"""Plug-and-play models: GGUF headers, discovery, preferences, and settings.

The GGUF reader is checked against bytes this file writes itself, so the tests
need no model on disk. That matters more than it sounds: the header is the
thing that decides how much RAM a dropped-in model is allowed to ask for, and
a test that needed a 700 MB file to check it would not be run.

`RealHeaderTests` reads the actual weights folder when one is present, and is
skipped otherwise.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from models import discovery
from models.discovery import (
    Discovered,
    choose_context,
    discover,
    estimate_min_free_mb,
    labelise,
    slugify,
)
from models.gguf import GgufInfo, read_metadata
from models.manager import ModelManager, ModelManagerError, load_registry
from models.preferences import ModelPreferences

# GGUF value type ids, as models/gguf.py reads them.
_UINT32, _UINT64, _STRING = 4, 10, 8


def write_gguf(
    path: Path,
    *,
    architecture: str = "llama",
    name: str = "Test Model",
    context_length: int = 8192,
    block_count: int = 16,
    head_count: int = 8,
    head_count_kv: int = 4,
    embedding_length: int = 1024,
    version: int = 3,
    padding: int = 0,
) -> Path:
    """Write a file with a valid GGUF header and no tensors.

    Enough for the reader to parse; the body is padding, so a 'model' here is
    a few hundred bytes rather than a few hundred megabytes.
    """
    fields: list[tuple[str, int, object]] = [
        ("general.architecture", _STRING, architecture),
        ("general.name", _STRING, name),
        (f"{architecture}.context_length", _UINT32, context_length),
        (f"{architecture}.block_count", _UINT32, block_count),
        (f"{architecture}.attention.head_count", _UINT32, head_count),
        (f"{architecture}.attention.head_count_kv", _UINT32, head_count_kv),
        (f"{architecture}.embedding_length", _UINT32, embedding_length),
    ]

    out = bytearray(b"GGUF")
    out += struct.pack("<I", version)
    out += struct.pack("<Q", 0)  # tensor count
    out += struct.pack("<Q", len(fields))
    for key, value_type, value in fields:
        encoded = key.encode("utf-8")
        out += struct.pack("<Q", len(encoded)) + encoded
        out += struct.pack("<I", value_type)
        if value_type == _STRING:
            text = str(value).encode("utf-8")
            out += struct.pack("<Q", len(text)) + text
        elif value_type == _UINT32:
            out += struct.pack("<I", int(value))
        else:
            out += struct.pack("<Q", int(value))

    out += b"\0" * padding
    path.write_bytes(bytes(out))
    return path


class GgufHeaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_the_fields_the_sizing_needs(self):
        path = write_gguf(self.tmp / "m.gguf", architecture="qwen3")
        info = read_metadata(path)
        self.assertIsNotNone(info)
        self.assertEqual(info.architecture, "qwen3")
        self.assertEqual(info.name, "Test Model")
        self.assertEqual(info.training_context, 8192)
        self.assertEqual(info.block_count, 16)
        self.assertEqual(info.head_count_kv, 4)

    def test_kv_bytes_per_token_is_arithmetic_not_a_guess(self):
        path = write_gguf(
            self.tmp / "m.gguf",
            block_count=17,
            head_count=8,
            head_count_kv=8,
            embedding_length=768,
        )
        info = read_metadata(path)
        # 2 caches x 17 layers x 8 KV heads x 96 head-dim x 2 bytes = 52,224.
        # This is the figure models.json worked out by hand for GLM-OCR.
        self.assertEqual(info.head_dim, 96)
        self.assertEqual(info.kv_bytes_per_token, 52224)
        self.assertEqual(info.kv_cache_mb(4096), 204)

    def test_a_missing_kv_head_count_falls_back_to_the_head_count(self):
        # Absent means multi-head, so the two are equal - not zero, which would
        # make the cache look free.
        info = GgufInfo(path=Path("x"), block_count=4, head_count=8,
                        head_count_kv=8, embedding_length=512)
        self.assertEqual(info.head_count_kv, 8)
        self.assertGreater(info.kv_bytes_per_token, 0)

    def test_an_incomplete_header_reports_zero_rather_than_a_wrong_number(self):
        info = GgufInfo(path=Path("x"), block_count=0)
        self.assertEqual(info.kv_bytes_per_token, 0)
        self.assertEqual(info.kv_cache_mb(4096), 0)

    def test_a_projector_is_recognised_by_its_architecture(self):
        path = write_gguf(self.tmp / "mmproj.gguf", architecture="clip")
        info = read_metadata(path)
        self.assertTrue(info.is_projector)

    def test_a_file_that_is_not_gguf_returns_none(self):
        path = self.tmp / "notes.txt"
        path.write_bytes(b"this is not a model")
        self.assertIsNone(read_metadata(path))

    def test_a_missing_file_returns_none(self):
        self.assertIsNone(read_metadata(self.tmp / "nope.gguf"))

    def test_a_truncated_header_returns_none_rather_than_raising(self):
        path = write_gguf(self.tmp / "m.gguf")
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
        # Half a header cannot be parsed; the contract is None, never an
        # exception out of this module.
        self.assertIsNone(read_metadata(path))

    def test_an_unknown_version_is_refused(self):
        path = write_gguf(self.tmp / "m.gguf", version=99)
        self.assertIsNone(read_metadata(path))

    def test_a_hostile_length_field_does_not_allocate(self):
        # A uint64 length of 2^60 must be rejected on sight, not attempted.
        out = bytearray(b"GGUF")
        out += struct.pack("<I", 3)
        out += struct.pack("<Q", 0)
        out += struct.pack("<Q", 1)
        out += struct.pack("<Q", 1 << 60)
        path = self.tmp / "evil.gguf"
        path.write_bytes(bytes(out))
        self.assertIsNone(read_metadata(path))


class SizingTests(unittest.TestCase):
    def info(self, **overrides) -> GgufInfo:
        settings = dict(
            path=Path("x"), block_count=24, head_count=16,
            head_count_kv=2, embedding_length=2048, training_context=262144,
        )
        settings.update(overrides)
        return GgufInfo(**settings)

    def test_a_long_context_model_is_capped_to_fit_the_cache_budget(self):
        # The whole point: a 262k model must not try to allocate 262k of cache.
        context, note = choose_context(self.info())
        self.assertLessEqual(context, 8192)
        self.assertIn("capped", note)
        self.assertIn("262,144", note)

    def test_a_short_context_model_is_not_padded_upwards(self):
        context, note = choose_context(self.info(training_context=2048))
        self.assertEqual(context, 2048)
        self.assertEqual(note, "")

    def test_a_bigger_budget_buys_more_context(self):
        small, _ = choose_context(self.info(), kv_budget_mb=100)
        large, _ = choose_context(self.info(), kv_budget_mb=2000)
        self.assertLess(small, large)

    def test_an_unreadable_header_falls_back_to_a_safe_default(self):
        context, note = choose_context(None)
        self.assertEqual(context, 4096)
        self.assertEqual(note, "")

    def test_the_ram_estimate_includes_weights_cache_and_headroom(self):
        gigabyte = 1024 * 1024 * 1024
        estimate = estimate_min_free_mb(gigabyte, self.info(), 4096)
        # 1024 MB of weights at 0.8 is 819, plus cache, plus 250 headroom.
        self.assertGreater(estimate, 819 + 250)
        self.assertLess(estimate, 2000)

    def test_a_large_model_gets_more_than_the_linear_estimate(self):
        gigabyte = 1024 * 1024 * 1024
        small = estimate_min_free_mb(1 * gigabyte, self.info(), 4096)
        large = estimate_min_free_mb(5 * gigabyte, self.info(), 4096)
        # Not merely five times: past a threshold a model cannot stay resident
        # and the linear estimate stops describing what it needs.
        self.assertGreater(large, small * 5)

    def test_an_unknown_cache_is_assumed_rather_than_treated_as_free(self):
        estimate = estimate_min_free_mb(1024 * 1024 * 1024, None, 4096)
        self.assertGreater(estimate, 819 + 250)


class NamingTests(unittest.TestCase):
    def test_the_quant_suffix_is_stripped(self):
        self.assertEqual(slugify("Qwen3-8B-Q4_K_M.gguf"), "qwen3-8b")
        self.assertEqual(labelise("Qwen3-8B-Q4_K_M.gguf"), "Qwen3 8B")

    def test_keys_are_lowercase_and_url_safe(self):
        self.assertEqual(slugify("Ministral 3.3B Instruct.gguf"), "ministral-3-3b-instruct")

    def test_a_key_is_never_empty(self):
        self.assertTrue(slugify("!!!.gguf"))


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_dropped_in_file_is_found_and_sized(self):
        write_gguf(self.tmp / "Shiny-New-7B-Q4_K_M.gguf", padding=4096)
        found = discover(self.tmp)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].key, "shiny-new-7b")
        self.assertEqual(found[0].label, "Shiny New 7B")
        self.assertGreater(found[0].min_free_mb, 0)
        self.assertGreater(found[0].context, 0)

    def test_files_a_curated_entry_already_claims_are_skipped(self):
        write_gguf(self.tmp / "known.gguf")
        write_gguf(self.tmp / "new.gguf")
        found = discover(self.tmp, known_files=["known.gguf"])
        self.assertEqual([item.file.name for item in found], ["new.gguf"])

    def test_a_projector_is_paired_rather_than_offered_as_a_model(self):
        write_gguf(self.tmp / "Vision-7B.gguf")
        write_gguf(self.tmp / "mmproj-Vision-7B.gguf", architecture="clip")
        found = discover(self.tmp)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].file.name, "Vision-7B.gguf")
        self.assertIsNotNone(found[0].mmproj)

    def test_a_paired_projector_counts_towards_the_ram_a_model_needs(self):
        """A vision model holds both halves at once. Qwen3-VL 2B is a 1,056 MB
        model with a 445 MB projector, so leaving the projector out under-states
        it by two fifths - and on 8 GB that is the guard waving through exactly
        what it exists to catch."""
        write_gguf(self.tmp / "Vision-2B-Q4_K_M.gguf", padding=4 * 1024 * 1024)
        alone = discover(self.tmp)[0].min_free_mb

        write_gguf(
            self.tmp / "mmproj-Vision-2B-F16.gguf",
            architecture="clip",
            padding=4 * 1024 * 1024,
        )
        paired = discover(self.tmp)[0]

        self.assertIsNotNone(paired.mmproj)
        self.assertGreater(paired.min_free_mb, alone)

    def test_a_dropped_in_vision_model_is_still_a_chat_model(self):
        """It was briefly given GLM-OCR's role, which took it out of the model
        picker: discovered, listed, and impossible to talk to. GLM-OCR is an
        OCR backend because it cannot call tools, not because it has a
        projector - and that entry is curated, so it keeps its own role."""
        write_gguf(self.tmp / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf")
        write_gguf(
            self.tmp / "mmproj-Qwen3-VL-2B-Instruct-F16.gguf", architecture="clip"
        )
        found = discover(self.tmp)
        self.assertEqual(found[0].role, "chat")
        self.assertIsNotNone(found[0].mmproj)

    def test_a_projector_at_a_different_quantisation_still_pairs(self):
        """The normal case, not the exception: a projector ships at F16 or Q8_0
        while the model it belongs to is Q4_K_M. Comparing the stems whole
        misses every real pair."""
        write_gguf(self.tmp / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf")
        write_gguf(
            self.tmp / "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf", architecture="clip"
        )
        found = discover(self.tmp)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].mmproj.name, "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf")

    def test_punctuation_does_not_stop_a_pair_from_matching(self):
        """`Qwen3VL` and `Qwen3-VL` are the same model named by two different
        uploaders, and people mix the two in one folder without noticing."""
        write_gguf(self.tmp / "Qwen3VL-2B-Instruct-Q4_K_M.gguf")
        write_gguf(
            self.tmp / "mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf", architecture="clip"
        )
        found = discover(self.tmp)
        self.assertIsNotNone(found[0].mmproj)

    def test_a_projector_naming_no_model_pairs_when_there_is_only_one(self):
        """`mmproj-F16.gguf` is what the most-downloaded Qwen3-VL repository
        calls its projector. It says what it is, not what it is for."""
        write_gguf(self.tmp / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf")
        write_gguf(self.tmp / "mmproj-F16.gguf", architecture="clip")
        found = discover(self.tmp)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].mmproj.name, "mmproj-F16.gguf")

    def test_a_projector_naming_no_model_is_left_alone_when_several_could_claim_it(self):
        """Guessing would hand it to the wrong model, which then loads and
        reports no vision - which reads as a broken model rather than as a file
        that wants renaming."""
        write_gguf(self.tmp / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf")
        write_gguf(self.tmp / "Ministral-3B-Q4_K_M.gguf")
        write_gguf(self.tmp / "mmproj-F16.gguf", architecture="clip")
        found = discover(self.tmp)
        self.assertEqual(len(found), 2)
        self.assertTrue(all(item.mmproj is None for item in found))

    def test_a_curated_model_does_not_count_as_the_one_candidate(self):
        """`sole` is about models this scan is actually offering. A folder whose
        other model is already in models.json still has one candidate."""
        write_gguf(self.tmp / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf")
        write_gguf(self.tmp / "already-curated.gguf")
        write_gguf(self.tmp / "mmproj-F16.gguf", architecture="clip")
        found = discover(self.tmp, known_files=["already-curated.gguf"])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].mmproj.name, "mmproj-F16.gguf")

    def test_the_wrong_projector_is_not_handed_to_a_model(self):
        """Two vision models in one folder, each with its own projector."""
        write_gguf(self.tmp / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf")
        write_gguf(self.tmp / "GLM-OCR-Q8_0.gguf")
        write_gguf(
            self.tmp / "mmproj-Qwen3-VL-2B-Instruct-F16.gguf", architecture="clip"
        )
        write_gguf(self.tmp / "mmproj-GLM-OCR-Q8_0.gguf", architecture="clip")
        pairs = {item.file.name: item.mmproj.name for item in discover(self.tmp)}
        self.assertEqual(
            pairs,
            {
                "Qwen3-VL-2B-Instruct-Q4_K_M.gguf": "mmproj-Qwen3-VL-2B-Instruct-F16.gguf",
                "GLM-OCR-Q8_0.gguf": "mmproj-GLM-OCR-Q8_0.gguf",
            },
        )

    def test_a_projector_named_by_convention_is_recognised_without_a_header(self):
        (self.tmp / "mmproj-thing.gguf").write_bytes(b"not really gguf")
        (self.tmp / "thing.gguf").write_bytes(b"not really gguf")
        found = discover(self.tmp)
        self.assertEqual([item.file.name for item in found], ["thing.gguf"])

    def test_ports_do_not_collide_with_ones_already_taken(self):
        for n in range(3):
            write_gguf(self.tmp / f"model{n}.gguf")
        found = discover(self.tmp, taken_ports=[8090, 8091])
        ports = [item.port for item in found]
        self.assertEqual(len(set(ports)), 3)
        self.assertNotIn(8090, ports)
        self.assertNotIn(8091, ports)

    def test_keys_do_not_collide(self):
        write_gguf(self.tmp / "thing-Q4_K_M.gguf")
        write_gguf(self.tmp / "thing-Q8_0.gguf")
        keys = [item.key for item in discover(self.tmp)]
        self.assertEqual(len(set(keys)), 2)

    def test_a_key_already_used_by_a_curated_entry_is_not_reused(self):
        write_gguf(self.tmp / "mistral.gguf")
        found = discover(self.tmp, taken_keys=["mistral"])
        self.assertNotEqual(found[0].key, "mistral")

    def test_an_unreadable_file_is_still_offered_with_a_note(self):
        (self.tmp / "mystery.gguf").write_bytes(b"nonsense" * 100)
        found = discover(self.tmp)
        self.assertEqual(len(found), 1)
        self.assertTrue(any("header could not be read" in n for n in found[0].notes))

    def test_a_folder_with_nothing_in_it_is_not_an_error(self):
        self.assertEqual(discover(self.tmp), [])

    def test_a_folder_that_does_not_exist_is_not_an_error(self):
        self.assertEqual(discover(self.tmp / "nope"), [])

    def test_non_gguf_files_are_ignored(self):
        (self.tmp / "readme.txt").write_text("hello", encoding="utf-8")
        (self.tmp / "model.bin").write_bytes(b"x")
        self.assertEqual(discover(self.tmp), [])


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_missing_file_is_first_launch_not_an_error(self):
        prefs = ModelPreferences.load(self.tmp)
        self.assertEqual(prefs.primary, "")
        self.assertFalse(prefs.setup_complete)

    def test_choosing_a_primary_round_trips(self):
        prefs = ModelPreferences.load(self.tmp)
        prefs.set_primary("tiny")
        prefs.save()

        again = ModelPreferences.load(self.tmp)
        self.assertEqual(again.primary, "tiny")
        self.assertTrue(again.setup_complete)
        # The router's cheap end follows the primary when it was never set.
        self.assertEqual(again.router_fast, "tiny")

    def test_a_corrupt_file_returns_to_first_launch_rather_than_crashing(self):
        (self.tmp / "models.local.json").write_text("{not json", encoding="utf-8")
        prefs = ModelPreferences.load(self.tmp)
        self.assertFalse(prefs.setup_complete)

    def test_a_file_from_a_future_version_is_ignored(self):
        (self.tmp / "models.local.json").write_text(
            json.dumps({"version": 999, "primary": "ghost"}), encoding="utf-8"
        )
        self.assertEqual(ModelPreferences.load(self.tmp).primary, "")

    def test_only_safe_fields_can_be_overridden(self):
        prefs = ModelPreferences.load(self.tmp)
        applied = prefs.override(
            "tiny", {"label": "Mine", "port": 9999, "file": "evil.gguf", "role": "chat"}
        )
        # Port, file and role decide what a model is and where it runs.
        self.assertEqual(applied, {"label": "Mine"})

    def test_numeric_overrides_are_clamped(self):
        prefs = ModelPreferences.load(self.tmp)
        applied = prefs.override("tiny", {"context": 99_999_999, "threads": 0})
        self.assertLessEqual(applied["context"], 131_072)
        self.assertGreaterEqual(applied["threads"], 1)

    def test_a_context_of_zero_cannot_be_set(self):
        # llama.cpp reads 0 as "the model's trained maximum", which on a 262k
        # model is how you fill a machine.
        prefs = ModelPreferences.load(self.tmp)
        self.assertGreater(prefs.override("tiny", {"context": 0})["context"], 0)

    def test_hiding_and_unhiding(self):
        prefs = ModelPreferences.load(self.tmp)
        prefs.hide("tiny")
        self.assertTrue(prefs.is_hidden("tiny"))
        prefs.hide("tiny", False)
        self.assertFalse(prefs.is_hidden("tiny"))

    def test_saving_is_atomic(self):
        prefs = ModelPreferences.load(self.tmp)
        prefs.set_primary("tiny")
        prefs.save()
        # No temporary file left behind.
        self.assertEqual(
            sorted(p.name for p in self.tmp.iterdir()), ["models.local.json"]
        )


class RegistryLayeringTests(unittest.TestCase):
    """models.json, the folder, and the user's settings, in that order."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        (self.tmp / "weights").mkdir()
        (self.tmp / "llama-server.exe").write_text("x", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def registry(self, models: list[dict], **extra) -> Path:
        payload = {
            "server_exe": str(self.tmp / "llama-server.exe"),
            "models_dir": "weights",
            "max_active": 1,
            "models": models,
            **extra,
        }
        path = self.tmp / "models.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_curated_entry_is_never_overwritten_by_discovery(self):
        write_gguf(self.tmp / "weights" / "curated.gguf")
        path = self.registry([{
            "key": "curated", "label": "Hand tuned", "file": "curated.gguf",
            "port": 8080, "context": 1234, "min_free_mb": 4321,
        }])
        loaded = load_registry(path)
        spec = loaded["specs"]["curated"]
        # The measured values survive; discovery adds nothing for this file.
        self.assertEqual(spec.context, 1234)
        self.assertEqual(spec.min_free_mb, 4321)
        self.assertEqual(loaded["discovered"], [])

    def test_a_new_file_is_added_alongside_curated_entries(self):
        write_gguf(self.tmp / "weights" / "curated.gguf")
        write_gguf(self.tmp / "weights" / "Dropped-In.gguf")
        path = self.registry([{
            "key": "curated", "label": "Hand tuned", "file": "curated.gguf",
            "port": 8080,
        }])
        loaded = load_registry(path)
        self.assertIn("curated", loaded["specs"])
        self.assertIn("dropped-in", loaded["specs"])
        self.assertEqual(loaded["discovered"], ["dropped-in"])

    def test_no_registry_at_all_still_works(self):
        # The true plug-and-play case: a fresh install with one file in the
        # folder and no models.json.
        write_gguf(self.tmp / "weights" / "Only-Model.gguf")
        loaded = load_registry(self.tmp / "does-not-exist.json")
        self.assertIn("only-model", loaded["specs"])
        self.assertEqual(loaded["default"], "only-model")

    def test_an_empty_folder_and_no_registry_is_a_clear_error(self):
        with self.assertRaises(ModelManagerError) as caught:
            load_registry(self.tmp / "does-not-exist.json")
        self.assertIn(".gguf", str(caught.exception))

    def test_the_saved_primary_wins_over_the_registry_default(self):
        write_gguf(self.tmp / "weights" / "a.gguf")
        write_gguf(self.tmp / "weights" / "b.gguf")
        path = self.registry([
            {"key": "a", "file": "a.gguf", "port": 8080},
            {"key": "b", "file": "b.gguf", "port": 8081},
        ], default="a")

        prefs = ModelPreferences.load(self.tmp)
        prefs.set_primary("b")
        prefs.save()

        self.assertEqual(load_registry(path)["default"], "b")

    def test_an_override_is_applied_on_top_of_a_curated_entry(self):
        write_gguf(self.tmp / "weights" / "a.gguf")
        path = self.registry([{
            "key": "a", "label": "Original", "file": "a.gguf",
            "port": 8080, "context": 4096,
        }])

        prefs = ModelPreferences.load(self.tmp)
        prefs.override("a", {"label": "Renamed", "context": 2048})
        prefs.save()

        spec = load_registry(path)["specs"]["a"]
        self.assertEqual(spec.label, "Renamed")
        self.assertEqual(spec.context, 2048)

    def test_a_hidden_model_stays_in_the_catalogue_but_is_not_offered(self):
        write_gguf(self.tmp / "weights" / "a.gguf")
        write_gguf(self.tmp / "weights" / "b.gguf")
        path = self.registry([
            {"key": "a", "file": "a.gguf", "port": 8080},
            {"key": "b", "file": "b.gguf", "port": 8081},
        ], default="a")

        prefs = ModelPreferences.load(self.tmp)
        prefs.hide("b")
        prefs.save()

        loaded = load_registry(path)
        # Still known, so it can be un-hidden; just not offered.
        self.assertIn("b", loaded["specs"])
        self.assertNotIn("b", loaded["offered"])

    def test_setup_is_required_only_when_there_is_a_real_choice(self):
        write_gguf(self.tmp / "weights" / "only.gguf")
        path = self.registry([])
        self.assertFalse(load_registry(path)["setup_required"])

        write_gguf(self.tmp / "weights" / "second.gguf")
        self.assertTrue(load_registry(path)["setup_required"])

    def test_choosing_a_primary_ends_the_setup_prompt(self):
        write_gguf(self.tmp / "weights" / "a.gguf")
        write_gguf(self.tmp / "weights" / "b.gguf")
        path = self.registry([])
        self.assertTrue(load_registry(path)["setup_required"])

        prefs = ModelPreferences.load(self.tmp)
        prefs.set_primary("a")
        prefs.save()
        self.assertFalse(load_registry(path)["setup_required"])

    def test_two_models_sharing_a_port_is_reported(self):
        write_gguf(self.tmp / "weights" / "a.gguf")
        write_gguf(self.tmp / "weights" / "b.gguf")
        path = self.registry([
            {"key": "a", "file": "a.gguf", "port": 8080},
            {"key": "b", "file": "b.gguf", "port": 8080},
        ])
        with self.assertRaises(ModelManagerError) as caught:
            load_registry(path)
        self.assertIn("port", str(caught.exception).lower())


class PreferenceLocationTests(unittest.TestCase):
    """Where preferences are kept, given how the registry was named."""

    def test_the_same_registry_by_two_names_shares_its_preferences(self):
        """A relative path and an absolute one are the same file. Comparing
        them unspelt sent settings made from a script into a second file the
        running application never reads - silently, which is the bad part."""
        from models.manager import DEFAULT_PREFERENCES_DIR, DEFAULT_REGISTRY

        if not DEFAULT_REGISTRY.is_file():
            self.skipTest("no shipped registry to compare against")

        relative = Path(os.path.relpath(DEFAULT_REGISTRY, Path.cwd()))
        registry = load_registry(relative)
        self.assertEqual(
            registry["preferences"].path.parent.resolve(),
            DEFAULT_PREFERENCES_DIR.resolve(),
        )

    def test_a_registry_somewhere_else_keeps_its_preferences_beside_it(self):
        """This is what stops the suite reading and rewriting the real user's
        settings, so it matters as much as the case above."""
        with tempfile.TemporaryDirectory() as name:
            tmp = Path(name)
            (tmp / "llama-server.exe").write_text("x", encoding="utf-8")
            write_gguf(tmp / "something.gguf")
            path = tmp / "models.json"
            path.write_text(
                json.dumps({"models_dir": str(tmp), "models": []}), encoding="utf-8"
            )
            registry = load_registry(path)
            self.assertEqual(registry["preferences"].path.parent, tmp)


class ManagerSettingsTests(unittest.TestCase):
    """Choosing and retuning through the manager, and persistence."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        weights = self.tmp / "weights"
        weights.mkdir()
        (self.tmp / "llama-server.exe").write_text("x", encoding="utf-8")
        write_gguf(weights / "Alpha.gguf")
        write_gguf(weights / "Beta.gguf")
        write_gguf(weights / "Vision.gguf")
        write_gguf(weights / "mmproj-Vision.gguf", architecture="clip")
        write_gguf(weights / "Reader.gguf")

        self.path = self.tmp / "models.json"
        self.path.write_text(
            json.dumps({
                "server_exe": str(self.tmp / "llama-server.exe"),
                "models_dir": "weights",
                "max_active": 1,
                # Curated, because `role` is curated. A dropped-in vision model
                # is an ordinary chat model that can also see; an OCR backend
                # is one that cannot drive the loop at all, and only the person
                # writing models.json knows which they have.
                "models": [
                    {
                        "key": "reader",
                        "label": "Reader",
                        "file": "Reader.gguf",
                        "port": 8188,
                        "role": "ocr",
                        "min_free_mb": 0,
                    }
                ],
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def manager(self) -> ModelManager:
        return ModelManager(self.path, preferences_dir=self.tmp)

    def test_everything_in_the_folder_is_offered(self):
        manager = self.manager()
        keys = {status.spec.key for status in manager.statuses()}
        self.assertIn("alpha", keys)
        self.assertIn("beta", keys)
        self.assertIn("vision", keys)
        # The projector is paired, never offered on its own.
        self.assertNotIn("mmproj-vision", keys)

    def test_setting_a_primary_persists_across_a_restart(self):
        manager = self.manager()
        manager.set_primary("beta")
        self.assertEqual(manager.default_key, "beta")
        self.assertFalse(manager.setup_required)

        self.assertEqual(self.manager().default_key, "beta")

    def test_a_vision_backend_cannot_be_the_primary(self):
        with self.assertRaises(ModelManagerError) as caught:
            self.manager().set_primary("reader")
        self.assertIn("not a chat model", str(caught.exception))

    def test_a_dropped_in_vision_model_can_be_the_primary(self):
        """It is a chat model that can also see, so there is nothing to stop
        it. Inferring an OCR role from the projector used to, which made a
        model you had just installed impossible to select."""
        manager = self.manager()
        manager.set_primary("vision")
        self.assertEqual(manager.default_key, "vision")

    def test_an_unknown_key_cannot_be_the_primary(self):
        with self.assertRaises(ModelManagerError):
            self.manager().set_primary("ghost")

    def test_an_override_persists_and_is_applied(self):
        manager = self.manager()
        manager.set_override("alpha", {"label": "My Alpha", "context": 2048})
        self.assertEqual(manager.get_spec("alpha").label, "My Alpha")

        reopened = self.manager()
        self.assertEqual(reopened.get_spec("alpha").context, 2048)

    def test_clearing_an_override_restores_the_discovered_values(self):
        manager = self.manager()
        original = manager.get_spec("alpha").context
        manager.set_override("alpha", {"context": 2048})
        manager.clear_override("alpha")
        self.assertEqual(manager.get_spec("alpha").context, original)

    def test_hiding_a_model_and_bringing_it_back(self):
        manager = self.manager()
        manager.set_primary("alpha")
        manager.set_hidden("beta", True)
        self.assertTrue(manager.is_hidden("beta"))
        manager.set_hidden("beta", False)
        self.assertFalse(manager.is_hidden("beta"))

    def test_the_primary_cannot_be_hidden(self):
        manager = self.manager()
        manager.set_primary("alpha")
        with self.assertRaises(ModelManagerError) as caught:
            manager.set_hidden("alpha", True)
        self.assertIn("primary", str(caught.exception).lower())

    def test_rescan_finds_a_file_copied_in_while_running(self):
        manager = self.manager()
        self.assertNotIn("gamma", {s.spec.key for s in manager.statuses()})

        write_gguf(self.tmp / "weights" / "Gamma.gguf")
        added = manager.rescan()

        self.assertEqual(added, ["gamma"])
        self.assertIn("gamma", {s.spec.key for s in manager.statuses()})

    def test_rescan_with_nothing_new_reports_nothing(self):
        self.assertEqual(self.manager().rescan(), [])

    def test_rescan_keeps_the_chosen_primary(self):
        manager = self.manager()
        manager.set_primary("beta")
        write_gguf(self.tmp / "weights" / "Gamma.gguf")
        manager.rescan()
        self.assertEqual(manager.default_key, "beta")

    def test_the_router_can_be_pointed_at_chosen_models(self):
        manager = self.manager()
        manager.set_router(fast="alpha", strong="beta")
        self.assertEqual(manager.router_fast, "alpha")
        self.assertEqual(manager.router_strong, "beta")
        self.assertEqual(self.manager().router_strong, "beta")

    def test_the_router_cannot_point_at_a_model_that_is_not_there(self):
        with self.assertRaises(ModelManagerError):
            self.manager().set_router(fast="ghost")


@unittest.skipUnless(
    (Path(__file__).resolve().parent.parent / "weights").is_dir()
    and any((Path(__file__).resolve().parent.parent / "weights").glob("*.gguf")),
    "no weights folder on this machine",
)
class RealHeaderTests(unittest.TestCase):
    """Against the actual GGUF files, when there are any.

    The synthetic headers above prove the parser handles the format; these
    prove it handles what real quantisation tools actually emit.
    """

    def test_every_real_model_reports_a_usable_header(self):
        weights = Path(__file__).resolve().parent.parent / "weights"
        for path in sorted(weights.glob("*.gguf")):
            with self.subTest(model=path.name):
                info = read_metadata(path)
                self.assertIsNotNone(info, f"{path.name} produced no header")
                self.assertTrue(info.architecture)
                if not info.is_projector:
                    # A language model must yield enough to size its cache.
                    self.assertGreater(info.kv_bytes_per_token, 0)
                    self.assertGreater(info.training_context, 0)

    def test_real_models_are_capped_below_their_training_context(self):
        weights = Path(__file__).resolve().parent.parent / "weights"
        for item in discover(weights):
            with self.subTest(model=item.file.name):
                self.assertLessEqual(item.context, discovery.CONTEXT_LADDER[-1])
                self.assertGreater(item.min_free_mb, 0)
