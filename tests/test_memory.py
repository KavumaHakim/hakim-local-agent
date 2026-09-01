"""The memory system: storage, retrieval, consolidation, and model switching.

Two fakes, and a real one that matters.

`HashingEmbedder` stands in for BGE - a bag-of-words vector, so similarity is
real and testable without loading a 130 MB model.

`ScriptedModel` stands in for the auxiliary model, returning whatever JSON a
test wants.

The model *manager* is not faked. `ManagerHarness` is the real `ModelManager`
with only process spawning and health checks replaced, so the tests that assert
"the two models are never loaded together" are asserting it about the code that
actually enforces it, not about a stub written to agree.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
import unittest

# These need the optional document-search dependencies, and must be checked
# BEFORE the imports that use them: an import error reads as a broken test
# suite rather than as "that part is not installed", which is a supported way
# to run this project. unittest turns a SkipTest raised here into a skipped
# module.
try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover - depends on what is installed
    raise unittest.SkipTest(
        "document search dependencies are not installed "
        "(pip install -r requirements-rag.txt)"
    )
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import Config
from memory.context import ContextBuilder
from memory.manager import MemoryManager, MemoryOperationError, reset_shared
from memory.processor import MemoryProcessor, _json_from
from memory.store import MemoryStore, normalise
from memory.types import (
    Candidate,
    JobKind,
    JobStatus,
    MemoryStatus,
    MemoryType,
    ScoredMemory,
    WorkingMemory,
)
from memory import extraction, retrieval
from tests.test_manager import ManagerHarness, write_registry
from tools.base import ToolRegistry
from tools.memory_tool import MemoryToolError, build_memory_tools


DIMENSION = 384


class HashingEmbedder:
    """Deterministic bag-of-words vectors, so similarity is real but free."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension
        self.loaded = False
        self.passages_encoded = 0
        self.queries_encoded = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().split():
            word = word.strip(".,;:()[]!?\"'")
            if not word:
                continue
            digest = hashlib.md5(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        return vector / norm

    def encode_passages(self, texts, progress=None) -> np.ndarray:
        texts = list(texts)
        self.passages_encoded += len(texts)
        self.loaded = True
        return np.vstack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        self.queries_encoded += 1
        self.loaded = True
        return self._vector(text)

    def unload(self) -> bool:
        self.loaded = False
        return True


class ScriptedModel:
    """An auxiliary model that returns queued replies and records its prompts."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.prompts: list[tuple[str, str]] = []

    def chat(self, messages, *, tools=None):
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.prompts.append((system, user))
        reply = self.replies.pop(0) if self.replies else "[]"
        return {"role": "assistant", "content": reply}


class MemoryCase(unittest.TestCase):
    """A temporary database and a fake embedder."""

    def setUp(self):
        reset_shared()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.embedder = HashingEmbedder()

    def tearDown(self):
        reset_shared()
        self._tmp.cleanup()

    def manager(self, **overrides) -> MemoryManager:
        settings = dict(
            store_dir=self.tmp / "vectors",
            embedder=self.embedder,
            dimension=DIMENSION,
            top_k=5,
            # The production gate (0.55) is calibrated for BGE-small, whose
            # unrelated-text similarity sits around 0.4-0.55. HashingEmbedder
            # is lexical and has a completely different distribution -
            # unrelated text scores ~0, related text scores by word overlap -
            # so the fake needs its own gate or every test would measure the
            # fake's vocabulary rather than the system. The real number is
            # checked by `test_the_default_gate_is_the_measured_one`.
            min_similarity=0.25,
        )
        settings.update(overrides)
        return MemoryManager(self.tmp / "agent.db", **settings)


# --- storing ---------------------------------------------------------------


class StoreTests(MemoryCase):
    def test_a_memory_round_trips(self):
        store = MemoryStore(self.tmp / "agent.db")
        memory_id, created = store.add(
            Candidate(content="User prefers vim", type=MemoryType.PREFERENCE)
        )
        self.assertTrue(created)
        found = store.get(memory_id)
        self.assertEqual(found.content, "User prefers vim")
        self.assertIs(found.type, MemoryType.PREFERENCE)
        self.assertIs(found.status, MemoryStatus.ACTIVE)

    def test_storing_the_same_sentence_twice_reinforces_rather_than_duplicates(self):
        store = MemoryStore(self.tmp / "agent.db")
        first, created_a = store.add(Candidate(content="User prefers vim", confidence=0.6))
        second, created_b = store.add(Candidate(content="user   prefers VIM", confidence=0.6))
        self.assertTrue(created_a)
        self.assertFalse(created_b)
        self.assertEqual(first, second)
        self.assertEqual(store.counts()["active"], 1)
        # Seeing it again is evidence, so confidence rises.
        self.assertGreater(store.get(first).confidence, 0.6)

    def test_normalise_collapses_case_and_whitespace_only(self):
        self.assertEqual(normalise("  User   PREFERS vim "), "user prefers vim")
        # Punctuation is kept: catching that pair is the embedding's job.
        self.assertNotEqual(normalise("User prefers vim"), normalise("User prefers vim."))

    def test_empty_content_is_refused(self):
        store = MemoryStore(self.tmp / "agent.db")
        with self.assertRaises(ValueError):
            store.add(Candidate(content="   "))

    def test_forgetting_marks_deleted_without_losing_the_row(self):
        store = MemoryStore(self.tmp / "agent.db")
        memory_id, _ = store.add(Candidate(content="User prefers vim"))
        self.assertTrue(store.forget(memory_id))
        self.assertIs(store.get(memory_id).status, MemoryStatus.DELETED)

    def test_a_hard_forget_removes_the_row(self):
        store = MemoryStore(self.tmp / "agent.db")
        memory_id, _ = store.add(Candidate(content="User prefers vim"))
        store.forget(memory_id, hard=True)
        self.assertIsNone(store.get(memory_id))

    def test_superseding_keeps_both_and_links_them(self):
        store = MemoryStore(self.tmp / "agent.db")
        old, _ = store.add(Candidate(content="User uses Qwen", type=MemoryType.FACT))
        new, _ = store.add(Candidate(content="User uses Mistral", type=MemoryType.FACT))
        store.supersede(old, new)

        self.assertIs(store.get(old).status, MemoryStatus.SUPERSEDED)
        self.assertEqual(store.get(old).superseded_by, new)
        self.assertIs(store.get(new).status, MemoryStatus.ACTIVE)
        self.assertTrue(
            any(link["relation"] == "supersedes" for link in store.links_for(new))
        )

    def test_the_legacy_keyed_table_is_imported_and_renamed(self):
        path = self.tmp / "agent.db"
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE memories (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO memories VALUES ('editor', 'prefers vim', 'a', 'b')"
            )
            connection.commit()
        finally:
            connection.close()

        store = MemoryStore(path)
        contents = [memory.content for memory in store.list_memories()]
        self.assertIn("editor: prefers vim", contents)

        connection = sqlite3.connect(path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            connection.close()
        # Renamed, not dropped: an upgrade must not be what loses the data.
        self.assertIn("memories_legacy", tables)
        self.assertNotIn("memories", tables)

    def test_the_legacy_import_does_not_run_twice(self):
        path = self.tmp / "agent.db"
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE memories (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO memories VALUES ('editor', 'prefers vim', 'a', 'b')"
            )
            connection.commit()
        finally:
            connection.close()

        MemoryStore(path)
        again = MemoryStore(path)
        self.assertEqual(again.counts()["active"], 1)


# --- extraction ------------------------------------------------------------


class ExtractionTests(unittest.TestCase):
    def test_greetings_and_small_talk_are_noise(self):
        for text in ("hi", "thanks!", "ok", "good morning", "cheers", "yep"):
            self.assertTrue(extraction.is_noise(text), text)

    def test_a_real_statement_is_not_noise(self):
        self.assertFalse(extraction.is_noise("I always use vim for editing code"))

    def test_an_explicit_instruction_is_recognised(self):
        self.assertEqual(
            extraction.explicit_request("Remember that I prefer dark mode"),
            "I prefer dark mode",
        )
        self.assertEqual(
            extraction.explicit_request("please remember: I use fish shell"),
            "I use fish shell",
        )

    def test_a_passing_mention_of_remembering_is_not_an_instruction(self):
        self.assertIsNone(
            extraction.explicit_request("I should remember to buy milk")
        )

    def test_forget_and_recall_requests_are_recognised(self):
        self.assertEqual(
            extraction.forget_request("Forget that I prefer dark mode"),
            "I prefer dark mode",
        )
        self.assertEqual(
            extraction.recall_request("What do you remember about me?"), "me"
        )

    def test_a_durable_preference_is_extracted(self):
        found = extraction.heuristic("I always use ripgrep instead of grep")
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].type, MemoryType.PREFERENCE)
        self.assertGreater(found[0].confidence, 0.6)

    def test_a_hedged_statement_becomes_an_intention_not_a_preference(self):
        # The trap the design calls out: "I might try X tomorrow" must not
        # become a permanent fact.
        found = extraction.heuristic("I might always use ripgrep from tomorrow")
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].type, MemoryType.INTENTION)
        self.assertLess(found[0].confidence, 0.5)

    def test_a_time_boxed_statement_becomes_temporary(self):
        found = extraction.heuristic("I always use ripgrep today, just testing")
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].type, MemoryType.TEMPORARY)

    def test_current_state_is_extracted_as_a_fact(self):
        found = extraction.heuristic("I'm now using Mistral as my main model")
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].type, MemoryType.FACT)

    def test_a_question_produces_nothing(self):
        self.assertEqual(extraction.heuristic("What is photosynthesis?"), [])

    def test_a_document_event_records_the_event_not_the_document(self):
        candidate = extraction.document_event("Biology.pdf")
        self.assertIs(candidate.type, MemoryType.EVENT)
        self.assertEqual(candidate.subject, "Biology.pdf")
        self.assertIn("Biology.pdf", candidate.content)
        self.assertLess(len(candidate.content), 200)


# --- scoring and decay -----------------------------------------------------


class ScoringTests(MemoryCase):
    def test_the_default_gate_is_the_measured_one(self):
        # Measured against bge-small-en-v1.5 with five stored memories:
        # relevant queries scored 0.575-0.664, irrelevant 0.395-0.549.
        # Lowering this without re-measuring makes every question retrieve
        # the whole store.
        from config import Config

        self.assertEqual(retrieval.DEFAULT_MIN_SIMILARITY, 0.55)
        self.assertEqual(Config().memory_min_similarity, 0.55)

    def test_the_gate_runs_before_scoring(self):
        store = MemoryStore(self.tmp / "agent.db")
        memory_id, _ = store.add(
            Candidate(content="User prefers vim", importance=1.0, confidence=1.0)
        )
        memory = store.get(memory_id)
        # Perfect on every other axis, but not about the question.
        self.assertEqual(
            retrieval.rank([(memory, 0.4)], limit=5, min_similarity=0.55), []
        )
        self.assertTrue(
            retrieval.rank([(memory, 0.7)], limit=5, min_similarity=0.55)
        )

    def memory(self, **overrides):
        store = MemoryStore(self.tmp / "agent.db")
        settings = dict(content="User prefers vim", type=MemoryType.PREFERENCE)
        settings.update(overrides)
        memory_id, _ = store.add(Candidate(**settings))
        return store.get(memory_id)

    def test_a_fresh_memory_barely_decays(self):
        self.assertGreater(retrieval.recency_factor(self.memory()), 0.99)

    def test_a_temporary_memory_decays_far_faster_than_a_preference(self):
        old = datetime.now(timezone.utc) - timedelta(days=14)
        preference = self.memory(type=MemoryType.PREFERENCE)
        temporary = self.memory(content="User is testing X", type=MemoryType.TEMPORARY)

        import dataclasses

        aged_preference = dataclasses.replace(preference, updated_at=old.isoformat())
        aged_temporary = dataclasses.replace(temporary, updated_at=old.isoformat())

        self.assertGreater(retrieval.recency_factor(aged_preference), 0.95)
        self.assertLess(retrieval.recency_factor(aged_temporary), 0.2)

    def test_every_factor_can_veto(self):
        # A product, not a sum: zero confidence means zero score however well
        # the text matches.
        memory = self.memory(confidence=0.0)
        self.assertEqual(retrieval.score(memory, 1.0).score, 0.0)

    def test_usage_lifts_a_memory_but_only_so_far(self):
        import dataclasses

        base = self.memory()
        used = dataclasses.replace(base, access_count=50)
        self.assertGreater(retrieval.usage_factor(used), 1.0)
        self.assertLessEqual(retrieval.usage_factor(used), 1 + retrieval.MAX_USAGE_BONUS)

    def test_a_superseded_memory_ranks_below_an_active_one(self):
        import dataclasses

        active = self.memory()
        old = dataclasses.replace(active, status=MemoryStatus.SUPERSEDED)
        self.assertGreater(
            retrieval.score(active, 0.9).score, retrieval.score(old, 0.9).score
        )

    def test_ranking_drops_everything_under_the_floor(self):
        memory = self.memory()
        ranked = retrieval.rank([(memory, 0.01)], limit=5, floor=0.1)
        self.assertEqual(ranked, [])


# --- retrieval -------------------------------------------------------------


class RetrievalTests(MemoryCase):
    """Ranking and the retrieval path.

    `HashingEmbedder` matches on shared words, not meaning, so these queries
    deliberately reuse the memory's vocabulary. That is a limit of the fake,
    not of the system: what these check is that the pipeline retrieves, ranks,
    filters and records correctly. Whether "what editor do I use" finds "User
    prefers vim" is BGE's job, and `RealModelTests` checks it for real.
    """

    def test_a_relevant_question_retrieves_the_right_memory(self):
        manager = self.manager()
        manager.remember("User prefers vim for editing code", type="preference")
        manager.remember("User runs models locally on an 8 GB laptop", type="fact")

        found = manager.search("user prefers editing code")
        self.assertTrue(found)
        self.assertIn("vim", found[0].memory.content)

    def test_an_unrelated_question_retrieves_nothing(self):
        # The requirement: "What is photosynthesis?" must not drag personal
        # memories into the prompt.
        manager = self.manager()
        manager.remember("User prefers vim for editing code", type="preference")
        manager.remember("User runs models locally on an 8 GB laptop", type="fact")

        self.assertEqual(manager.search("what is photosynthesis"), [])

    def test_retrieval_needs_no_language_model(self):
        # Nothing in this path may touch a chat model; only the embedder is
        # used, and it is a separate short-lived process.
        manager = self.manager()
        manager.remember("User prefers vim", type="preference")
        manager.search("editor preference")
        self.assertGreater(self.embedder.queries_encoded, 0)
        self.assertIsNone(manager.processor)

    def test_retrieval_records_that_a_memory_was_used(self):
        manager = self.manager()
        manager.remember("User prefers vim for editing code", type="preference")
        manager.recall("editing code preference")
        self.assertEqual(manager.store.list_memories()[0].access_count, 1)

    def test_recall_without_a_query_lists_the_most_important(self):
        manager = self.manager()
        manager.remember("Trivial detail about nothing", importance=0.1)
        manager.remember("User prefers vim", importance=0.9)
        listed = manager.recall()
        self.assertEqual(listed["memories"][0]["content"], "User prefers vim")

    def test_search_falls_back_to_text_when_embeddings_are_unavailable(self):
        class Broken(HashingEmbedder):
            def encode_query(self, text):
                raise RuntimeError("no embedder")

            def encode_passages(self, texts, progress=None):
                raise RuntimeError("no embedder")

        manager = self.manager(embedder=Broken())
        manager.remember("User prefers vim for editing", type="preference")
        found = manager.search("vim")
        self.assertTrue(found)
        self.assertIn("vim", found[0].memory.content)

    def test_memories_survive_a_restart(self):
        first = self.manager()
        first.remember("User prefers vim for editing code", type="preference")

        # A brand new manager over the same files, with a fresh embedder.
        reopened = self.manager(embedder=HashingEmbedder())
        found = reopened.search("editing code preference vim")
        self.assertTrue(found)
        self.assertIn("vim", found[0].memory.content)


# --- explicit user control -------------------------------------------------


class ExplicitControlTests(MemoryCase):
    def test_remember_stores_immediately_without_a_queue(self):
        manager = self.manager()
        result = manager.remember("User prefers dark mode", type="preference")
        self.assertTrue(result["stored"])
        # No job was queued: an explicit instruction must take effect now.
        self.assertEqual(manager.store.pending_count(), 0)
        self.assertEqual(manager.store.counts()["active"], 1)

    def test_an_explicit_remember_in_a_turn_is_honoured_at_once(self):
        manager = self.manager()
        outcome = manager.observe_turn(
            prompt="Remember that I prefer tabs over spaces", conversation_id=1
        )
        self.assertEqual(outcome["stored"], 1)
        found = manager.search("tabs or spaces")
        self.assertTrue(found)

    def test_forgetting_by_description_archives_the_memory(self):
        manager = self.manager()
        manager.remember("User prefers dark mode in the editor", type="preference")
        result = manager.forget("prefers dark mode editor")
        self.assertEqual(result["forgotten"], 1)
        self.assertEqual(manager.store.counts()["active"], 0)
        self.assertEqual(manager.search("prefers dark mode editor"), [])

    def test_forgetting_by_id(self):
        manager = self.manager()
        stored = manager.remember("User prefers dark mode")
        result = manager.forget(stored["memory"]["id"])
        self.assertEqual(result["forgotten"], 1)

    def test_forgetting_everything_about_a_subject(self):
        manager = self.manager()
        manager.note_document("Biology.pdf")
        manager.remember("User discussed chapter 4", subject="Biology.pdf")
        result = manager.forget("Biology.pdf")
        self.assertGreaterEqual(result["forgotten"], 2)

    def test_forgetting_something_unknown_says_so(self):
        manager = self.manager()
        with self.assertRaises(MemoryOperationError):
            manager.forget("a thing never mentioned")

    def test_updating_a_memory_re_embeds_it(self):
        manager = self.manager()
        stored = manager.remember("User prefers vim")
        memory_id = stored["memory"]["id"]
        manager.update(memory_id, content="User prefers neovim")

        found = manager.search("neovim")
        self.assertTrue(found)
        self.assertEqual(found[0].memory.content, "User prefers neovim")

    def test_an_unknown_type_is_refused_with_the_valid_ones(self):
        manager = self.manager()
        with self.assertRaises(MemoryOperationError) as caught:
            manager.remember("User prefers vim", type="opinion")
        self.assertIn("preference", str(caught.exception))


# --- turn observation ------------------------------------------------------


class ObservationTests(MemoryCase):
    def test_noise_is_never_stored_or_queued(self):
        manager = self.manager()
        for text in ("hi", "thanks", "ok", "cheers"):
            manager.observe_turn(prompt=text, conversation_id=1)
        self.assertEqual(manager.store.counts()["total"], 0)
        self.assertEqual(manager.store.pending_count(), 0)

    def test_a_clear_preference_is_stored_with_no_model(self):
        manager = self.manager()
        outcome = manager.observe_turn(
            prompt="I always use ripgrep instead of grep", conversation_id=1
        )
        self.assertEqual(outcome["stored"], 1)
        stored = manager.store.list_memories()[0]
        self.assertIs(stored.type, MemoryType.PREFERENCE)

    def test_a_temporary_statement_is_not_stored_as_a_preference(self):
        manager = self.manager()
        manager.observe_turn(
            prompt="I always use ripgrep today, just testing it out",
            conversation_id=1,
        )
        stored = manager.store.list_memories(
            statuses=(MemoryStatus.ACTIVE,), limit=10
        )
        self.assertTrue(stored)
        self.assertIs(stored[0].type, MemoryType.TEMPORARY)
        # And it must not surface as the answer to "what do I normally use".
        self.assertNotIn(
            MemoryType.PREFERENCE, [memory.type for memory in stored]
        )

    def test_nothing_is_queued_when_no_processor_is_attached(self):
        manager = self.manager()
        manager.observe_turn(
            prompt="Some ordinary sentence about the weather outside",
            conversation_id=1,
            message_count=4,
        )
        self.assertEqual(manager.store.pending_count(), 0)

    def test_a_document_event_is_recorded_without_its_contents(self):
        manager = self.manager()
        manager.note_document("Biology.pdf", conversation_id=1)
        stored = manager.store.list_memories()[0]
        self.assertIs(stored.type, MemoryType.EVENT)
        self.assertEqual(stored.subject, "Biology.pdf")
        self.assertLess(len(stored.content), 200)


# --- consolidation and conflict -------------------------------------------


class ConsolidationTests(MemoryCase):
    def test_near_identical_memories_are_merged_without_a_model(self):
        manager = self.manager()
        manager.remember("User prefers vim for editing", type="preference")
        manager.remember("User prefers vim for editing code", type="preference")

        outcome = manager.consolidate()
        self.assertGreaterEqual(outcome["merged"] + outcome["superseded"], 1)
        self.assertEqual(manager.store.counts()["active"], 1)

    def test_a_newer_fact_supersedes_an_older_contradicting_one(self):
        manager = self.manager()
        manager.remember("User uses Qwen as the primary model", type="fact")
        time.sleep(0.01)
        manager.remember("User uses Mistral as the primary model", type="fact")

        manager.consolidate()

        active = [m.content for m in manager.store.list_memories()]
        superseded = manager.store.list_memories(statuses=(MemoryStatus.SUPERSEDED,))
        # Exactly one current answer, and the old one kept as history.
        self.assertEqual(len(active), 1)
        self.assertIn("Mistral", active[0])
        self.assertTrue(any("Qwen" in m.content for m in superseded))

    def test_unrelated_memories_are_left_alone(self):
        manager = self.manager()
        manager.remember("User prefers vim for editing", type="preference")
        manager.remember("User lives in a different timezone entirely", type="fact")

        outcome = manager.consolidate()
        self.assertEqual(outcome["merged"], 0)
        self.assertEqual(outcome["superseded"], 0)
        self.assertEqual(manager.store.counts()["active"], 2)

    def test_events_are_never_superseded_by_each_other(self):
        # Two things can both have happened; only current state competes.
        manager = self.manager()
        manager.remember("User added the document Biology.pdf", type="event")
        manager.remember("User added the document Biology.pdf again", type="event")
        manager.consolidate()
        self.assertGreaterEqual(manager.store.counts()["active"], 1)


# --- the context builder ---------------------------------------------------


class ContextBuilderTests(MemoryCase):
    def scored(
        self,
        content: str,
        score: float = 0.9,
        type: MemoryType = MemoryType.FACT,
    ) -> ScoredMemory:
        store = MemoryStore(self.tmp / "agent.db")
        memory_id, _ = store.add(Candidate(content=content, type=type))
        return ScoredMemory(
            memory=store.get(memory_id), score=score, similarity=score, recency=1.0
        )

    def test_the_system_prompt_always_survives(self):
        built = ContextBuilder(budget_tokens=100).build(
            system_prompt="SYSTEM", history=[]
        )
        self.assertEqual(built.messages[0]["role"], "system")
        self.assertIn("SYSTEM", built.messages[0]["content"])

    def test_memories_are_rendered_with_their_type(self):
        built = ContextBuilder(budget_tokens=3000).build(
            system_prompt="SYSTEM",
            history=[{"role": "user", "content": "hello"}],
            memories=[self.scored("User prefers vim")],
        )
        self.assertIn("User prefers vim", built.messages[0]["content"])
        self.assertIn("[fact]", built.messages[0]["content"])

    def test_a_low_confidence_memory_is_marked_unconfirmed(self):
        store = MemoryStore(self.tmp / "agent.db")
        memory_id, _ = store.add(Candidate(content="User might try zsh", confidence=0.3))
        item = ScoredMemory(
            memory=store.get(memory_id), score=0.5, similarity=0.5, recency=1.0
        )
        built = ContextBuilder().build(
            system_prompt="SYSTEM", history=[], memories=[item]
        )
        self.assertIn("unconfirmed", built.messages[0]["content"])

    def test_the_summary_is_included_and_reported(self):
        built = ContextBuilder(budget_tokens=3000).build(
            system_prompt="SYSTEM",
            history=[{"role": "user", "content": "hi"}],
            summary="They were debugging a PDF importer.",
        )
        self.assertTrue(built.summary_used)
        self.assertIn("PDF importer", built.messages[0]["content"])

    def test_memories_cannot_consume_the_whole_budget(self):
        many = [self.scored(f"User fact number {n} " + "x" * 200) for n in range(40)]
        built = ContextBuilder(budget_tokens=1000).build(
            system_prompt="SYSTEM",
            history=[{"role": "user", "content": "a question"}],
            memories=many,
        )
        self.assertLess(len(built.memories_used), 40)
        # The user's actual question still made it in.
        self.assertTrue(any(m["role"] == "user" for m in built.messages))

    def test_old_messages_are_dropped_when_the_budget_is_tight(self):
        history = [
            {"role": "user", "content": f"message {n} " + "y" * 300}
            for n in range(40)
        ]
        built = ContextBuilder(budget_tokens=500).build(
            system_prompt="SYSTEM", history=history
        )
        self.assertGreater(built.messages_dropped, 0)
        self.assertLess(built.messages_kept, 40)
        # The most recent message is the one that survives.
        self.assertIn("message 39", built.messages[-1]["content"])

    def test_the_window_never_starts_on_an_orphaned_tool_result(self):
        # A tool message whose assistant tool_calls were dropped is rejected by
        # the chat template, so the window has to begin at a user message.
        history = [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        built = ContextBuilder(budget_tokens=60).build(
            system_prompt="SYSTEM", history=history
        )
        conversation = built.messages[1:]
        if conversation:
            self.assertEqual(conversation[0]["role"], "user")

    def test_a_minimum_of_recent_messages_always_survives(self):
        history = [
            {"role": "user", "content": "x" * 5000} for _ in range(6)
        ]
        built = ContextBuilder(budget_tokens=50, min_recent_messages=2).build(
            system_prompt="SYSTEM", history=history
        )
        self.assertGreaterEqual(built.messages_kept, 2)

    def test_working_memory_reaches_the_prompt_but_is_never_stored(self):
        working = WorkingMemory(task="Fix the PDF importer", current_document="a.pdf")
        built = ContextBuilder().build(
            system_prompt="SYSTEM", history=[], working=working
        )
        self.assertIn("Fix the PDF importer", built.messages[0]["content"])

        # There is deliberately no table for it: nothing can persist it.
        store = MemoryStore(self.tmp / "agent.db")
        self.assertEqual(store.counts()["total"], 0)


# --- summarisation ---------------------------------------------------------


class SummaryTests(MemoryCase):
    def test_a_short_conversation_is_not_summarised(self):
        manager = self.manager(summarize_after=24)
        manager.attach_processor(object())
        messages = [{"id": n, "role": "user", "content": "hi"} for n in range(5)]
        self.assertFalse(manager.queue_summary(1, messages))
        self.assertEqual(manager.store.pending_count(), 0)

    def test_a_long_conversation_queues_a_summary(self):
        manager = self.manager(summarize_after=10)
        manager.attach_processor(object())
        messages = [
            {"id": n, "role": "user", "content": f"message {n}"} for n in range(30)
        ]
        self.assertTrue(manager.queue_summary(1, messages))
        self.assertEqual(manager.store.pending_count(), 1)

    def test_a_stored_summary_is_returned_for_the_context(self):
        manager = self.manager()
        manager.store.save_summary(1, "They discussed PDFs.", covers_up_to=10, message_count=10)
        self.assertEqual(manager.summary_for(1), "They discussed PDFs.")

    def test_a_summary_is_not_regenerated_for_messages_it_already_covers(self):
        manager = self.manager(summarize_after=10)
        manager.attach_processor(object())
        messages = [
            {"id": n, "role": "user", "content": f"message {n}"} for n in range(30)
        ]
        manager.store.save_summary(1, "Earlier.", covers_up_to=29, message_count=29)
        self.assertFalse(manager.queue_summary(1, messages))


# --- the tools -------------------------------------------------------------


class MemoryToolTests(MemoryCase):
    def registry(self) -> ToolRegistry:
        return ToolRegistry(build_memory_tools(self.manager()))

    def test_all_five_operations_are_exposed(self):
        self.assertEqual(
            self.registry().names(),
            ["forget_memory", "recall", "remember", "search_memory", "update_memory"],
        )

    def test_remember_then_recall_through_the_registry(self):
        manager = self.manager()
        registry = ToolRegistry(build_memory_tools(manager))

        stored = registry.execute(
            "remember",
            {"content": "User prefers vim for editing code", "type": "preference"},
        )
        self.assertTrue(stored.ok)

        found = registry.execute("recall", {"query": "editing code preference"})
        self.assertTrue(found.ok)
        self.assertEqual(found.payload["count"], 1)
        self.assertIn("vim", found.payload["memories"][0]["content"])

    def test_recall_of_something_unknown_is_an_honest_empty_answer(self):
        manager = self.manager()
        manager.remember("User prefers vim", type="preference")
        registry = ToolRegistry(build_memory_tools(manager))
        found = registry.execute("recall", {"query": "photosynthesis"})
        self.assertTrue(found.ok)
        self.assertEqual(found.payload["count"], 0)
        self.assertIn("note", found.payload)

    def test_forget_through_the_registry_actually_removes_it(self):
        manager = self.manager()
        manager.remember("User prefers dark mode in the editor", type="preference")
        registry = ToolRegistry(build_memory_tools(manager))

        result = registry.execute("forget_memory", {"target": "dark mode editor"})
        self.assertTrue(result.ok)
        self.assertEqual(manager.store.counts()["active"], 0)

    def test_search_memory_reports_scores(self):
        manager = self.manager()
        manager.remember("User prefers vim for editing code", type="preference")
        registry = ToolRegistry(build_memory_tools(manager))
        found = registry.execute("search_memory", {"query": "editing code"})
        self.assertTrue(found.ok)
        self.assertIn("score", found.payload["memories"][0])

    def test_update_memory_through_the_registry(self):
        manager = self.manager()
        stored = manager.remember("User prefers vim")
        registry = ToolRegistry(build_memory_tools(manager))
        result = registry.execute(
            "update_memory",
            {"memory_id": stored["memory"]["id"], "content": "User prefers neovim"},
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["memory"]["content"], "User prefers neovim")

    def test_a_bad_argument_is_a_failed_result_not_an_exception(self):
        result = self.registry().execute("remember", {"contents": "typo"})
        self.assertFalse(result.ok)
        self.assertIn("Unknown argument", result.payload["error"])

    def test_the_tool_raises_its_own_error_type(self):
        from tools.memory_tool import MemoryTools

        tools = MemoryTools(self.manager())
        with self.assertRaises(MemoryToolError):
            tools.search_memory("")

    def test_the_descriptions_say_what_not_to_store(self):
        description = self.registry().get_tool("remember").description.lower()
        self.assertIn("do not store", description)
        self.assertIn("greetings", description)


# --- the auxiliary model and the one-model-at-a-time rule ------------------


class ProcessorTests(MemoryCase):
    """The batch worker, against the REAL ModelManager.

    `ManagerHarness` replaces process spawning and health checks and nothing
    else, so `ensure()` here runs the same `_stop_others` path the UI's model
    picker does. That is the point: the rule these tests check is enforced by
    the manager, and testing it against a stub written to agree would prove
    nothing at all.
    """

    def setUp(self):
        super().setUp()
        self.registry_path = write_registry(self.tmp, files_exist=("a.gguf", "b.gguf"))
        self.harness = ManagerHarness(self.registry_path)
        # Every load and stop, in order.
        self.timeline: list[tuple[str, str]] = []
        self._wrap(self.harness)

    def _wrap(self, harness):
        real_ensure, real_stop = harness.ensure, harness.stop

        def ensure(key=None):
            url = real_ensure(key)
            self.timeline.append(("load", key or harness.default_key))
            return url

        def stop(key):
            stopped = real_stop(key)
            if stopped:
                self.timeline.append(("stop", key))
            return stopped

        harness.ensure = ensure
        harness.stop = stop

    def loaded_now(self) -> list[str]:
        """Which chat models the manager currently believes are READY."""
        from models.manager import ModelState

        return [
            status.spec.key
            for status in self.harness.statuses()
            if status.state is ModelState.READY and status.spec.role == "chat"
        ]

    def processor(self, manager, model, *, aux_key="big", batch_size=12):
        return MemoryProcessor(
            manager.store,
            manager.vectors,
            manager=self.harness,
            config=Config(workspace=self.tmp, db_path=self.tmp / "agent.db"),
            aux_key=aux_key,
            client_factory=lambda url: model,
            batch_size=batch_size,
        )

    def test_the_primary_model_is_stopped_before_the_auxiliary_one_starts(self):
        manager = self.manager()
        model = ScriptedModel(
            ['[{"content": "User prefers vim", "type": "preference", '
             '"importance": 0.8, "confidence": 0.9}]']
        )
        manager.attach_processor(self.processor(manager, model))

        # The primary model is loaded and answering, as during a normal turn.
        self.harness.ensure("fast")
        self.assertEqual(self.loaded_now(), ["fast"])

        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "I always use vim"}]},
            1,
        )
        result = manager.maybe_process(force=True)

        self.assertTrue(result["ran"])
        # Nothing is resident afterwards: the auxiliary model is stopped, and
        # the primary is left for the next turn to load.
        self.assertEqual(self.loaded_now(), [])
        self.assertIn(("load", "big"), self.timeline)
        self.assertIn(("stop", "big"), self.timeline)
        # Stopped last, so nothing outlives the batch.
        self.assertEqual(self.timeline[-1], ("stop", "big"))

    def test_the_two_models_are_never_ready_at_the_same_time(self):
        manager = self.manager()
        model = ScriptedModel(["[]"])
        manager.attach_processor(self.processor(manager, model))

        seen: list[list[str]] = []
        real_chat = model.chat

        def chat(messages, *, tools=None):
            # Sampled at the one moment both could possibly be resident.
            seen.append(self.loaded_now())
            return real_chat(messages, tools=tools)

        model.chat = chat

        self.harness.ensure("fast")
        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "something worth a look"}]},
            1,
        )
        manager.maybe_process(force=True)

        self.assertTrue(seen)
        for snapshot in seen:
            self.assertEqual(
                snapshot,
                ["big"],
                f"both models were resident together: {snapshot}",
            )

    def test_many_jobs_are_processed_in_one_model_session(self):
        manager = self.manager()
        model = ScriptedModel(["[]"] * 10)
        manager.attach_processor(self.processor(manager, model))

        for n in range(6):
            manager.store.enqueue(
                JobKind.EXTRACT,
                {"turns": [{"role": "user", "content": f"turn number {n}"}]},
                1,
            )

        result = manager.maybe_process(force=True)
        self.assertTrue(result["ran"])
        self.assertEqual(result["jobs"], 6)

        # One load and one stop for six jobs: the switch is the expensive
        # part, so batching is the whole reason the queue exists.
        loads = [entry for entry in self.timeline if entry == ("load", "big")]
        stops = [entry for entry in self.timeline if entry == ("stop", "big")]
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(stops), 1)
        self.assertEqual(len(model.prompts), 6)

    def test_a_batch_is_refused_while_a_turn_is_running(self):
        manager = self.manager()
        manager.attach_processor(self.processor(manager, ScriptedModel()))
        manager.store.enqueue(
            JobKind.EXTRACT, {"turns": [{"role": "user", "content": "anything"}]}, 1
        )

        result = manager.maybe_process(busy=lambda: True)
        self.assertFalse(result["ran"])
        self.assertIn("turn is running", result["reason"])
        # The job is still there for later, and nothing was loaded.
        self.assertEqual(manager.store.pending_count(), 1)
        self.assertEqual(self.loaded_now(), [])

    def test_a_batch_stops_early_when_a_turn_arrives(self):
        manager = self.manager()
        model = ScriptedModel(["[]"] * 10)
        manager.attach_processor(self.processor(manager, model))
        for n in range(5):
            manager.store.enqueue(
                JobKind.EXTRACT,
                {"turns": [{"role": "user", "content": f"turn {n}"}]},
                1,
            )

        calls = {"n": 0}

        def busy():
            calls["n"] += 1
            return calls["n"] > 2

        result = manager.maybe_process(busy=busy, force=True)
        self.assertTrue(result["stopped_early"])
        # The rest went back on the queue rather than being lost...
        self.assertGreater(manager.store.pending_count(), 0)
        # ...and the auxiliary model was still stopped on the way out.
        self.assertEqual(self.loaded_now(), [])

    def test_the_auxiliary_model_is_stopped_even_when_a_job_explodes(self):
        manager = self.manager()

        class Exploding(ScriptedModel):
            def chat(self, messages, *, tools=None):
                raise RuntimeError("model died")

        manager.attach_processor(self.processor(manager, Exploding()))
        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "anything at all"}]},
            1,
        )

        result = manager.maybe_process(force=True)
        self.assertEqual(result["failed"], 1)
        # The one outcome that would break the next turn's memory ceiling.
        self.assertEqual(self.loaded_now(), [])

    def test_a_failed_job_is_retried_then_left_failed(self):
        manager = self.manager()

        class Exploding(ScriptedModel):
            def chat(self, messages, *, tools=None):
                raise RuntimeError("model died")

        manager.attach_processor(self.processor(manager, Exploding()))
        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "anything at all"}]},
            1,
        )

        for _ in range(4):
            manager.maybe_process(force=True)
        self.assertEqual(manager.store.pending_count(), 0)

    def test_extraction_stores_what_the_model_returns(self):
        manager = self.manager()
        model = ScriptedModel(
            ['[{"content": "User prefers dark mode", "type": "preference", '
             '"importance": 0.8, "confidence": 0.9}]']
        )
        manager.attach_processor(self.processor(manager, model))
        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "I always use dark mode"}]},
            1,
        )

        result = manager.maybe_process(force=True)
        self.assertEqual(result["memories_created"], 1)
        stored = manager.store.list_memories()[0]
        self.assertEqual(stored.content, "User prefers dark mode")
        self.assertIs(stored.type, MemoryType.PREFERENCE)

    def test_a_low_confidence_extraction_never_becomes_a_permanent_fact(self):
        manager = self.manager()
        model = ScriptedModel(
            ['[{"content": "User will switch to zsh", "type": "preference", '
             '"importance": 0.5, "confidence": 0.2}]']
        )
        manager.attach_processor(self.processor(manager, model))
        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "I might switch to zsh"}]},
            1,
        )
        manager.maybe_process(force=True)

        stored = manager.store.list_memories()[0]
        # The model said "preference"; the confidence says otherwise, and an
        # uncertain intention must not become a permanent fact.
        self.assertIs(stored.type, MemoryType.UNCERTAIN)

    def test_the_model_returning_nothing_is_a_normal_outcome(self):
        manager = self.manager()
        manager.attach_processor(self.processor(manager, ScriptedModel(["[]"])))
        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "just chatting about nothing"}]},
            1,
        )

        result = manager.maybe_process(force=True)
        self.assertTrue(result["ran"])
        self.assertEqual(result["memories_created"], 0)
        self.assertEqual(manager.store.counts()["total"], 0)

    def test_summarisation_stores_a_summary(self):
        manager = self.manager()
        manager.attach_processor(
            self.processor(
                manager,
                ScriptedModel(
                    ["They debugged a PDF importer and chose pypdf over pdfminer."]
                ),
            )
        )
        manager.store.enqueue(
            JobKind.SUMMARIZE,
            {
                "turns": [
                    {"role": "user", "content": f"message {n}"} for n in range(8)
                ],
                "covers_up_to": 8,
            },
            1,
        )
        manager.maybe_process(force=True)

        stored = manager.store.get_summary(1)
        self.assertIsNotNone(stored)
        self.assertIn("pypdf", stored["summary"])
        self.assertEqual(stored["covers_up_to"], 8)

    def test_consolidation_by_the_model_merges_two_memories(self):
        manager = self.manager()
        first = manager.remember("User prefers ripgrep", type="preference")
        second = manager.remember(
            "User uses ripgrep because it is fast", type="preference"
        )
        manager.attach_processor(
            self.processor(
                manager,
                ScriptedModel(
                    ['{"content": "User prefers ripgrep because it is fast", '
                     '"keep": "merged"}']
                ),
            )
        )
        manager.store.enqueue(
            JobKind.CONSOLIDATE,
            {
                "first_id": first["memory"]["id"],
                "second_id": second["memory"]["id"],
            },
        )
        manager.maybe_process(force=True)

        active = manager.store.list_memories()
        self.assertEqual(len(active), 1)
        self.assertIn("because it is fast", active[0].content)

    def test_a_missing_auxiliary_model_is_reported_not_raised(self):
        manager = self.manager()
        manager.attach_processor(
            self.processor(manager, ScriptedModel(), aux_key="ghost")
        )
        manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "anything at all"}]},
            1,
        )

        result = manager.maybe_process(force=True)
        self.assertFalse(result["ran"])
        self.assertIn("ghost", result["reason"])
        # The work is not lost; it waits for a model that exists.
        self.assertEqual(manager.store.pending_count(), 1)

    def test_the_whole_system_works_with_no_auxiliary_model_at_all(self):
        # Section 26: the processor improves memory, it is not a dependency.
        manager = self.manager()
        self.assertIsNone(manager.processor)

        manager.remember("User prefers vim for editing code", type="preference")
        manager.observe_turn(prompt="I always use ripgrep instead of grep")
        found = manager.search("prefers vim editing code")

        self.assertTrue(found)
        self.assertEqual(manager.store.counts()["active"], 2)
        self.assertFalse(manager.maybe_process()["ran"])

    def test_jobs_left_running_by_a_crash_are_returned_to_the_queue(self):
        manager = self.manager()
        job_id = manager.store.enqueue(JobKind.EXTRACT, {"turns": []}, 1)
        manager.store.mark_running([job_id])
        self.assertEqual(manager.store.pending_count(), 0)

        # A fresh manager over the same database, as after a restart.
        reopened = self.manager()
        self.assertEqual(reopened.store.pending_count(), 1)

    def test_the_queue_waits_for_enough_work_before_switching_models(self):
        manager = self.manager(queue_high_water=5)
        manager.attach_processor(self.processor(manager, ScriptedModel()))
        manager.store.enqueue(
            JobKind.EXTRACT, {"turns": [{"role": "user", "content": "one"}]}, 1
        )

        ready, why = manager.should_process()
        self.assertFalse(ready)
        self.assertIn("waiting for a batch", why)


    def test_being_deferred_never_uses_up_a_jobs_retries(self):
        # A job postponed because the user was typing must not be treated as a
        # job that failed three times.
        manager = self.manager()
        manager.attach_processor(self.processor(manager, ScriptedModel(["[]"] * 20)))
        job_id = manager.store.enqueue(
            JobKind.EXTRACT,
            {"turns": [{"role": "user", "content": "something worth a look"}]},
            1,
        )

        for _ in range(5):
            manager.maybe_process(busy=lambda: True, force=True)

        pending = manager.store.pending_jobs()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].id, job_id)
        self.assertEqual(pending[0].attempts, 0)

        # And it still runs once the agent is free.
        result = manager.maybe_process(force=True)
        self.assertTrue(result["ran"])
        self.assertEqual(manager.store.pending_count(), 0)


class JsonParsingTests(unittest.TestCase):
    """Small models wrap JSON in prose however firmly they are told not to."""

    def test_plain_json(self):
        self.assertEqual(_json_from('[{"a": 1}]'), [{"a": 1}])

    def test_fenced_json(self):
        self.assertEqual(_json_from('```json\n[{"a": 1}]\n```'), [{"a": 1}])

    def test_json_buried_in_prose(self):
        self.assertEqual(
            _json_from('Sure! Here it is:\n[{"a": 1}]\nHope that helps.'),
            [{"a": 1}],
        )

    def test_unparseable_returns_none(self):
        self.assertIsNone(_json_from("I could not do that."))
        self.assertIsNone(_json_from(""))


# --- the HTTP API ----------------------------------------------------------


class MemoryApiTests(MemoryCase):
    """The routes, against a real store and a fake embedder."""

    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient

        from api.main import create_app
        from tests.test_api import HarnessRuntime
        from tests.test_api import write_registry as api_registry

        registry_path = api_registry(self.tmp)
        config = Config(
            workspace=self.tmp,
            db_path=self.tmp / "chat.db",
            memory_tool_enabled=True,
            memory_store=self.tmp / "vectors",
        )
        self.runtime = HarnessRuntime(config, ManagerHarness(registry_path))

        # Point the runtime's manager at the fake embedder, so no model loads.
        manager = MemoryManager(
            config.db_path,
            store_dir=config.memory_store,
            embedder=self.embedder,
            dimension=DIMENSION,
        )
        self.runtime._memory = manager
        self.memory_manager = manager

        self._client = TestClient(create_app(self.runtime))
        self.client = self._client.__enter__()

    def tearDown(self):
        self._client.__exit__(None, None, None)
        super().tearDown()

    def test_remember_then_list(self):
        stored = self.client.post(
            "/api/memory",
            json={"content": "User prefers vim for editing", "type": "preference"},
        )
        self.assertEqual(stored.status_code, 200)
        self.assertEqual(stored.json()["type"], "preference")

        listed = self.client.get("/api/memory")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["count"], 1)

    def test_search_returns_scores(self):
        self.client.post(
            "/api/memory",
            json={"content": "User prefers vim for editing", "type": "preference"},
        )
        found = self.client.post(
            "/api/memory/search", json={"query": "prefers vim editing"}
        )
        self.assertEqual(found.status_code, 200)
        self.assertIsNotNone(found.json()["memories"][0]["score"])

    def test_an_unrelated_search_returns_nothing(self):
        self.client.post(
            "/api/memory",
            json={"content": "User prefers vim for editing", "type": "preference"},
        )
        found = self.client.post(
            "/api/memory/search", json={"query": "photosynthesis chloroplast"}
        )
        self.assertEqual(found.json()["count"], 0)

    def test_update_and_forget(self):
        stored = self.client.post(
            "/api/memory", json={"content": "User prefers vim"}
        ).json()

        updated = self.client.patch(
            f"/api/memory/{stored['id']}", json={"content": "User prefers neovim"}
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["content"], "User prefers neovim")

        removed = self.client.delete(f"/api/memory/{stored['id']}")
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json()["forgotten"], 1)

    def test_forgetting_something_unknown_is_a_404(self):
        self.assertEqual(
            self.client.delete("/api/memory/never-mentioned").status_code, 404
        )

    def test_updating_with_no_fields_is_a_400(self):
        stored = self.client.post(
            "/api/memory", json={"content": "User prefers vim"}
        ).json()
        self.assertEqual(
            self.client.patch(f"/api/memory/{stored['id']}", json={}).status_code, 400
        )

    def test_stats_reports_the_processor_state(self):
        stats = self.client.get("/api/memory/stats")
        self.assertEqual(stats.status_code, 200)
        body = stats.json()
        self.assertIn("processor", body)
        self.assertFalse(body["processor"]["running"])

    def test_consolidate_is_deterministic_and_needs_no_model(self):
        # Differing only by punctuation is exactly the pair `normalise` leaves
        # alone on purpose - deduplicating it is the embedding's job, and this
        # is where that job gets done.
        for content in (
            "User prefers ripgrep for search",
            "User prefers ripgrep for search.",
        ):
            self.client.post("/api/memory", json={"content": content})

        result = self.client.post("/api/memory/consolidate")
        self.assertEqual(result.status_code, 200)
        self.assertGreaterEqual(
            result.json()["merged"] + result.json()["superseded"], 1
        )

    def test_processing_and_consolidating_are_refused_mid_turn(self):
        class Busy:
            def busy(self):
                return True

            def depth(self):
                return 1

        real = self.runtime.queue
        self.runtime.queue = Busy()
        try:
            self.assertEqual(
                self.client.post("/api/memory/process").status_code, 409
            )
            self.assertEqual(
                self.client.post("/api/memory/consolidate").status_code, 409
            )
            # Reading is always allowed: it loads no model.
            self.assertEqual(self.client.get("/api/memory").status_code, 200)
        finally:
            self.runtime.queue = real

    def test_processing_with_no_queued_work_says_so(self):
        result = self.client.post("/api/memory/process")
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()["ran"])

    def test_the_memory_switches_are_listed(self):
        switches = {
            item["id"]: item
            for item in self.client.get("/api/tools").json()["switches"]
        }
        self.assertIn("memory", switches)
        self.assertIn("memory_processing", switches)
        # Background processing is the sharp end of memory, so it nests under it.
        self.assertEqual(switches["memory_processing"]["depends_on"], "memory")

    def test_turning_memory_off_also_turns_off_its_processing(self):
        self.client.post("/api/tools/memory_processing", json={"enabled": True})
        self.client.post("/api/tools/memory", json={"enabled": False})
        switches = {
            item["id"]: item
            for item in self.client.get("/api/tools").json()["switches"]
        }
        self.assertFalse(switches["memory"]["enabled"])
        self.assertFalse(switches["memory_processing"]["enabled"])


# --- the whole pipeline ----------------------------------------------------


class EndToEndTests(MemoryCase):
    """A turn's context, built from real memories and a real summary."""

    def test_context_carries_memories_summary_and_recent_messages(self):
        manager = self.manager()
        manager.remember("User prefers vim for editing code", type="preference")
        manager.remember("User runs everything locally on an 8 GB laptop", type="fact")
        manager.store.save_summary(
            7, "They were adding PDF support to the indexer.", 10, 10
        )

        history = [
            {"role": "user", "content": "earlier question about something"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "what editing code preference do I have?"},
        ]
        built = manager.build_context(
            system_prompt="SYSTEM",
            history=history,
            query="what editing code preference do I have?",
            conversation_id=7,
        )

        preamble = built.messages[0]["content"]
        self.assertIn("SYSTEM", preamble)
        self.assertIn("PDF support", preamble)
        self.assertIn("vim", preamble)
        self.assertTrue(built.summary_used)
        self.assertTrue(built.memories_used)
        # The user's actual question is still the last message.
        self.assertEqual(built.messages[-1]["role"], "user")

    def test_an_unrelated_question_brings_no_memories_into_context(self):
        manager = self.manager()
        manager.remember("User prefers vim for editing code", type="preference")

        built = manager.build_context(
            system_prompt="SYSTEM",
            history=[{"role": "user", "content": "what is photosynthesis?"}],
            query="what is photosynthesis?",
        )
        self.assertEqual(built.memories_used, [])
        self.assertNotIn("vim", built.messages[0]["content"])

    def test_the_agent_loop_uses_the_context_builder(self):
        from agent.loop import Agent
        from tests.fake_client import FakeQwenClient, text_message

        manager = self.manager()
        manager.remember("User prefers vim for editing code", type="preference")

        client = FakeQwenClient([text_message("done")])
        agent = Agent(
            client,
            Config(workspace=self.tmp, db_path=self.tmp / "agent.db"),
            ToolRegistry([]),
            memory=manager,
            conversation_id=1,
        )
        agent.send("remind me of my editing code preference")

        sent = client.calls[-1]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("vim", sent[0]["content"])
        # And the turn reports what it retrieved, so it is not invisible.
        self.assertTrue(agent.context_report["memories"])

    def test_the_agent_loop_without_memory_is_unchanged(self):
        from agent.loop import Agent
        from tests.fake_client import FakeQwenClient, text_message

        client = FakeQwenClient([text_message("done")])
        agent = Agent(
            client,
            Config(workspace=self.tmp, db_path=self.tmp / "agent.db"),
            ToolRegistry([]),
        )
        agent.send("hello")

        sent = client.calls[-1]
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(sent[1]["content"], "hello")

    def test_a_document_upload_records_an_event_not_the_document(self):
        manager = self.manager()
        manager.note_document("Biology.pdf", conversation_id=1)

        found = manager.search("Biology.pdf document added")
        self.assertTrue(found)
        self.assertIs(found[0].memory.type, MemoryType.EVENT)
        # The event only: nothing resembling the document's contents.
        self.assertLess(len(found[0].memory.content), 200)
