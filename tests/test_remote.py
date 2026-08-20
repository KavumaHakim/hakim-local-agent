"""Hosted-model tests. No network, no keys, no provider.

Everything that would touch the internet is replaced: the connectivity probe
is stubbed and the HTTP session is a fake. A test that only passed when this
machine had internet would be worse than no test - and would fail on a train.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from models.connectivity import Connectivity
from models.manager import ModelManagerError, ModelState
from models.remote import MissingKeyError, RemoteClient, RemoteHTTPError
from tests.test_manager import ManagerHarness

KEY_VAR = "TEST_REMOTE_KEY"


def write_registry(tmp: Path) -> Path:
    exe = tmp / "llama-server.exe"
    exe.write_text("x", encoding="utf-8")
    (tmp / "local.gguf").write_bytes(b"gguf")
    registry = {
        "server_exe": str(exe),
        "models_dir": str(tmp),
        "default": "local",
        "max_active": 1,
        "idle_timeout_seconds": 0,
        "models": [
            {"key": "local", "label": "Local 2B", "file": "local.gguf",
             "port": 8080, "min_free_mb": 0},
            {"key": "hosted", "label": "Hosted 120B", "provider": "testcloud",
             "model": "cloud-120b", "api_key_env": KEY_VAR,
             "base_url": "https://cloud.invalid/v1/"},
        ],
    }
    path = tmp / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.manager = ManagerHarness(write_registry(Path(self._tmp.name)))
        self._previous = os.environ.pop(KEY_VAR, None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._previous is None:
            os.environ.pop(KEY_VAR, None)
        else:
            os.environ[KEY_VAR] = self._previous

    def test_a_hosted_entry_needs_no_file_or_port(self):
        spec = self.manager.get_spec("hosted")
        self.assertTrue(spec.remote)
        self.assertEqual(spec.provider, "testcloud")
        self.assertEqual(spec.model, "cloud-120b")
        self.assertEqual(spec.port, 0)

    def test_the_base_url_loses_its_trailing_slash(self):
        """Otherwise every request path becomes a double slash."""
        self.assertEqual(self.manager.get_spec("hosted").url, "https://cloud.invalid/v1")

    def test_availability_follows_the_key(self):
        spec = self.manager.get_spec("hosted")
        self.assertFalse(spec.has_key)
        self.assertFalse(spec.available)

        os.environ[KEY_VAR] = "secret"
        self.assertTrue(spec.has_key)
        self.assertTrue(spec.available)

    def test_whitespace_is_not_a_key(self):
        os.environ[KEY_VAR] = "   "
        self.assertFalse(self.manager.get_spec("hosted").has_key)

    def test_the_key_itself_is_never_stored_on_the_spec(self):
        """So it cannot be serialised out to the UI by accident."""
        os.environ[KEY_VAR] = "super-secret-value"
        spec = self.manager.get_spec("hosted")
        self.assertNotIn("super-secret-value", repr(spec))

    def test_hosted_models_do_not_share_the_port_check(self):
        """They all report port 0, which must not read as a collision."""
        # Two hosted models, both port 0. Building the manager is the assertion.
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)
        registry = json.loads(write_registry(tmp).read_text(encoding="utf-8"))
        registry["models"].append(
            {"key": "hosted2", "label": "Other", "provider": "testcloud",
             "model": "other", "api_key_env": KEY_VAR,
             "base_url": "https://other.invalid/v1"}
        )
        path = tmp / "models.json"
        path.write_text(json.dumps(registry), encoding="utf-8")
        manager = ManagerHarness(path)
        self.assertEqual(len(manager.specs()), 3)


class LifecycleTests(RegistryTests):
    def test_ready_when_keyed_and_stopped_otherwise(self):
        self.assertIs(self.manager.status("hosted").state, ModelState.STOPPED)
        self.assertIn(KEY_VAR, self.manager.status("hosted").error)

        os.environ[KEY_VAR] = "secret"
        self.assertIs(self.manager.status("hosted").state, ModelState.READY)

    def test_ensure_without_a_key_says_which_variable(self):
        with self.assertRaises(ModelManagerError) as ctx:
            self.manager.ensure("hosted")
        self.assertIn(KEY_VAR, str(ctx.exception))

    def test_ensure_starts_no_process(self):
        os.environ[KEY_VAR] = "secret"
        url = self.manager.ensure("hosted")
        self.assertEqual(url, "https://cloud.invalid/v1")
        self.assertEqual(self.manager.spawned, [])

    def test_a_hosted_model_does_not_evict_the_local_one(self):
        """It uses no RAM, so it has no business joining the rotation - and
        keeping the local model warm means switching back costs nothing."""
        os.environ[KEY_VAR] = "secret"
        self.manager.ensure("local")
        self.assertIs(self.manager.status("local").state, ModelState.READY)

        self.manager.ensure("hosted")
        self.assertIs(self.manager.status("local").state, ModelState.READY)

    def test_stopping_a_hosted_model_reports_that_there_was_nothing_to_stop(self):
        os.environ[KEY_VAR] = "secret"
        self.assertFalse(self.manager.stop("hosted"))

    def test_active_key_ignores_hosted_models(self):
        os.environ[KEY_VAR] = "secret"
        self.assertIsNone(self.manager.active_key())
        self.manager.ensure("local")
        self.assertEqual(self.manager.active_key(), "local")


class ConnectivityTests(unittest.TestCase):
    class Probing(Connectivity):
        def __init__(self, result=True, **kwargs):
            super().__init__(**kwargs)
            self.result = result
            self.probes = 0

        def _probe(self) -> bool:
            self.probes += 1
            return self.result

    def test_the_answer_is_cached(self):
        """A probe per model per listing is what made /api/models slow before."""
        checker = self.Probing(ttl=60)
        for _ in range(10):
            checker.online()
        self.assertEqual(checker.probes, 1)

    def test_force_re_probes(self):
        checker = self.Probing(ttl=60)
        checker.online()
        checker.online(force=True)
        self.assertEqual(checker.probes, 2)

    def test_invalidate_makes_the_next_call_re_probe(self):
        checker = self.Probing(ttl=60)
        checker.online()
        checker.invalidate()
        checker.online()
        self.assertEqual(checker.probes, 2)

    def test_an_expired_cache_re_probes(self):
        checker = self.Probing(ttl=0)
        checker.online()
        checker.online()
        self.assertEqual(checker.probes, 2)

    def test_offline_is_reported_as_offline(self):
        self.assertFalse(self.Probing(result=False).online())


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.manager = ManagerHarness(write_registry(Path(self._tmp.name)))
        self.spec = self.manager.get_spec("hosted")
        self._previous = os.environ.pop(KEY_VAR, None)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._previous is None:
            os.environ.pop(KEY_VAR, None)
        else:
            os.environ[KEY_VAR] = self._previous

    def test_a_local_spec_is_refused(self):
        with self.assertRaises(ValueError):
            RemoteClient(self.manager.get_spec("local"))

    def test_a_missing_key_is_reported_before_any_request(self):
        client = RemoteClient(self.spec)
        with self.assertRaises(MissingKeyError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn(KEY_VAR, str(ctx.exception))

    def test_the_payload_names_the_provider_model_not_our_key(self):
        os.environ[KEY_VAR] = "secret"
        client = RemoteClient(self.spec)
        payload = client._payload(
            [{"role": "user", "content": "hi"}],
            [{"type": "function", "function": {"name": "calculate"}}],
        )
        self.assertEqual(payload["model"], "cloud-120b")
        self.assertEqual(len(payload["tools"]), 1)

    def test_error_text_never_carries_the_key(self):
        """Some providers echo the request back in an error body, and that must
        not travel into a log, an SSE event or the browser."""
        os.environ[KEY_VAR] = "super-secret-value"
        client = RemoteClient(self.spec)
        scrubbed = client._scrub('{"error": "bad key super-secret-value"}')
        self.assertNotIn("super-secret-value", scrubbed)
        self.assertIn("***", scrubbed)

    def test_provider_errors_name_the_provider_not_llama_cpp(self):
        """A Cerebras 402 once read 'llama.cpp server returned HTTP 402',
        which sends you to inspect a local process that is working fine."""
        error = RemoteHTTPError("Cloud 120B", 402, '{"message": "billing"}')
        self.assertIn("Cloud 120B", str(error))
        self.assertNotIn("llama.cpp", str(error))
        self.assertIn("credit or quota", str(error))

    def test_each_status_gets_the_hint_that_matches_it(self):
        cases = {
            401: "key was rejected",
            402: "credit or quota",
            404: "model id",
            429: "Rate limited",
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                self.assertIn(expected, str(RemoteHTTPError("X", status, "")))

    def test_an_unmapped_status_still_reports_the_number(self):
        self.assertIn("503", str(RemoteHTTPError("X", 503, "")))

    def test_the_authorization_header_is_built_from_the_environment(self):
        os.environ[KEY_VAR] = "secret"
        headers = RemoteClient(self.spec)._headers()
        self.assertEqual(headers["Authorization"], "Bearer secret")


if __name__ == "__main__":
    unittest.main()
