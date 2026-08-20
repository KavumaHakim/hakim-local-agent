"""Memory store and tools."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import Config
from memory_store import MAX_VALUE_LENGTH, MemoryStore
from tools.base import ToolRegistry
from tools.memory_tool import MemoryToolError, MemoryTools, build_memory_tools
from tools.registry import build_default_registry


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "agent.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_remember_and_get(self):
        self.store.remember("editor", "prefers VS Code")
        self.assertEqual(self.store.get("editor").value, "prefers VS Code")

    def test_keys_are_normalised(self):
        self.store.remember("  Preferred   Editor ", "vim")
        self.assertEqual(self.store.get("preferred editor").value, "vim")

    def test_remembering_again_replaces(self):
        self.store.remember("editor", "vim")
        result = self.store.remember("editor", "emacs")
        self.assertTrue(result["replaced"])
        self.assertEqual(self.store.get("editor").value, "emacs")
        self.assertEqual(self.store.count(), 1)

    def test_created_at_survives_an_update(self):
        self.store.remember("k", "one")
        created = self.store.get("k").created_at
        self.store.remember("k", "two")
        self.assertEqual(self.store.get("k").created_at, created)

    def test_missing_key_returns_none(self):
        self.assertIsNone(self.store.get("absent"))

    def test_empty_key_refused(self):
        with self.assertRaises(ValueError):
            self.store.remember("  ", "x")

    def test_empty_value_refused(self):
        with self.assertRaises(ValueError):
            self.store.remember("k", "   ")

    def test_overlong_value_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.store.remember("k", "x" * (MAX_VALUE_LENGTH + 1))
        self.assertIn("not a transcript", str(ctx.exception))

    def test_overlong_key_refused(self):
        with self.assertRaises(ValueError):
            self.store.remember("k" * 200, "x")

    def test_reopening_keeps_memories(self):
        self.store.remember("k", "durable")
        again = MemoryStore(self.store.path)
        self.assertEqual(again.get("k").value, "durable")


class RecallTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "agent.db")
        self.store.remember("editor", "prefers VS Code")
        self.store.remember("models", "Qwen3.5 2B for quick work")
        self.store.remember("machine", "8 GB RAM, no GPU")

    def tearDown(self):
        self._tmp.cleanup()

    def test_recall_everything(self):
        self.assertEqual(self.store.recall()["count"], 3)

    def test_recall_matches_values(self):
        result = self.store.recall("Qwen")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["memories"][0]["key"], "models")

    def test_recall_matches_keys(self):
        self.assertEqual(self.store.recall("machine")["count"], 1)

    def test_recall_is_case_insensitive_on_values(self):
        # SQLite LIKE is case-insensitive for ASCII by default.
        self.assertEqual(self.store.recall("qwen")["count"], 1)

    def test_no_match_is_empty_not_an_error(self):
        result = self.store.recall("nothing like this")
        self.assertTrue(result["success"])
        self.assertEqual(result["memories"], [])

    def test_limit_is_respected(self):
        self.assertEqual(self.store.recall(limit=2)["count"], 2)

    def test_most_recently_updated_comes_first(self):
        self.store.remember("editor", "actually prefers vim")
        self.assertEqual(self.store.recall()["memories"][0]["key"], "editor")


class ForgetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "agent.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_forget_removes_one(self):
        self.store.remember("a", "one")
        self.store.remember("b", "two")
        self.assertTrue(self.store.forget("a")["success"])
        self.assertIsNone(self.store.get("a"))
        self.assertIsNotNone(self.store.get("b"))

    def test_forgetting_something_absent_is_reported(self):
        result = self.store.forget("never stored")
        self.assertFalse(result["success"])
        self.assertIn("Nothing stored", result["error"])

    def test_purge_clears_all(self):
        self.store.remember("a", "one")
        self.store.purge()
        self.assertEqual(self.store.count(), 0)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "agent.db"
        self.tools = MemoryTools(MemoryStore(self.path))

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_through_the_tools(self):
        self.tools.remember("project", "Hakim AI System")
        result = self.tools.recall("Hakim")
        self.assertEqual(result["memories"][0]["key"], "project")

    def test_bad_input_becomes_a_tool_error(self):
        with self.assertRaises(MemoryToolError):
            self.tools.remember("", "x")

    def test_three_tools_are_exposed(self):
        names = {tool.name for tool in self.tools.tools()}
        self.assertEqual(names, {"remember", "recall", "forget"})

    def test_registry_reports_failure_structurally(self):
        registry = ToolRegistry(build_memory_tools(self.path))
        result = registry.execute("remember", {"key": "", "value": "x"})
        self.assertFalse(result.ok)
        self.assertIn("non-empty", result.payload["error"])

    def test_recall_needs_no_arguments(self):
        registry = ToolRegistry(build_memory_tools(self.path))
        self.assertTrue(registry.execute("recall", {}).ok)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_unless_enabled(self):
        registry, disabled = build_default_registry(
            Config(workspace=self.root, db_path=self.root / "a.db")
        )
        self.assertNotIn("remember", registry.names())
        self.assertIn("memory", {item.category for item in disabled})

    def test_registered_when_enabled(self):
        registry, _ = build_default_registry(
            Config(
                workspace=self.root,
                db_path=self.root / "a.db",
                memory_tool_enabled=True,
            )
        )
        for name in ("remember", "recall", "forget"):
            self.assertIn(name, registry.names())

    def test_shares_the_history_database_file(self):
        db = self.root / "shared.db"
        build_default_registry(
            Config(workspace=self.root, db_path=db, memory_tool_enabled=True)
        )
        self.assertTrue(db.exists())


if __name__ == "__main__":
    unittest.main()
