"""Filesystem write tests.

Two separate questions: is the path inside the workspace (same check reading
uses), and should a path inside the workspace be written to at all.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import Config
from tools.base import ToolRegistry
from tools.filesystem import (
    PROJECT_ROOT,
    FilesystemToolError,
    WorkspaceFiles,
)
from tools.registry import build_default_registry


class WriteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.files = WorkspaceFiles(self.root, max_write_bytes=1000)

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_a_file(self):
        result = self.files.write_text_file("notes.txt", "hello")
        self.assertTrue(result["success"])
        self.assertFalse(result["replaced"])
        self.assertEqual(result["bytes_written"], 5)
        self.assertEqual((self.root / "notes.txt").read_text(encoding="utf-8"), "hello")

    def test_round_trips_through_read(self):
        self.files.write_text_file("a.txt", "round trip")
        self.assertEqual(self.files.read_text_file("a.txt")["content"], "round trip")

    def test_unicode_is_written_as_utf8(self):
        self.files.write_text_file("u.txt", "café ☕")
        self.assertEqual(self.files.read_text_file("u.txt")["content"], "café ☕")

    def test_existing_file_is_not_clobbered_by_default(self):
        self.files.write_text_file("a.txt", "original")
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.write_text_file("a.txt", "replacement")
        self.assertIn("overwrite=true", str(ctx.exception))
        # The original survives.
        self.assertEqual(self.files.read_text_file("a.txt")["content"], "original")

    def test_overwrite_when_asked(self):
        self.files.write_text_file("a.txt", "original")
        result = self.files.write_text_file("a.txt", "replacement", overwrite=True)
        self.assertTrue(result["replaced"])
        self.assertEqual(
            self.files.read_text_file("a.txt")["content"], "replacement"
        )

    def test_no_temp_file_is_left_behind(self):
        self.files.write_text_file("a.txt", "x")
        leftovers = [p.name for p in self.root.iterdir() if "agent-tmp" in p.name]
        self.assertEqual(leftovers, [])

    def test_missing_parent_is_reported(self):
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.write_text_file("nope/a.txt", "x")
        self.assertIn("does not exist", str(ctx.exception))

    def test_writing_over_a_directory_is_refused(self):
        (self.root / "sub").mkdir()
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.write_text_file("sub", "x", overwrite=True)
        self.assertIn("is a directory", str(ctx.exception))

    def test_oversized_content_refused(self):
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.write_text_file("big.txt", "x" * 1001)
        self.assertIn("over the", str(ctx.exception))
        self.assertFalse((self.root / "big.txt").exists())

    def test_non_string_content_refused(self):
        with self.assertRaises(FilesystemToolError):
            self.files.write_text_file("a.txt", 123)


class WorkspaceEscapeTests(unittest.TestCase):
    """Writing uses exactly the same containment check as reading."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.files = WorkspaceFiles(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_relative_escape_refused(self):
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.write_text_file("../escaped.txt", "x")
        self.assertIn("outside the workspace", str(ctx.exception))

    def test_deep_escape_refused(self):
        with self.assertRaises(FilesystemToolError):
            self.files.write_text_file("../../escaped.txt", "x")

    def test_absolute_path_refused(self):
        with self.assertRaises(FilesystemToolError):
            self.files.write_text_file(str(self.root.parent / "x.txt"), "x")

    def test_directory_creation_escape_refused(self):
        with self.assertRaises(FilesystemToolError):
            self.files.create_directory("../escaped")


class SelfProtectionTests(unittest.TestCase):
    """The agent must not be able to rewrite its own guards."""

    def setUp(self):
        # Workspace is the project itself, which is the default.
        self.files = WorkspaceFiles(PROJECT_ROOT)

    def test_cannot_rewrite_the_trust_boundary(self):
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.write_text_file("tools/base.py", "# neutered")
        self.assertIn("own source", str(ctx.exception))

    def test_cannot_rewrite_the_registry(self):
        with self.assertRaises(FilesystemToolError):
            self.files.write_text_file("tools/registry.py", "x")

    def test_cannot_rewrite_config(self):
        with self.assertRaises(FilesystemToolError):
            self.files.write_text_file("config.py", "x")

    def test_cannot_rewrite_the_agent_loop(self):
        with self.assertRaises(FilesystemToolError):
            self.files.write_text_file("agent/loop.py", "x")

    def test_cannot_rewrite_the_model_registry_file(self):
        with self.assertRaises(FilesystemToolError):
            self.files.write_text_file("models.json", "{}")

    def test_cannot_write_into_git(self):
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.write_text_file(".git/config", "x")
        self.assertIn("Refusing to write inside", str(ctx.exception))

    def test_cannot_create_directories_in_protected_areas(self):
        with self.assertRaises(FilesystemToolError):
            self.files.create_directory("tools/evil")

    def test_unprotected_areas_are_still_writable(self):
        # A scratch directory in the project is fine; only the source is not.
        # Cleaned up afterwards: a test must not leave anything behind in the
        # real project, which an earlier version of this one did.
        scratch = PROJECT_ROOT / "tmp-write-check"
        try:
            result = self.files.create_directory(scratch.name)
            self.assertTrue(result["success"])
            self.assertTrue(scratch.is_dir())
        finally:
            if scratch.is_dir():
                scratch.rmdir()


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.files = WorkspaceFiles(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_a_directory(self):
        result = self.files.create_directory("reports")
        self.assertTrue(result["created"])
        self.assertTrue((self.root / "reports").is_dir())

    def test_creates_parents(self):
        self.files.create_directory("a/b/c")
        self.assertTrue((self.root / "a" / "b" / "c").is_dir())

    def test_existing_directory_is_not_an_error(self):
        self.files.create_directory("reports")
        result = self.files.create_directory("reports")
        self.assertTrue(result["success"])
        self.assertFalse(result["created"])

    def test_existing_file_blocks_a_directory(self):
        self.files.write_text_file("thing", "x")
        with self.assertRaises(FilesystemToolError) as ctx:
            self.files.create_directory("thing")
        self.assertIn("is a file", str(ctx.exception))

    def test_write_into_a_created_directory(self):
        self.files.create_directory("out")
        self.files.write_text_file("out/report.md", "# Report")
        self.assertEqual(
            self.files.read_text_file("out/report.md")["content"], "# Report"
        )


class NoDestructiveOperationsTests(unittest.TestCase):
    def test_the_module_exposes_no_delete_or_rename(self):
        names = dir(WorkspaceFiles)
        for forbidden in ("delete", "remove", "unlink", "rename", "move", "chmod"):
            self.assertFalse(
                any(forbidden in name for name in names),
                msg=f"{forbidden} should not exist on WorkspaceFiles",
            )

    def test_write_tools_are_only_create_and_write(self):
        files = WorkspaceFiles(PROJECT_ROOT)
        names = {tool.name for tool in files.write_tools()}
        self.assertEqual(names, {"write_text_file", "create_directory"})


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_unless_enabled(self):
        registry, disabled = build_default_registry(Config(workspace=self.root))
        self.assertNotIn("write_text_file", registry.names())
        self.assertIn("file writes", {item.category for item in disabled})

    def test_registered_when_enabled(self):
        registry, disabled = build_default_registry(
            Config(workspace=self.root, file_writes_enabled=True)
        )
        self.assertIn("write_text_file", registry.names())
        self.assertIn("create_directory", registry.names())
        self.assertNotIn("file writes", {item.category for item in disabled})

    def test_refusal_through_the_registry_is_structured(self):
        files = WorkspaceFiles(self.root)
        registry = ToolRegistry(files.write_tools())
        result = registry.execute(
            "write_text_file", {"path": "../escape.txt", "content": "x"}
        )
        self.assertFalse(result.ok)
        self.assertIn("outside the workspace", result.payload["error"])

    def test_missing_content_argument_is_reported(self):
        files = WorkspaceFiles(self.root)
        registry = ToolRegistry(files.write_tools())
        result = registry.execute("write_text_file", {"path": "a.txt"})
        self.assertFalse(result.ok)
        self.assertIn("Missing required argument", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
