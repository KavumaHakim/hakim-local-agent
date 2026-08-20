"""Git tool tests, run against real throwaway repositories."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from config import Config
from tools.base import ToolRegistry
from tools.git_tool import GitRepository, GitToolError, build_git_tools
from tools.registry import build_default_registry

HAS_GIT = shutil.which("git") is not None


def git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments], cwd=root, check=True,
        capture_output=True, text=True,
    )


def make_repo(root: Path) -> None:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "a.txt").write_text("first\n", encoding="utf-8")
    git(root, "add", "a.txt")
    git(root, "commit", "-q", "-m", "initial commit")


@unittest.skipUnless(HAS_GIT, "git not installed")
class ReadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        make_repo(self.root)
        self.repo = GitRepository(self.root, timeout=30)

    def tearDown(self):
        self._tmp.cleanup()

    def test_is_a_repository(self):
        self.assertTrue(self.repo.is_repository())

    def test_clean_status(self):
        result = self.repo.status()
        self.assertTrue(result["clean"])
        self.assertEqual(result["branch"], "main")
        self.assertEqual(result["changes"], [])

    def test_status_reports_modified_files(self):
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")
        result = self.repo.status()
        self.assertFalse(result["clean"])
        self.assertEqual(result["changes"][0]["path"], "a.txt")
        self.assertEqual(result["changes"][0]["unstaged"], "modified")

    def test_status_reports_untracked_files(self):
        (self.root / "new.txt").write_text("x", encoding="utf-8")
        paths = {c["path"]: c for c in self.repo.status()["changes"]}
        self.assertEqual(paths["new.txt"]["staged"], "untracked")

    def test_log(self):
        result = self.repo.log()
        self.assertEqual(result["count"], 1)
        commit = result["commits"][0]
        self.assertEqual(commit["subject"], "initial commit")
        self.assertEqual(commit["author"], "Test")
        self.assertTrue(commit["hash"])

    def test_log_count_is_clamped(self):
        self.assertEqual(self.repo.log(count=99999)["count"], 1)

    def test_diff_shows_changes(self):
        (self.root / "a.txt").write_text("second\n", encoding="utf-8")
        result = self.repo.diff()
        self.assertIn("a.txt", result["summary"])
        self.assertIn("second", result["patch"])

    def test_branches(self):
        result = self.repo.branches()
        self.assertEqual(result["current"], "main")
        self.assertIn("main", result["branches"])

    def test_not_a_repository_is_explained(self):
        with tempfile.TemporaryDirectory() as plain:
            repo = GitRepository(Path(plain), timeout=30)
            self.assertFalse(repo.is_repository())
            with self.assertRaises(GitToolError) as ctx:
                repo.status()
            self.assertIn("not a git repository", str(ctx.exception))


@unittest.skipUnless(HAS_GIT, "git not installed")
class WriteGatingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        make_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_commit_refused_by_default(self):
        repo = GitRepository(self.root, timeout=30)
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(GitToolError) as ctx:
            repo.commit("nope")
        self.assertIn("AGENT_GIT_ALLOW_WRITES", str(ctx.exception))

    def test_branch_creation_refused_by_default(self):
        repo = GitRepository(self.root, timeout=30)
        with self.assertRaises(GitToolError):
            repo.create_branch("feature")

    def test_write_tools_absent_by_default(self):
        names = {t.name for t in GitRepository(self.root, timeout=30).tools()}
        self.assertNotIn("git_commit", names)
        self.assertNotIn("git_create_branch", names)

    def test_no_remote_or_destructive_tools_exist_at_all(self):
        repo = GitRepository(self.root, timeout=30, allow_writes=True)
        names = {t.name for t in repo.tools()}
        for forbidden in ("push", "pull", "fetch", "reset", "checkout",
                          "clean", "stash", "merge", "rebase"):
            self.assertFalse(
                any(forbidden in name for name in names),
                msg=f"{forbidden} must not be reachable",
            )


@unittest.skipUnless(HAS_GIT, "git not installed")
class CommitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        make_repo(self.root)
        self.repo = GitRepository(self.root, timeout=30, allow_writes=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_commits_tracked_changes(self):
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")
        result = self.repo.commit("update a")

        self.assertTrue(result["success"])
        self.assertEqual(result["files"], ["a.txt"])
        self.assertTrue(self.repo.status()["clean"])
        self.assertEqual(self.repo.log()["commits"][0]["subject"], "update a")

    def test_commits_named_paths_only(self):
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")
        (self.root / "b.txt").write_text("new file\n", encoding="utf-8")
        result = self.repo.commit("just b", paths=["b.txt"])

        self.assertEqual(result["files"], ["b.txt"])
        # a.txt is still modified and uncommitted.
        remaining = [c["path"] for c in self.repo.status()["changes"]]
        self.assertIn("a.txt", remaining)

    def test_untracked_files_are_not_swept_in(self):
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")
        (self.root / "stray.txt").write_text("not mine\n", encoding="utf-8")
        result = self.repo.commit("only tracked")

        self.assertEqual(result["files"], ["a.txt"])
        untracked = [c["path"] for c in self.repo.status()["changes"]]
        self.assertIn("stray.txt", untracked)

    def test_nothing_to_commit_is_reported(self):
        with self.assertRaises(GitToolError) as ctx:
            self.repo.commit("empty")
        self.assertIn("nothing to commit", str(ctx.exception).lower())

    def test_empty_message_refused(self):
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(GitToolError):
            self.repo.commit("   ")

    def test_overlong_message_refused(self):
        (self.root / "a.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(GitToolError):
            self.repo.commit("x" * 600)

    def test_create_branch_does_not_switch(self):
        result = self.repo.create_branch("feature")
        self.assertFalse(result["switched"])
        self.assertEqual(self.repo.branches()["current"], "main")
        self.assertIn("feature", self.repo.branches()["branches"])

    def test_bad_branch_name_refused(self):
        for name in ("-x", "has space", "  "):
            with self.assertRaises(GitToolError, msg=name):
                self.repo.create_branch(name)


@unittest.skipUnless(HAS_GIT, "git not installed")
class PathspecTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        make_repo(self.root)
        self.repo = GitRepository(self.root, timeout=30, allow_writes=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_parent_traversal_refused(self):
        with self.assertRaises(GitToolError) as ctx:
            self.repo.commit("x", paths=["../outside.txt"])
        self.assertIn("climb out", str(ctx.exception))

    def test_absolute_path_refused(self):
        with self.assertRaises(GitToolError):
            self.repo.commit("x", paths=["C:/Windows/win.ini"])

    def test_option_like_path_refused(self):
        with self.assertRaises(GitToolError) as ctx:
            self.repo.diff(path="--exec=calc")
        self.assertIn("looks like an option", str(ctx.exception))


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_unless_enabled(self):
        registry, disabled = build_default_registry(Config(workspace=self.root))
        self.assertNotIn("git_status", registry.names())
        self.assertIn("git", {item.category for item in disabled})

    def test_read_tools_when_enabled(self):
        registry, _ = build_default_registry(
            Config(workspace=self.root, git_tool_enabled=True)
        )
        self.assertIn("git_status", registry.names())
        self.assertNotIn("git_commit", registry.names())

    def test_write_tools_need_the_second_flag(self):
        registry, _ = build_default_registry(
            Config(
                workspace=self.root, git_tool_enabled=True, git_allow_writes=True
            )
        )
        self.assertIn("git_commit", registry.names())

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_failure_through_the_registry_is_structured(self):
        registry = ToolRegistry(
            build_git_tools(self.root, timeout=30, allow_writes=False)
        )
        result = registry.execute("git_status", {})
        self.assertFalse(result.ok)
        self.assertIn("not a git repository", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
