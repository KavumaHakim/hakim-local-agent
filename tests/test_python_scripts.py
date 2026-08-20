"""run_python_file: restricted mode, unrestricted mode, and the guard between."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import Config
from tools.base import ToolRegistry
from tools.filesystem import WorkspaceFiles
from tools.python_tool import (
    PROJECT_ROOT,
    PythonToolError,
    build_python_file_tool,
    run_python_file,
)
from tools.registry import build_default_registry


def run(tmp: Path, path: str, unrestricted=False, timeout=25.0):
    return run_python_file(
        path,
        workspace=WorkspaceFiles(tmp),
        timeout=timeout,
        max_output_chars=4000,
        unrestricted=unrestricted,
    )


class RestrictedModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, body):
        (self.tmp / name).write_text(body, encoding="utf-8")
        return name

    def test_runs_a_simple_script(self):
        self.write("calc.py", "print(sum(range(10)))")
        result = run(self.tmp, "calc.py")
        self.assertTrue(result["success"])
        self.assertEqual(result["output"].strip(), "45")

    def test_preloaded_modules_work(self):
        self.write("m.py", "print(round(math.sqrt(2), 3))")
        self.assertIn("1.414", run(self.tmp, "m.py")["output"])

    def test_a_script_on_disk_gets_no_extra_privileges(self):
        # The same code refused inline must be refused from a file.
        self.write("bad.py", "import os\nprint(os.getcwd())")
        with self.assertRaises(PythonToolError) as ctx:
            run(self.tmp, "bad.py")
        self.assertIn("Imports are not allowed", str(ctx.exception))

    def test_file_access_still_refused(self):
        self.write("bad.py", "open('secret.txt')")
        with self.assertRaises(PythonToolError):
            run(self.tmp, "bad.py")

    def test_runtime_error_is_captured(self):
        self.write("boom.py", "print(1/0)")
        result = run(self.tmp, "boom.py")
        self.assertFalse(result["success"])
        self.assertIn("ZeroDivisionError", result["error"])


class PathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file(self):
        with self.assertRaises(PythonToolError) as ctx:
            run(self.tmp, "nope.py")
        self.assertIn("No such file", str(ctx.exception))

    def test_non_python_file_refused(self):
        (self.tmp / "notes.txt").write_text("hello", encoding="utf-8")
        with self.assertRaises(PythonToolError) as ctx:
            run(self.tmp, "notes.txt")
        self.assertIn("not a .py file", str(ctx.exception))

    def test_workspace_escape_refused(self):
        with self.assertRaises(PythonToolError) as ctx:
            run(self.tmp, "../outside.py")
        self.assertIn("outside the workspace", str(ctx.exception))

    def test_absolute_path_outside_refused(self):
        with self.assertRaises(PythonToolError):
            run(self.tmp, str(self.tmp.parent / "x.py"))


class UnrestrictedModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_imports_work(self):
        (self.tmp / "s.py").write_text(
            "import json\nprint(json.dumps({'ok': True}))", encoding="utf-8"
        )
        result = run(self.tmp, "s.py", unrestricted=True)
        self.assertTrue(result["success"])
        self.assertIn('"ok": true', result["stdout"])
        self.assertEqual(result["exit_code"], 0)

    def test_working_directory_is_the_workspace(self):
        (self.tmp / "data.txt").write_text("payload", encoding="utf-8")
        (self.tmp / "s.py").write_text(
            "print(open('data.txt').read())", encoding="utf-8"
        )
        self.assertIn("payload", run(self.tmp, "s.py", unrestricted=True)["stdout"])

    def test_nonzero_exit_is_reported_not_raised(self):
        (self.tmp / "s.py").write_text("raise SystemExit(3)", encoding="utf-8")
        result = run(self.tmp, "s.py", unrestricted=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 3)

    def test_traceback_comes_back_on_stderr(self):
        (self.tmp / "s.py").write_text("raise ValueError('nope')", encoding="utf-8")
        result = run(self.tmp, "s.py", unrestricted=True)
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("ValueError", result["stderr"])

    def test_timeout_is_enforced(self):
        (self.tmp / "s.py").write_text(
            "while True:\n    pass", encoding="utf-8"
        )
        with self.assertRaises(PythonToolError) as ctx:
            run(self.tmp, "s.py", unrestricted=True, timeout=3.0)
        self.assertIn("timeout", str(ctx.exception))


class SelfProtectionTests(unittest.TestCase):
    """Unrestricted Python must not run against the agent's own directory."""

    def test_refused_when_workspace_is_the_project(self):
        with self.assertRaises(PythonToolError) as ctx:
            run_python_file(
                "main.py",
                workspace=WorkspaceFiles(PROJECT_ROOT),
                timeout=5,
                max_output_chars=100,
                unrestricted=True,
            )
        self.assertIn("workspace is the", str(ctx.exception))

    def test_restricted_mode_is_not_blocked_by_the_workspace(self):
        # The screen and stripped child still apply, so restricted mode in the
        # project directory is the same risk as an inline snippet, which is
        # already permitted. main.py is refused - but for importing, not for
        # where it lives.
        with self.assertRaises(PythonToolError) as ctx:
            run_python_file(
                "main.py",
                workspace=WorkspaceFiles(PROJECT_ROOT),
                timeout=5,
                max_output_chars=100,
                unrestricted=False,
            )
        self.assertIn("Imports are not allowed", str(ctx.exception))
        self.assertNotIn("workspace is the", str(ctx.exception))


class ToolWiringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_description_states_the_mode(self):
        workspace = WorkspaceFiles(self.tmp)
        restricted = build_python_file_tool(
            workspace=workspace, timeout=5, max_output_chars=100,
            unrestricted=False,
        )
        unrestricted = build_python_file_tool(
            workspace=workspace, timeout=5, max_output_chars=100,
            unrestricted=True,
        )
        self.assertIn("no imports", restricted.description)
        self.assertIn("Full Python", unrestricted.description)

    def test_registered_with_the_python_tool(self):
        registry, _ = build_default_registry(
            Config(workspace=self.tmp, python_tool_enabled=True)
        )
        self.assertIn("run_python_file", registry.names())
        self.assertIn("run_python", registry.names())

    def test_absent_when_python_is_off(self):
        registry, _ = build_default_registry(Config(workspace=self.tmp))
        self.assertNotIn("run_python_file", registry.names())

    def test_refusal_through_the_registry_is_structured(self):
        tool = build_python_file_tool(
            workspace=WorkspaceFiles(self.tmp), timeout=5,
            max_output_chars=100, unrestricted=False,
        )
        result = ToolRegistry([tool]).execute(
            "run_python_file", {"path": "../escape.py"}
        )
        self.assertFalse(result.ok)
        self.assertIn("outside the workspace", result.payload["error"])


class WriteThenRunTests(unittest.TestCase):
    """The pairing this was built for: write a script, then run it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.files = WorkspaceFiles(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_then_run_restricted(self):
        self.files.write_text_file(
            "report.py", "total = sum(x * x for x in range(1, 11))\nprint(total)"
        )
        self.assertEqual(run(self.tmp, "report.py")["output"].strip(), "385")

    def test_write_then_run_unrestricted(self):
        self.files.create_directory("scripts")
        self.files.write_text_file(
            "scripts/go.py",
            "import statistics\nprint(statistics.median([3, 1, 2]))",
        )
        result = run(self.tmp, "scripts/go.py", unrestricted=True)
        self.assertEqual(result["stdout"].strip(), "2")


if __name__ == "__main__":
    unittest.main()
