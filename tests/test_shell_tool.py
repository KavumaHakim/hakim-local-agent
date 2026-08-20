"""Terminal tool tests.

Most of these are refusals. The point of the tool is what it will not do, so
that is what the suite spends its time on. The few that execute run only `git`
and expect it to be present.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from config import Config
from tools.base import ToolRegistry
from tools.registry import build_default_registry
from tools.shell_tool import (
    COMMANDS,
    ShellRunner,
    ShellToolError,
    build_shell_tool,
    tokenize,
    validate,
)

HAS_GIT = shutil.which("git") is not None


def runner(tmp: Path, **kwargs) -> ShellRunner:
    kwargs.setdefault("timeout", 20.0)
    kwargs.setdefault("max_output_chars", 4000)
    return ShellRunner(tmp, **kwargs)


class TokenizeTests(unittest.TestCase):
    def test_simple_split(self):
        self.assertEqual(tokenize("git status"), ["git", "status"])

    def test_quotes_are_stripped(self):
        self.assertEqual(
            tokenize('git log --pretty="%h %s"'),
            ["git", "log", "--pretty=%h %s"],
        )

    def test_backslashes_are_refused_with_guidance(self):
        # POSIX splitting would silently eat them, so they are refused instead.
        with self.assertRaises(ShellToolError) as ctx:
            tokenize(r"git show sub\file.txt")
        self.assertIn("forward slashes", str(ctx.exception))

    def test_empty_is_rejected(self):
        for value in ("", "   ", None):
            with self.assertRaises(ShellToolError):
                tokenize(value)

    def test_newlines_are_rejected(self):
        with self.assertRaises(ShellToolError) as ctx:
            tokenize("git status\ngit log")
        self.assertIn("single command", str(ctx.exception))

    def test_overlong_is_rejected(self):
        with self.assertRaises(ShellToolError):
            tokenize("git " + "x" * 600)


class AllowlistTests(unittest.TestCase):
    def check(self, command):
        return validate(command, COMMANDS)

    def test_allowed_git_verbs(self):
        for command in (
            "git status",
            "git log --oneline -5",
            "git diff HEAD",
            "git branch -a",
            "git rev-parse --short HEAD",
        ):
            self.assertEqual(self.check(command)[0], "git", msg=command)

    def test_unknown_program_refused(self):
        with self.assertRaises(ShellToolError) as ctx:
            self.check("curl https://example.com")
        self.assertIn("not an allowed command", str(ctx.exception))

    def test_writing_git_verbs_refused(self):
        for command in (
            "git commit -m x",
            "git push",
            "git reset --hard",
            "git checkout main",
            "git clean -fd",
            "git rebase main",
            "git stash",
            "git pull",
            "git fetch",
        ):
            with self.assertRaises(ShellToolError, msg=command):
                self.check(command)

    def test_git_config_is_refused(self):
        # It writes as readily as it reads.
        with self.assertRaises(ShellToolError):
            self.check("git config user.email attacker@example.com")

    def test_path_to_binary_refused(self):
        for command in (
            "C:/Windows/System32/cmd.exe /c dir",
            "./evil.exe",
            "../git status",
            "bin/git status",
        ):
            with self.assertRaises(ShellToolError, msg=command):
                self.check(command)


class ShellEscapeTests(unittest.TestCase):
    """Chaining is not filtered - it is simply never interpreted."""

    def check(self, command):
        return validate(command, COMMANDS)

    def test_chaining_does_not_produce_two_commands(self):
        # The tokens after ';' stay literal arguments to git, and git rejects
        # them itself. Nothing reaches a shell.
        argv = self.check("git status ; whoami")
        self.assertEqual(argv[0], "git")
        self.assertIn(";", argv)
        self.assertNotIn("whoami", argv[:1])

    def test_ampersand_chaining_is_literal(self):
        argv = self.check("git status && whoami")
        self.assertEqual(argv[0], "git")
        self.assertNotEqual(argv[0], "whoami")

    def test_pipe_is_literal(self):
        argv = self.check("git log | more")
        self.assertEqual(argv[0], "git")

    def test_redirection_is_literal(self):
        argv = self.check("git status > out.txt")
        self.assertEqual(argv[0], "git")

    def test_command_substitution_is_literal(self):
        argv = self.check("git log $(whoami)")
        self.assertEqual(argv[0], "git")
        self.assertIn("$(whoami)", argv)


class DangerousOptionTests(unittest.TestCase):
    def check(self, command):
        return validate(command, COMMANDS)

    def test_git_dash_c_refused(self):
        # `git -c core.pager='sh -c ...' log` would execute arbitrary code.
        with self.assertRaises(ShellToolError) as ctx:
            self.check("git -c core.pager=calc.exe log")
        self.assertIn("not allowed", str(ctx.exception))

    def test_exec_path_refused(self):
        with self.assertRaises(ShellToolError):
            self.check("git --exec-path=C:/evil log")

    def test_upload_pack_refused(self):
        with self.assertRaises(ShellToolError):
            self.check("git --upload-pack=calc.exe log")

    def test_directory_change_refused(self):
        with self.assertRaises(ShellToolError):
            self.check("git -C .. status")


class PathConfinementTests(unittest.TestCase):
    def check(self, command):
        return validate(command, COMMANDS)

    def test_absolute_windows_path_refused(self):
        with self.assertRaises(ShellToolError) as ctx:
            self.check("git show C:/Users/SHAMI/secrets.txt")
        self.assertIn("Absolute paths", str(ctx.exception))

    def test_backslash_windows_path_refused(self):
        with self.assertRaises(ShellToolError):
            self.check(r"git show C:\Users\SHAMI\secrets.txt")

    def test_absolute_posix_path_refused(self):
        with self.assertRaises(ShellToolError):
            self.check("git show /etc/passwd")

    def test_parent_traversal_refused(self):
        with self.assertRaises(ShellToolError) as ctx:
            self.check("git show ../../secrets.txt")
        self.assertIn("climb out", str(ctx.exception))

    def test_relative_paths_allowed(self):
        self.assertEqual(self.check("git show sub/inner.txt")[0], "git")

    def test_options_are_not_treated_as_paths(self):
        # A leading dash means an option; '--pretty=../x' is not a path.
        self.assertEqual(self.check("git log --pretty=format:%h")[0], "git")


class InterpreterTests(unittest.TestCase):
    def check(self, command):
        return validate(command, COMMANDS)

    def test_python_version_allowed(self):
        self.assertEqual(self.check("python --version")[0], "python")

    def test_python_dash_c_refused(self):
        # The whole point of the allowed_args restriction.
        with self.assertRaises(ShellToolError) as ctx:
            self.check("python -c \"import os; os.system('calc')\"")
        self.assertIn("only accepts", str(ctx.exception))

    def test_python_script_refused(self):
        with self.assertRaises(ShellToolError):
            self.check("python evil.py")

    def test_python_module_refused(self):
        with self.assertRaises(ShellToolError):
            self.check("python -m http.server")

    def test_pip_install_refused(self):
        with self.assertRaises(ShellToolError):
            self.check("pip install requests")

    def test_pip_list_allowed(self):
        self.assertEqual(self.check("pip list")[0], "pip")


class ExtraCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_extra_command_becomes_allowed(self):
        run = runner(self.tmp, extra_commands=("node",))
        self.assertIn("node", run.allowed_commands)

    def test_extra_command_does_not_weaken_others(self):
        run = runner(self.tmp, extra_commands=("node",))
        with self.assertRaises(ShellToolError):
            validate("git commit -m x", run._allowed)

    def test_empty_extras_ignored(self):
        run = runner(self.tmp, extra_commands=("", "  "))
        self.assertEqual(run.allowed_commands, sorted(COMMANDS))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.run = runner(self.tmp)

    def tearDown(self):
        self._tmp.cleanup()

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_runs_in_the_workspace(self):
        result = self.run.run("git rev-parse --show-toplevel")
        # Not a repository, so git fails - but it ran, in our directory.
        self.assertTrue(result["success"])
        self.assertNotEqual(result["exit_code"], 0)
        self.assertIn("not a git repository", result["stderr"].lower())

    @unittest.skipUnless(HAS_GIT, "git not installed")
    def test_nonzero_exit_is_reported_not_raised(self):
        result = self.run.run("git status")
        self.assertTrue(result["success"])
        self.assertIn("exit_code", result)

    def test_missing_binary_is_explained(self):
        run = runner(self.tmp, extra_commands=("definitely-not-installed",))
        with self.assertRaises(ShellToolError) as ctx:
            run.run("definitely-not-installed --help")
        self.assertIn("not installed", str(ctx.exception))

    def test_output_is_truncated(self):
        from tools.shell_tool import _truncate

        self.assertIn("truncated", _truncate("x" * 100, 20))

    def test_refusal_through_the_registry_is_structured(self):
        tool = build_shell_tool(
            self.tmp, timeout=10, max_output_chars=1000
        )
        registry = ToolRegistry([tool])
        result = registry.execute("run_command", {"command": "git push"})
        self.assertFalse(result.ok)
        self.assertIn("not allowed", result.payload["error"])


class EnvironmentTests(unittest.TestCase):
    def test_secrets_are_not_inherited(self):
        from tools.shell_tool import _child_environment

        env = _child_environment()
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        self.assertIn("PATH", env)

    def test_git_never_prompts(self):
        from tools.shell_tool import _child_environment

        self.assertEqual(_child_environment()["GIT_TERMINAL_PROMPT"], "0")


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_unless_enabled(self):
        registry, disabled = build_default_registry(Config(workspace=self.tmp))
        self.assertNotIn("run_command", registry.names())
        self.assertIn("terminal", {item.category for item in disabled})

    def test_registered_when_enabled(self):
        registry, disabled = build_default_registry(
            Config(workspace=self.tmp, shell_tool_enabled=True)
        )
        self.assertIn("run_command", registry.names())
        self.assertNotIn("terminal", {item.category for item in disabled})

    def test_tool_metadata(self):
        tool = build_shell_tool(self.tmp, timeout=10, max_output_chars=100)
        self.assertEqual(tool.category, "terminal")
        self.assertEqual(tool.parameters["required"], ["command"])
        self.assertIn("git", tool.description)


if __name__ == "__main__":
    unittest.main()
