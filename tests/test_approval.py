"""Commands that need a person, and the handshake that asks one.

Two halves. Which commands are free, which are gated and which are refused
outright - and the threading underneath, where a worker blocks mid-tool-call
while a request thread answers.
"""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from api.turns import Turn, TurnRequest
from tools.shell_tool import (
    COMMANDS,
    NEVER_ALLOWED,
    ShellRunner,
    ShellToolError,
    validate,
)


def verdict(command: str) -> str:
    """'free', 'approve' or 'refuse' for one command line."""
    try:
        return "approve" if validate(command, COMMANDS).needs_approval else "free"
    except ShellToolError:
        return "refuse"


class ClassificationTests(unittest.TestCase):
    def test_reading_runs_without_asking(self):
        for command in (
            "git status --short",
            "git log --oneline -5",
            "ls -la",
            "head -20 README.md",
            "wc -l setup.py",
            "rg pattern src",
            "node --version",
            "npm list",
            "pip list",
            "docker ps",
        ):
            self.assertEqual(verdict(command), "free", command)

    def test_changing_things_asks_first(self):
        for command in (
            "git commit -m 'x'",
            "git push origin main",
            "git checkout main",
            "pip install requests",
            "npm install left-pad",
            "npm run build",
            "mkdir reports",
            "cp a.txt b.txt",
            "curl https://example.com",
            "make all",
            "cargo build",
            "docker run alpine",
        ):
            self.assertEqual(verdict(command), "approve", command)

    def test_interpreters_are_refused_even_with_approval(self):
        """A prompt on `bash -c` asks someone to audit an arbitrary program."""
        for command in (
            "bash -c 'rm -rf /'",
            "sh -c whoami",
            "powershell -Command x",
            "cmd /c dir",
            "perl -e print",
            "sudo rm -rf /",
            "xargs rm",
            "python -c 'import os'",
        ):
            self.assertEqual(verdict(command), "refuse", command)

    def test_the_refusal_explains_itself(self):
        with self.assertRaises(ShellToolError) as caught:
            validate("bash -c ls", COMMANDS)
        self.assertIn("never allowed", str(caught.exception))
        self.assertIn("approval", str(caught.exception))

    def test_find_exec_is_refused_wherever_the_option_sits(self):
        """find's options follow its path, so the git-style scan missed them."""
        self.assertEqual(verdict("find . -exec rm {} ;"), "refuse")
        self.assertEqual(verdict("find . -delete"), "refuse")
        self.assertEqual(verdict("find src -name '*.py'"), "free")

    def test_the_git_option_scan_still_stops_at_the_verb(self):
        """`git log -c` is a diff format; only a leading -c is dangerous."""
        self.assertEqual(verdict("git log -c"), "free")
        self.assertEqual(verdict("git -c core.pager=sh log"), "refuse")

    def test_an_unknown_verb_lists_both_halves_of_the_allowlist(self):
        """Naming only the free verbs would read as though the rest were banned."""
        with self.assertRaises(ShellToolError) as caught:
            validate("git bisect", COMMANDS)
        message = str(caught.exception)
        self.assertIn("status", message)  # a free verb
        self.assertIn("commit", message)  # a gated one

    def test_every_never_allowed_name_is_refused(self):
        for name in NEVER_ALLOWED:
            self.assertEqual(verdict(f"{name} something"), "refuse", name)


class RunnerGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Path(self._tmp.name)

    def runner(self, approve=None) -> ShellRunner:
        return ShellRunner(self.workspace, timeout=5, approve=approve)

    def test_without_anyone_to_ask_it_refuses_rather_than_running(self):
        """The gate must not quietly vanish in the CLI or a script."""
        with self.assertRaises(ShellToolError) as caught:
            self.runner().run("mkdir made-anyway")
        self.assertIn("nobody to ask", str(caught.exception))
        self.assertFalse((self.workspace / "made-anyway").exists())

    def test_declining_returns_a_refusal_and_runs_nothing(self):
        asked: list[tuple[str, str]] = []

        def decline(command: str, reason: str) -> bool:
            asked.append((command, reason))
            return False

        result = self.runner(decline).run("mkdir nope")

        self.assertFalse(result["success"])
        self.assertTrue(result["declined"])
        self.assertFalse((self.workspace / "nope").exists())
        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0][0], "mkdir nope")
        self.assertIn("creates a directory", asked[0][1])

    def test_approving_actually_runs_it(self):
        result = self.runner(lambda command, reason: True).run("mkdir yes-please")

        self.assertTrue(result["success"], result)
        self.assertTrue((self.workspace / "yes-please").is_dir())

    def test_a_free_command_never_asks(self):
        asked: list[str] = []

        def watch(command: str, reason: str) -> bool:
            asked.append(command)
            return True

        self.runner(watch).run("git --version")
        self.assertEqual(asked, [])


def a_turn() -> Turn:
    return Turn(
        request=TurnRequest(
            conversation_id=1, prompt="hello", user_message_id=1, model_key="fast"
        )
    )


def drain(turn: Turn) -> list[dict]:
    """Every event queued so far."""
    events = []
    while not turn.events.empty():
        item = turn.events.get_nowait()
        if item is not None:
            events.append(item)
    return events


class HandshakeTests(unittest.TestCase):
    """The worker blocks; a request thread answers. Both sides, for real."""

    def ask_on_a_thread(self, turn: Turn, timeout: float = 5.0):
        answer: list[bool] = []
        worker = threading.Thread(
            target=lambda: answer.append(
                turn.ask("mkdir x", "creates a directory", timeout=timeout)
            )
        )
        worker.start()
        return worker, answer

    def wait_for_request(self, turn: Turn) -> str:
        """The id of the approval the worker is now waiting on."""
        deadline = time.time() + 5
        while time.time() < deadline:
            with turn.lock:
                if turn.pending:
                    return next(iter(turn.pending))
            time.sleep(0.01)
        self.fail("the worker never asked")

    def test_a_yes_reaches_the_waiting_worker(self):
        turn = a_turn()
        worker, answer = self.ask_on_a_thread(turn)
        request_id = self.wait_for_request(turn)

        self.assertTrue(turn.answer(request_id, True))
        worker.join(timeout=5)

        self.assertEqual(answer, [True])
        kinds = [event["type"] for event in drain(turn)]
        self.assertIn("approval", kinds)
        self.assertIn("approval_closed", kinds)

    def test_a_no_reaches_it_too(self):
        turn = a_turn()
        worker, answer = self.ask_on_a_thread(turn)
        request_id = self.wait_for_request(turn)

        turn.answer(request_id, False)
        worker.join(timeout=5)

        self.assertEqual(answer, [False])

    def test_the_request_carries_the_command_and_the_reason(self):
        turn = a_turn()
        worker, _ = self.ask_on_a_thread(turn)
        request_id = self.wait_for_request(turn)
        turn.answer(request_id, False)
        worker.join(timeout=5)

        asked = next(e for e in drain(turn) if e["type"] == "approval")
        self.assertEqual(asked["command"], "mkdir x")
        self.assertEqual(asked["reason"], "creates a directory")
        self.assertEqual(asked["request_id"], request_id)

    def test_silence_declines(self):
        """Nobody watching means nobody agreed."""
        turn = a_turn()
        worker, answer = self.ask_on_a_thread(turn, timeout=0.2)
        worker.join(timeout=5)

        self.assertEqual(answer, [False])
        self.assertEqual(turn.pending, {})

    def test_stopping_the_turn_declines_without_waiting_it_out(self):
        turn = a_turn()
        started = time.time()
        worker, answer = self.ask_on_a_thread(turn, timeout=30)
        self.wait_for_request(turn)

        turn.stop()
        worker.join(timeout=5)

        self.assertEqual(answer, [False])
        # It must not have sat out the 30s timeout to notice.
        self.assertLess(time.time() - started, 5)

    def test_answering_twice_is_reported_as_stale(self):
        turn = a_turn()
        worker, _ = self.ask_on_a_thread(turn)
        request_id = self.wait_for_request(turn)

        self.assertTrue(turn.answer(request_id, True))
        worker.join(timeout=5)
        # The worker has cleaned it up, so the second answer finds nothing.
        self.assertFalse(turn.answer(request_id, True))

    def test_an_unknown_request_is_not_an_error(self):
        self.assertFalse(a_turn().answer("never-existed", True))


if __name__ == "__main__":
    unittest.main()
