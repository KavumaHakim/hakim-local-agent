"""What a turn reports about its own context.

The report is what the interface shows: how much of the window this turn
used, how much of it was tool schemas, what had to be dropped. It used to
exist only when memory was attached, which is off by default - so for most
turns anybody actually ran, there was nothing to show.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import unittest
from pathlib import Path

from agent.loop import CHARS_PER_TOKEN, Agent
from config import Config
from tests.fake_client import FakeQwenClient, text_message, tool_call_message
from tools.base import Tool, ToolRegistry

ROOT = Path(__file__).resolve().parent.parent


def registry(run=None) -> ToolRegistry:
    def ok(**_):
        return {"success": True, "value": 4}

    schema = {"type": "object", "properties": {}}
    return ToolRegistry(
        [
            Tool("calculate", "calculator", "adds", schema, run or ok),
            Tool("git_status", "git", "status", schema, run or ok),
        ]
    )


def agent_for(client, **settings) -> Agent:
    config = dataclasses.replace(Config(), **settings)
    return Agent(client, config, registry())


class WithoutMemoryTests(unittest.TestCase):
    """Memory is off by default, so this is the path that matters most."""

    def test_a_plain_turn_reports_its_context(self):
        client = FakeQwenClient([text_message("hi")])
        agent = agent_for(client, model_context=4096)

        agent.send("hello")

        report = agent.context_report
        # One: the question. The report describes the request that was built,
        # not the history afterwards - the answer did not exist when the
        # context was assembled, and reporting it would overstate what the
        # model was actually given.
        self.assertEqual(report["messages_kept"], 1)
        self.assertEqual(report["messages_dropped"], 0)
        self.assertEqual(report["memories"], [])
        self.assertFalse(report["summary_used"])
        self.assertGreater(report["estimated_tokens"], 0)

    def test_it_names_the_window_it_has_to_fit(self):
        """A token count with nothing to compare it to says very little."""
        client = FakeQwenClient([text_message("hi")])
        agent = agent_for(client, model_context=8192)

        agent.send("hello")

        self.assertEqual(agent.context_report["context_limit"], 8192)

    def test_tool_schemas_are_counted_separately_and_in_the_total(self):
        client = FakeQwenClient([text_message("hi")])
        agent = agent_for(client, model_context=4096)

        agent.send("hello")

        report = agent.context_report
        self.assertGreater(report["tool_tokens"], 0)
        self.assertEqual(
            report["total_estimated_tokens"],
            report["estimated_tokens"] + report["tool_tokens"],
        )

    def test_the_estimate_uses_the_shared_ratio(self):
        client = FakeQwenClient([text_message("hi")])
        agent = agent_for(client)

        agent.send("hello")

        report = agent.context_report
        self.assertEqual(
            report["estimated_tokens"], int(report["characters"] / CHARS_PER_TOKEN)
        )


class DroppedAndTruncatedTests(unittest.TestCase):
    def test_messages_the_trimmer_dropped_are_counted(self):
        client = FakeQwenClient([text_message("ok")], repeat_last=True)
        # Two messages of history is nothing; a limit of 2 forces a cut.
        agent = agent_for(client, max_history_messages=2)

        for _ in range(4):
            agent.send("hello")

        self.assertGreater(agent.context_report["messages_dropped"], 0)

    def test_dropped_resets_between_turns(self):
        """It reports what this turn cost, not a running total for the session."""
        client = FakeQwenClient([text_message("ok")], repeat_last=True)
        agent = agent_for(client, max_history_messages=2)

        for _ in range(4):
            agent.send("hello")
        self.assertGreater(agent.context_report["messages_dropped"], 0)

        agent._history.clear()
        agent.send("hello")

        self.assertEqual(agent.context_report["messages_dropped"], 0)

    def test_a_cut_tool_result_is_reported(self):
        def huge(**_):
            return {"success": True, "text": "x" * 50_000}

        client = FakeQwenClient(
            [tool_call_message(("calculate", {})), text_message("done")]
        )
        # max_tool_result_chars is derived from the window and this share,
        # so the small window is what makes the result too big to fit.
        config = dataclasses.replace(
            Config(), model_context=2048, tool_result_share=0.05
        )
        agent = Agent(client, config, registry(run=huge))

        agent.send("hello")

        self.assertEqual(agent.context_report["truncated_results"], 1)

    def test_a_result_that_fits_is_not_reported_as_cut(self):
        client = FakeQwenClient(
            [tool_call_message(("calculate", {})), text_message("done")]
        )
        agent = agent_for(client)

        agent.send("hello")

        self.assertEqual(agent.context_report["truncated_results"], 0)


class DependencyIsolationTests(unittest.TestCase):
    """The base install has no numpy, so the agent must not need it.

    This is a subprocess because the rest of the suite imports numpy long
    before this runs; asking `sys.modules` in-process would always pass.
    """

    def test_importing_the_agent_does_not_pull_numpy(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import agent.loop;"
                " sys.exit(1 if 'numpy' in sys.modules else 0)",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            "importing agent.loop pulled in numpy, which lives in "
            "requirements-rag.txt - a base install would fail at startup.\n"
            f"{result.stdout}{result.stderr}",
        )

    def test_the_duplicated_ratio_matches_its_source(self):
        """agent/loop.py copies this rather than importing it. Keep them equal."""
        from memory.context import CHARS_PER_TOKEN as canonical

        self.assertEqual(CHARS_PER_TOKEN, canonical)


if __name__ == "__main__":
    unittest.main()
