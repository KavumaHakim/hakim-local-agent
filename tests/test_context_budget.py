"""Keeping a turn inside the model's context window.

These exist because of a real failure: asking the agent to find a file made it
list several directories, and the turn arrived at a 4,096-token model carrying
roughly 13,000 tokens of history. Two things let that happen and both are
covered here.

The per-result cap only knew how to cut *strings* - it was written for a page
of OCR and a file read - so a directory listing, which is hundreds of small
dicts under one key, walked straight past it.

And `_trim_history` can only cut immediately before a user message. Inside one
turn there is exactly one of those, at the very front, so it finds nothing safe
to drop and returns, leaving the request oversized rather than trimmed. Nothing
bounded the *sum* of results in a turn, only each one on its own.
"""

from __future__ import annotations

import json
import unittest

from agent.loop import Agent
try:
    from agent.loop import MIN_RESULT_CHARS
except ImportError:
    MIN_RESULT_CHARS = 400
from config import Config
from tests.fake_client import tool_call_message
from tools.base import Tool, ToolRegistry, ToolResult


def listing(count: int) -> dict:
    """What a directory listing looks like: a list, not a string."""
    return {
        "success": True,
        "path": "/somewhere",
        "files": [
            {"name": f"entry_{number}.py", "type": "file", "size_bytes": 1234}
            for number in range(count)
        ],
    }


class ResultCapTests(unittest.TestCase):
    """The per-result cap, against payloads that are not one long string."""

    def test_a_long_string_is_cut(self):
        result = ToolResult(name="list_directory", payload={"success": True, "text": "x" * 20_000})
        text = result.content_within(2_000)
        self.assertLessEqual(len(text), 2_000)
        self.assertIn("truncated", text)

    def test_a_long_list_is_cut(self):
        """The case that was missed: nothing here is a long string."""
        result = ToolResult(name="list_directory", payload=listing(2_000))
        text = result.content_within(2_000)
        self.assertLessEqual(len(text), 2_000)
        self.assertIn("truncated", text)

    def test_what_survives_a_cut_list_is_still_valid_json(self):
        result = ToolResult(name="list_directory", payload=listing(2_000))
        parsed = json.loads(result.content_within(2_000))
        self.assertTrue(parsed["success"])
        self.assertLess(len(parsed["files"]), 2_000)
        self.assertGreater(len(parsed["files"]), 0)

    def test_the_cut_says_how_much_is_missing(self):
        """A model handed the first forty entries with no indication answers
        as though it had seen the directory."""
        result = ToolResult(name="list_directory", payload=listing(2_000))
        parsed = json.loads(result.content_within(2_000))
        shown = len(parsed["files"])
        self.assertIn(f"{2_000 - shown:,}", parsed["truncated"])
        self.assertIn("2,000", parsed["truncated"])

    def test_a_payload_that_cannot_be_cut_is_refused_rather_than_sent(self):
        """Nested structure reaches neither the string pass nor the list pass.
        Returning it oversized is what overflows the window."""
        payload = {
            "success": True,
            "tree": {str(n): {"child": list(range(50))} for n in range(200)},
        }
        text = ToolResult(name="list_directory", payload=payload).content_within(1_500)
        self.assertLessEqual(len(text), 1_500)
        parsed = json.loads(text)
        self.assertIn("truncated", parsed)

    def test_a_small_result_is_left_exactly_alone(self):
        payload = {"success": True, "files": [{"name": "one.py"}]}
        text = ToolResult(name="list_directory", payload=payload).content_within(10_000)
        self.assertEqual(json.loads(text), payload)
        self.assertNotIn("truncated", text)

    def test_no_limit_means_no_limit(self):
        result = ToolResult(name="list_directory", payload=listing(500))
        self.assertEqual(len(json.loads(result.content_within(0))["files"]), 500)


class TurnBudgetTests(unittest.TestCase):
    """The sum of a turn's tool results, which nothing used to bound."""

    def build(self, rounds: int, entries: int = 400):
        """An agent whose model hunts for a file across `rounds` directories."""

        tool = Tool(
            name="list_directory",
            category="filesystem",
            description="List a directory.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            run=lambda **_: listing(entries),
        )

        class Hunting:
            def __init__(self) -> None:
                self.left = rounds
                self.sizes: list[int] = []

            def chat(self, messages, *, tools=None):
                self.sizes.append(
                    sum(
                        len(str(message.get("content") or ""))
                        + len(json.dumps(message.get("tool_calls") or []))
                        for message in messages
                    )
                )
                if self.left > 0:
                    self.left -= 1
                    return tool_call_message(("list_directory", {"path": "."}))
                return {"role": "assistant", "content": "not found"}

        config = Config(model_context=4096, max_iterations=10)
        client = Hunting()
        return Agent(client, config, ToolRegistry([tool])), client, config

    def test_one_listing_does_not_fill_the_window(self):
        agent, client, config = self.build(rounds=1)
        agent.send("find config.py")
        self.assertLessEqual(max(client.sizes), config.max_history_chars)

    def test_four_listings_stay_within_the_model(self):
        """Each result is inside its own cap; their sum used to be three times
        the whole window."""
        agent, client, config = self.build(rounds=4)
        agent.send("find config.py")

        worst = max(client.sizes)
        tokens = worst / 3.27
        # The window, less what the system prompt and tool schemas cost.
        self.assertLess(tokens, config.model_context - 1_080)

    def test_the_growth_is_bounded_not_linear_in_results(self):
        """Every round used to add a full-sized result. Now the later ones are
        floored, so a long hunt degrades instead of exploding."""
        short_agent, short_client, _ = self.build(rounds=2)
        short_agent.send("find it")

        long_agent, long_client, _ = self.build(rounds=8)
        long_agent.send("find it")

        # Four times the rounds must not cost four times the context. Before
        # the fix it was very nearly exactly linear.
        self.assertLess(max(long_client.sizes), max(short_client.sizes) * 2)

    def test_a_result_is_never_cut_below_the_floor(self):
        """Cut to nothing, a result tells the model less than an honest 'this
        did not fit', and the note explaining the cut does not fit either."""
        agent, _, _ = self.build(rounds=8)
        agent.send("find it")

        results = [m for m in agent.history if m.get("role") == "tool"]
        self.assertTrue(results)
        for message in results:
            self.assertGreaterEqual(len(message["content"]), MIN_RESULT_CHARS // 2)


class HistoryAccountingTests(unittest.TestCase):
    def test_tool_calls_count_towards_the_budget(self):
        """They are sent to the model, so leaving them out of the measurement
        under-reports exactly the turns that run out of room."""
        agent = Agent(object(), Config(), ToolRegistry([]))
        agent.load_history(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {
                                "name": "list_directory",
                                "arguments": '{"path": "."}',
                            },
                        }
                    ],
                }
            ]
        )
        self.assertGreater(agent._history_chars(), 50)


if __name__ == "__main__":
    unittest.main()
