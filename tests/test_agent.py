"""Agent loop tests. No llama.cpp server required."""

from __future__ import annotations

import json
import unittest

from agent.loop import Agent, IterationLimitError
from config import Config
from tests.fake_client import FakeQwenClient, text_message, tool_call_message
from tools.base import Tool, ToolError, ToolRegistry
from tools.calculator import CALCULATOR_TOOL


def _boom(**_: object) -> dict:
    raise RuntimeError("tool exploded")


def _refuse(**_: object) -> dict:
    raise ToolError("not today")


FAILING_TOOL = Tool(
    name="explode",
    category="test",
    description="Always raises.",
    parameters={"type": "object", "properties": {}, "required": []},
    run=_boom,
)

REFUSING_TOOL = Tool(
    name="refuse",
    category="test",
    description="Always refuses.",
    parameters={"type": "object", "properties": {}, "required": []},
    run=_refuse,
)


def build_agent(responses, *, repeat_last=False, max_iterations=5, extra_tools=()):
    config = Config(max_iterations=max_iterations)
    registry = ToolRegistry([CALCULATOR_TOOL, *extra_tools])
    client = FakeQwenClient(responses, repeat_last=repeat_last)
    return Agent(client, config, registry), client


def tool_messages(messages):
    return [m for m in messages if m["role"] == "tool"]


def payload_of(message):
    return json.loads(message["content"])


class NormalResponseTests(unittest.TestCase):
    def test_plain_answer_is_returned(self):
        agent, client = build_agent([text_message("Paris.")])
        turn = agent.send("Capital of France?")

        self.assertEqual(turn.content, "Paris.")
        self.assertFalse(turn.wants_tools)
        self.assertEqual(len(client.calls), 1)

    def test_system_prompt_is_sent_first(self):
        agent, client = build_agent([text_message("ok")])
        agent.send("hello")

        self.assertEqual(client.calls[0][0]["role"], "system")
        self.assertIn("You are Qwen", client.calls[0][0]["content"])

    def test_tool_definitions_are_sent(self):
        agent, client = build_agent([text_message("ok")])
        agent.send("hello")

        names = {t["function"]["name"] for t in client.tools_seen[0]}
        self.assertIn("calculate", names)

    def test_conversation_continues_across_turns(self):
        agent, client = build_agent([text_message("first"), text_message("second")])
        agent.send("one")
        agent.send("two")

        roles = [m["role"] for m in client.calls[1]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_clear_resets_history(self):
        agent, client = build_agent([text_message("a"), text_message("b")])
        agent.send("one")
        agent.clear()
        agent.send("two")

        self.assertEqual(len(client.calls[1]), 2)  # system + user only


class SingleToolCallTests(unittest.TestCase):
    def test_tool_runs_and_result_feeds_back(self):
        agent, client = build_agent(
            [
                tool_call_message(("calculate", {"expression": "sqrt(144) + 25**2"})),
                text_message("637"),
            ]
        )
        turn = agent.send("Calculate sqrt(144) + 25^2")

        self.assertEqual(turn.content, "637")

        messages = tool_messages(client.calls[1])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["tool_call_id"], "call_0")

        payload = payload_of(messages[0])
        self.assertTrue(payload["success"])
        self.assertEqual(payload["formatted"], "637")

    def test_assistant_tool_call_is_preserved_in_history(self):
        agent, client = build_agent(
            [
                tool_call_message(("calculate", {"expression": "2**10"})),
                text_message("1024"),
            ]
        )
        agent.send("2^10?")

        assistant = [m for m in client.calls[1] if m["role"] == "assistant"][0]
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "calculate")

    def test_observer_sees_the_call(self):
        agent, _ = build_agent(
            [tool_call_message(("calculate", {"expression": "1+1"})), text_message("2")]
        )
        seen = []
        agent.send("1+1?", observer=seen.append)

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].call.name, "calculate")
        self.assertTrue(seen[0].result.ok)


class MultipleToolCallTests(unittest.TestCase):
    def test_two_calls_in_one_message(self):
        agent, client = build_agent(
            [
                tool_call_message(
                    ("calculate", {"expression": "6*7"}),
                    ("calculate", {"expression": "10**2"}),
                ),
                text_message("42 and 100"),
            ]
        )
        turn = agent.send("both please")

        self.assertEqual(turn.content, "42 and 100")
        messages = tool_messages(client.calls[1])
        self.assertEqual(
            [payload_of(m)["formatted"] for m in messages], ["42", "100"]
        )
        self.assertEqual([m["tool_call_id"] for m in messages], ["call_0", "call_1"])

    def test_calls_across_several_rounds(self):
        agent, client = build_agent(
            [
                tool_call_message(("calculate", {"expression": "2+2"})),
                tool_call_message(("calculate", {"expression": "4*4"})),
                text_message("16"),
            ]
        )
        turn = agent.send("chain it")

        self.assertEqual(turn.content, "16")
        self.assertEqual(len(client.calls), 3)


class ToolErrorTests(unittest.TestCase):
    def test_unknown_tool_name_is_reported_not_raised(self):
        agent, client = build_agent(
            [tool_call_message(("teleport", {})), text_message("Sorry, I cannot.")]
        )
        turn = agent.send("teleport me")

        self.assertEqual(turn.content, "Sorry, I cannot.")
        payload = payload_of(tool_messages(client.calls[1])[0])
        self.assertFalse(payload["success"])
        self.assertIn("No tool named 'teleport'", payload["error"])

    def test_missing_required_argument(self):
        agent, client = build_agent(
            [tool_call_message(("calculate", {})), text_message("recovered")]
        )
        agent.send("calculate something")

        payload = payload_of(tool_messages(client.calls[1])[0])
        self.assertIn("Missing required argument", payload["error"])

    def test_unknown_argument(self):
        agent, client = build_agent(
            [tool_call_message(("calculate", {"expr": "1+1"})), text_message("ok")]
        )
        agent.send("go")

        payload = payload_of(tool_messages(client.calls[1])[0])
        # A misspelled key is both a missing required key and an unknown one.
        self.assertIn("Unknown argument(s): expr", payload["error"])
        self.assertIn("Missing required argument(s): expression", payload["error"])

    def test_wrong_argument_type(self):
        agent, client = build_agent(
            [tool_call_message(("calculate", {"expression": 42})), text_message("ok")]
        )
        agent.send("go")

        payload = payload_of(tool_messages(client.calls[1])[0])
        self.assertIn("must be a string", payload["error"])

    def test_tool_raising_unexpectedly_is_contained(self):
        agent, client = build_agent(
            [tool_call_message(("explode", {})), text_message("recovered")],
            extra_tools=[FAILING_TOOL],
        )
        turn = agent.send("explode")

        self.assertEqual(turn.content, "recovered")
        payload = payload_of(tool_messages(client.calls[1])[0])
        self.assertIn("RuntimeError: tool exploded", payload["error"])

    def test_tool_error_is_reported_cleanly(self):
        agent, client = build_agent(
            [tool_call_message(("refuse", {})), text_message("ok")],
            extra_tools=[REFUSING_TOOL],
        )
        agent.send("refuse")

        payload = payload_of(tool_messages(client.calls[1])[0])
        self.assertEqual(payload["error"], "not today")


class IterationLimitTests(unittest.TestCase):
    def test_limit_is_enforced(self):
        agent, client = build_agent(
            [tool_call_message(("calculate", {"expression": "1+1"}))],
            repeat_last=True,
            max_iterations=3,
        )

        with self.assertRaises(IterationLimitError):
            agent.send("loop forever")

        self.assertEqual(len(client.calls), 3)


class MalformedResponseTests(unittest.TestCase):
    def test_bad_tool_call_json_is_an_agent_error(self):
        broken = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": "{not json"},
                }
            ],
        }
        agent, _ = build_agent([broken])

        with self.assertRaises(Exception) as ctx:
            agent.send("go")
        self.assertIn("Could not read the model's reply", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
