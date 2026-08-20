"""Streaming client tests: SSE assembly, without a live server."""

from __future__ import annotations

import json
import unittest

from config import Config
from models.qwen import QwenClient, _merge_tool_call


def sse(*chunks: dict) -> list[str]:
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return lines


def delta(**fields) -> dict:
    return {"choices": [{"delta": fields}]}


class FakeResponse:
    """Stands in for a streaming requests.Response."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200
        self.text = ""

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.payloads: list[dict] = []

    def post(self, url, json=None, timeout=None, stream=False):
        self.payloads.append(json)
        return FakeResponse(self._lines)


class FakeStreamClient(QwenClient):
    """QwenClient with only the socket replaced, so real SSE parsing runs."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(Config())
        self._session = FakeSession(lines)

    @property
    def sent(self) -> list[dict]:
        return self._session.payloads


class TokenStreamTests(unittest.TestCase):
    def test_content_is_assembled_and_streamed(self):
        client = FakeStreamClient(
            sse(delta(content="Hello"), delta(content=" there"), delta(content="!"))
        )
        seen: list[str] = []
        message = client.chat_stream([], on_token=seen.append)

        self.assertEqual(message["content"], "Hello there!")
        self.assertEqual(seen, ["Hello", " there", "!"])

    def test_reasoning_is_counted_but_never_streamed(self):
        client = FakeStreamClient(
            sse(
                delta(reasoning_content="secret thinking"),
                delta(content="answer"),
            )
        )
        seen: list[str] = []
        message = client.chat_stream([], on_token=seen.append)

        self.assertEqual(seen, ["answer"])
        self.assertEqual(message["content"], "answer")
        self.assertEqual(message["reasoning_chars"], len("secret thinking"))
        self.assertNotIn("reasoning_content", message)

    def test_works_without_a_callback(self):
        client = FakeStreamClient(sse(delta(content="quiet")))
        self.assertEqual(client.chat_stream([])["content"], "quiet")


class ToolCallAssemblyTests(unittest.TestCase):
    def test_tool_call_split_across_chunks(self):
        client = FakeStreamClient(
            sse(
                delta(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "calculate", "arguments": '{"expr'},
                        }
                    ]
                ),
                delta(
                    tool_calls=[
                        {"index": 0, "function": {"arguments": 'ession": "2+2"}'}}
                    ]
                ),
            )
        )
        message = client.chat_stream([])

        call = message["tool_calls"][0]
        self.assertEqual(call["id"], "call_a")
        self.assertEqual(call["function"]["name"], "calculate")
        self.assertEqual(
            json.loads(call["function"]["arguments"]), {"expression": "2+2"}
        )

    def test_two_tool_calls_keep_their_order(self):
        client = FakeStreamClient(
            sse(
                delta(
                    tool_calls=[
                        {
                            "index": 1,
                            "id": "b",
                            "function": {"name": "second", "arguments": "{}"},
                        }
                    ]
                ),
                delta(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "a",
                            "function": {"name": "first", "arguments": "{}"},
                        }
                    ]
                ),
            )
        )
        message = client.chat_stream([])

        names = [c["function"]["name"] for c in message["tool_calls"]]
        self.assertEqual(names, ["first", "second"])

    def test_whole_tool_call_in_one_chunk(self):
        client = FakeStreamClient(
            sse(
                delta(
                    tool_calls=[
                        {
                            "index": 0,
                            "id": "x",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"expression": "1+1"}',
                            },
                        }
                    ]
                )
            )
        )
        call = client.chat_stream([])["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "calculate")

    def test_malformed_chunks_are_skipped(self):
        client = FakeStreamClient(
            ["data: not json", "data: " + json.dumps(delta(content="ok")), "data: [DONE]"]
        )
        self.assertEqual(client.chat_stream([])["content"], "ok")

    def test_merge_helper_defaults_missing_index(self):
        store: dict[int, dict] = {}
        _merge_tool_call(store, {"function": {"name": "n", "arguments": "{}"}})
        self.assertEqual(store[0]["function"]["name"], "n")

    def test_stream_flag_is_set_on_the_request(self):
        client = FakeStreamClient(sse(delta(content="x")))
        client.chat_stream([], tools=[{"type": "function"}])
        self.assertTrue(client.sent[0]["stream"])
        self.assertEqual(client.sent[0]["tool_choice"], "auto")

    def test_blank_and_comment_lines_are_ignored(self):
        client = FakeStreamClient(
            ["", ": keepalive", "data: " + json.dumps(delta(content="ok")), "data: [DONE]"]
        )
        self.assertEqual(client.chat_stream([])["content"], "ok")


class AgentStreamingTests(unittest.TestCase):
    def test_agent_uses_chat_stream_when_given_a_callback(self):
        from agent.loop import Agent
        from tools.base import ToolRegistry
        from tools.calculator import CALCULATOR_TOOL

        class Recorder:
            def __init__(self):
                self.streamed = False

            def chat(self, messages, *, tools=None):
                return {"role": "assistant", "content": "non-streamed"}

            def chat_stream(self, messages, *, tools=None, on_token=None, on_reasoning=None):
                self.streamed = True
                if on_token:
                    on_token("streamed")
                return {"role": "assistant", "content": "streamed"}

        client = Recorder()
        agent = Agent(client, Config(), ToolRegistry([CALCULATOR_TOOL]))

        seen: list[str] = []
        turn = agent.send("hi", on_token=seen.append)

        self.assertTrue(client.streamed)
        self.assertEqual(turn.content, "streamed")
        self.assertEqual(seen, ["streamed"])

    def test_agent_falls_back_when_no_callback(self):
        from agent.loop import Agent
        from tools.base import ToolRegistry

        class Recorder:
            def __init__(self):
                self.streamed = False

            def chat(self, messages, *, tools=None):
                return {"role": "assistant", "content": "non-streamed"}

            def chat_stream(self, messages, *, tools=None, on_token=None, on_reasoning=None):
                self.streamed = True
                return {"role": "assistant", "content": "streamed"}

        client = Recorder()
        agent = Agent(client, Config(), ToolRegistry([]))
        turn = agent.send("hi")

        self.assertFalse(client.streamed)
        self.assertEqual(turn.content, "non-streamed")

    def test_load_history_replaces_conversation(self):
        from agent.loop import Agent
        from tools.base import ToolRegistry

        class Stub:
            def chat(self, messages, *, tools=None):
                return {"role": "assistant", "content": "ok"}

        agent = Agent(Stub(), Config(), ToolRegistry([]))
        agent.load_history([{"role": "user", "content": "earlier"}])

        self.assertEqual(agent.history, [{"role": "user", "content": "earlier"}])


if __name__ == "__main__":
    unittest.main()
