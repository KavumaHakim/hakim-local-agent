"""Streaming client tests: SSE assembly, and the decode beneath it.

Almost everything here runs against a fake session, because the parser is
the part worth testing. The exception is the encoding test, which needs a
real socket to reproduce what a real server's headers do.
"""

from __future__ import annotations

import http.server
import json
import threading
import unittest
from dataclasses import replace

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
        # Whether the response was left. Worth recording: dropping the
        # connection is what actually makes llama-server stop generating, so
        # a stop that did not close it would have freed nothing.
        self.exited = False

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


class FakeSession:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.payloads: list[dict] = []
        self.responses: list[FakeResponse] = []

    def post(self, url, json=None, timeout=None, stream=False):
        self.payloads.append(json)
        response = FakeResponse(self._lines)
        self.responses.append(response)
        return response


class FakeStreamClient(QwenClient):
    """QwenClient with only the socket replaced, so real SSE parsing runs."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(Config())
        self._session = FakeSession(lines)

    @property
    def sent(self) -> list[dict]:
        return self._session.payloads

    @property
    def responses(self) -> list[FakeResponse]:
        return self._session.responses


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


class StoppingAStreamTests(unittest.TestCase):
    """Ending a model round part-way, which is where a turn's minutes go."""

    def test_it_stops_and_returns_what_it_had(self):
        client = FakeStreamClient(
            sse(delta(content="one "), delta(content="two "), delta(content="three"))
        )
        seen: list[str] = []
        message = client.chat_stream(
            [], on_token=seen.append, should_stop=lambda: len(seen) >= 2
        )

        self.assertEqual(seen, ["one ", "two "])
        self.assertEqual(message["content"], "one two ")

    def test_it_drops_the_connection(self):
        """The flag alone frees nothing; llama-server stops when we leave."""
        client = FakeStreamClient(sse(delta(content="a"), delta(content="b")))
        client.chat_stream([], should_stop=lambda: True)

        self.assertTrue(client.responses[0].exited)

    def test_a_stop_that_never_fires_changes_nothing(self):
        client = FakeStreamClient(sse(delta(content="all"), delta(content=" of it")))
        message = client.chat_stream([], should_stop=lambda: False)
        self.assertEqual(message["content"], "all of it")


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


class ProviderFieldPreservationTests(unittest.TestCase):
    """Fields we do not recognise must survive the merge.

    Gemini 3 attaches a thought_signature to each tool call and rejects the
    next round if it does not come back. Rebuilding a tool call from only the
    known fields dropped it, and the failure surfaced one round later as
    "Function call is missing a thought_signature" - which reads like a tool
    bug, not a lossy merge.
    """

    def test_unknown_fields_are_carried_through(self):
        store: dict[int, dict] = {}
        signature = {"google": {"thought_signature": "EswCCskCARFNMg9"}}
        _merge_tool_call(
            store,
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "extra_content": signature,
                "function": {"name": "calculate", "arguments": '{"expression":"1+1"}'},
            },
        )
        self.assertEqual(store[0]["extra_content"], signature)
        self.assertEqual(store[0]["function"]["name"], "calculate")

    def test_index_is_not_carried_into_the_finished_call(self):
        """It keys the accumulator; it is not part of a tool call."""
        store: dict[int, dict] = {}
        _merge_tool_call(store, {"index": 2, "id": "c", "function": {"name": "n"}})
        self.assertNotIn("index", store[2])

    def test_a_later_fragment_does_not_erase_an_earlier_extra(self):
        store: dict[int, dict] = {}
        _merge_tool_call(
            store, {"index": 0, "extra_content": {"google": {"thought_signature": "s"}}}
        )
        _merge_tool_call(
            store, {"index": 0, "function": {"arguments": '{"a":1}'}}
        )
        self.assertIn("extra_content", store[0])
        self.assertEqual(store[0]["function"]["arguments"], '{"a":1}')


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

            def chat_stream(
                self, messages, *, tools=None, on_token=None, on_reasoning=None,
                should_stop=None,
            ):
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

            def chat_stream(
                self, messages, *, tools=None, on_token=None, on_reasoning=None,
                should_stop=None,
            ):
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


class SSEEncodingTests(unittest.TestCase):
    """Decoding the wire, over a real socket.

    The fakes above hand back text that is already `str`, so they can only
    exercise the SSE parser and never the decode that happens under it. That
    is where mojibake came from: llama-server labels the stream
    `text/event-stream` and names no charset, and requests answers a missing
    charset on any `text/*` body with ISO-8859-1. So this one test pays for a
    real HTTP server.
    """

    text = "It’s a “quote” — café → 100%"

    @classmethod
    def setUpClass(cls):
        # Two well-formed events, as a real server sends them.
        body = b""
        for piece in (cls.text[:9], cls.text[9:]):
            chunk = {"choices": [{"delta": {"content": piece}}]}
            # Raw UTF-8 in the JSON, not \u escapes - what llama-server sends,
            # and the reason a wrong charset is able to do any damage.
            line = json.dumps(chunk, ensure_ascii=False)
            body += b"data: " + line.encode("utf-8") + b"\n\n"
        body += b"data: [DONE]\n\n"

        # Then cut the byte stream inside a character. Nothing stops a server
        # from flushing mid-sequence, and only an incremental decoder survives
        # it, so the wire split belongs in the test.
        cut = body.index("’".encode("utf-8")) + 1

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                # Read the prompt before answering. Windows resets a socket
                # that is closed with an undrained request body, and the reset
                # reaches the client before the reply does.
                self.rfile.read(int(self.headers.get("Content-Length", 0)))

                self.send_response(200)
                # No charset, which is the whole point.
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for part in (body[:cut], body[cut:]):
                    self.wfile.write(part)
                    self.wfile.flush()

            def log_message(self, *args):
                pass

        cls.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_punctuation_survives_the_stream(self):
        host, port = self.server.server_address
        config = replace(Config(), qwen_url=f"http://{host}:{port}")
        client = QwenClient(config)

        message = client.chat_stream([])

        self.assertEqual(message["content"], self.text)


if __name__ == "__main__":
    unittest.main()
