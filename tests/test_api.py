"""API tests. No llama-server, no model files, no network.

The model manager is the harness the model tests already use, and the chat
client is scripted, so these exercise the real routes, the real queue and the
real SQLite store while staying deterministic. A test that passed only because
nothing happened to be listening on port 8080 would be worse than no test.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Iterable
from unittest import mock

from fastapi.testclient import TestClient

from api.main import create_app
from api.runtime import Runtime
from api.turns import Turn, TurnQueueFull, TurnRequest
from config import Config
from tests.fake_client import tool_call_message
from tests.test_manager import ManagerHarness


def write_registry(tmp: Path) -> Path:
    """A two-model registry with a router, written into a temp directory."""
    exe = tmp / "llama-server.exe"
    exe.write_text("x", encoding="utf-8")
    for name in ("fast.gguf", "big.gguf", "ocr.gguf", "mmproj.gguf"):
        (tmp / name).write_bytes(b"gguf")

    registry = {
        "server_exe": str(exe),
        "models_dir": str(tmp),
        "default": "fast",
        "max_active": 1,
        "idle_timeout_seconds": 0,
        "router": {"fast": "fast", "strong": "big"},
        "models": [
            {"key": "fast", "label": "Fast 2B", "file": "fast.gguf",
             "port": 8080, "min_free_mb": 0},
            {"key": "big", "label": "Big 8B", "file": "big.gguf",
             "port": 8082, "min_free_mb": 0},
            {"key": "cloud", "label": "Cloud 120B", "provider": "testcloud",
             "model": "cloud-120b", "api_key_env": "TEST_CLOUD_KEY",
             "base_url": "https://cloud.invalid/v1"},
            {"key": "ocr", "label": "OCR", "role": "ocr", "file": "ocr.gguf",
             "mmproj": "mmproj.gguf", "port": 8081, "min_free_mb": 0},
        ],
    }
    path = tmp / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


class ScriptedClient:
    """A chat client that streams, so token events are exercised too."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.seen: list[list[dict[str, Any]]] = []
        # What the model was actually offered, which is how the tool switches
        # are checked: the registry is what decides, not the roster endpoint.
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    def _next(
        self,
        messages: Iterable[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.seen.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        if not self._responses:
            raise AssertionError("ScriptedClient ran out of responses")
        return self._responses.pop(0)

    def chat(self, messages, *, tools=None):
        return self._next(messages, tools)

    def chat_stream(
        self, messages, *, tools=None, on_token=None, on_reasoning=None, should_stop=None
    ):
        message = dict(self._next(messages, tools))
        # `_reasoning` stands in for llama.cpp's reasoning_content deltas. The
        # real client never puts them in the assembled message either, so it is
        # popped rather than returned.
        thinking = message.pop("_reasoning", "")
        if on_reasoning and thinking:
            on_reasoning(thinking)

        content = message.get("content") or ""
        if on_token and content:
            # Two fragments, so the test can tell streaming from a single
            # buffered write.
            middle = max(1, len(content) // 2)
            on_token(content[:middle])
            on_token(content[middle:])
        return message


class FakeConnectivity:
    """Connectivity with the network under the test's control.

    The real one opens a socket. A test that depended on whether this machine
    happened to have internet would pass or fail for reasons that have nothing
    to do with the code.
    """

    def __init__(self, online: bool = True) -> None:
        self.is_online = online
        self.invalidated = 0

    def online(self, *, force: bool = False) -> bool:
        return self.is_online

    def invalidate(self) -> None:
        self.invalidated += 1


class HarnessRuntime(Runtime):
    """Runtime with the model client scripted and the manager stubbed."""

    def __init__(self, config: Config, manager: ManagerHarness) -> None:
        super().__init__(config, manager)
        self.responses: list[dict[str, Any]] = []
        self.clients: list[ScriptedClient] = []
        self.connectivity = FakeConnectivity()

    def make_client(self, config: Config, spec: Any) -> ScriptedClient:
        client = ScriptedClient(self.responses)
        self.clients.append(client)
        return client


def make_turn_request(prompt: str) -> Turn:
    """A turn that can be handed straight to the queue, bypassing the route."""
    return Turn(
        request=TurnRequest(
            conversation_id=1,
            prompt=prompt,
            user_message_id=1,
            model_key="fast",
        )
    )


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Turn a server-sent event stream into (event, data) pairs."""
    events = []
    for block in body.split("\n\n"):
        name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if name is not None:
            events.append((name, data or {}))
    return events


def kinds(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [name for name, _ in events]


def first(events: list[tuple[str, dict[str, Any]]], kind: str) -> dict[str, Any]:
    for name, data in events:
        if name == kind:
            return data
    raise AssertionError(f"no {kind!r} event in {kinds(events)}")


class ApiTestCase(unittest.TestCase):
    """Shared setup: temp database, temp workspace, stubbed manager."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        registry_path = write_registry(tmp)
        self.manager = ManagerHarness(registry_path)
        config = dataclasses.replace(
            Config(), db_path=tmp / "chat.db", workspace=tmp
        )
        self.runtime = HarnessRuntime(config, self.manager)
        self.client = TestClient(create_app(self.runtime))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def say(self, prompt: str, **body: Any) -> list[tuple[str, dict[str, Any]]]:
        response = self.client.post("/api/chat", json={"prompt": prompt, **body})
        self.assertEqual(response.status_code, 200, response.text)
        return parse_sse(response.text)


class ChatStreamTests(ApiTestCase):
    def test_a_plain_answer_streams_and_is_stored(self):
        self.runtime.responses = [{"role": "assistant", "content": "Paris."}]
        events = self.say("capital of France?")

        self.assertEqual(
            kinds(events),
            ["accepted", "model", "model", "start", "token", "token", "done"],
        )

        tokens = "".join(data["text"] for name, data in events if name == "token")
        self.assertEqual(tokens, "Paris.")

        done = first(events, "done")
        self.assertEqual(done["content"], "Paris.")
        self.assertEqual(done["model_key"], "fast")
        self.assertIsInstance(done["elapsed"], float)

        conversation_id = first(events, "accepted")["conversation_id"]
        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        self.assertEqual(
            [(m["role"], m["content"]) for m in stored["messages"]],
            [("user", "capital of France?"), ("assistant", "Paris.")],
        )

    def test_the_model_is_reported_loading_then_ready(self):
        """The long silence at the front of a cold turn needs its own events."""
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        events = self.say("hello")
        states = [data["state"] for name, data in events if name == "model"]
        self.assertEqual(states, ["loading", "ready"])

    def test_a_conversation_is_created_and_titled_from_the_prompt(self):
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        events = self.say("Summarise the build log")
        conversation_id = first(events, "accepted")["conversation_id"]

        listed = self.client.get("/api/conversations").json()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], conversation_id)
        self.assertEqual(listed[0]["title"], "Summarise the build log")

    def test_a_second_turn_continues_the_same_conversation(self):
        self.runtime.responses = [{"role": "assistant", "content": "first"}]
        conversation_id = first(self.say("one"), "accepted")["conversation_id"]

        self.runtime.responses = [{"role": "assistant", "content": "second"}]
        events = self.say("two", conversation_id=conversation_id)
        self.assertEqual(first(events, "accepted")["conversation_id"], conversation_id)

        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        self.assertEqual(
            [m["content"] for m in stored["messages"]],
            ["one", "first", "two", "second"],
        )

    def test_history_is_replayed_without_duplicating_the_new_prompt(self):
        """The agent appends the prompt itself, so history must stop before it."""
        self.runtime.responses = [{"role": "assistant", "content": "first"}]
        conversation_id = first(self.say("one"), "accepted")["conversation_id"]

        self.runtime.responses = [{"role": "assistant", "content": "second"}]
        self.say("two", conversation_id=conversation_id)

        sent = self.runtime.clients[-1].seen[0]
        self.assertEqual(
            [(m["role"], m.get("content")) for m in sent],
            [
                ("system", sent[0].get("content")),
                ("user", "one"),
                ("assistant", "first"),
                ("user", "two"),
            ],
        )

    def test_a_tool_round_reports_the_call(self):
        self.runtime.responses = [
            tool_call_message(("calculate", {"expression": "2+2"})),
            {"role": "assistant", "content": "4"},
        ]
        events = self.say("what is 2+2?")

        tool = first(events, "tool")
        self.assertEqual(tool["name"], "calculate")
        self.assertTrue(tool["ok"])
        self.assertIn("4", tool["summary"])

        done = first(events, "done")
        self.assertEqual([t["name"] for t in done["tools"]], ["calculate"])

        # Stored with the message, so a reload shows which tools ran.
        conversation_id = first(events, "accepted")["conversation_id"]
        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        self.assertEqual(stored["messages"][-1]["tools"][0]["name"], "calculate")

    def test_a_tool_call_carries_what_was_sent_and_what_came_back(self):
        """A one-line summary is not enough to check the agent's work."""
        self.runtime.responses = [
            tool_call_message(("calculate", {"expression": "2+2"})),
            {"role": "assistant", "content": "4"},
        ]
        events = self.say("what is 2+2?")

        tool = first(events, "tool")
        self.assertIn("2+2", tool["arguments"])
        self.assertIn("expression", tool["arguments"])
        self.assertIn("4", tool["output"])
        self.assertFalse(tool["clipped"])

        # And it survives a reload, not just the live stream.
        conversation_id = first(events, "accepted")["conversation_id"]
        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        recorded = stored["messages"][-1]["tools"][0]
        self.assertIn("2+2", recorded["arguments"])
        self.assertIn("4", recorded["output"])

    def test_a_huge_tool_result_is_clipped_and_says_so(self):
        """A file read may be 200,000 characters, and all of it would end up in
        the message row and go to the browser on every reload."""
        from api.runtime import TOOL_DISPLAY_LIMIT, tool_entry
        from agent.loop import ToolEvent
        from agent.parser import ToolCall as ParsedCall
        from tools.base import ToolResult

        event = ToolEvent(
            call=ParsedCall(id="1", name="read_text_file", arguments={"path": "x"}),
            result=ToolResult(
                name="read_text_file",
                payload={"success": True, "text": "x" * 50_000},
            ),
        )
        entry = tool_entry(event)
        self.assertTrue(entry["clipped"])
        self.assertLessEqual(len(entry["output"]), TOOL_DISPLAY_LIMIT)

    def test_an_unknown_model_is_refused_before_anything_is_queued(self):
        response = self.client.post(
            "/api/chat", json={"prompt": "hi", "model_key": "nope"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("nope", response.json()["detail"])
        self.assertEqual(self.client.get("/api/conversations").json(), [])

    def test_an_empty_prompt_is_refused(self):
        response = self.client.post("/api/chat", json={"prompt": "   "})
        self.assertEqual(response.status_code, 422)

    def test_a_model_error_is_reported_as_an_event(self):
        """By the time a turn runs the response has started; there is no status
        code left to set, so failures arrive as events."""
        from models.qwen import QwenConnectionError

        class Broken(ScriptedClient):
            def chat_stream(
                self, messages, *, tools=None, on_token=None, on_reasoning=None,
                should_stop=None,
            ):
                raise QwenConnectionError("server went away")

        self.runtime.make_client = lambda config, spec: Broken([])
        events = self.say("hello")

        error = first(events, "error")
        self.assertEqual(error["kind"], "agent")
        self.assertIn("server went away", error["message"])


class RoutingTests(ApiTestCase):
    def test_auto_routing_announces_the_switch_and_says_why(self):
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        events = self.say(
            "Please refactor and debug this architecture, step by step",
            auto_route=True,
        )

        route = first(events, "route")
        self.assertEqual(route["key"], "big")
        self.assertEqual(route["label"], "Big 8B")
        self.assertTrue(route["reason"])
        self.assertEqual(first(events, "done")["model_key"], "big")

    def test_a_simple_prompt_stays_on_the_fast_model(self):
        self.runtime.responses = [{"role": "assistant", "content": "hi"}]
        events = self.say("hello", auto_route=True)
        self.assertNotIn("route", kinds(events))
        self.assertEqual(first(events, "done")["model_key"], "fast")

    def test_a_conversation_never_routes_back_down(self):
        """Once the strong model is loaded, switching back costs more than the
        RAM is worth - so escalation has to survive into the next turn."""
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        events = self.say("debug this architecture and refactor it", auto_route=True)
        conversation_id = first(events, "accepted")["conversation_id"]
        self.assertEqual(first(events, "done")["model_key"], "big")

        self.runtime.responses = [{"role": "assistant", "content": "still big"}]
        events = self.say("hi", conversation_id=conversation_id, auto_route=True)
        self.assertEqual(first(events, "done")["model_key"], "big")

        detail = self.client.get(f"/api/conversations/{conversation_id}").json()
        self.assertTrue(detail["escalated"])


class ConversationRouteTests(ApiTestCase):
    def make_conversation(self, prompt: str = "hello") -> int:
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        return first(self.say(prompt), "accepted")["conversation_id"]

    def test_rename(self):
        conversation_id = self.make_conversation()
        response = self.client.patch(
            f"/api/conversations/{conversation_id}", json={"title": "Renamed"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Renamed")

    def test_delete_removes_it_and_its_messages(self):
        conversation_id = self.make_conversation()
        self.assertEqual(
            self.client.delete(f"/api/conversations/{conversation_id}").status_code,
            204,
        )
        self.assertEqual(
            self.client.get(f"/api/conversations/{conversation_id}").status_code, 404
        )
        self.assertEqual(self.client.get("/api/conversations").json(), [])

    def test_missing_conversation_is_a_404_not_a_crash(self):
        self.assertEqual(self.client.get("/api/conversations/999").status_code, 404)
        self.assertEqual(self.client.delete("/api/conversations/999").status_code, 404)
        self.assertEqual(
            self.client.patch(
                "/api/conversations/999", json={"title": "x"}
            ).status_code,
            404,
        )


class ModelRouteTests(ApiTestCase):
    def test_listing_reports_every_model_and_the_router_pair(self):
        body = self.client.get("/api/models").json()
        # Membership, not the exact list: adding a model to the fixture should
        # not break a test about the router pair.
        keys = [m["key"] for m in body["models"]]
        self.assertIn("fast", keys)
        self.assertIn("big", keys)
        self.assertEqual(body["router_fast"], "fast")
        self.assertEqual(body["router_strong"], "big")
        self.assertEqual(body["default_key"], "fast")
        self.assertEqual(body["max_active"], 1)

    def test_load_then_unload(self):
        body = self.client.post("/api/models/big/load").json()
        states = {m["key"]: m["state"] for m in body["models"]}
        self.assertEqual(states["big"], "ready")
        self.assertEqual(body["active_key"], "big")

        body = self.client.post("/api/models/big/unload").json()
        states = {m["key"]: m["state"] for m in body["models"]}
        self.assertEqual(states["big"], "stopped")

    def test_loading_one_model_unloads_the_other(self):
        """max_active is 1: 8 GB cannot hold two."""
        self.client.post("/api/models/fast/load")
        body = self.client.post("/api/models/big/load").json()
        states = {m["key"]: m["state"] for m in body["models"]}
        self.assertEqual(states["big"], "ready")
        self.assertEqual(states["fast"], "stopped")

    def test_unknown_model_is_a_404(self):
        self.assertEqual(
            self.client.post("/api/models/nope/unload").status_code, 404
        )

    def test_gpu_layers_and_the_server_binary_are_both_reported(self):
        """One is useless without the other. The number only does anything
        against a llama-server with a GPU backend compiled in, so a panel that
        asked for it without naming the binary would be asking blind."""
        body = self.client.get("/api/models").json()
        fast = next(m for m in body["models"] if m["key"] == "fast")
        self.assertEqual(fast["gpu_layers"], 0)
        self.assertTrue(body["server_exe"])

    def test_gpu_layers_can_be_retuned(self):
        body = self.client.patch(
            "/api/models/fast", json={"gpu_layers": 99}
        ).json()
        fast = next(m for m in body["models"] if m["key"] == "fast")
        self.assertEqual(fast["gpu_layers"], 99)
        self.assertTrue(fast["customised"])

    def test_turning_offloading_back_off_is_not_mistaken_for_no_change(self):
        """0 is a value, not an empty field. `exclude_none` has to be what
        drops an untouched field, never falsiness."""
        self.client.patch("/api/models/fast", json={"gpu_layers": 99})
        body = self.client.patch("/api/models/fast", json={"gpu_layers": 0}).json()
        fast = next(m for m in body["models"] if m["key"] == "fast")
        self.assertEqual(fast["gpu_layers"], 0)

    def test_a_negative_layer_count_is_refused_at_the_edge(self):
        response = self.client.patch("/api/models/fast", json={"gpu_layers": -1})
        self.assertEqual(response.status_code, 422)


class SpeechRouteTests(ApiTestCase):
    """Dictation, with whisper itself replaced.

    What is worth testing at this layer is what the route refuses and what it
    leaves behind - not whether whisper.cpp transcribes, which has its own
    tests in test_speech.py.
    """

    def test_the_status_says_it_is_off_when_nothing_is_installed(self):
        with mock.patch("api.routes.speech.probe", return_value=None):
            body = self.client.get("/api/speech").json()
        self.assertFalse(body["available"])
        # Says what is missing, because the person can fix this and the UI
        # otherwise just silently has no microphone.
        self.assertIn("whisper", body["detail"].lower())

    def test_the_status_names_the_model_when_it_is_installed(self):
        from speech.whisper import WhisperInfo

        found = WhisperInfo(path="/x/whisper-cli", model="/x/ggml-base.en.bin")
        with mock.patch("api.routes.speech.probe", return_value=found):
            body = self.client.get("/api/speech").json()
        self.assertTrue(body["available"])
        self.assertEqual(body["model"], "base.en")

    def test_a_clip_comes_back_as_text(self):
        with mock.patch(
            "api.routes.speech.transcribe", return_value="open the file"
        ):
            response = self.client.post(
                "/api/speech/transcribe",
                files={"file": ("dictation.wav", b"RIFF" + b"\0" * 100, "audio/wav")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "open the file")

    def test_the_clip_is_deleted_whether_or_not_it_worked(self):
        """A recording of somebody's voice is the most personal thing this
        handles, and the only reason to keep one would be to debug this
        route."""
        seen = {}

        def remember(path, **kwargs):
            seen["path"] = Path(path)
            seen["existed"] = Path(path).is_file()
            return "hello"

        with mock.patch("api.routes.speech.transcribe", side_effect=remember):
            self.client.post(
                "/api/speech/transcribe",
                files={"file": ("dictation.wav", b"RIFF" + b"\0" * 100, "audio/wav")},
            )
        self.assertTrue(seen["existed"], "the clip was never written")
        self.assertFalse(seen["path"].exists(), "the clip was left behind")

    def test_the_clip_is_deleted_even_when_transcription_fails(self):
        from speech.whisper import WhisperError

        seen = {}

        def blow_up(path, **kwargs):
            seen["path"] = Path(path)
            raise WhisperError("no model")

        with mock.patch("api.routes.speech.transcribe", side_effect=blow_up):
            response = self.client.post(
                "/api/speech/transcribe",
                files={"file": ("dictation.wav", b"RIFF" + b"\0" * 100, "audio/wav")},
            )
        self.assertEqual(response.status_code, 503)
        self.assertFalse(seen["path"].exists())

    def test_something_that_is_not_audio_is_refused(self):
        response = self.client.post(
            "/api/speech/transcribe",
            files={"file": ("clip.exe", b"MZ", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 415)

    def test_an_empty_clip_is_refused(self):
        response = self.client.post(
            "/api/speech/transcribe",
            files={"file": ("dictation.wav", b"", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)

    def test_an_oversized_clip_is_refused_while_it_is_still_arriving(self):
        """Enforced during the read, not after: the cap exists so a large body
        is never materialised, and checking afterwards would defeat it."""
        with mock.patch("api.routes.speech.MAX_BYTES", 1024):
            response = self.client.post(
                "/api/speech/transcribe",
                files={"file": ("dictation.wav", b"\0" * 8192, "audio/wav")},
            )
        self.assertEqual(response.status_code, 413)


class ReasoningTests(ApiTestCase):
    """Thinking is shown to the user but never fed back to the model."""

    def test_reasoning_streams_as_its_own_event(self):
        self.runtime.responses = [
            {
                "role": "assistant",
                "content": "Paris.",
                "_reasoning": "The capital of France is Paris.",
            }
        ]
        events = self.say("capital of France?")

        self.assertIn("reasoning", kinds(events))
        self.assertEqual(
            first(events, "reasoning")["text"], "The capital of France is Paris."
        )
        # It arrives on a separate channel, so it can never be mistaken for
        # part of the answer.
        tokens = "".join(data["text"] for name, data in events if name == "token")
        self.assertEqual(tokens, "Paris.")

    def test_reasoning_is_not_stored(self):
        self.runtime.responses = [
            {"role": "assistant", "content": "Paris.", "_reasoning": "thinking hard"}
        ]
        conversation_id = first(
            self.say("capital of France?"), "accepted"
        )["conversation_id"]

        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        blob = json.dumps(stored)
        self.assertNotIn("thinking hard", blob)
        self.assertEqual(stored["messages"][-1]["content"], "Paris.")

    def test_reasoning_is_never_replayed_to_the_model(self):
        """The chat template does not expect a thinking trace coming back in."""
        self.runtime.responses = [
            {"role": "assistant", "content": "first", "_reasoning": "private thoughts"}
        ]
        conversation_id = first(self.say("one"), "accepted")["conversation_id"]

        self.runtime.responses = [{"role": "assistant", "content": "second"}]
        self.say("two", conversation_id=conversation_id)

        sent = json.dumps(self.runtime.clients[-1].seen[0])
        self.assertNotIn("private thoughts", sent)


class ToolSwitchTests(ApiTestCase):
    def switches(self) -> dict[str, dict[str, Any]]:
        body = self.client.get("/api/tools").json()
        return {entry["id"]: entry for entry in body["switches"]}

    def test_everything_risky_is_off_to_begin_with(self):
        for identifier, entry in self.switches().items():
            self.assertFalse(entry["enabled"], f"{identifier} was on by default")

    def test_turning_a_switch_on_offers_the_tool_to_the_model(self):
        body = self.client.post("/api/tools/python", json={"enabled": True}).json()
        names = {tool["name"] for tool in body["tools"]}
        self.assertIn("run_python", names)

        # And the next turn actually sees it.
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        self.say("hello")
        offered = {
            definition["function"]["name"]
            for definition in (self.runtime.clients[-1].tools_seen[0] or [])
        }
        self.assertIn("run_python", offered)

    def test_turning_it_off_again_withdraws_it(self):
        self.client.post("/api/tools/python", json={"enabled": True})
        body = self.client.post("/api/tools/python", json={"enabled": False}).json()
        self.assertNotIn(
            "run_python", {tool["name"] for tool in body["tools"]}
        )

    def test_a_sharp_end_needs_its_parent_first(self):
        response = self.client.post(
            "/api/tools/git_writes", json={"enabled": True}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Git", response.json()["detail"])

    def test_turning_a_parent_off_takes_its_sharp_end_with_it(self):
        """Otherwise you could end up with unrestricted Python and no Python."""
        self.client.post("/api/tools/python", json={"enabled": True})
        self.client.post("/api/tools/python_unrestricted", json={"enabled": True})
        self.assertTrue(self.switches()["python_unrestricted"]["enabled"])

        self.client.post("/api/tools/python", json={"enabled": False})
        self.assertFalse(self.switches()["python_unrestricted"]["enabled"])

    def test_the_risk_text_survives_being_enabled(self):
        """The warning matters most when the switch is on, so it must not be
        harvested from the disabled list alone."""
        before = self.switches()["terminal"]["risk"]
        self.assertIn("AGENT_ENABLE_SHELL_TOOL", before)

        self.client.post("/api/tools/terminal", json={"enabled": True})
        after = self.switches()["terminal"]
        self.assertTrue(after["enabled"])
        self.assertEqual(after["risk"], before)

    def test_the_ocr_switch_starts_and_stops_its_server(self):
        """One toggle, because the tool is useless without the server and the
        server is dead weight without the tool."""
        from models.manager import ModelState

        self.client.post("/api/tools/ocr", json={"enabled": True})
        self.assertIs(self.manager.status("ocr").state, ModelState.READY)
        self.assertIn("ocr_image", {t["name"] for t in self.client.get("/api/tools").json()["tools"]})

        self.client.post("/api/tools/ocr", json={"enabled": False})
        self.assertIs(self.manager.status("ocr").state, ModelState.STOPPED)

    def test_ocr_reads_as_off_while_its_server_is_down(self):
        """The flag alone would show a switch that is on while every use fails."""
        self.client.post("/api/tools/ocr", json={"enabled": True})
        self.assertTrue(self.switches()["ocr"]["enabled"])

        # The server dies underneath us, as a llama-server can.
        self.manager.healthy_ports.discard(8081)
        self.manager._processes.pop("ocr", None)
        self.assertFalse(self.switches()["ocr"]["enabled"])

    def test_an_unknown_switch_is_a_404(self):
        response = self.client.post("/api/tools/nonsense", json={"enabled": True})
        self.assertEqual(response.status_code, 404)

    def test_overrides_do_not_claim_to_come_from_the_environment(self):
        self.client.post("/api/tools/memory", json={"enabled": True})
        entry = self.switches()["memory"]
        self.assertTrue(entry["enabled"])
        self.assertFalse(entry["from_env"])


class RemoteModelTests(ApiTestCase):
    """Hosted models: consent before routing to one, and offline fallback."""

    def setUp(self) -> None:
        super().setUp()
        # A key has to look present, or the model is unavailable for a reason
        # unrelated to what each test is about.
        self._previous = os.environ.get("TEST_CLOUD_KEY")
        os.environ["TEST_CLOUD_KEY"] = "not-a-real-key"
        self.addCleanup(self._restore_key)

    def _restore_key(self) -> None:
        if self._previous is None:
            os.environ.pop("TEST_CLOUD_KEY", None)
        else:
            os.environ["TEST_CLOUD_KEY"] = self._previous

    def route_to_cloud(self) -> None:
        """Make the auto-router's strong model the hosted one."""
        self.manager.router_strong = "cloud"

    def test_a_hosted_model_is_listed_with_its_provider(self):
        body = self.client.get("/api/models").json()
        cloud = next(m for m in body["models"] if m["key"] == "cloud")
        self.assertTrue(cloud["remote"])
        self.assertEqual(cloud["provider"], "testcloud")
        self.assertTrue(cloud["has_key"])
        self.assertTrue(cloud["usable"])
        self.assertTrue(body["online"])

    def test_a_hosted_model_is_never_the_active_one(self):
        """active_key answers 'what is holding RAM', and a hosted model is not."""
        body = self.client.get("/api/models").json()
        self.assertIsNone(body["active_key"])

    def test_no_key_means_unusable_and_says_which_variable(self):
        os.environ.pop("TEST_CLOUD_KEY", None)
        body = self.client.get("/api/models").json()
        cloud = next(m for m in body["models"] if m["key"] == "cloud")
        self.assertFalse(cloud["has_key"])
        self.assertFalse(cloud["usable"])
        self.assertIn("TEST_CLOUD_KEY", cloud["error"])

    def test_offline_makes_it_unusable_but_still_keyed(self):
        self.runtime.connectivity.is_online = False
        body = self.client.get("/api/models").json()
        cloud = next(m for m in body["models"] if m["key"] == "cloud")
        self.assertTrue(cloud["has_key"])
        self.assertFalse(cloud["usable"])
        self.assertFalse(body["online"])

    def test_routing_to_a_hosted_model_needs_confirmation_first(self):
        """Nothing may be stored or queued before the user agrees."""
        self.route_to_cloud()
        response = self.client.post(
            "/api/chat",
            json={
                "prompt": "debug this architecture and refactor it",
                "auto_route": True,
            },
        )
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["kind"], "remote_confirmation_required")
        self.assertEqual(detail["model_key"], "cloud")
        self.assertEqual(detail["provider"], "testcloud")

        # And nothing happened.
        self.assertEqual(self.client.get("/api/conversations").json(), [])

    def test_confirming_lets_the_turn_run(self):
        self.route_to_cloud()
        self.runtime.responses = [{"role": "assistant", "content": "done"}]
        events = self.say(
            "debug this architecture and refactor it",
            auto_route=True,
            confirm_remote=True,
        )
        self.assertEqual(first(events, "done")["model_key"], "cloud")
        self.assertTrue(first(events, "route")["remote"])

    def test_choosing_a_hosted_model_yourself_needs_no_confirmation(self):
        """Picking it is already deliberate; being moved onto it is not."""
        self.runtime.responses = [{"role": "assistant", "content": "done"}]
        events = self.say("hello", model_key="cloud")
        self.assertEqual(first(events, "done")["model_key"], "cloud")
        self.assertNotIn("route", kinds(events))

    def test_offline_falls_back_to_local_and_says_so(self):
        self.runtime.connectivity.is_online = False
        self.runtime.responses = [{"role": "assistant", "content": "local answer"}]
        events = self.say("hello", model_key="cloud")

        fallback = first(events, "fallback")
        self.assertEqual(fallback["from"], "cloud")
        self.assertEqual(fallback["to"], "fast")
        self.assertIn("internet", fallback["reason"])
        self.assertEqual(first(events, "done")["model_key"], "fast")

    def test_the_model_event_says_whether_the_turn_left_the_machine(self):
        self.runtime.responses = [{"role": "assistant", "content": "done"}]
        events = self.say("hello", model_key="cloud")
        model = first(events, "model")
        self.assertTrue(model["remote"])
        self.assertEqual(model["provider"], "testcloud")


class UploadTests(ApiTestCase):
    """Uploads land in the workspace, because that is the only place the OCR
    tool can read from."""

    PNG = bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000a49444154789c6360000002000100ffff0300000600"
        "0557bfabd40000000049454e44ae426082"
    )

    def upload(self, name: str, data: bytes):
        return self.client.post(
            "/api/uploads", files={"file": (name, data, "image/png")}
        )

    def test_an_image_is_stored_and_its_workspace_path_returned(self):
        response = self.upload("note.png", self.PNG)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertTrue(body["path"].startswith("uploads/"))
        self.assertTrue(body["path"].endswith(".png"))
        self.assertEqual(body["size"], len(self.PNG))
        # Really on disk, inside the workspace, where ocr_image resolves from.
        stored = Path(self.runtime.config.workspace) / body["path"]
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.read_bytes(), self.PNG)

    def test_a_hostile_filename_cannot_escape_the_directory(self):
        body = self.upload("../../../etc/passwd.png", self.PNG).json()
        self.assertNotIn("..", body["path"])
        resolved = (Path(self.runtime.config.workspace) / body["path"]).resolve()
        self.assertIn(
            Path(self.runtime.config.workspace).resolve(), resolved.parents
        )

    def test_two_uploads_of_one_name_do_not_collide(self):
        first_path = self.upload("scan.png", self.PNG).json()["path"]
        second_path = self.upload("scan.png", self.PNG).json()["path"]
        self.assertNotEqual(first_path, second_path)

    def test_a_non_image_is_refused(self):
        response = self.client.post(
            "/api/uploads", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        self.assertEqual(response.status_code, 415)

    def test_an_oversized_file_is_refused_and_leaves_nothing_behind(self):
        limit = self.runtime.config.ocr_max_image_bytes
        response = self.upload("huge.png", b"\x89PNG" + b"x" * (limit + 10))
        self.assertEqual(response.status_code, 413)

        directory = Path(self.runtime.config.workspace) / "uploads"
        self.assertEqual(list(directory.glob("*")) if directory.exists() else [], [])

    def test_an_empty_file_is_refused(self):
        self.assertEqual(self.upload("empty.png", b"").status_code, 400)

    def test_it_says_when_ocr_is_off_rather_than_failing_later(self):
        body = self.upload("note.png", self.PNG).json()
        self.assertFalse(body["ocr_ready"])
        self.assertIn("OCR", body["hint"])

    def test_an_attachment_is_named_in_the_prompt(self):
        """The model has no other way to know the file exists."""
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        events = self.say("read this", attachments=["uploads/ab-note.png"])

        conversation_id = first(events, "accepted")["conversation_id"]
        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        asked = stored["messages"][0]["content"]
        self.assertIn("read this", asked)
        self.assertIn("uploads/ab-note.png", asked)
        self.assertIn("ocr_image", asked)

    def test_an_attachment_alone_is_a_valid_turn(self):
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        events = self.say("", attachments=["uploads/ab-note.png"])
        self.assertIn("done", kinds(events))

    def test_no_attachment_leaves_the_prompt_untouched(self):
        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        events = self.say("plain question")
        conversation_id = first(events, "accepted")["conversation_id"]
        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        self.assertEqual(stored["messages"][0]["content"], "plain question")


class StoppingTurnTests(ApiTestCase):
    """Ending a turn for real, rather than only stopping watching it."""

    def stop_midway(self, after: int = 1):
        """A client that asks the queue to stop this turn while it generates.

        Standing in for the click: the request thread cannot reach the running
        turn while the test is blocked reading its stream, so the stop is asked
        for from inside - through the same `stop_turn` the endpoint calls.
        """
        runtime = self.runtime

        class StopsItself(ScriptedClient):
            def chat_stream(
                self, messages, *, tools=None, on_token=None, on_reasoning=None,
                should_stop=None,
            ):
                message = dict(self._next(messages, tools))
                content = message.get("content") or ""
                for index, word in enumerate(content.split(" ")):
                    if should_stop is not None and should_stop():
                        return {"role": "assistant", "content": " ".join(
                            content.split(" ")[:index]
                        )}
                    if on_token:
                        on_token(word + " ")
                    if index + 1 >= after:
                        current = runtime.queue._current
                        if current is not None:
                            runtime.queue.stop_turn(current.id)
                return message

        self.runtime.make_client = lambda config, spec: StopsItself(
            self.runtime.responses
        )

    def test_a_running_turn_ends_and_says_so(self):
        self.stop_midway()
        self.runtime.responses = [
            {"role": "assistant", "content": "one two three four five"}
        ]
        events = self.say("go on for a while")

        self.assertIn("stopped", kinds(events))
        self.assertNotIn("done", kinds(events))
        # Not an error: it did exactly what it was told.
        self.assertNotIn("error", kinds(events))

    def test_what_had_been_generated_is_kept(self):
        """Two minutes of prose is not worth throwing away over a missing
        last word, on a machine this slow."""
        self.stop_midway(after=2)
        self.runtime.responses = [
            {"role": "assistant", "content": "one two three four five"}
        ]
        events = self.say("go on")

        stopped = first(events, "stopped")
        self.assertTrue(stopped["content"].startswith("one two"))
        self.assertIsNotNone(stopped["message_id"])

    def test_the_stored_message_admits_it_was_cut_short(self):
        """Otherwise a reload shows a truncated answer as a finished one."""
        self.stop_midway()
        self.runtime.responses = [
            {"role": "assistant", "content": "one two three four five"}
        ]
        events = self.say("go on")

        conversation_id = first(events, "accepted")["conversation_id"]
        stored = self.client.get(f"/api/conversations/{conversation_id}").json()
        answer = stored["messages"][-1]
        self.assertEqual(answer["role"], "assistant")
        self.assertIn("stopped before finishing", answer["content"])

    def test_the_queue_is_free_afterwards(self):
        """The reason for stopping at all: the next turn can start."""
        self.stop_midway()
        self.runtime.responses = [{"role": "assistant", "content": "one two three"}]
        self.say("stop me")

        self.assertFalse(self.client.get("/api/health").json()["busy"])

        self.runtime.make_client = HarnessRuntime.make_client.__get__(
            self.runtime, HarnessRuntime
        )
        self.runtime.responses = [{"role": "assistant", "content": "second turn"}]
        self.assertEqual(first(self.say("and now?"), "done")["content"], "second turn")

    def test_stopping_a_finished_turn_is_not_an_error(self):
        """By the time anyone clicks, the turn may have got there first."""
        self.runtime.responses = [{"role": "assistant", "content": "quick"}]
        turn_id = first(self.say("hello"), "accepted")["turn_id"]

        response = self.client.post(f"/api/chat/{turn_id}/stop")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "unknown")

    def test_a_queued_turn_is_dropped_through_the_endpoint(self):
        """Proves the route is wired to the queue, not only that it answers.

        The worker is held on a turn that will not finish until the test lets
        it, so the second one is genuinely queued rather than queued-if-the-
        scheduler-cooperates.
        """
        held = threading.Event()
        self.addCleanup(held.set)

        class Blocking(ScriptedClient):
            def chat_stream(self, messages, *, tools=None, **_):
                held.wait(timeout=5)
                return {"role": "assistant", "content": "let go"}

        self.runtime.make_client = lambda config, spec: Blocking([])
        self.runtime.queue.submit(make_turn_request("blocker"))
        for _ in range(200):
            if self.runtime.queue.busy():
                break
            time.sleep(0.01)

        waiting = self.runtime.queue.submit(make_turn_request("waiting"))
        response = self.client.post(f"/api/chat/{waiting.id}/stop")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "queued")
        self.assertEqual(self.runtime.queue.depth(), 0)


class RewindTests(ApiTestCase):
    """Editing a question is built on deleting from it onwards."""

    def conversation_with(self, *contents: str) -> tuple[int, list[int]]:
        conversation = self.runtime.store.create_conversation(title="Old title")
        ids = []
        for index, content in enumerate(contents):
            ids.append(
                self.runtime.store.add_message(
                    conversation,
                    "user" if index % 2 == 0 else "assistant",
                    content,
                )
            )
        return conversation, ids

    def rewind(self, conversation: int, message_id: int):
        return self.client.delete(
            f"/api/conversations/{conversation}/messages/{message_id}"
        )

    def test_it_takes_the_message_and_everything_after_it(self):
        """The answers were a reply to a question that is about to change."""
        conversation, ids = self.conversation_with("one", "first", "two", "second")

        response = self.rewind(conversation, ids[2])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"removed": 2, "emptied": False})

        left = self.client.get(f"/api/conversations/{conversation}").json()
        self.assertEqual([m["content"] for m in left["messages"]], ["one", "first"])

    def test_it_says_when_nothing_is_left(self):
        conversation, ids = self.conversation_with("only", "answer")
        self.assertEqual(self.rewind(conversation, ids[0]).json()["emptied"], True)

    def test_a_conversation_left_empty_is_retitled_by_the_next_question(self):
        """Otherwise the history list goes on naming a question that is gone."""
        conversation, ids = self.conversation_with("what is 2+2?", "4")
        self.rewind(conversation, ids[0])

        self.runtime.responses = [{"role": "assistant", "content": "8"}]
        self.say("what is 4+4?", conversation_id=conversation)

        listed = self.client.get("/api/conversations").json()
        titles = {item["id"]: item["title"] for item in listed}
        self.assertEqual(titles[conversation], "what is 4+4?")

    def test_another_conversation_is_untouched(self):
        """Ids are global, so "every id from here up" has to be scoped."""
        first, _ = self.conversation_with("keep", "me")
        second, second_ids = self.conversation_with("drop", "me")

        self.assertEqual(self.rewind(second, second_ids[0]).json()["removed"], 2)

        left = self.client.get(f"/api/conversations/{first}").json()
        self.assertEqual([m["content"] for m in left["messages"]], ["keep", "me"])

    def test_a_message_from_another_conversation_is_a_404(self):
        first, first_ids = self.conversation_with("mine", "answer")
        second, _ = self.conversation_with("theirs")

        response = self.rewind(second, first_ids[0])
        self.assertEqual(response.status_code, 404)
        # And nothing was taken from either one.
        kept = self.client.get(f"/api/conversations/{first}").json()
        self.assertEqual(len(kept["messages"]), 2)

    def test_an_unknown_conversation_is_a_404(self):
        self.assertEqual(self.rewind(9999, 1).status_code, 404)

    def test_it_is_refused_while_a_turn_is_in_flight(self):
        """A queued turn is identified by its own user message id, and reads
        its history when it runs. Deleting underneath it breaks both."""
        conversation, ids = self.conversation_with("one", "first")
        self.runtime.queue.busy = lambda: True  # type: ignore[method-assign]

        response = self.rewind(conversation, ids[0])
        self.assertEqual(response.status_code, 409)
        self.assertIn("stop it first", response.json()["detail"])

    def test_the_rewound_history_is_what_the_next_turn_sees(self):
        """The point of all of it: the model must not be told about the
        question that was replaced."""
        conversation, ids = self.conversation_with(
            "what is 2+2?", "4", "and times 3?", "12"
        )
        self.rewind(conversation, ids[2])

        self.runtime.responses = [{"role": "assistant", "content": "2"}]
        self.say("and minus 2?", conversation_id=conversation)

        sent = self.runtime.clients[-1].seen[0]
        self.assertEqual(
            [(m["role"], m.get("content")) for m in sent[1:]],
            [
                ("user", "what is 2+2?"),
                ("assistant", "4"),
                ("user", "and minus 2?"),
            ],
        )


class BacklogTests(ApiTestCase):
    """The queue is bounded, and a refusal has to reach the person asking."""

    def test_a_refused_backlog_is_a_429_that_says_why(self):
        def full(turn):
            raise TurnQueueFull(
                "8 turns are already waiting. On this hardware that is well "
                "over an hour of work."
            )

        self.runtime.queue.submit = full  # type: ignore[method-assign]
        response = self.client.post("/api/chat", json={"prompt": "one more"})

        self.assertEqual(response.status_code, 429)
        self.assertIn("already waiting", response.json()["detail"])


class QueuedTurnHistoryTests(ApiTestCase):
    """What a turn sees when it was queued behind another one.

    Rows are written in an order that is not conversation order: a queued
    turn's question is stored when it is accepted, which is *before* the
    running turn's answer is stored. Selecting history by id alone therefore
    drops the answer the user is most likely replying to.
    """

    def run_directly(self, turn: Turn) -> None:
        """Run one turn on this thread, bypassing the queue.

        The queue is not what is under test here, and driving it through the
        endpoint would mean two overlapping SSE streams in a synchronous test
        client - which tests the test, not the code.
        """
        self.runtime.run_turn(turn)

    def test_a_queued_turn_sees_the_answer_that_came_before_it(self):
        conversation = self.runtime.store.create_conversation(title="t")
        store = self.runtime.store
        first_question = store.add_message(conversation, "user", "what is 2+2?")
        # Accepted while the first turn is still running, so its row lands
        # before the answer to the question ahead of it.
        second_question = store.add_message(conversation, "user", "and times 3?")
        store.add_message(conversation, "assistant", "4")

        self.runtime.responses = [{"role": "assistant", "content": "12"}]
        self.run_directly(
            Turn(
                request=TurnRequest(
                    conversation_id=conversation,
                    prompt="and times 3?",
                    user_message_id=second_question,
                    model_key="fast",
                )
            )
        )

        sent = self.runtime.clients[-1].seen[0]
        self.assertEqual(
            [(m["role"], m.get("content")) for m in sent[1:]],
            [
                ("user", "what is 2+2?"),
                ("assistant", "4"),
                ("user", "and times 3?"),
            ],
        )
        self.assertNotEqual(first_question, second_question)

    def test_it_does_not_see_questions_still_waiting_behind_it(self):
        """A prompt queued after this one has no answer yet, and putting it in
        the history would have the model answering the wrong question."""
        conversation = self.runtime.store.create_conversation(title="t")
        store = self.runtime.store
        mine = store.add_message(conversation, "user", "mine")
        store.add_message(conversation, "user", "queued behind me")

        self.runtime.responses = [{"role": "assistant", "content": "ok"}]
        self.run_directly(
            Turn(
                request=TurnRequest(
                    conversation_id=conversation,
                    prompt="mine",
                    user_message_id=mine,
                    model_key="fast",
                )
            )
        )

        sent = self.runtime.clients[-1].seen[0]
        self.assertEqual(
            [m.get("content") for m in sent[1:]], ["mine"]
        )


class WorkspaceRouteTests(ApiTestCase):
    """Moving the jail, which is the point: an agent that can only ever look
    at its own source is not much use."""

    def setUp(self) -> None:
        super().setUp()
        self.elsewhere = Path(self._tmp.name) / "elsewhere"
        self.elsewhere.mkdir()
        (self.elsewhere / "marker.txt").write_text("found me", encoding="utf-8")

    def workspace(self) -> dict[str, Any]:
        return self.client.get("/api/workspace").json()

    def move_to(self, path: Any):
        return self.client.post("/api/workspace", json={"path": str(path)})

    def test_it_starts_where_the_environment_put_it(self):
        body = self.workspace()
        self.assertEqual(body["path"], str(self.runtime.config.workspace))
        self.assertEqual(body["default"], str(self.runtime.config.workspace))
        self.assertTrue(body["from_env"])

    def test_moving_it_moves_what_the_tools_actually_read(self):
        """The roster saying so proves nothing; the tool result does."""
        self.assertEqual(self.move_to(self.elsewhere).status_code, 200)

        self.runtime.responses = [
            tool_call_message(("list_directory", {"path": "."})),
            {"role": "assistant", "content": "one file"},
        ]
        events = self.say("what is in here?")
        self.assertIn("marker.txt", first(events, "tool")["output"])

    def test_it_is_resolved_rather_than_taken_as_typed(self):
        body = self.move_to(f"{self.elsewhere}{os.sep}.{os.sep}").json()
        self.assertEqual(body["path"], str(self.elsewhere.resolve()))
        self.assertFalse(body["from_env"])

    def test_uploads_follow_it(self):
        """Anywhere else and the agent could not read its own attachment."""
        self.move_to(self.elsewhere)
        body = self.client.post(
            "/api/uploads",
            files={"file": ("note.png", UploadTests.PNG, "image/png")},
        ).json()
        self.assertTrue((self.elsewhere / body["path"]).is_file())

    def test_resetting_returns_to_the_environment(self):
        self.move_to(self.elsewhere)
        body = self.client.delete("/api/workspace").json()
        self.assertEqual(body["path"], str(self.runtime.config.workspace))
        self.assertTrue(body["from_env"])

    def test_recent_folders_are_offered_back(self):
        self.move_to(self.elsewhere)
        recent = self.workspace()["recent"]
        self.assertEqual(recent[0], str(self.elsewhere))
        self.assertIn(str(self.runtime.config.workspace), recent)

    def test_a_file_is_not_a_workspace(self):
        response = self.move_to(self.elsewhere / "marker.txt")
        self.assertEqual(response.status_code, 400)
        self.assertIn("not a folder", response.json()["detail"])

    def test_a_folder_that_is_not_there_is_refused(self):
        response = self.move_to(self.elsewhere / "nothing-here")
        self.assertEqual(response.status_code, 400)
        self.assertIn("no such folder", response.json()["detail"])

    def test_a_drive_root_is_refused(self):
        """A jail around the whole disk is not a jail."""
        root = Path(self.elsewhere.anchor or "/")
        response = self.move_to(root)
        self.assertEqual(response.status_code, 400)
        self.assertIn("root of a drive", response.json()["detail"])

    def test_the_operating_system_is_refused(self):
        system = os.environ.get("SystemRoot") or "/etc"
        if not Path(system).is_dir():
            self.skipTest("no system directory to point at")
        response = self.move_to(system)
        self.assertEqual(response.status_code, 400)
        self.assertIn("operating system", response.json()["detail"])

    def test_it_will_not_move_mid_turn(self):
        self.runtime.queue.busy = lambda: True  # type: ignore[method-assign]
        self.assertEqual(self.move_to(self.elsewhere).status_code, 409)
        self.assertEqual(self.client.delete("/api/workspace").status_code, 409)

    def test_it_reports_which_tools_would_act_on_the_folder(self):
        self.assertEqual(self.workspace()["active_tools"], [])
        self.assertFalse(self.workspace()["writable"])

        self.client.post("/api/tools/file_writes", json={"enabled": True})
        body = self.workspace()
        self.assertIn("File writes", body["active_tools"])
        self.assertTrue(body["writable"])

    def test_health_reports_the_folder_in_force_not_the_startup_one(self):
        self.move_to(self.elsewhere)
        self.assertEqual(
            self.client.get("/api/health").json()["workspace"], str(self.elsewhere)
        )


class WorkspaceBrowseTests(ApiTestCase):
    """The picker walks the filesystem here, because a browser cannot tell a
    page the real path of a folder someone chose."""

    def setUp(self) -> None:
        super().setUp()
        self.root = Path(self._tmp.name)
        (self.root / "notes").mkdir()
        (self.root / "notes" / "deeper").mkdir()
        (self.root / "not-a-folder.txt").write_text("x", encoding="utf-8")

    def browse(self, path: Any = "") -> dict[str, Any]:
        response = self.client.get("/api/workspace/browse", params={"path": str(path)})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_it_lists_sub_folders_and_nothing_else(self):
        body = self.browse(self.root)
        names = {entry["name"] for entry in body["entries"]}
        self.assertIn("notes", names)
        self.assertNotIn("not-a-folder.txt", names)

    def test_an_entry_carries_the_absolute_path_the_api_needs(self):
        entry = next(e for e in self.browse(self.root)["entries"] if e["name"] == "notes")
        self.assertEqual(Path(entry["path"]), self.root / "notes")
        # And it is directly usable, which is the whole contract with the UI.
        self.assertEqual(
            self.client.post("/api/workspace", json={"path": entry["path"]}).json()[
                "path"
            ],
            str(self.root / "notes"),
        )

    def test_it_walks_up_and_down(self):
        deeper = self.browse(self.root / "notes")
        self.assertEqual(Path(deeper["parent"]), self.root)
        self.assertEqual(
            [entry["name"] for entry in deeper["entries"]], ["deeper"]
        )

    def test_no_path_means_the_current_workspace(self):
        self.assertEqual(
            Path(self.browse()["path"]), Path(self.runtime.workspace)
        )

    def test_it_offers_somewhere_to_start(self):
        roots = self.browse(self.root)["roots"]
        self.assertTrue(roots)
        self.assertTrue(all(Path(entry["path"]).is_dir() for entry in roots))

    def test_a_missing_folder_is_a_404(self):
        response = self.client.get(
            "/api/workspace/browse", params={"path": str(self.root / "nope")}
        )
        self.assertEqual(response.status_code, 404)


class MetaRouteTests(ApiTestCase):
    def test_tools_lists_the_enabled_ones_and_why_the_rest_are_not(self):
        body = self.client.get("/api/tools").json()
        names = {tool["name"] for tool in body["tools"]}
        self.assertIn("calculate", names)
        self.assertIn("read_text_file", names)

        disabled = {item["category"]: item["reason"] for item in body["disabled"]}
        # The reason has to name the flag, or there is no way to find it.
        self.assertIn("AGENT_ENABLE_FILE_WRITES", disabled["file writes"])
        self.assertIn("terminal", disabled)

    def test_write_tools_are_absent_by_default(self):
        """Off by default is the whole security posture; assert it directly."""
        names = {t["name"] for t in self.client.get("/api/tools").json()["tools"]}
        self.assertNotIn("write_text_file", names)
        self.assertNotIn("run_command", names)
        self.assertNotIn("run_python", names)

    def test_health_reports_an_idle_server(self):
        body = self.client.get("/api/health").json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["busy"])
        self.assertEqual(body["queue_depth"], 0)


if __name__ == "__main__":
    unittest.main()
