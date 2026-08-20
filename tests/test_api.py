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
import unittest
from pathlib import Path
from typing import Any, Iterable

from fastapi.testclient import TestClient

from api.main import create_app
from api.runtime import Runtime
from config import Config
from tests.fake_client import tool_call_message
from tests.test_manager import ManagerHarness


def write_registry(tmp: Path) -> Path:
    """A two-model registry with a router, written into a temp directory."""
    exe = tmp / "llama-server.exe"
    exe.write_text("x", encoding="utf-8")
    for name in ("fast.gguf", "big.gguf"):
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

    def chat_stream(self, messages, *, tools=None, on_token=None, on_reasoning=None):
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
            def chat_stream(self, messages, *, tools=None, on_token=None, on_reasoning=None):
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
