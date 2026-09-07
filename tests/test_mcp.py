"""The MCP client, against a real server subprocess.

Nothing is mocked. `fake_mcp_server.py` is started the way any other server
would be, speaks the real protocol over stdin and stdout, and is stopped
afterwards - because the parts of this worth testing are exactly the parts a
mock would paper over: the handshake, matching a reply to its id, a server
that logs before answering, and one that dies.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from tools.mcp_client import (
    McpConfigError,
    McpConnection,
    McpHttpConnection,
    McpError,
    McpManager,
    ServerSpec,
    check_name,
    check_url,
    edit_config,
    load_servers,
    open_connection,
    resolve_env,
)
from tests import fake_mcp_http_server
from tools import mcp_catalog

SERVER = str(Path(__file__).resolve().parent / "fake_mcp_server.py")


def spec(name: str = "fake", *, noisy: bool = False, **kwargs) -> ServerSpec:
    args = [SERVER] + (["--noisy"] if noisy else [])
    return ServerSpec(name=name, command=sys.executable, args=tuple(args), **kwargs)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "mcp.json"

    def write(self, data) -> Path:
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return self.path

    def test_the_standard_shape_is_read(self):
        """A config copied from another MCP client should just work."""
        servers = load_servers(
            self.write(
                {
                    "mcpServers": {
                        "files": {"command": "npx", "args": ["-y", "server"]}
                    }
                }
            )
        )
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0].name, "files")
        self.assertEqual(servers[0].args, ("-y", "server"))
        self.assertEqual(servers[0].category, "mcp:files")

    def test_trusted_and_enabled_are_read(self):
        servers = load_servers(
            self.write(
                {
                    "mcpServers": {
                        "a": {"command": "x", "trusted": True},
                        "b": {"command": "x", "enabled": False},
                    }
                }
            )
        )
        by_name = {s.name: s for s in servers}
        self.assertTrue(by_name["a"].trusted)
        self.assertFalse(by_name["b"].enabled)

    def test_a_missing_file_is_no_servers_not_a_crash(self):
        self.assertEqual(load_servers(Path("nowhere.json")), [])

    def test_broken_json_is_no_servers_not_a_crash(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(load_servers(self.path), [])

    def test_an_entry_with_no_command_is_skipped(self):
        servers = load_servers(
            self.write({"mcpServers": {"bad": {"args": ["x"]}, "ok": {"command": "y"}}})
        )
        self.assertEqual([s.name for s in servers], ["ok"])


class ConnectionTests(unittest.TestCase):
    def connect(self, **kwargs) -> McpConnection:
        connection = McpConnection(spec(**kwargs))
        self.addCleanup(connection.stop)
        return connection

    def test_it_handshakes_and_lists(self):
        tools = self.connect().list_tools()
        self.assertEqual(
            sorted(t["name"] for t in tools), ["echo", "explode", "wipe"]
        )

    def test_a_call_returns_the_text_content(self):
        connection = self.connect()
        result = connection.call("echo", {"text": "hello"})
        self.assertEqual(result["content"][0]["text"], "hello")

    def test_a_notification_before_the_reply_is_skipped(self):
        """A client that read one line and called it the answer breaks here."""
        connection = self.connect(noisy=True)
        result = connection.call("echo", {"text": "still works"})
        self.assertEqual(result["content"][0]["text"], "still works")

    def test_a_protocol_error_is_raised_with_the_servers_message(self):
        connection = self.connect()
        with self.assertRaises(McpError) as caught:
            connection.call("nonexistent", {})
        self.assertIn("no tool", str(caught.exception))

    def test_a_command_that_does_not_exist_says_so(self):
        connection = McpConnection(
            ServerSpec(name="ghost", command="definitely-not-a-real-program-xyz")
        )
        self.addCleanup(connection.stop)
        with self.assertRaises(McpError) as caught:
            connection.start()
        self.assertIn("Could not start", str(caught.exception))

    def test_stopping_is_idempotent(self):
        connection = self.connect()
        connection.start()
        self.assertTrue(connection.alive)
        connection.stop()
        connection.stop()
        self.assertFalse(connection.alive)


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = self.root / "mcp.json"
        self.cache = self.root / "cache.json"
        self.config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "fake": {
                            "command": sys.executable,
                            "args": [SERVER],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def manager(self, **kwargs) -> McpManager:
        made = McpManager(self.config, self.cache, **kwargs)
        self.addCleanup(made.stop_all)
        return made

    def test_no_tools_before_a_refresh(self):
        """The registry is built every turn; it must never start a server."""
        self.assertEqual(self.manager().tools(), [])

    def test_refresh_caches_what_the_server_offers(self):
        report = self.manager().refresh()

        self.assertEqual(report["servers"], {"fake": 3})
        self.assertEqual(report["errors"], {})
        self.assertTrue(self.cache.is_file())

    def test_refresh_leaves_nothing_running(self):
        """Refreshing exists to make later turns cheap, not to hold servers."""
        made = self.manager()
        made.refresh()
        self.assertEqual(made.stop_all(), [])

    def test_tools_come_from_the_cache_after_a_refresh(self):
        made = self.manager()
        made.refresh()

        names = sorted(t.name for t in made.tools())

        self.assertEqual(names, ["fake__echo", "fake__explode", "fake__wipe"])
        self.assertTrue(all(t.category == "mcp:fake" for t in made.tools()))

    def test_a_second_manager_reads_the_cache_without_refreshing(self):
        self.manager().refresh()
        self.assertEqual(len(self.manager().tools()), 3)

    def test_a_read_only_tool_does_not_ask(self):
        made = self.manager()
        made.refresh()
        asked = []
        echo = next(
            t
            for t in made.tools(approve=lambda w, y: asked.append(w) or True)
            if t.name == "fake__echo"
        )

        self.assertEqual(echo.run(text="hi")["output"], "hi")
        self.assertEqual(asked, [])

    def test_anything_not_declared_read_only_asks(self):
        made = self.manager()
        made.refresh()
        asked = []

        def decline(what, why):
            asked.append((what, why))
            return False

        wipe = next(
            t for t in made.tools(approve=decline) if t.name == "fake__wipe"
        )
        result = wipe.run()

        self.assertFalse(result["success"])
        self.assertTrue(result["declined"])
        self.assertIn("wipe", asked[0][0])
        self.assertIn("read-only", asked[0][1])

    def test_a_trusted_server_does_not_ask(self):
        self.config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "fake": {
                            "command": sys.executable,
                            "args": [SERVER],
                            "trusted": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        made = self.manager()
        made.refresh()
        asked = []
        wipe = next(
            t
            for t in made.tools(approve=lambda w, y: asked.append(w) or True)
            if t.name == "fake__wipe"
        )

        self.assertEqual(wipe.run()["output"], "wiped")
        self.assertEqual(asked, [])

    def test_with_nobody_to_ask_a_gated_tool_refuses(self):
        made = self.manager()
        made.refresh()
        wipe = next(t for t in made.tools() if t.name == "fake__wipe")
        with self.assertRaises(McpError) as caught:
            wipe.run()
        self.assertIn("nobody to ask", str(caught.exception))

    def test_a_tool_level_failure_is_reported_not_raised(self):
        """isError is the tool failing, not the protocol. The model needs both."""
        made = self.manager()
        made.refresh()
        explode = next(t for t in made.tools() if t.name == "fake__explode")

        result = explode.run()

        self.assertFalse(result["success"])
        self.assertIn("it went wrong", result["error"])

    def test_calling_starts_the_server_and_keeps_it(self):
        made = self.manager()
        made.refresh()
        echo = next(t for t in made.tools() if t.name == "fake__echo")

        echo.run(text="one")

        self.assertEqual(made.stop_all(), ["fake"])

    def test_an_idle_server_is_swept(self):
        made = self.manager(idle_timeout=0.01)
        made.refresh()
        next(t for t in made.tools() if t.name == "fake__echo").run(text="x")

        import time

        time.sleep(0.05)
        self.assertEqual(made.sweep(), ["fake"])

    def test_a_zero_idle_timeout_never_sweeps(self):
        made = self.manager(idle_timeout=0)
        made.refresh()
        next(t for t in made.tools() if t.name == "fake__echo").run(text="x")
        self.assertEqual(made.sweep(), [])

    def test_a_server_that_will_not_start_is_reported_not_fatal(self):
        self.config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "broken": {"command": "definitely-not-real-xyz"},
                        "fake": {"command": sys.executable, "args": [SERVER]},
                    }
                }
            ),
            encoding="utf-8",
        )
        report = self.manager().refresh()

        self.assertIn("broken", report["errors"])
        self.assertEqual(report["servers"].get("fake"), 3)


class ReloadTests(unittest.TestCase):
    """Adding a server to mcp.json while the API is running.

    The manager the API holds is built once at startup. Before `reload`, a
    server added to the file was invisible to it forever - and `refresh` would
    report success having asked the old set.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = self.root / "mcp.json"
        self.cache = self.root / "cache.json"
        self.write({"fake": self.entry()})

    def entry(self, **extra) -> dict:
        return {"command": sys.executable, "args": [SERVER], **extra}

    def write(self, servers: dict) -> None:
        self.config.write_text(
            json.dumps({"mcpServers": servers}), encoding="utf-8"
        )

    def manager(self) -> McpManager:
        made = McpManager(self.config, self.cache)
        self.addCleanup(made.stop_all)
        return made

    def test_a_server_added_to_the_file_is_seen_after_a_reload(self):
        made = self.manager()
        self.assertEqual([s.name for s in made.servers], ["fake"])

        self.write({"fake": self.entry(), "second": self.entry()})
        made.reload()

        self.assertEqual([s.name for s in made.servers], ["fake", "second"])

    def test_refreshing_picks_up_a_new_server_without_being_asked_to_reload(self):
        """The whole point: pressing Refresh after editing the file works."""
        made = self.manager()
        self.write({"fake": self.entry(), "second": self.entry()})

        report = made.refresh()

        self.assertEqual(report["servers"], {"fake": 3, "second": 3})

    def test_a_server_removed_from_the_file_goes(self):
        made = self.manager()
        self.write({})
        made.reload()
        self.assertEqual(made.servers, [])

    def test_a_running_server_that_was_removed_is_stopped(self):
        """It is a subprocess started from a command line that no longer exists."""
        made = self.manager()
        made.refresh()
        next(t for t in made.tools() if t.name == "fake__echo").run(text="x")

        self.write({})
        made.reload()

        self.assertEqual(made.stop_all(), [])

    def test_a_running_server_whose_command_changed_is_stopped(self):
        made = self.manager()
        made.refresh()
        next(t for t in made.tools() if t.name == "fake__echo").run(text="x")

        self.write({"fake": self.entry(trusted=True)})
        made.reload()

        self.assertEqual(made.stop_all(), [])

    def test_a_server_left_alone_keeps_its_connection(self):
        """Reloading is not an excuse to restart everything."""
        made = self.manager()
        made.refresh()
        next(t for t in made.tools() if t.name == "fake__echo").run(text="x")

        self.write({"fake": self.entry(), "second": self.entry()})
        made.reload()

        self.assertEqual(made.stop_all(), ["fake"])

    def test_disabling_a_server_in_the_file_removes_it(self):
        made = self.manager()
        self.write({"fake": self.entry(enabled=False)})
        made.reload()
        self.assertEqual(made.servers, [])

    def test_cached_tools_reads_the_file_and_starts_nothing(self):
        made = self.manager()
        self.assertEqual(made.cached_tools(), {})

        made.refresh()

        self.assertEqual(sorted(made.cached_tools()), ["fake"])
        self.assertEqual(made.stop_all(), [])


class EditConfigTests(unittest.TestCase):
    """Changing mcp.json without losing what else is in it.

    This file is git-ignored because it holds `env` blocks with API keys, which
    means there is no second copy. Every test here is about the same property:
    a write touches one entry and leaves the rest of the document alone.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "mcp.json"

    def write(self, document: dict) -> None:
        self.path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_it_creates_the_file_when_there_is_none(self):
        edit_config(self.path, lambda s: s.__setitem__("a", {"command": "x"}))
        self.assertEqual(self.read()["mcpServers"]["a"]["command"], "x")

    def test_another_servers_api_key_survives(self):
        """The reason this function exists rather than a rewrite from a model."""
        self.write(
            {
                "mcpServers": {
                    "github": {"command": "x", "env": {"TOKEN": "secret-value"}},
                }
            }
        )

        edit_config(self.path, lambda s: s.__setitem__("new", {"command": "y"}))

        self.assertEqual(
            self.read()["mcpServers"]["github"]["env"], {"TOKEN": "secret-value"}
        )

    def test_keys_this_code_knows_nothing_about_survive(self):
        """Another client's settings, or a future version of this one's."""
        self.write(
            {
                "mcpServers": {"a": {"command": "x", "somethingElse": 7}},
                "globalShortcut": "cmd+m",
            }
        )

        edit_config(self.path, lambda s: s.__setitem__("b", {"command": "y"}))

        after = self.read()
        self.assertEqual(after["globalShortcut"], "cmd+m")
        self.assertEqual(after["mcpServers"]["a"]["somethingElse"], 7)

    def test_removing_one_leaves_the_others(self):
        self.write({"mcpServers": {"a": {"command": "x"}, "b": {"command": "y"}}})

        edit_config(self.path, lambda s: s.pop("a", None))

        self.assertEqual(list(self.read()["mcpServers"]), ["b"])

    def test_broken_json_is_refused_rather_than_overwritten(self):
        """Overwriting it would destroy whatever keys it holds, uncopied."""
        self.path.write_text('{"mcpServers": {broken', encoding="utf-8")

        with self.assertRaises(McpConfigError) as caught:
            edit_config(self.path, lambda s: s.__setitem__("a", {"command": "x"}))

        self.assertIn("by hand", str(caught.exception))
        # And it really is untouched.
        self.assertIn("broken", self.path.read_text(encoding="utf-8"))

    def test_a_json_document_that_is_not_an_object_is_refused(self):
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(McpConfigError):
            edit_config(self.path, lambda s: s.__setitem__("a", {"command": "x"}))

    def test_a_missing_servers_key_is_created_not_fatal(self):
        self.write({"otherThing": True})
        edit_config(self.path, lambda s: s.__setitem__("a", {"command": "x"}))
        after = self.read()
        self.assertTrue(after["otherThing"])
        self.assertIn("a", after["mcpServers"])

    def test_no_temporary_file_is_left_behind(self):
        edit_config(self.path, lambda s: s.__setitem__("a", {"command": "x"}))
        leftovers = [p.name for p in self.path.parent.iterdir()]
        self.assertEqual(leftovers, ["mcp.json"])

    def test_what_it_writes_reads_back_as_servers(self):
        """The round trip, since load_servers is what actually consumes this."""
        edit_config(
            self.path,
            lambda s: s.__setitem__("a", {"command": "x", "args": ["--flag"]}),
        )
        specs = load_servers(self.path)
        self.assertEqual([s.name for s in specs], ["a"])
        self.assertEqual(specs[0].args, ("--flag",))


class NameTests(unittest.TestCase):
    def test_a_plain_name_passes(self):
        self.assertEqual(check_name("  Files  "), "files")

    def test_names_that_would_break_a_tool_prefix_are_refused(self):
        for bad in ("", "has space", "Slash/es", "-leading", "a" * 60, "dots.here"):
            with self.assertRaises(McpConfigError, msg=bad):
                check_name(bad)

    def test_hyphens_and_underscores_are_fine(self):
        self.assertEqual(check_name("my_server-2"), "my_server-2")


class CatalogTests(unittest.TestCase):
    """The offered servers, checked for the mistakes a table invites."""

    def test_the_package_named_matches_the_one_actually_run(self):
        """Caught a real one: the args said `server-sequentialthinking`,
        which does not exist on npm, while the label said the same thing and
        looked right. The label and the argument have to agree, and the
        argument is what runs."""
        for entry in mcp_catalog.CATALOG:
            package = entry.package.split(" ")[0]
            self.assertTrue(package, entry.name)
            self.assertTrue(
                any(package == arg for arg in entry.args),
                f"{entry.name}: package {package!r} is not in args {entry.args}",
            )

    def test_names_are_usable_as_a_tool_prefix(self):
        for entry in mcp_catalog.CATALOG:
            self.assertEqual(check_name(entry.name), entry.name)

    def test_every_entry_says_what_runtime_it_needs(self):
        for entry in mcp_catalog.CATALOG:
            self.assertIn(entry.runtime, ("node", "python"), entry.name)

    def test_a_uvx_entry_is_python_and_an_npx_entry_is_node(self):
        for entry in mcp_catalog.CATALOG:
            expected = "python" if entry.command == "uvx" else "node"
            self.assertEqual(entry.runtime, expected, entry.name)

    def test_anything_reaching_a_service_asks_for_a_credential(self):
        """A service server with no `needs` starts and then fails opaquely."""
        for entry in mcp_catalog.CATALOG:
            if entry.unmaintained:
                self.assertTrue(entry.needs, entry.name)

    def test_every_credential_says_where_to_get_one(self):
        for entry in mcp_catalog.CATALOG:
            for need in entry.needs:
                self.assertTrue(need.label, f"{entry.name}/{need.variable}")
                self.assertTrue(need.hint, f"{entry.name}/{need.variable}")

    def test_resolving_fills_in_the_workspace(self):
        entry = mcp_catalog.entry("git")
        workspace = Path("/somewhere/else")
        resolved = entry.resolve(workspace)
        # str(), not the literal: a Windows path renders with backslashes.
        self.assertIn(str(workspace), resolved["args"])
        self.assertTrue(all("{workspace}" not in a for a in resolved["args"]))

    def test_lookup_is_case_insensitive_and_trims(self):
        self.assertIsNotNone(mcp_catalog.entry("  GitHub "))
        self.assertIsNone(mcp_catalog.entry("nothing-like-this"))


class EnvReferenceTests(unittest.TestCase):
    """`${VAR}` values, so a secret need not be written to mcp.json at all."""

    def test_a_plain_value_is_passed_through(self):
        self.assertEqual(resolve_env({"K": "literal"}), {"K": "literal"})

    def test_a_reference_is_read_from_the_environment(self):
        with mock.patch.dict(os.environ, {"SOME_TOKEN": "from-the-shell"}):
            self.assertEqual(
                resolve_env({"K": "${SOME_TOKEN}"}), {"K": "from-the-shell"}
            )

    def test_surrounding_whitespace_is_tolerated(self):
        with mock.patch.dict(os.environ, {"SOME_TOKEN": "v"}):
            self.assertEqual(resolve_env({"K": "  ${SOME_TOKEN}  "}), {"K": "v"})

    def test_an_unset_reference_becomes_empty_not_the_literal(self):
        """A server handed the literal `${X}` sends it to its API as a token."""
        os.environ.pop("DEFINITELY_NOT_SET_XYZ", None)
        self.assertEqual(resolve_env({"K": "${DEFINITELY_NOT_SET_XYZ}"}), {"K": ""})

    def test_only_a_whole_value_is_a_reference(self):
        """No half-expansion: it is a source of silent mistakes."""
        with mock.patch.dict(os.environ, {"T": "secret"}):
            self.assertEqual(
                resolve_env({"K": "Bearer ${T}"}), {"K": "Bearer ${T}"}
            )


class HttpTransportTests(unittest.TestCase):
    """Streamable HTTP, against a real server on a real socket."""

    def serve(self, **kwargs):
        handle = fake_mcp_http_server.serve(**kwargs)
        self.addCleanup(handle.stop)
        return handle

    def connect(self, handle, **spec_kwargs) -> McpHttpConnection:
        connection = McpHttpConnection(
            ServerSpec(name="remote", url=handle.url, **spec_kwargs)
        )
        self.addCleanup(connection.stop)
        return connection

    def test_it_handshakes_and_lists(self):
        tools = self.connect(self.serve()).list_tools()
        self.assertEqual(sorted(t["name"] for t in tools), ["echo", "wipe"])

    def test_a_call_returns_the_text_content(self):
        connection = self.connect(self.serve())
        result = connection.call("echo", {"text": "over http"})
        self.assertEqual(result["content"][0]["text"], "over http")

    def test_an_event_stream_answer_is_read(self):
        """The server chooses per response; both shapes have to work."""
        connection = self.connect(self.serve(stream=True))
        result = connection.call("echo", {"text": "streamed"})
        self.assertEqual(result["content"][0]["text"], "streamed")

    def test_a_notification_before_the_reply_is_skipped(self):
        """A client taking the first event as its answer breaks here."""
        connection = self.connect(self.serve(stream=True, noisy=True))
        result = connection.call("echo", {"text": "still works"})
        self.assertEqual(result["content"][0]["text"], "still works")

    def test_a_tool_error_is_raised_with_the_servers_message(self):
        connection = self.connect(self.serve())
        with self.assertRaises(McpError) as caught:
            connection.call("nonexistent", {})
        self.assertIn("no tool", str(caught.exception))

    def test_the_protocol_version_is_sent(self):
        handle = self.serve()
        self.connect(handle).list_tools()
        self.assertTrue(
            all("MCP-Protocol-Version" in h for h in handle.seen_headers)
        )

    def test_both_content_types_are_accepted(self):
        """Saying only one would make the server's choice fail half the time."""
        handle = self.serve()
        self.connect(handle).list_tools()
        accept = handle.seen_headers[0]["Accept"]
        self.assertIn("application/json", accept)
        self.assertIn("text/event-stream", accept)

    def test_configured_headers_are_sent(self):
        handle = self.serve(require_auth="Bearer example-token")
        connection = self.connect(
            handle, headers={"Authorization": "Bearer example-token"}
        )
        self.assertEqual(len(connection.list_tools()), 2)

    def test_a_header_can_come_from_the_environment(self):
        """So a token need not be written into mcp.json at all.

        Only a whole value expands, so the variable holds the entire header
        including the `Bearer` prefix. Written the other way round -
        `"Bearer ${VAR}"` - nothing is substituted, which the next test is
        here to pin down.
        """
        handle = self.serve(require_auth="Bearer from-the-shell")
        with mock.patch.dict(
            os.environ, {"MY_MCP_TOKEN": "Bearer from-the-shell"}
        ):
            connection = self.connect(
                handle, headers={"Authorization": "${MY_MCP_TOKEN}"}
            )
            self.assertEqual(len(connection.list_tools()), 2)
        # The value went out, not the name of it.
        sent = handle.seen_headers[0]["Authorization"]
        self.assertEqual(sent, "Bearer from-the-shell")

    def test_a_reference_inside_a_larger_header_is_not_expanded(self):
        """Half-substituted credentials fail confusingly; they fail plainly.

        `resolve_env` only expands a value that is entirely `${VAR}`. A header
        written as `Bearer ${VAR}` is sent literally, and the server refuses
        it - which is the error someone can act on, rather than a token that
        is silently half a token.
        """
        handle = self.serve(require_auth="Bearer from-the-shell")
        with mock.patch.dict(os.environ, {"MY_MCP_TOKEN": "from-the-shell"}):
            connection = self.connect(
                handle, headers={"Authorization": "Bearer ${MY_MCP_TOKEN}"}
            )
            with self.assertRaises(McpError):
                connection.list_tools()
        self.assertEqual(
            handle.seen_headers[0]["Authorization"], "Bearer ${MY_MCP_TOKEN}"
        )

    def test_a_missing_credential_says_what_to_do(self):
        connection = self.connect(self.serve(require_auth="Bearer needed"))
        with self.assertRaises(McpError) as caught:
            connection.list_tools()
        message = str(caught.exception)
        self.assertIn("credential", message)
        self.assertIn("headers", message)
        self.assertIn("OAuth", message)

    def test_a_session_id_is_kept_and_sent_back(self):
        handle = self.serve(sessions=True)
        self.connect(handle).list_tools()

        later = [h for h in handle.seen_headers if "Mcp-Session-Id" in h]
        self.assertTrue(later)
        self.assertEqual(later[0]["Mcp-Session-Id"], "test-session-1")

    def test_an_expired_session_is_dropped_rather_than_resent(self):
        """A 404 against a session we hold is the server saying it is gone.

        Holding on to it would make every later call fail the same way
        forever. The connection forgets it and marks itself not alive, so the
        next call handshakes again - which is what the message promises.
        """
        connection = self.connect(self.serve(sessions=True, expire_session=True))
        with self.assertRaises(McpError) as caught:
            connection.list_tools()

        self.assertIn("session", str(caught.exception))
        self.assertFalse(connection.alive)
        self.assertEqual(connection._session_id, "")

    def test_stopping_ends_the_session_on_the_server(self):
        handle = self.serve(sessions=True)
        connection = McpHttpConnection(ServerSpec(name="remote", url=handle.url))
        connection.list_tools()

        connection.stop()

        self.assertEqual(handle.deleted, ["test-session-1"])
        self.assertFalse(connection.alive)

    def test_an_http_error_is_reported_with_its_status(self):
        connection = self.connect(self.serve(status=500))
        with self.assertRaises(McpError) as caught:
            connection.list_tools()
        self.assertIn("500", str(caught.exception))

    def test_a_server_that_is_not_there_says_so(self):
        connection = McpHttpConnection(
            ServerSpec(name="ghost", url="http://127.0.0.1:9/mcp")
        )
        self.addCleanup(connection.stop)
        with self.assertRaises(McpError) as caught:
            connection.list_tools()
        self.assertIn("Could not reach", str(caught.exception))

    def test_stopping_is_idempotent(self):
        connection = self.connect(self.serve())
        connection.start()
        self.assertTrue(connection.alive)
        connection.stop()
        connection.stop()
        self.assertFalse(connection.alive)

    def test_it_can_be_started_again_after_stopping(self):
        """The sweeper stops idle connections; the next call must revive one."""
        connection = self.connect(self.serve())
        connection.list_tools()
        connection.stop()

        self.assertEqual(connection.call("echo", {"text": "again"})
                         ["content"][0]["text"], "again")


class UrlValidationTests(unittest.TestCase):
    def test_a_plain_url_passes(self):
        self.assertEqual(check_url("https://x.test/mcp", "n"), "https://x.test/mcp")

    def test_a_non_http_scheme_is_refused(self):
        for url in ("file:///etc/passwd", "ftp://x.test/", "ws://x.test/"):
            with self.assertRaises(McpError, msg=url):
                check_url(url, "n")

    def test_credentials_in_the_url_are_refused(self):
        """Nobody should be asked to eyeball a password in a url."""
        with self.assertRaises(McpError) as caught:
            check_url("https://user:secret@x.test/mcp", "n")
        self.assertIn("headers", str(caught.exception))

    def test_a_url_with_no_host_is_refused(self):
        with self.assertRaises(McpError):
            check_url("http:///mcp", "n")


class TransportChoiceTests(unittest.TestCase):
    """`url` against `command`, and what the config makes of each."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "mcp.json"

    def write(self, servers: dict) -> Path:
        self.path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
        return self.path

    def test_a_url_entry_is_read_as_remote(self):
        spec = load_servers(
            self.write({"remote": {"url": "https://x.test/mcp"}})
        )[0]
        self.assertTrue(spec.remote)
        self.assertEqual(spec.display, "https://x.test/mcp")

    def test_a_command_entry_is_read_as_local(self):
        spec = load_servers(self.write({"local": {"command": "npx", "args": ["a"]}}))[0]
        self.assertFalse(spec.remote)
        self.assertEqual(spec.display, "npx a")

    def test_headers_are_read(self):
        spec = load_servers(
            self.write(
                {"r": {"url": "https://x.test/mcp", "headers": {"A": "b"}}}
            )
        )[0]
        self.assertEqual(spec.headers, {"A": "b"})

    def test_an_entry_with_both_is_skipped_rather_than_guessed_at(self):
        servers = load_servers(
            self.write(
                {
                    "both": {"command": "npx", "url": "https://x.test/"},
                    "ok": {"command": "npx"},
                }
            )
        )
        self.assertEqual([s.name for s in servers], ["ok"])

    def test_an_entry_with_neither_is_skipped(self):
        servers = load_servers(self.write({"empty": {"args": ["x"]}, "ok": {"command": "y"}}))
        self.assertEqual([s.name for s in servers], ["ok"])

    def test_the_factory_picks_the_transport(self):
        local = open_connection(ServerSpec(name="a", command="x"))
        remote = open_connection(ServerSpec(name="b", url="https://x.test/mcp"))
        self.addCleanup(local.stop)
        self.addCleanup(remote.stop)
        self.assertIsInstance(local, McpConnection)
        self.assertIsInstance(remote, McpHttpConnection)


class HttpThroughTheManagerTests(unittest.TestCase):
    """A remote server has to behave like any other in the roster."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.handle = fake_mcp_http_server.serve()
        self.addCleanup(self.handle.stop)
        (self.root / "mcp.json").write_text(
            json.dumps({"mcpServers": {"remote": {"url": self.handle.url}}}),
            encoding="utf-8",
        )
        self.made = McpManager(self.root / "mcp.json", self.root / "cache.json")
        self.addCleanup(self.made.stop_all)

    def test_refresh_caches_what_it_offers(self):
        report = self.made.refresh()
        self.assertEqual(report["servers"], {"remote": 2})
        self.assertEqual(report["errors"], {})

    def test_its_tools_are_registered_like_any_other(self):
        self.made.refresh()
        names = sorted(t.name for t in self.made.tools())
        self.assertEqual(names, ["remote__echo", "remote__wipe"])
        self.assertTrue(all(t.category == "mcp:remote" for t in self.made.tools()))

    def test_read_only_still_runs_without_asking(self):
        self.made.refresh()
        asked = []
        echo = next(
            t
            for t in self.made.tools(approve=lambda w, y: asked.append(w) or True)
            if t.name == "remote__echo"
        )
        self.assertEqual(echo.run(text="hi")["output"], "hi")
        self.assertEqual(asked, [])

    def test_anything_else_still_asks(self):
        self.made.refresh()
        asked = []
        wipe = next(
            t
            for t in self.made.tools(approve=lambda w, y: asked.append(w) or False)
            if t.name == "remote__wipe"
        )
        result = wipe.run()
        self.assertFalse(result["success"])
        self.assertTrue(asked)

    def test_an_idle_remote_connection_is_swept(self):
        made = McpManager(
            self.root / "mcp.json", self.root / "cache.json", idle_timeout=0.01
        )
        self.addCleanup(made.stop_all)
        made.refresh()
        next(t for t in made.tools() if t.name == "remote__echo").run(text="x")

        time.sleep(0.05)

        self.assertEqual(made.sweep(), ["remote"])


if __name__ == "__main__":
    unittest.main()
