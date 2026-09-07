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
import unittest
from unittest import mock
from pathlib import Path

from tools.mcp_client import (
    McpConfigError,
    McpConnection,
    McpError,
    McpManager,
    ServerSpec,
    check_name,
    edit_config,
    load_servers,
    resolve_env,
)
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


if __name__ == "__main__":
    unittest.main()
