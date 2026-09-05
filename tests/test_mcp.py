"""The MCP client, against a real server subprocess.

Nothing is mocked. `fake_mcp_server.py` is started the way any other server
would be, speaks the real protocol over stdin and stdout, and is stopped
afterwards - because the parts of this worth testing are exactly the parts a
mock would paper over: the handshake, matching a reply to its id, a server
that logs before answering, and one that dies.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools.mcp_client import (
    McpConnection,
    McpError,
    McpManager,
    ServerSpec,
    load_servers,
)

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


if __name__ == "__main__":
    unittest.main()
