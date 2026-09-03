"""Loopback HTTP must ignore the proxy environment.

The bug this guards against was invisible to the rest of the suite, because
every other test replaces `_healthy` or the HTTP client with a fake. Nothing
here is faked: a real socket, the real sessions, and an environment set up the
way this machine was actually found - a system-wide HTTP_PROXY and no NO_PROXY.

In that environment `requests` sends a request for 127.0.0.1 to the proxy,
which cannot reach this machine's loopback, and the failure comes back as a
ProxyError in tens of milliseconds. The manager read that as "not healthy",
600 seconds at a time, for a llama-server that was loaded and fine. The first
test below reproduces that on purpose; the rest show the sessions that talk to
local servers no longer take the bait.
"""

from __future__ import annotations

import http.server
import os
import tempfile
import threading
import unittest
from pathlib import Path

import requests

from config import Config
from models.manager import ModelManager
from models.qwen import QwenClient
from tests.test_manager import write_registry
from tools.filesystem import WorkspaceFiles
from tools.ocr_tool import OcrClient

# A port with nothing on it. A refused connection fails at once, which keeps
# the hostile case fast and deterministic rather than waiting on a timeout.
DEAD_PROXY = "http://127.0.0.1:9"


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class LoopbackServer:
    """A local HTTP server that answers 200 to everything."""

    def __enter__(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _OkHandler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False


class HostileProxyEnvironment(unittest.TestCase):
    """The environment this machine was found in: a proxy, and no NO_PROXY."""

    def setUp(self):
        self._saved = dict(os.environ)
        for key in list(os.environ):
            if key.upper() in {"NO_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
                del os.environ[key]
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            os.environ[key] = DEAD_PROXY

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)


class TheEnvironmentIsHostileTests(HostileProxyEnvironment):
    def test_a_default_session_cannot_reach_loopback(self):
        """The control. If this ever passes, the other tests prove nothing."""
        with LoopbackServer() as server:
            with self.assertRaises(requests.RequestException):
                requests.Session().get(f"{server.url}/health", timeout=2.0)


class ManagerHealthTests(HostileProxyEnvironment):
    def test_the_real_health_check_reaches_a_local_server(self):
        with tempfile.TemporaryDirectory() as tmp, LoopbackServer() as server:
            root = Path(tmp).resolve()
            manager = ModelManager(write_registry(root), preferences_dir=root)
            self.assertTrue(manager._healthy(server.port))


class QwenClientTests(HostileProxyEnvironment):
    def test_health_reaches_a_local_server(self):
        with LoopbackServer() as server:
            client = QwenClient(Config(qwen_url=server.url))
            self.assertTrue(client.health())

    def test_loopback_ignores_the_environment_and_a_lan_host_does_not(self):
        local = QwenClient(Config(qwen_url="http://127.0.0.1:8084"))
        named = QwenClient(Config(qwen_url="http://localhost:8084"))
        ipv6 = QwenClient(Config(qwen_url="http://[::1]:8084"))
        lan = QwenClient(Config(qwen_url="http://192.168.1.50:8080"))

        self.assertFalse(local._session.trust_env)
        self.assertFalse(named._session.trust_env)
        self.assertFalse(ipv6._session.trust_env)
        # Another machine may genuinely be behind the proxy; leave that alone.
        self.assertTrue(lan._session.trust_env)


class OcrClientTests(HostileProxyEnvironment):
    def test_health_reaches_a_local_server(self):
        with tempfile.TemporaryDirectory() as tmp, LoopbackServer() as server:
            client = OcrClient(Config(ocr_url=server.url), WorkspaceFiles(Path(tmp)))
            self.assertTrue(client.health())

    def test_loopback_ignores_the_environment_and_a_lan_host_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = WorkspaceFiles(Path(tmp))
            local = OcrClient(Config(ocr_url="http://127.0.0.1:8081"), files)
            lan = OcrClient(Config(ocr_url="http://10.0.0.7:8081"), files)
            self.assertFalse(local._session.trust_env)
            self.assertTrue(lan._session.trust_env)


if __name__ == "__main__":
    unittest.main()
