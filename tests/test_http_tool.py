"""HTTP tool tests.

The allowlist, scheme and redirect checks run against a stubbed session. The
few that touch the network hit loopback only and skip when nothing answers.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import requests

from config import Config
from tools.base import ToolRegistry
from tools.http_tool import HttpClient, HttpToolError, build_http_tool
from tools.registry import build_default_registry


class FakeRaw:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, size, decode_content=True):
        return self._payload[:size]


class FakeResponse:
    def __init__(self, status=200, headers=None, payload=b"", is_redirect=False):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/plain"}
        self.raw = FakeRaw(payload)
        self.is_redirect = is_redirect

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self):
        self.trust_env = True
        self.calls: list[dict] = []
        self.response = FakeResponse(payload=b"pong")
        self.error: Exception | None = None

    def request(self, method, url, data=None, headers=None, timeout=None,
                allow_redirects=None, stream=None):
        self.calls.append(
            {"method": method, "url": url, "data": data, "headers": headers,
             "allow_redirects": allow_redirects}
        )
        if self.error:
            raise self.error
        return self.response


def client(**kwargs) -> HttpClient:
    kwargs.setdefault("allowed_hosts", ("127.0.0.1", "localhost"))
    kwargs.setdefault("timeout", 5.0)
    kwargs.setdefault("max_bytes", 1000)
    kwargs.setdefault("allow_writes", False)
    c = HttpClient(**kwargs)
    c._session = FakeSession()
    return c


class HostAllowlistTests(unittest.TestCase):
    def test_loopback_allowed(self):
        c = client()
        self.assertEqual(c.check_url("http://127.0.0.1:8080/health"), "127.0.0.1")
        self.assertEqual(c.check_url("http://localhost:8080/x"), "localhost")

    def test_external_host_refused(self):
        c = client()
        with self.assertRaises(HttpToolError) as ctx:
            c.check_url("https://example.com/data")
        self.assertIn("not an allowed host", str(ctx.exception))

    def test_host_match_is_case_insensitive(self):
        self.assertEqual(client().check_url("http://LOCALHOST:80/"), "localhost")

    def test_widened_allowlist_permits_the_named_host(self):
        c = client(allowed_hosts=("127.0.0.1", "example.com"))
        self.assertEqual(c.check_url("https://example.com/x"), "example.com")

    def test_lookalike_host_refused(self):
        # localhost.evil.com must not pass because it starts with localhost.
        c = client()
        with self.assertRaises(HttpToolError):
            c.check_url("http://localhost.evil.com/x")

    def test_no_host_refused(self):
        with self.assertRaises(HttpToolError):
            client().check_url("http:///nohost")


class SchemeTests(unittest.TestCase):
    def test_file_scheme_refused(self):
        # Would otherwise be an unrestricted file reader ignoring the jail.
        with self.assertRaises(HttpToolError) as ctx:
            client().check_url("file:///C:/Windows/win.ini")
        self.assertIn("only http", str(ctx.exception).lower())

    def test_other_schemes_refused(self):
        for url in ("ftp://127.0.0.1/x", "gopher://127.0.0.1/", "data:text/plain,hi"):
            with self.assertRaises(HttpToolError, msg=url):
                client().check_url(url)

    def test_missing_scheme_refused(self):
        with self.assertRaises(HttpToolError):
            client().check_url("127.0.0.1:8080/health")

    def test_credentials_in_url_refused(self):
        with self.assertRaises(HttpToolError) as ctx:
            client().check_url("http://user:secret@127.0.0.1/x")
        self.assertIn("Credentials", str(ctx.exception))

    def test_overlong_url_refused(self):
        with self.assertRaises(HttpToolError):
            client().check_url("http://127.0.0.1/" + "a" * 2100)

    def test_url_with_newline_refused(self):
        # Seen for real: a small model leaked its tool-call scaffolding into
        # the url argument. urlparse keeps the junk in the path and the request
        # goes out mangled, so it is refused with an instruction instead.
        bad = "http://127.0.0.1:8081/health\n<parameter=method>\nGET"
        with self.assertRaises(HttpToolError) as ctx:
            client().check_url(bad)
        self.assertIn("whitespace or control characters", str(ctx.exception))

    def test_url_with_space_refused(self):
        with self.assertRaises(HttpToolError):
            client().check_url("http://127.0.0.1:8080/a path")

    def test_url_with_carriage_return_refused(self):
        with self.assertRaises(HttpToolError):
            client().check_url("http://127.0.0.1:8080/x\r\nHost: evil")

    def test_surrounding_whitespace_is_fine(self):
        # Trimmed, not rejected - only whitespace *inside* is a problem.
        self.assertEqual(
            client().check_url("  http://127.0.0.1:8080/health  "), "127.0.0.1"
        )


class MethodTests(unittest.TestCase):
    def test_read_methods_allowed_by_default(self):
        c = client()
        self.assertEqual(c.check_method("get"), "GET")
        self.assertEqual(c.check_method("HEAD"), "HEAD")

    def test_write_methods_refused_by_default(self):
        c = client()
        for verb in ("POST", "PUT", "PATCH", "DELETE"):
            with self.assertRaises(HttpToolError, msg=verb) as ctx:
                c.check_method(verb)
            self.assertIn("AGENT_HTTP_ALLOW_WRITES", str(ctx.exception))

    def test_write_methods_allowed_when_enabled(self):
        c = client(allow_writes=True)
        self.assertEqual(c.check_method("post"), "POST")

    def test_unknown_method_refused(self):
        with self.assertRaises(HttpToolError):
            client().check_method("TRACE")


class RedirectTests(unittest.TestCase):
    """A redirect must never be followed out of the allowlist."""

    def test_redirects_are_disabled_on_the_request(self):
        c = client()
        c.request("http://127.0.0.1:8080/x")
        self.assertFalse(c._session.calls[0]["allow_redirects"])

    def test_redirect_is_reported_not_followed(self):
        c = client()
        c._session.response = FakeResponse(
            status=302,
            headers={"Location": "https://evil.example.com/steal",
                     "Content-Type": "text/html"},
            is_redirect=True,
        )
        result = c.request("http://127.0.0.1:8080/x")

        self.assertEqual(result["status"], 302)
        self.assertEqual(result["redirect_to"], "https://evil.example.com/steal")
        self.assertIn("not followed", result["body"])
        # Only the original request was made.
        self.assertEqual(len(c._session.calls), 1)


class ResponseTests(unittest.TestCase):
    def test_body_is_returned(self):
        result = client().request("http://127.0.0.1:8080/health")
        self.assertTrue(result["success"])
        self.assertEqual(result["body"], "pong")
        self.assertEqual(result["status"], 200)

    def test_non_2xx_is_reported_not_raised(self):
        c = client()
        c._session.response = FakeResponse(status=404, payload=b"nope")
        result = c.request("http://127.0.0.1:8080/missing")
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], 404)

    def test_large_body_is_truncated(self):
        c = client(max_bytes=10)
        c._session.response = FakeResponse(payload=b"x" * 500)
        result = c.request("http://127.0.0.1:8080/big")
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["body"]), 10)

    def test_binary_body_is_described_not_dumped(self):
        c = client()
        c._session.response = FakeResponse(
            headers={"Content-Type": "image/png"}, payload=b"\x89PNG\x00\x01\x02"
        )
        result = c.request("http://127.0.0.1:8080/img")
        self.assertIn("not text", result["body"])
        self.assertIn("image/png", result["body"])

    def test_json_content_type_is_treated_as_text(self):
        c = client()
        c._session.response = FakeResponse(
            headers={"Content-Type": "application/json"}, payload=b'{"ok":true}'
        )
        self.assertEqual(c.request("http://127.0.0.1:8080/j")["body"], '{"ok":true}')

    def test_connection_error_is_explained(self):
        c = client()
        c._session.error = requests.ConnectionError("refused")
        with self.assertRaises(HttpToolError) as ctx:
            c.request("http://127.0.0.1:9/x")
        self.assertIn("Could not connect", str(ctx.exception))

    def test_timeout_is_explained(self):
        c = client()
        c._session.error = requests.Timeout("slow")
        with self.assertRaises(HttpToolError) as ctx:
            c.request("http://127.0.0.1:8080/x")
        self.assertIn("No response within", str(ctx.exception))


class BodyTests(unittest.TestCase):
    def test_json_body_gets_a_json_content_type(self):
        c = client(allow_writes=True)
        c.request("http://127.0.0.1:8080/x", method="POST", body='{"a":1}')
        self.assertEqual(
            c._session.calls[0]["headers"]["Content-Type"], "application/json"
        )

    def test_plain_body_gets_text_content_type(self):
        c = client(allow_writes=True)
        c.request("http://127.0.0.1:8080/x", method="POST", body="hello")
        self.assertEqual(
            c._session.calls[0]["headers"]["Content-Type"], "text/plain"
        )

    def test_explicit_content_type_wins(self):
        c = client(allow_writes=True)
        c.request("http://127.0.0.1:8080/x", method="POST", body="a=1",
                  content_type="application/x-www-form-urlencoded")
        self.assertEqual(
            c._session.calls[0]["headers"]["Content-Type"],
            "application/x-www-form-urlencoded",
        )

    def test_body_is_encoded_as_utf8(self):
        c = client(allow_writes=True)
        c.request("http://127.0.0.1:8080/x", method="POST", body="café")
        self.assertEqual(c._session.calls[0]["data"], "café".encode("utf-8"))


class EnvironmentTests(unittest.TestCase):
    def test_session_ignores_ambient_credentials(self):
        # trust_env off means no proxies and no .netrc.
        real = HttpClient(allowed_hosts=("127.0.0.1",), timeout=1, max_bytes=10)
        self.assertFalse(real._session.trust_env)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_unless_enabled(self):
        registry, disabled = build_default_registry(Config(workspace=self.tmp))
        self.assertNotIn("http_request", registry.names())
        self.assertIn("http", {item.category for item in disabled})

    def test_registered_when_enabled(self):
        registry, disabled = build_default_registry(
            Config(workspace=self.tmp, http_tool_enabled=True)
        )
        self.assertIn("http_request", registry.names())
        self.assertNotIn("http", {item.category for item in disabled})

    def test_description_lists_hosts_and_methods(self):
        tool = build_http_tool(
            allowed_hosts=("127.0.0.1",), timeout=5, max_bytes=100,
            allow_writes=False,
        )
        self.assertIn("127.0.0.1", tool.description)
        self.assertIn("GET", tool.description)
        self.assertNotIn("DELETE", tool.description)

    def test_refusal_through_the_registry_is_structured(self):
        tool = build_http_tool(
            allowed_hosts=("127.0.0.1",), timeout=5, max_bytes=100,
            allow_writes=False,
        )
        registry = ToolRegistry([tool])
        result = registry.execute("http_request", {"url": "https://example.com"})
        self.assertFalse(result.ok)
        self.assertIn("not an allowed host", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
