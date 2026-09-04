"""The HTTP tool's three tiers, and the redirects it now follows.

The tool used to refuse everything off its host allowlist and every write.
Both are now shown to a person instead, on the same pattern as the terminal.
The refusals that remain are the ones a prompt could not sensibly replace:
`file://` is not a network request at all, and credentials in a url are not
something anyone should be asked to eyeball.

Redirects are the other half. They were never followed, which made
`http://host/x` -> `http://host/x/` on the *same allowed host* a dead end -
friction protecting nothing, since the second url passes exactly the check the
first one did.
"""

from __future__ import annotations

import http.server
import threading
import unittest

from tools.http_tool import HttpClient, HttpToolError


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves a few fixed paths, including two kinds of redirect."""

    def _send(self, status: int, body: bytes = b"", **headers):
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/end":
            self._send(200, b"arrived")
        elif self.path == "/here":
            # A bare path, the common shape - resolved against the current url.
            self._send(302, Location="/end")
        elif self.path == "/away":
            self._send(302, Location="https://elsewhere.invalid/x")
        elif self.path == "/loop":
            self._send(302, Location="/loop")
        else:
            self._send(404, b"no")

    def do_POST(self):
        self._send(200, b"posted")

    def log_message(self, *args):
        pass


class Server:
    def __enter__(self):
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        return False


def client(**kwargs) -> HttpClient:
    kwargs.setdefault("allowed_hosts", ("127.0.0.1", "localhost"))
    kwargs.setdefault("timeout", 5)
    return HttpClient(**kwargs)


class TierTests(unittest.TestCase):
    def gate(self, url: str, method: str = "GET", **kwargs) -> str:
        try:
            verdict = client(**kwargs).plan(url, method)
            return "approve" if verdict.needs_approval else "free"
        except HttpToolError:
            return "refuse"

    def test_reading_an_allowed_host_is_free(self):
        self.assertEqual(self.gate("http://127.0.0.1:8080/health"), "free")
        self.assertEqual(self.gate("http://localhost:8000/x"), "free")

    def test_a_host_off_the_list_asks(self):
        self.assertEqual(self.gate("https://example.com/"), "approve")

    def test_a_write_asks_even_on_an_allowed_host(self):
        self.assertEqual(self.gate("http://127.0.0.1/x", "POST"), "approve")
        self.assertEqual(self.gate("http://127.0.0.1/x", "DELETE"), "approve")

    def test_allow_writes_turns_that_prompt_off(self):
        """The flag now means 'do not ask', not 'do not permit'."""
        self.assertEqual(
            self.gate("http://127.0.0.1/x", "POST", allow_writes=True), "free"
        )
        # But it says nothing about leaving the allowlist.
        self.assertEqual(
            self.gate("https://example.com/", "POST", allow_writes=True), "approve"
        )

    def test_the_reason_names_both_problems_when_there_are_two(self):
        verdict = client().plan("https://api.example.com/v1", "POST")
        self.assertIn("POST", verdict.reason)
        self.assertIn("api.example.com", verdict.reason)
        self.assertIn("not on the allowed list", verdict.reason)

    def test_what_stays_refused(self):
        """A prompt could not sensibly stand in for any of these."""
        self.assertEqual(self.gate("file:///etc/passwd"), "refuse")
        self.assertEqual(self.gate("ftp://127.0.0.1/x"), "refuse")
        self.assertEqual(self.gate("http://user:pass@127.0.0.1/"), "refuse")
        self.assertEqual(self.gate("http://127.0.0.1/ with junk"), "refuse")
        self.assertEqual(self.gate("http://127.0.0.1/", "TRACE"), "refuse")


class GateTests(unittest.TestCase):
    def test_with_nobody_to_ask_it_refuses(self):
        with self.assertRaises(HttpToolError) as caught:
            client().request("https://example.com/")
        self.assertIn("nobody to ask", str(caught.exception))

    def test_declining_sends_nothing(self):
        asked = []

        def decline(what, why):
            asked.append((what, why))
            return False

        with Server() as server:
            result = client(approve=decline).request(f"{server.base}/end", "POST")

        self.assertFalse(result["success"])
        self.assertTrue(result["declined"])
        self.assertEqual(len(asked), 1)
        self.assertIn("POST", asked[0][0])

    def test_approving_sends_it(self):
        with Server() as server:
            result = client(approve=lambda w, y: True).request(
                f"{server.base}/end", "POST"
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], 200)

    def test_a_free_request_never_asks(self):
        asked = []
        with Server() as server:
            client(approve=lambda w, y: asked.append(w) or True).request(
                f"{server.base}/end"
            )
        self.assertEqual(asked, [])


class RedirectTests(unittest.TestCase):
    def test_a_redirect_within_the_allowed_hosts_is_followed(self):
        with Server() as server:
            result = client().request(f"{server.base}/here")

        self.assertEqual(result["status"], 200)
        self.assertEqual(result["body"], "arrived")
        self.assertTrue(result["url"].endswith("/end"))
        self.assertEqual(len(result["redirects"]), 1)

    def test_a_redirect_off_the_list_is_reported_not_followed(self):
        """The approval was for the url that was asked about, not its target."""
        with Server() as server:
            result = client(approve=lambda w, y: True).request(f"{server.base}/away")

        self.assertEqual(result["redirect_to"], "https://elsewhere.invalid/x")
        self.assertIn("not on the allowed list", result["body"])

    def test_a_redirect_loop_ends(self):
        with Server() as server:
            with self.assertRaises(HttpToolError) as caught:
                client().request(f"{server.base}/loop")
        self.assertIn("redirects", str(caught.exception))

    def test_a_plain_response_reports_no_redirects(self):
        with Server() as server:
            result = client().request(f"{server.base}/end")
        self.assertNotIn("redirects", result)


if __name__ == "__main__":
    unittest.main()
