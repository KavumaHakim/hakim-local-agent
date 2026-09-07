"""A real MCP server over Streamable HTTP, for the tests to talk to.

The counterpart to `fake_mcp_server.py`, which does the same job over stdio.
Nothing is mocked: this binds a socket, speaks the real protocol, and is torn
down afterwards - because the parts worth testing are exactly the ones a mock
would paper over. A server answering with an event stream instead of JSON, one
that sends a log line before the reply, one that wants a session header back,
one that demands a credential: those are the shapes that break a client, and
they only appear if something real produces them.

Started with `serve()`, which returns a handle with `.url` and `.stop()`.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "echo",
        "description": "Returns what it was given.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "wipe",
        "description": "Claims no annotation, so it must be asked about.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Options:
    """How this instance should misbehave."""

    def __init__(
        self,
        *,
        stream: bool = False,
        noisy: bool = False,
        sessions: bool = False,
        require_auth: str = "",
        status: int = 0,
        expire_session: bool = False,
    ) -> None:
        # Answer with text/event-stream rather than application/json.
        self.stream = stream
        # Send a log notification before the reply. A client that reads one
        # message and calls it the answer breaks here.
        self.noisy = noisy
        # Hand out an Mcp-Session-Id and require it on later requests. It is
        # enforced rather than merely issued: a client that takes the header
        # and never sends it back would otherwise pass.
        self.sessions = sessions
        # Refuse unless this exact Authorization header arrives.
        self.require_auth = require_auth
        # Answer everything with this status instead, to test the reporting.
        self.status = status
        # Forget the session after the handshake, so the next request gets
        # the 404 a real server sends when a session has aged out.
        self.expire_session = expire_session


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Silence: the test output is the interesting thing, not the access log.
    def log_message(self, *args) -> None:  # noqa: D102
        pass

    @property
    def options(self) -> Options:
        return self.server.options  # type: ignore[attr-defined]

    def do_DELETE(self) -> None:  # noqa: N802
        self.server.deleted.append(self.headers.get("Mcp-Session-Id", ""))  # type: ignore[attr-defined]
        self._send(202, b"", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self.server.seen_headers.append(dict(self.headers))  # type: ignore[attr-defined]

        if self.options.status:
            self._send(self.options.status, b'{"error":"as asked"}')
            return

        if self.options.require_auth:
            if self.headers.get("Authorization") != self.options.require_auth:
                self._send(401, b'{"error":"unauthorized"}')
                return

        try:
            message = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._send(400, b'{"error":"not json"}')
            return

        # Sessions, the way a real server enforces them: everything after the
        # handshake must carry the header, and a session the server has
        # forgotten is a 404 rather than a 400. A client that stores the id
        # and never sends it back has to fail here, or the option is testing
        # nothing.
        if self.options.sessions and message.get("method") != "initialize":
            sent = self.headers.get("Mcp-Session-Id", "")
            if sent != self.server.session:  # type: ignore[attr-defined]
                self._send(404, b'{"error":"no such session"}')
                return

        # A notification has no id and gets no reply.
        if "id" not in message:
            self._send(202, b"", "text/plain")
            return

        reply = self._reply_to(message)
        session = ""
        if self.options.sessions and message.get("method") == "initialize":
            session = "test-session-1"
            # Handed to the client and immediately forgotten, when the point
            # is to watch the client meet an expired one.
            self.server.session = "" if self.options.expire_session else session  # type: ignore[attr-defined]

        if self.options.stream:
            body = b""
            if self.options.noisy:
                body += _event(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/message",
                        "params": {"level": "info", "data": "still working"},
                    }
                )
            body += _event(reply)
            self._send(200, body, "text/event-stream", session)
        else:
            self._send(200, json.dumps(reply).encode("utf-8"), session=session)

    def _reply_to(self, message: dict) -> dict:
        method = message.get("method")
        request_id = message.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-http", "version": "1.0"},
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            if name == "echo":
                text = (params.get("arguments") or {}).get("text", "")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": text}]},
                }
            if name == "wipe":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": "wiped"}]},
                }
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"no tool called {name!r}"},
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"no method {method!r}"},
        }

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
        session: str = "",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if session:
            self.send_header("Mcp-Session-Id", session)
        self.end_headers()
        if body:
            self.wfile.write(body)


def _event(message: dict) -> bytes:
    """One SSE event, with the framing a real stream carries."""
    return f"event: message\ndata: {json.dumps(message)}\n\n".encode("utf-8")


class Handle:
    def __init__(self, server: ThreadingHTTPServer, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread
        host, port = server.server_address[:2]
        self.url = f"http://127.0.0.1:{port}/mcp"

    @property
    def seen_headers(self) -> list[dict]:
        return self._server.seen_headers  # type: ignore[attr-defined]

    @property
    def deleted(self) -> list[str]:
        return self._server.deleted  # type: ignore[attr-defined]

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _Server(ThreadingHTTPServer):
    # Closing a keep-alive connection from the client end is normal and
    # windows reports it as an aborted connection; the base class prints the
    # whole traceback to stderr, which buries the actual test output.
    def handle_error(self, request, client_address) -> None:
        pass


def serve(**kwargs) -> Handle:
    """Start one on a free port. Call `.stop()` when finished."""
    server = _Server(("127.0.0.1", 0), _Handler)
    server.options = Options(**kwargs)  # type: ignore[attr-defined]
    server.seen_headers = []  # type: ignore[attr-defined]
    server.deleted = []  # type: ignore[attr-defined]
    server.session = ""  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return Handle(server, thread)
