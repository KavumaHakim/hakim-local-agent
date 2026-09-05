"""A real MCP server, run as a subprocess by the client tests.

Stdlib only and no import of the project, because it is executed with
`python fake_mcp_server.py` and has to behave like somebody else's program.

It answers the protocol properly: an `initialize` handshake, a `tools/list`
with one read-only tool and one that is not, and `tools/call`. It also emits a
log notification before its first reply, which is the case a client that
assumed the next line was its answer would fail on.
"""

from __future__ import annotations

import json
import sys


TOOLS = [
    {
        "name": "echo",
        "description": "Return whatever it is given.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "wipe",
        "description": "Pretends to delete something.",
        "inputSchema": {"type": "object", "properties": {}},
        # No readOnlyHint, so the client must treat it as needing approval.
    },
    {
        "name": "explode",
        "description": "Reports a tool-level failure.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
]


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    noisy = "--noisy" in sys.argv
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue

        method = request.get("method")
        request_id = request.get("id")

        if method == "notifications/initialized":
            continue

        if noisy:
            # A notification with no id, sent before the real reply. A client
            # that reads one line and calls it the answer breaks here.
            send({"jsonrpc": "2.0", "method": "notifications/message",
                  "params": {"level": "info", "data": "starting up"}})

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"},
            }})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": request_id,
                  "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "echo":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text",
                                 "text": str(arguments.get("text", ""))}]
                }})
            elif name == "explode":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "it went wrong"}],
                }})
            elif name == "wipe":
                send({"jsonrpc": "2.0", "id": request_id, "result": {
                    "content": [{"type": "text", "text": "wiped"}]
                }})
            else:
                send({"jsonrpc": "2.0", "id": request_id,
                      "error": {"code": -32602, "message": f"no tool {name!r}"}})
        else:
            send({"jsonrpc": "2.0", "id": request_id,
                  "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()
