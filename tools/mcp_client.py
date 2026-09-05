"""Model Context Protocol servers, as tools the agent can call.

MCP is how most of the tool ecosystem is now written: a server exposes tools
over JSON-RPC, and any client can use them. Supporting it means the agent
gains tools nobody here has to write.

Three decisions shape this file, and all three come from the hardware.

**The manifest is cached; servers start lazily.** To register a server's tools
you need their schemas, and getting those means starting the server and asking
it. But the registry is rebuilt on *every turn* - it is derived from the
config, and a settings change must not leave a stale one behind. Starting
every MCP server every turn would be minutes of subprocess churn on a machine
that cannot spare it. So the tool list is fetched once, written to disk, and
served from there; a server is only actually started when one of its tools is
called. `refresh` is the one operation that pays the cost, and a person asks
for it.

**Each server is a lens category.** A filesystem MCP server is a dozen tools
and several thousand tokens of schema. The lens already exists to keep that
out of the prompt until it is wanted, and a server is exactly the right grain
for a category - so `mcp:github` is a group that opens when the conversation
mentions github, and costs nothing before that.

**Anything not declared read-only asks first.** MCP tools carry annotations,
and `readOnlyHint` is the one that matters: a server that says a tool only
reads is taken at its word, and everything else goes in front of a person the
way `git commit` and `POST` do. A hint is the server's claim rather than a
guarantee, which is why it can only ever move a tool into the safer tier -
absent annotations mean "ask".

No SDK. The stdio transport is newline-delimited JSON-RPC 2.0, the client side
of it is one request and one response, and this project does not take a
dependency it can write in a page. HTTP transports and OAuth are not here; the
servers people run locally are stdio.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolError

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "hakim", "version": "1.0"}

# A server that has not answered in this long is not going to.
DEFAULT_TIMEOUT = 30.0
# Handshake and listing are slower: `npx` may be fetching the package.
STARTUP_TIMEOUT = 120.0


class McpError(ToolError):
    """A server could not be reached, or refused the call."""


@dataclass(frozen=True)
class ServerSpec:
    """How to start one server, from mcp.json."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    # Named here so an operator can trust a whole server without annotating
    # every tool. Off by default: trust is a thing someone states.
    trusted: bool = False
    enabled: bool = True

    @property
    def category(self) -> str:
        return f"mcp:{self.name}"


def load_servers(path: Path) -> list[ServerSpec]:
    """Read mcp.json. A missing or broken file means no servers, not a crash.

    The shape is the one every other MCP client uses, so a config can be
    copied from elsewhere:

        {"mcpServers": {"files": {"command": "npx", "args": ["-y", "..."]}}}
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        return []

    found: list[ServerSpec] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("command"), str):
            continue
        if not str(name).strip():
            continue
        found.append(
            ServerSpec(
                name=str(name).strip(),
                command=entry["command"],
                args=tuple(str(a) for a in entry.get("args", []) or ()),
                env={
                    str(k): str(v) for k, v in (entry.get("env") or {}).items()
                },
                cwd=str(entry.get("cwd", "") or ""),
                trusted=bool(entry.get("trusted", False)),
                enabled=entry.get("enabled", True) is not False,
            )
        )
    return found


class McpConnection:
    """One running server, spoken to over its stdin and stdout.

    Synchronous on purpose. The agent loop runs one tool at a time on one
    thread, so there is no concurrency to manage - and a request/response
    client that reads until it sees its own id is a page of code, where an
    async one would be a subsystem.
    """

    def __init__(self, spec: ServerSpec, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._spec = spec
        self._timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        self.last_used = 0.0

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.alive:
            return

        # The child gets a minimal environment plus whatever the server
        # declares, for the same reason the shell tool scrubs its own: an MCP
        # server is somebody else's code, and the API keys in your shell are
        # not part of what it was asked to do.
        environment = {
            name: os.environ[name]
            for name in (
                "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
                "TEMP", "TMP", "HOME", "USERPROFILE", "APPDATA",
                "LOCALAPPDATA", "LANG", "LC_ALL",
            )
            if name in os.environ
        }
        environment.update(self._spec.env)

        try:
            self._process = subprocess.Popen(
                [self._spec.command, *self._spec.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Drained nowhere, so it must not be a pipe: a server that
                # logs steadily would fill the buffer and deadlock, which is
                # the same trap the model manager avoids the same way.
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=self._spec.cwd or None,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise McpError(
                f"Could not start the MCP server {self._spec.name!r}: {exc}. "
                f"Check the command in mcp.json."
            ) from None

        try:
            self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
                timeout=STARTUP_TIMEOUT,
            )
            self._notify("notifications/initialized")
        except McpError:
            self.stop()
            raise

        self.last_used = time.time()

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            # stdout as well as stdin. Closing only the one the client writes
            # to leaks a descriptor per server started, which the tests caught
            # as a ResourceWarning - harmless once, and not once over a long
            # session with a sweeper restarting servers.
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    # --- the protocol ---

    def _send(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise McpError(f"The MCP server {self._spec.name!r} is not running.")
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise McpError(
                f"The MCP server {self._spec.name!r} closed its input: {exc}"
            ) from None

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """One call, waiting for the reply with a matching id.

        Anything else the server sends - log notifications, progress, a
        request of its own - is skipped rather than treated as an answer. A
        client that assumed the next line was its reply would break the first
        time a server logged something.
        """
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )

            deadline = time.time() + (timeout or self._timeout)
            while True:
                if time.time() > deadline:
                    raise McpError(
                        f"The MCP server {self._spec.name!r} did not answer "
                        f"{method!r} within {timeout or self._timeout:.0f}s."
                    )
                if self._process is None or self._process.stdout is None:
                    raise McpError(f"{self._spec.name!r} is not running.")

                line = self._process.stdout.readline()
                if not line:
                    code = self._process.poll()
                    raise McpError(
                        f"The MCP server {self._spec.name!r} stopped"
                        + (f" (exit code {code})" if code is not None else "")
                        + "."
                    )
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue  # a server writing prose to stdout, not a reply
                if not isinstance(message, dict) or message.get("id") != request_id:
                    continue

                error = message.get("error")
                if isinstance(error, dict):
                    raise McpError(
                        f"{self._spec.name}: {error.get('message', 'call failed')}"
                    )
                result = message.get("result")
                return result if isinstance(result, dict) else {}

    def list_tools(self) -> list[dict[str, Any]]:
        self.start()
        result = self._request("tools/list", timeout=STARTUP_TIMEOUT)
        tools = result.get("tools")
        return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.start()
        self.last_used = time.time()
        return self._request("tools/call", {"name": tool, "arguments": arguments})


def _readonly(tool: dict[str, Any]) -> bool:
    """Whether a server claims this tool only reads.

    A hint, not a guarantee - it is the server describing itself. So it can
    only ever move a tool into the tier that does not ask; nothing here lets
    an annotation make something *more* permitted than the default, and the
    default is to ask.
    """
    annotations = tool.get("annotations")
    if not isinstance(annotations, dict):
        return False
    return annotations.get("readOnlyHint") is True


def _text_of(result: dict[str, Any]) -> str:
    """The readable part of an MCP call result.

    Content is a list of typed blocks. Text is what a model can use; an image
    or an embedded resource is described rather than inlined, because the
    alternative is base64 filling the context window.
    """
    blocks = result.get("content")
    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif kind == "image":
            parts.append(f"({block.get('mimeType', 'an image')}, not shown)")
        elif kind == "resource":
            resource = block.get("resource")
            uri = resource.get("uri") if isinstance(resource, dict) else None
            parts.append(f"(resource {uri or 'attached'})")
    return "\n".join(parts).strip()


class McpManager:
    """Every configured server, their cached tool lists, and their lifetimes.

    The cache is what makes this affordable. Building the registry happens on
    every turn; starting a subprocess per server to ask what it offers does
    not. So the list is fetched once by `refresh`, written beside the other
    generated state, and read from there afterwards.
    """

    def __init__(
        self,
        config_path: Path,
        cache_path: Path,
        *,
        idle_timeout: float = 300.0,
    ) -> None:
        self._config_path = Path(config_path)
        self._cache_path = Path(cache_path)
        self._idle_timeout = idle_timeout
        self._specs = {
            spec.name: spec
            for spec in load_servers(self._config_path)
            if spec.enabled
        }
        self._connections: dict[str, McpConnection] = {}
        self._lock = threading.Lock()

    @property
    def servers(self) -> list[ServerSpec]:
        return [self._specs[name] for name in sorted(self._specs)]

    # --- the cached manifest ---

    def _read_cache(self) -> dict[str, list[dict[str, Any]]]:
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        servers = raw.get("servers")
        if not isinstance(servers, dict):
            return {}
        return {
            name: [t for t in tools if isinstance(t, dict)]
            for name, tools in servers.items()
            if isinstance(tools, list)
        }

    def _write_cache(self, manifest: dict[str, list[dict[str, Any]]]) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(
                    {"servers": manifest, "refreshed_at": time.time()}, indent=2
                ),
                encoding="utf-8",
            )
        except OSError:
            # A cache that cannot be written means asking again next time,
            # which is slow rather than broken.
            pass

    def refresh(self) -> dict[str, Any]:
        """Ask every server what it offers, and remember. The expensive one.

        Each is started and stopped again around the question: the point of
        refreshing is to make the *next* turns cheap, not to leave a dozen
        subprocesses resident on an 8 GB machine.
        """
        manifest: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, str] = {}

        for spec in self.servers:
            connection = McpConnection(spec)
            try:
                manifest[spec.name] = connection.list_tools()
            except McpError as exc:
                errors[spec.name] = str(exc)
            finally:
                connection.stop()

        self._write_cache(manifest)
        return {
            "success": True,
            "servers": {name: len(tools) for name, tools in manifest.items()},
            "errors": errors,
        }

    # --- connections ---

    def connection(self, name: str) -> McpConnection:
        with self._lock:
            existing = self._connections.get(name)
            if existing is not None and existing.alive:
                return existing
            spec = self._specs.get(name)
            if spec is None:
                raise McpError(f"No MCP server named {name!r} is configured.")
            connection = McpConnection(spec)
            self._connections[name] = connection
            return connection

    def sweep(self) -> list[str]:
        """Stop servers nobody has used lately, on the same sweep as models."""
        if self._idle_timeout <= 0:
            return []
        stopped = []
        now = time.time()
        with self._lock:
            for name, connection in list(self._connections.items()):
                if connection.alive and now - connection.last_used > self._idle_timeout:
                    connection.stop()
                    self._connections.pop(name, None)
                    stopped.append(name)
        return stopped

    def stop_all(self) -> list[str]:
        with self._lock:
            names = [n for n, c in self._connections.items() if c.alive]
            for connection in self._connections.values():
                connection.stop()
            self._connections.clear()
            return names

    # --- becoming Hakim tools ---

    def tools(self, *, approve: Any = None) -> list[Tool]:
        """One Hakim tool per cached MCP tool.

        Names carry the server so two servers may both offer `search` without
        colliding, and so the model can see where a tool came from - which
        matters when weighing what it returns.
        """
        built: list[Tool] = []
        manifest = self._read_cache()

        for spec in self.servers:
            for entry in manifest.get(spec.name, []):
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    continue
                schema = entry.get("inputSchema")
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}

                needs_approval = not (spec.trusted or _readonly(entry))
                description = entry.get("description") or f"{name} on {spec.name}"
                if needs_approval:
                    description += " Asks for permission before running."

                built.append(
                    Tool(
                        name=f"{spec.name}__{name}",
                        category=spec.category,
                        description=description,
                        parameters=schema,
                        run=self._runner(spec, name, needs_approval, approve),
                    )
                )
        return built

    def _runner(
        self, spec: ServerSpec, tool: str, needs_approval: bool, approve: Any
    ):
        def run(**arguments: Any) -> dict[str, Any]:
            if needs_approval:
                asked = f"{spec.name}: {tool}"
                if approve is None:
                    raise McpError(
                        f"{asked} needs approval before it can run, and there "
                        f"is nobody to ask in this context. Mark the server "
                        f"trusted in mcp.json, or use the web interface where "
                        f"the prompt can be shown."
                    )
                summary = json.dumps(arguments, ensure_ascii=False)[:300]
                if not approve(
                    f"{asked} {summary}",
                    f"calls {tool!r} on the {spec.name!r} MCP server, which "
                    f"does not declare itself read-only",
                ):
                    return {
                        "success": False,
                        "error": (
                            f"Not approved: {asked} was declined, or the "
                            f"request timed out."
                        ),
                        "declined": True,
                    }

            result = self.connection(spec.name).call(tool, arguments)
            text = _text_of(result)
            # `isError` is a tool-level failure, as distinct from the protocol
            # failing. The model needs to know which it was, so it is reported
            # rather than raised.
            failed = result.get("isError") is True
            payload: dict[str, Any] = {
                "success": not failed,
                "server": spec.name,
                "tool": tool,
            }
            if failed:
                payload["error"] = text or "The tool reported a failure."
            else:
                payload["output"] = text or "(no output)"
            return payload

        return run
