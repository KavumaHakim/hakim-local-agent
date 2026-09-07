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

**Two transports, one protocol.** A server with a `command` is a subprocess
here, spoken to over its stdin and stdout; one with a `url` is somebody else's,
over Streamable HTTP. After the handshake they are the same JSON-RPC, so
`open_connection` picks the transport and nothing above that line knows which
it got. The 2024 two-endpoint HTTP+SSE transport is deprecated upstream and is
not implemented.

No SDK. Stdio MCP is newline-delimited JSON-RPC 2.0 and the HTTP one is a POST
that answers with either JSON or an event stream; the client side of both is a
page of code, and this project does not take a dependency it can write in a
page. Interactive OAuth is not here - static credentials in `headers` cover
the servers people actually run, and a 401 says what is missing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from tools.base import Tool, ToolError

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "hakim", "version": "1.0"}

# A server that has not answered in this long is not going to.
DEFAULT_TIMEOUT = 30.0
# Handshake and listing are slower: `npx` may be fetching the package.
STARTUP_TIMEOUT = 120.0

# How much a streaming HTTP server may send before answering. A server that
# logs forever must not hold a turn open forever.
MAX_STREAM_BYTES = 2_000_000


class McpError(ToolError):
    """A server could not be reached, or refused the call."""


@dataclass(frozen=True)
class ServerSpec:
    """How to reach one server, from mcp.json.

    Two transports, and `url` is what chooses. A spec with a command is a
    subprocess on this machine; a spec with a url is somebody else's, over
    HTTP. Everything after the handshake is the same protocol either way,
    which is why one dataclass covers both.
    """

    name: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    # Named here so an operator can trust a whole server without annotating
    # every tool. Off by default: trust is a thing someone states.
    trusted: bool = False
    enabled: bool = True
    # Streamable HTTP. Set instead of `command`, never as well.
    url: str = ""
    # Sent with every HTTP request. `${VAR}` values are read from the
    # environment, so a bearer token need not live in the file.
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def category(self) -> str:
        return f"mcp:{self.name}"

    @property
    def remote(self) -> bool:
        return bool(self.url)

    @property
    def display(self) -> str:
        """What to show as 'how this server is reached'."""
        return self.url if self.remote else " ".join([self.command, *self.args])


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
        if not isinstance(entry, dict) or not str(name).strip():
            continue
        command = entry.get("command")
        url = entry.get("url") or entry.get("serverUrl") or entry.get("endpoint")
        # One or the other. An entry with neither is not a server, and an
        # entry with both does not say which was meant, so neither is guessed
        # at - the pane shows what is configured and this stays predictable.
        if not isinstance(command, str) and not isinstance(url, str):
            continue
        if isinstance(command, str) and isinstance(url, str):
            continue
        found.append(
            ServerSpec(
                name=str(name).strip(),
                command=command if isinstance(command, str) else "",
                args=tuple(str(a) for a in entry.get("args", []) or ()),
                env={
                    str(k): str(v) for k, v in (entry.get("env") or {}).items()
                },
                cwd=str(entry.get("cwd", "") or ""),
                trusted=bool(entry.get("trusted", False)),
                enabled=entry.get("enabled", True) is not False,
                url=(url or "").strip() if isinstance(url, str) else "",
                headers={
                    str(k): str(v) for k, v in (entry.get("headers") or {}).items()
                },
            )
        )
    return found


# A server name has to survive being a JSON key, a tool-name prefix and a lens
# category, so it is kept to what all three read the same way.
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


class McpConfigError(ToolError):
    """The config file could not be changed."""


def edit_config(path: Path, change) -> None:
    """Apply `change` to the `mcpServers` map and write the file back.

    **Everything outside the entry being changed is preserved**, because this
    file is hand-written and git-ignored: it holds `env` blocks with API keys
    in them, and may hold comments-by-convention, ordering, or keys a future
    version of some other client understands. Rewriting it from a parsed model
    of what this code happens to know about would quietly delete a person's
    credentials, and they have no copy - that is the point of git-ignoring it.

    So: read the raw JSON, hand `change` the servers dict to mutate, write the
    whole document back. A file that does not exist yet is created; one that is
    unreadable is an error rather than something to overwrite, for the same
    reason.
    """
    path = Path(path)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise McpConfigError(f"Could not read {path.name}: {exc}") from None
        except ValueError as exc:
            raise McpConfigError(
                f"{path.name} is not valid JSON ({exc}). Fix it by hand - "
                f"overwriting it would lose whatever else is in there, "
                f"including any API keys."
            ) from None
        if not isinstance(raw, dict):
            raise McpConfigError(f"{path.name} does not contain a JSON object.")
    else:
        raw = {}

    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    change(servers)
    raw["mcpServers"] = servers

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and moved into place, so a failure
        # half-way through leaves the original rather than a truncated file.
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise McpConfigError(f"Could not write {path.name}: {exc}") from None


_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def resolve_env(env: dict[str, str]) -> dict[str, str]:
    """Expand `${VAR}` values from this process's environment.

    The better way to give a server a credential: the value stays in the
    environment and mcp.json holds only the name of it, so the secret is not
    written to a file at all. The alternative - pasting the value - is still
    supported, because plenty of people would rather have it in one place than
    manage a shell variable.

    Only an entire value of exactly `${NAME}` is treated as a reference. No
    substitution inside a larger string: half-expanded values are a source of
    silent mistakes, and nothing here needs them.

    A name that is not set resolves to empty rather than being left as the
    literal `${NAME}`, which a server would otherwise send to its API as
    though it were a token.
    """
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _REFERENCE.match(str(value).strip())
        resolved[key] = os.environ.get(match.group(1), "") if match else str(value)
    return resolved


def check_name(name: str) -> str:
    """The server name, or an error saying what a name may be."""
    cleaned = (name or "").strip().lower()
    if not _NAME.match(cleaned):
        raise McpConfigError(
            "A server name must be lowercase letters, digits, hyphens or "
            "underscores, start with a letter or digit, and be at most 41 "
            "characters. It becomes the prefix on every tool the server "
            "offers."
        )
    return cleaned


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
        environment.update(resolve_env(self._spec.env))

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


class McpHttpConnection:
    """One server reached over Streamable HTTP.

    The same protocol as the stdio transport, over a different pipe, so this
    exposes the identical surface - `start`, `list_tools`, `call`, `stop`,
    `alive`, `last_used` - and `open_connection` picks between them. Nothing
    above this line knows which a server is.

    **Streamable HTTP, not the older HTTP+SSE.** One endpoint: every request
    is a POST, and the server answers either with a single JSON object or with
    an event stream carrying the reply among whatever else it wants to send.
    Both are handled, because which one you get is the server's choice per
    request and not a property of the server. The 2024 two-endpoint transport
    (`GET /sse` plus a separate POST url) is deprecated upstream and is not
    implemented.

    **A session is a header.** If the server returns `Mcp-Session-Id` on the
    initialize response, every later request carries it, and `stop` sends a
    DELETE so the server can drop the state rather than waiting for it to age
    out. Servers that do not use sessions simply never send the header.

    **What is not here: interactive OAuth.** The spec's authorization flow
    wants a browser round trip, a redirect listener and dynamic client
    registration; that is a feature, not a detail. Static credentials cover
    the servers people actually run - `headers` in the config, with `${VAR}`
    read from the environment - and a 401 says plainly which is missing.
    """

    def __init__(self, spec: ServerSpec, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._spec = spec
        self._timeout = timeout
        self._session_id = ""
        self._started = False
        self._next_id = 0
        self._lock = threading.Lock()
        self.last_used = 0.0

        self._http = requests.Session()
        # No ambient credentials, the same rule the HTTP tool follows: a
        # proxy setting or a .netrc entry is not part of what this server was
        # configured to receive.
        self._http.trust_env = False

    @property
    def alive(self) -> bool:
        """Whether the handshake has been done and not torn down.

        There is no process to poll, so this is the honest equivalent: a
        connection that has initialized and still holds whatever session the
        server gave it.
        """
        return self._started

    def start(self) -> None:
        if self._started:
            return
        check_url(self._spec.url, self._spec.name)
        # Set before the handshake so `_request` will send it, and cleared
        # again if the handshake fails.
        self._started = True
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
            self._started = False
            self._session_id = ""
            raise
        self.last_used = time.time()

    def stop(self) -> None:
        """End the session, then forget it.

        Best effort: a server that is gone, or one that never used sessions,
        is not a failure to report. Stopping is what the sweeper does to an
        idle connection, and it must not raise into it.
        """
        session, self._session_id = self._session_id, ""
        self._started = False
        if session:
            try:
                self._http.delete(
                    self._spec.url,
                    headers={**self._headers(), "Mcp-Session-Id": session},
                    timeout=5,
                )
            except requests.RequestException:
                pass
        self._http.close()
        # A closed Session cannot be reused, so a later start gets a new one.
        self._http = requests.Session()
        self._http.trust_env = False

    # --- the protocol ---

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            # Both, because the server chooses per response which to send.
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        headers.update(resolve_env(self._spec.headers))
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _post(self, payload: dict[str, Any], timeout: float):
        try:
            return self._http.post(
                self._spec.url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                timeout=timeout,
                stream=True,
            )
        except requests.RequestException as exc:
            raise McpError(
                f"Could not reach the MCP server {self._spec.name!r} at "
                f"{self._spec.url}: {exc}"
            ) from None

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """A message with no id, so there is no reply to wait for."""
        response = self._post(
            {"jsonrpc": "2.0", "method": method, "params": params or {}},
            self._timeout,
        )
        # 202 Accepted is the documented answer; anything 2xx is fine, and a
        # notification failing is not worth ending a turn over.
        response.close()

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            wait = timeout or self._timeout
            response = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                },
                wait,
            )

            try:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self._session_id = session
                self._check_status(response, method)
                message = self._read_reply(response, request_id, method, wait)
            finally:
                response.close()

            error = message.get("error")
            if isinstance(error, dict):
                raise McpError(
                    f"{self._spec.name}: {error.get('message', 'call failed')}"
                )
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def _check_status(self, response, method: str) -> None:
        if response.status_code < 400:
            return
        if response.status_code in (401, 403):
            raise McpError(
                f"The MCP server {self._spec.name!r} refused the request "
                f"({response.status_code}). It wants a credential this client "
                f"did not send - add one under 'headers' for this server, for "
                f"example an Authorization header. Interactive OAuth sign-in "
                f"is not supported here."
            )
        if response.status_code == 404 and self._session_id:
            # The documented way a server says a session has expired.
            self._session_id = ""
            self._started = False
            raise McpError(
                f"The MCP server {self._spec.name!r} no longer has this "
                f"session. It will be re-established on the next call."
            )
        raise McpError(
            f"The MCP server {self._spec.name!r} answered {method!r} with "
            f"HTTP {response.status_code}."
        )

    def _read_reply(
        self, response, request_id: int, method: str, wait: float
    ) -> dict[str, Any]:
        """The JSON-RPC message with our id, from either response shape."""
        content_type = (response.headers.get("Content-Type") or "").lower()

        if "text/event-stream" in content_type:
            return self._read_event_stream(response, request_id, method, wait)

        try:
            body = json.loads(response.content.decode("utf-8", errors="replace"))
        except ValueError:
            raise McpError(
                f"The MCP server {self._spec.name!r} answered {method!r} with "
                f"something that is not JSON."
            ) from None
        # A server may batch, exactly as the stdio transport may interleave.
        for message in body if isinstance(body, list) else [body]:
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
        raise McpError(
            f"The MCP server {self._spec.name!r} answered {method!r} without "
            f"a reply to it."
        )

    def _read_event_stream(
        self, response, request_id: int, method: str, wait: float
    ) -> dict[str, Any]:
        """Read events until ours arrives.

        The same rule as the stdio transport: anything that is not the reply
        to this id - a log, a progress notification, a keep-alive - is
        skipped rather than treated as an answer.
        """
        deadline = time.time() + wait
        seen = 0
        for raw in response.iter_lines(decode_unicode=True):
            if time.time() > deadline:
                break
            if raw is None:
                continue
            line = raw.strip()
            if not line or not line.startswith("data:"):
                continue  # event:, id:, retry: and blank separators
            seen += len(line)
            if seen > MAX_STREAM_BYTES:
                raise McpError(
                    f"The MCP server {self._spec.name!r} sent more than "
                    f"{MAX_STREAM_BYTES:,} bytes without answering {method!r}."
                )
            try:
                message = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
        raise McpError(
            f"The MCP server {self._spec.name!r} did not answer {method!r} "
            f"within {wait:.0f}s."
        )

    def list_tools(self) -> list[dict[str, Any]]:
        self.start()
        result = self._request("tools/list", timeout=STARTUP_TIMEOUT)
        tools = result.get("tools")
        return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []

    def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.start()
        self.last_used = time.time()
        return self._request("tools/call", {"name": tool, "arguments": arguments})


def check_url(url: str, name: str) -> str:
    """Validate a server url, or say what is wrong with it."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise McpError(
            f"The MCP server {name!r} needs an http or https url, got "
            f"{parsed.scheme or 'none'!r}."
        )
    if not parsed.hostname:
        raise McpError(f"The url for the MCP server {name!r} has no host.")
    if parsed.username or parsed.password:
        # The same refusal the HTTP tool makes: nobody should be asked to
        # eyeball a password embedded in a url and judge it.
        raise McpError(
            f"The url for the MCP server {name!r} has credentials in it. Put "
            f"them in 'headers' instead, where they can be a ${{VAR}}."
        )
    return url.strip()


def open_connection(spec: ServerSpec, *, timeout: float = DEFAULT_TIMEOUT):
    """The connection for this spec, whichever transport it names."""
    if spec.remote:
        return McpHttpConnection(spec, timeout=timeout)
    return McpConnection(spec, timeout=timeout)


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

    def reload(self) -> None:
        """Re-read mcp.json.

        The long-lived manager the API holds is built once at startup, so
        without this, adding a server to the file and pressing Refresh would
        cheerfully refresh the *old* set and report success. The registry does
        not have this problem - it builds a new manager every turn - which is
        exactly the kind of inconsistency that only shows up once there is a
        panel displaying both.

        A connection whose spec has changed or disappeared is stopped: it is a
        subprocess started from a command line that no longer exists.
        """
        with self._lock:
            fresh = {
                spec.name: spec
                for spec in load_servers(self._config_path)
                if spec.enabled
            }
            for name, connection in list(self._connections.items()):
                if fresh.get(name) != self._specs.get(name):
                    connection.stop()
                    self._connections.pop(name, None)
            self._specs = fresh

    # --- the cached manifest ---

    def cached_tools(self) -> dict[str, list[dict[str, Any]]]:
        """What each server last said it offers. Reads the file, starts nothing."""
        return self._read_cache()

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

        The config is re-read first, so this is also how a server added to
        mcp.json while the API is running becomes real.
        """
        self.reload()
        manifest: dict[str, list[dict[str, Any]]] = {}
        errors: dict[str, str] = {}

        for spec in self.servers:
            connection = open_connection(spec)
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
            connection = open_connection(spec)
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
