"""HTTP requests, restricted to an allowlist of hosts.

SECURITY NOTE - READ BEFORE ENABLING
------------------------------------
Disabled by default (config.http_tool_enabled, env AGENT_ENABLE_HTTP_TOOL=1).

This is the tool that lets the agent reach off the process, so the boundary is
a host allowlist that defaults to loopback only. The agent can inspect your own
services - the llama servers included - without being able to reach the
internet unless you say so.

The layers, and what each one is actually for:

1. **Host allowlist.** Checked against the parsed hostname, defaulting to
   127.0.0.1 / localhost / ::1. Adding a public host is a deliberate act.
2. **Scheme allowlist.** http and https only. Without this, file:// would turn
   a network tool into an unrestricted file reader that ignores the workspace
   jail, and requests supports more schemes than you might expect.
3. **Redirects are refused, not followed.** This is the one that bites: a
   permitted host answering 302 with a Location pointing anywhere would walk
   straight out of the allowlist, and following it would validate the first URL
   while fetching a different one. The redirect is reported to the model
   instead, which can then request the new URL and have it checked properly.
4. **Read-only by default.** GET and HEAD always; POST, PUT, PATCH and DELETE
   only when http_allow_writes is on, because a request that changes state on
   another service is a different risk from reading one.
5. **No ambient credentials.** The session runs with trust_env off, so proxy
   settings and .netrc entries are not picked up, and no credentials of yours
   ride along on a request the model composed.
6. **URL credentials refused.** user:pass@host in a URL is rejected rather than
   quietly forwarded.
7. **Size cap and timeout**, and binary bodies are described rather than dumped
   into the conversation.

A non-2xx status is reported, not raised: "404" is an answer to a question, and
the model should see it rather than a tool failure.

What this is not: a browser, and not a general internet client. It has no
cookie jar, no session state, and no JavaScript.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse

import requests

from tools.base import Tool, ToolError

READ_METHODS = ("GET", "HEAD")
WRITE_METHODS = ("POST", "PUT", "PATCH", "DELETE")
ALLOWED_SCHEMES = ("http", "https")

MAX_URL_LENGTH = 2000
# Content types we will put into the conversation as text.
TEXT_HINTS = ("text/", "json", "xml", "javascript", "x-www-form-urlencoded")


class HttpToolError(ToolError):
    """The request was refused, or could not be made."""


def _is_texty(content_type: str) -> bool:
    lowered = (content_type or "").lower()
    return any(hint in lowered for hint in TEXT_HINTS)


class HttpClient:
    """Makes allowlisted HTTP requests on the agent's behalf."""

    def __init__(
        self,
        *,
        allowed_hosts: tuple[str, ...],
        timeout: float = 20.0,
        max_bytes: int = 100_000,
        allow_writes: bool = False,
    ) -> None:
        self._allowed = tuple(h.strip().lower() for h in allowed_hosts if h.strip())
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._allow_writes = allow_writes

        self._session = requests.Session()
        # Do not inherit proxies or .netrc credentials from the environment.
        self._session.trust_env = False

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        return self._allowed

    @property
    def methods(self) -> tuple[str, ...]:
        return READ_METHODS + (WRITE_METHODS if self._allow_writes else ())

    # --- validation ---

    def check_url(self, url: str) -> str:
        """Return the hostname if the URL is permitted, else raise."""
        if not isinstance(url, str) or not url.strip():
            raise HttpToolError("url must be a non-empty string.")
        if len(url) > MAX_URL_LENGTH:
            raise HttpToolError(f"url is too long (limit {MAX_URL_LENGTH}).")

        # A url with whitespace or control characters inside it is malformed,
        # and urlparse will not complain: it keeps the junk in the path and the
        # request goes out mangled. A small model leaking its own tool-call
        # scaffolding into the argument is the common cause, and an embedded
        # CR/LF is worth refusing on its own account. Saying so plainly lets
        # the model correct itself on the next round.
        inner = url.strip()
        if any(ch.isspace() or ord(ch) < 32 for ch in inner):
            raise HttpToolError(
                "The url contains whitespace or control characters. Send only "
                "the url itself, e.g. http://127.0.0.1:8080/health, and put "
                "the method in the method argument."
            )

        try:
            parsed = urlparse(url.strip())
        except ValueError as exc:
            raise HttpToolError(f"Could not parse the url: {exc}") from None

        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            raise HttpToolError(
                f"Only {' and '.join(ALLOWED_SCHEMES)} urls are allowed, got "
                f"{parsed.scheme or 'no scheme'!r}."
            )
        if parsed.username or parsed.password:
            raise HttpToolError("Credentials in the url are not allowed.")

        host = (parsed.hostname or "").lower()
        if not host:
            raise HttpToolError("The url has no host.")
        if host not in self._allowed:
            raise HttpToolError(
                f"{host!r} is not an allowed host. Allowed: "
                f"{', '.join(self._allowed)}."
            )
        return host

    def check_method(self, method: str) -> str:
        upper = (method or "GET").strip().upper()
        if upper not in self.methods:
            if upper in WRITE_METHODS:
                raise HttpToolError(
                    f"{upper} changes state on the other service and is off. "
                    f"Set AGENT_HTTP_ALLOW_WRITES=1 to permit it."
                )
            raise HttpToolError(
                f"{upper!r} is not a supported method. Allowed: "
                f"{', '.join(self.methods)}."
            )
        return upper

    # --- the tool entry point ---

    def request(
        self,
        url: str,
        method: str = "GET",
        body: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        self.check_url(url)
        verb = self.check_method(method)

        headers = {"Accept": "*/*"}
        if body is not None and content_type:
            headers["Content-Type"] = content_type
        elif body is not None:
            # Guess only between the two the agent realistically sends.
            headers["Content-Type"] = (
                "application/json" if _looks_like_json(body) else "text/plain"
            )

        started = time.time()
        try:
            response = self._session.request(
                verb,
                url.strip(),
                data=body.encode("utf-8") if isinstance(body, str) else None,
                headers=headers,
                timeout=self._timeout,
                # See the module note: following a redirect would fetch a URL
                # that was never checked.
                allow_redirects=False,
                stream=True,
            )
        except requests.Timeout:
            raise HttpToolError(
                f"No response within {self._timeout:g}s."
            ) from None
        except requests.ConnectionError:
            raise HttpToolError(
                f"Could not connect to {url}. Is the service running?"
            ) from None
        except requests.RequestException as exc:
            raise HttpToolError(f"Request failed: {exc}") from None

        elapsed_ms = int((time.time() - started) * 1000)

        with response:
            content_type_header = response.headers.get("Content-Type", "")
            raw = response.raw.read(self._max_bytes + 1, decode_content=True) or b""

        truncated = len(raw) > self._max_bytes
        raw = raw[: self._max_bytes]

        result: dict[str, Any] = {
            "success": True,
            "status": response.status_code,
            "url": url.strip(),
            "method": verb,
            "content_type": content_type_header,
            "elapsed_ms": elapsed_ms,
            "truncated": truncated,
        }

        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            # Report rather than follow, so the next hop gets checked too.
            result["redirect_to"] = response.headers.get("Location", "")
            result["body"] = (
                f"The server redirected to {result['redirect_to']!r}. Redirects "
                f"are not followed automatically; request that url directly if "
                f"it is on an allowed host."
            )
            return result

        if _is_texty(content_type_header) or not raw:
            result["body"] = raw.decode("utf-8", errors="replace")
        else:
            result["body"] = (
                f"({len(raw)} bytes of {content_type_header or 'unknown type'} "
                f"- not text, so not shown)"
            )
        return result

    def tool(self) -> Tool:
        return Tool(
            name="http_request",
            category="http",
            description=(
                "Make an HTTP request to an allowed host and return the status "
                "and body. Allowed hosts: "
                + ", ".join(self._allowed)
                + ". Methods: "
                + ", ".join(self.methods)
                + ". Use this to check local services and their APIs. "
                "Redirects are reported, not followed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full url, e.g. http://127.0.0.1:8080/health",
                    },
                    "method": {
                        "type": "string",
                        "description": f"One of {', '.join(self.methods)}. Defaults to GET.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body, for methods that take one.",
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Content-Type for the body, e.g. application/json.",
                    },
                },
                "required": ["url"],
            },
            run=self.request,
        )


def _looks_like_json(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except ValueError:
        return False
    return True


def build_http_tool(
    *,
    allowed_hosts: tuple[str, ...],
    timeout: float,
    max_bytes: int,
    allow_writes: bool,
) -> Tool:
    return HttpClient(
        allowed_hosts=allowed_hosts,
        timeout=timeout,
        max_bytes=max_bytes,
        allow_writes=allow_writes,
    ).tool()
