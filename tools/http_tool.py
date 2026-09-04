"""HTTP requests, restricted to an allowlist of hosts.

SECURITY NOTE - READ BEFORE ENABLING
------------------------------------
Disabled by default (config.http_tool_enabled, env AGENT_ENABLE_HTTP_TOOL=1).

This is the tool that lets the agent reach off the process. The boundary is no
longer a wall but a door with someone standing at it: the host allowlist says
what goes through freely, and everything else is shown to a person first.

Three tiers, the same shape as the terminal tool:

* **Free.** A GET or HEAD to an allowlisted host - loopback by default. This
  is the case the tool exists for: inspecting your own services, the llama
  servers included, with no prompt and no ceremony.
* **Approved.** Any other host, and any method that changes state. The turn
  blocks, the method and full url are shown, and it goes only on a yes.
  Silence declines. This replaces two flat refusals that were friction rather
  than protection: reaching a public host used to need an env var set before
  the process started, and a POST to your own local API was impossible without
  a second one.
* **Refused.** Things a prompt could not sensibly stand in for - see below.

The layers, and what each one is actually for:

1. **Host allowlist.** Checked against the parsed hostname, defaulting to
   127.0.0.1 / localhost / ::1. Now it decides *free versus asked*, not
   *allowed versus refused*, so a lookalike like `localhost.evil.com` still
   fails to be free and still ends up in front of a person.
2. **Scheme allowlist.** http and https only, and this one stays a refusal.
   `file://` is not a network request at all - it would be an unrestricted
   file reader that ignores the workspace jail - so there is no version of it
   worth asking about.
3. **Redirects are followed only within the free list.** A permitted host
   answering 302 with a Location pointing anywhere would otherwise walk
   straight out of the allowlist. Each hop is re-parsed and re-checked; one
   landing on a free host is followed (up to MAX_REDIRECTS), and one landing
   anywhere else is reported so the model requests it separately and the
   person sees *that* host in a prompt of its own. Refusing every redirect was
   the old behaviour and it made `http://host/x` -> `http://host/x/` a dead
   end - friction protecting nothing, since the second url passes exactly the
   check the first one did.
4. **Writes ask rather than being switched off.** `http_allow_writes` now
   means "do not ask about writes", for someone driving a local API who does
   not want a prompt on every call. It no longer means "writes are impossible".
5. **No ambient credentials.** The session runs with trust_env off, so proxy
   settings and .netrc entries are not picked up, and no credentials of yours
   ride along on a request the model composed.
6. **URL credentials refused**, and this stays a refusal too: nobody should be
   asked to eyeball a password embedded in a url and judge it.
7. **Size cap and timeout**, and binary bodies are described rather than dumped
   into the conversation.

**What approval is not.** It means a person read the method and the url before
it went. It does not mean the response is safe to act on: anything that comes
back is still text a model will read, from a host you agreed to once.

A non-2xx status is reported, not raised: "404" is an answer to a question, and
the model should see it rather than a tool failure.

What this is not: a browser, and not a general internet client. It has no
cookie jar, no session state, and no JavaScript.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import requests

from tools.base import Tool, ToolError

READ_METHODS = ("GET", "HEAD")
WRITE_METHODS = ("POST", "PUT", "PATCH", "DELETE")
ALLOWED_SCHEMES = ("http", "https")

MAX_URL_LENGTH = 2000
# Content types we will put into the conversation as text.
TEXT_HINTS = ("text/", "json", "xml", "javascript", "x-www-form-urlencoded")

# How many same-host redirects to follow before giving up. Enough for the
# ordinary http->https and /path -> /path/ hops; short enough that a loop
# between two allowed hosts ends quickly rather than spinning.
MAX_REDIRECTS = 5

# Asked before a request that leaves the allowlist or changes state: the
# thing being done and why it is being asked about, in; yes or no, out.
ApprovalCheck = Callable[[str, str], bool]


class HttpToolError(ToolError):
    """The request was refused, or could not be made."""


@dataclass(frozen=True)
class Verdict:
    """What checking a request decided."""

    url: str
    method: str
    host: str
    needs_approval: bool = False
    reason: str = ""


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
        approve: ApprovalCheck | None = None,
    ) -> None:
        self._allowed = tuple(h.strip().lower() for h in allowed_hosts if h.strip())
        self._timeout = timeout
        self._max_bytes = max_bytes
        # Now means "do not ask about writes", not "permit writes at all".
        # Writes used to be impossible without it; they are now possible with
        # a person's agreement, and this turns that prompt off for someone
        # who is driving a local API and does not want one every time.
        self._allow_writes = allow_writes
        # None means nobody is listening - the CLI, a test. Anything that
        # would need approval is refused there rather than sent.
        self._approve = approve

        self._session = requests.Session()
        # Do not inherit proxies or .netrc credentials from the environment.
        self._session.trust_env = False

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        return self._allowed

    @property
    def methods(self) -> tuple[str, ...]:
        """Every method the model may ask for. Some of them will ask first."""
        return READ_METHODS + WRITE_METHODS

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
        return host

    def is_free_host(self, host: str) -> bool:
        """Whether this host may be reached without asking anyone."""
        return host.lower() in self._allowed

    def check_method(self, method: str) -> str:
        upper = (method or "GET").strip().upper()
        if upper not in self.methods:
            raise HttpToolError(
                f"{upper!r} is not a supported method. Allowed: "
                f"{', '.join(self.methods)}."
            )
        return upper

    def plan(self, url: str, method: str) -> Verdict:
        """Decide whether a request may go, and whether to ask first.

        Two things make a request worth asking about, and they are different
        risks: leaving the allowlist means bytes go somewhere new, and a write
        means something changes at the other end. A request can be both, and
        then the prompt says both - "sends a POST to api.example.com, which is
        not on the allowed list" is what someone needs to decide on.
        """
        host = self.check_url(url)
        verb = self.check_method(method)

        off_list = not self.is_free_host(host)
        writing = verb in WRITE_METHODS and not self._allow_writes

        if not off_list and not writing:
            return Verdict(url.strip(), verb, host)

        if off_list and writing:
            reason = (
                f"sends a {verb} to {host}, which is not on the allowed list - "
                f"so it both leaves this machine and changes something there"
            )
        elif off_list:
            reason = (
                f"fetches from {host}, which is not on the allowed list "
                f"({', '.join(self._allowed)}) - the request leaves this machine"
            )
        else:
            reason = f"sends a {verb} to {host}, which changes state there"

        return Verdict(url.strip(), verb, host, True, reason)

    # --- the tool entry point ---

    def request(
        self,
        url: str,
        method: str = "GET",
        body: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        verdict = self.plan(url, method)
        verb = verdict.method

        if verdict.needs_approval:
            asked = f"{verb} {verdict.url}"
            if self._approve is None:
                raise HttpToolError(
                    f"{asked} needs approval before it can be sent, and there "
                    f"is nobody to ask in this context. Add the host to "
                    f"AGENT_HTTP_ALLOWED_HOSTS, or use the web interface "
                    f"where the prompt can be shown."
                )
            if not self._approve(asked, verdict.reason):
                return {
                    "success": False,
                    "error": (
                        f"Not approved: {asked} was declined, or the request "
                        f"timed out. Do not try to reach it another way."
                    ),
                    "url": verdict.url,
                    "method": verb,
                    "declined": True,
                }

        headers = {"Accept": "*/*"}
        if body is not None and content_type:
            headers["Content-Type"] = content_type
        elif body is not None:
            # Guess only between the two the agent realistically sends.
            headers["Content-Type"] = (
                "application/json" if _looks_like_json(body) else "text/plain"
            )

        started = time.time()
        current = verdict.url
        hops: list[str] = []
        payload = body.encode("utf-8") if isinstance(body, str) else None

        for _ in range(MAX_REDIRECTS + 1):
            try:
                response = self._session.request(
                    verb,
                    current,
                    data=payload,
                    headers=headers,
                    timeout=self._timeout,
                    # Never requests' own following: it would fetch a url that
                    # was never checked. The loop here re-checks every hop.
                    allow_redirects=False,
                    stream=True,
                )
            except requests.Timeout:
                raise HttpToolError(
                    f"No response within {self._timeout:g}s."
                ) from None
            except requests.ConnectionError:
                raise HttpToolError(
                    f"Could not connect to {current}. Is the service running?"
                ) from None
            except requests.RequestException as exc:
                raise HttpToolError(f"Request failed: {exc}") from None

            redirecting = response.status_code in (301, 302, 303, 307, 308)
            location = response.headers.get("Location", "") if redirecting else ""
            if not redirecting or not location:
                break

            # Resolved against the current url, because a Location is
            # commonly a bare path.
            target = urljoin(current, location)
            try:
                target_host = self.check_url(target)
            except HttpToolError:
                target_host = ""

            # Followed only into the free list. A redirect somewhere else is
            # reported instead, so the model re-requests it and the person
            # sees the new host in a prompt of its own - the approval was for
            # the url that was asked about, not for wherever it points.
            if not target_host or not self.is_free_host(target_host):
                response.close()
                elapsed_ms = int((time.time() - started) * 1000)
                return {
                    "success": True,
                    "status": response.status_code,
                    "url": current,
                    "method": verb,
                    "content_type": response.headers.get("Content-Type", ""),
                    "elapsed_ms": elapsed_ms,
                    "truncated": False,
                    "redirect_to": target,
                    "redirects": hops,
                    "body": (
                        f"The server redirected to {target!r}, which is not on "
                        f"the allowed list. Request that url directly if you "
                        f"want it - it will be checked, and asked about, on "
                        f"its own."
                    ),
                }

            response.close()
            hops.append(target)
            current = target
        else:
            raise HttpToolError(
                f"More than {MAX_REDIRECTS} redirects starting from "
                f"{verdict.url}; giving up."
            )

        elapsed_ms = int((time.time() - started) * 1000)

        with response:
            content_type_header = response.headers.get("Content-Type", "")
            raw = response.raw.read(self._max_bytes + 1, decode_content=True) or b""

        truncated = len(raw) > self._max_bytes
        raw = raw[: self._max_bytes]

        result: dict[str, Any] = {
            "success": True,
            "status": response.status_code,
            "url": current,
            "method": verb,
            "content_type": content_type_header,
            "elapsed_ms": elapsed_ms,
            "truncated": truncated,
        }
        if hops:
            # So the model can see it did not end up where it asked.
            result["redirects"] = hops

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
                "Make an HTTP request and return the status and body. "
                + ", ".join(self._allowed)
                + " are reached straight away; any other host is shown to the "
                "user for approval first, so expect a pause - and if one is "
                "declined, say so rather than looking for another way to "
                "reach it. Methods: "
                + ", ".join(self.methods)
                + (
                    "; the ones that change state also ask."
                    if not self._allow_writes
                    else "."
                )
                + " Redirects within the allowed hosts are followed; one "
                "leaving them is reported so you can request it separately."
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
    approve: ApprovalCheck | None = None,
) -> Tool:
    return HttpClient(
        allowed_hosts=allowed_hosts,
        timeout=timeout,
        max_bytes=max_bytes,
        allow_writes=allow_writes,
        approve=approve,
    ).tool()
