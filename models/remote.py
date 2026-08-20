"""Chat client for hosted, OpenAI-compatible APIs.

Speed, when there is internet. A turn that costs minutes on this CPU costs
seconds on someone else's accelerator, and that is the entire reason this
module exists.

Both providers speak the OpenAI chat-completions shape, so one client covers
them:

  * **Cerebras** — `https://api.cerebras.ai/v1`, natively OpenAI-compatible.
  * **Gemini** — Google's OpenAI-compatibility endpoint, rather than the native
    `generateContent` API. That is a deliberate trade: the compatibility layer
    exposes the same `tools` array and the same streaming deltas the agent loop
    already speaks, so no translation layer is needed and the loop stays
    identical whichever model answers.

What this does NOT do is pretend to be local. `models/qwen.py` is the only
module that talks to llama-server; this is the only one that leaves the
machine. Keeping them apart is what makes it obvious, reading the code, which
turns go where.

The API key is read from the environment at call time and never stored on a
spec, logged, or included in an error message.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import requests

from models.manager import ModelSpec
from models.qwen import (
    Message,
    QwenConnectionError,
    QwenError,
    QwenResponseError,
    QwenTimeoutError,
    TokenCallback,
    _merge_tool_call,
)


class RemoteError(QwenError):
    """A hosted provider failed.

    Subclasses QwenError so the agent loop and the API error paths treat a
    cloud failure exactly like a local one - there is nothing a caller can
    usefully do differently.
    """


class MissingKeyError(RemoteError):
    """The provider's API key is not in the environment."""


class RemoteResponseError(RemoteError):
    """The provider returned something unreadable."""


class RemoteHTTPError(RemoteError):
    """A hosted provider returned a non-2xx status.

    Its own class rather than QwenHTTPError, whose message is hardcoded to
    "llama.cpp server returned HTTP ...". A Cerebras 402 reported as a
    llama.cpp failure sends you to look at a local process that is working
    fine - and a wrong hint costs more than no hint.
    """

    def __init__(self, label: str, status_code: int, body: str) -> None:
        hint = ""
        if status_code in (401, 403):
            hint = " The API key was rejected."
        elif status_code == 402:
            hint = " The account has no credit or quota for this model."
        elif status_code == 404:
            hint = " Check the model id in models.json against list_models()."
        elif status_code == 429:
            hint = " Rate limited; try again shortly or use a local model."
        super().__init__(f"{label} returned HTTP {status_code}.{hint} {body[:300]}")
        self.status_code = status_code
        self.body = body


class RemoteClient:
    """Chat client for a hosted model. Implements the ChatClient protocol."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = -1,
        request_timeout: float = 120.0,
        connect_timeout: float = 10.0,
    ) -> None:
        if not spec.remote:
            raise ValueError(f"{spec.key} is a local model, not a hosted one.")
        self._spec = spec
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        # Far shorter than the local timeout. Twenty minutes is the right
        # allowance for an 8B on a 2-core CPU and absurd for a hosted API: if
        # one of these has not answered in two minutes, something is wrong and
        # falling back to local is more useful than waiting.
        self._timeout = (connect_timeout, request_timeout)
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self._spec.base_url

    def _key(self) -> str:
        key = os.environ.get(self._spec.api_key_env, "").strip()
        if not key:
            raise MissingKeyError(
                f"{self._spec.api_key_env} is not set, so {self._spec.label} "
                f"cannot be used. Put it in .env and restart the API."
            )
        return key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
        }

    def health(self) -> bool:
        """Whether the provider answers. Costs a request, so call it rarely."""
        try:
            response = self._session.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=(5, 10)
            )
        except (requests.RequestException, MissingKeyError):
            return False
        return response.status_code == 200

    def list_models(self) -> list[str]:
        """Model ids this key can actually see.

        Used to pin the right id rather than guessing at one: provider models
        get renamed, and a wrong string fails as a 404 that looks like an
        outage.
        """
        response = self._session.get(
            f"{self.base_url}/models", headers=self._headers(), timeout=(5, 20)
        )
        if response.status_code != 200:
            raise RemoteHTTPError(
                self._spec.label, response.status_code, self._scrub(response.text)
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteResponseError(f"Model list was not JSON: {exc}") from None
        return [entry.get("id", "") for entry in payload.get("data", [])]


    # --- the ChatClient protocol ---

    def chat(
        self,
        messages: Iterable[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Message:
        payload = self._payload(messages, tools)
        data = self._post(payload)
        return self._extract(data)

    def chat_stream(
        self,
        messages: Iterable[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_token: TokenCallback | None = None,
        on_reasoning: TokenCallback | None = None,
    ) -> Message:
        """Stream a reply, assembling the same message `chat` would return."""
        payload = self._payload(messages, tools)
        payload["stream"] = True

        content: list[str] = []
        partial_calls: dict[int, dict[str, Any]] = {}
        reasoning_chars = 0

        for chunk in self._stream(payload):
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue

            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                content.append(piece)
                if on_token is not None:
                    on_token(piece)

            # Providers disagree on the field name: llama.cpp and Cerebras use
            # reasoning_content, others use reasoning. Both are accepted so a
            # thinking model shows its working whichever it sends.
            thought = delta.get("reasoning_content")
            if not isinstance(thought, str):
                thought = delta.get("reasoning")
            if isinstance(thought, str) and thought:
                reasoning_chars += len(thought)
                if on_reasoning is not None:
                    on_reasoning(thought)

            for fragment in delta.get("tool_calls") or []:
                if isinstance(fragment, dict):
                    _merge_tool_call(partial_calls, fragment)

        message: Message = {"role": "assistant", "content": "".join(content)}
        if partial_calls:
            message["tool_calls"] = [
                partial_calls[index] for index in sorted(partial_calls)
            ]
        if reasoning_chars:
            # Length only, exactly as the local client does: the trace is shown
            # live and never stored or replayed.
            message["reasoning_chars"] = reasoning_chars
        return message

    # --- internals ---

    def _payload(
        self, messages: Iterable[Message], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._spec.model,
            "messages": list(messages),
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        if self._max_tokens and self._max_tokens > 0:
            payload["max_tokens"] = self._max_tokens
        if tools:
            payload["tools"] = tools
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except requests.Timeout as exc:
            raise QwenTimeoutError(f"{self._spec.label} timed out: {exc}") from None
        except requests.RequestException as exc:
            raise QwenConnectionError(
                f"Could not reach {self._spec.label}: {exc}"
            ) from None

        if response.status_code != 200:
            raise RemoteHTTPError(
                self._spec.label, response.status_code, self._scrub(response.text)
            )
        try:
            return response.json()
        except ValueError as exc:
            raise QwenResponseError(f"Reply was not JSON: {exc}") from None

    def _stream(self, payload: dict[str, Any]):
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
                stream=True,
            )
        except requests.Timeout as exc:
            raise QwenTimeoutError(f"{self._spec.label} timed out: {exc}") from None
        except requests.RequestException as exc:
            raise QwenConnectionError(
                f"Could not reach {self._spec.label}: {exc}"
            ) from None

        with response:
            if response.status_code != 200:
                raise RemoteHTTPError(
                    self._spec.label, response.status_code, self._scrub(response.text)
                )

            for raw in response.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                body = raw[len("data:"):].strip()
                if body == "[DONE]":
                    return
                try:
                    yield json.loads(body)
                except ValueError:
                    # One malformed chunk is not worth ending the turn over.
                    continue

    def _scrub(self, text: str) -> str:
        """Remove the API key from provider error text before it is shown.

        Some providers echo the request back in an error body. The key must not
        travel from there into a log, an SSE event, or the browser.
        """
        body = text[:500]
        try:
            key = self._key()
        except MissingKeyError:
            return body
        return body.replace(key, "***") if key else body

    @staticmethod
    def _extract(data: dict[str, Any]) -> Message:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise QwenResponseError("Reply contained no choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise QwenResponseError("Reply contained no message.")
        return message

