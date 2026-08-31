"""HTTP client for the Qwen3 llama.cpp server.

This module is the only place that speaks HTTP. It knows nothing about tools,
prompts or the agent loop - it just forwards tool definitions and returns the
assistant message the server produced.

Verified against llama-server build 10373 (this machine's binary):
  - POST /v1/chat/completions   OpenAI-compatible, accepts `tools`
  - GET  /health                readiness check
  - --jinja is enabled by default, so the server applies Qwen3's own chat
    template, parses the model's native tool-call syntax, and returns standard
    OpenAI `tool_calls` objects. No custom protocol is needed here.
  - With the default --reasoning-format deepseek, thinking is returned in
    `message.reasoning_content` and kept out of `message.content`.
  - --chat-template-kwargs exists, so per-request `chat_template_kwargs`
    (used below for Qwen3's enable_thinking switch) is supported.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Protocol

import requests

from config import Config

# Called with each content fragment as it streams in.
TokenCallback = Callable[[str], None]

# Asked between chunks: "should this stream be abandoned?" Returning True ends
# it, and the client returns whatever it had assembled so far.
StopCheck = Callable[[], bool]


class QwenError(Exception):
    """Base class for all Qwen client failures."""


class QwenConnectionError(QwenError):
    """The server could not be reached."""


class QwenTimeoutError(QwenError):
    """The server did not respond in time."""


class QwenHTTPError(QwenError):
    """The server returned a non-2xx status."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"llama.cpp server returned HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class QwenResponseError(QwenError):
    """The server returned a response we could not understand."""


Message = dict[str, Any]


class ChatClient(Protocol):
    """What the agent loop needs from a model client.

    Declared so the loop can be tested against a fake without a live server.
    """

    def chat(
        self,
        messages: Iterable[Message],
        *,
        tools: list[dict[str, Any]] | None = ...,
    ) -> Message: ...

    # Optional, and the loop checks for it before calling: a client that
    # cannot stream is used through `chat` instead. A client that can stream
    # must accept `should_stop`, because that is the only handle the loop has
    # on a model round once it has started.


class QwenClient:
    """Chat client for a llama.cpp server running Qwen3."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self._config.qwen_url

    def health(self) -> bool:
        """Return True if the server is up and a model is loaded.

        Never raises: this is used to print a friendly message at startup.
        """
        try:
            response = self._session.get(
                f"{self._config.qwen_url}/health",
                timeout=self._config.connect_timeout,
            )
        except requests.RequestException:
            return False
        return response.status_code == 200

    def chat(
        self,
        messages: Iterable[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> Message:
        """Send a conversation and return the assistant message.

        The return value is the raw OpenAI-shaped message dict, e.g.
        ``{"role": "assistant", "content": "...", "tool_calls": [...]}``.
        Interpreting it is the parser's job, not this client's.
        """
        payload = self._build_payload(messages, tools, temperature)
        data = self._post("/v1/chat/completions", payload)
        return self._extract_message(data)

    def chat_stream(
        self,
        messages: Iterable[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_token: TokenCallback | None = None,
        on_reasoning: TokenCallback | None = None,
        should_stop: StopCheck | None = None,
    ) -> Message:
        """Stream a reply, then return the same assembled message `chat` would.

        Tokens are handed to `on_token` as they arrive. On this hardware a turn
        takes minutes, so streaming is the difference between a usable
        interface and a blank screen.

        `reasoning_content` deltas go to `on_reasoning`, kept on a separate
        channel from the answer. They are still never added to the assembled
        message: thinking is per-turn and replaying it to the model on the next
        round is exactly what the chat template does not expect. Showing it and
        storing it are different questions, and only the first is the caller's
        to make.
        """
        payload = self._build_payload(messages, tools)
        payload["stream"] = True

        content: list[str] = []
        # Tool call fragments arrive spread over chunks, keyed by index.
        partial_calls: dict[int, dict[str, Any]] = {}
        reasoning_chars = 0

        for chunk in self._stream("/v1/chat/completions", payload):
            if should_stop is not None and should_stop():
                # Leaving the loop closes the generator, which closes the
                # response, which drops the connection - and llama-server
                # stops generating when its client goes away. That is what
                # actually gives the CPU back; the flag alone would not.
                break

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

            thought = delta.get("reasoning_content")
            if isinstance(thought, str) and thought:
                reasoning_chars += len(thought)
                if on_reasoning is not None:
                    on_reasoning(thought)

            for fragment in delta.get("tool_calls") or []:
                if isinstance(fragment, dict):
                    _merge_tool_call(partial_calls, fragment)

        # Whatever arrived before the stop, assembled the same way a finished
        # reply is. It is the caller that knows a partial answer is not an
        # answer; this returns what it has and says nothing about why.
        message: Message = {"role": "assistant", "content": "".join(content)}
        if partial_calls:
            message["tool_calls"] = [
                partial_calls[index] for index in sorted(partial_calls)
            ]
        if reasoning_chars:
            # Length only - the text itself is deliberately dropped.
            message["reasoning_chars"] = reasoning_chars
        return message

    # --- internals ---

    def _build_payload(
        self,
        messages: Iterable[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.qwen_model,
            "messages": list(messages),
            "temperature": (
                self._config.temperature if temperature is None else temperature
            ),
            "top_p": self._config.top_p,
            "stream": False,
        }
        if self._config.max_tokens > 0:
            payload["max_tokens"] = self._config.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if not self._config.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _stream(self, path: str, payload: dict[str, Any]):
        """Yield decoded SSE chunks from a streaming completion."""
        url = f"{self._config.qwen_url}{path}"
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=(self._config.connect_timeout, self._config.request_timeout),
                stream=True,
            )
        except requests.Timeout as exc:
            raise QwenTimeoutError(
                f"No response from {url} within "
                f"{self._config.request_timeout:.0f}s."
            ) from exc
        except requests.ConnectionError as exc:
            raise QwenConnectionError(
                f"Could not reach the Qwen server at {self._config.qwen_url}. "
                f"Is llama-server running on that port?"
            ) from exc
        except requests.RequestException as exc:
            raise QwenError(f"Request to {url} failed: {exc}") from exc

        with response:
            if response.status_code >= 400:
                raise QwenHTTPError(response.status_code, response.text)

            try:
                lines = response.iter_lines(decode_unicode=True)
                for line in lines:
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        return
                    try:
                        chunk = json.loads(body)
                    except ValueError:
                        # A malformed keepalive is not worth failing the turn.
                        continue
                    if isinstance(chunk, dict):
                        yield chunk
            except requests.Timeout as exc:
                raise QwenTimeoutError(
                    f"Stream from {url} stalled for more than "
                    f"{self._config.request_timeout:.0f}s."
                ) from exc
            except requests.RequestException as exc:
                raise QwenError(f"Stream from {url} failed: {exc}") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._config.qwen_url}{path}"
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=(self._config.connect_timeout, self._config.request_timeout),
            )
        except requests.Timeout as exc:
            raise QwenTimeoutError(
                f"No response from {url} within "
                f"{self._config.request_timeout:.0f}s. Local generation can be "
                f"slow; raise AGENT_REQUEST_TIMEOUT if this is expected."
            ) from exc
        except requests.ConnectionError as exc:
            raise QwenConnectionError(
                f"Could not reach the Qwen server at {self._config.qwen_url}. "
                f"Is llama-server running on that port?"
            ) from exc
        except requests.RequestException as exc:
            raise QwenError(f"Request to {url} failed: {exc}") from exc

        if response.status_code >= 400:
            raise QwenHTTPError(response.status_code, response.text)

        try:
            data = response.json()
        except ValueError as exc:
            raise QwenResponseError(
                f"Server response was not valid JSON: {response.text[:500]}"
            ) from exc

        if not isinstance(data, dict):
            raise QwenResponseError(f"Expected a JSON object, got {type(data).__name__}")
        return data

    @staticmethod
    def _extract_message(data: dict[str, Any]) -> Message:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise QwenResponseError(f"Response contained no choices: {data!r}")

        first = choices[0]
        if not isinstance(first, dict):
            raise QwenResponseError(f"Malformed choice entry: {first!r}")

        message = first.get("message")
        if not isinstance(message, dict):
            raise QwenResponseError(f"Choice contained no message: {first!r}")

        return message


def _merge_tool_call(store: dict[int, dict[str, Any]], fragment: dict[str, Any]) -> None:
    """Fold one streamed tool-call fragment into the accumulator.

    llama.cpp may send a tool call whole, or split its `arguments` string
    across several chunks. Fragments are keyed by `index`, and argument text is
    concatenated in arrival order.

    **Fields this function does not recognise are carried through untouched.**
    Rebuilding a tool call from only the parts we know about silently discards
    whatever else the provider attached, and at least one provider requires
    getting it back: Gemini 3 puts a `thought_signature` in
    `extra_content.google` on the call, and replaying the assistant message
    without it is rejected with

        400 ... Function call is missing a thought_signature in functionCall
        parts. This is required for tools to work correctly

    which only ever appears on the *second* round, once tool results are sent
    back - so it looks like a tool bug rather than a lossy merge. The
    non-streaming path never had this problem because it returns the
    provider's message as it arrived.
    """
    index = fragment.get("index", 0)
    if not isinstance(index, int):
        index = 0

    entry = store.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )

    if fragment.get("id"):
        entry["id"] = fragment["id"]
    if fragment.get("type"):
        entry["type"] = fragment["type"]

    # Anything else the provider sent, preserved as-is. `index` is the
    # accumulator's own key and is not part of a finished tool call.
    for name, value in fragment.items():
        if name in ("index", "id", "type", "function"):
            continue
        if value is not None:
            entry[name] = value

    function = fragment.get("function")
    if isinstance(function, dict):
        if function.get("name"):
            entry["function"]["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            entry["function"]["arguments"] += arguments
