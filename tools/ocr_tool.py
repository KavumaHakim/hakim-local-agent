"""OCR tool backed by a GLM-OCR llama.cpp server.

VERIFIED against a running server (GLM-OCR-Q8_0 + mmproj-GLM-OCR-Q8_0):
the endpoint below returns correct transcriptions, /props reports
{"vision": true, "video": true, "audio": false}, and a 760x300 note takes
about 30 seconds on this machine.

llama.cpp serves vision models through mtmd, and its OpenAI-compatible endpoint
takes images as content parts on a user message:

    {"role": "user", "content": [
        {"type": "text",      "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]}

One quirk worth knowing: GLM-OCR emits tables as HTML (<table><tr><td>)
whatever format you ask for. The data is accurate; the markup is not
negotiable. The default prompt avoids mentioning tables at all, because doing
so makes the model wrap even plain lines in table markup.

The tool refuses early and clearly when the server is not multimodal, so a
missing projector produces an explanation rather than a puzzling failure.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import requests

from config import Config
from tools.base import Tool, ToolError
from tools.filesystem import WorkspaceFiles

# Tuned against the running model. Mentioning tables at all makes GLM-OCR wrap
# even plain lines in <table> markup, so this prompt does not - and asks for
# plain text explicitly, which reliably produces clean line-by-line output.
DEFAULT_PROMPT = (
    "Read this image and return the text exactly as it appears, one line per "
    "line. Plain text only: no HTML, no markdown, no commentary."
)

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


class OcrError(ToolError):
    """An OCR request was rejected or failed."""


class OcrClient:
    """Validates image paths and sends them to the GLM-OCR server."""

    def __init__(self, config: Config, workspace: WorkspaceFiles) -> None:
        self._config = config
        self._workspace = workspace
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self._config.ocr_url

    # --- server checks ---

    def health(self) -> bool:
        """True if the OCR server is up. Never raises."""
        try:
            response = self._session.get(
                f"{self._config.ocr_url}/health",
                timeout=self._config.connect_timeout,
            )
        except requests.RequestException:
            return False
        return response.status_code == 200

    def properties(self) -> dict[str, Any]:
        """Whatever /props reports, or an empty dict if unreachable."""
        try:
            response = self._session.get(
                f"{self._config.ocr_url}/props",
                timeout=self._config.connect_timeout,
            )
            if response.status_code != 200:
                return {}
            data = response.json()
        except (requests.RequestException, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def supports_images(self) -> bool | None:
        """True/False if /props says so, None if it does not mention it.

        llama.cpp has reported this under more than one key over time, so an
        unknown answer is returned as None and treated as "try anyway" rather
        than blocking a server that would have worked.
        """
        props = self.properties()
        if not props:
            return None

        modalities = props.get("modalities")
        if isinstance(modalities, dict) and "vision" in modalities:
            return bool(modalities["vision"])
        if isinstance(modalities, list):
            return "vision" in modalities
        for key in ("has_multimodal", "multimodal", "has_mtmd"):
            if key in props:
                return bool(props[key])
        return None

    # --- validation ---

    def validate(self, path: str) -> Path:
        """Check the path is a readable image inside the workspace."""
        try:
            # Reuses the workspace jail, so OCR cannot reach files the
            # filesystem tool could not.
            target = self._workspace.resolve(path)
        except ToolError as exc:
            raise OcrError(str(exc)) from None

        if not target.exists():
            raise OcrError(f"File does not exist: {path}")
        if target.is_dir():
            raise OcrError(f"{path} is a directory, not an image.")

        suffix = target.suffix.lower()
        allowed = self._config.ocr_allowed_extensions
        if suffix not in allowed:
            raise OcrError(
                f"Unsupported file type {suffix or '(none)'}. Allowed: "
                f"{', '.join(allowed)}."
            )

        try:
            size = target.stat().st_size
        except OSError as exc:
            raise OcrError(f"Could not stat file: {exc}") from None

        if size == 0:
            raise OcrError(f"{path} is empty.")
        if size > self._config.ocr_max_image_bytes:
            raise OcrError(
                f"Image is {size} bytes, over the "
                f"{self._config.ocr_max_image_bytes} byte limit."
            )

        return target

    # --- the tool entry point ---

    def ocr_image(self, path: str, prompt: str | None = None) -> dict[str, Any]:
        """Extract text from an image inside the workspace."""
        target = self.validate(path)
        text = self._send(target, prompt or DEFAULT_PROMPT)
        return {
            "success": True,
            "path": path,
            "text": text,
            "characters": len(text),
        }

    def _data_uri(self, path: Path) -> str:
        mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise OcrError(f"Could not read image: {exc}") from None
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    def build_payload(self, path: Path, prompt: str) -> dict[str, Any]:
        """The request body. Separated so tests can inspect it."""
        return {
            "model": self._config.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._data_uri(path)},
                        },
                    ],
                }
            ],
            # OCR should transcribe, not invent.
            "temperature": 0.1,
            "stream": False,
        }

    def _send(self, path: Path, prompt: str) -> str:
        if self.supports_images() is False:
            raise OcrError(
                f"The server at {self._config.ocr_url} has no vision support "
                f"loaded. Start it with --mmproj pointing at the GLM-OCR "
                f"projector file."
            )

        payload = self.build_payload(path, prompt)
        url = f"{self._config.ocr_url}/v1/chat/completions"

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=(self._config.connect_timeout, self._config.request_timeout),
            )
        except requests.Timeout:
            raise OcrError(
                f"No response from the OCR server within "
                f"{self._config.request_timeout:.0f}s."
            ) from None
        except requests.ConnectionError:
            raise OcrError(
                f"Could not reach the OCR server at {self._config.ocr_url}. "
                f"Is llama-server running there with --mmproj?"
            ) from None
        except requests.RequestException as exc:
            raise OcrError(f"OCR request failed: {exc}") from None

        if response.status_code >= 400:
            detail = response.text[:300]
            # A 400 here is usually one of two things: the file is not a valid
            # image, or the server has no projector loaded. Name both rather
            # than guessing - the wrong hint sends you looking in the wrong
            # place, which is worse than no hint.
            hint = (
                " Either the file is not a readable image, or the server was "
                "started without --mmproj and cannot accept images."
                if response.status_code in (400, 500)
                else ""
            )
            raise OcrError(f"OCR server returned HTTP {response.status_code}: {detail}{hint}")

        try:
            data = response.json()
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise OcrError(
                f"Could not read the OCR server's reply: {response.text[:300]}"
            ) from None

        if not isinstance(text, str) or not text.strip():
            raise OcrError("The OCR server returned no text for that image.")
        return text.strip()

    def tool(self) -> Tool:
        return Tool(
            name="ocr_image",
            category="ocr",
            description=(
                "Read the text in an image or scanned document inside the "
                "workspace. Use for photos, scans, screenshots, tables and "
                "handwriting. Give a path relative to the workspace root. "
                "Optionally say what to extract. Tables come back as HTML."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Image file, relative to the workspace root.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Optional instruction, e.g. 'extract the table as "
                            "markdown'."
                        ),
                    },
                },
                "required": ["path"],
            },
            run=self.ocr_image,
        )
