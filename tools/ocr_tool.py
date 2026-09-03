"""OCR with a choice of backend: the GLM-OCR model, or Tesseract.

`OCR_BACKEND` picks between them, and they are genuinely different trades
rather than one being better:

                      RAM        one page      understands layout
    GLM-OCR         ~1.4 GB      ~30 s         yes - tables, columns, headings
    Tesseract       ~50 MB       <1 s          no - lines of text, in order

Tesseract is the default because most OCR here is "read the text off this
screenshot", and paying 1.4 GB and half a minute for that on an 8 GB machine is
a poor trade. GLM-OCR earns its cost on a page whose structure matters.

Everything above the backend is shared: the same workspace jail, the same size
and extension checks, and one `ocr_image` tool whose description changes to
match whichever backend is active - so the model is told what it is actually
getting.

The model backend is documented below.


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
from urllib.parse import urlsplit

import requests

from config import Config
from tools.base import Tool, ToolError
from tools.filesystem import WorkspaceFiles
from tools.tesseract import TesseractBackend, TesseractError

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


def _is_loopback(url: str) -> bool:
    """Whether `url` names this machine, where a proxy can only get in the way."""
    host = (urlsplit(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


class OcrClient:
    """Validates image paths and sends them to the GLM-OCR server."""

    def __init__(
        self,
        config: Config,
        workspace: WorkspaceFiles,
        tesseract: TesseractBackend | None = None,
    ) -> None:
        self._config = config
        # Injectable, so the tests can point at a stub binary rather than
        # needing Tesseract installed to exercise the dispatch.
        self._tesseract = tesseract or TesseractBackend(
            command=config.tesseract_cmd,
            language=config.tesseract_lang,
            psm=config.tesseract_psm,
            timeout=config.ocr_timeout,
        )
        self._workspace = workspace
        self._session = requests.Session()
        # Loopback bypasses the proxy environment, for the reason recorded on
        # the manager's session in models/manager.py: with a system proxy set
        # and no NO_PROXY, a request to 127.0.0.1 never reaches it.
        self._session.trust_env = not _is_loopback(self._config.ocr_url)

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

    @property
    def backend(self) -> str:
        """Which backend is active: "tesseract" or "model"."""
        return "tesseract" if self._config.ocr_backend == "tesseract" else "model"

    def backend_ready(self) -> tuple[bool, str]:
        """Whether the active backend can actually run, and why not."""
        if self.backend == "tesseract":
            if self._tesseract.available():
                return True, ""
            return False, self._tesseract.missing_message()
        if not self.health():
            return False, (
                f"The GLM-OCR server is not answering on {self._config.ocr_url}. "
                f"Start it, or switch OCR to Tesseract, which needs no server."
            )
        return True, ""

    def ocr_image(self, path: str, prompt: str | None = None) -> dict[str, Any]:
        """Extract text from an image inside the workspace.

        The path is validated the same way whichever backend reads it: the
        workspace jail and the size limit are properties of the tool, not of
        the reader behind it.
        """
        target = self.validate(path)

        if self.backend == "tesseract":
            try:
                text = self._tesseract.read(target)
            except TesseractError as exc:
                raise OcrError(str(exc)) from None
            return {
                "success": True,
                "path": path,
                "text": text,
                "characters": len(text),
                "backend": "tesseract",
                # Said plainly, because a prompt silently doing nothing is the
                # kind of thing that wastes an afternoon.
                **(
                    {"note": (
                        "Tesseract transcribes only; the prompt was ignored. "
                        "Switch OCR to the GLM-OCR model to direct extraction."
                    )}
                    if prompt
                    else {}
                ),
            }

        text = self._send(target, prompt or DEFAULT_PROMPT)
        return {
            "success": True,
            "path": path,
            "text": text,
            "characters": len(text),
            "backend": "model",
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

    def _description(self) -> str:
        """What the model is told, which depends on the backend.

        The two behave differently enough that one description would be wrong
        for whichever is running: Tesseract cannot follow an instruction and
        cannot see a table, and telling the model otherwise wastes a round
        trip and produces a confident wrong answer about the result.
        """
        common = (
            "Read the text in an image or scanned document inside the "
            "workspace. Give a path relative to the workspace root. "
        )
        if self.backend == "tesseract":
            return common + (
                "Backed by Tesseract: fast and cheap, and it transcribes text "
                "line by line. It does not follow instructions and does not "
                "preserve tables or columns, so do not ask it to extract a "
                "particular field - read the text and pick the field yourself. "
                "Best on screenshots and clean scans; weak on photographs and "
                "handwriting."
            )
        return common + (
            "Backed by the GLM-OCR vision model: slower and heavier, but it "
            "understands layout. Use for photos, scans, tables and "
            "handwriting, and say what to extract if you want something "
            "specific. Tables come back as HTML."
        )

    def tool(self) -> Tool:
        return Tool(
            name="ocr_image",
            category="ocr",
            description=self._description(),
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
                            "markdown'. Ignored by the Tesseract backend, "
                            "which only transcribes."
                        ),
                    },
                },
                "required": ["path"],
            },
            run=self.ocr_image,
        )
