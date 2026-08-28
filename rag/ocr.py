"""Reading scanned pages, through whichever OCR engine is configured.

The rest of the pipeline calls `ocr_pdf_pages(path, pages)` and gets back
`{page_number: text}`. It never learns which engine answered, which is the
whole point: swapping GLM-OCR for Tesseract or a cloud service later means
writing one class here and changing one line of configuration.

`GlmOcrBackend` is the adapter for the OCR server this project already runs.
It reuses `tools.ocr_tool.OcrClient` rather than re-implementing the request,
so the vision check, the payload format, the timeouts and the error messages
stay in one place and cannot drift apart.

The bridge that needs explaining is the file. `OcrClient.ocr_image` takes a
*path inside the workspace*, because that is the jail every other tool
resolves against. A rendered PDF page is bytes in memory. So the page is
written to a temporary file inside the workspace, read, and deleted - the same
bridge `api/routes/uploads.py` makes for browser uploads, for the same reason.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

from rag.extract.pdf import PdfError, render_page_png

# Rendered pages land here, under the workspace, and are removed immediately.
SCRATCH_DIRNAME = ".ocr-scratch"

# What the OCR model is asked for. Deliberately does not mention tables: the
# note in tools/ocr_tool.py records that mentioning them at all makes GLM-OCR
# wrap plain lines in table markup.
PAGE_PROMPT = (
    "Transcribe all text on this page exactly as it appears, preserving the "
    "reading order and line breaks. Output only the text."
)


class OcrUnavailable(Exception):
    """No OCR engine is configured, or the configured one is not reachable."""


class OcrBackend(Protocol):
    """What the extractor needs from an OCR engine."""

    def ocr_pdf_pages(
        self, path: Path, pages: tuple[int, ...] | list[int]
    ) -> dict[int, str]:
        """Return `{page number: text}` for the pages it could read."""


class GlmOcrBackend:
    """The project's existing GLM-OCR server, adapted to whole pages."""

    def __init__(self, client, workspace, *, dpi: int = 200) -> None:
        # `client` is a tools.ocr_tool.OcrClient; `workspace` a WorkspaceFiles.
        # Typed loosely on purpose - importing them here would drag the tools
        # package into every ingest, including the ones with no PDF in them.
        self._client = client
        self._workspace = workspace
        self._dpi = dpi

    def available(self) -> bool:
        """Whether the OCR server is up and has vision loaded."""
        try:
            if not self._client.health():
                return False
        except Exception:
            return False
        # None means "the server did not say", which the client treats as
        # worth trying rather than as a refusal.
        return self._client.supports_images() is not False

    def ocr_pdf_pages(
        self, path: Path, pages: tuple[int, ...] | list[int]
    ) -> dict[int, str]:
        if not pages:
            return {}
        if not self.available():
            raise OcrUnavailable(
                "The OCR server is not reachable, so scanned pages cannot be "
                "read. Turn on OCR in the sidebar and try again."
            )

        scratch = Path(self._workspace.root) / SCRATCH_DIRNAME
        try:
            scratch.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OcrUnavailable(f"Could not create {scratch}: {exc}") from None

        found: dict[int, str] = {}
        try:
            for number in pages:
                text = self._one_page(path, number, scratch)
                if text:
                    found[number] = text
        finally:
            # Best effort: a leftover scratch directory is untidy, not broken.
            try:
                if scratch.is_dir() and not any(scratch.iterdir()):
                    scratch.rmdir()
            except OSError:
                pass
        return found

    def _one_page(self, path: Path, number: int, scratch: Path) -> str:
        try:
            png = render_page_png(path, number, dpi=self._dpi)
        except PdfError:
            return ""  # a page that will not render is one page lost, not all

        target = scratch / f"{uuid.uuid4().hex[:8]}-p{number}.png"
        try:
            target.write_bytes(png)
        except OSError:
            return ""

        try:
            relative = target.relative_to(Path(self._workspace.root)).as_posix()
            result = self._client.ocr_image(relative, PAGE_PROMPT)
            return str(result.get("text", "") or "")
        except Exception:
            # One unreadable page must not abandon the other twenty.
            return ""
        finally:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass


def build_ocr_backend(config) -> OcrBackend | None:
    """The configured OCR backend, or None when OCR is switched off.

    Returning None rather than a backend that always fails is deliberate: the
    extractor reports "OCR is not configured" differently from "OCR tried and
    could not read it", and those have different fixes.
    """
    if not getattr(config, "ocr_enabled", False):
        return None

    from tools.filesystem import WorkspaceFiles
    from tools.ocr_tool import OcrClient

    workspace = WorkspaceFiles(
        config.workspace,
        max_read_bytes=config.max_read_bytes,
        max_write_bytes=config.max_write_bytes,
    )
    return GlmOcrBackend(
        OcrClient(config, workspace),
        workspace,
        dpi=getattr(config, "rag_ocr_dpi", 200),
    )
