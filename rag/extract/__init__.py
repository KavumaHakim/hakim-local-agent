"""Choosing a producer for a file, and applying the OCR fallback.

This is the only module the rest of the application imports for extraction. It
decides three things and delegates everything else:

  * which producer handles the file, by extension;
  * whether the result is thin enough to be a scan;
  * whether OCR should be asked to fill the gaps.

OCR is never run speculatively. A PDF with a text layer is read in
milliseconds; a scanned one costs a vision-model call per page, which on this
hardware is tens of seconds each. So the fallback runs only for pages the PDF
producer positively identified as image-only, and only when a backend is
actually configured.
"""

from __future__ import annotations

from pathlib import Path

from rag.elements import DocumentElement, ExtractedDocument
from rag.extract.clean import clean_text
from rag.extract.docx import DocxError, extract_docx
from rag.extract.pdf import PdfError, extract_pdf
from rag.extract.text import extract_text_file


class ExtractionError(Exception):
    """A document could not be read, or contained no usable text."""


TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        # prose and notes
        ".txt", ".md", ".markdown", ".rst", ".text", ".log",
        # structured text people keep notes in
        ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        # source code
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php",
        ".swift", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
        ".sql", ".jl", ".lua", ".pl",
        # markup and styles
        ".html", ".htm", ".xml", ".css", ".scss", ".sass", ".less",
        ".vue", ".svelte",
    }
)

PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
DOCX_EXTENSIONS: frozenset[str] = frozenset({".docx"})

SUPPORTED_EXTENSIONS: frozenset[str] = (
    TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS
)

# The formats a person is likely to actually upload, named separately so the
# upload endpoint and the UI can offer a short list rather than all fifty-five.
DOCUMENT_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")


def is_supported(path: str | Path) -> bool:
    """Whether this file's extension is one the extractor reads."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def extract(
    path: str | Path,
    *,
    max_bytes: int = 20_000_000,
    ocr=None,
) -> ExtractedDocument:
    """Read `path` into a normalised document.

    `ocr` is anything with `ocr_pdf_pages(path, pages)` returning
    `{page: text}`. When it is None a scanned PDF still succeeds, but comes
    back with `ocr_note` explaining why it is empty rather than pretending the
    file had no content.
    """
    target = Path(path)

    try:
        stat = target.stat()
    except FileNotFoundError:
        raise ExtractionError(f"File does not exist: {target}") from None
    except OSError as exc:
        raise ExtractionError(f"Could not read {target}: {exc}") from None

    if target.is_dir():
        raise ExtractionError(f"{target} is a directory, not a file.")

    suffix = target.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ExtractionError(
            f"{suffix or 'That file'} is not a type this reads. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if stat.st_size == 0:
        raise ExtractionError(f"{target.name} is empty.")
    if stat.st_size > max_bytes:
        raise ExtractionError(
            f"{target.name} is {stat.st_size:,} bytes, over the "
            f"{max_bytes:,}-byte limit. Raise RAG_MAX_FILE_BYTES to index it."
        )

    try:
        if suffix in PDF_EXTENSIONS:
            document = extract_pdf(target)
            document = _fill_scanned_pages(target, document, ocr)
        elif suffix in DOCX_EXTENSIONS:
            document = extract_docx(target)
        else:
            document = extract_text_file(target)
    except (PdfError, DocxError) as exc:
        raise ExtractionError(str(exc)) from None
    except PermissionError as exc:
        raise ExtractionError(str(exc)) from None
    except OSError as exc:
        raise ExtractionError(f"Could not read {target.name}: {exc}") from None

    if not any(element.searchable for element in document.elements):
        raise ExtractionError(_nothing_found(target, document))

    return document


def _fill_scanned_pages(
    path: Path, document: ExtractedDocument, ocr
) -> ExtractedDocument:
    """Run OCR over the pages the PDF producer found no text layer on.

    Elements are re-sorted by page afterwards so a document that was half text
    and half scan still reads in order.
    """
    if not document.ocr_pages:
        return document

    if ocr is None:
        return ExtractedDocument(
            elements=document.elements,
            kind=document.kind,
            page_count=document.page_count,
            ocr_used=False,
            ocr_pages=document.ocr_pages,
            ocr_note=(
                f"{len(document.ocr_pages)} page(s) have no text layer and "
                f"need OCR, which is not configured. Turn on the OCR tool and "
                f"index this again to read them."
            ),
        )

    try:
        recovered = ocr.ocr_pdf_pages(path, document.ocr_pages)
    except Exception as exc:
        return ExtractedDocument(
            elements=document.elements,
            kind=document.kind,
            page_count=document.page_count,
            ocr_used=False,
            ocr_pages=document.ocr_pages,
            # An OCR failure must not fail the document: the pages that did
            # have text are still worth indexing.
            ocr_note=f"OCR could not read the scanned pages: {exc}",
        )

    elements = list(document.elements)
    read: list[int] = []
    for number in document.ocr_pages:
        text = clean_text(recovered.get(number, "") or "", join_broken_words=True)
        if not text.strip():
            continue
        elements.append(DocumentElement(type="text", content=text, page=number))
        read.append(number)

    # Page order, with None last - only text formats produce None, and they
    # never reach this function.
    elements.sort(key=lambda element: (element.page is None, element.page or 0))

    missed = [n for n in document.ocr_pages if n not in read]
    note = ""
    if missed:
        note = (
            f"OCR returned no text for page(s) "
            f"{', '.join(str(n) for n in missed)}."
        )

    return ExtractedDocument(
        elements=elements,
        kind=document.kind,
        page_count=document.page_count,
        ocr_used=bool(read),
        ocr_pages=tuple(read),
        ocr_note=note,
    )


def _nothing_found(target: Path, document: ExtractedDocument) -> str:
    """Say why a document produced nothing, specifically enough to act on."""
    if document.ocr_note:
        return f"{target.name} contains no readable text. {document.ocr_note}"
    if document.kind == "pdf":
        return (
            f"{target.name} contains no extractable text. If it is a scan, "
            f"turn on the OCR tool so its pages can be read."
        )
    return f"{target.name} contains no extractable text."
