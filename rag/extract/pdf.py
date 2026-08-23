"""PDFs, via PyMuPDF.

PyMuPDF rather than pypdf for three things pypdf cannot do well:

  * `find_tables()`, so a table in a report survives as a table.
  * `get_pixmap()`, which is what lets a scanned page be rendered and sent to
    OCR - without it the OCR fallback would need a second imaging dependency.
  * Text extraction that keeps reading order on multi-column pages.

**Detecting a scan.** A page with no text layer extracts as an empty string.
That is not the same as a blank page, and the difference decides whether OCR is
worth minutes of CPU. `page_needs_ocr` calls a page scanned when it yields
almost no text *and* the page carries an image large enough to be the content.
Both halves matter: a genuinely blank separator page has no text and no image,
and running OCR on it would cost time and produce nothing.
"""

from __future__ import annotations

from pathlib import Path

from rag.elements import DocumentElement, ExtractedDocument, render_table
from rag.extract.clean import clean_text

# Below this many characters, a page has no usable text layer. A page number
# and a running header come to perhaps twenty; real body text on any page is
# far more.
MIN_PAGE_CHARACTERS = 48

# An image covering less of the page than this is a logo or a figure, not the
# page's content, so its presence does not make the page a scan.
MIN_IMAGE_COVERAGE = 0.35


class PdfError(Exception):
    """A PDF could not be opened or read."""


def _fitz():
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise PdfError(
            "Reading PDFs needs PyMuPDF. Install it with: "
            "pip install -r requirements-documents.txt"
        ) from None
    return fitz


def open_document(path: Path):
    """Open a PDF, decrypting it if it has an empty user password.

    The file is read into memory and handed to PyMuPDF as a stream rather than
    opened by name, for one reason that only shows up on Windows: when
    `fitz.open(path)` raises part-way through parsing a damaged file, the
    handle it already took is never released, and the file stays locked for the
    life of the process. Indexing a folder with one corrupt PDF in it would
    then leave that file undeletable.

    Reading it ourselves means the handle is closed before PyMuPDF sees a byte,
    so a failed parse locks nothing. The cost is holding the file in memory,
    which `extract()` has already capped at RAG_MAX_FILE_BYTES.
    """
    fitz = _fitz()
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PdfError(f"Could not read {path.name}: {exc}") from None

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfError(
            f"{path.name} could not be opened as a PDF "
            f"({type(exc).__name__}: {exc}). It may be corrupt."
        ) from None

    if document.needs_pass:
        # An empty user password is common and opens silently. A real one
        # cannot be guessed, and is reported rather than retried.
        if not document.authenticate(""):
            document.close()
            raise PdfError(
                f"{path.name} is password-protected, so its text cannot be read."
            )
    return document


def page_needs_ocr(page, text: str) -> bool:
    """Whether this page's content is an image rather than text."""
    if len(text.strip()) >= MIN_PAGE_CHARACTERS:
        return False

    try:
        rect = page.rect
        page_area = float(rect.width) * float(rect.height)
        if page_area <= 0:
            return False
        for image in page.get_images(full=True):
            bounds = page.get_image_rects(image[0])
            for box in bounds:
                covered = float(box.width) * float(box.height)
                if covered / page_area >= MIN_IMAGE_COVERAGE:
                    return True
    except Exception:
        # A page whose images cannot be inspected is treated as not-a-scan:
        # OCR is the expensive path, so an unclear signal should not choose it.
        return False
    return False


def extract_pdf(path: Path, *, want_tables: bool = True) -> ExtractedDocument:
    """Read a PDF into elements, one group per page.

    Pages whose text layer is missing come back with no text element and are
    listed in `ocr_pages`; the caller decides whether to run OCR over them.
    """
    document = open_document(path)
    elements: list[DocumentElement] = []
    scanned: list[int] = []

    try:
        total = document.page_count
        for index in range(total):
            number = index + 1
            try:
                page = document.load_page(index)
            except Exception:
                continue  # one damaged page must not lose the rest

            try:
                raw = page.get_text("text") or ""
            except Exception:
                raw = ""

            text = clean_text(raw, join_broken_words=True)

            if page_needs_ocr(page, text):
                scanned.append(number)
                # No element: there is nothing to index yet. OCR fills this in.
                continue

            table_text: list[str] = []
            if want_tables:
                table_text = _tables(page)

            if text.strip():
                elements.append(
                    DocumentElement(type="text", content=text, page=number)
                )
            for rendered in table_text:
                elements.append(
                    DocumentElement(type="table", content=rendered, page=number)
                )
    finally:
        document.close()

    return ExtractedDocument(
        elements=elements,
        kind="pdf",
        page_count=total,
        ocr_pages=tuple(scanned),
    )


def _tables(page) -> list[str]:
    """Rendered tables on one page, or nothing when detection fails.

    Table finding is heuristic and occasionally throws on unusual page
    structures. A missing table costs a little recall; an exception here would
    cost the whole document.
    """
    try:
        found = page.find_tables()
    except Exception:
        return []

    rendered: list[str] = []
    try:
        for table in found.tables:
            try:
                rows = table.extract()
            except Exception:
                continue
            text = render_table([[cell for cell in row] for row in rows])
            # A one-line "table" is almost always a false positive on ruled
            # text, and it duplicates the page text that already covers it.
            if text and text.count("\n") >= 2:
                rendered.append(text)
    except Exception:
        return []
    return rendered


def render_page_png(path: Path, number: int, *, dpi: int = 200) -> bytes:
    """Rasterise one page, for sending to OCR.

    200 dpi is the usable floor for OCR on body text. Higher makes a much
    larger image for the vision model to process, which on this hardware costs
    more than the accuracy is worth.
    """
    document = open_document(path)
    try:
        if not 1 <= number <= document.page_count:
            raise PdfError(f"{path.name} has no page {number}.")
        page = document.load_page(number - 1)
        try:
            pixmap = page.get_pixmap(dpi=dpi)
            return pixmap.tobytes("png")
        except Exception as exc:
            raise PdfError(
                f"Could not render page {number} of {path.name}: {exc}"
            ) from None
    finally:
        document.close()
