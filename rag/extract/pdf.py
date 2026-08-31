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

import re
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

# A raster image smaller than this in either direction is furniture - a bullet,
# a rule, a logo, an icon in a running header. Extracting them would bury the
# real figures in noise.
MIN_FIGURE_PIXELS = 120

# And one covering less of the page than this is decoration rather than a
# figure someone would refer to.
MIN_FIGURE_COVERAGE = 0.02

# How far below an image to look for its caption, as a fraction of page height.
# Captions sit directly under the figure; body text further down belongs to the
# page, not to the picture.
CAPTION_GAP = 0.08

# What a caption looks like. Deliberately narrow: claiming a paragraph of body
# prose is a figure's caption would be worse than admitting there is none.
CAPTION_PATTERN = re.compile(
    r"^\s*(figure|fig\.?|chart|graph|diagram|illustration|plate|exhibit|scheme)"
    r"\s*[\d.:\-]*\s*\S",
    re.IGNORECASE,
)


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


def extract_pdf(
    path: Path,
    *,
    want_tables: bool = True,
    figure_dir: Path | None = None,
) -> ExtractedDocument:
    """Read a PDF into elements, one group per page.

    Pages whose text layer is missing come back with no text element and are
    listed in `ocr_pages`; the caller decides whether to run OCR over them.

    With `figure_dir`, embedded raster images large enough to be figures are
    written there as PNG and reported as `image` elements carrying their
    caption. Without it no images are touched, which is what every caller that
    only wants text should pass.
    """
    document = open_document(path)
    elements: list[DocumentElement] = []
    scanned: list[int] = []

    try:
        total = document.page_count
        outline = _outline(document)
        for index in range(total):
            number = index + 1
            # The book's own table of contents, turned into headings the
            # chunker already knows what to do with. Without this a PDF is the
            # one format that produces no structure at all, which is backwards:
            # a textbook is exactly where knowing the chapter matters most.
            for level, title in outline.get(number, ()):
                elements.append(
                    DocumentElement(type="heading", content=title, page=number, level=level)
                )
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
            if want_tables and _may_hold_table(page):
                table_text = _tables(page)

            if text.strip():
                elements.append(
                    DocumentElement(type="text", content=text, page=number)
                )
            for rendered in table_text:
                elements.append(
                    DocumentElement(type="table", content=rendered, page=number)
                )

            if figure_dir is not None:
                elements.extend(_figures(document, page, number, figure_dir))
    finally:
        document.close()

    return ExtractedDocument(
        elements=elements,
        kind="pdf",
        page_count=total,
        ocr_pages=tuple(scanned),
    )


def _outline(document) -> dict[int, list[tuple[int, str]]]:
    """The PDF's own table of contents, keyed by the page each entry starts on.

    PyMuPDF returns `[level, title, page]` rows straight from the file's
    bookmarks, so this is the author's structure rather than a guess made from
    font sizes.

    The limitation is worth stating: a bookmark points at a *page*, not at a
    position on it. A section therefore begins at the top of its page, and a
    page holding the end of one section and the start of the next is credited
    entirely to the new one. Getting that exactly right means reasoning about
    text coordinates, which is a great deal of work to move a boundary by a
    paragraph.

    Not every PDF has bookmarks. One without simply gets no headings, exactly
    as before.
    """
    try:
        rows = document.get_toc(simple=True)
    except Exception:
        # get_toc is not universally implemented across the formats fitz will
        # open, and a missing outline costs structure rather than content.
        return {}

    found: dict[int, list[tuple[int, str]]] = {}
    for row in rows or ():
        try:
            level, title, page = int(row[0]), str(row[1]).strip(), int(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        # Page 0 or -1 means the entry points at nothing resolvable.
        if page < 1 or not title:
            continue
        found.setdefault(page, []).append((max(1, level), title))
    return found


def _figures(document, page, number: int, figure_dir: Path) -> list[DocumentElement]:
    """Raster images on one page that are big enough to be figures.

    Each is written as a PNG and returned as an `image` element whose content
    is its caption, when one can be found. A figure with no caption still gets
    an element - the file is what a vision model would later read - but its
    empty content means it contributes nothing to the index, which is correct:
    there is nothing yet to search for.

    Vector artwork is not covered. A chart drawn as lines and rectangles is not
    an image at all as far as the file is concerned, and pulling one out means
    rendering a region of the page rather than extracting an object.
    """
    try:
        rect = page.rect
        page_area = float(rect.width) * float(rect.height)
        images = page.get_images(full=True)
    except Exception:
        return []
    if page_area <= 0 or not images:
        return []

    blocks = _text_blocks(page)
    found: list[DocumentElement] = []

    for index, image in enumerate(images):
        xref = image[0]
        try:
            boxes = page.get_image_rects(xref)
        except Exception:
            continue
        if not boxes:
            continue
        box = boxes[0]

        covered = float(box.width) * float(box.height)
        if covered / page_area < MIN_FIGURE_COVERAGE:
            continue

        try:
            pixmap = _fitz().Pixmap(document, xref)
        except Exception:
            continue
        try:
            if pixmap.width < MIN_FIGURE_PIXELS or pixmap.height < MIN_FIGURE_PIXELS:
                continue
            # CMYK and other exotic colour spaces cannot be written as PNG
            # directly; converting is cheaper than losing the figure.
            if pixmap.n - pixmap.alpha >= 4:
                pixmap = _fitz().Pixmap(_fitz().csRGB, pixmap)
            target = figure_dir / f"page{number:04d}-{index}.png"
            try:
                figure_dir.mkdir(parents=True, exist_ok=True)
                pixmap.save(str(target))
            except Exception:
                continue
        finally:
            pixmap = None

        found.append(
            DocumentElement(
                type="image",
                content=_caption(box, blocks, float(rect.height)),
                page=number,
                path=str(target),
            )
        )
    return found


def _text_blocks(page) -> list[tuple[float, float, float, float, str]]:
    """(x0, y0, x1, y1, text) for the page's text blocks."""
    try:
        raw = page.get_text("blocks") or []
    except Exception:
        return []
    blocks = []
    for entry in raw:
        try:
            x0, y0, x1, y1, body = entry[0], entry[1], entry[2], entry[3], entry[4]
        except (IndexError, TypeError):
            continue
        if isinstance(body, str) and body.strip():
            blocks.append((float(x0), float(y0), float(x1), float(y1), body.strip()))
    return blocks


def _caption(box, blocks, page_height: float) -> str:
    """The caption below an image, if there is one that looks like a caption.

    Below rather than above because that is where captions overwhelmingly sit,
    and horizontally overlapping the figure so a caption in the next column is
    not stolen. Only text matching `CAPTION_PATTERN` is accepted: an empty
    caption is honest, and calling a paragraph of body prose one is not.
    """
    limit = float(box.y1) + page_height * CAPTION_GAP
    best: tuple[float, str] | None = None

    for x0, y0, x1, y1, body in blocks:
        if y0 < float(box.y1) or y0 > limit:
            continue
        # Some horizontal overlap with the figure, or it belongs to something
        # else on the page.
        if x1 < float(box.x0) or x0 > float(box.x1):
            continue
        if not CAPTION_PATTERN.match(body):
            continue
        distance = y0 - float(box.y1)
        if best is None or distance < best[0]:
            best = (distance, " ".join(body.split()))

    return best[1] if best else ""


def _may_hold_table(page) -> bool:
    """Whether table detection could possibly find anything on this page.

    `find_tables` defaults to `vertical_strategy="lines"` and
    `horizontal_strategy="lines"`, so it only ever finds tables delimited by
    drawn lines. A page with no vector drawings therefore cannot yield one,
    and asking is a waste of the most expensive call in extraction.

    Measured on a 120-page prose book: detection took 16.06 s and found
    nothing, where extraction without it took 0.34 s. Checking for drawings
    first costs 0.10 s across the same book - a 47x saving on exactly the kind
    of document someone indexes a book for.

    This is a filter, not a heuristic: it removes only pages where the
    detector's own strategy guarantees no result. A borderless table was never
    found by this configuration and still is not; its text is still extracted
    and indexed as text either way.
    """
    try:
        return bool(page.get_drawings())
    except Exception:
        # If the page will not say, let the real detector decide. Being wrong
        # here costs time, and being wrong the other way costs a table.
        return True


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
