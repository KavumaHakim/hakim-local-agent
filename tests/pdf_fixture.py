"""A minimal PDF writer, so the PDF tests need no extra dependency.

reportlab would be a 10 MB install to produce a file the tests then read back,
and matplotlib embeds subset fonts whose extracted text is not reliably the
text that went in. A hand-built PDF using the base-14 Helvetica keeps the bytes
under a hundred lines and makes the page numbers exact, which is the thing the
page metadata tests actually care about.

Only what pypdf needs to find the text: a catalog, a page tree, one content
stream per page, and a correct cross-reference table.
"""

from __future__ import annotations

from pathlib import Path

# Points. US Letter, which is what pypdf assumes when a page omits MediaBox.
PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 72
TOP_BASELINE = 720
LINE_HEIGHT = 14


def _escape(text: str) -> str:
    """Escape a PDF literal string. Backslash first, or it escapes itself."""
    for character, replacement in (("\\", "\\\\"), ("(", "\\("), (")", "\\)")):
        text = text.replace(character, replacement)
    return text


def _content_stream(lines: list[str]) -> bytes:
    """The drawing operators for one page: a text block, one Tj per line."""
    parts = ["BT", "/F1 12 Tf", f"{LEFT_MARGIN} {TOP_BASELINE} Td"]
    for position, line in enumerate(lines):
        if position:
            parts.append(f"0 -{LINE_HEIGHT} Td")
        parts.append(f"({_escape(line)}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("ascii", errors="replace")


def write_pdf(path: str | Path, pages: list[list[str]]) -> Path:
    """Write a PDF where `pages[i]` is the list of text lines on page i + 1."""
    if not pages:
        raise ValueError("A PDF needs at least one page.")

    target = Path(path)

    # Object numbering: 1 catalog, 2 page tree, 3 font, then a page object and
    # a content object per page.
    page_ids = [4 + index * 2 for index in range(len(pages))]
    content_ids = [identifier + 1 for identifier in page_ids]

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{identifier} 0 R" for identifier in page_ids)
            + f"] /Count {len(pages)} >>"
        ).encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for index, lines in enumerate(pages):
        stream = _content_stream(lines)
        objects[page_ids[index]] = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_ids[index]} 0 R >>"
        ).encode("ascii")
        objects[content_ids[index]] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )

    # Body, recording where each object starts: the xref table is byte offsets,
    # and a wrong one makes the file unreadable.
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for identifier in sorted(objects):
        offsets[identifier] = len(out)
        out += f"{identifier} 0 obj\n".encode("ascii")
        out += objects[identifier]
        out += b"\nendobj\n"

    highest = max(objects)
    xref_at = len(out)
    out += f"xref\n0 {highest + 1}\n".encode("ascii")
    # Object 0 is always the head of the free list.
    out += b"0000000000 65535 f \n"
    for identifier in range(1, highest + 1):
        if identifier in offsets:
            out += f"{offsets[identifier]:010d} 00000 n \n".encode("ascii")
        else:
            out += b"0000000000 65535 f \n"

    out += f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\n".encode("ascii")
    out += f"startxref\n{xref_at}\n%%EOF\n".encode("ascii")

    target.write_bytes(bytes(out))
    return target


def add_outline(path: str | Path, entries: list[tuple[int, str, int]]) -> Path:
    """Attach a table of contents to a PDF written by `write_pdf`.

    `entries` is (level, title, page), the shape PyMuPDF uses.

    Built with fitz rather than by hand, unlike the rest of this file. An
    outline is a linked tree of objects with /First, /Last, /Count and /Parent
    all having to agree, which is a great deal of fragile plumbing to write in
    order to test the code that *reads* one. Letting fitz write it still leaves
    the part under test - turning a table of contents into headings, and
    headings into chunk sections - entirely ours.
    """
    import fitz

    target = Path(path)
    document = fitz.open(target)
    try:
        document.set_toc([[level, title, page] for level, title, page in entries])
        document.save(str(target), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    finally:
        document.close()
    return target


def add_ruled_table(path: str | Path, page_number: int = 1) -> Path:
    """Draw a lined table on one page of an existing PDF.

    Ruled rather than whitespace-aligned on purpose: PyMuPDF's table finder
    defaults to `strategy="lines"`, so a drawn grid is the only kind it ever
    detects, and it is therefore the only kind worth testing against.
    """
    import fitz

    target = Path(path)
    document = fitz.open(target)
    try:
        page = document[page_number - 1]
        top, left, rows, columns, height, width = 100, 72, 4, 3, 20, 120
        for row in range(rows + 1):
            page.draw_line(
                (left, top + row * height),
                (left + columns * width, top + row * height),
            )
        for column in range(columns + 1):
            page.draw_line(
                (left + column * width, top),
                (left + column * width, top + rows * height),
            )
        for row in range(rows):
            for column in range(columns):
                page.insert_text(
                    (left + column * width + 5, top + row * height + 14),
                    f"r{row}c{column}",
                )
        document.save(str(target), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    finally:
        document.close()
    return target


def add_raster_figure(
    path: str | Path,
    page_number: int = 1,
    *,
    size: int = 300,
    box: tuple[float, float, float, float] = (72, 110, 372, 310),
) -> Path:
    """Embed a raster image on one page of an existing PDF.

    A generated gradient rather than a real photograph: what the tests care
    about is that an image object of a given pixel size is present, so the
    extractor can be asked whether it treats it as a figure or as furniture.
    """
    import fitz

    target = Path(path)
    document = fitz.open(target)
    try:
        page = document[page_number - 1]
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, size, size), False)
        for x in range(size):
            for y in range(size):
                pixmap.set_pixel(x, y, (x % 256, y % 256, (x + y) % 256))
        page.insert_image(fitz.Rect(*box), pixmap=pixmap)
        document.save(str(target), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    finally:
        document.close()
    return target
