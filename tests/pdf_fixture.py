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
