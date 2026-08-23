"""The normalised shape every document is reduced to, whatever it came from.

A PDF page, a DOCX paragraph and a Markdown heading have nothing in common
until something makes them comparable. This is that something: extraction ends
here, and everything downstream - chunking, embedding, citation - reads only
these two types and never has to know which library produced them.

`DocumentElement.type` is the extension point. Text, headings and tables are
handled today; `image` exists with a `path` so that adding figure extraction
later is a new producer and a new branch in the chunker, not a change to the
representation everything else depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Extend by adding a member, a producer that emits it, and a rule in the
# chunker for how it becomes searchable text.
ElementType = Literal["heading", "text", "table", "image"]


@dataclass(frozen=True)
class DocumentElement:
    """One piece of a document, with where it came from."""

    type: ElementType
    # The searchable text. For a table this is a rendered form; for an image it
    # is the caption or the OCR text, and may be empty.
    content: str = ""
    # 1-based, and only for formats that have pages. None everywhere else -
    # inventing a page number for a DOCX paragraph would put a field in a
    # citation that looks authoritative and is not.
    page: int | None = None
    # Headings only: 1 for a title, 2 for a section, and so on.
    level: int | None = None
    # Images only: where the extracted file was written, if it was.
    path: str | None = None

    @property
    def is_heading(self) -> bool:
        return self.type == "heading"

    @property
    def searchable(self) -> bool:
        """Whether this element contributes text worth embedding."""
        return bool(self.content.strip())


@dataclass(frozen=True)
class ExtractedDocument:
    """Everything extraction learned about one file.

    Carries the provenance the tools have to report honestly: whether OCR was
    needed, which pages it was used on, and how much text came out. A caller
    that wants to say "this was a scan" must be able to know it, and must not
    be able to guess it.
    """

    elements: list[DocumentElement] = field(default_factory=list)
    # "pdf", "docx" or "text".
    kind: str = ""
    # Real pages, for formats that have them.
    page_count: int | None = None
    # True when at least one page's text came from OCR rather than the file.
    ocr_used: bool = False
    # Which pages those were, so a citation can be honest about its source.
    ocr_pages: tuple[int, ...] = ()
    # Set when a scan was detected but OCR could not run - the caller reports
    # this rather than silently indexing an almost-empty document.
    ocr_note: str = ""

    @property
    def text(self) -> str:
        """The whole document as plain text. For `read_document`."""
        return "\n\n".join(
            element.content for element in self.elements if element.searchable
        )

    @property
    def characters(self) -> int:
        return sum(len(element.content) for element in self.elements)

    @property
    def tables(self) -> int:
        return sum(1 for element in self.elements if element.type == "table")

    @property
    def headings(self) -> list[str]:
        return [element.content for element in self.elements if element.is_heading]

    def with_elements(self, elements: list[DocumentElement]) -> "ExtractedDocument":
        """A copy carrying different elements. Used by the OCR fallback."""
        return ExtractedDocument(
            elements=elements,
            kind=self.kind,
            page_count=self.page_count,
            ocr_used=self.ocr_used,
            ocr_pages=self.ocr_pages,
            ocr_note=self.ocr_note,
        )


def render_table(rows: list[list[str]]) -> str:
    """Turn a table into text a language model reads correctly.

    Pipe-delimited rather than aligned: alignment spaces are tokens that carry
    no information, and on a machine generating at a few tokens per second
    that is a real cost. The header separator is kept because it is what tells
    the model the first row is headers.
    """
    cleaned = [
        [" ".join(str(cell or "").split()) for cell in row]
        for row in rows
        if any(str(cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    lines = [" | ".join(cleaned[0])]
    if len(cleaned) > 1:
        lines.append(" | ".join("---" for _ in range(width)))
        lines.extend(" | ".join(row) for row in cleaned[1:])
    return "\n".join(lines)
