"""Plain text, Markdown, and source files.

The only structure available here is what the text itself marks. Markdown
headings are real structure and become heading elements, which is what lets a
chunk from a `## Photosynthesis` section carry that section in its metadata.
A `.py` file has no headings, so it produces one text element per paragraph and
no false structure.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag.elements import DocumentElement, ExtractedDocument
from rag.extract.clean import clean_text

# `# Heading`, up to six levels. Requires the space, so a `#!/bin/sh` shebang
# and a `#comment` in a config file are not mistaken for titles.
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")

# Setext headings: a line underlined with === or ---.
_SETEXT = re.compile(r"^(=+|-{3,})\s*$")

_PARAGRAPH = re.compile(r"\n\s*\n")

# Only these get their headings parsed. Treating a '#' comment in Python as a
# section title would fill the metadata with nonsense.
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".rst", ".txt", ".text"})


def extract_text_file(path: Path) -> ExtractedDocument:
    """Read a text or source file into elements."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        raise PermissionError(f"Permission denied: {path}") from None

    body = clean_text(raw)
    if not body.strip():
        return ExtractedDocument(elements=[], kind="text")

    if path.suffix.lower() in MARKDOWN_SUFFIXES:
        elements = _with_headings(body)
    else:
        elements = [
            DocumentElement(type="text", content=block.strip())
            for block in _PARAGRAPH.split(body)
            if block.strip()
        ]

    return ExtractedDocument(elements=elements, kind="text")


def _with_headings(body: str) -> list[DocumentElement]:
    """Split into paragraphs, promoting Markdown headings to heading elements."""
    elements: list[DocumentElement] = []

    for block in _PARAGRAPH.split(body):
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        buffer: list[str] = []

        def flush() -> None:
            joined = "\n".join(buffer).strip()
            if joined:
                elements.append(DocumentElement(type="text", content=joined))
            buffer.clear()

        index = 0
        while index < len(lines):
            line = lines[index]
            match = _ATX_HEADING.match(line.strip())
            if match:
                flush()
                elements.append(
                    DocumentElement(
                        type="heading",
                        content=match.group(2).strip(),
                        level=len(match.group(1)),
                    )
                )
                index += 1
                continue

            # A setext heading is only a heading if there is a line above it to
            # underline, and that line is the one being buffered.
            following = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if line.strip() and _SETEXT.match(following) and not buffer:
                elements.append(
                    DocumentElement(
                        type="heading",
                        content=line.strip(),
                        level=1 if following.startswith("=") else 2,
                    )
                )
                index += 2
                continue

            buffer.append(line)
            index += 1

        flush()

    return elements
