"""Normalising extracted text without destroying its structure.

Shared by every format's producer, so a PDF, a DOCX and a Markdown file are
cleaned exactly one way. Blank lines survive because the chunker uses them as
paragraph boundaries, and indentation survives because in a source file it is
the structure.
"""

from __future__ import annotations

import re
import unicodedata


# Zero-width and formatting characters: they carry no meaning but do cost
# tokens and break matching. ZWSP/ZWNJ/ZWJ/LRM/RLM, the line and paragraph
# separators, the BOM, and the soft hyphen PDFs are full of.
#
# Written as codepoints rather than escapes so the source stays pure ASCII
# and no editor or transfer can silently turn an escape into the character
# it names - which is exactly the bug this line is meant to clean up.
_INVISIBLE_CODEPOINTS = (
    0x00AD, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2028, 0x2029, 0xFEFF,
)
_INVISIBLE = re.compile("[" + "".join(map(chr, _INVISIBLE_CODEPOINTS)) + "]")

# A word split across a line break by a hyphen: "mitochon-\ndria". Very common
# in PDFs, and it breaks both search and the model's reading if left in.
_LINE_BROKEN_WORD = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")

_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str, *, join_broken_words: bool = False) -> str:
    """Normalise extracted text without destroying its structure.

    Blank lines survive because the chunker uses them as paragraph boundaries,
    and indentation survives because in a source file it is the structure.

    `join_broken_words` is for PDFs only. Applying it to code would corrupt
    real hyphens at line ends, which is why it is not the default.
    """
    if not text:
        return ""

    # NFKC folds ligatures and the full-width forms PDF extraction produces,
    # so a search for "find" matches a PDF's "fi"-ligature spelling of it.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)

    if join_broken_words:
        text = _LINE_BROKEN_WORD.sub(r"\1\2", text)
        # PDF columns arrive as one line per visual line. Runs of spaces are
        # layout, not content, so they collapse.
        text = _MANY_SPACES.sub(" ", text)

    text = _TRAILING_SPACE.sub("", text)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()
