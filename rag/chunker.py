"""Splitting extracted text into overlapping chunks worth embedding.

Two rules shape everything here.

**Split on structure, not on a character count.** A chunk that starts
mid-sentence embeds badly, because the embedding is of a fragment whose
meaning is elsewhere. So text is broken into paragraphs, paragraphs into
sentences, and only a single sentence longer than the whole budget is ever cut
by force.

**Never merge across pages.** Two adjacent pages could pack into one chunk and
save a little space, but then `page` in the metadata would be a lie for half of
it, and citing the page is most of the point of indexing a PDF.

Sizes are configured in tokens because that is the unit the embedding model
truncates on, but counted in characters, because counting tokens would mean
loading the tokenizer - and ingestion has to be able to plan its work without
the model in memory. `CHARS_PER_TOKEN` is the conversion, set low enough to
stay conservative: over-estimating tokens makes chunks smaller, which is safe,
while under-estimating would let the model silently truncate their tails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.elements import DocumentElement, ExtractedDocument

# BGE-small-en-v1.5 truncates at 512 tokens. Chunks are planned against a
# budget below that so a bad estimate costs a little space rather than losing
# the end of a chunk without saying so.
MODEL_MAX_TOKENS = 512

# English prose runs about 4.2 characters per token; source code, dense with
# punctuation and short identifiers, closer to 3. The lower figure is used for
# everything, so prose chunks come out a little under budget and code chunks
# still fit.
CHARS_PER_TOKEN = 3.5

DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 75

# A chunk this short carries no retrievable meaning - a stray heading, a page
# number, the tail of a split paragraph. Indexing it only adds noise to the
# results.
MIN_CHUNK_CHARS = 40

_PARAGRAPH = re.compile(r"\n\s*\n")

# End of sentence: . ! ? or their closing-quote forms, followed by whitespace
# and something that starts a new sentence. Deliberately not a full sentence
# tokeniser - the cost of an occasional bad split here is one slightly odd
# chunk boundary, not a wrong answer.
_SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]*\s+(?=["\'(\[]*[A-Z0-9])')


@dataclass(frozen=True)
class Chunk:
    """One unit of text to embed, with where it came from."""

    text: str
    # Position within the document, from 0. Part of the chunk's stable id.
    ordinal: int
    # 1-based page for PDFs, None for formats without pages.
    page: int | None = None
    # The heading this chunk sits under, when the format had one. Carried into
    # the search result so a citation can say "under Fire Safety" for a DOCX,
    # which has no page number to give instead.
    section: str | None = None


def token_budget(tokens: int) -> int:
    """Characters allowed for a target of `tokens` tokens."""
    capped = max(1, min(int(tokens), MODEL_MAX_TOKENS))
    return max(1, int(capped * CHARS_PER_TOKEN))


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split an extracted document into overlapping chunks.

    Elements are grouped into runs that share a page and a section, and each
    run is packed independently. That is what keeps `page` and `section`
    truthful: a chunk never spans two pages, so its citation is exact.

    Headings are not chunks of their own - a heading alone embeds to almost
    nothing useful. Instead a heading opens a new run and is prepended to the
    text of the chunks under it, which gives those chunks the context a reader
    would have had from the page.
    """
    size = token_budget(chunk_tokens)
    # Overlap has to leave room for new text, or chunking cannot advance.
    overlap = max(0, min(token_budget(overlap_tokens), size // 2))

    chunks: list[Chunk] = []
    ordinal = 0
    for page, section, body in _runs(document):
        for text in _split_one(body, size=size, overlap=overlap):
            chunks.append(
                Chunk(text=text, ordinal=ordinal, page=page, section=section)
            )
            ordinal += 1
    return chunks


def _runs(document: ExtractedDocument) -> list[tuple[int | None, str | None, str]]:
    """(page, section, text) for each run of same-page, same-section text.

    A run ends when the page changes or a heading starts a new section, which
    are exactly the boundaries a citation has to respect.
    """
    runs: list[tuple[int | None, str | None, str]] = []
    section: str | None = None
    page: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        """Close the current run. Plain function, so the buffer clears now."""
        joined = "\n\n".join(buffer).strip()
        if joined:
            runs.append((page, section, joined))
        buffer.clear()

    for element in document.elements:
        if element.is_heading:
            flush()
            section = element.content.strip() or None
            page = element.page
            # The heading itself leads the section's first chunk rather than
            # becoming a chunk of its own: a heading alone embeds to almost
            # nothing, but as the opening line of the text beneath it, it is
            # exactly the context a reader would have had.
            buffer.append(element.content.strip())
            continue

        if not element.searchable:
            continue

        if buffer and element.page != page:
            flush()
        if not buffer:
            page = element.page

        buffer.append(element.content.strip())

    flush()
    return runs


def _split_one(text: str, *, size: int, overlap: int) -> list[str]:
    """Split one run of text into chunks of at most `size` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    units = _units(text, size=size)

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in units:
        # +1 for the separator that will join them.
        addition = len(unit) + (1 if current else 0)
        if current and length + addition > size:
            chunks.append(" ".join(current).strip())
            current, length = _carry_over(current, overlap)
            addition = len(unit) + (1 if current else 0)
        current.append(unit)
        length += addition

    if current:
        tail = " ".join(current).strip()
        # A final scrap shorter than the minimum is appended to the previous
        # chunk instead of becoming a chunk of its own. Dropping it would lose
        # the end of the document.
        if chunks and len(tail) < MIN_CHUNK_CHARS:
            chunks[-1] = f"{chunks[-1]} {tail}".strip()
        elif tail:
            chunks.append(tail)

    return [chunk for chunk in chunks if chunk.strip()]


def _carry_over(units: list[str], overlap: int) -> tuple[list[str], int]:
    """The trailing units of a finished chunk that seed the next one.

    Overlap is what stops a fact that straddles a boundary from being
    unfindable in either chunk.
    """
    if overlap <= 0:
        return [], 0

    carried: list[str] = []
    length = 0
    for unit in reversed(units):
        addition = len(unit) + (1 if carried else 0)
        if length + addition > overlap:
            break
        carried.insert(0, unit)
        length += addition
    return carried, length


def _units(text: str, *, size: int) -> list[str]:
    """Break text into the smallest pieces chunking is allowed to reorder.

    Paragraphs first, because a paragraph is a complete thought and a blank
    line is the one boundary every format agrees on. Only paragraphs that do
    not fit are broken into sentences, and only sentences that still do not fit
    are cut by force.
    """
    units: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
            continue

        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= size:
                units.append(sentence)
            else:
                units.extend(_hard_wrap(sentence, size))
    return units


def _hard_wrap(text: str, size: int) -> list[str]:
    """Last resort: cut on whitespace, and mid-word only if there is none.

    Reached by minified JavaScript, a CSV row with no spaces, and PDF pages
    that extract as one unbroken line.
    """
    pieces: list[str] = []
    remaining = text
    while len(remaining) > size:
        window = remaining[:size]
        cut = window.rfind(" ")
        # Only break on a space if it is reasonably far in; a space at
        # character 3 of a 1750-character window would produce a useless chunk.
        if cut < size // 2:
            cut = size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [piece for piece in pieces if piece]
