"""Document search: extraction, chunking, the index, and the agent tool.

No embedding model. `HashingEmbedder` stands in for BGE and produces real,
comparable vectors from a bag of words, so retrieval can be asserted on
meaning-ish grounds without a 130 MB download and eighty seconds of import in
every run. The tests that need the real model live in `RealModelTests` and are
skipped unless RAG_MODEL_TESTS=1 asks for them.

What the fake cannot check is whether BGE is any good, and that is fine - that
is the model's job, not this code's. What these check is the part that is ours:
that a chunk keeps its page, that re-indexing does not duplicate, that a freed
row is reused and never returned by a search, and that compaction does not
scramble which vector belongs to which chunk.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import Config
from rag.chunker import CHARS_PER_TOKEN, chunk_document, token_budget
from rag.elements import DocumentElement, ExtractedDocument
from rag.extract import ExtractionError, clean_text, extract, is_supported
from rag.extract.pdf import extract_pdf
from rag.index import VectorIndex, VectorIndexError
from rag.manager import RagError, RagManager
from rag.metadata import MetadataStore
from tests.pdf_fixture import (
    add_outline,
    add_raster_figure,
    add_ruled_table,
    write_pdf,
)
from tools.base import ToolRegistry
from tools.document_search import DocumentSearchError, DocumentSearchTools
from tools.registry import build_default_registry

DIMENSION = 384


def text_document(*elements: DocumentElement, kind: str = "text") -> ExtractedDocument:
    """An ExtractedDocument built inline, for the chunker tests."""
    return ExtractedDocument(elements=list(elements), kind=kind)


def paragraph(content: str, page: int | None = None) -> DocumentElement:
    return DocumentElement(type="text", content=content, page=page)


def heading(content: str, page: int | None = None, level: int = 1) -> DocumentElement:
    return DocumentElement(type="heading", content=content, page=page, level=level)


class HashingEmbedder:
    """A deterministic stand-in for the real embedder.

    Each word is hashed to a dimension and counted, then the vector is
    normalised - so two texts sharing words score high and two that share none
    score near zero. That is enough structure to test retrieval, ordering and
    thresholds without loading anything.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension
        self.loaded = False
        self.passages_encoded = 0
        self.queries_encoded = 0
        self.unloads = 0

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().split():
            word = word.strip(".,;:()[]!?\"'")
            if not word:
                continue
            digest = hashlib.md5(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            # An all-punctuation chunk still needs a unit vector, or the dot
            # product stops being a cosine.
            vector[0] = 1.0
            norm = 1.0
        return vector / norm

    def encode_passages(self, texts, progress=None) -> np.ndarray:
        texts = list(texts)
        self.passages_encoded += len(texts)
        self.loaded = True
        if progress is not None:
            progress(len(texts), len(texts))
        return np.vstack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        self.queries_encoded += 1
        self.loaded = True
        return self._vector(text)

    def unload(self) -> bool:
        self.unloads += 1
        was = self.loaded
        self.loaded = False
        return was


class TempCase(unittest.TestCase):
    """A temporary directory for the store and for sample documents."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.docs = self.tmp / "docs"
        self.docs.mkdir()
        self.embedder = HashingEmbedder()

    def tearDown(self):
        self._tmp.cleanup()

    def manager(self, store_dir_name: str = "store", **overrides) -> RagManager:
        settings = dict(
            embedder=self.embedder,
            dimension=DIMENSION,
            min_score=0.0,
            top_k=5,
        )
        settings.update(overrides)
        return RagManager(self.tmp / store_dir_name, **settings)

    def write(self, name: str, text: str) -> Path:
        path = self.docs / name
        path.write_text(text, encoding="utf-8")
        return path


# --- extraction -----------------------------------------------------------


class ExtractionTests(TempCase):
    def test_reads_a_text_file_with_no_page_numbers(self):
        path = self.write("notes.txt", "Hello there.\n\nSecond paragraph.")
        document = extract(path)
        self.assertEqual(document.kind, "text")
        self.assertIsNone(document.page_count)
        # A format without pages must not invent one: a page number in a
        # citation is either true or absent.
        self.assertTrue(all(e.page is None for e in document.elements))
        self.assertIn("Second paragraph", document.text)

    def test_reads_a_pdf_page_by_page_with_numbers(self):
        path = write_pdf(
            self.docs / "doc.pdf",
            [["Page one text"], ["Page two text"], ["Page three text"]],
        )
        document = extract(path)
        self.assertEqual(document.kind, "pdf")
        self.assertEqual(document.page_count, 3)
        self.assertEqual(
            sorted({e.page for e in document.elements if e.searchable}), [1, 2, 3]
        )
        page_two = " ".join(e.content for e in document.elements if e.page == 2)
        self.assertIn("two", page_two)

    def test_a_text_pdf_does_not_ask_for_ocr(self):
        path = write_pdf(self.docs / "doc.pdf", [["Readable text layer"]])
        document = extract(path)
        self.assertFalse(document.ocr_used)
        self.assertEqual(document.ocr_pages, ())
        self.assertEqual(document.ocr_note, "")

    def test_source_code_keeps_its_indentation(self):
        path = self.write("app.py", "def f():\n    return 1\n")
        self.assertIn("    return 1", extract(path).text)

    def test_missing_file(self):
        with self.assertRaises(ExtractionError) as caught:
            extract(self.docs / "nope.txt")
        self.assertIn("does not exist", str(caught.exception))

    def test_unsupported_type_lists_what_is_supported(self):
        path = self.write("thing.xyz", "data")
        with self.assertRaises(ExtractionError) as caught:
            extract(path)
        self.assertIn(".pdf", str(caught.exception))

    def test_empty_file(self):
        path = self.write("empty.txt", "")
        with self.assertRaises(ExtractionError) as caught:
            extract(path)
        self.assertIn("empty", str(caught.exception))

    def test_whitespace_only_file_has_no_extractable_text(self):
        path = self.write("blank.txt", "   \n\n   \t\n")
        with self.assertRaises(ExtractionError) as caught:
            extract(path)
        self.assertIn("no extractable text", str(caught.exception))

    def test_corrupt_pdf_is_reported_not_raised_raw(self):
        path = self.docs / "broken.pdf"
        path.write_bytes(b"%PDF-1.4 this is not really a pdf")
        with self.assertRaises(ExtractionError) as caught:
            extract(path)
        self.assertIn("corrupt", str(caught.exception).lower())

    def test_a_directory_is_not_a_document(self):
        with self.assertRaises(ExtractionError):
            extract(self.docs)

    def test_oversized_file_is_refused(self):
        path = self.write("big.txt", "x" * 2000)
        with self.assertRaises(ExtractionError) as caught:
            extract(path, max_bytes=100)
        self.assertIn("limit", str(caught.exception))

    def test_is_supported(self):
        self.assertTrue(is_supported("a.pdf"))
        self.assertTrue(is_supported("a.PY"))
        self.assertFalse(is_supported("a.exe"))


class TableDetectionTests(TempCase):
    """Table finding is the most expensive call in extraction, and on a book
    it is nearly all of it."""

    def test_a_ruled_table_is_still_extracted(self):
        path = self.docs / "report.pdf"
        write_pdf(path, [["Introduction text"], ["Table 3.1 Boiling points"]])
        add_ruled_table(path, page_number=2)

        document = extract_pdf(path)
        self.assertIn("table", [element.type for element in document.elements])

    def test_a_page_with_no_drawings_cannot_hold_a_line_ruled_table(self):
        """The filter is exact, not a guess: find_tables defaults to
        strategy='lines', so no lines means no possible result."""
        import fitz

        from rag.extract.pdf import _may_hold_table

        path = self.docs / "prose.pdf"
        write_pdf(path, [["Nothing but prose on this page."]])
        add_ruled_table(path, page_number=1)

        document = fitz.open(path)
        try:
            self.assertTrue(_may_hold_table(document[0]))
        finally:
            document.close()

        plain = self.docs / "plain.pdf"
        write_pdf(plain, [["Nothing but prose on this page."]])
        document = fitz.open(plain)
        try:
            self.assertFalse(_may_hold_table(document[0]))
        finally:
            document.close()

    def test_prose_pages_produce_no_table_elements(self):
        path = self.docs / "book.pdf"
        write_pdf(path, [["Alkenes contain a double bond."], ["More prose here."]])
        document = extract_pdf(path)
        self.assertEqual(
            [element.type for element in document.elements], ["text", "text"]
        )


class FigureTests(TempCase):
    """Raster figures: captions become searchable, pictures become findable."""

    def book(self, *, caption: str = "Figure 3.4 Infrared spectrum of ethene"):
        path = self.docs / "spectroscopy.pdf"
        write_pdf(
            path,
            [
                [
                    "3.4 Spectroscopy of alkenes",
                    "The absorption band is diagnostic of the double bond.",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    caption,
                    "Body text continues after the figure here.",
                ]
            ],
        )
        # write_pdf lays text out in PDF coordinates (y upwards from the
        # bottom) while an image box is in fitz coordinates (y downwards from
        # the top). Line 14 lands at y=268 from the top, so the picture has to
        # end just above it for the caption to be *below* the figure.
        add_raster_figure(path, box=(72, 100, 372, 250))
        return path

    def test_a_figure_is_written_out_and_recorded(self):
        self.book()
        manager = self.manager()
        manager.index_path(self.docs)

        figures = manager.outline("spectroscopy.pdf")["figures"]
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0]["page"], 1)
        self.assertTrue(Path(figures[0]["path"]).is_file())

    def test_the_caption_is_found_and_attached(self):
        self.book()
        manager = self.manager()
        manager.index_path(self.docs)

        figure = manager.outline("spectroscopy.pdf")["figures"][0]
        self.assertIn("Infrared spectrum of ethene", figure["caption"])

    def test_body_text_is_never_mistaken_for_a_caption(self):
        """An empty caption is honest; calling a paragraph of prose one is
        not."""
        self.book(caption="The following discussion concerns bond vibration.")
        manager = self.manager()
        manager.index_path(self.docs)

        figure = manager.outline("spectroscopy.pdf")["figures"][0]
        self.assertEqual(figure["caption"], "")

    def test_furniture_is_not_a_figure(self):
        """A logo in a running header would bury the real figures."""
        path = self.docs / "letterhead.pdf"
        write_pdf(path, [["Some text on the page."]])
        add_raster_figure(path, size=20, box=(520, 40, 540, 60))

        manager = self.manager()
        manager.index_path(self.docs)
        self.assertEqual(manager.outline("letterhead.pdf")["figures"], [])

    def test_removing_a_document_removes_its_figures(self):
        """They are derived data nothing else refers to, so leaving them
        behind is a directory of orphans that only grows."""
        self.book()
        manager = self.manager()
        manager.index_path(self.docs)
        stored = Path(manager.outline("spectroscopy.pdf")["figures"][0]["path"])
        self.assertTrue(stored.is_file())

        manager.remove("spectroscopy.pdf")
        self.assertFalse(stored.exists())

    def test_reindexing_does_not_accumulate_figures(self):
        self.book()
        manager = self.manager()
        manager.index_path(self.docs)
        manager.index_path(self.docs, force=True)

        self.assertEqual(len(manager.outline("spectroscopy.pdf")["figures"]), 1)

    def test_it_can_be_turned_off(self):
        self.book()
        manager = self.manager(figures=False)
        manager.index_path(self.docs)

        self.assertEqual(manager.outline("spectroscopy.pdf")["figures"], [])
        self.assertFalse((self.tmp / "store" / "figures").exists())

    def test_a_document_with_no_figures_reports_none(self):
        self.write("plain.txt", "Nothing but text in this file.")
        manager = self.manager()
        manager.index_path(self.docs)
        self.assertEqual(manager.outline("plain.txt")["figures"], [])


class CleaningTests(unittest.TestCase):
    def test_strips_zero_width_characters(self):
        self.assertEqual(clean_text("a" + chr(0x200B) + "b"), "ab")

    def test_collapses_runs_of_blank_lines(self):
        self.assertEqual(clean_text("a\n\n\n\n\nb"), "a\n\nb")

    def test_keeps_a_single_blank_line_as_a_paragraph_break(self):
        self.assertEqual(clean_text("a\n\nb"), "a\n\nb")

    def test_joins_words_broken_across_lines_only_for_pdfs(self):
        self.assertEqual(
            clean_text("mitochon-\ndria", join_broken_words=True), "mitochondria"
        )
        # A real hyphen at a line end in source code must survive.
        self.assertEqual(clean_text("well-\nknown"), "well-\nknown")

    def test_normalises_line_endings(self):
        self.assertEqual(clean_text("a\r\nb"), "a\nb")


# --- chunking -------------------------------------------------------------


class ChunkingTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        chunks = chunk_document(text_document(paragraph("Short enough.")))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short enough.")

    def test_long_text_splits_and_numbers_chunks_in_order(self):
        body = "This sentence is here to take up room. " * 12
        document = text_document(*(paragraph(body) for _ in range(6)))
        chunks = chunk_document(document, chunk_tokens=100, overlap_tokens=20)
        self.assertGreater(len(chunks), 2)
        self.assertEqual([chunk.ordinal for chunk in chunks], list(range(len(chunks))))

    def test_chunks_stay_within_the_character_budget(self):
        document = text_document(
            *(paragraph("Filler sentence number %d here. " % n * 8) for n in range(12))
        )
        budget = token_budget(100)
        for chunk in chunk_document(document, chunk_tokens=100, overlap_tokens=20):
            self.assertLessEqual(len(chunk.text), budget)

    def test_consecutive_chunks_overlap(self):
        sentences = " ".join(
            f"Sentence number {n} carries some words." for n in range(60)
        )
        chunks = chunk_document(
            text_document(paragraph(sentences)), chunk_tokens=100, overlap_tokens=25
        )
        self.assertGreater(len(chunks), 1)
        carried = max(
            (
                n
                for n in range(10, len(chunks[0].text) + 1)
                if chunks[1].text.startswith(chunks[0].text[-n:])
            ),
            default=0,
        )
        self.assertGreater(carried, 0, "the second chunk carries none of the first")

    def test_pages_are_never_merged_so_the_page_number_stays_true(self):
        chunks = chunk_document(
            text_document(paragraph("Alpha text.", 1), paragraph("Beta text.", 2))
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk.page for chunk in chunks], [1, 2])

    def test_ordinals_run_across_pages_not_within_them(self):
        body = "Words words words. " * 200
        chunks = chunk_document(
            text_document(paragraph(body, 1), paragraph(body, 2)),
            chunk_tokens=100,
            overlap_tokens=10,
        )
        self.assertEqual([chunk.ordinal for chunk in chunks], list(range(len(chunks))))
        self.assertEqual(chunks[0].page, 1)
        self.assertEqual(chunks[-1].page, 2)

    def test_a_heading_becomes_the_section_of_the_chunks_under_it(self):
        chunks = chunk_document(
            text_document(
                heading("Fire Safety"),
                paragraph("Use the CO2 extinguisher on an electrical fire."),
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Fire Safety")
        # The heading leads the text rather than being a chunk of its own: on
        # its own it embeds to almost nothing.
        self.assertTrue(chunks[0].text.startswith("Fire Safety"))
        self.assertIn("extinguisher", chunks[0].text)

    def test_a_new_heading_starts_a_new_chunk(self):
        chunks = chunk_document(
            text_document(
                heading("Alpha"),
                paragraph("First body text."),
                heading("Beta"),
                paragraph("Second body text."),
            )
        )
        self.assertEqual([chunk.section for chunk in chunks], ["Alpha", "Beta"])

    def test_a_table_is_chunked_as_searchable_text(self):
        table = DocumentElement(
            type="table", content="Name | Role\n--- | ---\nAda | Engineer"
        )
        chunks = chunk_document(text_document(table))
        self.assertEqual(len(chunks), 1)
        self.assertIn("Ada", chunks[0].text)

    def test_text_with_no_whitespace_is_hard_wrapped(self):
        chunks = chunk_document(
            text_document(paragraph("x" * 5000)), chunk_tokens=100, overlap_tokens=0
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), token_budget(100))

    def test_overlap_larger_than_the_chunk_cannot_stall_the_split(self):
        # Overlap is clamped to half the chunk; without that, chunking a long
        # document would never advance.
        body = "Sentence with several words in it. " * 100
        chunks = chunk_document(
            text_document(paragraph(body)), chunk_tokens=50, overlap_tokens=500
        )
        self.assertGreater(len(chunks), 1)

    def test_an_empty_element_yields_nothing(self):
        self.assertEqual(chunk_document(text_document(paragraph("   "))), [])

    def test_token_budget_is_capped_at_the_models_window(self):
        # 5,000 tokens asked for, but BGE truncates at 512.
        self.assertEqual(token_budget(5000), int(512 * CHARS_PER_TOKEN))


# --- the vector index -----------------------------------------------------


class VectorIndexTests(TempCase):
    def index(self, dimension: int = 4) -> VectorIndex:
        return VectorIndex(self.tmp / "vectors.f32", dimension)

    def test_write_then_search_finds_the_nearest_row(self):
        index = self.index()
        index.write(
            [0, 1, 2],
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0.7, 0.7, 0, 0]], dtype=np.float32
            ),
        )
        hits = index.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
        self.assertEqual(hits[0][0], 0)
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)
        self.assertEqual(hits[1][0], 2)

    def test_specific_rows_can_be_scored_without_a_search(self):
        """What a keyword hit needs: it arrives with a row and no cosine."""
        index = self.index()
        index.write(
            [0, 1, 2],
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0.6, 0.8, 0, 0]], dtype=np.float32
            ),
        )
        scores = index.score_rows(np.array([1, 0, 0, 0], dtype=np.float32), [2, 0])

        self.assertEqual(sorted(scores), [0, 2])
        self.assertAlmostEqual(scores[0], 1.0, places=5)
        self.assertAlmostEqual(scores[2], 0.6, places=5)

    def test_scoring_ignores_rows_that_are_not_there(self):
        index = self.index()
        index.write([0], np.array([[1, 0, 0, 0]], dtype=np.float32))
        scores = index.score_rows(np.array([1, 0, 0, 0], dtype=np.float32), [0, 99])
        self.assertEqual(list(scores), [0])

    def test_scoring_nothing_needs_no_index(self):
        self.assertEqual(self.index().score_rows(np.zeros(4, np.float32), []), {})

    def test_capacity_counts_rows(self):
        index = self.index()
        self.assertEqual(index.capacity(), 0)
        index.write([0, 1], np.zeros((2, 4), dtype=np.float32))
        self.assertEqual(index.capacity(), 2)

    def test_scattered_rows_are_written_where_they_belong(self):
        index = self.index()
        index.write([5], np.array([[1, 0, 0, 0]], dtype=np.float32))
        self.assertEqual(index.capacity(), 6)
        hits = index.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
        self.assertEqual(hits[0][0], 5)

    def test_freed_rows_are_skipped_by_a_search(self):
        index = self.index()
        index.write(
            [0, 1], np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        )
        hits = index.search(
            np.array([1, 0, 0, 0], dtype=np.float32), top_k=2, skip={0}
        )
        self.assertEqual([row for row, _ in hits], [1])

    def test_a_truncated_file_is_reported_as_corrupt(self):
        index = self.index()
        index.write([0, 1], np.zeros((2, 4), dtype=np.float32))
        with index.path.open("r+b") as handle:
            handle.truncate(index.path.stat().st_size - 3)
        with self.assertRaises(VectorIndexError) as caught:
            index.capacity()
        self.assertIn("corrupt", str(caught.exception).lower())

    def test_verify_catches_an_index_shorter_than_the_metadata(self):
        index = self.index()
        index.write([0], np.zeros((1, 4), dtype=np.float32))
        with self.assertRaises(VectorIndexError) as caught:
            index.verify(expected_rows=5)
        self.assertIn("rebuild", str(caught.exception).lower())

    def test_a_query_of_the_wrong_width_says_to_rebuild(self):
        index = self.index()
        index.write([0], np.zeros((1, 4), dtype=np.float32))
        with self.assertRaises(VectorIndexError) as caught:
            index.search(np.zeros(8, dtype=np.float32), top_k=1)
        self.assertIn("rebuild", str(caught.exception).lower())

    def test_search_spans_blocks(self):
        # More rows than one search block, so the running top-k is exercised
        # rather than a single pass that happens to see everything.
        from rag import index as index_module

        rows = index_module.SEARCH_BLOCK_ROWS + 50
        vectors = np.zeros((rows, 4), dtype=np.float32)
        vectors[:, 0] = 0.1
        vectors[rows - 1] = [1, 0, 0, 0]
        index = self.index()
        index.write(list(range(rows)), vectors)
        hits = index.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
        self.assertEqual(hits[0][0], rows - 1)

    def test_compact_returns_the_row_mapping_and_shrinks_the_file(self):
        index = self.index()
        index.write(
            [0, 1, 2],
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32
            ),
        )
        mapping = index.compact([0, 2])
        self.assertEqual(mapping, {0: 0, 2: 1})
        self.assertEqual(index.capacity(), 2)
        hits = index.search(np.array([0, 0, 1, 0], dtype=np.float32), top_k=1)
        self.assertEqual(hits[0][0], 1)

    def test_mismatched_row_and_vector_counts_are_refused(self):
        with self.assertRaises(VectorIndexError):
            self.index().write([0, 1], np.zeros((3, 4), dtype=np.float32))


# --- the metadata store ---------------------------------------------------


class MetadataStoreTests(TempCase):
    def store(self) -> MetadataStore:
        return MetadataStore(self.tmp / "chunks.db")

    def add(self, store: MetadataStore, path: str, chunks: list[tuple]) -> int:
        """Store a document. `chunks` is (ordinal, page, text, vector_row).

        The section is filled in here rather than in every caller: these tests
        are about row bookkeeping, and spelling out `None` for a section in
        thirty tuples would bury what each one is actually checking.
        """
        document_id, freed = store.replace_document(
            path=path, name=Path(path).name, suffix=".txt", size_bytes=1,
            mtime_ns=1, sha256="abc", pages=None, kind="text",
            chunks=[
                (ordinal, page, None, text, vector_row)
                for ordinal, page, text, vector_row in chunks
            ],
        )
        store.release_rows(freed)
        return document_id

    def test_replacing_a_document_does_not_duplicate_its_chunks(self):
        store = self.store()
        self.add(store, "/a.txt", [(0, None, "one", 0), (1, None, "two", 1)])
        self.add(store, "/a.txt", [(0, None, "new", 2)])
        self.assertEqual(store.counts(), (1, 1))

    def test_replacing_a_document_frees_its_old_rows(self):
        store = self.store()
        self.add(store, "/a.txt", [(0, None, "one", 0), (1, None, "two", 1)])
        self.add(store, "/a.txt", [(0, None, "new", 2)])
        self.assertEqual(store.free_row_set(), {0, 1})

    def test_deleting_a_document_removes_chunks_and_frees_rows(self):
        store = self.store()
        document_id = self.add(store, "/a.txt", [(0, None, "one", 0)])
        name, freed = store.delete_document(document_id)
        self.assertEqual(name, "a.txt")
        self.assertEqual(freed, [0])
        self.assertEqual(store.counts(), (0, 0))
        self.assertEqual(store.free_row_set(), {0})

    def test_deleting_an_unknown_document_raises(self):
        with self.assertRaises(KeyError):
            self.store().delete_document(999)

    def test_free_rows_are_handed_out_once(self):
        store = self.store()
        store.release_rows([3, 4])
        self.assertEqual(store.take_free_rows(2), [3, 4])
        self.assertEqual(store.take_free_rows(2), [])

    def test_max_vector_row_counts_free_rows_too(self):
        # The trap this guards: taking rows off the free list must not lower
        # the boundary, or the next append hands out a row already in use.
        store = self.store()
        self.add(store, "/a.txt", [(0, None, "one", 5)])
        store.release_rows([9])
        self.assertEqual(store.max_vector_row(), 9)

    def test_chunks_are_fetched_by_vector_row(self):
        store = self.store()
        self.add(store, "/a.txt", [(0, 7, "the text", 2)])
        found = store.chunks_by_rows([2])
        self.assertEqual(found[2].text, "the text")
        self.assertEqual(found[2].page, 7)
        self.assertEqual(found[2].document, "a.txt")

    def test_chunk_id_identifies_document_and_position(self):
        store = self.store()
        document_id = self.add(store, "/a.txt", [(4, None, "t", 0)])
        self.assertEqual(store.chunks_by_rows([0])[0].chunk_id, f"{document_id}#4")

    def test_remap_rows_renumbers_without_colliding(self):
        store = self.store()
        self.add(store, "/a.txt", [(0, None, "one", 1), (1, None, "two", 3)])
        store.release_rows([0, 2])
        # 1 -> 0 collides with nothing only because the update goes via an
        # offset; a direct update would hit the UNIQUE constraint.
        store.remap_rows({1: 0, 3: 1})
        rows = dict(store.all_chunk_rows())
        self.assertEqual(sorted(rows.values()), [0, 1])
        self.assertEqual(store.free_row_set(), set())

    def test_settings_round_trip(self):
        store = self.store()
        store.save_settings({"model": "m", "dimension": "384"})
        self.assertEqual(store.settings()["dimension"], "384")


# --- the manager ----------------------------------------------------------


class ManagerTests(TempCase):
    def test_index_then_search_returns_the_matching_chunk(self):
        self.write("notes.txt", "Mitochondria generate ATP by oxidative phosphorylation.")
        manager = self.manager()
        manager.index_path(self.docs)
        result = manager.search("mitochondria ATP")
        self.assertEqual(result["count"], 1)
        self.assertIn("Mitochondria", result["results"][0]["text"])
        self.assertEqual(result["results"][0]["document"], "notes.txt")

    def test_search_ranks_the_better_match_first(self):
        self.write("a.txt", "Photosynthesis happens in the chloroplast thylakoid.")
        self.write("b.txt", "Enzymes are catalysts that lower activation energy.")
        manager = self.manager()
        manager.index_path(self.docs)
        result = manager.search("chloroplast thylakoid photosynthesis")
        self.assertEqual(result["results"][0]["document"], "a.txt")

    def test_pdf_results_carry_the_page(self):
        write_pdf(
            self.docs / "handbook.pdf",
            [["Introduction chapter"], ["Fire safety extinguisher rules"]],
        )
        manager = self.manager()
        manager.index_path(self.docs)
        result = manager.search("fire extinguisher")
        self.assertEqual(result["results"][0]["page"], 2)

    def test_text_results_omit_the_page_rather_than_reporting_null(self):
        self.write("notes.txt", "Osmosis moves water across a membrane.")
        manager = self.manager()
        manager.index_path(self.docs)
        self.assertNotIn("page", manager.search("osmosis water")["results"][0])

    def test_unchanged_files_are_skipped_and_not_re_embedded(self):
        self.write("notes.txt", "Ribosomes translate messenger RNA.")
        manager = self.manager()
        manager.index_path(self.docs)
        after_first = self.embedder.passages_encoded

        result = manager.index_path(self.docs)
        self.assertEqual(result["skipped"], ["notes.txt"])
        self.assertEqual(result["indexed"], [])
        self.assertEqual(self.embedder.passages_encoded, after_first)

    def test_re_indexing_does_not_duplicate_chunks(self):
        self.write("notes.txt", "Ribosomes translate messenger RNA.")
        manager = self.manager()
        manager.index_path(self.docs)
        manager.index_path(self.docs)
        manager.index_path(self.docs, force=True)
        self.assertEqual(manager.stats()["chunks"], 1)
        self.assertEqual(manager.stats()["documents"], 1)

    def test_a_changed_file_is_re_indexed_and_the_new_text_is_findable(self):
        path = self.write("notes.txt", "Ribosomes translate messenger RNA.")
        manager = self.manager()
        manager.index_path(self.docs)

        path.write_text(
            "The Golgi apparatus packages proteins into vesicles.", encoding="utf-8"
        )
        # st_mtime_ns can be too coarse to notice a write this fast, so the
        # hash is what has to catch it. Forcing the timestamp back proves the
        # hash path runs rather than the timestamp path.
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        manager.store.touch_document(1, mtime_ns=1_000_000_000, size_bytes=34)

        result = manager.index_path(self.docs)
        self.assertEqual([entry["document"] for entry in result["indexed"]], ["notes.txt"])
        found = manager.search("Golgi vesicles packages")
        self.assertIn("Golgi", found["results"][0]["text"])
        self.assertEqual(manager.stats()["chunks"], 1)

    def test_a_touched_but_unchanged_file_is_not_re_embedded(self):
        path = self.write("notes.txt", "Ribosomes translate messenger RNA.")
        manager = self.manager()
        manager.index_path(self.docs)
        before = self.embedder.passages_encoded

        # Same bytes, new timestamp: what a sync tool or a copy produces.
        os.utime(path, ns=(2_000_000_000, 2_000_000_000))
        result = manager.index_path(self.docs)
        self.assertEqual(result["skipped"], ["notes.txt"])
        self.assertEqual(self.embedder.passages_encoded, before)

    def test_the_index_survives_a_new_manager_without_re_embedding(self):
        self.write("notes.txt", "Mitochondria generate ATP.")
        self.manager().index_path(self.docs)

        fresh_embedder = HashingEmbedder()
        reopened = self.manager(embedder=fresh_embedder)
        result = reopened.search("mitochondria ATP")
        self.assertEqual(result["count"], 1)
        # One query embedded, no passages: the vectors came off disk.
        self.assertEqual(fresh_embedder.passages_encoded, 0)
        self.assertEqual(fresh_embedder.queries_encoded, 1)

    def test_removing_a_document_takes_its_chunks_out_of_search(self):
        self.write("a.txt", "Photosynthesis in the chloroplast.")
        self.write("b.txt", "Enzymes lower activation energy.")
        manager = self.manager()
        manager.index_path(self.docs)

        manager.remove("a.txt")
        result = manager.search("photosynthesis chloroplast")
        self.assertNotIn("a.txt", [hit["document"] for hit in result["results"]])
        self.assertEqual(manager.stats()["documents"], 1)

    def test_a_freed_row_is_reused_by_the_next_document(self):
        path = self.write("a.txt", "Alpha text here.")
        manager = self.manager()
        manager.index_path(self.docs)
        manager.remove("a.txt")
        path.unlink()  # or indexing the folder again simply re-adds it

        self.write("b.txt", "Beta text here.")
        manager.index_path(self.docs)

        connection = sqlite3.connect(manager.store.path)
        try:
            rows = [row[0] for row in connection.execute("SELECT vector_row FROM chunks")]
            free = [row[0] for row in connection.execute("SELECT vector_row FROM free_rows")]
        finally:
            connection.close()
        self.assertEqual(rows, [0])
        self.assertEqual(free, [])

    def test_a_freed_row_never_surfaces_in_results(self):
        path = self.write("a.txt", "Photosynthesis chloroplast thylakoid stroma.")
        manager = self.manager()
        manager.index_path(self.docs)
        manager.remove("a.txt")
        path.unlink()

        self.write("b.txt", "Completely different words about enzymes.")
        manager = self.manager(min_score=-1.0)
        manager.index_path(self.docs)
        # The deleted vector is gone; nothing may cite a chunk that is not
        # there, whatever it scores.
        for hit in manager.search("photosynthesis chloroplast")["results"]:
            self.assertEqual(hit["document"], "b.txt")

    def test_compact_preserves_which_vector_belongs_to_which_chunk(self):
        for name, text in (
            ("a.txt", "Alpha alpha alpha unique words."),
            ("b.txt", "Beta beta beta different words."),
            ("c.txt", "Gamma gamma gamma other words."),
        ):
            self.write(name, text)
        manager = self.manager()
        manager.index_path(self.docs)
        manager.remove("b.txt")

        manager.compact()
        result = manager.search("gamma other words")
        self.assertEqual(result["results"][0]["document"], "c.txt")
        self.assertEqual(manager.stats()["chunks"], 2)

    def test_compact_with_nothing_to_reclaim_says_so(self):
        self.write("a.txt", "Alpha text.")
        manager = self.manager()
        manager.index_path(self.docs)
        self.assertFalse(manager.compact()["compacted"])

    def test_rebuild_re_reads_every_document(self):
        self.write("a.txt", "Alpha text here.")
        self.write("b.txt", "Beta text here.")
        manager = self.manager()
        manager.index_path(self.docs)

        result = manager.rebuild()
        self.assertEqual(len(result["rebuilt"]), 2)
        self.assertEqual(result["dropped"], [])
        self.assertEqual(manager.stats()["chunks"], 2)
        self.assertTrue(manager.search("alpha text")["results"])

    def test_rebuild_drops_documents_whose_source_is_gone(self):
        path = self.write("a.txt", "Alpha text here.")
        self.write("b.txt", "Beta text here.")
        manager = self.manager()
        manager.index_path(self.docs)

        path.unlink()
        result = manager.rebuild()
        self.assertEqual(result["dropped"], ["a.txt"])
        self.assertEqual(manager.stats()["documents"], 1)

    def test_rebuild_with_an_empty_index_is_an_error_not_a_no_op(self):
        with self.assertRaises(RagError):
            self.manager().rebuild()

    def test_searching_an_empty_index_explains_rather_than_failing(self):
        result = self.manager().search("anything at all")
        self.assertEqual(result["count"], 0)
        self.assertIn("Nothing has been indexed", result["note"])

    def test_the_similarity_threshold_filters_weak_matches(self):
        self.write("a.txt", "Photosynthesis chloroplast thylakoid.")
        manager = self.manager(min_score=0.99)
        manager.index_path(self.docs)
        result = manager.search("completely unrelated banking terminology")
        self.assertEqual(result["count"], 0)
        self.assertIn("threshold", result["note"])

    def test_top_k_limits_the_results(self):
        for n in range(6):
            self.write(f"n{n}.txt", f"Shared common words document number {n}.")
        manager = self.manager()
        manager.index_path(self.docs)
        self.assertEqual(len(manager.search("shared common words", top_k=2)["results"]), 2)

    def test_indexing_a_folder_skips_directories_that_are_never_wanted(self):
        (self.docs / ".venv").mkdir()
        (self.docs / ".venv" / "junk.py").write_text("x = 1", encoding="utf-8")
        (self.docs / "node_modules").mkdir()
        (self.docs / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
        self.write("real.txt", "Real content here.")

        manager = self.manager()
        result = manager.index_path(self.docs)
        self.assertEqual([entry["document"] for entry in result["indexed"]], ["real.txt"])

    def test_one_bad_file_does_not_abandon_the_rest(self):
        self.write("good.txt", "Good content here.")
        (self.docs / "bad.pdf").write_bytes(b"%PDF-1.4 not a pdf")
        manager = self.manager()
        result = manager.index_path(self.docs)
        self.assertEqual([entry["document"] for entry in result["indexed"]], ["good.txt"])
        self.assertEqual([entry["document"] for entry in result["failed"]], ["bad.pdf"])

    def test_indexing_a_missing_path_is_refused(self):
        with self.assertRaises(RagError) as caught:
            self.manager().index_path(self.docs / "nope.txt")
        self.assertIn("does not exist", str(caught.exception))

    def test_indexing_a_folder_with_nothing_supported_is_refused(self):
        (self.docs / "thing.exe").write_bytes(b"MZ")
        with self.assertRaises(RagError) as caught:
            self.manager().index_path(self.docs)
        self.assertIn("No supported files", str(caught.exception))

    def test_an_empty_query_is_refused(self):
        with self.assertRaises(RagError):
            self.manager().search("   ")

    def test_a_query_the_size_of_a_document_is_refused(self):
        with self.assertRaises(RagError) as caught:
            self.manager().search("word " * 1000)
        self.assertIn("question", str(caught.exception))

    def test_removing_something_not_indexed_is_refused(self):
        with self.assertRaises(RagError) as caught:
            self.manager().remove("ghost.pdf")
        self.assertIn("not in the index", str(caught.exception))

    def test_a_document_can_be_removed_by_id(self):
        self.write("a.txt", "Alpha.")
        manager = self.manager()
        manager.index_path(self.docs)
        document_id = manager.list_documents()["documents"][0]["id"]
        self.assertTrue(manager.remove(document_id)["success"])

    def test_changing_the_model_is_refused_rather_than_mixing_vectors(self):
        self.write("a.txt", "Alpha text.")
        self.manager().index_path(self.docs)

        other = self.manager(model="some/other-model")
        with self.assertRaises(RagError) as caught:
            other.search("alpha")
        self.assertIn("rebuild", str(caught.exception).lower())

    def test_changing_the_dimension_is_refused(self):
        self.write("a.txt", "Alpha text.")
        self.manager().index_path(self.docs)

        with self.assertRaises(RagError):
            self.manager(dimension=128).index_path(self.docs, force=True)

    def test_list_documents_reports_chunks_and_pages(self):
        write_pdf(self.docs / "doc.pdf", [["One"], ["Two"]])
        manager = self.manager()
        manager.index_path(self.docs)
        entry = manager.list_documents()["documents"][0]
        self.assertEqual(entry["document"], "doc.pdf")
        self.assertEqual(entry["pages"], 2)
        self.assertEqual(entry["chunks"], 2)

    def test_stats_reports_the_model_the_index_was_built_with(self):
        self.write("a.txt", "Alpha text.")
        manager = self.manager()
        manager.index_path(self.docs)
        stats = manager.stats()
        self.assertEqual(stats["dimension"], DIMENSION)
        self.assertEqual(stats["chunks"], 1)
        self.assertEqual(stats["model"], "BAAI/bge-small-en-v1.5")

        # Unloading is what the sweeper does; the index stays searchable.
        manager.unload()
        self.assertFalse(manager.stats()["embedder_loaded"])
        self.assertEqual(manager.search("alpha text")["count"], 1)

    def test_a_failed_embedding_leaves_the_index_untouched(self):
        self.write("a.txt", "Alpha text.")
        manager = self.manager()
        manager.index_path(self.docs)
        before = manager.stats()

        class Broken(HashingEmbedder):
            def encode_passages(self, texts, progress=None):
                from rag.embeddings import EmbeddingError

                raise EmbeddingError("out of memory")

        self.write("b.txt", "Beta text.")
        broken = self.manager(embedder=Broken())
        with self.assertRaises(RagError) as caught:
            broken.index_path(self.docs, force=True)
        self.assertIn("out of memory", str(caught.exception))

        # Nothing was written, so the counts are exactly as they were and the
        # document that was already there is still searchable.
        self.assertEqual(manager.stats()["chunks"], before["chunks"])
        self.assertEqual(manager.stats()["documents"], before["documents"])
        self.assertEqual(manager.search("alpha text")["count"], 1)


# --- the agent tool -------------------------------------------------------


class HybridSearchTests(TempCase):
    """Keyword matching beside the embeddings.

    The failure this exists for is specific: a term that has to *appear* -
    "E2", a formula, a surname - is what an embedding model is worst at, and
    bge-small's noise floor leaves little room to tell a weak match from none.
    The fake embedder here has the same weakness for the same reason, so the
    case is reproducible without the real model.
    """

    def library(self) -> None:
        self.write(
            "chapter10.txt",
            "Section 10.7 Dehydration. When a secondary alcohol is heated with "
            "acid the major product follows the Zaitsev rule, giving the more "
            "substituted alkene as the principal product.",
        )
        self.write(
            "chapter04.txt",
            "Removing a water molecule from an alcohol produces a double bond "
            "between two carbon atoms, the general pattern of an elimination.",
        )
        self.write(
            "biology.txt",
            "Enzymes are biological catalysts that lower the activation energy "
            "of a reaction without being consumed by it.",
        )

    def test_an_exact_term_survives_a_similarity_score_below_the_threshold(self):
        """The whole point. Vector-only finds nothing; the words are right there."""
        self.library()

        without = self.manager(min_score=0.3, hybrid=False)
        without.index_path(self.docs)
        self.assertEqual(without.search("Zaitsev rule")["count"], 0)

        with_keywords = self.manager(
            store_dir_name="hybrid", min_score=0.3, hybrid=True
        )
        with_keywords.index_path(self.docs)
        found = with_keywords.search("Zaitsev rule")
        self.assertEqual(found["count"], 1)
        self.assertEqual(found["results"][0]["document"], "chapter10.txt")

    def test_a_result_says_how_it_was_found(self):
        """A weak score means something different for each, so it is reported."""
        self.library()
        manager = self.manager(min_score=0.3)
        manager.index_path(self.docs)

        hit = manager.search("Zaitsev")["results"][0]
        self.assertIn(hit["match"], ("keyword", "both"))
        # The score stays cosine similarity, not a fused number: one column,
        # one meaning.
        self.assertLess(hit["score"], 0.3)

    def test_the_threshold_still_gates_a_purely_semantic_guess(self):
        """Only evidence is exempt. A vague near-miss is still filtered."""
        self.library()
        manager = self.manager(min_score=0.9)
        manager.index_path(self.docs)
        self.assertEqual(manager.search("photosynthesis chloroplast")["count"], 0)

    def test_turning_it_off_restores_semantic_only_search(self):
        self.library()
        manager = self.manager(min_score=0.3, hybrid=False)
        manager.index_path(self.docs)
        self.assertEqual(manager.search("Zaitsev rule")["count"], 0)

    def test_deleting_a_document_takes_it_out_of_keyword_search(self):
        """The FTS index is kept in step by triggers; assert it, because a
        stale one would return chunks that no longer exist."""
        self.library()
        manager = self.manager(min_score=0.3)
        manager.index_path(self.docs)
        self.assertEqual(manager.search("Zaitsev")["count"], 1)

        manager.remove("chapter10.txt")
        self.assertEqual(manager.search("Zaitsev")["count"], 0)

    def test_reindexing_does_not_duplicate_a_keyword_hit(self):
        self.library()
        manager = self.manager(min_score=0.3)
        manager.index_path(self.docs)
        manager.index_path(self.docs, force=True)

        found = manager.search("Zaitsev")
        self.assertEqual(found["count"], 1)

    def test_an_index_built_before_keyword_search_is_backfilled(self):
        """The migration that matters: the chunk text is already in SQLite, so
        adding keyword search must cost a rebuild of the FTS index and not a
        single re-embedded chunk."""
        self.library()
        manager = self.manager(min_score=0.3)
        manager.index_path(self.docs)
        embedded = self.embedder.passages_encoded
        self.assertGreater(embedded, 0)

        # Wind the store back to what it looked like before this feature.
        from rag.manager import METADATA_FILE

        connection = sqlite3.connect(self.tmp / "store" / METADATA_FILE)
        connection.executescript(
            "DROP TRIGGER IF EXISTS chunks_search_insert;"
            "DROP TRIGGER IF EXISTS chunks_search_delete;"
            "DROP TRIGGER IF EXISTS chunks_search_update;"
            "DROP TABLE IF EXISTS chunk_search;"
            "DELETE FROM index_meta WHERE key = 'keyword_index_built';"
        )
        connection.commit()
        connection.close()

        reopened = self.manager(min_score=0.3)
        self.assertEqual(reopened.search("Zaitsev rule")["count"], 1)
        # Nothing was embedded to get there.
        self.assertEqual(self.embedder.passages_encoded, embedded)

    def test_a_query_of_only_operators_finds_nothing_rather_than_failing(self):
        """FTS5 has its own query language and a person's question is not
        written in it, so every term is quoted before it gets there."""
        self.library()
        manager = self.manager(min_score=0.3)
        manager.index_path(self.docs)

        for query in ('NEAR AND OR', '"', 'alcohol - "acid', 'a * b'):
            result = manager.search(query)
            self.assertTrue(result["success"], query)

    def test_the_search_terms_are_quoted_not_interpreted(self):
        from rag.metadata import _fts_query

        self.assertEqual(_fts_query("Zaitsev rule E2"), '"Zaitsev" OR "rule" OR "E2"')
        self.assertEqual(_fts_query("NEAR AND"), '"NEAR" OR "AND"')
        self.assertEqual(_fts_query("!!! ?"), "")

    def test_keyword_hits_are_scored_the_same_way_semantic_ones_are(self):
        """A BM25 rank in a column documented as cosine similarity would put
        two different measurements in one field."""
        self.library()
        manager = self.manager(min_score=0.3)
        manager.index_path(self.docs)

        hit = manager.search("Zaitsev")["results"][0]
        rows = manager.store.search_text("Zaitsev", 5)
        vector = manager.embedder.encode_query("Zaitsev")
        expected = manager.index.score_rows(vector, rows)[rows[0]]
        self.assertAlmostEqual(hit["score"], round(expected, 4), places=4)


class OutlineTests(TempCase):
    """Structure: what a document is made of, rather than what matches."""

    def book(self) -> Path:
        path = self.docs / "chemistry.pdf"
        write_pdf(
            path,
            [
                ["Contents"],
                ["Alkanes are saturated hydrocarbons with single bonds only"],
                ["Alkenes contain a carbon carbon double bond"],
                ["Dehydration of an alcohol follows the Zaitsev rule"],
            ],
        )
        add_outline(
            path,
            [
                (1, "Chapter 2 Alkanes", 2),
                (1, "Chapter 3 Alkenes", 3),
                (2, "3.7 Dehydration", 4),
            ],
        )
        return path

    def indexed(self) -> RagManager:
        self.book()
        manager = self.manager(min_score=0.0)
        manager.index_path(self.docs)
        return manager

    def test_a_pdf_gets_its_sections_from_its_own_bookmarks(self):
        """Without this a PDF is the one format producing no structure at all,
        which is backwards: a textbook is where the chapter matters most."""
        manager = self.indexed()
        outline = manager.outline("chemistry.pdf")

        self.assertEqual(
            [section["section"] for section in outline["sections"]],
            ["Chapter 2 Alkanes", "Chapter 3 Alkenes", "3.7 Dehydration"],
        )

    def test_the_outline_says_where_each_section_starts(self):
        outline = self.indexed().outline("chemistry.pdf")
        pages = {s["section"]: s["first_page"] for s in outline["sections"]}
        self.assertEqual(pages["Chapter 2 Alkanes"], 2)
        self.assertEqual(pages["3.7 Dehydration"], 4)

    def test_sections_come_back_in_document_order_not_alphabetical(self):
        outline = self.indexed().outline("chemistry.pdf")
        first = [s["section"] for s in outline["sections"]][0]
        self.assertEqual(first, "Chapter 2 Alkanes")

    def test_a_document_without_headings_says_so_rather_than_looking_empty(self):
        self.write("plain.txt", "Just a paragraph with no headings at all.")
        manager = self.manager()
        manager.index_path(self.docs)

        outline = manager.outline("plain.txt")
        self.assertEqual(outline["count"], 0)
        self.assertIn("no headings", outline["note"])

    def test_an_unknown_document_is_an_error_not_an_empty_outline(self):
        manager = self.indexed()
        with self.assertRaises(RagError):
            manager.outline("nothing.pdf")

    def test_a_search_can_be_narrowed_to_one_section(self):
        """The difference between the best passages in a chapter and whichever
        of the best overall happened to land in it."""
        manager = self.indexed()

        anywhere = manager.search("double bond")
        self.assertIn("Alkenes", anywhere["results"][0]["text"])

        scoped = manager.search("double bond", section="Alkanes")
        self.assertEqual(scoped["count"], 1)
        self.assertIn("Alkanes are saturated", scoped["results"][0]["text"])
        self.assertEqual(scoped["scope"], "Alkanes")

    def test_a_section_is_matched_loosely_because_people_do_not_paste_headings(self):
        manager = self.indexed()
        self.assertEqual(manager.search("alcohol", section="Dehydration")["count"], 1)

    def test_a_search_can_be_narrowed_to_one_document(self):
        self.book()
        self.write("biology.txt", "Alkenes are never mentioned in this file at all.")
        manager = self.manager(min_score=0.0)
        manager.index_path(self.docs)

        scoped = manager.search("alkenes", document="biology.txt")
        self.assertEqual(scoped["count"], 1)
        self.assertEqual(scoped["results"][0]["document"], "biology.txt")

    def test_a_scope_that_matches_nothing_says_what_to_do(self):
        manager = self.indexed()
        empty = manager.search("anything", section="Chapter 99")
        self.assertEqual(empty["count"], 0)
        self.assertIn("get_document_outline", empty["note"])

    def test_an_unknown_document_in_a_search_fails_before_the_model_runs(self):
        manager = self.indexed()
        encoded = self.embedder.queries_encoded
        with self.assertRaises(RagError):
            manager.search("anything", document="nothing.pdf")
        self.assertEqual(self.embedder.queries_encoded, encoded)

    def test_keyword_matching_still_applies_inside_a_scope(self):
        manager = self.indexed()
        found = manager.search("Zaitsev", section="Dehydration", )
        self.assertEqual(found["count"], 1)
        self.assertIn(found["results"][0]["match"], ("keyword", "both"))

    def test_markdown_headings_produce_an_outline_too(self):
        """PDFs are the new part; the other formats already had structure and
        must keep it."""
        self.write(
            "notes.md",
            "# Photosynthesis\n\nLight reactions occur in the thylakoid.\n\n"
            "# Respiration\n\nGlycolysis happens in the cytosol.\n",
        )
        manager = self.manager()
        manager.index_path(self.docs)

        sections = [s["section"] for s in manager.outline("notes.md")["sections"]]
        self.assertEqual(sections, ["Photosynthesis", "Respiration"])


class DocumentToolTests(TempCase):
    def tools(self, **overrides) -> ToolRegistry:
        manager = self.manager()
        manager.index_path(self.docs)
        return ToolRegistry(DocumentSearchTools(manager, **overrides).tools())

    def test_the_tools_are_registered_with_a_schema_the_model_can_read(self):
        self.write("a.txt", "Alpha text.")
        registry = self.tools()
        self.assertEqual(
            registry.names(),
            ["get_document_outline", "list_documents", "search_documents"],
        )
        definition = registry.get_tool("search_documents").definition()
        self.assertEqual(definition["type"], "function")
        self.assertIn("query", definition["function"]["parameters"]["properties"])

    def test_the_description_says_when_to_search_and_when_not_to(self):
        self.write("a.txt", "Alpha text.")
        description = self.tools().get_tool("search_documents").description.lower()
        self.assertIn("my notes", description)
        self.assertIn("do not use", description)
        self.assertIn("arithmetic", description)

    def test_searching_through_the_registry_returns_a_structured_result(self):
        self.write("notes.txt", "Mitochondria generate ATP.")
        result = self.tools().execute("search_documents", {"query": "mitochondria ATP"})
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["results"][0]["document"], "notes.txt")

    def test_a_bad_argument_comes_back_as_an_error_not_an_exception(self):
        self.write("a.txt", "Alpha.")
        result = self.tools().execute("search_documents", {"quer": "typo"})
        self.assertFalse(result.ok)
        self.assertIn("Unknown argument", result.payload["error"])

    def test_an_empty_query_comes_back_as_a_failed_result(self):
        self.write("a.txt", "Alpha.")
        result = self.tools().execute("search_documents", {"query": "  "})
        self.assertFalse(result.ok)

    def test_the_context_budget_drops_the_weakest_results(self):
        for n in range(5):
            self.write(f"n{n}.txt", "Shared words here. " * 40)
        registry = self.tools(context_budget=900)
        result = registry.execute("search_documents", {"query": "shared words here"})
        self.assertTrue(result.ok)
        self.assertLess(len(result.payload["results"]), 5)
        self.assertIn("truncated", result.payload)

    def test_the_budget_always_keeps_at_least_one_result(self):
        self.write("big.txt", "Long document text. " * 200)
        registry = self.tools(context_budget=500)
        result = registry.execute("search_documents", {"query": "long document text"})
        self.assertEqual(len(result.payload["results"]), 1)

    def test_list_documents_through_the_registry(self):
        self.write("a.txt", "Alpha.")
        result = self.tools().execute("list_documents", {})
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["documents"][0]["document"], "a.txt")

    def test_the_tool_raises_its_own_error_type(self):
        manager = self.manager()
        tools = DocumentSearchTools(manager)
        with self.assertRaises(DocumentSearchError):
            tools.search_documents("")


# --- registry wiring ------------------------------------------------------


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config(
            workspace=Path(self._tmp.name),
            rag_store=Path(self._tmp.name) / "rag",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_document_search_is_off_by_default(self):
        registry, disabled = build_default_registry(self.config)
        self.assertNotIn("search_documents", registry)
        self.assertIn("documents", [item.category for item in disabled])

    def test_the_disabled_reason_says_how_to_turn_it_on(self):
        _, disabled = build_default_registry(self.config)
        reason = next(item.reason for item in disabled if item.category == "documents")
        self.assertIn("AGENT_ENABLE_RAG=1", reason)
        self.assertIn("python -m rag index", reason)

    def test_enabling_it_registers_both_tools(self):
        registry, disabled = build_default_registry(
            dataclasses.replace(self.config, rag_enabled=True)
        )
        self.assertIn("search_documents", registry)
        self.assertIn("list_documents", registry)
        self.assertNotIn("documents", [item.category for item in disabled])

    def test_enabling_it_does_not_start_the_embedding_model(self):
        # Building the registry must stay cheap: it happens on every turn.
        from rag import embeddings

        build_default_registry(dataclasses.replace(self.config, rag_enabled=True))
        self.assertFalse(embeddings.sweep_shared())

    def test_the_calculator_and_filesystem_tools_still_work(self):
        registry, _ = build_default_registry(
            dataclasses.replace(self.config, rag_enabled=True)
        )
        self.assertIn("calculate", registry)
        self.assertIn("read_text_file", registry)
        self.assertEqual(
            registry.execute("calculate", {"expression": "25 * 17"}).payload["result"],
            425,
        )

    def test_the_tools_are_grouped_under_their_own_category(self):
        registry, _ = build_default_registry(
            dataclasses.replace(self.config, rag_enabled=True)
        )
        self.assertIn("documents", registry.categories())


# --- the real embedding model ---------------------------------------------


@unittest.skipUnless(
    os.environ.get("RAG_MODEL_TESTS") == "1",
    "set RAG_MODEL_TESTS=1 to run the tests that load BGE (slow: minutes)",
)
class RealModelTests(TempCase):
    """The parts a fake embedder cannot check.

    Skipped by default: loading torch and the model costs more than the whole
    rest of the suite, and it needs the model to have been downloaded.
    """

    def test_the_worker_loads_and_returns_384_dimensional_unit_vectors(self):
        from rag.embeddings import Embedder

        embedder = Embedder(idle_seconds=0.0)
        try:
            vectors = embedder.encode_passages(["hello world", "goodbye world"])
            self.assertEqual(vectors.shape, (2, 384))
            for vector in vectors:
                self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=4)
        finally:
            embedder.unload()

    def test_unloading_stops_the_worker_process(self):
        from rag.embeddings import Embedder

        embedder = Embedder(idle_seconds=0.0)
        embedder.encode_query("anything")
        self.assertTrue(embedder.loaded)
        embedder.unload()
        self.assertFalse(embedder.loaded)

    def test_real_retrieval_finds_the_right_passage(self):
        from rag.embeddings import Embedder

        self.write(
            "biology.txt",
            "Mitochondria generate ATP by oxidative phosphorylation.\n\n"
            "The Calvin cycle fixes carbon dioxide in the chloroplast stroma.\n\n"
            "Ribosomes translate messenger RNA into polypeptide chains.",
        )
        embedder = Embedder(idle_seconds=0.0)
        try:
            manager = self.manager(embedder=embedder, chunk_tokens=20, overlap_tokens=0)
            manager.index_path(self.docs)
            result = manager.search("how does the cell make energy?")
            self.assertTrue(result["results"])
            self.assertIn("ATP", result["results"][0]["text"])
            # A real match should be well clear of the noise floor.
            self.assertGreater(result["results"][0]["score"], 0.5)
        finally:
            embedder.unload()


# --- through the agent loop -----------------------------------------------


class AgentLoopTests(TempCase):
    """The tool as the agent actually reaches it.

    Which tool the model picks is the model's decision and cannot be asserted
    with a scripted client. What these check is everything around that
    decision: that the definition is offered, that a call routes to the RAG
    manager, that the result is what comes back in the tool message, and that a
    turn which calls no tool touches no document.
    """

    def agent(self, **overrides):
        from agent.loop import Agent
        from tests.fake_client import FakeQwenClient

        manager = self.manager()
        manager.index_path(self.docs)
        registry = ToolRegistry(DocumentSearchTools(manager).tools())
        registry.register(
            __import__("tools.calculator", fromlist=["CALCULATOR_TOOL"]).CALCULATOR_TOOL
        )
        client = FakeQwenClient(overrides.pop("responses", []))
        config = Config(workspace=self.tmp, rag_store=self.tmp / "store")
        return Agent(client, config, registry), client

    def test_the_search_tool_is_offered_to_the_model(self):
        from tests.fake_client import text_message

        self.write("notes.txt", "Alpha text.")
        agent, client = self.agent(responses=[text_message("done")])
        agent.send("hello")
        offered = {
            definition["function"]["name"] for definition in client.tools_seen[-1]
        }
        self.assertIn("search_documents", offered)

    def test_a_search_call_reaches_the_documents_and_comes_back_as_a_tool_message(self):
        from tests.fake_client import text_message, tool_call_message

        self.write("biology.txt", "Mitochondria generate ATP for the cell.")
        agent, _ = self.agent(
            responses=[
                tool_call_message(
                    ("search_documents", {"query": "mitochondria ATP"})
                ),
                text_message("Mitochondria generate ATP."),
            ]
        )
        seen = []
        turn = agent.send(
            "according to my notes, what do mitochondria do?",
            observer=lambda event: seen.append(event),
        )

        self.assertEqual([event.call.name for event in seen], ["search_documents"])
        self.assertTrue(seen[0].result.ok)
        self.assertIn("Mitochondria", seen[0].result.payload["results"][0]["text"])
        self.assertEqual(turn.content, "Mitochondria generate ATP.")

        # The retrieved text has to reach the model, or retrieval was pointless.
        tool_messages = [m for m in agent.history if m.get("role") == "tool"]
        self.assertIn("Mitochondria", tool_messages[0]["content"])

    def test_an_ordinary_question_touches_no_documents(self):
        from tests.fake_client import text_message, tool_call_message

        self.write("biology.txt", "Mitochondria generate ATP for the cell.")
        agent, _ = self.agent(
            responses=[
                tool_call_message(("calculate", {"expression": "25 * 17"})),
                text_message("425"),
            ]
        )
        before = self.embedder.queries_encoded
        seen = []
        agent.send("what is 25 x 17?", observer=lambda event: seen.append(event))

        self.assertEqual([event.call.name for event in seen], ["calculate"])
        self.assertEqual(seen[0].result.payload["result"], 425)
        # No query was embedded, so the embedding model was never needed.
        self.assertEqual(self.embedder.queries_encoded, before)

    def test_a_search_that_finds_nothing_tells_the_model_so(self):
        from tests.fake_client import text_message, tool_call_message

        self.write("biology.txt", "Mitochondria generate ATP.")
        manager = self.manager(min_score=0.99)
        manager.index_path(self.docs)
        registry = ToolRegistry(DocumentSearchTools(manager).tools())

        result = registry.execute(
            "search_documents", {"query": "medieval french tapestry restoration"}
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.payload["count"], 0)
        self.assertIn("threshold", result.payload["note"])


# --- the HTTP API ---------------------------------------------------------


class RagApiTests(TempCase):
    """The routes, against a real store and a fake embedder.

    The runtime is the real one with the model manager stubbed, so these
    exercise the actual routes and the actual RagManager rather than a
    convenient re-implementation.
    """

    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient

        from api.main import create_app
        from api.runtime import Runtime
        from tests.test_api import HarnessRuntime, write_registry
        from tests.test_manager import ManagerHarness

        registry_path = write_registry(self.tmp)
        config = Config(
            workspace=self.tmp,
            db_path=self.tmp / "chat.db",
            rag_enabled=True,
            rag_store=self.tmp / "store",
            rag_min_score=0.0,
        )
        harness = ManagerHarness(registry_path)
        self.runtime = HarnessRuntime(config, harness)

        # The routes build their own manager from config; point it at the fake
        # embedder so no model is loaded.
        embedder = self.embedder
        import api.routes.rag as rag_routes

        self._real_manager = rag_routes._manager
        rag_routes._manager = lambda runtime: RagManager(
            runtime.effective_config().rag_store,
            embedder=embedder,
            dimension=DIMENSION,
            min_score=0.0,
        )
        self._client = TestClient(create_app(self.runtime))
        self.client = self._client.__enter__()

    def tearDown(self):
        import api.routes.rag as rag_routes

        rag_routes._manager = self._real_manager
        self._client.__exit__(None, None, None)
        super().tearDown()

    def test_indexing_then_searching_over_http(self):
        self.write("biology.txt", "Mitochondria generate ATP for the cell.")
        indexed = self.client.post("/api/rag/index", json={"path": str(self.docs)})
        self.assertEqual(indexed.status_code, 200)
        self.assertEqual(indexed.json()["indexed"][0]["document"], "biology.txt")

        found = self.client.post(
            "/api/rag/search", json={"query": "mitochondria ATP"}
        )
        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["results"][0]["document"], "biology.txt")

    def test_the_outline_of_a_document_over_http(self):
        self.write(
            "notes.md",
            "# Photosynthesis\n\nLight reactions occur in the thylakoid.\n",
        )
        self.client.post("/api/rag/index", json={"path": str(self.docs)})

        response = self.client.get("/api/rag/documents/notes.md/outline")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["document"], "notes.md")
        self.assertEqual(
            [section["section"] for section in body["sections"]], ["Photosynthesis"]
        )

    def test_an_outline_for_a_document_that_is_not_there_is_a_404(self):
        response = self.client.get("/api/rag/documents/nothing.md/outline")
        self.assertEqual(response.status_code, 404)

    def test_a_search_can_be_scoped_over_http(self):
        self.write(
            "notes.md",
            "# Alkanes\n\nSaturated hydrocarbons with single bonds.\n\n"
            "# Alkenes\n\nContain a carbon carbon double bond.\n",
        )
        self.client.post("/api/rag/index", json={"path": str(self.docs)})

        scoped = self.client.post(
            "/api/rag/search", json={"query": "bond", "section": "Alkanes"}
        )
        self.assertEqual(scoped.status_code, 200, scoped.text)
        body = scoped.json()
        self.assertEqual(body["scope"], "Alkanes")
        self.assertTrue(
            all("Alkanes" in hit["text"] for hit in body["results"]), body["results"]
        )

    def test_a_hit_reports_how_it_was_found_over_http(self):
        self.write("notes.md", "The Zaitsev rule predicts the major product.")
        self.client.post("/api/rag/index", json={"path": str(self.docs)})

        found = self.client.post("/api/rag/search", json={"query": "Zaitsev"})
        self.assertIn(found.json()["results"][0]["match"], ("keyword", "both"))

    def test_listing_and_deleting_a_document(self):
        self.write("a.txt", "Alpha text.")
        self.client.post("/api/rag/index", json={"path": str(self.docs)})

        listing = self.client.get("/api/rag/documents")
        self.assertEqual(listing.status_code, 200)
        document_id = listing.json()["documents"][0]["id"]

        removed = self.client.delete(f"/api/rag/documents/{document_id}")
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json()["document"], "a.txt")
        self.assertEqual(self.client.get("/api/rag/documents").json()["count"], 0)

    def test_deleting_something_that_is_not_there_is_a_404(self):
        self.assertEqual(
            self.client.delete("/api/rag/documents/999").status_code, 404
        )

    def test_indexing_a_missing_path_is_a_400_not_a_500(self):
        response = self.client.post(
            "/api/rag/index", json={"path": str(self.tmp / "nope")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("does not exist", response.json()["detail"])

    def test_an_empty_query_is_rejected_by_the_schema(self):
        self.assertEqual(
            self.client.post("/api/rag/search", json={"query": ""}).status_code, 422
        )

    def test_stats_reports_the_index(self):
        self.write("a.txt", "Alpha text.")
        self.client.post("/api/rag/index", json={"path": str(self.docs)})
        stats = self.client.get("/api/rag/stats").json()
        self.assertEqual(stats["documents"], 1)
        self.assertEqual(stats["dimension"], DIMENSION)

    def test_rebuild_reports_what_it_rebuilt(self):
        self.write("a.txt", "Alpha text.")
        self.client.post("/api/rag/index", json={"path": str(self.docs)})
        response = self.client.post("/api/rag/rebuild")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["indexed"][0]["document"], "a.txt")

    def test_searching_is_allowed_mid_turn_but_indexing_is_not(self):
        # The agent's own tool call must not be refused while its turn runs;
        # an ingest competing for both cores must be.
        self.write("a.txt", "Alpha text.")
        self.client.post("/api/rag/index", json={"path": str(self.docs)})

        class Busy:
            def busy(self):
                return True

            def depth(self):
                return 1

        real_queue = self.runtime.queue
        self.runtime.queue = Busy()
        try:
            self.assertEqual(
                self.client.post("/api/rag/index", json={"path": str(self.docs)}).status_code,
                409,
            )
            self.assertEqual(
                self.client.post("/api/rag/search", json={"query": "alpha"}).status_code,
                200,
            )
        finally:
            self.runtime.queue = real_queue

    def test_the_document_switch_is_listed_and_can_be_toggled(self):
        switches = {
            switch["id"]: switch for switch in self.client.get("/api/tools").json()["switches"]
        }
        self.assertIn("documents", switches)
        self.assertTrue(switches["documents"]["enabled"])

        off = self.client.post("/api/tools/documents", json={"enabled": False})
        self.assertEqual(off.status_code, 200)
        names = [tool["name"] for tool in off.json()["tools"]]
        self.assertNotIn("search_documents", names)

    def test_the_switch_explains_what_turning_it_on_costs(self):
        switches = {
            switch["id"]: switch for switch in self.client.get("/api/tools").json()["switches"]
        }
        self.assertIn("AGENT_ENABLE_RAG=1", switches["documents"]["risk"])
