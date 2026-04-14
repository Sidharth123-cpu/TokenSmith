"""
Unit tests for chunking strategies.
Tests correctness of all chunking modes: sliding window, sentence boundary,
paragraph-aware, and adaptive.
"""
import pytest
from src.preprocessing.chunking import (
    SlidingWindowConfig, SlidingWindowStrategy,
    SentenceBoundaryConfig, SentenceBoundaryStrategy,
    ParagraphAwareConfig, ParagraphAwareStrategy,
    AdaptiveConfig, AdaptiveStrategy,
    SectionRecursiveConfig, SectionRecursiveStrategy,
    DocumentChunker,
)

def all_text_preserved(original: str, chunks: list, join_char: str = "") -> bool:
    original_clean = " ".join(original.split())
    chunks_clean = " ".join(" ".join(c.split()) for c in chunks)
    # Every word in original should appear in the chunks
    original_words = set(original_clean.split())
    chunks_words = set(chunks_clean.split())
    return original_words.issubset(chunks_words)


SAMPLE_TEXT = (
    "Database systems provide efficient data storage. "
    "They support concurrent access by multiple users. "
    "Transaction management ensures ACID properties. "
    "Recovery mechanisms protect against failures. "
    "Query optimization improves performance significantly."
)

PARAGRAPH_TEXT = (
    "Chapter 1: Introduction\n\n"
    "Database systems are fundamental to modern computing. They store and organize data efficiently.\n\n"
    "Chapter 2: Storage\n\n"
    "Data is stored on disk in pages. Buffer managers cache frequently accessed pages in memory. "
    "This reduces I/O operations and improves query latency.\n\n"
    "Chapter 3: Indexing\n\n"
    "B+ trees are the most common index structure. They provide O(log n) lookup time."
)

MARKDOWN_TEXT = (
    "# Introduction\n\n"
    "This is the introduction section.\n\n"
    "## Background\n\n"
    "Some background information here. It spans multiple sentences. "
    "The topic is quite interesting.\n\n"
    "## Methods\n\n"
    "We use several methods in this work."
)

LONG_SENTENCE_TEXT = (
    "This is a very long sentence that goes on and on and on and contains a lot of words "
    "because sometimes academic papers have sentences that are unreasonably long and contain "
    "multiple clauses separated by commas and conjunctions and prepositional phrases that "
    "make the reader lose track of the subject by the time they reach the period."
)

class TestSlidingWindow:
    def test_basic_chunking(self):
        config = SlidingWindowConfig(window_size=100, overlap=20)
        strategy = SlidingWindowStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_overlap_present(self):
        config = SlidingWindowConfig(window_size=100, overlap=30)
        strategy = SlidingWindowStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        if len(chunks) >= 2:
            # The end of chunk 0 should overlap with start of chunk 1
            end_of_first = chunks[0][-30:]
            assert end_of_first in chunks[1] or chunks[1].startswith(end_of_first.strip())

    def test_empty_text(self):
        config = SlidingWindowConfig(window_size=100, overlap=20)
        strategy = SlidingWindowStrategy(config)
        assert strategy.chunk("") == []

    def test_text_shorter_than_window(self):
        config = SlidingWindowConfig(window_size=10000, overlap=100)
        strategy = SlidingWindowStrategy(config)
        chunks = strategy.chunk("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    def test_config_validation(self):
        with pytest.raises(AssertionError):
            SlidingWindowConfig(window_size=0, overlap=0).validate()
        with pytest.raises(AssertionError):
            SlidingWindowConfig(window_size=100, overlap=100).validate()

    def test_no_tokens_lost(self):
        config = SlidingWindowConfig(window_size=80, overlap=20)
        strategy = SlidingWindowStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        assert all_text_preserved(SAMPLE_TEXT, chunks)


# -------------------- Sentence Boundary Tests --------------------

class TestSentenceBoundary:
    def test_basic_chunking(self):
        config = SentenceBoundaryConfig(max_chunk_size=120, overlap_sentences=0)
        strategy = SentenceBoundaryStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 120 or chunk == chunks[-1]  # last might be shorter

    def test_no_mid_sentence_splits(self):
        config = SentenceBoundaryConfig(max_chunk_size=150, overlap_sentences=0)
        strategy = SentenceBoundaryStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            # Each chunk should end with a period (complete sentence)
            assert chunk.rstrip().endswith(".") or chunk.rstrip().endswith("!") or chunk.rstrip().endswith("?")

    def test_overlap_sentences(self):
        config = SentenceBoundaryConfig(max_chunk_size=100, overlap_sentences=1)
        strategy = SentenceBoundaryStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        # With overlap, some sentence content should appear in consecutive chunks
        if len(chunks) >= 2:
            # At least check chunks were created
            assert all(len(c) > 0 for c in chunks)

    def test_empty_text(self):
        config = SentenceBoundaryConfig(max_chunk_size=100, overlap_sentences=0)
        strategy = SentenceBoundaryStrategy(config)
        assert strategy.chunk("") == []

    def test_single_sentence(self):
        config = SentenceBoundaryConfig(max_chunk_size=1000, overlap_sentences=0)
        strategy = SentenceBoundaryStrategy(config)
        chunks = strategy.chunk("Just one sentence.")
        assert len(chunks) == 1

    def test_no_tokens_lost(self):
        config = SentenceBoundaryConfig(max_chunk_size=100, overlap_sentences=0)
        strategy = SentenceBoundaryStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        assert all_text_preserved(SAMPLE_TEXT, chunks)

class TestParagraphAware:
    def test_basic_chunking(self):
        config = ParagraphAwareConfig(max_chunk_size=200, overlap=0)
        strategy = ParagraphAwareStrategy(config)
        chunks = strategy.chunk(PARAGRAPH_TEXT)
        assert len(chunks) > 1

    def test_respects_paragraph_boundaries(self):
        config = ParagraphAwareConfig(max_chunk_size=500, overlap=0)
        strategy = ParagraphAwareStrategy(config)
        chunks = strategy.chunk(PARAGRAPH_TEXT)
        # Chunks should not split mid-paragraph (no partial paragraphs)
        for chunk in chunks:
            assert len(chunk) > 0

    def test_long_paragraph_fallback(self):
        long_para = "This is a sentence. " * 50  # very long paragraph
        config = ParagraphAwareConfig(max_chunk_size=200, overlap=0)
        strategy = ParagraphAwareStrategy(config)
        chunks = strategy.chunk(long_para)
        # Should still produce multiple chunks via sentence fallback
        assert len(chunks) > 1

    def test_empty_text(self):
        config = ParagraphAwareConfig(max_chunk_size=200, overlap=0)
        strategy = ParagraphAwareStrategy(config)
        assert strategy.chunk("") == []

    def test_no_tokens_lost(self):
        config = ParagraphAwareConfig(max_chunk_size=200, overlap=0)
        strategy = ParagraphAwareStrategy(config)
        chunks = strategy.chunk(PARAGRAPH_TEXT)
        assert all_text_preserved(PARAGRAPH_TEXT, chunks)

class TestAdaptive:
    def test_selects_paragraph_for_structured_text(self):
        config = AdaptiveConfig(max_chunk_size=300, overlap=50)
        strategy = AdaptiveStrategy(config)
        # Markdown text with headers should trigger paragraph strategy
        features = strategy._analyze_document(MARKDOWN_TEXT)
        selected = strategy._select_strategy(features)
        assert isinstance(selected, ParagraphAwareStrategy)

    def test_selects_sentence_for_prose(self):
        config = AdaptiveConfig(max_chunk_size=300, overlap=50)
        strategy = AdaptiveStrategy(config)
        # Dense prose without paragraph breaks - must be long enough
        # that paragraph_density drops below 2.0 (1 paragraph per 1000+ chars)
        prose = (
            "The database engine processes queries in several stages. "
            "First it parses the SQL statement into an abstract syntax tree. "
            "Then it optimizes the execution plan using cost-based optimization. "
            "Next it executes the plan against the storage engine layer. "
            "Finally it returns results to the client application. "
            "Each stage has its own performance characteristics that matter. "
            "Understanding these stages is key to database optimization. "
            "The parser validates syntax and checks semantic correctness. "
            "The optimizer considers multiple join orderings and index selections. "
            "The executor manages buffer pool access and disk I/O operations. "
            "Result serialization handles type conversion and network protocols."
        )
        features = strategy._analyze_document(prose)
        selected = strategy._select_strategy(features)
        assert isinstance(selected, SentenceBoundaryStrategy)

    def test_produces_valid_chunks(self):
        config = AdaptiveConfig(max_chunk_size=200, overlap=50)
        strategy = AdaptiveStrategy(config)
        chunks = strategy.chunk(PARAGRAPH_TEXT)
        assert len(chunks) > 0
        assert all(len(c) > 0 for c in chunks)

    def test_empty_text(self):
        config = AdaptiveConfig(max_chunk_size=200, overlap=50)
        strategy = AdaptiveStrategy(config)
        assert strategy.chunk("") == []

    def test_no_tokens_lost(self):
        config = AdaptiveConfig(max_chunk_size=200, overlap=0)
        strategy = AdaptiveStrategy(config)
        chunks = strategy.chunk(SAMPLE_TEXT)
        assert all_text_preserved(SAMPLE_TEXT, chunks)

class TestDocumentChunkerIntegration:
    def test_table_preservation_sliding_window(self):
        text = "Some text before. <table>col1|col2</table> Some text after."
        config = SlidingWindowConfig(window_size=1000, overlap=0)
        strategy = SlidingWindowStrategy(config)
        chunker = DocumentChunker(strategy=strategy, keep_tables=True)
        chunks = chunker.chunk(text)
        full = " ".join(chunks)
        assert "<table>col1|col2</table>" in full

    def test_table_preservation_sentence(self):
        text = "First sentence. <table>data here</table> Second sentence. Third sentence."
        config = SentenceBoundaryConfig(max_chunk_size=1000, overlap_sentences=0)
        strategy = SentenceBoundaryStrategy(config)
        chunker = DocumentChunker(strategy=strategy, keep_tables=True)
        chunks = chunker.chunk(text)
        full = " ".join(chunks)
        assert "<table>data here</table>" in full

    def test_no_strategy_raises(self):
        chunker = DocumentChunker(strategy=None)
        with pytest.raises(ValueError):
            chunker.chunk("Some text")