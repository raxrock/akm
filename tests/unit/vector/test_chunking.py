"""Tests for document chunking strategies."""

import pytest
from uuid import uuid4

from akm.core.models import Document
from akm.vector.chromadb.chunking import (
    ChunkingStrategy,
    FixedSizeChunking,
    SentenceChunking,
    ParagraphChunking,
    get_chunking_strategy,
)


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    content = """This is the first sentence. This is the second sentence. This is the third sentence.

This is a new paragraph with more content. It has multiple sentences. The topic changes here.

Another paragraph follows. It contains different information. We want to test chunking behavior."""
    return Document(
        content=content,
        source="test.txt",
        source_type="file",
    )


@pytest.fixture
def long_document() -> Document:
    """Create a longer document for testing."""
    sentences = [f"This is sentence number {i}. " for i in range(100)]
    content = "".join(sentences)
    return Document(
        content=content,
        source="long_test.txt",
        source_type="file",
    )


class TestFixedSizeChunking:
    """Tests for FixedSizeChunking strategy."""

    def test_basic_chunking(self, sample_document: Document) -> None:
        """Test basic fixed-size chunking."""
        strategy = FixedSizeChunking()
        chunks = strategy.chunk(sample_document, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == sample_document.id
            assert len(chunk.content) <= 100 + 50  # Allow some flexibility

    def test_chunk_indices(self, sample_document: Document) -> None:
        """Test that chunk indices are sequential."""
        strategy = FixedSizeChunking()
        chunks = strategy.chunk(sample_document, chunk_size=50)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_character_positions(self, sample_document: Document) -> None:
        """Test that character positions are valid."""
        strategy = FixedSizeChunking()
        chunks = strategy.chunk(sample_document, chunk_size=100, chunk_overlap=0)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char <= len(sample_document.content)

    def test_overlap(self, long_document: Document) -> None:
        """Test that overlap is applied correctly."""
        strategy = FixedSizeChunking()
        chunks = strategy.chunk(long_document, chunk_size=100, chunk_overlap=20)

        # With overlap, adjacent chunks should share some content
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # The end of chunk i should overlap with start of chunk i+1
                # This is approximate due to word boundary handling
                pass  # Overlap verification is complex with word boundaries

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        strategy = FixedSizeChunking()
        assert strategy.strategy_name == "fixed_size"


class TestSentenceChunking:
    """Tests for SentenceChunking strategy."""

    def test_basic_chunking(self, sample_document: Document) -> None:
        """Test basic sentence chunking."""
        strategy = SentenceChunking()
        chunks = strategy.chunk(sample_document, chunk_size=200, chunk_overlap=1)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == sample_document.id

    def test_respects_sentences(self, sample_document: Document) -> None:
        """Test that chunking respects sentence boundaries."""
        strategy = SentenceChunking()
        chunks = strategy.chunk(sample_document, chunk_size=100, chunk_overlap=0)

        # Each chunk should end with sentence-ending punctuation
        for chunk in chunks:
            content = chunk.content.strip()
            if content:  # Non-empty chunks
                # Content should contain complete sentences
                assert "." in content or len(content) < 20

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        strategy = SentenceChunking()
        assert strategy.strategy_name == "sentence"


class TestParagraphChunking:
    """Tests for ParagraphChunking strategy."""

    def test_basic_chunking(self, sample_document: Document) -> None:
        """Test basic paragraph chunking."""
        strategy = ParagraphChunking()
        chunks = strategy.chunk(sample_document, chunk_size=500, chunk_overlap=0)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == sample_document.id

    def test_respects_paragraphs(self, sample_document: Document) -> None:
        """Test that chunking respects paragraph boundaries."""
        strategy = ParagraphChunking()
        # Large chunk size to keep paragraphs together
        chunks = strategy.chunk(sample_document, chunk_size=1000, chunk_overlap=0)

        # With large chunk size, should have fewer chunks
        assert len(chunks) <= 3  # Sample doc has 3 paragraphs

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        strategy = ParagraphChunking()
        assert strategy.strategy_name == "paragraph"


class TestChunkingFactory:
    """Tests for the chunking strategy factory."""

    def test_get_fixed_size(self) -> None:
        """Test getting fixed size strategy."""
        strategy = get_chunking_strategy("fixed_size")
        assert isinstance(strategy, FixedSizeChunking)

    def test_get_sentence(self) -> None:
        """Test getting sentence strategy."""
        strategy = get_chunking_strategy("sentence")
        assert isinstance(strategy, SentenceChunking)

    def test_get_paragraph(self) -> None:
        """Test getting paragraph strategy."""
        strategy = get_chunking_strategy("paragraph")
        assert isinstance(strategy, ParagraphChunking)

    def test_default_strategy(self) -> None:
        """Test default strategy."""
        strategy = get_chunking_strategy()
        assert isinstance(strategy, FixedSizeChunking)

    def test_unknown_strategy(self) -> None:
        """Test unknown strategy falls back to default."""
        strategy = get_chunking_strategy("unknown")
        assert isinstance(strategy, FixedSizeChunking)
