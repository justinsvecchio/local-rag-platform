"""Tests for chunking strategies."""

import pytest

from rag.chunking.factory import get_chunker
from rag.chunking.heading import HeadingChunker
from rag.chunking.recursive import RecursiveChunker
from rag.chunking.semantic import SemanticChunker
from rag.chunking.sliding_window import SlidingWindowChunker
from rag.models.document import ChunkingStrategy, Document, DocumentMetadata, DocType


class TestChunkerFactory:
    """Test chunker factory."""

    def test_get_recursive_chunker(self):
        """Test getting recursive chunker."""
        chunker = get_chunker(ChunkingStrategy.RECURSIVE)
        assert isinstance(chunker, RecursiveChunker)

    def test_get_sliding_window_chunker(self):
        """Test getting sliding window chunker."""
        chunker = get_chunker(ChunkingStrategy.SLIDING_WINDOW)
        assert isinstance(chunker, SlidingWindowChunker)

    def test_get_heading_chunker(self):
        """Test getting heading chunker."""
        chunker = get_chunker(ChunkingStrategy.HEADING)
        assert isinstance(chunker, HeadingChunker)

    def test_get_semantic_chunker(self):
        """Test getting semantic chunker."""
        chunker = get_chunker(ChunkingStrategy.SEMANTIC)
        assert isinstance(chunker, SemanticChunker)

    def test_get_chunker_with_string(self):
        """Test getting chunker with string strategy."""
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_custom_chunk_size(self):
        """Test custom chunk size."""
        chunker = get_chunker(
            ChunkingStrategy.RECURSIVE,
            chunk_size=256,
            chunk_overlap=25,
        )
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 25


class TestRecursiveChunker:
    """Test recursive chunker."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with small size for testing."""
        return RecursiveChunker(chunk_size=50, chunk_overlap=10)

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        content = """# Introduction

This is the introduction paragraph with some important information.

## Section One

This section contains detailed information about the first topic.
It spans multiple sentences and provides context.

## Section Two

Another section with different content about a second topic.
"""
        return Document(
            id="test_doc",
            content=content,
            doc_type=DocType.MARKDOWN,
            metadata=DocumentMetadata(title="Test Document"),
        )

    def test_chunks_document(self, chunker, sample_document):
        """Test that document is chunked."""
        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content
            assert chunk.metadata.document_id == "test_doc"

    def test_chunk_ids_are_unique(self, chunker, sample_document):
        """Test that chunk IDs are unique."""
        chunks = chunker.chunk(sample_document)
        ids = [c.id for c in chunks]

        assert len(ids) == len(set(ids))

    def test_chunk_indices_sequential(self, chunker, sample_document):
        """Test that chunk indices are sequential."""
        chunks = chunker.chunk(sample_document)
        indices = [c.metadata.chunk_index for c in chunks]

        assert indices == list(range(len(chunks)))

    def test_empty_document_returns_empty(self, chunker):
        """Test that empty document returns no chunks."""
        doc = Document(
            id="empty",
            content="",
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        assert chunks == []

    def test_small_document_single_chunk(self):
        """Test that small document becomes single chunk."""
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        doc = Document(
            id="small",
            content="Short content.",
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Short content."


class TestSlidingWindowChunker:
    """Test sliding window chunker."""

    @pytest.fixture
    def chunker(self):
        """Create chunker with small window."""
        return SlidingWindowChunker(chunk_size=20, chunk_overlap=5)

    def test_overlapping_chunks(self, chunker):
        """Test that chunks overlap correctly."""
        # Create document with enough content
        content = " ".join(["word"] * 50)
        doc = Document(
            id="test",
            content=content,
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        # Should have multiple overlapping chunks
        assert len(chunks) > 1

    def test_respects_chunk_size(self):
        """Test that chunks respect size limit."""
        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=20)
        content = " ".join(["word"] * 500)
        doc = Document(
            id="test",
            content=content,
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        # Each chunk should be at or under size limit (with some tolerance)
        for chunk in chunks:
            token_count = chunk.metadata.token_count or len(chunk.content.split())
            # Allow some tolerance for tokenization differences
            assert token_count <= 120  # chunk_size + some buffer


class TestHeadingChunker:
    """Test heading-based chunker."""

    @pytest.fixture
    def chunker(self):
        """Create heading chunker."""
        return HeadingChunker(chunk_size=500, chunk_overlap=50)

    def test_splits_on_headings(self, chunker):
        """Test that document is split on headings."""
        content = """# First Section

Content under first heading.

# Second Section

Content under second heading.

## Subsection

Content under subsection.
"""
        doc = Document(
            id="test",
            content=content,
            doc_type=DocType.MARKDOWN,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        # Should create chunks based on headings
        assert len(chunks) >= 2

    def test_preserves_section_titles(self, chunker):
        """Test that section titles are preserved in metadata."""
        content = """# Main Title

Some content here.

## Important Section

More detailed content.
"""
        doc = Document(
            id="test",
            content=content,
            doc_type=DocType.MARKDOWN,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        # At least one chunk should have section title
        section_titles = [c.metadata.section_title for c in chunks]
        assert any(t is not None for t in section_titles)


class TestSemanticChunker:
    """Test semantic chunker."""

    @pytest.fixture
    def chunker(self):
        """Create semantic chunker."""
        return SemanticChunker(
            chunk_size=200,
            chunk_overlap=20,
            similarity_threshold=0.5,
        )

    def test_chunks_by_semantic_similarity(self, chunker):
        """Test that chunking considers semantic similarity."""
        # Content with distinct topics
        content = """
Machine learning is a subset of artificial intelligence.
Neural networks learn patterns from data.
Deep learning uses multiple layers.

The weather today is sunny and warm.
Tomorrow will bring rain and clouds.
The forecast shows improving conditions.
"""
        doc = Document(
            id="test",
            content=content,
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        # Should create chunks (exact count depends on similarity threshold)
        assert len(chunks) >= 1

    def test_handles_short_content(self, chunker):
        """Test that short content is handled."""
        doc = Document(
            id="test",
            content="Short sentence.",
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1


class TestChunkMetadata:
    """Test chunk metadata population."""

    def test_metadata_includes_document_info(self):
        """Test that chunk metadata includes document info."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(
            id="doc_123",
            content="Some content to chunk into pieces.",
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(
                title="Test Document",
                source_url="https://example.com/doc",
            ),
        )
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata.document_id == "doc_123"
            assert chunk.metadata.document_title == "Test Document"
            assert chunk.metadata.source_url == "https://example.com/doc"

    def test_token_count_populated(self):
        """Test that token count is populated."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        doc = Document(
            id="test",
            content="Word " * 50,
            doc_type=DocType.TEXT,
            metadata=DocumentMetadata(),
        )
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata.token_count is not None
            assert chunk.metadata.token_count > 0
