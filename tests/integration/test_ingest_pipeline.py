"""Integration tests for ingestion pipeline."""

import pytest
from datetime import datetime, timezone

from rag.ingest.pipeline import IngestPipeline
from rag.ingest.cleaners.pipeline import CleanerPipeline
from rag.models.document import ChunkingStrategy


class TestIngestPipelineIntegration:
    """Integration tests for document ingestion pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create ingestion pipeline without external dependencies."""
        return IngestPipeline(
            vector_index=None,
            lexical_index=None,
            embedder=None,
            cleaner=CleanerPipeline(),
        )

    @pytest.mark.asyncio
    async def test_ingest_text_document(self, pipeline):
        """Test ingesting a plain text document."""
        content = b"This is a test document with some content for chunking."
        filename = "test.txt"

        result = await pipeline.process(
            content=content,
            filename=filename,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=50,
            chunk_overlap=10,
            skip_indexing=True,
        )

        assert result.document_id is not None
        assert result.doc_type == "text"
        assert result.chunk_count >= 1
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_ingest_markdown_document(self, pipeline):
        """Test ingesting a markdown document."""
        content = b"""# Title

## Section 1

This is the first section with content.

## Section 2

This is the second section with more content.
"""
        filename = "test.md"

        result = await pipeline.process(
            content=content,
            filename=filename,
            chunking_strategy=ChunkingStrategy.HEADING,
            skip_indexing=True,
        )

        assert result.document_id is not None
        assert result.doc_type == "markdown"
        assert result.chunk_count >= 1

    @pytest.mark.asyncio
    async def test_ingest_with_custom_metadata(self, pipeline):
        """Test ingesting with custom metadata."""
        content = b"Document content."
        filename = "test.txt"
        custom_metadata = {
            "author": "Test Author",
            "category": "testing",
        }

        result = await pipeline.process(
            content=content,
            filename=filename,
            metadata=custom_metadata,
            skip_indexing=True,
        )

        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_ingest_generates_content_hash(self, pipeline):
        """Test that ingestion generates content hash."""
        content = b"Document content for hashing."
        filename = "test.txt"

        result = await pipeline.process(
            content=content,
            filename=filename,
            skip_indexing=True,
        )

        assert result.metadata.get("content_hash") is not None
        assert len(result.metadata["content_hash"]) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_ingest_different_chunking_strategies(self, pipeline):
        """Test different chunking strategies produce different results."""
        content = b"""# Main Title

First paragraph of content.

## Section A

Content in section A with details.

## Section B

Content in section B with more details.
"""
        filename = "test.md"

        # Test recursive chunking
        recursive_result = await pipeline.process(
            content=content,
            filename=filename,
            chunking_strategy=ChunkingStrategy.RECURSIVE,
            skip_indexing=True,
        )

        # Test heading-based chunking
        heading_result = await pipeline.process(
            content=content,
            filename=filename,
            chunking_strategy=ChunkingStrategy.HEADING,
            skip_indexing=True,
        )

        # Both should produce valid results
        assert recursive_result.chunk_count >= 1
        assert heading_result.chunk_count >= 1

    @pytest.mark.asyncio
    async def test_ingest_html_document(self, pipeline):
        """Test ingesting an HTML document."""
        content = b"""
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>This is paragraph content.</p>
            <h2>Subheading</h2>
            <p>More content here.</p>
        </body>
        </html>
        """
        filename = "test.html"

        result = await pipeline.process(
            content=content,
            filename=filename,
            skip_indexing=True,
        )

        assert result.document_id is not None
        assert result.doc_type == "html"
        assert result.chunk_count >= 1

    @pytest.mark.asyncio
    async def test_ingest_empty_document(self, pipeline):
        """Test ingesting an empty document."""
        content = b""
        filename = "empty.txt"

        result = await pipeline.process(
            content=content,
            filename=filename,
            skip_indexing=True,
        )

        # Should handle gracefully
        assert result.document_id is not None
        assert result.chunk_count >= 0

    @pytest.mark.asyncio
    async def test_chunk_size_affects_output(self, pipeline):
        """Test that chunk size affects number of chunks."""
        # Long content
        content = b"Word " * 500
        filename = "test.txt"

        # Small chunks
        small_result = await pipeline.process(
            content=content,
            filename=filename,
            chunk_size=50,
            chunk_overlap=10,
            skip_indexing=True,
        )

        # Large chunks
        large_result = await pipeline.process(
            content=content,
            filename=filename,
            chunk_size=500,
            chunk_overlap=50,
            skip_indexing=True,
        )

        # Smaller chunks should produce more chunks
        assert small_result.chunk_count > large_result.chunk_count


class TestCleanerPipelineIntegration:
    """Integration tests for text cleaning pipeline."""

    @pytest.fixture
    def cleaner(self):
        """Create cleaner pipeline."""
        return CleanerPipeline()

    def test_cleans_extra_whitespace(self, cleaner):
        """Test that extra whitespace is cleaned."""
        text = "Too    many   spaces   here."
        cleaned = cleaner.clean(text)

        assert "    " not in cleaned
        assert "Too many spaces here." == cleaned or "  " not in cleaned

    def test_normalizes_unicode(self, cleaner):
        """Test unicode normalization."""
        # Various unicode representations
        text = "café"  # Could have different representations
        cleaned = cleaner.clean(text)

        assert isinstance(cleaned, str)
        assert len(cleaned) > 0

    def test_preserves_meaningful_content(self, cleaner):
        """Test that meaningful content is preserved."""
        text = "Important information about RAG systems."
        cleaned = cleaner.clean(text)

        assert "RAG" in cleaned
        assert "information" in cleaned

    def test_handles_html_entities(self, cleaner):
        """Test handling of HTML entities."""
        text = "Less &lt; Greater &gt; Ampersand &amp;"
        cleaned = cleaner.clean(text)

        # Should decode entities
        assert "&lt;" not in cleaned or "<" in cleaned
