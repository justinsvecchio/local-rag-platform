"""Pytest configuration and fixtures."""

import pytest

from rag.models.document import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    DocumentType,
)
from rag.models.retrieval import SearchResult, SearchSource


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    return Document(
        id="doc_001",
        content="This is a sample document for testing. It contains multiple sentences. Each sentence provides some information.",
        doc_type=DocumentType.TXT,
        metadata=DocumentMetadata(
            title="Sample Document",
            author="Test Author",
        ),
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="chunk_001",
            content="This is the first chunk of content.",
            metadata=ChunkMetadata(
                document_id="doc_001",
                chunk_index=0,
                start_char=0,
                end_char=40,
            ),
        ),
        Chunk(
            id="chunk_002",
            content="This is the second chunk of content.",
            metadata=ChunkMetadata(
                document_id="doc_001",
                chunk_index=1,
                start_char=40,
                end_char=80,
            ),
        ),
    ]


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample search results for testing."""
    return [
        SearchResult(
            chunk_id="chunk_001",
            document_id="doc_001",
            content="First result content about the query topic.",
            score=0.95,
            source=SearchSource.VECTOR,
            metadata={"title": "First Doc"},
        ),
        SearchResult(
            chunk_id="chunk_002",
            document_id="doc_002",
            content="Second result content with related information.",
            score=0.85,
            source=SearchSource.VECTOR,
            metadata={"title": "Second Doc"},
        ),
        SearchResult(
            chunk_id="chunk_003",
            document_id="doc_003",
            content="Third result with additional context.",
            score=0.75,
            source=SearchSource.LEXICAL,
            metadata={"title": "Third Doc"},
        ),
    ]


@pytest.fixture
def vector_results() -> list[SearchResult]:
    """Create vector search results for fusion testing."""
    return [
        SearchResult(
            chunk_id="A",
            document_id="doc_A",
            content="Content A",
            score=0.95,
            source=SearchSource.VECTOR,
        ),
        SearchResult(
            chunk_id="B",
            document_id="doc_B",
            content="Content B",
            score=0.85,
            source=SearchSource.VECTOR,
        ),
        SearchResult(
            chunk_id="C",
            document_id="doc_C",
            content="Content C",
            score=0.75,
            source=SearchSource.VECTOR,
        ),
    ]


@pytest.fixture
def lexical_results() -> list[SearchResult]:
    """Create lexical search results for fusion testing."""
    return [
        SearchResult(
            chunk_id="C",
            document_id="doc_C",
            content="Content C",
            score=0.90,
            source=SearchSource.LEXICAL,
        ),
        SearchResult(
            chunk_id="B",
            document_id="doc_B",
            content="Content B",
            score=0.80,
            source=SearchSource.LEXICAL,
        ),
        SearchResult(
            chunk_id="D",
            document_id="doc_D",
            content="Content D",
            score=0.70,
            source=SearchSource.LEXICAL,
        ),
    ]
