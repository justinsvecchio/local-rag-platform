"""Protocol definitions for RAG platform components.

These protocols define the interfaces that all implementations must follow,
enabling dependency injection, testing, and component swapping.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from rag.models.document import Chunk, Document
from rag.models.generation import GeneratedAnswer
from rag.models.retrieval import RankedResult, SearchResult


@runtime_checkable
class Parser(Protocol):
    """Protocol for document parsers.

    Parsers convert raw document bytes into structured Document objects
    with extracted text and metadata.
    """

    @abstractmethod
    async def parse(self, content: bytes, filename: str) -> Document:
        """Parse raw bytes into a Document.

        Args:
            content: Raw document bytes
            filename: Original filename (used for type detection)

        Returns:
            Parsed Document with text content and metadata

        Raises:
            ParseError: If parsing fails
        """
        ...

    @property
    @abstractmethod
    def supported_mimetypes(self) -> set[str]:
        """Return set of MIME types this parser supports."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return set of file extensions this parser supports."""
        ...


@runtime_checkable
class Cleaner(Protocol):
    """Protocol for text cleaners.

    Cleaners normalize and clean text content before chunking.
    """

    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Protocol for document chunkers.

    Chunkers split documents into smaller, retrievable chunks
    while preserving context and metadata.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers.

    Embedders convert text into vector representations
    for semantic similarity search.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Some embedding models use different prompts for queries vs documents.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector storage and retrieval.

    Vector indexes store embeddings and enable similarity search
    with optional metadata filtering.
    """

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors.

        Args:
            ids: Vector IDs
            vectors: Embedding vectors
            payloads: Metadata payloads for each vector
        """
        ...

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            vector: Query vector
            limit: Maximum results to return
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: Vector IDs to delete
        """
        ...

    @abstractmethod
    async def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get vectors by ID.

        Args:
            ids: Vector IDs to retrieve

        Returns:
            List of vector records with payloads
        """
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Protocol for lexical/BM25 search.

    Lexical indexes enable keyword-based search
    to complement semantic search.
    """

    @abstractmethod
    def index(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add document to index.

        Args:
            doc_id: Document/chunk ID
            text: Text content to index
            metadata: Document metadata
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using BM25.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        ...

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> None:
        """Delete documents from index.

        Args:
            doc_ids: Document IDs to delete
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk.

        Args:
            path: Path to save index
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk.

        Args:
            path: Path to load index from
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol for retrievers.

    Retrievers coordinate search across different indexes
    and return relevant results.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Protocol for result fusion.

    Fusion strategies combine results from multiple
    retrieval sources (e.g., vector + lexical).
    """

    @abstractmethod
    def fuse(
        self,
        result_lists: list[list[SearchResult]],
        weights: list[float] | None = None,
    ) -> list[SearchResult]:
        """Combine multiple result lists.

        Args:
            result_lists: List of result lists to fuse
            weights: Optional weights for each list

        Returns:
            Fused and sorted results
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerankers.

    Rerankers re-score results based on the query
    to improve ranking quality.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[RankedResult]:
        """Rerank results.

        Args:
            query: The search query
            results: Candidate results to rerank
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the reranker model name."""
        ...


@runtime_checkable
class FreshnessScorer(Protocol):
    """Protocol for freshness scoring.

    Freshness scorers adjust result scores based on
    document age/recency.
    """

    @abstractmethod
    def score(
        self,
        result: SearchResult,
        reference_time: datetime | None = None,
    ) -> float:
        """Calculate freshness score for a result.

        Args:
            result: Search result with document timestamp
            reference_time: Reference time (default: now)

        Returns:
            Freshness score between 0 and 1
        """
        ...

    @abstractmethod
    def apply_to_results(
        self,
        results: list[SearchResult],
        weight: float,
        reference_time: datetime | None = None,
    ) -> list[SearchResult]:
        """Apply freshness scoring to results.

        Args:
            results: Results to score
            weight: Weight for freshness component
            reference_time: Reference time

        Returns:
            Results with adjusted scores
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """Protocol for answer generation.

    Generators use LLMs to produce answers from
    retrieved context with citations.
    """

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: list[RankedResult],
    ) -> GeneratedAnswer:
        """Generate answer with citations.

        Args:
            query: User query
            context: Retrieved and ranked context

        Returns:
            Generated answer with citations
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the LLM model name."""
        ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Protocol for query rewriting.

    Query rewriters expand or reformulate queries
    to improve retrieval quality.
    """

    @abstractmethod
    async def rewrite(self, query: str) -> str:
        """Rewrite a query for better retrieval.

        Args:
            query: Original query

        Returns:
            Rewritten query
        """
        ...

    @abstractmethod
    async def expand(self, query: str, num_expansions: int = 3) -> list[str]:
        """Generate multiple query variations.

        Args:
            query: Original query
            num_expansions: Number of variations to generate

        Returns:
            List of query variations including original
        """
        ...
