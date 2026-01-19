"""Vector retriever using semantic search."""

from typing import Any

from rag.models.retrieval import SearchResult
from rag.protocols import Embedder, VectorIndex


class VectorRetriever:
    """Retriever using vector similarity search.

    Converts queries to embeddings and searches the vector index
    for semantically similar documents.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_index: VectorIndex,
    ):
        """Initialize vector retriever.

        Args:
            embedder: Embedder for query encoding
            vector_index: Vector index for similarity search
        """
        self._embedder = embedder
        self._vector_index = vector_index

    async def retrieve(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents using vector search.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = await self._embedder.embed_query(query)

        # Search vector index
        results = await self._vector_index.search(
            vector=query_embedding,
            limit=limit,
            filters=filters,
        )

        return results

    @property
    def embedder(self) -> Embedder:
        """Get the embedder."""
        return self._embedder

    @property
    def index(self) -> VectorIndex:
        """Get the vector index."""
        return self._vector_index
