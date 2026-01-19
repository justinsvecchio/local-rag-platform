"""Index manager for coordinating vector and lexical indexes."""

from typing import Any

from rag.config import get_settings
from rag.index.bm25.index import BM25Index
from rag.index.qdrant.client import QdrantIndex
from rag.models.retrieval import SearchResult


class IndexManager:
    """Manager for coordinating multiple indexes.

    Provides a unified interface for indexing and searching
    across vector and lexical indexes.
    """

    def __init__(
        self,
        vector_index: QdrantIndex | None = None,
        lexical_index: BM25Index | None = None,
        embedding_dimension: int = 1536,
    ):
        """Initialize index manager.

        Args:
            vector_index: Vector index instance
            lexical_index: Lexical index instance
            embedding_dimension: Embedding dimension for vector index
        """
        self._vector_index = vector_index
        self._lexical_index = lexical_index
        self._embedding_dimension = embedding_dimension

    @classmethod
    def from_config(cls) -> "IndexManager":
        """Create index manager from configuration.

        Returns:
            Configured IndexManager instance
        """
        settings = get_settings()

        vector_index = QdrantIndex(
            url=settings.qdrant.url,
            collection_name=settings.qdrant.collection,
            api_key=settings.qdrant.api_key,
        )

        lexical_index = BM25Index()

        # Try to load persisted BM25 index
        try:
            lexical_index.load(settings.bm25_index_path)
        except Exception:
            pass  # No existing index

        return cls(
            vector_index=vector_index,
            lexical_index=lexical_index,
        )

    @property
    def vector_index(self) -> QdrantIndex | None:
        """Get vector index."""
        return self._vector_index

    @property
    def lexical_index(self) -> BM25Index | None:
        """Get lexical index."""
        return self._lexical_index

    async def index_chunks(
        self,
        chunk_ids: list[str],
        contents: list[str],
        embeddings: list[list[float]],
        metadata_list: list[dict[str, Any]],
    ) -> None:
        """Index chunks in both vector and lexical indexes.

        Args:
            chunk_ids: Chunk IDs
            contents: Chunk contents
            embeddings: Chunk embeddings
            metadata_list: Chunk metadata
        """
        # Index in vector store
        if self._vector_index:
            payloads = [
                {"content": content, **metadata}
                for content, metadata in zip(contents, metadata_list)
            ]
            await self._vector_index.upsert(chunk_ids, embeddings, payloads)

        # Index in lexical store
        if self._lexical_index:
            for chunk_id, content, metadata in zip(
                chunk_ids, contents, metadata_list
            ):
                self._lexical_index.index(chunk_id, content, metadata)

    async def search_vector(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search vector index.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        if not self._vector_index:
            return []

        return await self._vector_index.search(
            query_embedding,
            limit,
            filters,
        )

    def search_lexical(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search lexical index.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        if not self._lexical_index:
            return []

        return self._lexical_index.search(query, limit, filters)

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks from both indexes.

        Args:
            chunk_ids: Chunk IDs to delete
        """
        if self._vector_index:
            await self._vector_index.delete(chunk_ids)

        if self._lexical_index:
            self._lexical_index.delete(chunk_ids)

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID
        """
        if self._vector_index:
            await self._vector_index.delete_by_document(document_id)

        # For BM25, we need to find and delete by document_id
        if self._lexical_index:
            to_delete = [
                doc_id
                for doc_id, doc in self._lexical_index._documents.items()
                if doc.get("metadata", {}).get("document_id") == document_id
            ]
            self._lexical_index.delete(to_delete)

    def save_lexical_index(self, path: str | None = None) -> None:
        """Save lexical index to disk.

        Args:
            path: Path to save to (uses config default if None)
        """
        if not self._lexical_index:
            return

        if path is None:
            settings = get_settings()
            path = settings.bm25_index_path

        self._lexical_index.save(path)

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dict with index stats
        """
        stats = {}

        if self._lexical_index:
            stats["lexical_count"] = self._lexical_index.count()

        return stats
