"""Qdrant vector index implementation."""

from datetime import datetime
from typing import Any

from rag.models.retrieval import SearchResult, SearchSource


class QdrantIndex:
    """Qdrant vector index for semantic search.

    Provides vector storage with metadata filtering.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "rag_chunks",
        api_key: str | None = None,
        embedding_dimension: int = 1536,
        distance: str = "Cosine",
    ):
        """Initialize Qdrant index.

        Args:
            url: Qdrant server URL
            collection_name: Name of the collection
            api_key: Optional API key for Qdrant Cloud
            embedding_dimension: Dimension of embedding vectors
            distance: Distance metric ('Cosine', 'Euclid', 'Dot')
        """
        self._url = url
        self._collection_name = collection_name
        self._api_key = api_key
        self._embedding_dimension = embedding_dimension
        self._distance = distance
        self._client = None

    @property
    def client(self):
        """Lazy load Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key,
                )
            except ImportError as e:
                raise ImportError(
                    "Qdrant index requires 'qdrant-client' package"
                ) from e
        return self._client

    async def ensure_collection(self) -> None:
        """Ensure collection exists with proper schema."""
        from qdrant_client.models import Distance, VectorParams

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        collections = self.client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dimension,
                    distance=distance_map.get(self._distance, Distance.COSINE),
                ),
            )

            # Create payload indexes for common filters
            self.client.create_payload_index(
                collection_name=self._collection_name,
                field_name="document_id",
                field_schema="keyword",
            )

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
            payloads: Metadata payloads
        """
        from qdrant_client.models import PointStruct

        await self.ensure_collection()

        points = [
            PointStruct(id=id_, vector=vec, payload=payload)
            for id_, vec, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self,
        vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            vector: Query vector
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        await self.ensure_collection()

        query_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=self._collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            SearchResult(
                chunk_id=str(r.id),
                document_id=r.payload.get("document_id", ""),
                content=r.payload.get("content", ""),
                score=r.score,
                source=SearchSource.VECTOR,
                metadata={
                    k: v
                    for k, v in r.payload.items()
                    if k not in ("document_id", "content")
                },
                document_created_at=self._parse_datetime(
                    r.payload.get("document_created_at")
                ),
            )
            for r in results
        ]

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: Vector IDs to delete
        """
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=ids),
        )

    async def delete_by_document(self, document_id: str) -> None:
        """Delete all vectors for a document.

        Args:
            document_id: Document ID
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )

    async def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get vectors by ID.

        Args:
            ids: Vector IDs to retrieve

        Returns:
            List of records with payloads
        """
        results = self.client.retrieve(
            collection_name=self._collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": str(r.id),
                **r.payload,
            }
            for r in results
        ]

    def _build_filter(self, filters: dict[str, Any]):
        """Build Qdrant filter from dict.

        Args:
            filters: Filter dict

        Returns:
            Qdrant Filter object
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
            Range,
        )

        conditions = []

        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(
                        FieldCondition(key=key, range=Range(**value))
                    )
                elif "any" in value:
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value["any"]))
                    )
            elif isinstance(value, list):
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse datetime string."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    async def count(self) -> int:
        """Get total number of vectors in collection."""
        await self.ensure_collection()
        info = self.client.get_collection(self._collection_name)
        return info.points_count
