"""Hybrid retriever combining vector and lexical search."""

import asyncio
import time
from typing import Any

from rag.models.generation import RetrievalTrace
from rag.models.retrieval import RetrievalConfig, SearchResult
from rag.protocols import Embedder, FusionStrategy, LexicalIndex, VectorIndex
from rag.retrieve.fusion.rrf import RRFFusion
from rag.retrieve.lexical import LexicalRetriever
from rag.retrieve.vector import VectorRetriever


class HybridRetriever:
    """Hybrid retriever combining vector and lexical search.

    Uses both semantic (vector) and keyword (BM25) search,
    then fuses results using RRF or weighted fusion.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_index: VectorIndex,
        lexical_index: LexicalIndex,
        fusion_strategy: FusionStrategy | None = None,
        config: RetrievalConfig | None = None,
    ):
        """Initialize hybrid retriever.

        Args:
            embedder: Embedder for query encoding
            vector_index: Vector index for semantic search
            lexical_index: Lexical index for BM25 search
            fusion_strategy: Strategy for combining results
            config: Retrieval configuration
        """
        self._vector_retriever = VectorRetriever(embedder, vector_index)
        self._lexical_retriever = LexicalRetriever(lexical_index)
        self._fusion = fusion_strategy or RRFFusion(k=60)
        self._config = config or RetrievalConfig()

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        trace: RetrievalTrace | None = None,
    ) -> list[SearchResult]:
        """Retrieve using hybrid search.

        Runs vector and lexical search in parallel, then fuses results.

        Args:
            query: Search query
            limit: Maximum results (uses config initial_limit if None)
            filters: Optional metadata filters
            trace: Optional trace object for recording steps

        Returns:
            Fused list of SearchResult objects
        """
        limit = limit or self._config.initial_limit

        # Run both searches in parallel
        start_time = time.perf_counter()

        vector_task = self._vector_retriever.retrieve(query, limit, filters)
        lexical_task = self._lexical_retriever.retrieve(query, limit, filters)

        vector_results, lexical_results = await asyncio.gather(
            vector_task, lexical_task
        )

        vector_time = (time.perf_counter() - start_time) * 1000

        # Record to trace
        if trace:
            trace.add_vector_results(vector_results, vector_time)
            trace.add_lexical_results(lexical_results, vector_time)

        # Fuse results
        fusion_start = time.perf_counter()

        weights = [
            self._config.vector_weight,
            self._config.lexical_weight,
        ]

        fused_results = self._fusion.fuse(
            [vector_results, lexical_results],
            weights=weights,
        )

        fusion_time = (time.perf_counter() - fusion_start) * 1000

        if trace:
            trace.add_fused_results(fused_results, fusion_time)

        return fused_results

    async def retrieve_vector_only(
        self,
        query: str,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using only vector search.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters

        Returns:
            List of SearchResult objects
        """
        limit = limit or self._config.initial_limit
        return await self._vector_retriever.retrieve(query, limit, filters)

    async def retrieve_lexical_only(
        self,
        query: str,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using only lexical search.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional filters

        Returns:
            List of SearchResult objects
        """
        limit = limit or self._config.initial_limit
        return await self._lexical_retriever.retrieve(query, limit, filters)

    @property
    def config(self) -> RetrievalConfig:
        """Get retrieval configuration."""
        return self._config

    @config.setter
    def config(self, value: RetrievalConfig) -> None:
        """Set retrieval configuration."""
        self._config = value

    @property
    def fusion(self) -> FusionStrategy:
        """Get fusion strategy."""
        return self._fusion

    @fusion.setter
    def fusion(self, value: FusionStrategy) -> None:
        """Set fusion strategy."""
        self._fusion = value
