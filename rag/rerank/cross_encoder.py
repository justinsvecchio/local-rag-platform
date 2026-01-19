"""Cross-encoder reranker using sentence-transformers."""

import asyncio
from typing import Any

from rag.models.retrieval import RankedResult, SearchResult


class CrossEncoderReranker:
    """Cross-encoder reranker for improving result ranking.

    Uses a cross-encoder model to jointly encode query-document
    pairs and produce relevance scores. More accurate than
    bi-encoder similarity but slower.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for reranking
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None

    @property
    def model(self):
        """Lazy load cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self._model_name,
                    device=self._device,
                    max_length=self._max_length,
                )
            except ImportError as e:
                raise ImportError(
                    "Cross-encoder reranking requires 'sentence-transformers' package"
                ) from e
        return self._model

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[RankedResult]:
        """Rerank results using cross-encoder.

        Args:
            query: The search query
            results: Candidate results to rerank
            top_k: Number of results to return

        Returns:
            Top-k reranked results
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [(query, r.content) for r in results]

        # Run inference in thread pool (model is synchronous)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            ),
        )

        # Combine with original results
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build ranked results
        ranked = []
        for rank, (result, rerank_score) in enumerate(
            scored_results[:top_k], start=1
        ):
            ranked.append(
                RankedResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    original_score=result.score,
                    rerank_score=float(rerank_score),
                    final_score=float(rerank_score),
                    rank=rank,
                    source=result.source,
                    metadata=result.metadata,
                    freshness_score=result.metadata.get("freshness_score"),
                    document_created_at=result.document_created_at,
                    document_updated_at=result.document_updated_at,
                )
            )

        return ranked

    async def rerank_with_freshness(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
        freshness_weight: float = 0.2,
    ) -> list[RankedResult]:
        """Rerank results with freshness integration.

        Combines rerank score with freshness score:
        final = rerank * (1 - weight) + rerank * freshness * weight

        Args:
            query: Search query
            results: Candidate results
            top_k: Number of results to return
            freshness_weight: Weight for freshness (0-1)

        Returns:
            Reranked results with freshness-adjusted scores
        """
        # First, get reranked results
        ranked = await self.rerank(query, results, len(results))

        # Adjust scores based on freshness
        for result in ranked:
            freshness = result.freshness_score or 1.0
            result.final_score = (
                result.rerank_score * (1 - freshness_weight) +
                result.rerank_score * freshness * freshness_weight
            )

        # Re-sort by final score
        ranked.sort(key=lambda x: x.final_score, reverse=True)

        # Re-assign ranks and limit
        for i, result in enumerate(ranked[:top_k], start=1):
            result.rank = i

        return ranked[:top_k]
