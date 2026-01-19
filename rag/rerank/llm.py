"""LLM-based reranker using language models."""

import asyncio
import json
import re
from typing import Any

from rag.models.retrieval import RankedResult, SearchResult


class LLMReranker:
    """LLM-based reranker for result reranking.

    Uses a language model to score query-document relevance.
    More expensive but can handle complex relevance judgments.
    """

    # Prompt template for reranking
    RERANK_PROMPT = """You are a relevance judge. Given a query and a document, rate the document's relevance to the query on a scale of 0-10.

Query: {query}

Document: {document}

Provide only a number from 0-10 representing relevance, where:
- 0: Completely irrelevant
- 5: Somewhat relevant
- 10: Highly relevant and directly answers the query

Relevance score:"""

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 5,
    ):
        """Initialize LLM reranker.

        Args:
            llm_client: OpenAI client instance
            model: Model name to use
            max_concurrent: Maximum concurrent API calls
        """
        self._client = llm_client
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            except ImportError as e:
                raise ImportError(
                    "LLM reranking requires 'openai' package"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[RankedResult]:
        """Rerank results using LLM.

        Args:
            query: Search query
            results: Candidate results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Score all results concurrently
        scores = await asyncio.gather(
            *[self._score_result(query, r) for r in results]
        )

        # Combine and sort
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build ranked results
        ranked = []
        for rank, (result, score) in enumerate(scored_results[:top_k], start=1):
            ranked.append(
                RankedResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    original_score=result.score,
                    rerank_score=score,
                    final_score=score,
                    rank=rank,
                    source=result.source,
                    metadata=result.metadata,
                    freshness_score=result.metadata.get("freshness_score"),
                    document_created_at=result.document_created_at,
                    document_updated_at=result.document_updated_at,
                )
            )

        return ranked

    async def _score_result(
        self,
        query: str,
        result: SearchResult,
    ) -> float:
        """Score a single query-document pair.

        Args:
            query: Search query
            result: Search result to score

        Returns:
            Relevance score (0-10 normalized to 0-1)
        """
        async with self._semaphore:
            try:
                prompt = self.RERANK_PROMPT.format(
                    query=query,
                    document=result.content[:2000],  # Truncate long docs
                )

                response = await self.client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0,
                )

                # Parse score from response
                score_text = response.choices[0].message.content.strip()
                score = self._parse_score(score_text)

                # Normalize to 0-1
                return score / 10.0

            except Exception:
                # Return original score on error
                return result.score

    def _parse_score(self, text: str) -> float:
        """Parse score from LLM response.

        Args:
            text: Response text

        Returns:
            Parsed score (0-10)
        """
        # Try to extract a number
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            score = float(match.group(1))
            return min(10.0, max(0.0, score))

        # Default to middle score if parsing fails
        return 5.0
