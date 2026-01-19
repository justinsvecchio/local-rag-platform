"""Weighted score fusion implementation."""

from collections import defaultdict

from rag.models.retrieval import SearchResult, SearchSource


class WeightedFusion:
    """Weighted score fusion for combining result lists.

    Combines results by normalizing scores and applying weights.
    More sensitive to score magnitudes than RRF.
    """

    def __init__(
        self,
        normalize: bool = True,
        score_floor: float = 0.0,
    ):
        """Initialize weighted fusion.

        Args:
            normalize: Whether to normalize scores to [0, 1]
            score_floor: Minimum score floor after normalization
        """
        self.normalize = normalize
        self.score_floor = score_floor

    def fuse(
        self,
        result_lists: list[list[SearchResult]],
        weights: list[float] | None = None,
    ) -> list[SearchResult]:
        """Combine result lists using weighted score fusion.

        Args:
            result_lists: List of ranked result lists
            weights: Weights for each list (default: equal)

        Returns:
            Fused results sorted by combined score
        """
        if not result_lists:
            return []

        result_lists = [r for r in result_lists if r]
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Optionally normalize scores within each list
        if self.normalize:
            result_lists = [
                self._normalize_scores(results)
                for results in result_lists
            ]

        # Combine scores
        combined_scores: dict[str, float] = defaultdict(float)
        result_data: dict[str, SearchResult] = {}

        for weight, results in zip(weights, result_lists):
            for result in results:
                combined_scores[result.chunk_id] += weight * result.score

                if result.chunk_id not in result_data:
                    result_data[result.chunk_id] = result

        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        # Build fused results
        fused_results = []
        for chunk_id in sorted_ids:
            original = result_data[chunk_id]
            score = max(combined_scores[chunk_id], self.score_floor)

            fused_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=original.document_id,
                    content=original.content,
                    score=score,
                    source=SearchSource.HYBRID,
                    metadata=original.metadata,
                    document_created_at=original.document_created_at,
                    document_updated_at=original.document_updated_at,
                )
            )

        return fused_results

    def _normalize_scores(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Normalize scores to [0, 1] range.

        Uses min-max normalization.

        Args:
            results: Results to normalize

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score

        if range_score == 0:
            # All scores are the same
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=1.0,
                    source=r.source,
                    metadata=r.metadata,
                    document_created_at=r.document_created_at,
                    document_updated_at=r.document_updated_at,
                )
                for r in results
            ]

        return [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                content=r.content,
                score=(r.score - min_score) / range_score,
                source=r.source,
                metadata=r.metadata,
                document_created_at=r.document_created_at,
                document_updated_at=r.document_updated_at,
            )
            for r in results
        ]
