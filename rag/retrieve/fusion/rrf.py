"""Reciprocal Rank Fusion (RRF) implementation."""

from collections import defaultdict

from rag.models.retrieval import SearchResult, SearchSource


class RRFFusion:
    """Reciprocal Rank Fusion for combining ranked result lists.

    RRF is a simple but effective method for combining results
    from multiple retrieval systems. It's robust to score
    differences between systems and doesn't require normalization.

    Formula: RRF_score(d) = sum(1 / (k + rank_i(d))) for all lists i
    """

    def __init__(self, k: int = 60):
        """Initialize RRF fusion.

        Args:
            k: RRF constant. Higher values reduce the impact of
               top-ranked documents. Default of 60 is empirically
               effective across many benchmarks.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def fuse(
        self,
        result_lists: list[list[SearchResult]],
        weights: list[float] | None = None,
    ) -> list[SearchResult]:
        """Combine multiple ranked result lists using RRF.

        Args:
            result_lists: List of ranked result lists
            weights: Optional weights for each list (default: equal)

        Returns:
            Fused results sorted by RRF score
        """
        if not result_lists:
            return []

        # Filter empty lists
        result_lists = [r for r in result_lists if r]
        if not result_lists:
            return []

        # Default to equal weights
        if weights is None:
            weights = [1.0] * len(result_lists)
        elif len(weights) != len(result_lists):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of result lists ({len(result_lists)})"
            )

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]

        # Calculate RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        result_data: dict[str, SearchResult] = {}

        for weight, results in zip(weights, result_lists):
            for rank, result in enumerate(results, start=1):
                # RRF formula: weight / (k + rank)
                rrf_score = weight / (self.k + rank)
                rrf_scores[result.chunk_id] += rrf_score

                # Keep the result with highest individual score
                if result.chunk_id not in result_data or \
                   result.score > result_data[result.chunk_id].score:
                    result_data[result.chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )

        # Build fused results
        fused_results = []
        for chunk_id in sorted_ids:
            original = result_data[chunk_id]
            fused_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=original.document_id,
                    content=original.content,
                    score=rrf_scores[chunk_id],
                    source=SearchSource.HYBRID,
                    metadata=original.metadata,
                    document_created_at=original.document_created_at,
                    document_updated_at=original.document_updated_at,
                )
            )

        return fused_results

    def fuse_with_scores(
        self,
        result_lists: list[list[SearchResult]],
        weights: list[float] | None = None,
    ) -> list[tuple[SearchResult, dict[str, float]]]:
        """Fuse results and return detailed scoring information.

        Useful for debugging and analysis.

        Args:
            result_lists: List of ranked result lists
            weights: Optional weights for each list

        Returns:
            List of (result, score_breakdown) tuples
        """
        if not result_lists:
            return []

        result_lists = [r for r in result_lists if r]
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Track individual contributions
        rrf_scores: dict[str, float] = defaultdict(float)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        result_data: dict[str, SearchResult] = {}

        for list_idx, (weight, results) in enumerate(zip(weights, result_lists)):
            for rank, result in enumerate(results, start=1):
                rrf_score = weight / (self.k + rank)
                rrf_scores[result.chunk_id] += rrf_score
                contributions[result.chunk_id][f"list_{list_idx}"] = rrf_score

                if result.chunk_id not in result_data or \
                   result.score > result_data[result.chunk_id].score:
                    result_data[result.chunk_id] = result

        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )

        results = []
        for chunk_id in sorted_ids:
            original = result_data[chunk_id]
            fused = SearchResult(
                chunk_id=chunk_id,
                document_id=original.document_id,
                content=original.content,
                score=rrf_scores[chunk_id],
                source=SearchSource.HYBRID,
                metadata=original.metadata,
                document_created_at=original.document_created_at,
                document_updated_at=original.document_updated_at,
            )
            score_breakdown = {
                "total": rrf_scores[chunk_id],
                **contributions[chunk_id],
            }
            results.append((fused, score_breakdown))

        return results
