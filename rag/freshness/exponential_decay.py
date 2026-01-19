"""Exponential decay freshness scorer."""

import math
from datetime import datetime, timezone

from rag.models.retrieval import SearchResult, SearchSource


class ExponentialDecayScorer:
    """Freshness scorer using exponential time decay.

    Documents decay in relevance over time according to:
    freshness = exp(-lambda * age_days)

    where lambda = ln(2) / half_life_days

    This means a document's freshness score is 0.5 at the half-life,
    and continues to decay exponentially.
    """

    def __init__(
        self,
        half_life_days: float = 30.0,
        min_score: float = 0.1,
        max_boost: float = 1.5,
    ):
        """Initialize exponential decay scorer.

        Args:
            half_life_days: Days until freshness decays to 50%
            min_score: Minimum freshness score floor
            max_boost: Maximum boost for very fresh content
        """
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")

        self.half_life_days = half_life_days
        self.decay_constant = math.log(2) / half_life_days
        self.min_score = min_score
        self.max_boost = max_boost

    def score(
        self,
        result: SearchResult,
        reference_time: datetime | None = None,
    ) -> float:
        """Calculate freshness score for a result.

        Args:
            result: Search result with document timestamp
            reference_time: Reference time (default: now UTC)

        Returns:
            Freshness score between min_score and max_boost
        """
        if result.document_created_at is None:
            # Neutral score for undated documents
            return 1.0

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        # Ensure timezone awareness
        doc_time = result.document_created_at
        if doc_time.tzinfo is None:
            doc_time = doc_time.replace(tzinfo=timezone.utc)

        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        # Calculate age in days
        age_delta = reference_time - doc_time
        age_days = age_delta.total_seconds() / (24 * 3600)

        # Handle future dates (slight boost for very fresh content)
        if age_days < 0:
            # Linear boost for future dates (capped)
            boost = min(self.max_boost, 1.0 + abs(age_days) / 7)
            return boost

        # Exponential decay
        raw_score = math.exp(-self.decay_constant * age_days)

        # Apply floor
        return max(self.min_score, raw_score)

    def apply_to_results(
        self,
        results: list[SearchResult],
        weight: float = 0.2,
        reference_time: datetime | None = None,
    ) -> list[SearchResult]:
        """Apply freshness scoring to search results.

        Adjusts scores using the formula:
        final_score = original_score * (1 - weight) + original_score * freshness * weight

        This blends the original relevance score with freshness-adjusted score.

        Args:
            results: Results to score
            weight: Weight for freshness component (0-1)
            reference_time: Reference time for scoring

        Returns:
            Results with adjusted scores, sorted by new score
        """
        if not results:
            return results

        weight = max(0.0, min(1.0, weight))  # Clamp to [0, 1]

        boosted = []
        for result in results:
            freshness = self.score(result, reference_time)

            # Blend original score with freshness-adjusted score
            adjusted_score = (
                result.score * (1 - weight) +
                result.score * freshness * weight
            )

            # Create new result with adjusted score and freshness in metadata
            boosted.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    score=adjusted_score,
                    source=result.source,
                    metadata={
                        **result.metadata,
                        "freshness_score": freshness,
                        "original_score": result.score,
                    },
                    document_created_at=result.document_created_at,
                    document_updated_at=result.document_updated_at,
                )
            )

        # Re-sort by adjusted score
        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted

    def get_decay_at_days(self, days: float) -> float:
        """Get the decay factor at a given number of days.

        Useful for understanding/debugging the decay curve.

        Args:
            days: Number of days

        Returns:
            Decay factor (freshness score)
        """
        if days < 0:
            return min(self.max_boost, 1.0 + abs(days) / 7)
        raw = math.exp(-self.decay_constant * days)
        return max(self.min_score, raw)

    def days_to_score(self, target_score: float) -> float:
        """Calculate days until score reaches target.

        Args:
            target_score: Target freshness score

        Returns:
            Days until that score is reached
        """
        if target_score >= 1.0:
            return 0.0
        if target_score <= self.min_score:
            return float("inf")

        # Solve: target = exp(-lambda * days)
        # days = -ln(target) / lambda
        return -math.log(target_score) / self.decay_constant
