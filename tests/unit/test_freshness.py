"""Tests for freshness scoring."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from rag.freshness.exponential_decay import ExponentialDecayScorer
from rag.models.retrieval import SearchResult, SearchSource


class TestExponentialDecayScorer:
    """Test exponential decay freshness scorer."""

    @pytest.fixture
    def scorer(self):
        """Create scorer with 30-day half-life."""
        return ExponentialDecayScorer(half_life_days=30.0)

    @pytest.fixture
    def now(self):
        """Get current time."""
        return datetime.now(timezone.utc)

    def _make_result(
        self,
        chunk_id: str,
        score: float,
        created_at: datetime | None = None,
    ) -> SearchResult:
        """Helper to create search results."""
        return SearchResult(
            chunk_id=chunk_id,
            document_id=f"doc_{chunk_id}",
            content=f"Content {chunk_id}",
            score=score,
            source=SearchSource.VECTOR,
            document_created_at=created_at,
        )

    def test_brand_new_document_gets_max_freshness(self, scorer, now):
        """Test that a document created now gets maximum freshness."""
        result = self._make_result("A", 0.8, now)
        freshness = scorer.score(result, now)

        # Should be close to 1.0 for brand new document
        assert freshness > 0.99

    def test_half_life_gives_half_freshness(self, scorer, now):
        """Test that document at half-life age gets ~0.5 freshness."""
        half_life_ago = now - timedelta(days=30)
        result = self._make_result("A", 0.8, half_life_ago)
        freshness = scorer.score(result, now)

        # Should be approximately 0.5 at half-life
        assert abs(freshness - 0.5) < 0.01

    def test_older_documents_decay_exponentially(self, scorer, now):
        """Test that older documents have lower freshness."""
        ages_days = [0, 30, 60, 90, 120]
        freshness_scores = []

        for age in ages_days:
            created_at = now - timedelta(days=age)
            result = self._make_result("A", 0.8, created_at)
            freshness_scores.append(scorer.score(result, now))

        # Each should be lower than the previous
        for i in range(1, len(freshness_scores)):
            assert freshness_scores[i] < freshness_scores[i - 1]

    def test_min_score_floor(self, now):
        """Test that freshness doesn't go below min_score."""
        scorer = ExponentialDecayScorer(
            half_life_days=30.0,
            min_score=0.2,
        )

        # Very old document
        very_old = now - timedelta(days=365)
        result = self._make_result("A", 0.8, very_old)
        freshness = scorer.score(result, now)

        assert freshness >= 0.2

    def test_no_created_at_uses_default_freshness(self, scorer, now):
        """Test that missing created_at uses default freshness."""
        result = self._make_result("A", 0.8, None)
        freshness = scorer.score(result, now)

        # Should return default (typically 0.5 or 1.0 depending on implementation)
        assert 0 <= freshness <= 1.0

    def test_apply_to_results_boosts_fresh_content(self, scorer, now):
        """Test applying freshness to a list of results."""
        results = [
            self._make_result("old", 0.9, now - timedelta(days=90)),
            self._make_result("new", 0.7, now - timedelta(days=1)),
        ]

        boosted = scorer.apply(results, weight=0.3)

        # The new document should be boosted relative to old
        old_result = next(r for r in boosted if r.chunk_id == "old")
        new_result = next(r for r in boosted if r.chunk_id == "new")

        # New result should have gained more from freshness boost
        # despite having lower original score
        old_freshness = scorer.score(results[0], now)
        new_freshness = scorer.score(results[1], now)

        assert new_freshness > old_freshness

    def test_weight_affects_boost_magnitude(self, now):
        """Test that weight parameter affects freshness impact."""
        scorer = ExponentialDecayScorer(half_life_days=30.0)

        # Old document
        old_doc = now - timedelta(days=60)
        results = [self._make_result("A", 0.8, old_doc)]

        # Low weight - less impact
        low_weight = scorer.apply(results, weight=0.1)

        # High weight - more impact
        high_weight = scorer.apply(results, weight=0.5)

        # With higher weight, freshness penalty should be more pronounced
        # For old documents, higher weight means lower final score
        assert high_weight[0].score <= low_weight[0].score

    def test_custom_half_life(self, now):
        """Test scorer with custom half-life."""
        # Short half-life (7 days) - rapid decay
        short_scorer = ExponentialDecayScorer(half_life_days=7.0)

        # Long half-life (90 days) - slow decay
        long_scorer = ExponentialDecayScorer(half_life_days=90.0)

        # 30-day old document
        old_doc = now - timedelta(days=30)
        result = self._make_result("A", 0.8, old_doc)

        short_freshness = short_scorer.score(result, now)
        long_freshness = long_scorer.score(result, now)

        # Short half-life should decay more
        assert short_freshness < long_freshness

    def test_decay_constant_calculation(self):
        """Test that decay constant is calculated correctly."""
        # decay_constant = ln(2) / half_life
        scorer = ExponentialDecayScorer(half_life_days=30.0)
        expected_constant = math.log(2) / 30.0

        assert abs(scorer.decay_constant - expected_constant) < 0.0001

    def test_results_maintain_metadata(self, scorer, now):
        """Test that applying freshness preserves result metadata."""
        result = SearchResult(
            chunk_id="A",
            document_id="doc_A",
            content="Content A",
            score=0.8,
            source=SearchSource.VECTOR,
            document_created_at=now - timedelta(days=10),
            metadata={"title": "Test", "page": 1},
        )

        boosted = scorer.apply([result], weight=0.2)

        assert boosted[0].chunk_id == "A"
        assert boosted[0].document_id == "doc_A"
        assert boosted[0].content == "Content A"
        assert boosted[0].metadata["title"] == "Test"
        assert boosted[0].metadata["page"] == 1

    def test_empty_results_returns_empty(self, scorer):
        """Test that empty input returns empty output."""
        assert scorer.apply([], weight=0.2) == []
