"""Integration tests for retrieval pipeline."""

import pytest
from datetime import datetime, timedelta, timezone

from rag.models.retrieval import SearchResult, SearchSource, RetrievalConfig
from rag.retrieve.fusion.rrf import RRFFusion
from rag.freshness.exponential_decay import ExponentialDecayScorer


class TestRetrievalPipelineIntegration:
    """Integration tests for the retrieval pipeline."""

    @pytest.fixture
    def vector_results(self):
        """Simulated vector search results."""
        now = datetime.now(timezone.utc)
        return [
            SearchResult(
                chunk_id="v1",
                document_id="doc_1",
                content="Machine learning is a subset of artificial intelligence.",
                score=0.95,
                source=SearchSource.VECTOR,
                document_created_at=now - timedelta(days=5),
            ),
            SearchResult(
                chunk_id="v2",
                document_id="doc_2",
                content="Deep learning uses neural networks with many layers.",
                score=0.88,
                source=SearchSource.VECTOR,
                document_created_at=now - timedelta(days=30),
            ),
            SearchResult(
                chunk_id="v3",
                document_id="doc_3",
                content="Natural language processing enables text understanding.",
                score=0.82,
                source=SearchSource.VECTOR,
                document_created_at=now - timedelta(days=60),
            ),
        ]

    @pytest.fixture
    def lexical_results(self):
        """Simulated lexical search results."""
        now = datetime.now(timezone.utc)
        return [
            SearchResult(
                chunk_id="l1",
                document_id="doc_4",
                content="AI and machine learning are transforming industries.",
                score=0.90,
                source=SearchSource.LEXICAL,
                document_created_at=now - timedelta(days=2),
            ),
            SearchResult(
                chunk_id="v1",  # Overlap with vector
                document_id="doc_1",
                content="Machine learning is a subset of artificial intelligence.",
                score=0.85,
                source=SearchSource.LEXICAL,
                document_created_at=now - timedelta(days=5),
            ),
            SearchResult(
                chunk_id="l3",
                document_id="doc_5",
                content="Python is commonly used for machine learning projects.",
                score=0.75,
                source=SearchSource.LEXICAL,
                document_created_at=now - timedelta(days=90),
            ),
        ]

    def test_rrf_fusion_combines_results(self, vector_results, lexical_results):
        """Test that RRF fusion combines results from both sources."""
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([vector_results, lexical_results])

        # Should have all unique chunks
        chunk_ids = {r.chunk_id for r in fused}
        expected_ids = {"v1", "v2", "v3", "l1", "l3"}
        assert chunk_ids == expected_ids

    def test_rrf_boosts_overlapping_results(self, vector_results, lexical_results):
        """Test that items in both lists rank higher."""
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([vector_results, lexical_results])

        scores = {r.chunk_id: r.score for r in fused}

        # v1 appears in both lists, should have highest score
        v1_score = scores["v1"]
        for chunk_id, score in scores.items():
            if chunk_id != "v1":
                assert v1_score >= score, f"v1 should rank highest, but {chunk_id} has higher score"

    def test_freshness_boosts_recent_content(self, vector_results):
        """Test that freshness scoring boosts recent content."""
        scorer = ExponentialDecayScorer(half_life_days=30.0)

        boosted = scorer.apply(vector_results, weight=0.3)

        # Recent content (v1, 5 days old) should get boosted
        # Old content (v3, 60 days old) should be penalized
        v1 = next(r for r in boosted if r.chunk_id == "v1")
        v3 = next(r for r in boosted if r.chunk_id == "v3")

        # v1 started with 0.95, v3 with 0.82
        # After freshness boost, v1 should still be higher
        assert v1.score > v3.score

    def test_full_pipeline_flow(self, vector_results, lexical_results):
        """Test the full retrieval pipeline flow."""
        # 1. Fusion
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([vector_results, lexical_results])

        # 2. Freshness scoring
        scorer = ExponentialDecayScorer(half_life_days=30.0)
        fresh_boosted = scorer.apply(fused, weight=0.2)

        # 3. Results should be ordered by score
        for i in range(len(fresh_boosted) - 1):
            assert fresh_boosted[i].score >= fresh_boosted[i + 1].score

        # 4. All results should have HYBRID source after fusion
        for result in fresh_boosted:
            assert result.source == SearchSource.HYBRID

    def test_weighted_fusion(self, vector_results, lexical_results):
        """Test weighted RRF fusion."""
        fusion = RRFFusion(k=60)

        # Heavy weight on vector
        vector_heavy = fusion.fuse(
            [vector_results, lexical_results],
            weights=[0.8, 0.2],
        )

        # Heavy weight on lexical
        lexical_heavy = fusion.fuse(
            [vector_results, lexical_results],
            weights=[0.2, 0.8],
        )

        # Results should differ in ranking
        vector_heavy_ids = [r.chunk_id for r in vector_heavy]
        lexical_heavy_ids = [r.chunk_id for r in lexical_heavy]

        # The orderings might differ due to weighting
        # At minimum, scores should differ
        vector_heavy_scores = {r.chunk_id: r.score for r in vector_heavy}
        lexical_heavy_scores = {r.chunk_id: r.score for r in lexical_heavy}

        # v2 only in vector, should rank higher with vector weight
        # l1 only in lexical, should rank higher with lexical weight
        assert vector_heavy_scores.get("v2", 0) >= lexical_heavy_scores.get("v2", 0) - 0.01
        assert lexical_heavy_scores.get("l1", 0) >= vector_heavy_scores.get("l1", 0) - 0.01


class TestRetrievalConfigIntegration:
    """Test retrieval configuration options."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = RetrievalConfig()

        assert config.top_k > 0
        assert config.vector_weight >= 0
        assert config.lexical_weight >= 0
        assert config.enable_reranking is not None
        assert config.enable_freshness is not None

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = RetrievalConfig(
            top_k=20,
            vector_weight=0.7,
            lexical_weight=0.3,
            enable_reranking=True,
            enable_freshness=True,
            freshness_weight=0.3,
        )

        assert config.top_k == 20
        assert config.vector_weight == 0.7
        assert config.lexical_weight == 0.3
        assert config.enable_reranking is True
        assert config.enable_freshness is True
        assert config.freshness_weight == 0.3


class TestMetadataFilteringIntegration:
    """Test metadata filtering in retrieval."""

    @pytest.fixture
    def results_with_metadata(self):
        """Create results with various metadata."""
        return [
            SearchResult(
                chunk_id="a1",
                document_id="doc_a",
                content="Content A",
                score=0.9,
                source=SearchSource.VECTOR,
                metadata={"category": "tech", "language": "en"},
            ),
            SearchResult(
                chunk_id="b1",
                document_id="doc_b",
                content="Content B",
                score=0.85,
                source=SearchSource.VECTOR,
                metadata={"category": "science", "language": "en"},
            ),
            SearchResult(
                chunk_id="c1",
                document_id="doc_c",
                content="Content C",
                score=0.8,
                source=SearchSource.VECTOR,
                metadata={"category": "tech", "language": "es"},
            ),
        ]

    def test_filter_by_category(self, results_with_metadata):
        """Test filtering results by category."""
        tech_results = [
            r for r in results_with_metadata
            if r.metadata.get("category") == "tech"
        ]

        assert len(tech_results) == 2
        assert all(r.metadata["category"] == "tech" for r in tech_results)

    def test_filter_by_multiple_criteria(self, results_with_metadata):
        """Test filtering by multiple criteria."""
        filtered = [
            r for r in results_with_metadata
            if r.metadata.get("category") == "tech"
            and r.metadata.get("language") == "en"
        ]

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "a1"
