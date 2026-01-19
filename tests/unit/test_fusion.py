"""Tests for RRF fusion."""

import pytest

from rag.models.retrieval import SearchResult, SearchSource
from rag.retrieve.fusion.rrf import RRFFusion


class TestRRFFusion:
    """Test RRF fusion implementation."""

    @pytest.fixture
    def fusion(self):
        """Create RRF fusion with default k."""
        return RRFFusion(k=60)

    def test_fuse_empty_lists(self, fusion):
        """Test fusion with empty lists."""
        assert fusion.fuse([]) == []
        assert fusion.fuse([[], []]) == []

    def test_fuse_single_list(self, fusion):
        """Test fusion with single list."""
        results = [
            SearchResult(
                chunk_id="A",
                document_id="doc_A",
                content="A",
                score=0.9,
                source=SearchSource.VECTOR,
            )
        ]

        fused = fusion.fuse([results])

        assert len(fused) == 1
        assert fused[0].chunk_id == "A"
        # RRF score = 1/(60+1) for first rank
        expected_score = 1 / 61
        assert abs(fused[0].score - expected_score) < 0.0001

    def test_fuse_combines_results(self, vector_results, lexical_results):
        """Test that fusion combines results from both lists."""
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([vector_results, lexical_results])

        chunk_ids = {r.chunk_id for r in fused}
        assert chunk_ids == {"A", "B", "C", "D"}

    def test_fuse_overlapping_items_rank_higher(
        self, vector_results, lexical_results
    ):
        """Test that items in both lists rank higher."""
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([vector_results, lexical_results])

        # B and C appear in both lists, should have higher scores
        scores = {r.chunk_id: r.score for r in fused}

        # Items in both lists should score higher than items in only one
        assert scores["B"] > scores["A"]  # B in both, A only in vector
        assert scores["C"] > scores["D"]  # C in both, D only in lexical

    def test_fuse_with_weights(self, vector_results, lexical_results):
        """Test weighted fusion."""
        fusion = RRFFusion(k=60)

        # Heavy weight on vector
        fused_vector_heavy = fusion.fuse(
            [vector_results, lexical_results],
            weights=[0.9, 0.1],
        )

        # Heavy weight on lexical
        fused_lexical_heavy = fusion.fuse(
            [vector_results, lexical_results],
            weights=[0.1, 0.9],
        )

        # A (only in vector) should rank higher with vector weight
        vector_heavy_rank = next(
            i for i, r in enumerate(fused_vector_heavy) if r.chunk_id == "A"
        )
        lexical_heavy_rank = next(
            i for i, r in enumerate(fused_lexical_heavy) if r.chunk_id == "A"
        )

        assert vector_heavy_rank < lexical_heavy_rank

    def test_fuse_source_is_hybrid(self, vector_results, lexical_results):
        """Test that fused results have HYBRID source."""
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([vector_results, lexical_results])

        for result in fused:
            assert result.source == SearchSource.HYBRID

    def test_fuse_preserves_metadata(self):
        """Test that fusion preserves original metadata."""
        results = [
            SearchResult(
                chunk_id="A",
                document_id="doc_A",
                content="Content A",
                score=0.9,
                source=SearchSource.VECTOR,
                metadata={"title": "Title A", "page": 1},
            )
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse([results])

        assert fused[0].metadata["title"] == "Title A"
        assert fused[0].metadata["page"] == 1

    def test_rrf_score_formula(self):
        """Test that RRF score formula is correct."""
        fusion = RRFFusion(k=60)

        # Create result at rank 1
        results = [
            SearchResult(
                chunk_id="A",
                document_id="doc",
                content="A",
                score=1.0,
                source=SearchSource.VECTOR,
            )
        ]

        fused = fusion.fuse([results])

        # RRF score for rank 1 with k=60: 1/(60+1) = 1/61
        expected = 1 / 61
        assert abs(fused[0].score - expected) < 0.0001

    def test_fuse_with_scores_detail(self, vector_results, lexical_results):
        """Test fusion with detailed score breakdown."""
        fusion = RRFFusion(k=60)
        fused_with_scores = fusion.fuse_with_scores(
            [vector_results, lexical_results]
        )

        # Should have score breakdown
        for result, breakdown in fused_with_scores:
            assert "total" in breakdown
            assert breakdown["total"] == result.score
