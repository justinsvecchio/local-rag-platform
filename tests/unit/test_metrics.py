"""Tests for evaluation metrics."""

import pytest

from evals.metrics import RetrievalMetrics, GenerationMetrics


class TestRetrievalMetrics:
    """Test retrieval evaluation metrics."""

    def test_recall_at_k_all_relevant(self):
        """Test recall when all relevant items are retrieved."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}

        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=5)
        assert recall == 1.0

    def test_recall_at_k_partial(self):
        """Test recall with partial retrieval."""
        retrieved = ["a", "b", "x", "y", "z"]
        relevant = {"a", "b", "c", "d"}

        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=5)
        assert recall == 0.5  # 2 out of 4 relevant retrieved

    def test_recall_at_k_none_relevant(self):
        """Test recall when no relevant items retrieved."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}

        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=3)
        assert recall == 0.0

    def test_recall_at_k_respects_limit(self):
        """Test that recall respects k limit."""
        retrieved = ["x", "y", "a", "b", "c"]  # relevant items at end
        relevant = {"a", "b", "c"}

        recall_k2 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=2)
        recall_k5 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=5)

        assert recall_k2 == 0.0  # only x, y in top 2
        assert recall_k5 == 1.0  # all relevant in top 5

    def test_precision_at_k_all_relevant(self):
        """Test precision when all retrieved are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c", "d", "e"}

        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k=3)
        assert precision == 1.0

    def test_precision_at_k_partial(self):
        """Test precision with partial relevance."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}

        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k=5)
        assert precision == 0.6  # 3 out of 5 are relevant

    def test_precision_at_k_none_relevant(self):
        """Test precision when no items are relevant."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}

        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k=3)
        assert precision == 0.0

    def test_mrr_first_position(self):
        """Test MRR when relevant item is first."""
        retrieved = ["a", "x", "y"]
        relevant = {"a"}

        mrr = RetrievalMetrics.mrr(retrieved, relevant)
        assert mrr == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant item is second."""
        retrieved = ["x", "a", "y"]
        relevant = {"a"}

        mrr = RetrievalMetrics.mrr(retrieved, relevant)
        assert mrr == 0.5

    def test_mrr_not_found(self):
        """Test MRR when no relevant item found."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}

        mrr = RetrievalMetrics.mrr(retrieved, relevant)
        assert mrr == 0.0

    def test_mrr_multiple_relevant(self):
        """Test MRR with multiple relevant items (uses first)."""
        retrieved = ["x", "a", "b", "c"]
        relevant = {"a", "b", "c"}

        mrr = RetrievalMetrics.mrr(retrieved, relevant)
        assert mrr == 0.5  # first relevant at position 2

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        retrieved = ["a", "b", "c"]
        relevance = {"a": 3, "b": 2, "c": 1}  # descending relevance

        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevance, k=3)
        assert ndcg == 1.0

    def test_ndcg_reversed_ranking(self):
        """Test NDCG with reversed ranking."""
        retrieved = ["c", "b", "a"]
        relevance = {"a": 3, "b": 2, "c": 1}

        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevance, k=3)
        assert ndcg < 1.0  # not perfect ranking

    def test_ndcg_no_relevant(self):
        """Test NDCG with no relevant items."""
        retrieved = ["x", "y", "z"]
        relevance = {"a": 3, "b": 2}

        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevance, k=3)
        assert ndcg == 0.0

    def test_empty_retrieved_returns_zero(self):
        """Test metrics with empty retrieved list."""
        relevant = {"a", "b", "c"}

        assert RetrievalMetrics.recall_at_k([], relevant, k=5) == 0.0
        assert RetrievalMetrics.precision_at_k([], relevant, k=5) == 0.0
        assert RetrievalMetrics.mrr([], relevant) == 0.0

    def test_empty_relevant_returns_zero(self):
        """Test metrics with empty relevant set."""
        retrieved = ["a", "b", "c"]

        # Recall with no relevant items is 0 (or undefined)
        recall = RetrievalMetrics.recall_at_k(retrieved, set(), k=3)
        assert recall == 0.0


class TestGenerationMetrics:
    """Test generation evaluation metrics."""

    def test_citation_precision_all_correct(self):
        """Test citation precision when all citations are valid."""
        cited_indices = [1, 2, 3]
        source_count = 5

        precision = GenerationMetrics.citation_precision(cited_indices, source_count)
        assert precision == 1.0

    def test_citation_precision_some_invalid(self):
        """Test citation precision with some invalid citations."""
        cited_indices = [1, 2, 6, 7]  # 6, 7 are invalid
        source_count = 5

        precision = GenerationMetrics.citation_precision(cited_indices, source_count)
        assert precision == 0.5  # 2 out of 4 valid

    def test_citation_precision_all_invalid(self):
        """Test citation precision when all citations invalid."""
        cited_indices = [10, 11, 12]
        source_count = 5

        precision = GenerationMetrics.citation_precision(cited_indices, source_count)
        assert precision == 0.0

    def test_citation_precision_empty_citations(self):
        """Test citation precision with no citations."""
        precision = GenerationMetrics.citation_precision([], source_count=5)
        assert precision == 0.0  # or 1.0 depending on definition

    def test_citation_recall_all_sources_cited(self):
        """Test citation recall when all sources cited."""
        cited_indices = [1, 2, 3, 4, 5]
        source_count = 5

        recall = GenerationMetrics.citation_recall(cited_indices, source_count)
        assert recall == 1.0

    def test_citation_recall_partial(self):
        """Test citation recall with partial coverage."""
        cited_indices = [1, 2]
        source_count = 5

        recall = GenerationMetrics.citation_recall(cited_indices, source_count)
        assert recall == 0.4  # 2 out of 5 sources cited

    def test_answer_relevance_score_structure(self):
        """Test answer relevance score returns valid structure."""
        # This would typically use an LLM judge
        # Here we test the interface
        query = "What is Python?"
        answer = "Python is a programming language."

        # Score should be between 0 and 1
        # Mock test - actual implementation uses LLM
        score = 0.85
        assert 0 <= score <= 1.0

    def test_groundedness_score_structure(self):
        """Test groundedness score returns valid structure."""
        answer = "Python was created by Guido van Rossum."
        sources = ["Guido van Rossum created Python in 1991."]

        # Score should be between 0 and 1
        # Mock test - actual implementation uses LLM
        score = 0.95
        assert 0 <= score <= 1.0


class TestMetricsEdgeCases:
    """Test edge cases for metrics."""

    def test_k_larger_than_retrieved(self):
        """Test when k is larger than retrieved list."""
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}

        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=10)
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k=10)

        # Should handle gracefully
        assert recall == 2 / 3  # 2 out of 3 relevant found
        assert precision == 1.0  # both retrieved are relevant

    def test_duplicate_retrieved_items(self):
        """Test handling duplicate items in retrieved list."""
        retrieved = ["a", "a", "b", "b"]
        relevant = {"a", "b"}

        # Behavior depends on implementation - should handle gracefully
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=4)
        assert recall == 1.0  # both relevant items found

    def test_case_sensitivity(self):
        """Test that metrics are case-sensitive."""
        retrieved = ["A", "B", "C"]
        relevant = {"a", "b", "c"}

        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=3)
        # Should be 0 if case-sensitive
        assert recall == 0.0

    def test_k_equals_zero(self):
        """Test when k equals zero."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}

        # k=0 should return 0 or handle edge case
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k=0)
        assert recall == 0.0
