"""Evaluation metrics for RAG system."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvalResult:
    """Result of an evaluation metric."""

    metric_name: str
    score: float
    details: dict[str, Any] | None = None


class RetrievalMetrics:
    """Retrieval evaluation metrics."""

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> EvalResult:
        """Calculate Recall@K.

        Recall@K = |relevant in top-k| / |total relevant|

        Args:
            retrieved_ids: Ordered list of retrieved IDs
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            EvalResult with recall score
        """
        if not relevant_ids:
            return EvalResult(
                metric_name=f"recall@{k}",
                score=1.0,
                details={"k": k, "note": "no relevant docs"},
            )

        top_k = set(retrieved_ids[:k])
        hits = len(top_k & relevant_ids)
        score = hits / len(relevant_ids)

        return EvalResult(
            metric_name=f"recall@{k}",
            score=score,
            details={"hits": hits, "total_relevant": len(relevant_ids), "k": k},
        )

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> EvalResult:
        """Calculate Precision@K.

        Precision@K = |relevant in top-k| / k

        Args:
            retrieved_ids: Ordered list of retrieved IDs
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            EvalResult with precision score
        """
        top_k = retrieved_ids[:k]
        if not top_k:
            return EvalResult(
                metric_name=f"precision@{k}",
                score=0.0,
                details={"k": k},
            )

        hits = len(set(top_k) & relevant_ids)
        score = hits / len(top_k)

        return EvalResult(
            metric_name=f"precision@{k}",
            score=score,
            details={"hits": hits, "retrieved": len(top_k), "k": k},
        )

    @staticmethod
    def mrr(
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> EvalResult:
        """Calculate Mean Reciprocal Rank.

        MRR = 1 / rank of first relevant result

        Args:
            retrieved_ids: Ordered list of retrieved IDs
            relevant_ids: Set of relevant document IDs

        Returns:
            EvalResult with MRR score
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return EvalResult(
                    metric_name="mrr",
                    score=1.0 / rank,
                    details={"first_relevant_rank": rank},
                )

        return EvalResult(
            metric_name="mrr",
            score=0.0,
            details={"first_relevant_rank": None},
        )

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevance_scores: dict[str, float],
        k: int,
    ) -> EvalResult:
        """Calculate Normalized Discounted Cumulative Gain at K.

        Args:
            retrieved_ids: Ordered list of retrieved IDs
            relevance_scores: Dict mapping doc IDs to relevance scores
            k: Number of top results to consider

        Returns:
            EvalResult with NDCG score
        """

        def dcg(scores: list[float]) -> float:
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))

        # Actual DCG
        actual_scores = [
            relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]
        ]
        actual_dcg = dcg(actual_scores)

        # Ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_scores)

        if ideal_dcg == 0:
            score = 0.0
        else:
            score = actual_dcg / ideal_dcg

        return EvalResult(
            metric_name=f"ndcg@{k}",
            score=score,
            details={"actual_dcg": actual_dcg, "ideal_dcg": ideal_dcg, "k": k},
        )


class GenerationMetrics:
    """Generation quality metrics."""

    @staticmethod
    def citation_precision(
        citations: list[dict],
        relevant_chunk_ids: set[str],
    ) -> EvalResult:
        """Calculate citation precision.

        Citation Precision = |correct citations| / |total citations|

        Args:
            citations: List of citation dicts with chunk_id
            relevant_chunk_ids: Set of relevant chunk IDs

        Returns:
            EvalResult with citation precision
        """
        if not citations:
            return EvalResult(
                metric_name="citation_precision",
                score=1.0,
                details={"note": "no citations"},
            )

        correct = sum(
            1
            for c in citations
            if c.get("chunk_id") in relevant_chunk_ids
        )
        score = correct / len(citations)

        return EvalResult(
            metric_name="citation_precision",
            score=score,
            details={"correct": correct, "total": len(citations)},
        )

    @staticmethod
    async def groundedness(
        answer: str,
        context: list[str],
        judge_fn,
    ) -> EvalResult:
        """Calculate groundedness using LLM judge.

        Checks if all claims in answer are supported by context.

        Args:
            answer: Generated answer
            context: List of context strings
            judge_fn: Async function that takes prompt and returns score

        Returns:
            EvalResult with groundedness score
        """
        prompt = f"""Evaluate if the answer is fully grounded in the provided context.

Context:
{chr(10).join(context)}

Answer:
{answer}

Score from 0 to 1 where:
- 1.0: All claims are directly supported by context
- 0.5: Some claims supported, some unsupported
- 0.0: Most claims are unsupported or hallucinated

Return only the numeric score."""

        score = await judge_fn(prompt)

        return EvalResult(
            metric_name="groundedness",
            score=score,
            details={"method": "llm_judge"},
        )

    @staticmethod
    def answer_relevance(
        answer: str,
        query: str,
    ) -> EvalResult:
        """Simple heuristic for answer relevance.

        Checks if answer contains query terms.
        For production, use LLM-based evaluation.

        Args:
            answer: Generated answer
            query: Original query

        Returns:
            EvalResult with relevance score
        """
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())

        if not query_terms:
            return EvalResult(
                metric_name="answer_relevance",
                score=1.0,
                details={"method": "term_overlap"},
            )

        overlap = len(query_terms & answer_terms)
        score = overlap / len(query_terms)

        return EvalResult(
            metric_name="answer_relevance",
            score=min(1.0, score),  # Cap at 1.0
            details={"overlap": overlap, "query_terms": len(query_terms)},
        )
