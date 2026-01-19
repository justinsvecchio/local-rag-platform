"""Evaluation runner for RAG system."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.metrics import GenerationMetrics, RetrievalMetrics


@dataclass
class EvalCase:
    """A single evaluation case."""

    query: str
    relevant_chunk_ids: set[str]
    relevance_scores: dict[str, float] = field(default_factory=dict)
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Evaluation report."""

    dataset_name: str
    total_cases: int
    metrics: dict[str, float]
    per_case_results: list[dict[str, Any]]
    config: dict[str, Any]


class EvalRunner:
    """Orchestrates evaluation runs."""

    def __init__(
        self,
        retriever,
        generator=None,
        judge_fn=None,
    ):
        """Initialize runner.

        Args:
            retriever: Retriever to evaluate
            generator: Optional generator for end-to-end eval
            judge_fn: Optional LLM judge function
        """
        self.retriever = retriever
        self.generator = generator
        self.judge_fn = judge_fn

    async def run(
        self,
        dataset_path: Path | str,
        k_values: list[int] | None = None,
        include_generation: bool = True,
    ) -> EvalReport:
        """Run evaluation on a dataset.

        Args:
            dataset_path: Path to evaluation dataset JSON
            k_values: K values for recall/precision
            include_generation: Whether to evaluate generation

        Returns:
            Evaluation report
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        # Load dataset
        with open(dataset_path) as f:
            data = json.load(f)

        cases = [
            EvalCase(
                query=item["query"],
                relevant_chunk_ids=set(item["relevant_chunk_ids"]),
                relevance_scores=item.get("relevance_scores", {}),
                expected_answer=item.get("expected_answer"),
                metadata=item.get("metadata", {}),
            )
            for item in data.get("cases", [])
        ]

        # Run evaluations
        per_case_results = []
        aggregated: dict[str, list[float]] = {
            f"recall@{k}": [] for k in k_values
        }
        aggregated.update({f"precision@{k}": [] for k in k_values})
        aggregated.update({
            "mrr": [],
            "citation_precision": [],
            "groundedness": [],
        })

        for case in cases:
            result = await self._evaluate_case(
                case, k_values, include_generation
            )
            per_case_results.append(result)

            # Aggregate scores
            for metric_name, score in result["metrics"].items():
                if metric_name in aggregated:
                    aggregated[metric_name].append(score)

        # Calculate averages
        avg_metrics = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in aggregated.items()
        }

        return EvalReport(
            dataset_name=data.get("name", "unknown"),
            total_cases=len(cases),
            metrics=avg_metrics,
            per_case_results=per_case_results,
            config={"k_values": k_values, "include_generation": include_generation},
        )

    async def _evaluate_case(
        self,
        case: EvalCase,
        k_values: list[int],
        include_generation: bool,
    ) -> dict[str, Any]:
        """Evaluate a single case."""
        # Retrieve
        results = await self.retriever.retrieve(
            query=case.query,
            limit=max(k_values),
        )
        retrieved_ids = [r.chunk_id for r in results]

        # Calculate retrieval metrics
        metrics = {}

        for k in k_values:
            recall = RetrievalMetrics.recall_at_k(
                retrieved_ids, case.relevant_chunk_ids, k
            )
            precision = RetrievalMetrics.precision_at_k(
                retrieved_ids, case.relevant_chunk_ids, k
            )
            metrics[recall.metric_name] = recall.score
            metrics[precision.metric_name] = precision.score

        mrr = RetrievalMetrics.mrr(retrieved_ids, case.relevant_chunk_ids)
        metrics["mrr"] = mrr.score

        # Generation metrics
        if include_generation and self.generator:
            from rag.models.retrieval import RankedResult

            # Convert to ranked results
            ranked = [
                RankedResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    original_score=r.score,
                    rerank_score=r.score,
                    final_score=r.score,
                    rank=i + 1,
                    source=r.source,
                    metadata=r.metadata,
                )
                for i, r in enumerate(results[:5])
            ]

            answer = await self.generator.generate(
                query=case.query,
                context=ranked,
            )

            citation_prec = GenerationMetrics.citation_precision(
                [c.model_dump() for c in answer.citations],
                case.relevant_chunk_ids,
            )
            metrics["citation_precision"] = citation_prec.score

            if self.judge_fn:
                groundedness = await GenerationMetrics.groundedness(
                    answer.answer,
                    [r.content for r in results[:5]],
                    self.judge_fn,
                )
                metrics["groundedness"] = groundedness.score

        return {
            "query": case.query,
            "retrieved_ids": retrieved_ids,
            "metrics": metrics,
        }


def main():
    """CLI entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evals/reports/",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated k values for metrics",
    )

    args = parser.parse_args()
    k_values = [int(k) for k in args.k_values.split(",")]

    # This would need proper initialization in real usage
    print(f"Would run evaluation on {args.dataset}")
    print(f"K values: {k_values}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
