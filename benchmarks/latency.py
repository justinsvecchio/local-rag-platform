"""Latency benchmarks for RAG platform components."""

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from rag.chunking.factory import get_chunker
from rag.models.document import ChunkingStrategy, Document, DocumentMetadata, DocType
from rag.retrieve.fusion.rrf import RRFFusion
from rag.freshness.exponential_decay import ExponentialDecayScorer
from rag.models.retrieval import SearchResult, SearchSource


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    mean_ms: float
    median_ms: float
    std_dev_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float

    def __str__(self) -> str:
        """Format benchmark result."""
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_ms:.2f}ms\n"
            f"  Median: {self.median_ms:.2f}ms\n"
            f"  Std Dev: {self.std_dev_ms:.2f}ms\n"
            f"  Min: {self.min_ms:.2f}ms\n"
            f"  Max: {self.max_ms:.2f}ms\n"
            f"  P95: {self.p95_ms:.2f}ms\n"
            f"  P99: {self.p99_ms:.2f}ms"
        )


def benchmark_sync(
    func: Callable[[], Any],
    name: str,
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark a synchronous function.

    Args:
        func: Function to benchmark
        name: Name for the benchmark
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times_ms = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    total_time = (time.perf_counter() - start_total) * 1000

    # Calculate statistics
    times_sorted = sorted(times_ms)
    p95_idx = int(len(times_sorted) * 0.95)
    p99_idx = int(len(times_sorted) * 0.99)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time,
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        std_dev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        p95_ms=times_sorted[p95_idx] if p95_idx < len(times_sorted) else times_sorted[-1],
        p99_ms=times_sorted[p99_idx] if p99_idx < len(times_sorted) else times_sorted[-1],
    )


async def benchmark_async(
    func: Callable[[], Any],
    name: str,
    iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark an asynchronous function.

    Args:
        func: Async function to benchmark
        name: Name for the benchmark
        iterations: Number of iterations to run
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        await func()

    # Benchmark
    times_ms = []
    start_total = time.perf_counter()

    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    total_time = (time.perf_counter() - start_total) * 1000

    # Calculate statistics
    times_sorted = sorted(times_ms)
    p95_idx = int(len(times_sorted) * 0.95)
    p99_idx = int(len(times_sorted) * 0.99)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time,
        mean_ms=statistics.mean(times_ms),
        median_ms=statistics.median(times_ms),
        std_dev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        p95_ms=times_sorted[p95_idx] if p95_idx < len(times_sorted) else times_sorted[-1],
        p99_ms=times_sorted[p99_idx] if p99_idx < len(times_sorted) else times_sorted[-1],
    )


def create_sample_document(word_count: int = 1000) -> Document:
    """Create a sample document for benchmarking."""
    content = " ".join(["word"] * word_count)
    return Document(
        id="bench_doc",
        content=content,
        doc_type=DocType.TEXT,
        metadata=DocumentMetadata(title="Benchmark Document"),
    )


def create_sample_results(count: int = 100) -> list[SearchResult]:
    """Create sample search results for benchmarking."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    return [
        SearchResult(
            chunk_id=f"chunk_{i}",
            document_id=f"doc_{i % 10}",
            content=f"Sample content for chunk {i}",
            score=1.0 - (i * 0.01),
            source=SearchSource.VECTOR,
            document_created_at=now - timedelta(days=i),
        )
        for i in range(count)
    ]


def run_chunking_benchmarks() -> list[BenchmarkResult]:
    """Run benchmarks for chunking strategies."""
    results = []
    doc = create_sample_document(5000)

    strategies = [
        (ChunkingStrategy.RECURSIVE, "Recursive Chunking"),
        (ChunkingStrategy.SLIDING_WINDOW, "Sliding Window Chunking"),
        (ChunkingStrategy.HEADING, "Heading Chunking"),
    ]

    for strategy, name in strategies:
        chunker = get_chunker(strategy, chunk_size=512, chunk_overlap=50)
        result = benchmark_sync(
            lambda c=chunker, d=doc: c.chunk(d),
            name=name,
            iterations=50,
        )
        results.append(result)

    return results


def run_fusion_benchmarks() -> list[BenchmarkResult]:
    """Run benchmarks for fusion strategies."""
    results = []

    # Create two result lists to fuse
    list1 = create_sample_results(50)
    list2 = create_sample_results(50)

    fusion = RRFFusion(k=60)
    result = benchmark_sync(
        lambda: fusion.fuse([list1, list2]),
        name="RRF Fusion (100 results)",
        iterations=100,
    )
    results.append(result)

    # Larger scale
    list1_large = create_sample_results(500)
    list2_large = create_sample_results(500)
    result_large = benchmark_sync(
        lambda: fusion.fuse([list1_large, list2_large]),
        name="RRF Fusion (1000 results)",
        iterations=50,
    )
    results.append(result_large)

    return results


def run_freshness_benchmarks() -> list[BenchmarkResult]:
    """Run benchmarks for freshness scoring."""
    results = []

    scorer = ExponentialDecayScorer(half_life_days=30.0)

    # Small result set
    small_results = create_sample_results(20)
    result = benchmark_sync(
        lambda: scorer.apply(small_results, weight=0.2),
        name="Freshness Scoring (20 results)",
        iterations=100,
    )
    results.append(result)

    # Large result set
    large_results = create_sample_results(200)
    result_large = benchmark_sync(
        lambda: scorer.apply(large_results, weight=0.2),
        name="Freshness Scoring (200 results)",
        iterations=50,
    )
    results.append(result_large)

    return results


def run_all_benchmarks() -> dict[str, list[BenchmarkResult]]:
    """Run all benchmarks."""
    return {
        "chunking": run_chunking_benchmarks(),
        "fusion": run_fusion_benchmarks(),
        "freshness": run_freshness_benchmarks(),
    }


def print_benchmark_report(all_results: dict[str, list[BenchmarkResult]]) -> None:
    """Print a formatted benchmark report."""
    print("=" * 60)
    print("RAG Platform Latency Benchmarks")
    print("=" * 60)

    for category, results in all_results.items():
        print(f"\n{category.upper()}")
        print("-" * 40)
        for result in results:
            print(f"\n{result}")


if __name__ == "__main__":
    all_results = run_all_benchmarks()
    print_benchmark_report(all_results)
