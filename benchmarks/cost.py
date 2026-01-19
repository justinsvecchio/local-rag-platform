"""Cost tracking and estimation for RAG platform."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EmbeddingModel(Enum):
    """Available embedding models."""

    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    OPENAI_ADA_002 = "text-embedding-ada-002"
    LOCAL = "local"  # Free


class LLMModel(Enum):
    """Available LLM models."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    LOCAL = "local"  # Free (Ollama)


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    model: str
    input_cost_per_1k: float  # USD per 1K tokens
    output_cost_per_1k: float  # USD per 1K tokens (for LLMs)
    notes: str = ""


# Pricing as of early 2024 - update as needed
EMBEDDING_PRICING: dict[str, ModelPricing] = {
    EmbeddingModel.OPENAI_3_SMALL.value: ModelPricing(
        model=EmbeddingModel.OPENAI_3_SMALL.value,
        input_cost_per_1k=0.00002,
        output_cost_per_1k=0,
        notes="Best price/performance ratio",
    ),
    EmbeddingModel.OPENAI_3_LARGE.value: ModelPricing(
        model=EmbeddingModel.OPENAI_3_LARGE.value,
        input_cost_per_1k=0.00013,
        output_cost_per_1k=0,
        notes="Highest quality embeddings",
    ),
    EmbeddingModel.OPENAI_ADA_002.value: ModelPricing(
        model=EmbeddingModel.OPENAI_ADA_002.value,
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0,
        notes="Legacy model",
    ),
    EmbeddingModel.LOCAL.value: ModelPricing(
        model=EmbeddingModel.LOCAL.value,
        input_cost_per_1k=0,
        output_cost_per_1k=0,
        notes="Free - runs locally",
    ),
}

LLM_PRICING: dict[str, ModelPricing] = {
    LLMModel.GPT_4O.value: ModelPricing(
        model=LLMModel.GPT_4O.value,
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.015,
        notes="Latest GPT-4 Omni model",
    ),
    LLMModel.GPT_4O_MINI.value: ModelPricing(
        model=LLMModel.GPT_4O_MINI.value,
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        notes="Cost-effective GPT-4 variant",
    ),
    LLMModel.GPT_4_TURBO.value: ModelPricing(
        model=LLMModel.GPT_4_TURBO.value,
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        notes="GPT-4 Turbo with 128K context",
    ),
    LLMModel.GPT_35_TURBO.value: ModelPricing(
        model=LLMModel.GPT_35_TURBO.value,
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        notes="Fast and cost-effective",
    ),
    LLMModel.CLAUDE_3_OPUS.value: ModelPricing(
        model=LLMModel.CLAUDE_3_OPUS.value,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        notes="Most capable Claude model",
    ),
    LLMModel.CLAUDE_3_SONNET.value: ModelPricing(
        model=LLMModel.CLAUDE_3_SONNET.value,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        notes="Balanced Claude model",
    ),
    LLMModel.CLAUDE_3_HAIKU.value: ModelPricing(
        model=LLMModel.CLAUDE_3_HAIKU.value,
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        notes="Fastest Claude model",
    ),
    LLMModel.LOCAL.value: ModelPricing(
        model=LLMModel.LOCAL.value,
        input_cost_per_1k=0,
        output_cost_per_1k=0,
        notes="Free - runs locally via Ollama",
    ),
}


@dataclass
class CostEstimate:
    """Cost estimate for an operation."""

    operation: str
    embedding_cost: float
    llm_input_cost: float
    llm_output_cost: float
    total_cost: float
    details: dict[str, Any]


class CostTracker:
    """Track and estimate costs for RAG operations."""

    def __init__(
        self,
        embedding_model: str = EmbeddingModel.OPENAI_3_SMALL.value,
        llm_model: str = LLMModel.GPT_4O_MINI.value,
    ):
        """Initialize cost tracker.

        Args:
            embedding_model: Model used for embeddings
            llm_model: Model used for LLM generation
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self._total_embedding_tokens = 0
        self._total_llm_input_tokens = 0
        self._total_llm_output_tokens = 0
        self._operations: list[CostEstimate] = []

    @property
    def embedding_pricing(self) -> ModelPricing:
        """Get embedding model pricing."""
        return EMBEDDING_PRICING.get(
            self.embedding_model,
            EMBEDDING_PRICING[EmbeddingModel.LOCAL.value],
        )

    @property
    def llm_pricing(self) -> ModelPricing:
        """Get LLM model pricing."""
        return LLM_PRICING.get(
            self.llm_model,
            LLM_PRICING[LLMModel.LOCAL.value],
        )

    def estimate_ingestion_cost(
        self,
        document_tokens: int,
        chunk_count: int,
    ) -> CostEstimate:
        """Estimate cost for document ingestion.

        Args:
            document_tokens: Total tokens in document
            chunk_count: Number of chunks created

        Returns:
            CostEstimate for the ingestion
        """
        # Embedding cost: embed each chunk
        # Assuming average chunk is ~500 tokens
        avg_chunk_tokens = document_tokens / max(chunk_count, 1)
        total_embedding_tokens = chunk_count * avg_chunk_tokens

        embedding_cost = (
            total_embedding_tokens / 1000 * self.embedding_pricing.input_cost_per_1k
        )

        estimate = CostEstimate(
            operation="ingestion",
            embedding_cost=embedding_cost,
            llm_input_cost=0,
            llm_output_cost=0,
            total_cost=embedding_cost,
            details={
                "document_tokens": document_tokens,
                "chunk_count": chunk_count,
                "embedding_tokens": total_embedding_tokens,
            },
        )

        self._total_embedding_tokens += int(total_embedding_tokens)
        self._operations.append(estimate)

        return estimate

    def estimate_query_cost(
        self,
        query_tokens: int,
        context_tokens: int,
        output_tokens: int = 500,
        rerank_count: int = 0,
    ) -> CostEstimate:
        """Estimate cost for a query operation.

        Args:
            query_tokens: Tokens in the query
            context_tokens: Tokens in retrieved context
            output_tokens: Expected output tokens
            rerank_count: Number of documents to rerank

        Returns:
            CostEstimate for the query
        """
        # Embedding cost: embed the query
        embedding_cost = query_tokens / 1000 * self.embedding_pricing.input_cost_per_1k

        # LLM cost: input = query + context, output = answer
        llm_input_tokens = query_tokens + context_tokens
        llm_input_cost = llm_input_tokens / 1000 * self.llm_pricing.input_cost_per_1k
        llm_output_cost = output_tokens / 1000 * self.llm_pricing.output_cost_per_1k

        # Reranking cost (if using cross-encoder, it's local/free)
        # If using LLM reranker, add that cost
        rerank_cost = 0  # Cross-encoder is local

        total_cost = embedding_cost + llm_input_cost + llm_output_cost + rerank_cost

        estimate = CostEstimate(
            operation="query",
            embedding_cost=embedding_cost,
            llm_input_cost=llm_input_cost,
            llm_output_cost=llm_output_cost,
            total_cost=total_cost,
            details={
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "output_tokens": output_tokens,
                "rerank_count": rerank_count,
            },
        )

        self._total_embedding_tokens += query_tokens
        self._total_llm_input_tokens += llm_input_tokens
        self._total_llm_output_tokens += output_tokens
        self._operations.append(estimate)

        return estimate

    def get_total_cost(self) -> float:
        """Get total cost across all tracked operations."""
        return sum(op.total_cost for op in self._operations)

    def get_summary(self) -> dict[str, Any]:
        """Get cost summary."""
        return {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "total_operations": len(self._operations),
            "total_embedding_tokens": self._total_embedding_tokens,
            "total_llm_input_tokens": self._total_llm_input_tokens,
            "total_llm_output_tokens": self._total_llm_output_tokens,
            "total_cost_usd": self.get_total_cost(),
            "breakdown": {
                "ingestion": sum(
                    op.total_cost for op in self._operations if op.operation == "ingestion"
                ),
                "queries": sum(
                    op.total_cost for op in self._operations if op.operation == "query"
                ),
            },
        }

    def reset(self) -> None:
        """Reset all tracked costs."""
        self._total_embedding_tokens = 0
        self._total_llm_input_tokens = 0
        self._total_llm_output_tokens = 0
        self._operations = []


def estimate_monthly_cost(
    documents_per_month: int,
    avg_tokens_per_document: int,
    avg_chunks_per_document: int,
    queries_per_month: int,
    avg_context_tokens: int = 2000,
    embedding_model: str = EmbeddingModel.OPENAI_3_SMALL.value,
    llm_model: str = LLMModel.GPT_4O_MINI.value,
) -> dict[str, Any]:
    """Estimate monthly costs for a RAG workload.

    Args:
        documents_per_month: Documents ingested per month
        avg_tokens_per_document: Average tokens per document
        avg_chunks_per_document: Average chunks per document
        queries_per_month: Queries per month
        avg_context_tokens: Average context tokens per query
        embedding_model: Embedding model to use
        llm_model: LLM model to use

    Returns:
        Monthly cost breakdown
    """
    tracker = CostTracker(embedding_model=embedding_model, llm_model=llm_model)

    # Simulate ingestion
    for _ in range(documents_per_month):
        tracker.estimate_ingestion_cost(
            document_tokens=avg_tokens_per_document,
            chunk_count=avg_chunks_per_document,
        )

    # Simulate queries
    for _ in range(queries_per_month):
        tracker.estimate_query_cost(
            query_tokens=50,  # Average query length
            context_tokens=avg_context_tokens,
            output_tokens=500,  # Average answer length
        )

    summary = tracker.get_summary()
    summary["inputs"] = {
        "documents_per_month": documents_per_month,
        "avg_tokens_per_document": avg_tokens_per_document,
        "avg_chunks_per_document": avg_chunks_per_document,
        "queries_per_month": queries_per_month,
    }

    return summary


def print_cost_comparison() -> None:
    """Print cost comparison for different configurations."""
    print("=" * 70)
    print("RAG Platform Monthly Cost Estimates")
    print("=" * 70)

    # Small workload
    small = estimate_monthly_cost(
        documents_per_month=100,
        avg_tokens_per_document=5000,
        avg_chunks_per_document=10,
        queries_per_month=1000,
    )

    # Medium workload
    medium = estimate_monthly_cost(
        documents_per_month=1000,
        avg_tokens_per_document=5000,
        avg_chunks_per_document=10,
        queries_per_month=10000,
    )

    # Large workload
    large = estimate_monthly_cost(
        documents_per_month=10000,
        avg_tokens_per_document=5000,
        avg_chunks_per_document=10,
        queries_per_month=100000,
    )

    workloads = [
        ("Small (100 docs, 1K queries)", small),
        ("Medium (1K docs, 10K queries)", medium),
        ("Large (10K docs, 100K queries)", large),
    ]

    for name, summary in workloads:
        print(f"\n{name}")
        print("-" * 50)
        print(f"  Embedding model: {summary['embedding_model']}")
        print(f"  LLM model: {summary['llm_model']}")
        print(f"  Ingestion cost: ${summary['breakdown']['ingestion']:.4f}")
        print(f"  Query cost: ${summary['breakdown']['queries']:.4f}")
        print(f"  Total monthly cost: ${summary['total_cost_usd']:.4f}")

    # Local-first comparison
    print("\n" + "=" * 70)
    print("Local-First Mode (No API Costs)")
    print("=" * 70)
    print("  Using local embeddings (sentence-transformers) and Ollama: $0.00/month")
    print("  Note: Requires local compute resources")


if __name__ == "__main__":
    print_cost_comparison()
