"""Reranker factory."""

from typing import Literal

from rag.config import get_settings
from rag.protocols import Reranker
from rag.rerank.cross_encoder import CrossEncoderReranker
from rag.rerank.llm import LLMReranker


def get_reranker(
    reranker_type: Literal["cross_encoder", "llm"] = "cross_encoder",
    **kwargs,
) -> Reranker:
    """Get a reranker instance.

    Args:
        reranker_type: Type of reranker ('cross_encoder' or 'llm')
        **kwargs: Additional arguments for the reranker

    Returns:
        Reranker instance
    """
    settings = get_settings()

    match reranker_type:
        case "cross_encoder":
            return CrossEncoderReranker(
                model_name=kwargs.get("model_name", settings.reranker.model),
                batch_size=kwargs.get("batch_size", settings.reranker.batch_size),
                **{k: v for k, v in kwargs.items() if k not in ("model_name", "batch_size")},
            )
        case "llm":
            return LLMReranker(**kwargs)
        case _:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
