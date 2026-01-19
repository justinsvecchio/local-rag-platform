"""FastAPI dependency injection providers."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from rag.config import get_settings
from rag.embed.factory import get_embedder
from rag.freshness.exponential_decay import ExponentialDecayScorer
from rag.generate.factory import get_generator as _get_generator
from rag.index.bm25.index import BM25Index
from rag.index.manager import IndexManager
from rag.index.qdrant.client import QdrantIndex
from rag.ingest.pipeline import IngestPipeline
from rag.models.retrieval import RetrievalConfig
from rag.protocols import Embedder, Generator
from rag.rerank.cross_encoder import CrossEncoderReranker
from rag.retrieve.fusion.rrf import RRFFusion
from rag.retrieve.hybrid import HybridRetriever


@lru_cache
def get_embedder_instance() -> Embedder:
    """Get cached embedder instance."""
    return get_embedder()


@lru_cache
def get_qdrant_index() -> QdrantIndex:
    """Get cached Qdrant index."""
    settings = get_settings()
    return QdrantIndex(
        url=settings.qdrant.url,
        collection_name=settings.qdrant.collection,
        api_key=settings.qdrant.api_key,
        embedding_dimension=1536,  # Default for OpenAI
    )


@lru_cache
def get_bm25_index() -> BM25Index:
    """Get cached BM25 index."""
    settings = get_settings()
    index = BM25Index()

    # Try to load persisted index
    try:
        index.load(settings.bm25_index_path)
    except Exception:
        pass

    return index


def get_index_manager(
    qdrant: Annotated[QdrantIndex, Depends(get_qdrant_index)],
    bm25: Annotated[BM25Index, Depends(get_bm25_index)],
) -> IndexManager:
    """Get index manager with both indexes."""
    return IndexManager(
        vector_index=qdrant,
        lexical_index=bm25,
    )


def get_retrieval_config() -> RetrievalConfig:
    """Get retrieval configuration."""
    settings = get_settings()
    return RetrievalConfig(
        rrf_k=settings.retrieval.rrf_k,
        freshness_half_life_days=settings.retrieval.freshness_half_life_days,
        freshness_weight=settings.retrieval.freshness_weight,
        initial_limit=settings.retrieval.initial_limit,
        rerank_limit=settings.retrieval.rerank_limit,
        final_limit=settings.retrieval.final_limit,
    )


def get_hybrid_retriever(
    embedder: Annotated[Embedder, Depends(get_embedder_instance)],
    qdrant: Annotated[QdrantIndex, Depends(get_qdrant_index)],
    bm25: Annotated[BM25Index, Depends(get_bm25_index)],
    config: Annotated[RetrievalConfig, Depends(get_retrieval_config)],
) -> HybridRetriever:
    """Get hybrid retriever."""
    return HybridRetriever(
        embedder=embedder,
        vector_index=qdrant,
        lexical_index=bm25,
        fusion_strategy=RRFFusion(k=config.rrf_k),
        config=config,
    )


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Get cached cross-encoder reranker."""
    settings = get_settings()
    return CrossEncoderReranker(
        model_name=settings.reranker.model,
        batch_size=settings.reranker.batch_size,
    )


@lru_cache
def get_generator() -> Generator:
    """Get cached generator."""
    return _get_generator()


def get_freshness_scorer() -> ExponentialDecayScorer:
    """Get freshness scorer."""
    settings = get_settings()
    return ExponentialDecayScorer(
        half_life_days=settings.retrieval.freshness_half_life_days,
    )


def get_ingest_pipeline(
    embedder: Annotated[Embedder, Depends(get_embedder_instance)],
    qdrant: Annotated[QdrantIndex, Depends(get_qdrant_index)],
    bm25: Annotated[BM25Index, Depends(get_bm25_index)],
) -> IngestPipeline:
    """Get ingestion pipeline."""
    return IngestPipeline(
        vector_index=qdrant,
        lexical_index=bm25,
        embedder=embedder,
    )
