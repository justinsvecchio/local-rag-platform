"""Query endpoint for RAG retrieval and generation."""

import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import (
    get_freshness_scorer,
    get_generator,
    get_hybrid_retriever,
    get_reranker,
)
from apps.api.schemas.requests import QueryRequest
from apps.api.schemas.responses import ErrorResponse, QueryResponse, SourceDocument
from rag.freshness.exponential_decay import ExponentialDecayScorer
from rag.generate.openai import OpenAIGenerator
from rag.models.generation import RetrievalTrace
from rag.models.retrieval import RetrievalConfig
from rag.rerank.cross_encoder import CrossEncoderReranker
from rag.retrieve.hybrid import HybridRetriever

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="Retrieve relevant documents and generate an answer with citations.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    generator: OpenAIGenerator = Depends(get_generator),
    freshness_scorer: ExponentialDecayScorer = Depends(get_freshness_scorer),
) -> QueryResponse:
    """Execute a RAG query.

    Performs hybrid retrieval (vector + BM25), optional reranking,
    and generates an answer with citations.
    """
    start_time = time.perf_counter()

    # Initialize trace if requested
    trace = None
    if request.return_trace:
        trace = RetrievalTrace(
            trace_id=str(uuid4()),
            query=request.query,
            total_latency_ms=0,  # Will be updated
        )

    # Build filters from request
    filters = _build_filters(request.filters) if request.filters else None

    # Optional query rewriting
    query_text = request.query
    query_rewritten = None
    if request.enable_query_rewrite:
        # TODO: Implement query rewriting
        pass

    if trace:
        trace.query_rewritten = query_rewritten

    # Retrieve
    try:
        results = await retriever.retrieve(
            query=query_text,
            limit=request.limit * 4,  # Over-fetch for reranking
            filters=filters,
            trace=trace,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval failed: {e}",
        )

    if not results:
        # No results found
        return QueryResponse(
            query=request.query,
            query_rewritten=query_rewritten,
            answer="I couldn't find any relevant information to answer your question.",
            citations=[],
            sources=[],
            model=generator.model_name,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            trace=trace,
        )

    # Apply freshness scoring
    if request.enable_freshness:
        results = freshness_scorer.apply_to_results(
            results,
            weight=retriever.config.freshness_weight,
            reference_time=request.freshness_reference,
        )

    # Rerank
    if request.enable_reranking and len(results) > 0:
        try:
            rerank_start = time.perf_counter()
            ranked_results = await reranker.rerank(
                query=query_text,
                results=results,
                top_k=request.limit,
            )
            rerank_time = (time.perf_counter() - rerank_start) * 1000

            if trace:
                trace.add_reranked_results(ranked_results, rerank_time)
        except Exception as e:
            # Fall back to un-reranked results
            ranked_results = [
                results[i].__class__.from_search_result(
                    result=results[i],
                    rerank_score=results[i].score,
                    rank=i + 1,
                )
                for i in range(min(len(results), request.limit))
            ]
    else:
        # Use results directly as ranked
        from rag.models.retrieval import RankedResult

        ranked_results = [
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
                freshness_score=r.metadata.get("freshness_score"),
                document_created_at=r.document_created_at,
                document_updated_at=r.document_updated_at,
            )
            for i, r in enumerate(results[: request.limit])
        ]

    # Generate answer
    try:
        gen_start = time.perf_counter()
        answer = await generator.generate(
            query=request.query,
            context=ranked_results,
        )
        gen_time = (time.perf_counter() - gen_start) * 1000

        if trace:
            trace.generation_latency_ms = gen_time
            trace.answer = answer
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Generation failed: {e}",
        )

    # Build response
    total_latency = (time.perf_counter() - start_time) * 1000

    if trace:
        trace.total_latency_ms = total_latency

    sources = [
        SourceDocument(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            content=r.content[:500] + "..." if len(r.content) > 500 else r.content,
            score=r.final_score,
            metadata=r.metadata,
        )
        for r in ranked_results
    ]

    return QueryResponse(
        query=request.query,
        query_rewritten=query_rewritten,
        answer=answer.answer,
        citations=answer.citations,
        sources=sources,
        model=answer.model,
        latency_ms=total_latency,
        trace=trace if request.return_trace else None,
    )


def _build_filters(filters: list) -> dict:
    """Build filter dict from MetadataFilter list."""
    if not filters:
        return {}

    result = {}
    for f in filters:
        if f.operator == "eq":
            result[f.field] = f.value
        elif f.operator in ("gt", "gte", "lt", "lte"):
            if f.field not in result:
                result[f.field] = {}
            result[f.field][f.operator] = f.value
        elif f.operator == "in":
            result[f.field] = {"any": f.value}

    return result
