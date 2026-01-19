"""API response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from rag.models.generation import Citation, RetrievalTrace


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    doc_type: str = Field(..., description="Detected document type")
    chunk_count: int = Field(..., description="Number of chunks created")
    total_tokens: int = Field(..., description="Total tokens across all chunks")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )


class SourceDocument(BaseModel):
    """A source document in query response."""

    document_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata"
    )


class QueryResponse(BaseModel):
    """Response to a query."""

    query: str = Field(..., description="Original query")
    query_rewritten: str | None = Field(
        default=None, description="Rewritten query (if enabled)"
    )
    answer: str = Field(..., description="Generated answer")
    citations: list[Citation] = Field(
        default_factory=list, description="Citations used in the answer"
    )
    sources: list[SourceDocument] = Field(
        default_factory=list, description="Source documents used"
    )
    model: str = Field(..., description="LLM model used for generation")
    latency_ms: float = Field(..., description="Total processing time in ms")
    trace: RetrievalTrace | None = Field(
        default=None, description="Detailed retrieval trace (if requested)"
    )


class DocumentResponse(BaseModel):
    """Response for document retrieval."""

    document_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    doc_type: str = Field(..., description="Document type")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    chunks: list[dict[str, Any]] | None = Field(
        default=None, description="Chunk details (if requested)"
    )
    created_at: datetime = Field(..., description="Document creation time")
    updated_at: datetime | None = Field(
        default=None, description="Last update time"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    detail: str | None = Field(default=None, description="Error details")
    trace_id: str | None = Field(default=None, description="Request trace ID")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    qdrant_status: str | None = Field(
        default=None, description="Qdrant connection status"
    )
    index_stats: dict[str, Any] | None = Field(
        default=None, description="Index statistics"
    )
