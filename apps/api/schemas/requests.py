"""API request schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from rag.models.document import ChunkingStrategy


class IngestRequest(BaseModel):
    """Request to ingest a document."""

    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.RECURSIVE,
        description="Strategy for chunking the document",
    )
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to attach to the document",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        """Validate that overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class MetadataFilter(BaseModel):
    """Filter for metadata queries."""

    field: str = Field(..., description="Metadata field name to filter on")
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"] = Field(
        default="eq",
        description="Comparison operator",
    )
    value: Any = Field(..., description="Value to compare against")


class QueryRequest(BaseModel):
    """Request to query the RAG system."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question or search query",
    )
    filters: list[MetadataFilter] = Field(
        default_factory=list,
        description="Metadata filters to apply",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )
    enable_reranking: bool = Field(
        default=True,
        description="Whether to use cross-encoder reranking",
    )
    enable_freshness: bool = Field(
        default=True,
        description="Whether to apply freshness boosting",
    )
    freshness_reference: datetime | None = Field(
        default=None,
        description="Reference time for freshness scoring (default: now)",
    )
    enable_query_rewrite: bool = Field(
        default=False,
        description="Whether to rewrite the query for better retrieval",
    )
    return_trace: bool = Field(
        default=False,
        description="Whether to return detailed retrieval trace",
    )
    # Multi-hop settings (bonus feature)
    enable_multi_hop: bool = Field(
        default=False,
        description="Whether to enable multi-hop retrieval",
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum number of retrieval hops",
    )


class DocumentQueryRequest(BaseModel):
    """Request to get a specific document."""

    include_chunks: bool = Field(
        default=False,
        description="Whether to include chunk details",
    )
    include_embeddings: bool = Field(
        default=False,
        description="Whether to include embedding vectors",
    )
