"""Document management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from apps.api.dependencies import get_index_manager
from apps.api.schemas.responses import DocumentResponse, ErrorResponse
from rag.index.manager import IndexManager

router = APIRouter(prefix="/docs", tags=["documents"])


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document by ID",
    description="Retrieve a document and optionally its chunks.",
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
)
async def get_document(
    document_id: str,
    include_chunks: bool = Query(
        default=False, description="Include chunk details"
    ),
    index_manager: IndexManager = Depends(get_index_manager),
) -> DocumentResponse:
    """Get a document by its ID.

    Returns document metadata and optionally chunk details.
    """
    # Query Qdrant for chunks with this document_id
    if index_manager.vector_index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector index not available",
        )

    # Search for all chunks of this document
    # We need to use a filter search with a dummy vector
    # This is a limitation - in production you'd want a separate document store
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = index_manager.vector_index.client.scroll(
            collection_name=index_manager.vector_index._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        points = results[0]

        if not points:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Reconstruct document from chunks
        # Sort chunks by index
        points.sort(key=lambda p: p.payload.get("chunk_index", 0))

        # Get first chunk for document metadata
        first = points[0].payload
        content = "\n\n".join(p.payload.get("content", "") for p in points)

        chunks = None
        if include_chunks:
            chunks = [
                {
                    "chunk_id": str(p.id),
                    "chunk_index": p.payload.get("chunk_index"),
                    "content": p.payload.get("content", ""),
                    "section_title": p.payload.get("section_title"),
                    "token_count": p.payload.get("token_count"),
                }
                for p in points
            ]

        return DocumentResponse(
            document_id=document_id,
            content=content,
            doc_type=first.get("doc_type", "unknown"),
            metadata={
                "title": first.get("document_title"),
                "source_url": first.get("source_url"),
                "chunk_count": len(points),
            },
            chunks=chunks,
            created_at=first.get("document_created_at") or first.get("created_at"),
            updated_at=first.get("document_updated_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {e}",
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document",
    description="Delete a document and all its chunks.",
    responses={
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
)
async def delete_document(
    document_id: str,
    index_manager: IndexManager = Depends(get_index_manager),
) -> None:
    """Delete a document and all its chunks.

    Removes the document from both vector and lexical indexes.
    """
    try:
        await index_manager.delete_document(document_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {e}",
        )
