"""Ingest endpoint for document ingestion."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from apps.api.dependencies import get_ingest_pipeline
from apps.api.schemas.requests import IngestRequest
from apps.api.schemas.responses import ErrorResponse, IngestResponse
from rag.config import get_settings
from rag.ingest.pipeline import IngestPipeline
from rag.ingest.parsers.base import ParseError

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description="Parse, chunk, embed, and index a document.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    request: str = Form(default="{}", description="Ingestion options as JSON"),
    pipeline: IngestPipeline = Depends(get_ingest_pipeline),
) -> IngestResponse:
    """Ingest a document into the RAG system.

    Accepts various document formats (PDF, HTML, DOCX, TXT, Markdown).
    The document is parsed, cleaned, chunked, embedded, and indexed.
    """
    settings = get_settings()

    # Parse request JSON
    try:
        ingest_request = IngestRequest.model_validate_json(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid request JSON: {e}",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > settings.max_upload_size:
        max_mb = settings.max_upload_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {max_mb}MB limit",
        )

    # Validate file has content
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    # Process document
    try:
        result = await pipeline.process(
            content=content,
            filename=file.filename or "document",
            chunking_strategy=ingest_request.chunking_strategy,
            chunk_size=ingest_request.chunk_size,
            chunk_overlap=ingest_request.chunk_overlap,
            metadata=ingest_request.metadata,
        )
    except ParseError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse document: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {e}",
        )

    return IngestResponse(
        document_id=result.document_id,
        filename=file.filename or "document",
        doc_type=result.doc_type,
        chunk_count=result.chunk_count,
        total_tokens=result.total_tokens,
        processing_time_ms=result.processing_time_ms,
        metadata=result.metadata,
    )
