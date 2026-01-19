"""FastAPI application for RAG platform."""

import logging
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.routers import documents_router, ingest_router, query_router
from apps.api.schemas.responses import ErrorResponse, HealthResponse
from rag import __version__
from rag.config import get_settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting RAG Platform API", version=__version__)

    settings = get_settings()
    logger.info(
        "Configuration loaded",
        embedding_provider=settings.embedding.provider,
        llm_provider=settings.llm.provider,
        qdrant_url=settings.qdrant.url,
    )

    yield

    # Shutdown
    logger.info("Shutting down RAG Platform API")


# Create FastAPI app
app = FastAPI(
    title="RAG Platform API",
    description="Production-quality Retrieval-Augmented Generation platform",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred",
        ).model_dump(),
    )


# Include routers
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(documents_router)


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and connectivity.",
    tags=["health"],
)
async def health_check() -> HealthResponse:
    """Check service health."""
    settings = get_settings()
    qdrant_status = "unknown"

    # Check Qdrant connectivity
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant.url)
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {e}"

    return HealthResponse(
        status="healthy",
        version=__version__,
        qdrant_status=qdrant_status,
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "RAG Platform API",
        "version": __version__,
        "docs": "/docs",
    }


def run() -> None:
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "apps.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
