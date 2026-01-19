"""API routers."""

from apps.api.routers.documents import router as documents_router
from apps.api.routers.ingest import router as ingest_router
from apps.api.routers.query import router as query_router

__all__ = ["ingest_router", "query_router", "documents_router"]
