"""Lexical retriever using BM25 search."""

from typing import Any

from rag.models.retrieval import SearchResult
from rag.protocols import LexicalIndex


class LexicalRetriever:
    """Retriever using BM25 lexical search.

    Performs keyword-based search to complement semantic search.
    Good for exact matches, names, codes, and specific terms.
    """

    def __init__(self, lexical_index: LexicalIndex):
        """Initialize lexical retriever.

        Args:
            lexical_index: BM25 index for keyword search
        """
        self._lexical_index = lexical_index

    async def retrieve(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant documents using BM25 search.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        # BM25 search is synchronous but wrapped in async for consistency
        results = self._lexical_index.search(
            query=query,
            limit=limit,
            filters=filters,
        )

        return results

    @property
    def index(self) -> LexicalIndex:
        """Get the lexical index."""
        return self._lexical_index
