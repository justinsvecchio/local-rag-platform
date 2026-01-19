"""BM25 lexical index implementation."""

import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.models.retrieval import SearchResult, SearchSource


class BM25Index:
    """BM25 lexical index for keyword search.

    Uses rank-bm25 for efficient BM25 scoring.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """Initialize BM25 index.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
            epsilon: BM25 epsilon parameter
        """
        self._k1 = k1
        self._b = b
        self._epsilon = epsilon
        self._bm25 = None
        self._documents: dict[str, dict[str, Any]] = {}
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []

    def index(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add document to index.

        Args:
            doc_id: Document/chunk ID
            text: Text content to index
            metadata: Document metadata
        """
        # Tokenize text
        tokens = self._tokenize(text)

        # Store document
        self._documents[doc_id] = {
            "content": text,
            "tokens": tokens,
            "metadata": metadata,
        }
        self._doc_ids.append(doc_id)
        self._corpus.append(tokens)

        # Invalidate BM25 model (will be rebuilt on search)
        self._bm25 = None

    def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using BM25.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        if not self._corpus:
            return []

        # Build BM25 model if needed
        if self._bm25 is None:
            self._build_bm25()

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Combine with doc IDs and filter
        results = []
        for idx, (doc_id, score) in enumerate(zip(self._doc_ids, scores)):
            if score <= 0:
                continue

            doc = self._documents[doc_id]
            metadata = doc.get("metadata", {})

            # Apply filters
            if filters and not self._matches_filter(metadata, filters):
                continue

            results.append(
                SearchResult(
                    chunk_id=doc_id,
                    document_id=metadata.get("document_id", doc_id),
                    content=doc["content"],
                    score=float(score),
                    source=SearchSource.LEXICAL,
                    metadata=metadata,
                    document_created_at=self._parse_datetime(
                        metadata.get("document_created_at")
                    ),
                )
            )

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def delete(self, doc_ids: list[str]) -> None:
        """Delete documents from index.

        Args:
            doc_ids: Document IDs to delete
        """
        for doc_id in doc_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                idx = self._doc_ids.index(doc_id)
                self._doc_ids.pop(idx)
                self._corpus.pop(idx)

        # Invalidate BM25 model
        self._bm25 = None

    def save(self, path: str) -> None:
        """Persist index to disk.

        Args:
            path: Path to save index
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self._documents,
            "doc_ids": self._doc_ids,
            "corpus": self._corpus,
            "k1": self._k1,
            "b": self._b,
            "epsilon": self._epsilon,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load index from disk.

        Args:
            path: Path to load index from
        """
        if not Path(path).exists():
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._documents = data["documents"]
        self._doc_ids = data["doc_ids"]
        self._corpus = data["corpus"]
        self._k1 = data.get("k1", self._k1)
        self._b = data.get("b", self._b)
        self._epsilon = data.get("epsilon", self._epsilon)
        self._bm25 = None

    def _build_bm25(self) -> None:
        """Build BM25 model from corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "BM25 index requires 'rank-bm25' package"
            ) from e

        self._bm25 = BM25Okapi(
            self._corpus,
            k1=self._k1,
            b=self._b,
            epsilon=self._epsilon,
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Simple whitespace + lowercasing tokenization.
        Override for more sophisticated tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _matches_filter(
        self,
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        """Check if metadata matches filters.

        Args:
            metadata: Document metadata
            filters: Filter conditions

        Returns:
            True if metadata matches all filters
        """
        for key, value in filters.items():
            meta_value = metadata.get(key)

            if isinstance(value, dict):
                # Range filter
                if "gte" in value and (meta_value is None or meta_value < value["gte"]):
                    return False
                if "lte" in value and (meta_value is None or meta_value > value["lte"]):
                    return False
                if "gt" in value and (meta_value is None or meta_value <= value["gt"]):
                    return False
                if "lt" in value and (meta_value is None or meta_value >= value["lt"]):
                    return False
            elif isinstance(value, list):
                # IN filter
                if meta_value not in value:
                    return False
            else:
                # Equality filter
                if meta_value != value:
                    return False

        return True

    def _parse_datetime(self, value: Any) -> datetime | None:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def count(self) -> int:
        """Get total number of documents in index."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents from index."""
        self._documents = {}
        self._doc_ids = []
        self._corpus = []
        self._bm25 = None
