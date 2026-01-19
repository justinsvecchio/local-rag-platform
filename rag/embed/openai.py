"""OpenAI embedding provider."""

import asyncio
from typing import Literal

from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingError(Exception):
    """Raised when embedding fails."""

    pass


class OpenAIEmbedder:
    """OpenAI embedding provider.

    Uses OpenAI's text-embedding models for high-quality embeddings.
    """

    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        dimensions: int | None = None,
    ):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: Model name to use
            batch_size: Maximum texts per API call
            dimensions: Optional dimension override (for 3-large)
        """
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError as e:
                raise EmbeddingError(
                    "OpenAI embeddings require 'openai' package"
                ) from e
        return self._client

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimensions:
            return self._dimensions
        return self.MODEL_DIMENSIONS.get(self._model, 1536)

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """
        if not texts:
            return []

        try:
            all_embeddings = []

            # Process in batches
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]

                # Prepare request kwargs
                kwargs = {
                    "input": batch,
                    "model": self._model,
                }
                if self._dimensions and "3-large" in self._model:
                    kwargs["dimensions"] = self._dimensions

                response = await self.client.embeddings.create(**kwargs)

                # Sort by index to ensure order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        OpenAI doesn't differentiate between query and document embeddings,
        so this is the same as embed() for a single text.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        embeddings = await self.embed([query])
        return embeddings[0]
