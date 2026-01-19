"""Sentence Transformers embedding provider for local embeddings."""

import asyncio
from functools import lru_cache

import numpy as np

from rag.embed.openai import EmbeddingError


class SentenceTransformerEmbedder:
    """Local embedding provider using sentence-transformers.

    Provides high-quality embeddings without API calls.
    Runs entirely locally on CPU or GPU.
    """

    # Popular models and their dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """Initialize sentence transformer embedder.

        Args:
            model_name: Model name from HuggingFace
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = None

    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
            except ImportError as e:
                raise EmbeddingError(
                    "Local embeddings require 'sentence-transformers' package"
                ) from e
        return self._model

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._model_name in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self._model_name]
        # Load model to get dimension if not known
        return self.model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

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
            # Run encoding in thread pool since it's CPU/GPU bound
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                ),
            )

            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return [e.tolist() for e in embeddings]

        except Exception as e:
            raise EmbeddingError(f"Local embedding failed: {e}") from e

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Some sentence-transformers models use different prompts for queries.
        This method handles that automatically.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        # Check if model has query instruction (BGE models)
        if "bge" in self._model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embeddings = await self.embed([query])
        return embeddings[0]
