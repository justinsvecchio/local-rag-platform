"""Embedder factory for creating embedding providers."""

from typing import Literal

from rag.config import get_settings
from rag.embed.openai import EmbeddingError, OpenAIEmbedder
from rag.embed.sentence_transformers import SentenceTransformerEmbedder
from rag.protocols import Embedder


class EmbedderFactory:
    """Factory for creating embedding providers."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        default_provider: Literal["openai", "local"] = "openai",
    ):
        """Initialize factory.

        Args:
            openai_api_key: OpenAI API key (uses env var if None)
            default_provider: Default provider to use
        """
        self._openai_api_key = openai_api_key
        self._default_provider = default_provider

    def get_embedder(
        self,
        provider: Literal["openai", "local"] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> Embedder:
        """Get an embedder instance.

        Args:
            provider: Provider to use ('openai' or 'local')
            model: Model name (uses default for provider if None)
            **kwargs: Additional arguments for embedder

        Returns:
            Embedder instance

        Raises:
            ValueError: If provider is unknown
            EmbeddingError: If provider is unavailable
        """
        provider = provider or self._default_provider

        match provider:
            case "openai":
                return self._create_openai_embedder(model, **kwargs)
            case "local":
                return self._create_local_embedder(model, **kwargs)
            case _:
                raise ValueError(f"Unknown embedding provider: {provider}")

    def _create_openai_embedder(
        self,
        model: str | None = None,
        **kwargs,
    ) -> OpenAIEmbedder:
        """Create OpenAI embedder."""
        settings = get_settings()

        api_key = self._openai_api_key or settings.openai_api_key
        if not api_key:
            raise EmbeddingError(
                "OpenAI embeddings require OPENAI_API_KEY environment variable"
            )

        return OpenAIEmbedder(
            api_key=api_key,
            model=model or settings.embedding.openai_model,
            **kwargs,
        )

    def _create_local_embedder(
        self,
        model: str | None = None,
        **kwargs,
    ) -> SentenceTransformerEmbedder:
        """Create local sentence-transformers embedder."""
        settings = get_settings()

        return SentenceTransformerEmbedder(
            model_name=model or settings.embedding.local_model,
            **kwargs,
        )

    def get_best_available(self) -> Embedder:
        """Get the best available embedder.

        Tries OpenAI first, falls back to local.

        Returns:
            Best available embedder
        """
        settings = get_settings()

        # Try OpenAI if configured
        if settings.has_openai:
            try:
                return self._create_openai_embedder()
            except EmbeddingError:
                pass

        # Fall back to local
        return self._create_local_embedder()


# Module-level factory instance
_factory: EmbedderFactory | None = None


def get_embedder(
    provider: Literal["openai", "local"] | None = None,
    model: str | None = None,
    **kwargs,
) -> Embedder:
    """Get an embedder using the default factory.

    Args:
        provider: Provider to use (uses configured default if None)
        model: Model name
        **kwargs: Additional arguments

    Returns:
        Embedder instance
    """
    global _factory

    if _factory is None:
        settings = get_settings()
        _factory = EmbedderFactory(
            openai_api_key=settings.openai_api_key,
            default_provider=settings.embedding.provider,
        )

    if provider is None and model is None:
        return _factory.get_best_available()

    return _factory.get_embedder(provider, model, **kwargs)
