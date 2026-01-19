"""Generator factory."""

from typing import Literal

from rag.config import get_settings
from rag.generate.anthropic import AnthropicGenerator
from rag.generate.openai import OpenAIGenerator
from rag.protocols import Generator


def get_generator(
    provider: Literal["openai", "anthropic"] | None = None,
    model: str | None = None,
    **kwargs,
) -> Generator:
    """Get a generator instance.

    Args:
        provider: Provider to use ('openai' or 'anthropic')
        model: Model name (uses config default if None)
        **kwargs: Additional arguments

    Returns:
        Generator instance
    """
    settings = get_settings()

    # Use configured provider if not specified
    if provider is None:
        provider = settings.llm.provider
        if provider == "ollama":
            # Ollama uses OpenAI-compatible API
            provider = "openai"

    match provider:
        case "openai":
            return OpenAIGenerator(
                api_key=settings.openai_api_key,
                model=model or settings.llm.openai_model,
                **kwargs,
            )
        case "anthropic":
            return AnthropicGenerator(
                api_key=settings.anthropic_api_key,
                model=model or settings.llm.anthropic_model,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown generator provider: {provider}")


def get_best_available_generator() -> Generator:
    """Get the best available generator based on configuration.

    Tries providers in order: configured provider, OpenAI, Anthropic.

    Returns:
        Best available generator
    """
    settings = get_settings()

    # Try configured provider first
    try:
        if settings.llm.provider == "openai" and settings.has_openai:
            return get_generator("openai")
        if settings.llm.provider == "anthropic" and settings.has_anthropic:
            return get_generator("anthropic")
    except Exception:
        pass

    # Fall back to any available provider
    if settings.has_openai:
        return get_generator("openai")
    if settings.has_anthropic:
        return get_generator("anthropic")

    raise RuntimeError(
        "No LLM provider available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
    )
