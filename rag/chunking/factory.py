"""Chunker factory for creating appropriate chunkers."""

from rag.chunking.heading import HeadingChunker
from rag.chunking.recursive import RecursiveChunker
from rag.chunking.semantic import SemanticChunker
from rag.chunking.sliding_window import SlidingWindowChunker
from rag.models.document import ChunkingStrategy
from rag.protocols import Chunker


class ChunkerFactory:
    """Factory for creating document chunkers."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer: str = "cl100k_base",
    ):
        """Initialize factory with default settings.

        Args:
            chunk_size: Default chunk size in tokens
            chunk_overlap: Default overlap between chunks
            tokenizer: Default tokenizer name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def get_chunker(
        self,
        strategy: ChunkingStrategy | str,
        **kwargs,
    ) -> Chunker:
        """Get a chunker for the specified strategy.

        Args:
            strategy: Chunking strategy to use
            **kwargs: Additional arguments passed to chunker

        Returns:
            Chunker instance

        Raises:
            ValueError: If strategy is unknown
        """
        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy)

        # Merge default settings with kwargs
        settings = {
            "chunk_size": kwargs.pop("chunk_size", self.chunk_size),
            "chunk_overlap": kwargs.pop("chunk_overlap", self.chunk_overlap),
            "tokenizer": kwargs.pop("tokenizer", self.tokenizer),
        }

        match strategy:
            case ChunkingStrategy.RECURSIVE:
                return RecursiveChunker(**settings, **kwargs)
            case ChunkingStrategy.HEADING:
                return HeadingChunker(**settings, **kwargs)
            case ChunkingStrategy.SLIDING_WINDOW:
                return SlidingWindowChunker(**settings, **kwargs)
            case ChunkingStrategy.SEMANTIC:
                return SemanticChunker(**settings, **kwargs)
            case _:
                raise ValueError(f"Unknown chunking strategy: {strategy}")

    def create_recursive(self, **kwargs) -> RecursiveChunker:
        """Create a recursive chunker."""
        return self.get_chunker(ChunkingStrategy.RECURSIVE, **kwargs)

    def create_heading(self, **kwargs) -> HeadingChunker:
        """Create a heading-based chunker."""
        return self.get_chunker(ChunkingStrategy.HEADING, **kwargs)

    def create_sliding_window(self, **kwargs) -> SlidingWindowChunker:
        """Create a sliding window chunker."""
        return self.get_chunker(ChunkingStrategy.SLIDING_WINDOW, **kwargs)

    def create_semantic(self, **kwargs) -> SemanticChunker:
        """Create a semantic chunker."""
        return self.get_chunker(ChunkingStrategy.SEMANTIC, **kwargs)


# Default factory instance
_factory = ChunkerFactory()


def get_chunker(
    strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
    **kwargs,
) -> Chunker:
    """Get a chunker using the default factory.

    Args:
        strategy: Chunking strategy to use
        **kwargs: Additional arguments

    Returns:
        Chunker instance
    """
    return _factory.get_chunker(strategy, **kwargs)
