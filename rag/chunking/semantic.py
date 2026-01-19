"""Semantic chunker using sentence embeddings."""

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from rag.chunking.base import BaseChunker
from rag.models.document import Chunk, Document

if TYPE_CHECKING:
    from rag.protocols import Embedder


class SemanticChunker(BaseChunker):
    """Chunker that uses semantic similarity to find chunk boundaries.

    Splits text at points where semantic similarity between
    consecutive sentences is low, creating more coherent chunks.
    """

    def __init__(
        self,
        embedder: "Embedder | None" = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.5,
        min_sentences: int = 2,
        tokenizer: str = "cl100k_base",
    ):
        """Initialize semantic chunker.

        Args:
            embedder: Embedder for computing sentence similarities.
                     If None, will use a lightweight local model.
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            similarity_threshold: Threshold below which to split
            min_sentences: Minimum sentences per chunk
            tokenizer: Tiktoken tokenizer name
        """
        super().__init__(chunk_size, chunk_overlap, tokenizer)
        self._embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences
        self._local_model = None

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document using semantic boundaries.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        # Run async embedding in sync context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._chunk_async(document))
        finally:
            loop.close()

    async def _chunk_async(self, document: Document) -> list[Chunk]:
        """Async implementation of chunking."""
        text = document.content

        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [
                self._create_chunk(
                    content=text,
                    document=document,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                )
            ]

        # Get embeddings for sentences
        embeddings = await self._get_embeddings([s for s, _, _ in sentences])

        # Find split points based on semantic similarity
        split_points = self._find_split_points(embeddings, sentences)

        # Create chunks from split points
        chunks = self._create_chunks_from_splits(
            sentences, split_points, document
        )

        return chunks

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with positions.

        Returns list of (sentence, start_pos, end_pos) tuples.
        """
        import re

        # Simple sentence splitting
        pattern = r"(?<=[.!?])\s+"
        parts = re.split(pattern, text)

        sentences = []
        pos = 0

        for part in parts:
            part = part.strip()
            if part:
                start = text.find(part, pos)
                if start == -1:
                    start = pos
                end = start + len(part)
                sentences.append((part, start, end))
                pos = end

        return sentences

    async def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for texts."""
        if self._embedder:
            embeddings = await self._embedder.embed(texts)
            return np.array(embeddings)
        else:
            # Use lightweight local model
            return self._get_local_embeddings(texts)

    def _get_local_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings using a local model."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                # Fall back to random embeddings (for testing)
                return np.random.rand(len(texts), 384)

        return self._local_model.encode(texts, show_progress_bar=False)

    def _find_split_points(
        self,
        embeddings: np.ndarray,
        sentences: list[tuple[str, int, int]],
    ) -> list[int]:
        """Find optimal split points based on semantic similarity.

        Returns indices where chunks should be split.
        """
        if len(embeddings) <= 1:
            return []

        # Compute cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find split points where similarity is below threshold
        # and respecting min_sentences constraint
        split_points = []
        last_split = 0
        current_tokens = 0

        for i, sim in enumerate(similarities):
            sent_text = sentences[i][0]
            sent_tokens = self.count_tokens(sent_text)
            current_tokens += sent_tokens

            # Check if we should split here
            sentences_since_split = i - last_split + 1
            should_split = (
                sim < self.similarity_threshold
                and sentences_since_split >= self.min_sentences
            )

            # Also split if we're approaching chunk size limit
            if current_tokens >= self.chunk_size * 0.8:
                should_split = True

            if should_split:
                split_points.append(i + 1)
                last_split = i + 1
                current_tokens = 0

        return split_points

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _create_chunks_from_splits(
        self,
        sentences: list[tuple[str, int, int]],
        split_points: list[int],
        document: Document,
    ) -> list[Chunk]:
        """Create chunks from split points."""
        chunks = []
        split_indices = [0] + split_points + [len(sentences)]

        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]
            if not chunk_sentences:
                continue

            content = " ".join(s[0] for s in chunk_sentences)
            start_char = chunk_sentences[0][1]
            end_char = chunk_sentences[-1][2]

            chunks.append(
                self._create_chunk(
                    content=content,
                    document=document,
                    chunk_index=len(chunks),
                    start_char=start_char,
                    end_char=end_char,
                )
            )

        return chunks
