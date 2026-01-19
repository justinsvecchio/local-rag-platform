"""Sliding window chunker."""

from rag.chunking.base import BaseChunker
from rag.models.document import Chunk, Document


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window chunker.

    Creates chunks of a fixed token size with a specified overlap.
    Simple but effective for uniform chunking.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer: str = "cl100k_base",
    ):
        """Initialize sliding window chunker.

        Args:
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            tokenizer: Tiktoken tokenizer name
        """
        super().__init__(chunk_size, chunk_overlap, tokenizer)

    @property
    def strategy_name(self) -> str:
        return "sliding_window"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content
        tokens = self.tokenizer.encode(text)

        if len(tokens) == 0:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        # Track character positions for token ranges
        char_positions = self._get_char_positions(text, tokens)

        i = 0
        chunk_index = 0

        while i < len(tokens):
            # Get token range for this chunk
            end_idx = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]

            # Decode tokens to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Get character positions
            start_char = char_positions[i]
            end_char = char_positions[end_idx - 1] + len(
                self.tokenizer.decode([tokens[end_idx - 1]])
            ) if end_idx > 0 else len(text)

            chunks.append(
                self._create_chunk(
                    content=chunk_text,
                    document=document,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                )
            )

            chunk_index += 1
            i += step

            # Stop if we've reached the end
            if end_idx >= len(tokens):
                break

        return chunks

    def _get_char_positions(self, text: str, tokens: list[int]) -> list[int]:
        """Map token indices to character positions.

        This is an approximation since tokenization isn't always
        reversible at character boundaries.

        Args:
            text: Original text
            tokens: Token list

        Returns:
            List of starting character positions for each token
        """
        positions = []
        current_pos = 0

        for i, token in enumerate(tokens):
            positions.append(current_pos)
            # Decode single token to get approximate length
            token_text = self.tokenizer.decode([token])
            current_pos += len(token_text)

        return positions
