"""Recursive text splitter chunker."""

from rag.chunking.base import BaseChunker
from rag.models.document import Chunk, Document


class RecursiveChunker(BaseChunker):
    """Recursive text splitter that respects document structure.

    Splits text recursively using a hierarchy of separators,
    trying to keep semantically related text together.
    """

    # Default separators in order of preference
    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Multiple paragraph breaks
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentence boundaries
        "? ",
        "! ",
        "; ",
        ", ",      # Clause boundaries
        " ",       # Word boundaries
        "",        # Character level (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        tokenizer: str = "cl100k_base",
    ):
        """Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            separators: List of separators to use (in order of preference)
            tokenizer: Tiktoken tokenizer name
        """
        super().__init__(chunk_size, chunk_overlap, tokenizer)
        self.separators = separators or self.DEFAULT_SEPARATORS

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks recursively.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        text = document.content
        splits = self._split_text(text, self.separators)

        # Merge splits into chunks respecting size limits
        chunks = []
        current_texts = []
        current_tokens = 0
        char_position = 0

        for split in splits:
            split_tokens = self.count_tokens(split)

            # If this split alone exceeds chunk size, we need to further split
            if split_tokens > self.chunk_size:
                # First, flush current buffer
                if current_texts:
                    chunk_text = "".join(current_texts)
                    chunks.append(
                        self._create_chunk(
                            content=chunk_text,
                            document=document,
                            chunk_index=len(chunks),
                            start_char=char_position - len(chunk_text),
                            end_char=char_position,
                        )
                    )
                    current_texts = []
                    current_tokens = 0

                # Split the large text further
                sub_splits = self._force_split(split, self.chunk_size)
                for sub_split in sub_splits:
                    chunks.append(
                        self._create_chunk(
                            content=sub_split,
                            document=document,
                            chunk_index=len(chunks),
                            start_char=char_position,
                            end_char=char_position + len(sub_split),
                        )
                    )
                    char_position += len(sub_split)
                continue

            # Check if adding this split would exceed chunk size
            if current_tokens + split_tokens > self.chunk_size and current_texts:
                # Create chunk from current buffer
                chunk_text = "".join(current_texts)
                start_char = char_position - len(chunk_text)
                chunks.append(
                    self._create_chunk(
                        content=chunk_text,
                        document=document,
                        chunk_index=len(chunks),
                        start_char=start_char,
                        end_char=char_position,
                    )
                )

                # Handle overlap
                overlap_texts, overlap_tokens = self._get_overlap(
                    current_texts, self.chunk_overlap
                )
                current_texts = overlap_texts
                current_tokens = overlap_tokens

            current_texts.append(split)
            current_tokens += split_tokens
            char_position += len(split)

        # Don't forget the last chunk
        if current_texts:
            chunk_text = "".join(current_texts)
            chunks.append(
                self._create_chunk(
                    content=chunk_text.strip(),
                    document=document,
                    chunk_index=len(chunks),
                    start_char=char_position - len(chunk_text),
                    end_char=char_position,
                )
            )

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of text splits
        """
        if not text:
            return []

        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level split as last resort
            return list(text)

        if separator not in text:
            # Try next separator
            return self._split_text(text, remaining_separators)

        # Split by this separator
        parts = text.split(separator)
        splits = []

        for i, part in enumerate(parts):
            if not part:
                continue

            # Check if this part needs further splitting
            if self.count_tokens(part) > self.chunk_size:
                # Recursively split with remaining separators
                sub_splits = self._split_text(part, remaining_separators)
                splits.extend(sub_splits)
            else:
                splits.append(part)

            # Add separator back (except for last part)
            if i < len(parts) - 1 and separator:
                splits.append(separator)

        return splits

    def _force_split(self, text: str, max_tokens: int) -> list[str]:
        """Force split text to fit within max tokens.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per split

        Returns:
            List of text splits
        """
        tokens = self.tokenizer.encode(text)
        splits = []

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            splits.append(chunk_text)

        return splits

    def _get_overlap(
        self, texts: list[str], overlap_tokens: int
    ) -> tuple[list[str], int]:
        """Get texts for overlap from the end of current chunk.

        Args:
            texts: Current text segments
            overlap_tokens: Target overlap in tokens

        Returns:
            Tuple of (overlap texts, total overlap tokens)
        """
        if not texts or overlap_tokens <= 0:
            return [], 0

        overlap_texts = []
        total_tokens = 0

        for text in reversed(texts):
            text_tokens = self.count_tokens(text)
            if total_tokens + text_tokens <= overlap_tokens:
                overlap_texts.insert(0, text)
                total_tokens += text_tokens
            else:
                break

        return overlap_texts, total_tokens
