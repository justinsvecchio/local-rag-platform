"""Base chunker classes and utilities."""

from abc import ABC, abstractmethod

import tiktoken

from rag.models.document import Chunk, ChunkMetadata, Document


class BaseChunker(ABC):
    """Base class for document chunkers."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer: str = "cl100k_base",
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer: Tiktoken tokenizer name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer_name = tokenizer
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = tiktoken.get_encoding(self._tokenizer_name)
        return self._tokenizer

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Token count
        """
        return len(self.tokenizer.encode(text))

    def _create_chunk(
        self,
        content: str,
        document: Document,
        chunk_index: int,
        start_char: int,
        end_char: int,
        section_title: str | None = None,
        section_path: list[str] | None = None,
        page_number: int | None = None,
    ) -> Chunk:
        """Create a Chunk with standard metadata.

        Args:
            content: Chunk text content
            document: Source document
            chunk_index: Index of this chunk in the document
            start_char: Starting character position
            end_char: Ending character position
            section_title: Optional section title
            section_path: Optional section hierarchy path
            page_number: Optional page number

        Returns:
            Chunk object
        """
        token_count = self.count_tokens(content)

        metadata = ChunkMetadata(
            document_id=document.id,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            token_count=token_count,
            section_title=section_title,
            section_path=section_path or [],
            page_number=page_number,
            # Copy relevant document metadata
            source_url=document.metadata.source_url,
            document_title=document.metadata.title,
            document_created_at=document.metadata.created_at,
            document_updated_at=document.metadata.updated_at,
        )

        return Chunk(content=content, metadata=metadata)


def merge_small_chunks(
    chunks: list[Chunk],
    min_size: int = 100,
    tokenizer: str = "cl100k_base",
) -> list[Chunk]:
    """Merge chunks that are too small.

    Args:
        chunks: List of chunks to potentially merge
        min_size: Minimum chunk size in tokens
        tokenizer: Tokenizer name for counting

    Returns:
        List of chunks with small ones merged
    """
    if not chunks:
        return chunks

    enc = tiktoken.get_encoding(tokenizer)
    merged = []
    current = None

    for chunk in chunks:
        tokens = len(enc.encode(chunk.content))

        if current is None:
            current = chunk
            continue

        current_tokens = len(enc.encode(current.content))

        if current_tokens < min_size:
            # Merge with next chunk
            merged_content = current.content + "\n\n" + chunk.content
            merged_metadata = ChunkMetadata(
                document_id=current.metadata.document_id,
                chunk_index=current.metadata.chunk_index,
                start_char=current.metadata.start_char,
                end_char=chunk.metadata.end_char,
                token_count=len(enc.encode(merged_content)),
                section_title=current.metadata.section_title,
                section_path=current.metadata.section_path,
                source_url=current.metadata.source_url,
                document_title=current.metadata.document_title,
                document_created_at=current.metadata.document_created_at,
                document_updated_at=current.metadata.document_updated_at,
            )
            current = Chunk(content=merged_content, metadata=merged_metadata)
        else:
            merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    # Re-index chunks
    for i, chunk in enumerate(merged):
        chunk.metadata.chunk_index = i

    return merged
