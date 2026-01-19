"""Document ingestion pipeline."""

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from rag.chunking.factory import get_chunker
from rag.ingest.cleaners.pipeline import CleanerPipeline
from rag.ingest.parsers.factory import get_parser
from rag.models.document import Chunk, ChunkingStrategy, Document, DocumentMetadata
from rag.protocols import Chunker, Embedder, LexicalIndex, VectorIndex


@dataclass
class IngestResult:
    """Result of document ingestion."""

    document_id: str
    doc_type: str
    chunk_count: int
    total_tokens: int
    processing_time_ms: float
    metadata: dict[str, Any]


class IngestPipeline:
    """Pipeline for ingesting documents into the RAG system.

    Handles parsing, cleaning, chunking, embedding, and indexing.
    """

    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        lexical_index: LexicalIndex | None = None,
        embedder: Embedder | None = None,
        cleaner: CleanerPipeline | None = None,
        default_chunker: Chunker | None = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            vector_index: Vector index for storing embeddings
            lexical_index: Lexical index for BM25 search
            embedder: Embedder for generating embeddings
            cleaner: Text cleaner pipeline
            default_chunker: Default chunker to use
        """
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.embedder = embedder
        self.cleaner = cleaner or CleanerPipeline()
        self.default_chunker = default_chunker

    async def process(
        self,
        content: bytes,
        filename: str,
        chunking_strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        metadata: dict[str, Any] | None = None,
        skip_indexing: bool = False,
    ) -> IngestResult:
        """Process a document through the ingestion pipeline.

        Args:
            content: Raw document bytes
            filename: Original filename
            chunking_strategy: Chunking strategy to use
            chunk_size: Chunk size in tokens
            chunk_overlap: Chunk overlap in tokens
            metadata: Additional metadata to attach
            skip_indexing: If True, skip vector/lexical indexing

        Returns:
            IngestResult with processing details
        """
        start_time = time.perf_counter()

        # 1. Parse document
        parser = get_parser(filename)
        document = await parser.parse(content, filename)

        # 2. Add custom metadata
        if metadata:
            document.metadata = DocumentMetadata(
                **document.metadata.model_dump(),
                **metadata,
            )

        # 3. Compute content hash
        document.metadata.content_hash = self._compute_hash(document.content)

        # 4. Clean text
        document = Document(
            id=document.id,
            content=self.cleaner.clean(document.content),
            doc_type=document.doc_type,
            metadata=document.metadata,
            raw_content=document.raw_content,
            parsed_at=document.parsed_at,
        )

        # 5. Chunk document
        chunker = get_chunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk(document)

        # 6. Index chunks (if not skipped)
        if not skip_indexing:
            await self._index_chunks(chunks)

        # Calculate processing time
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Calculate total tokens
        total_tokens = sum(c.metadata.token_count or 0 for c in chunks)

        return IngestResult(
            document_id=document.id,
            doc_type=document.doc_type.value,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            processing_time_ms=processing_time_ms,
            metadata={
                "title": document.metadata.title,
                "source_url": document.metadata.source_url,
                "source_path": document.metadata.source_path,
                "content_hash": document.metadata.content_hash,
                "chunking_strategy": chunking_strategy
                if isinstance(chunking_strategy, str)
                else chunking_strategy.value,
            },
        )

    async def process_document(
        self,
        document: Document,
        chunking_strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        skip_indexing: bool = False,
    ) -> tuple[list[Chunk], IngestResult]:
        """Process an already-parsed document.

        Args:
            document: Parsed document
            chunking_strategy: Chunking strategy
            chunk_size: Chunk size in tokens
            chunk_overlap: Chunk overlap in tokens
            skip_indexing: If True, skip indexing

        Returns:
            Tuple of (chunks, IngestResult)
        """
        start_time = time.perf_counter()

        # Clean text
        cleaned_content = self.cleaner.clean(document.content)
        document = Document(
            id=document.id,
            content=cleaned_content,
            doc_type=document.doc_type,
            metadata=document.metadata,
            raw_content=document.raw_content,
            parsed_at=document.parsed_at,
        )

        # Chunk
        chunker = get_chunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunker.chunk(document)

        # Index
        if not skip_indexing:
            await self._index_chunks(chunks)

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        total_tokens = sum(c.metadata.token_count or 0 for c in chunks)

        result = IngestResult(
            document_id=document.id,
            doc_type=document.doc_type.value,
            chunk_count=len(chunks),
            total_tokens=total_tokens,
            processing_time_ms=processing_time_ms,
            metadata={
                "title": document.metadata.title,
                "chunking_strategy": chunking_strategy
                if isinstance(chunking_strategy, str)
                else chunking_strategy.value,
            },
        )

        return chunks, result

    async def _index_chunks(self, chunks: list[Chunk]) -> None:
        """Index chunks in vector and lexical indexes.

        Args:
            chunks: Chunks to index
        """
        if not chunks:
            return

        # Generate embeddings
        if self.embedder and self.vector_index:
            texts = [c.content for c in chunks]
            embeddings = await self.embedder.embed(texts)

            # Prepare data for vector index
            ids = [c.id for c in chunks]
            payloads = [
                {
                    "document_id": c.metadata.document_id,
                    "content": c.content,
                    "chunk_index": c.metadata.chunk_index,
                    "section_title": c.metadata.section_title,
                    "document_title": c.metadata.document_title,
                    "source_url": c.metadata.source_url,
                    "document_created_at": (
                        c.metadata.document_created_at.isoformat()
                        if c.metadata.document_created_at
                        else None
                    ),
                    "token_count": c.metadata.token_count,
                }
                for c in chunks
            ]

            await self.vector_index.upsert(ids, embeddings, payloads)

        # Index in lexical index
        if self.lexical_index:
            for chunk in chunks:
                self.lexical_index.index(
                    doc_id=chunk.id,
                    text=chunk.content,
                    metadata={
                        "document_id": chunk.metadata.document_id,
                        "document_title": chunk.metadata.document_title,
                        "source_url": chunk.metadata.source_url,
                        "document_created_at": chunk.metadata.document_created_at,
                    },
                )

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content.

        Args:
            content: Text content

        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(content.encode()).hexdigest()

    async def delete_document(self, document_id: str) -> None:
        """Delete a document and all its chunks from indexes.

        Args:
            document_id: ID of document to delete
        """
        # This would require querying for all chunk IDs for this document
        # Implementation depends on how we track document-chunk relationships
        # For now, this is a placeholder
        pass
