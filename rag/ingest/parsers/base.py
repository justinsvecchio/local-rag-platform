"""Base parser classes and exceptions."""

from abc import ABC, abstractmethod
from datetime import datetime

from rag.models.document import Document, DocumentMetadata, DocumentType


class ParseError(Exception):
    """Raised when document parsing fails."""

    def __init__(
        self,
        message: str,
        filename: str | None = None,
        partial_content: str | None = None,
    ):
        super().__init__(message)
        self.filename = filename
        self.partial_content = partial_content


class BaseParser(ABC):
    """Base class for document parsers."""

    @abstractmethod
    async def parse(self, content: bytes, filename: str) -> Document:
        """Parse raw bytes into a Document.

        Args:
            content: Raw document bytes
            filename: Original filename

        Returns:
            Parsed Document

        Raises:
            ParseError: If parsing fails
        """
        ...

    @property
    @abstractmethod
    def supported_mimetypes(self) -> set[str]:
        """Return set of supported MIME types."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return set of supported file extensions."""
        ...

    def _create_document(
        self,
        content: str,
        doc_type: DocumentType,
        filename: str,
        raw_content: bytes,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        """Helper to create a Document with standard metadata."""
        doc_metadata = DocumentMetadata(
            title=title or filename,
            source_path=filename,
            **(metadata or {}),
        )

        return Document(
            content=content,
            doc_type=doc_type,
            metadata=doc_metadata,
            raw_content=raw_content,
        )

    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract title from filename by removing extension."""
        import re

        # Remove extension
        name = filename.rsplit(".", 1)[0]
        # Replace separators with spaces
        name = re.sub(r"[-_]+", " ", name)
        # Title case
        return name.title()
