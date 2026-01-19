"""PDF document parser using unstructured."""

import asyncio
from io import BytesIO

from rag.ingest.parsers.base import BaseParser, ParseError
from rag.models.document import Document, DocumentType


class PDFParser(BaseParser):
    """Parser for PDF documents using unstructured library."""

    @property
    def supported_mimetypes(self) -> set[str]:
        return {"application/pdf"}

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    async def parse(self, content: bytes, filename: str) -> Document:
        """Parse PDF content into a Document.

        Uses unstructured library for robust PDF parsing including
        OCR for scanned documents.
        """
        try:
            # Run parsing in thread pool since unstructured is sync
            loop = asyncio.get_event_loop()
            text, metadata = await loop.run_in_executor(
                None, self._parse_sync, content, filename
            )

            return self._create_document(
                content=text,
                doc_type=DocumentType.PDF,
                filename=filename,
                raw_content=content,
                title=metadata.get("title"),
                metadata=metadata,
            )
        except ImportError as e:
            raise ParseError(
                f"PDF parsing requires 'unstructured[pdf]' package: {e}",
                filename=filename,
            ) from e
        except Exception as e:
            raise ParseError(
                f"Failed to parse PDF: {e}",
                filename=filename,
            ) from e

    def _parse_sync(self, content: bytes, filename: str) -> tuple[str, dict]:
        """Synchronous PDF parsing."""
        try:
            from unstructured.partition.pdf import partition_pdf
        except ImportError:
            # Fallback to basic PDF parsing
            return self._parse_with_pypdf(content, filename)

        # Parse with unstructured
        elements = partition_pdf(
            file=BytesIO(content),
            strategy="fast",  # Use 'hi_res' for OCR
            include_page_breaks=True,
        )

        # Extract text
        texts = []
        for element in elements:
            if hasattr(element, "text") and element.text:
                texts.append(element.text)

        text = "\n\n".join(texts)

        # Extract metadata
        metadata = {}
        if elements:
            first_elem = elements[0]
            if hasattr(first_elem, "metadata"):
                elem_meta = first_elem.metadata
                if hasattr(elem_meta, "filename"):
                    metadata["source_filename"] = elem_meta.filename
                if hasattr(elem_meta, "page_number"):
                    metadata["total_pages"] = max(
                        e.metadata.page_number
                        for e in elements
                        if hasattr(e.metadata, "page_number")
                    )

        return text, metadata

    def _parse_with_pypdf(self, content: bytes, filename: str) -> tuple[str, dict]:
        """Fallback PDF parsing using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ParseError(
                "No PDF parser available. Install 'unstructured[pdf]' or 'pypdf'",
                filename=filename,
            )

        reader = PdfReader(BytesIO(content))

        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

        text = "\n\n".join(texts)

        metadata = {
            "total_pages": len(reader.pages),
        }

        # Extract PDF metadata
        if reader.metadata:
            if reader.metadata.title:
                metadata["title"] = reader.metadata.title
            if reader.metadata.author:
                metadata["author"] = reader.metadata.author
            if reader.metadata.creation_date:
                metadata["created_at"] = reader.metadata.creation_date

        return text, metadata
