"""Parser factory for creating appropriate parsers."""

import mimetypes
from pathlib import Path

from rag.ingest.parsers.base import BaseParser, ParseError
from rag.ingest.parsers.docx import DocxParser
from rag.ingest.parsers.html import HTMLParser
from rag.ingest.parsers.pdf import PDFParser
from rag.ingest.parsers.text import MarkdownParser, TextParser


class ParserFactory:
    """Factory for creating document parsers."""

    def __init__(self):
        """Initialize factory with default parsers."""
        self._parsers: list[BaseParser] = [
            PDFParser(),
            HTMLParser(),
            DocxParser(),
            MarkdownParser(),
            TextParser(),
        ]

        # Build lookup tables
        self._by_mimetype: dict[str, BaseParser] = {}
        self._by_extension: dict[str, BaseParser] = {}

        for parser in self._parsers:
            for mime in parser.supported_mimetypes:
                self._by_mimetype[mime] = parser
            for ext in parser.supported_extensions:
                self._by_extension[ext.lower()] = parser

    def get_parser(self, filename: str, mimetype: str | None = None) -> BaseParser:
        """Get appropriate parser for a file.

        Args:
            filename: Filename to determine parser
            mimetype: Optional MIME type hint

        Returns:
            Appropriate parser for the file type

        Raises:
            ParseError: If no parser is available
        """
        # Try MIME type first
        if mimetype and mimetype in self._by_mimetype:
            return self._by_mimetype[mimetype]

        # Try extension
        ext = Path(filename).suffix.lower()
        if ext in self._by_extension:
            return self._by_extension[ext]

        # Try to guess MIME type
        guessed_mime, _ = mimetypes.guess_type(filename)
        if guessed_mime and guessed_mime in self._by_mimetype:
            return self._by_mimetype[guessed_mime]

        raise ParseError(
            f"No parser available for file type: {filename} (mimetype: {mimetype})",
            filename=filename,
        )

    def register_parser(self, parser: BaseParser) -> None:
        """Register a custom parser.

        Args:
            parser: Parser to register
        """
        self._parsers.append(parser)

        for mime in parser.supported_mimetypes:
            self._by_mimetype[mime] = parser
        for ext in parser.supported_extensions:
            self._by_extension[ext.lower()] = parser

    @property
    def supported_extensions(self) -> set[str]:
        """Get all supported file extensions."""
        return set(self._by_extension.keys())

    @property
    def supported_mimetypes(self) -> set[str]:
        """Get all supported MIME types."""
        return set(self._by_mimetype.keys())


# Global factory instance
_factory = ParserFactory()


def get_parser(filename: str, mimetype: str | None = None) -> BaseParser:
    """Get parser for a file using the global factory.

    Args:
        filename: Filename to determine parser
        mimetype: Optional MIME type hint

    Returns:
        Appropriate parser
    """
    return _factory.get_parser(filename, mimetype)
