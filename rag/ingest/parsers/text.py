"""Plain text and Markdown parser."""

import asyncio
import re

from rag.ingest.parsers.base import BaseParser, ParseError
from rag.models.document import Document, DocumentType


class TextParser(BaseParser):
    """Parser for plain text files."""

    @property
    def supported_mimetypes(self) -> set[str]:
        return {"text/plain"}

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt", ".text"}

    async def parse(self, content: bytes, filename: str) -> Document:
        """Parse text content into a Document."""
        try:
            # Detect encoding
            encoding = self._detect_encoding(content)
            text = content.decode(encoding, errors="replace")

            return self._create_document(
                content=text.strip(),
                doc_type=DocumentType.TXT,
                filename=filename,
                raw_content=content,
            )
        except Exception as e:
            raise ParseError(
                f"Failed to parse text file: {e}",
                filename=filename,
            ) from e

    def _detect_encoding(self, content: bytes) -> str:
        """Detect text encoding."""
        # Check for BOM
        if content.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if content.startswith(b"\xff\xfe"):
            return "utf-16-le"
        if content.startswith(b"\xfe\xff"):
            return "utf-16-be"

        # Try UTF-8 first
        try:
            content.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass

        # Try common encodings
        for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
            try:
                content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue

        return "utf-8"


class MarkdownParser(BaseParser):
    """Parser for Markdown files."""

    @property
    def supported_mimetypes(self) -> set[str]:
        return {"text/markdown", "text/x-markdown"}

    @property
    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown", ".mdown", ".mkd"}

    async def parse(self, content: bytes, filename: str) -> Document:
        """Parse Markdown content into a Document."""
        try:
            loop = asyncio.get_event_loop()
            text, metadata = await loop.run_in_executor(
                None, self._parse_sync, content, filename
            )

            return self._create_document(
                content=text,
                doc_type=DocumentType.MARKDOWN,
                filename=filename,
                raw_content=content,
                title=metadata.get("title"),
                metadata=metadata,
            )
        except Exception as e:
            raise ParseError(
                f"Failed to parse Markdown: {e}",
                filename=filename,
            ) from e

    def _parse_sync(self, content: bytes, filename: str) -> tuple[str, dict]:
        """Synchronous Markdown parsing."""
        text = content.decode("utf-8", errors="replace")

        # Extract YAML frontmatter if present
        metadata, text = self._extract_frontmatter(text)

        # Extract title from first H1 if not in frontmatter
        if "title" not in metadata:
            title = self._extract_title(text)
            if title:
                metadata["title"] = title

        return text.strip(), metadata

    def _extract_frontmatter(self, text: str) -> tuple[dict, str]:
        """Extract YAML frontmatter from Markdown."""
        metadata = {}

        # Check for YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                text = parts[2].strip()

                # Parse simple YAML
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip().strip("\"'")
                        if value:
                            metadata[key] = value

        return metadata, text

    def _extract_title(self, text: str) -> str | None:
        """Extract title from first H1 heading."""
        # Look for # heading
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Look for underline-style heading
        lines = text.split("\n")
        for i, line in enumerate(lines[:-1]):
            if lines[i + 1].strip() and all(c == "=" for c in lines[i + 1].strip()):
                return line.strip()

        return None
