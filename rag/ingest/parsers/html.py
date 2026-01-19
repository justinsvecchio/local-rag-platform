"""HTML document parser using BeautifulSoup."""

import asyncio
import re
from datetime import datetime

from rag.ingest.parsers.base import BaseParser, ParseError
from rag.models.document import Document, DocumentType


class HTMLParser(BaseParser):
    """Parser for HTML documents using BeautifulSoup."""

    # Tags to remove entirely
    REMOVE_TAGS = {
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "canvas",
        "nav",
        "footer",
        "aside",
        "header",
    }

    # Tags that indicate structural breaks
    BLOCK_TAGS = {
        "p",
        "div",
        "article",
        "section",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "tr",
        "blockquote",
        "pre",
    }

    @property
    def supported_mimetypes(self) -> set[str]:
        return {"text/html", "application/xhtml+xml"}

    @property
    def supported_extensions(self) -> set[str]:
        return {".html", ".htm", ".xhtml"}

    async def parse(self, content: bytes, filename: str) -> Document:
        """Parse HTML content into a Document."""
        try:
            # Run parsing in thread pool
            loop = asyncio.get_event_loop()
            text, metadata = await loop.run_in_executor(
                None, self._parse_sync, content, filename
            )

            return self._create_document(
                content=text,
                doc_type=DocumentType.HTML,
                filename=filename,
                raw_content=content,
                title=metadata.get("title"),
                metadata=metadata,
            )
        except Exception as e:
            raise ParseError(
                f"Failed to parse HTML: {e}",
                filename=filename,
            ) from e

    def _parse_sync(self, content: bytes, filename: str) -> tuple[str, dict]:
        """Synchronous HTML parsing."""
        from bs4 import BeautifulSoup

        # Detect encoding
        encoding = self._detect_encoding(content)
        html_text = content.decode(encoding, errors="replace")

        # Parse HTML
        soup = BeautifulSoup(html_text, "lxml")

        # Extract metadata
        metadata = self._extract_metadata(soup)

        # Remove unwanted elements
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract main content
        main_content = self._find_main_content(soup)

        # Convert to text with structure preserved
        text = self._extract_text(main_content)

        # Clean up text
        text = self._clean_text(text)

        return text, metadata

    def _detect_encoding(self, content: bytes) -> str:
        """Detect HTML encoding."""
        # Check for BOM
        if content.startswith(b"\xef\xbb\xbf"):
            return "utf-8"
        if content.startswith(b"\xff\xfe"):
            return "utf-16-le"
        if content.startswith(b"\xfe\xff"):
            return "utf-16-be"

        # Try to find encoding in meta tag
        try:
            # Quick regex search in first 2KB
            head = content[:2048].decode("ascii", errors="ignore")
            match = re.search(
                r'<meta[^>]+charset=["\']?([^"\'\s>]+)', head, re.IGNORECASE
            )
            if match:
                return match.group(1).strip()

            # Check content-type meta
            match = re.search(
                r'<meta[^>]+content=["\'][^"\']*charset=([^"\'\s;]+)',
                head,
                re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
        except Exception:
            pass

        return "utf-8"

    def _extract_metadata(self, soup) -> dict:
        """Extract metadata from HTML head."""
        metadata = {}

        # Title
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        # Meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            prop = meta.get("property", "").lower()
            content = meta.get("content", "")

            if not content:
                continue

            if name == "description" or prop == "og:description":
                metadata["description"] = content
            elif name == "author":
                metadata["author"] = content
            elif name == "keywords":
                metadata["tags"] = [k.strip() for k in content.split(",")]
            elif prop == "article:published_time":
                try:
                    metadata["created_at"] = datetime.fromisoformat(
                        content.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            elif prop == "article:modified_time":
                try:
                    metadata["updated_at"] = datetime.fromisoformat(
                        content.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

        # Canonical URL
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            metadata["source_url"] = canonical["href"]

        return metadata

    def _find_main_content(self, soup):
        """Find the main content area of the page."""
        # Look for common content containers
        selectors = [
            "article",
            "main",
            '[role="main"]',
            ".content",
            ".post-content",
            ".article-content",
            ".entry-content",
            "#content",
            "#main",
        ]

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element

        # Fall back to body
        return soup.body or soup

    def _extract_text(self, element) -> str:
        """Extract text from element preserving structure."""
        from bs4 import NavigableString

        parts = []

        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    parts.append(text)
            elif child.name in self.BLOCK_TAGS:
                # Add newlines around block elements
                child_text = self._extract_text(child)
                if child_text:
                    parts.append(f"\n{child_text}\n")
            else:
                child_text = self._extract_text(child)
                if child_text:
                    parts.append(child_text)

        return " ".join(parts)

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return text.strip()
