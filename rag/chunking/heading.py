"""Heading-based chunker for structured documents."""

import re
from dataclasses import dataclass

from rag.chunking.base import BaseChunker, merge_small_chunks
from rag.models.document import Chunk, Document, DocumentType


@dataclass
class Section:
    """A document section with heading and content."""

    title: str
    level: int
    content: str
    start_char: int
    end_char: int
    path: list[str]


class HeadingChunker(BaseChunker):
    """Chunker that splits documents by headings.

    Preserves document structure by creating chunks that align
    with heading boundaries. Useful for well-structured documents.
    """

    # Heading patterns for different formats
    MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    HTML_HEADING = re.compile(r"<h([1-6])[^>]*>(.+?)</h\1>", re.IGNORECASE | re.DOTALL)
    UNDERLINE_HEADING_1 = re.compile(r"^(.+)\n={3,}$", re.MULTILINE)
    UNDERLINE_HEADING_2 = re.compile(r"^(.+)\n-{3,}$", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        tokenizer: str = "cl100k_base",
        merge_small: bool = True,
    ):
        """Initialize heading chunker.

        Args:
            chunk_size: Maximum chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size (smaller chunks get merged)
            tokenizer: Tiktoken tokenizer name
            merge_small: Whether to merge small chunks
        """
        super().__init__(chunk_size, chunk_overlap, tokenizer)
        self.min_chunk_size = min_chunk_size
        self.merge_small = merge_small

    @property
    def strategy_name(self) -> str:
        return "heading"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document by headings.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        # Extract sections based on document type
        if document.doc_type == DocumentType.MARKDOWN:
            sections = self._parse_markdown_sections(document.content)
        elif document.doc_type == DocumentType.HTML:
            sections = self._parse_html_sections(document.content)
        else:
            # Fall back to markdown-style parsing for other types
            sections = self._parse_markdown_sections(document.content)

        if not sections:
            # No headings found, create single chunk
            return [
                self._create_chunk(
                    content=document.content,
                    document=document,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(document.content),
                )
            ]

        # Convert sections to chunks, splitting large ones
        chunks = []
        for section in sections:
            section_chunks = self._section_to_chunks(section, document, len(chunks))
            chunks.extend(section_chunks)

        # Optionally merge small chunks
        if self.merge_small:
            chunks = merge_small_chunks(
                chunks,
                min_size=self.min_chunk_size,
                tokenizer=self._tokenizer_name,
            )

        return chunks

    def _parse_markdown_sections(self, text: str) -> list[Section]:
        """Parse Markdown headings into sections."""
        sections = []
        heading_stack: list[tuple[str, int]] = []  # (title, level)

        # Find all headings with positions
        headings = []

        # Hash-style headings
        for match in self.MARKDOWN_HEADING.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append((match.start(), level, title))

        # Underline-style headings
        for match in self.UNDERLINE_HEADING_1.finditer(text):
            title = match.group(1).strip()
            headings.append((match.start(), 1, title))

        for match in self.UNDERLINE_HEADING_2.finditer(text):
            title = match.group(1).strip()
            headings.append((match.start(), 2, title))

        # Sort by position
        headings.sort(key=lambda x: x[0])

        # Create sections
        for i, (pos, level, title) in enumerate(headings):
            # Update heading stack
            while heading_stack and heading_stack[-1][1] >= level:
                heading_stack.pop()
            heading_stack.append((title, level))

            # Determine section end
            if i + 1 < len(headings):
                end_pos = headings[i + 1][0]
            else:
                end_pos = len(text)

            # Extract content (excluding heading line)
            content_start = text.find("\n", pos)
            if content_start == -1:
                content_start = pos
            else:
                content_start += 1

            content = text[content_start:end_pos].strip()

            if content:  # Only create section if there's content
                sections.append(
                    Section(
                        title=title,
                        level=level,
                        content=content,
                        start_char=content_start,
                        end_char=end_pos,
                        path=[h[0] for h in heading_stack],
                    )
                )

        # Handle content before first heading
        if headings and headings[0][0] > 0:
            preamble = text[: headings[0][0]].strip()
            if preamble:
                sections.insert(
                    0,
                    Section(
                        title="Introduction",
                        level=0,
                        content=preamble,
                        start_char=0,
                        end_char=headings[0][0],
                        path=["Introduction"],
                    ),
                )

        return sections

    def _parse_html_sections(self, text: str) -> list[Section]:
        """Parse HTML headings into sections.

        Note: This is a simplified parser. For complex HTML,
        consider using BeautifulSoup.
        """
        sections = []
        heading_stack: list[tuple[str, int]] = []

        # Find all headings
        headings = []
        for match in self.HTML_HEADING.finditer(text):
            level = int(match.group(1))
            # Strip HTML tags from title
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            headings.append((match.start(), match.end(), level, title))

        # Create sections
        for i, (start, heading_end, level, title) in enumerate(headings):
            # Update heading stack
            while heading_stack and heading_stack[-1][1] >= level:
                heading_stack.pop()
            heading_stack.append((title, level))

            # Determine section end
            if i + 1 < len(headings):
                end_pos = headings[i + 1][0]
            else:
                end_pos = len(text)

            content = text[heading_end:end_pos].strip()

            if content:
                sections.append(
                    Section(
                        title=title,
                        level=level,
                        content=content,
                        start_char=heading_end,
                        end_char=end_pos,
                        path=[h[0] for h in heading_stack],
                    )
                )

        return sections

    def _section_to_chunks(
        self,
        section: Section,
        document: Document,
        start_index: int,
    ) -> list[Chunk]:
        """Convert a section to one or more chunks.

        Large sections are split while preserving section metadata.
        """
        content = section.content
        tokens = self.count_tokens(content)

        if tokens <= self.chunk_size:
            # Section fits in one chunk
            return [
                self._create_chunk(
                    content=content,
                    document=document,
                    chunk_index=start_index,
                    start_char=section.start_char,
                    end_char=section.end_char,
                    section_title=section.title,
                    section_path=section.path,
                )
            ]

        # Split large section
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        encoded = self.tokenizer.encode(content)
        i = 0

        while i < len(encoded):
            end_idx = min(i + self.chunk_size, len(encoded))
            chunk_tokens = encoded[i:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Approximate character positions
            char_ratio = len(content) / len(encoded) if encoded else 1
            start_char = section.start_char + int(i * char_ratio)
            end_char = section.start_char + int(end_idx * char_ratio)

            chunks.append(
                self._create_chunk(
                    content=chunk_text,
                    document=document,
                    chunk_index=start_index + len(chunks),
                    start_char=start_char,
                    end_char=end_char,
                    section_title=section.title,
                    section_path=section.path,
                )
            )

            i += step
            if end_idx >= len(encoded):
                break

        return chunks
