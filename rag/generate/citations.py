"""Citation extraction and formatting."""

import re
from typing import Any

from rag.models.generation import Citation
from rag.models.retrieval import RankedResult


class CitationBuilder:
    """Builds citations from generated text and context."""

    def __init__(
        self,
        min_excerpt_length: int = 20,
        max_excerpt_length: int = 200,
    ):
        """Initialize citation builder.

        Args:
            min_excerpt_length: Minimum excerpt length
            max_excerpt_length: Maximum excerpt length
        """
        self.min_excerpt_length = min_excerpt_length
        self.max_excerpt_length = max_excerpt_length

    def extract_citations(
        self,
        answer: str,
        context: list[RankedResult],
        citation_pattern: str = r"\[(\d+)\]",
    ) -> tuple[str, list[Citation]]:
        """Extract citation references from answer and build Citation objects.

        Args:
            answer: Generated answer text with citation markers like [1], [2]
            context: Ranked context used for generation
            citation_pattern: Regex pattern for citation markers

        Returns:
            Tuple of (answer text, list of Citations)
        """
        citations = []
        used_indices = set()

        # Find all citation markers
        for match in re.finditer(citation_pattern, answer):
            try:
                idx = int(match.group(1)) - 1  # Convert to 0-indexed

                if 0 <= idx < len(context) and idx not in used_indices:
                    used_indices.add(idx)
                    result = context[idx]

                    citations.append(
                        Citation(
                            citation_index=idx + 1,
                            chunk_id=result.chunk_id,
                            document_id=result.document_id,
                            document_title=result.metadata.get("document_title"),
                            excerpt=self._extract_excerpt(result.content),
                            relevance_score=result.final_score,
                            source_url=result.metadata.get("source_url"),
                            page_number=result.metadata.get("page_number"),
                            section_title=result.metadata.get("section_title"),
                        )
                    )
            except (ValueError, IndexError):
                continue

        # Sort citations by index
        citations.sort(key=lambda c: c.citation_index)

        return answer, citations

    def _extract_excerpt(self, content: str) -> str:
        """Extract a meaningful excerpt from content.

        Args:
            content: Full content text

        Returns:
            Excerpt string
        """
        # Clean whitespace
        content = " ".join(content.split())

        if len(content) <= self.max_excerpt_length:
            return content

        # Try to cut at sentence boundary
        truncated = content[: self.max_excerpt_length]
        last_period = truncated.rfind(".")
        last_question = truncated.rfind("?")
        last_exclaim = truncated.rfind("!")

        cut_point = max(last_period, last_question, last_exclaim)

        if cut_point > self.min_excerpt_length:
            return content[: cut_point + 1]

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > self.min_excerpt_length:
            return content[:last_space] + "..."

        return truncated + "..."

    def format_context_for_prompt(
        self,
        context: list[RankedResult],
        max_tokens: int = 4000,
        chars_per_token: int = 4,
    ) -> str:
        """Format context for inclusion in generation prompt.

        Args:
            context: Ranked results to format
            max_tokens: Approximate token limit
            chars_per_token: Estimated characters per token

        Returns:
            Formatted context string with citation markers
        """
        formatted_parts = []
        total_chars = 0
        char_limit = max_tokens * chars_per_token

        for i, result in enumerate(context, start=1):
            # Build header with citation number
            header = f"[{i}]"

            title = result.metadata.get("document_title")
            if title:
                header += f" ({title})"

            section = result.metadata.get("section_title")
            if section:
                header += f" - {section}"

            content = result.content.strip()
            entry = f"{header}\n{content}\n\n"

            if total_chars + len(entry) > char_limit:
                # Truncate this entry to fit
                remaining = char_limit - total_chars - len(header) - 10
                if remaining > 100:
                    entry = f"{header}\n{content[:remaining]}...\n\n"
                else:
                    break

            formatted_parts.append(entry)
            total_chars += len(entry)

        return "".join(formatted_parts)

    def validate_citations(
        self,
        citations: list[Citation],
        context: list[RankedResult],
    ) -> list[Citation]:
        """Validate that citations reference valid context items.

        Args:
            citations: Citations to validate
            context: Original context

        Returns:
            List of valid citations
        """
        valid = []
        context_ids = {r.chunk_id for r in context}

        for citation in citations:
            if citation.chunk_id in context_ids:
                valid.append(citation)

        return valid
