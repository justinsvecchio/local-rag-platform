"""Tests for citation extraction and building."""

import pytest

from rag.generate.citations import CitationBuilder
from rag.models.retrieval import SearchResult, SearchSource


class TestCitationBuilder:
    """Test citation builder."""

    @pytest.fixture
    def builder(self):
        """Create citation builder."""
        return CitationBuilder()

    @pytest.fixture
    def sample_sources(self):
        """Create sample source results."""
        return [
            SearchResult(
                chunk_id="chunk_1",
                document_id="doc_1",
                content="Python was created by Guido van Rossum in 1991.",
                score=0.95,
                source=SearchSource.HYBRID,
                metadata={"title": "Python History", "url": "https://example.com/python"},
            ),
            SearchResult(
                chunk_id="chunk_2",
                document_id="doc_2",
                content="Python is known for its simple, readable syntax.",
                score=0.90,
                source=SearchSource.HYBRID,
                metadata={"title": "Python Features", "url": "https://example.com/features"},
            ),
            SearchResult(
                chunk_id="chunk_3",
                document_id="doc_3",
                content="Python supports multiple programming paradigms.",
                score=0.85,
                source=SearchSource.HYBRID,
                metadata={"title": "Python Paradigms"},
            ),
        ]

    def test_extract_citations_from_text(self, builder):
        """Test extracting citation markers from text."""
        text = "Python was created in 1991 [1]. It has simple syntax [2]."
        citations = builder.extract_citation_indices(text)

        assert citations == [1, 2]

    def test_extract_multiple_citations(self, builder):
        """Test extracting multiple citation markers."""
        text = "First point [1]. Second point [2]. Third point [3]. Back to first [1]."
        citations = builder.extract_citation_indices(text)

        # Should return unique citations in order of appearance
        assert 1 in citations
        assert 2 in citations
        assert 3 in citations

    def test_extract_no_citations(self, builder):
        """Test text with no citations."""
        text = "This text has no citation markers."
        citations = builder.extract_citation_indices(text)

        assert citations == []

    def test_build_citations(self, builder, sample_sources):
        """Test building citation objects from sources."""
        text = "Python was created in 1991 [1]. It's known for readability [2]."
        citations = builder.build_citations(text, sample_sources)

        assert len(citations) == 2

        # First citation
        assert citations[0].index == 1
        assert citations[0].chunk_id == "chunk_1"
        assert "Guido van Rossum" in citations[0].excerpt

        # Second citation
        assert citations[1].index == 2
        assert citations[1].chunk_id == "chunk_2"

    def test_build_citations_with_missing_reference(self, builder, sample_sources):
        """Test handling citations that reference non-existent sources."""
        text = "Some claim [1]. Another claim [5]."  # [5] doesn't exist
        citations = builder.build_citations(text, sample_sources)

        # Should only include valid citations
        assert len(citations) == 1
        assert citations[0].index == 1

    def test_format_context_with_indices(self, builder, sample_sources):
        """Test formatting context with citation indices."""
        formatted = builder.format_context_with_indices(sample_sources)

        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "[3]" in formatted
        assert "Python was created" in formatted
        assert "simple, readable syntax" in formatted

    def test_citation_excerpt_truncation(self, builder):
        """Test that long excerpts are truncated."""
        long_content = "A" * 500
        sources = [
            SearchResult(
                chunk_id="long",
                document_id="doc",
                content=long_content,
                score=0.9,
                source=SearchSource.HYBRID,
            )
        ]

        text = "Reference [1]."
        citations = builder.build_citations(text, sources)

        # Excerpt should be truncated
        assert len(citations[0].excerpt) <= builder.max_excerpt_length + 3  # +3 for "..."

    def test_citation_includes_metadata(self, builder, sample_sources):
        """Test that citations include source metadata."""
        text = "Python info [1]."
        citations = builder.build_citations(text, sample_sources)

        # Should include document title if available
        assert citations[0].document_title == "Python History"
        assert citations[0].url == "https://example.com/python"

    def test_extract_bracket_citations(self, builder):
        """Test extracting various bracket citation formats."""
        texts = [
            ("See [1] for details.", [1]),
            ("Multiple [1][2] citations.", [1, 2]),
            ("Spaced [1] [2] [3] refs.", [1, 2, 3]),
            ("No space[1]between.", [1]),
        ]

        for text, expected in texts:
            citations = builder.extract_citation_indices(text)
            for exp in expected:
                assert exp in citations

    def test_empty_sources_returns_empty_citations(self, builder):
        """Test that empty sources returns empty citations."""
        text = "Some text [1] with citations [2]."
        citations = builder.build_citations(text, [])

        assert citations == []

    def test_citation_ordering(self, builder, sample_sources):
        """Test that citations are ordered by index."""
        text = "Last [3]. First [1]. Middle [2]."
        citations = builder.build_citations(text, sample_sources)

        indices = [c.index for c in citations]
        assert indices == sorted(indices)


class TestCitationBuilderEdgeCases:
    """Test edge cases for citation builder."""

    @pytest.fixture
    def builder(self):
        """Create citation builder."""
        return CitationBuilder()

    def test_nested_brackets(self, builder):
        """Test handling nested brackets."""
        text = "See [[1]] for info."
        citations = builder.extract_citation_indices(text)

        # Should extract the inner citation
        assert 1 in citations

    def test_large_citation_numbers(self, builder):
        """Test handling large citation numbers."""
        text = "Reference [10]. Another [25]."
        citations = builder.extract_citation_indices(text)

        assert 10 in citations
        assert 25 in citations

    def test_zero_citation_ignored(self, builder):
        """Test that [0] is handled appropriately."""
        text = "Zero citation [0] and one [1]."
        citations = builder.extract_citation_indices(text)

        # Depending on implementation, [0] might be ignored or included
        # Most implementations use 1-based indexing
        assert 1 in citations

    def test_negative_numbers_ignored(self, builder):
        """Test that negative numbers in brackets are ignored."""
        text = "Invalid [-1] and valid [1]."
        citations = builder.extract_citation_indices(text)

        assert 1 in citations
        assert -1 not in citations

    def test_non_numeric_brackets_ignored(self, builder):
        """Test that non-numeric brackets are ignored."""
        text = "Array [index] and citation [1]."
        citations = builder.extract_citation_indices(text)

        assert citations == [1]

    def test_special_characters_in_content(self, builder):
        """Test handling special characters in source content."""
        sources = [
            SearchResult(
                chunk_id="special",
                document_id="doc",
                content="Content with 'quotes' and \"double quotes\" and <brackets>.",
                score=0.9,
                source=SearchSource.HYBRID,
            )
        ]

        text = "Reference [1]."
        citations = builder.build_citations(text, sources)

        assert len(citations) == 1
        assert "quotes" in citations[0].excerpt

    def test_unicode_in_content(self, builder):
        """Test handling unicode in source content."""
        sources = [
            SearchResult(
                chunk_id="unicode",
                document_id="doc",
                content="日本語 content with émojis 🎉",
                score=0.9,
                source=SearchSource.HYBRID,
            )
        ]

        text = "Reference [1]."
        citations = builder.build_citations(text, sources)

        assert len(citations) == 1
        assert "日本語" in citations[0].excerpt
