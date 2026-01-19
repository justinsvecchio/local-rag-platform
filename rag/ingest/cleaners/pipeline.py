"""Cleaner pipeline for composing multiple cleaners."""

from rag.ingest.cleaners.text import (
    BoilerplateRemover,
    HTMLEntityCleaner,
    TextCleaner,
)
from rag.protocols import Cleaner


class CleanerPipeline:
    """Pipeline for composing multiple text cleaners."""

    def __init__(self, cleaners: list[Cleaner] | None = None):
        """Initialize pipeline with cleaners.

        Args:
            cleaners: List of cleaners to apply in order.
                     If None, uses a default pipeline.
        """
        if cleaners is None:
            self.cleaners = self._default_cleaners()
        else:
            self.cleaners = cleaners

    def _default_cleaners(self) -> list[Cleaner]:
        """Create default cleaner pipeline."""
        return [
            HTMLEntityCleaner(),
            TextCleaner(
                normalize_unicode=True,
                normalize_whitespace=True,
                remove_control_chars=True,
                strip_lines=True,
                max_newlines=2,
            ),
        ]

    def clean(self, text: str) -> str:
        """Apply all cleaners to text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text

    def add_cleaner(self, cleaner: Cleaner) -> "CleanerPipeline":
        """Add a cleaner to the pipeline.

        Args:
            cleaner: Cleaner to add

        Returns:
            Self for chaining
        """
        self.cleaners.append(cleaner)
        return self

    def insert_cleaner(self, index: int, cleaner: Cleaner) -> "CleanerPipeline":
        """Insert a cleaner at a specific position.

        Args:
            index: Position to insert at
            cleaner: Cleaner to insert

        Returns:
            Self for chaining
        """
        self.cleaners.insert(index, cleaner)
        return self


def create_default_pipeline() -> CleanerPipeline:
    """Create a default cleaner pipeline for general use."""
    return CleanerPipeline()


def create_strict_pipeline() -> CleanerPipeline:
    """Create a strict cleaner pipeline that also removes boilerplate."""
    return CleanerPipeline(
        cleaners=[
            HTMLEntityCleaner(),
            BoilerplateRemover(),
            TextCleaner(
                normalize_unicode=True,
                normalize_whitespace=True,
                remove_control_chars=True,
                strip_lines=True,
                max_newlines=2,
            ),
        ]
    )
