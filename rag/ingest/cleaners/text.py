"""Text cleaning utilities."""

import re
import unicodedata


class TextCleaner:
    """Text cleaner for normalizing and cleaning document text."""

    def __init__(
        self,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        remove_control_chars: bool = True,
        strip_lines: bool = True,
        max_newlines: int = 2,
    ):
        """Initialize cleaner with options.

        Args:
            normalize_unicode: Normalize unicode to NFC form
            normalize_whitespace: Collapse multiple spaces/tabs
            remove_control_chars: Remove control characters
            strip_lines: Strip whitespace from each line
            max_newlines: Maximum consecutive newlines to allow
        """
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.remove_control_chars = remove_control_chars
        self.strip_lines = strip_lines
        self.max_newlines = max_newlines

    def clean(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Remove control characters (except newlines and tabs)
        if self.remove_control_chars:
            text = self._remove_control_chars(text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Strip lines
        if self.strip_lines:
            text = self._strip_lines(text)

        # Limit consecutive newlines
        if self.max_newlines > 0:
            text = self._limit_newlines(text)

        return text.strip()

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        # Keep: \n (0x0A), \r (0x0D), \t (0x09)
        return "".join(
            char
            for char in text
            if not unicodedata.category(char).startswith("C")
            or char in "\n\r\t"
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various space characters with regular space
        text = re.sub(r"[\t\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]+", " ", text)
        # Collapse multiple spaces (but not newlines)
        text = re.sub(r" {2,}", " ", text)
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text

    def _strip_lines(self, text: str) -> str:
        """Strip whitespace from each line."""
        lines = text.split("\n")
        return "\n".join(line.strip() for line in lines)

    def _limit_newlines(self, text: str) -> str:
        """Limit consecutive newlines."""
        pattern = r"\n{" + str(self.max_newlines + 1) + r",}"
        replacement = "\n" * self.max_newlines
        return re.sub(pattern, replacement, text)


class BoilerplateRemover:
    """Remove common boilerplate text from documents."""

    # Common boilerplate patterns
    PATTERNS = [
        # Copyright notices
        r"(?i)copyright\s*©?\s*\d{4}.*$",
        r"(?i)all rights reserved\.?.*$",
        # Page numbers
        r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
        r"^\s*\d+\s*$",  # Standalone numbers (likely page numbers)
        # Headers/footers
        r"(?i)^(draft|confidential|internal use only).*$",
        # URLs (unless they're the main content)
        r"(?i)^https?://\S+\s*$",
    ]

    def __init__(self, patterns: list[str] | None = None):
        """Initialize with patterns.

        Args:
            patterns: Optional custom patterns (uses defaults if None)
        """
        self.patterns = [
            re.compile(p, re.MULTILINE) for p in (patterns or self.PATTERNS)
        ]

    def clean(self, text: str) -> str:
        """Remove boilerplate from text.

        Args:
            text: Text to clean

        Returns:
            Text with boilerplate removed
        """
        for pattern in self.patterns:
            text = pattern.sub("", text)
        return text


class HTMLEntityCleaner:
    """Clean HTML entities from text."""

    # Common HTML entities
    ENTITIES = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&apos;": "'",
        "&#39;": "'",
        "&mdash;": "—",
        "&ndash;": "–",
        "&hellip;": "…",
        "&rsquo;": "'",
        "&lsquo;": "'",
        "&rdquo;": """,
        "&ldquo;": """,
    }

    def clean(self, text: str) -> str:
        """Clean HTML entities from text.

        Args:
            text: Text with potential HTML entities

        Returns:
            Text with entities decoded
        """
        # Handle named entities
        for entity, replacement in self.ENTITIES.items():
            text = text.replace(entity, replacement)

        # Handle numeric entities
        def replace_numeric(match: re.Match) -> str:
            try:
                code = int(match.group(1))
                return chr(code)
            except (ValueError, OverflowError):
                return match.group(0)

        text = re.sub(r"&#(\d+);", replace_numeric, text)

        # Handle hex entities
        def replace_hex(match: re.Match) -> str:
            try:
                code = int(match.group(1), 16)
                return chr(code)
            except (ValueError, OverflowError):
                return match.group(0)

        text = re.sub(r"&#x([0-9a-fA-F]+);", replace_hex, text)

        return text
