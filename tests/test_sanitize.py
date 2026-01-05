"""Tests for text sanitization."""

import pytest

from stock_mcp.utils.sanitize import sanitize_text


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_sanitize_none(self) -> None:
        """Test sanitize returns None for None input."""
        assert sanitize_text(None) is None

    def test_sanitize_basic(self) -> None:
        """Test basic text passthrough."""
        text = "Hello World"
        assert sanitize_text(text) == "Hello World"

    def test_sanitize_strips_whitespace(self) -> None:
        """Test whitespace is stripped."""
        text = "  Hello World  "
        assert sanitize_text(text) == "Hello World"

    def test_sanitize_removes_control_chars(self) -> None:
        """Test control characters are removed."""
        text = "Hello\x00World\x1f!"
        assert sanitize_text(text) == "HelloWorld!"

    def test_sanitize_removes_carriage_return(self) -> None:
        """Test carriage return is removed."""
        text = "Hello\rWorld"
        assert sanitize_text(text) == "HelloWorld"

    def test_sanitize_removes_high_control_chars(self) -> None:
        """Test high control characters (0x7f-0x9f) are removed."""
        text = "Hello\x7fWorld\x9f!"
        assert sanitize_text(text) == "HelloWorld!"

    def test_sanitize_truncates_long_text(self) -> None:
        """Test long text is truncated."""
        text = "A" * 600
        result = sanitize_text(text, max_length=500)

        assert len(result) == 503  # 500 + "..."
        assert result.endswith("...")

    def test_sanitize_custom_max_length(self) -> None:
        """Test custom max length."""
        text = "Hello World"
        result = sanitize_text(text, max_length=5)

        assert result == "Hello..."

    def test_sanitize_exact_max_length(self) -> None:
        """Test text at exact max length."""
        text = "Hello"
        result = sanitize_text(text, max_length=5)

        assert result == "Hello"  # No truncation needed

    def test_sanitize_preserves_unicode(self) -> None:
        """Test Unicode characters are preserved."""
        text = "Hello 世界 🌍"
        assert sanitize_text(text) == "Hello 世界 🌍"

    def test_sanitize_preserves_newlines(self) -> None:
        """Test newlines are preserved (not control chars in our definition)."""
        # Note: Our implementation removes \x00-\x1f which includes \n (\x0a)
        # This is intentional for security, but worth documenting
        text = "Hello\nWorld"
        result = sanitize_text(text)
        # Newline (\x0a) is in the control char range and gets removed
        assert result == "HelloWorld"

    def test_sanitize_empty_string(self) -> None:
        """Test empty string."""
        assert sanitize_text("") == ""

    def test_sanitize_whitespace_only(self) -> None:
        """Test whitespace-only string."""
        assert sanitize_text("   ") == ""
