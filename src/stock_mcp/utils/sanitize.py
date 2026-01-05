"""Text sanitization utilities."""

import re


def sanitize_text(text: str | None, max_length: int = 500) -> str | None:
    """
    Sanitize untrusted text fields.

    Removes control characters and truncates to max_length.
    Apply to: name, description, sector, industry, any free-text field.

    Args:
        text: Text to sanitize (may be None)
        max_length: Maximum length before truncation

    Returns:
        Sanitized text or None if input was None
    """
    if text is None:
        return None

    # Remove control characters (including \r, \x00-\x1f, \x7f-\x9f)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Truncate if needed
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text.strip()
