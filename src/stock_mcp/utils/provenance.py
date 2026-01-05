"""Data provenance and metadata utilities."""

from datetime import datetime
from typing import Any

from stock_mcp import SCHEMA_VERSION, SERVER_VERSION


def build_meta(tool: str, duration_ms: float | None = None) -> dict[str, Any]:
    """
    Build standard metadata block for responses.

    Args:
        tool: Name of the tool producing this response
        duration_ms: Execution time in milliseconds (optional)

    Returns:
        Metadata dict with version info
    """
    meta: dict[str, Any] = {
        "server_version": SERVER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
    }
    if duration_ms is not None:
        meta["duration_ms"] = round(duration_ms, 1)
    return meta


def build_provenance(
    source: str,
    as_of: datetime | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Build data provenance block for a single data source.

    Args:
        source: Data source name (e.g., "yfinance", "fred")
        as_of: Timestamp of data freshness
        **kwargs: Additional provenance fields

    Returns:
        Provenance dict for this data source
    """
    prov: dict[str, Any] = {"source": source}

    if as_of is not None:
        if isinstance(as_of, datetime):
            prov["as_of"] = as_of.isoformat()
        else:
            prov["as_of"] = as_of

    # Add any additional fields
    prov.update(kwargs)

    # Ensure warnings list exists
    if "warnings" not in prov:
        prov["warnings"] = []

    return prov


def build_error_response(
    error_type: str,
    message: str,
    symbol: str | None = None,
    retry_after_seconds: int | None = None,
) -> dict[str, Any]:
    """
    Build standardized error response.

    Args:
        error_type: Type of error (invalid_symbol, data_unavailable, rate_limited)
        message: Human-readable error message
        symbol: Symbol that caused the error (if applicable)
        retry_after_seconds: Seconds to wait before retry (for rate limiting)

    Returns:
        Error response dict
    """
    response: dict[str, Any] = {
        "error": True,
        "error_type": error_type,
        "message": message,
        "meta": build_meta("error"),
    }

    if symbol is not None:
        response["symbol"] = symbol

    if retry_after_seconds is not None:
        response["retry_after_seconds"] = retry_after_seconds

    return response
