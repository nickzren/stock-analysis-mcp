"""Symbol search tool."""

from time import perf_counter
from typing import Any

from stock_analysis.data.provider import read_symbol_matches
from stock_analysis.utils.provenance import (
    build_error_response,
    build_meta,
    build_provenance,
    utcnow_isoformat_z,
)
from stock_analysis.utils.sanitize import sanitize_text

# US exchanges we support
US_EXCHANGES = {"NYSE", "NASDAQ", "NMS", "NYQ", "NGM", "PCX", "AMEX", "BTS", "NCM"}


async def symbol_search(query: str, limit: int = 10) -> dict[str, Any]:
    """
    Search for stock symbols.

    Args:
        query: Search query (company name or ticker)
        limit: Maximum number of results (default: 10)

    Returns:
        Dict with search results and exact match info
    """
    start_time = perf_counter()

    # Read with HTTP status and envelope checks (WP1): a failed search is an
    # error, never an empty result.
    read = await read_symbol_matches(query)
    if not read.verified:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Symbol search failed upstream ({read.error_type})",
        )

    results = []
    for quote in [q for q in read.payload if "symbol" in q][:limit]:
        exchange = quote.get("exchange", "")
        if exchange in US_EXCHANGES:
            quote_type = quote.get("quoteType", "")
            results.append(
                {
                    "symbol": quote.get("symbol"),
                    "name": sanitize_text(
                        quote.get("shortname") or quote.get("longname")
                    ),
                    "exchange": exchange,
                    "type": "etf" if quote_type == "ETF" else "equity",
                    "is_valid": True,
                }
            )

    # Find exact match: compare normalized query to symbol
    # This fixes the issue where first result != exact ticker match
    normalized_query = query.upper().strip()
    exact_match = next(
        (r["symbol"] for r in results if r["symbol"] == normalized_query),
        None,
    )

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("symbol_search", duration_ms),
        "data_provenance": {
            "search": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
                query=query,
            ),
        },
        "results": results,
        "exact_match": exact_match,
    }
