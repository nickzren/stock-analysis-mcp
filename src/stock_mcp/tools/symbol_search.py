"""Symbol search tool."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import perf_counter
from typing import Any

from yfinance import Search

from stock_mcp.utils.provenance import build_meta, build_provenance
from stock_mcp.utils.sanitize import sanitize_text

# US exchanges we support
US_EXCHANGES = {"NYSE", "NASDAQ", "NMS", "NYQ", "NGM", "PCX", "AMEX", "BTS", "NCM"}

_executor = ThreadPoolExecutor(max_workers=2)


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

    loop = asyncio.get_running_loop()

    def _search() -> list[dict[str, Any]]:
        search = Search(query)
        results = []

        for quote in search.quotes[:limit]:
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

        return results

    results = await loop.run_in_executor(_executor, _search)

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
                as_of=datetime.utcnow().isoformat() + "Z",
                query=query,
            ),
        },
        "results": results,
        "exact_match": exact_match,
    }
