"""Stock summary tool."""

from datetime import datetime
from time import perf_counter
from typing import Any

from stock_mcp.data.yfinance_client import fetch_info
from stock_mcp.utils.helpers import safe_float, safe_int, safe_round
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance
from stock_mcp.utils.sanitize import sanitize_text


async def stock_summary(symbol: str) -> dict[str, Any]:
    """
    Get basic stock summary information.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with name, sector, industry, exchange, prices, market cap, volume
    """
    start_time = perf_counter()

    try:
        info = await fetch_info(symbol)
    except ValueError as e:
        return build_error_response(
            error_type="invalid_symbol",
            message=str(e),
            symbol=symbol,
        )
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch data: {e}",
            symbol=symbol,
        )

    # Extract and sanitize fields
    normalized_symbol = symbol.upper().strip()

    # Prices
    current_price = info.get("regularMarketPrice") or info.get("currentPrice")
    previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

    # Volume
    avg_volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")

    # Dividend yield (convert to decimal if present)
    div_yield = info.get("dividendYield")
    if div_yield is not None:
        # yfinance sometimes returns as decimal, sometimes as percent
        if div_yield > 1:  # Likely a percentage
            div_yield = div_yield / 100

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("stock_summary", duration_ms),
        "data_provenance": {
            "fundamentals": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
            ),
        },
        "symbol": normalized_symbol,
        "name": sanitize_text(info.get("shortName") or info.get("longName")),
        "sector": sanitize_text(info.get("sector")),
        "industry": sanitize_text(info.get("industry")),
        "exchange": info.get("exchange"),
        "currency": info.get("currency", "USD"),
        "current_price": safe_float(current_price),
        "previous_close": safe_float(previous_close),
        "market_cap": safe_int(info.get("marketCap")),
        "avg_volume_30d": safe_int(avg_volume),
        "shares_outstanding": safe_int(info.get("sharesOutstanding")),
        "dividend_yield": safe_round(div_yield, 4),
        "employees": safe_int(info.get("fullTimeEmployees")),
        "website": sanitize_text(info.get("website")),
        "description": sanitize_text(info.get("longBusinessSummary"), max_length=500),
    }


