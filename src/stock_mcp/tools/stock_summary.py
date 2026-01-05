"""Stock summary tool."""

from datetime import datetime
from time import perf_counter
from typing import Any

from stock_mcp.data.yfinance_client import fetch_info
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
        "current_price": _safe_float(current_price),
        "previous_close": _safe_float(previous_close),
        "market_cap": _safe_int(info.get("marketCap")),
        "avg_volume_30d": _safe_int(avg_volume),
        "shares_outstanding": _safe_int(info.get("sharesOutstanding")),
        "dividend_yield": _safe_round(div_yield, 4),
        "description": sanitize_text(info.get("longBusinessSummary"), max_length=500),
    }


def _safe_float(value: Any) -> float | None:
    """Convert to float or return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert to int or return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_round(value: Any, decimals: int) -> float | None:
    """Round to decimals or return None."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None
