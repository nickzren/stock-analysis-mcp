"""Price history tool."""

from time import perf_counter
from typing import Any

from stock_analysis.data.cache import price_cache
from stock_analysis.data.yfinance_client import fetch_history, get_market_state
from stock_analysis.utils.ohlcv import df_to_rows
from stock_analysis.utils.provenance import (
    FetchError,
    build_error_response,
    build_meta,
    build_provenance,
    fetch_or_error,
    utcnow_isoformat_z,
)
from stock_analysis.utils.validators import FetchParams


async def price_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    adjusted: bool = True,
    include_preview: bool = True,
) -> dict[str, Any]:
    """
    Fetch price history and return summary + Resource URI.

    Args:
        symbol: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Bar interval (1m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        adjusted: Whether to use adjusted prices (default: True)
        include_preview: Include last 5 bars in response (default: True)

    Returns:
        Dict with summary, preview (optional), and resource_uri
    """
    start_time = perf_counter()

    try:
        params = FetchParams(
            symbol=symbol,
            period=period,
            interval=interval,
            adjusted=adjusted,
        )
    except ValueError as e:
        return build_error_response(
            error_type="invalid_parameters",
            message=str(e),
            symbol=symbol,
        )

    try:
        # Fetch and standardize (standardization happens in fetch_history)
        df = await fetch_or_error(fetch_history(params), symbol)
    except FetchError as fe:
        return fe.response

    # Store in cache (df is already standardized)
    uri = price_cache.store(params, df)

    summary = _build_summary(df)

    # Get market state
    market_state = get_market_state(params.tz)

    # Get last bar date for provenance
    last_bar_date = df["date"].iloc[-1] if len(df) > 0 else None

    duration_ms = (perf_counter() - start_time) * 1000

    response: dict[str, Any] = {
        "meta": build_meta("price_history", duration_ms),
        "data_provenance": {
            "price": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
                bar_timezone=params.tz,
                price_adjustment="auto_adjust_true" if adjusted else "auto_adjust_false",
                market_state=market_state["state"],
                market_state_method=market_state["method"],
                last_bar_date=last_bar_date,
            ),
        },
        "symbol": params.symbol,
        "period": params.period,
        "interval": params.interval,
        "adjusted": adjusted,
        "summary": summary,
        "resource_uri": uri,
        "resource_rows": len(df),
    }

    if include_preview:
        response["preview"] = df_to_rows(df.tail(5))

    return response


def _build_summary(df: Any) -> dict[str, Any]:
    """Build price-history summary fields."""
    close_prices = df["close"].dropna()
    if len(close_prices) >= 2:
        start_price = float(close_prices.iloc[0])
        end_price = float(close_prices.iloc[-1])
        total_return = (end_price - start_price) / start_price if start_price != 0 else None
    else:
        start_price = float(close_prices.iloc[0]) if len(close_prices) > 0 else None
        end_price = float(close_prices.iloc[-1]) if len(close_prices) > 0 else None
        total_return = None

    return {
        "data_points": len(df),
        "start_date": df["date"].iloc[0] if len(df) > 0 else None,
        "end_date": df["date"].iloc[-1] if len(df) > 0 else None,
        "start_price": start_price,
        "end_price": end_price,
        "period_high": float(df["high"].max()) if not df["high"].isna().all() else None,
        "period_low": float(df["low"].min()) if not df["low"].isna().all() else None,
        "total_return": round(total_return, 4) if total_return is not None else None,
    }
