"""Technical analysis tool."""

import operator
from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from stock_mcp.data.yfinance_client import fetch_history
from stock_mcp.utils.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_macd,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
)
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance
from stock_mcp.utils.validators import FetchParams, check_rule, check_rule_expr


async def technicals(symbol: str) -> dict[str, Any]:
    """
    Calculate technical indicators for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with moving averages, RSI, MACD, ATR, price position, returns, volume
    """
    start_time = perf_counter()

    # Fetch 1 year of daily data for calculations
    params = FetchParams(
        symbol=symbol,
        period="1y",
        interval="1d",
        adjusted=True,
    )

    try:
        df = await fetch_history(params)
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

    if len(df) < 20:
        return build_error_response(
            error_type="insufficient_data",
            message=f"Need at least 20 data points, got {len(df)}",
            symbol=symbol,
        )

    # Extract price series
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    current_price = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else None

    # Moving Averages
    sma_20 = calculate_sma(close, 20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    ema_12 = calculate_ema(close, 12)
    ema_26 = calculate_ema(close, 26)

    sma_20_val = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None
    sma_50_val = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None
    sma_200_val = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None
    ema_12_val = float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else None
    ema_26_val = float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else None

    # Price vs SMA ratios
    price_vs_sma20 = (
        (current_price - sma_20_val) / sma_20_val
        if current_price and sma_20_val
        else None
    )
    price_vs_sma50 = (
        (current_price - sma_50_val) / sma_50_val
        if current_price and sma_50_val
        else None
    )
    price_vs_sma200 = (
        (current_price - sma_200_val) / sma_200_val
        if current_price and sma_200_val
        else None
    )

    moving_averages = {
        "sma_20": round(sma_20_val, 2) if sma_20_val else None,
        "sma_50": round(sma_50_val, 2) if sma_50_val else None,
        "sma_200": round(sma_200_val, 2) if sma_200_val else None,
        "ema_12": round(ema_12_val, 2) if ema_12_val else None,
        "ema_26": round(ema_26_val, 2) if ema_26_val else None,
        "price_vs_sma20": round(price_vs_sma20, 4) if price_vs_sma20 is not None else None,
        "price_vs_sma50": round(price_vs_sma50, 4) if price_vs_sma50 is not None else None,
        "price_vs_sma200": round(price_vs_sma200, 4) if price_vs_sma200 is not None else None,
        "rules": {
            "above_sma20": {
                "triggered": check_rule_expr(current_price, sma_20_val, operator.gt),
                "threshold": "price > sma20",
            },
            "above_sma50": {
                "triggered": check_rule_expr(current_price, sma_50_val, operator.gt),
                "threshold": "price > sma50",
            },
            "above_sma200": {
                "triggered": check_rule_expr(current_price, sma_200_val, operator.gt),
                "threshold": "price > sma200",
            },
            "golden_cross": {
                "triggered": check_rule_expr(sma_50_val, sma_200_val, operator.gt),
                "threshold": "sma50 > sma200",
            },
            "death_cross": {
                "triggered": check_rule_expr(sma_50_val, sma_200_val, operator.lt),
                "threshold": "sma50 < sma200",
            },
        },
    }

    # RSI
    rsi_series = calculate_rsi(close, 14)
    rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None

    rsi = {
        "value": round(rsi_val, 1) if rsi_val is not None else None,
        "period": 14,
        "rules": {
            "overbought": {
                "triggered": check_rule(rsi_val, 70, operator.gt),
                "threshold": 70,
            },
            "oversold": {
                "triggered": check_rule(rsi_val, 30, operator.lt),
                "threshold": 30,
            },
        },
    }

    # MACD
    macd_data = calculate_macd(close, 12, 26, 9)
    macd_line_val = (
        float(macd_data["macd_line"].iloc[-1])
        if not pd.isna(macd_data["macd_line"].iloc[-1])
        else None
    )
    signal_line_val = (
        float(macd_data["signal_line"].iloc[-1])
        if not pd.isna(macd_data["signal_line"].iloc[-1])
        else None
    )
    histogram_val = (
        float(macd_data["histogram"].iloc[-1])
        if not pd.isna(macd_data["histogram"].iloc[-1])
        else None
    )

    macd = {
        "macd_line": round(macd_line_val, 4) if macd_line_val is not None else None,
        "signal_line": round(signal_line_val, 4) if signal_line_val is not None else None,
        "histogram": round(histogram_val, 4) if histogram_val is not None else None,
        "settings": {"fast": 12, "slow": 26, "signal": 9},
        "rules": {
            "bullish_cross": {
                "triggered": check_rule_expr(macd_line_val, signal_line_val, operator.gt),
                "threshold": "macd > signal",
            },
            "bearish_cross": {
                "triggered": check_rule_expr(macd_line_val, signal_line_val, operator.lt),
                "threshold": "macd < signal",
            },
        },
    }

    # ATR
    atr_series = calculate_atr(high, low, close, 14)
    atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None
    atr_pct = atr_val / current_price if atr_val and current_price else None

    atr = {
        "value": round(atr_val, 2) if atr_val is not None else None,
        "value_pct": round(atr_pct, 4) if atr_pct is not None else None,
        "period": 14,
    }

    # Price Position (52-week and 1-month)
    week_52_high = float(high.max()) if not high.isna().all() else None
    week_52_low = float(low.min()) if not low.isna().all() else None

    # 1-month low (approx 21 trading days)
    low_1m = float(low.tail(21).min()) if len(low) >= 21 and not low.tail(21).isna().all() else None

    from_52w_high = (
        (current_price - week_52_high) / week_52_high
        if current_price and week_52_high
        else None
    )
    from_52w_low = (
        (current_price - week_52_low) / week_52_low
        if current_price and week_52_low
        else None
    )
    position_in_range = (
        (current_price - week_52_low) / (week_52_high - week_52_low)
        if current_price and week_52_high and week_52_low and week_52_high != week_52_low
        else None
    )

    price_position = {
        "week_52_high": round(week_52_high, 2) if week_52_high else None,
        "week_52_low": round(week_52_low, 2) if week_52_low else None,
        "low_1m": round(low_1m, 2) if low_1m else None,
        "from_52w_high": round(from_52w_high, 4) if from_52w_high is not None else None,
        "from_52w_low": round(from_52w_low, 4) if from_52w_low is not None else None,
        "position_in_range": round(position_in_range, 4) if position_in_range is not None else None,
    }

    # Returns
    returns = {
        "return_1w": _safe_round(calculate_returns(close, 5), 4),
        "return_1m": _safe_round(calculate_returns(close, 21), 4),
        "return_3m": _safe_round(calculate_returns(close, 63), 4),
        "return_6m": _safe_round(calculate_returns(close, 126), 4),
        "return_ytd": _calculate_ytd_return(df),
        "return_1y": _safe_round(calculate_returns(close, 252), 4),
    }

    # Volume
    current_volume = int(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else None
    avg_volume_20d = float(volume.tail(20).mean()) if len(volume) >= 20 else None
    volume_ratio = (
        current_volume / avg_volume_20d if current_volume and avg_volume_20d else None
    )

    volume_data = {
        "current": current_volume,
        "avg_20d": int(avg_volume_20d) if avg_volume_20d else None,
        "ratio": round(volume_ratio, 2) if volume_ratio else None,
    }

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("technicals", duration_ms),
        "data_provenance": {
            "price": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
                last_bar_date=df["date"].iloc[-1] if len(df) > 0 else None,
            ),
        },
        "symbol": params.symbol,
        "current_price": round(current_price, 2) if current_price else None,
        "moving_averages": moving_averages,
        "rsi": rsi,
        "macd": macd,
        "atr": atr,
        "price_position": price_position,
        "returns": returns,
        "volume": volume_data,
    }


def _safe_round(value: float | None, decimals: int) -> float | None:
    """Round value if not None."""
    if value is None:
        return None
    return round(value, decimals)


def _calculate_ytd_return(df: pd.DataFrame) -> float | None:
    """Calculate YTD return from DataFrame."""
    if len(df) < 2:
        return None

    # Parse dates
    dates = pd.to_datetime(df["date"])
    current_year = dates.iloc[-1].year

    # Find first trading day of current year
    year_start_mask = dates.dt.year == current_year
    if not year_start_mask.any():
        return None

    year_data = df[year_start_mask]
    if len(year_data) < 2:
        return None

    close = pd.to_numeric(year_data["close"], errors="coerce")
    start_price = close.iloc[0]
    end_price = close.iloc[-1]

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None

    return round((end_price - start_price) / start_price, 4)
