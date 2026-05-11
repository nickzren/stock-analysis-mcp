"""Technical analysis tool."""

import operator
from time import perf_counter
from typing import Any

import pandas as pd

from stock_analysis.data.yfinance_client import fetch_history
from stock_analysis.utils.helpers import safe_last_float, safe_round
from stock_analysis.utils.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_fibonacci_levels,
    calculate_macd,
    calculate_obv,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
)
from stock_analysis.utils.provenance import (
    FetchError,
    build_error_response,
    build_meta,
    build_provenance,
    fetch_or_error,
    utcnow_isoformat_z,
)
from stock_analysis.utils.validators import FetchParams, check_rule, check_rule_expr


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
        df = await fetch_or_error(fetch_history(params), symbol)
    except FetchError as fe:
        return fe.response

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

    current_price = safe_last_float(close)

    moving_averages = _build_moving_averages(close, current_price)
    rsi = _build_rsi(close)
    macd = _build_macd(close)
    atr = _build_atr(high, low, close, current_price)
    price_position = _build_price_position(high, low, current_price)
    returns = _build_returns(df, close)
    volume_data = _build_volume(volume)
    bollinger = _build_bollinger(close, current_price)
    obv_data = _build_obv(close, volume)
    fib_levels = _build_fibonacci(price_position, current_price)
    price_action = _build_price_action(close)

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("technicals", duration_ms),
        "data_provenance": {
            "price": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
                last_bar_date=df["date"].iloc[-1] if len(df) > 0 else None,
            ),
        },
        "symbol": params.symbol,
        "current_price": safe_round(current_price, 2),
        "moving_averages": moving_averages,
        "rsi": rsi,
        "macd": macd,
        "atr": atr,
        "price_position": price_position,
        "returns": returns,
        "volume": volume_data,
        "bollinger": bollinger,
        "obv": obv_data,
        "fibonacci": fib_levels,
        "price_action": price_action,
    }


def _build_moving_averages(
    close: pd.Series,
    current_price: float | None,
) -> dict[str, Any]:
    """Build moving average values and rules."""
    sma_20 = calculate_sma(close, 20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    ema_12 = calculate_ema(close, 12)
    ema_26 = calculate_ema(close, 26)

    sma_20_val = safe_last_float(sma_20)
    sma_50_val = safe_last_float(sma_50)
    sma_200_val = safe_last_float(sma_200)
    ema_12_val = safe_last_float(ema_12)
    ema_26_val = safe_last_float(ema_26)

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

    return {
        "sma_20": safe_round(sma_20_val, 2),
        "sma_50": safe_round(sma_50_val, 2),
        "sma_200": safe_round(sma_200_val, 2),
        "ema_12": safe_round(ema_12_val, 2),
        "ema_26": safe_round(ema_26_val, 2),
        "sma_200_slope_pct_per_day": safe_round(
            _calc_sma_slope_pct_per_day(sma_200, slope_window=20),
            6,
        ),
        "price_vs_sma20": safe_round(price_vs_sma20, 4),
        "price_vs_sma50": safe_round(price_vs_sma50, 4),
        "price_vs_sma200": safe_round(price_vs_sma200, 4),
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


def _build_rsi(close: pd.Series) -> dict[str, Any]:
    """Build RSI values, divergence, and rules."""
    rsi_series = calculate_rsi(close, 14)
    rsi_val = safe_last_float(rsi_series)
    bullish_divergence = _detect_bullish_rsi_divergence(
        close=close,
        rsi=rsi_series,
        pivot_window=5,
        lookback=60,
    )

    return {
        "value": safe_round(rsi_val, 1),
        "period": 14,
        "bullish_divergence": bullish_divergence,
        "divergence_lookback": 60,
        "divergence_pivot_window": 5,
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


def _build_macd(close: pd.Series) -> dict[str, Any]:
    """Build MACD values and rules."""
    macd_data = calculate_macd(close, 12, 26, 9)
    macd_line_val = safe_last_float(macd_data["macd_line"])
    signal_line_val = safe_last_float(macd_data["signal_line"])
    histogram_val = safe_last_float(macd_data["histogram"])

    hist_series = macd_data["histogram"].dropna()
    hist_rising_3d = None
    if len(hist_series) >= 4:
        hist_rising_3d = bool(
            hist_series.iloc[-1] > hist_series.iloc[-2] > hist_series.iloc[-3]
        )

    return {
        "macd_line": safe_round(macd_line_val, 4),
        "signal_line": safe_round(signal_line_val, 4),
        "histogram": safe_round(histogram_val, 4),
        "histogram_rising_3d": hist_rising_3d,
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


def _build_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    current_price: float | None,
) -> dict[str, Any]:
    """Build ATR values."""
    atr_series = calculate_atr(high, low, close, 14)
    atr_val = safe_last_float(atr_series)
    atr_pct: float | None = None
    if atr_val is not None and current_price is not None and current_price > 0:
        atr_pct = atr_val / current_price

    return {
        "value": safe_round(atr_val, 2),
        "value_pct": safe_round(atr_pct, 4),
        "period": 14,
    }


def _build_price_position(
    high: pd.Series,
    low: pd.Series,
    current_price: float | None,
) -> dict[str, Any]:
    """Build 52-week and recent high/low position fields."""
    week_52_high = float(high.max()) if not high.isna().all() else None
    week_52_low = float(low.min()) if not low.isna().all() else None
    low_1m = float(low.tail(21).min()) if len(low) >= 21 and not low.tail(21).isna().all() else None
    high_3m = float(high.tail(63).max()) if len(high) >= 63 and not high.tail(63).isna().all() else None
    high_6m = float(high.tail(126).max()) if len(high) >= 126 and not high.tail(126).isna().all() else None

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
    from_3m_high = (
        (current_price - high_3m) / high_3m
        if current_price and high_3m
        else None
    )
    from_6m_high = (
        (current_price - high_6m) / high_6m
        if current_price and high_6m
        else None
    )
    position_in_range = (
        (current_price - week_52_low) / (week_52_high - week_52_low)
        if current_price and week_52_high and week_52_low and week_52_high != week_52_low
        else None
    )

    return {
        "week_52_high": safe_round(week_52_high, 2),
        "week_52_low": safe_round(week_52_low, 2),
        "low_1m": safe_round(low_1m, 2),
        "from_52w_high": safe_round(from_52w_high, 4),
        "from_52w_low": safe_round(from_52w_low, 4),
        "from_3m_high": safe_round(from_3m_high, 4),
        "from_6m_high": safe_round(from_6m_high, 4),
        "days_since_52w_high": _days_since_extreme(high, kind="high"),
        "days_since_52w_low": _days_since_extreme(low, kind="low"),
        "position_in_range": safe_round(position_in_range, 4),
    }


def _build_returns(df: pd.DataFrame, close: pd.Series) -> dict[str, Any]:
    """Build trailing return fields."""
    return_1y = calculate_returns(close, 252)
    if return_1y is None and len(close) >= 200:
        return_1y = calculate_returns(close, len(close) - 1)

    return {
        "return_1w": safe_round(calculate_returns(close, 5), 4),
        "return_1w_zscore": safe_round(_zscore_weekly_return(close, lookback_weeks=104), 2),
        "return_1m": safe_round(calculate_returns(close, 21), 4),
        "return_3m": safe_round(calculate_returns(close, 63), 4),
        "return_6m": safe_round(calculate_returns(close, 126), 4),
        "return_ytd": _calculate_ytd_return(df),
        "return_1y": safe_round(return_1y, 4),
    }


def _build_volume(volume: pd.Series) -> dict[str, Any]:
    """Build volume summary fields."""
    current_volume = int(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else None
    avg_volume_20d = float(volume.tail(20).mean()) if len(volume) >= 20 else None
    volume_ratio: float | None = None
    if current_volume is not None and avg_volume_20d is not None and avg_volume_20d > 0:
        volume_ratio = current_volume / avg_volume_20d

    return {
        "current": current_volume,
        "avg_20d": int(avg_volume_20d) if avg_volume_20d is not None else None,
        "ratio": safe_round(volume_ratio, 2),
    }


def _build_bollinger(close: pd.Series, current_price: float | None) -> dict[str, Any]:
    """Build Bollinger Band values and rules."""
    bb = calculate_bollinger_bands(close)
    return {
        **bb,
        "rules": {
            "above_upper": {
                "triggered": current_price > bb["upper"] if bb["upper"] is not None else None,
                "threshold": "price > upper_band",
            },
            "below_lower": {
                "triggered": current_price < bb["lower"] if bb["lower"] is not None else None,
                "threshold": "price < lower_band",
            },
            "squeeze": {
                "triggered": bb["bandwidth"] is not None and bb["bandwidth"] < 0.05,
                "threshold": "bandwidth < 0.05",
            },
        },
    }


def _build_obv(close: pd.Series, volume: pd.Series) -> dict[str, Any]:
    """Build On-Balance Volume fields."""
    obv_series = calculate_obv(close, volume)
    if obv_series is None or len(obv_series) < 20:
        return {"current": None, "sma_20": None, "trend": None}

    obv_current = float(obv_series.iloc[-1])
    obv_sma20 = float(obv_series.tail(20).mean())
    return {
        "current": round(obv_current, 0),
        "sma_20": round(obv_sma20, 0),
        "trend": "rising" if obv_current > obv_sma20 else "falling",
    }


def _build_fibonacci(
    price_position: dict[str, Any],
    current_price: float | None,
) -> dict[str, Any] | None:
    """Build Fibonacci retracement levels from the 52-week range."""
    w52_high = price_position.get("week_52_high")
    w52_low = price_position.get("week_52_low")
    if w52_high is None or w52_low is None or w52_high <= w52_low:
        return None

    fib_levels = calculate_fibonacci_levels(w52_high, w52_low)
    fib_values = sorted(fib_levels.values())
    fib_levels["nearest_support"] = max((v for v in fib_values if v <= current_price), default=None)
    fib_levels["nearest_resistance"] = min((v for v in fib_values if v >= current_price), default=None)
    return fib_levels


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


def _calc_sma_slope_pct_per_day(
    sma_series: pd.Series,
    slope_window: int = 20,
) -> float | None:
    """Calculate SMA slope as percent change per day over slope_window."""
    series = sma_series.dropna()
    if len(series) < slope_window + 1:
        return None
    start = series.iloc[-(slope_window + 1)]
    end = series.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return None
    return float((end - start) / start / slope_window)


def _zscore_weekly_return(
    close_series: pd.Series,
    lookback_weeks: int = 104,
) -> float | None:
    """Z-score of the most recent 1-week return vs trailing weekly returns."""
    weekly = close_series.pct_change(5).dropna()
    if len(weekly) < 20:
        return None
    weekly = weekly.tail(lookback_weeks)
    std = weekly.std()
    if std == 0 or pd.isna(std):
        return None
    return float((weekly.iloc[-1] - weekly.mean()) / std)


def _days_since_extreme(series: pd.Series, *, kind: str) -> int | None:
    """Days since most recent extreme (high/low) in series."""
    clean = series.dropna().reset_index(drop=True)
    if len(clean) == 0:
        return None
    idx = int(clean.idxmax()) if kind == "high" else int(clean.idxmin())
    return int(len(clean) - 1 - idx)


def _detect_bullish_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    pivot_window: int = 5,
    lookback: int = 60,
) -> bool | None:
    """Detect bullish RSI divergence (price lower low, RSI higher low)."""
    c = close.dropna().tail(lookback).reset_index(drop=True)
    r = rsi.dropna().tail(lookback).reset_index(drop=True)
    if len(c) < (2 * pivot_window + 5) or len(r) != len(c):
        return None

    lows: list[int] = []
    for i in range(pivot_window, len(c) - pivot_window):
        window = c[i - pivot_window:i + pivot_window + 1]
        if c[i] == window.min():
            lows.append(i)

    if len(lows) < 2:
        return None

    i1, i2 = lows[-2], lows[-1]
    price_lower_low = c[i2] < c[i1]
    rsi_higher_low = r[i2] > r[i1]
    return bool(price_lower_low and rsi_higher_low)


def _build_price_action(close_series: pd.Series) -> dict[str, bool | None]:
    """Compute simple price action triggers for entry timing."""
    clean = close_series.dropna().reset_index(drop=True)
    if len(clean) < 3:
        return {
            "higher_closes_2d": None,
            "break_5d_high": None,
        }
    higher_closes_2d = bool(clean.iloc[-1] > clean.iloc[-2] > clean.iloc[-3])
    break_5d_high = None
    if len(clean) >= 6:
        prior_5d_high = clean.iloc[-6:-1].max()
        break_5d_high = bool(clean.iloc[-1] > prior_5d_high)
    return {
        "higher_closes_2d": higher_closes_2d,
        "break_5d_high": break_5d_high,
    }
