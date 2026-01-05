"""Technical indicator calculations."""

import math

import numpy as np
import pandas as pd


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        prices: Price series (typically close prices)
        period: Number of periods for the average

    Returns:
        SMA series
    """
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        prices: Price series (typically close prices)
        period: Number of periods for the average

    Returns:
        EMA series
    """
    return prices.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Uses Wilder's smoothing method (exponential moving average).

    Args:
        prices: Price series (typically close prices)
        period: RSI period (default: 14)

    Returns:
        RSI series (0-100 scale)
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing: alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero (when avg_loss is 0)
    rsi = rsi.replace([np.inf, -np.inf], 100)

    return rsi


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series (typically close prices)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Dict with 'macd_line', 'signal_line', 'histogram' series
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default: 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing for ATR
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return atr


def calculate_returns(prices: pd.Series, periods: int) -> float | None:
    """
    Calculate return over a specific number of periods.

    Args:
        prices: Price series
        periods: Number of periods to look back

    Returns:
        Return as decimal (0.15 = 15%), or None if insufficient data
    """
    if len(prices) < periods + 1:
        return None

    current = prices.iloc[-1]
    past = prices.iloc[-periods - 1]

    if pd.isna(current) or pd.isna(past) or past == 0:
        return None

    return (current - past) / past


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float | None:
    """
    Calculate volatility (standard deviation of returns).

    Args:
        returns: Returns series
        annualize: Whether to annualize (assumes daily data, 252 trading days)

    Returns:
        Volatility as decimal, or None if insufficient data
    """
    if len(returns) < 20:
        return None

    std = returns.std()
    if pd.isna(std):
        return None

    if annualize:
        return std * math.sqrt(252)
    return std


def calculate_max_drawdown(prices: pd.Series) -> float | None:
    """
    Calculate maximum drawdown.

    Args:
        prices: Price series

    Returns:
        Max drawdown as negative decimal (-0.20 = 20% drawdown), or None
    """
    if len(prices) < 2:
        return None

    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax

    min_dd = drawdown.min()
    if pd.isna(min_dd):
        return None

    return float(min_dd)


def calculate_current_drawdown(prices: pd.Series) -> tuple[float | None, int | None]:
    """
    Calculate current drawdown from peak.

    Args:
        prices: Price series

    Returns:
        Tuple of (current_drawdown, days_since_high)
    """
    if len(prices) < 2:
        return None, None

    current = prices.iloc[-1]
    peak = prices.max()
    peak_idx = prices.idxmax()

    if pd.isna(current) or pd.isna(peak) or peak == 0:
        return None, None

    drawdown = (current - peak) / peak

    # Calculate days since high
    if isinstance(peak_idx, pd.Timestamp):
        last_date = prices.index[-1]
        if isinstance(last_date, pd.Timestamp):
            days_since = (last_date - peak_idx).days
        else:
            days_since = len(prices) - prices.index.get_loc(peak_idx) - 1
    else:
        days_since = len(prices) - peak_idx - 1

    return float(drawdown), int(days_since)


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float | None:
    """
    Calculate Value at Risk (historical method).

    Args:
        returns: Returns series
        confidence: Confidence level (default: 0.95 for 95% VaR)

    Returns:
        VaR as positive decimal (0.03 = 3% loss), or None
    """
    if len(returns) < 100:
        return None

    percentile = (1 - confidence) * 100
    var = returns.quantile(percentile / 100)

    if pd.isna(var):
        return None

    # Return as positive number (represents potential loss)
    return -float(var)


def calculate_beta(
    symbol_returns: pd.Series,
    benchmark_returns: pd.Series,
    min_overlap: int = 200,
) -> dict[str, float | str | int | None]:
    """
    Calculate beta with strict date alignment.

    Args:
        symbol_returns: Symbol's daily returns
        benchmark_returns: Benchmark's daily returns
        min_overlap: Minimum required overlapping days

    Returns:
        Dict with beta value, overlap info, and any warnings
    """
    # Inner join on dates
    aligned = pd.concat([symbol_returns, benchmark_returns], axis=1, join="inner")
    aligned.columns = ["symbol", "benchmark"]
    aligned = aligned.dropna()

    if len(aligned) < min_overlap:
        return {
            "value": None,
            "warning": f"insufficient_overlap:{len(aligned)}_days",
            "overlap_days": len(aligned),
        }

    cov = aligned["symbol"].cov(aligned["benchmark"])
    var = aligned["benchmark"].var()

    if var == 0:
        return {"value": None, "warning": "zero_benchmark_variance", "overlap_days": len(aligned)}

    return {
        "value": round(cov / var, 3),
        "overlap_days": len(aligned),
        "date_range": {
            "start": str(aligned.index.min().date()),
            "end": str(aligned.index.max().date()),
        },
        "warning": None,
    }


def calculate_pairwise_correlations(
    returns_dict: dict[str, pd.Series],
    min_overlap: int = 100,
) -> dict[str, list | float | int | None]:
    """
    Calculate pairwise correlations with individual overlap tracking.

    Args:
        returns_dict: Dict mapping symbol to returns series
        min_overlap: Minimum required overlapping days per pair

    Returns:
        Dict with pairs, high correlation pairs, and summary stats
    """
    symbols = list(returns_dict.keys())
    pairs: list[dict] = []

    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i + 1 :]:
            aligned = pd.concat(
                [returns_dict[sym1], returns_dict[sym2]],
                axis=1,
                join="inner",
            ).dropna()

            overlap = len(aligned)

            if overlap < min_overlap:
                corr = None
                warning = f"insufficient_overlap:{overlap}"
            else:
                corr = round(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]), 3)
                warning = None

            pairs.append(
                {
                    "symbols": [sym1, sym2],
                    "correlation": corr,
                    "overlap_days": overlap,
                    "warning": warning,
                }
            )

    # High correlation: absolute value (catch negative correlations too)
    high_corr = [p for p in pairs if p["correlation"] and abs(p["correlation"]) > 0.7]

    valid_corrs = [p["correlation"] for p in pairs if p["correlation"] is not None]

    return {
        "pairs": pairs,
        "high_correlation_pairs": high_corr,
        "avg_correlation": round(sum(valid_corrs) / len(valid_corrs), 3) if valid_corrs else None,
        "avg_abs_correlation": (
            round(sum(abs(c) for c in valid_corrs) / len(valid_corrs), 3) if valid_corrs else None
        ),
        "total_pairs": len(pairs),
        "pairs_with_data": len(valid_corrs),
    }
