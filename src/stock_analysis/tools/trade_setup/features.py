"""Setup-level technical features computed from a standardized daily OHLCV frame."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stock_analysis.utils.helpers import safe_round

BANDWIDTH_WINDOW = 20
BANDWIDTH_STD_MULT = 4.0  # (upper - lower) / middle with 2.0-std bands
BANDWIDTH_LOOKBACK = 126  # ~6 months of trading days
SWING_PIVOT_WINDOW = 3
SWING_LOOKBACK = 20


def compute_setup_features(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """Pure feature extraction. Expects standardize_ohlcv columns.

    Returns None when fewer than 21 rows (cannot compute prior-20d levels).
    """
    if df is None or len(df) < 21:
        return None

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    last_close = float(close.iloc[-1])
    prior_close = float(close.iloc[-2])
    prior_day_high = float(high.iloc[-2])
    prior_day_low = float(low.iloc[-2])

    # Prior-20-day levels EXCLUDE the current bar so "close above 20d high"
    # is satisfiable by the last bar.
    high_20d_prior = float(high.iloc[-21:-1].max())
    low_20d_prior = float(low.iloc[-21:-1].min())

    swing_low = _most_recent_pivot_low(low)

    prior_volume = volume.iloc[-21:-1]
    avg_volume_20d = float(prior_volume.mean()) if not prior_volume.isna().all() else None
    last_volume = float(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else None
    last_volume_ratio = (
        last_volume / avg_volume_20d
        if last_volume is not None and avg_volume_20d
        else None
    )

    return {
        "last_close": safe_round(last_close, 2),
        "prior_close": safe_round(prior_close, 2),
        "prior_day_high": safe_round(prior_day_high, 2),
        "prior_day_low": safe_round(prior_day_low, 2),
        "high_20d_prior": safe_round(high_20d_prior, 2),
        "low_20d_prior": safe_round(low_20d_prior, 2),
        "swing_low": safe_round(swing_low, 2),
        "last_volume_ratio": safe_round(last_volume_ratio, 2),
        "bandwidth_pctile_6m": safe_round(_bandwidth_percentile(close), 2),
        "trigger_state": {
            "close_above_prior_day_high": last_close > prior_day_high,
            "close_above_20d_high": last_close > high_20d_prior,
        },
    }


def _most_recent_pivot_low(low: pd.Series) -> float | None:
    """Most recent strict pivot low (strictly below all neighbors within
    ±SWING_PIVOT_WINDOW) within SWING_LOOKBACK; falls back to the lookback min."""
    clean = low.dropna().reset_index(drop=True)
    if len(clean) < 2 * SWING_PIVOT_WINDOW + 1:
        return None
    start = max(SWING_PIVOT_WINDOW, len(clean) - SWING_LOOKBACK)
    pivot: float | None = None
    for i in range(start, len(clean) - SWING_PIVOT_WINDOW):
        window = clean[i - SWING_PIVOT_WINDOW: i + SWING_PIVOT_WINDOW + 1]
        neighbors_min = min(
            window.iloc[:SWING_PIVOT_WINDOW].min(),
            window.iloc[SWING_PIVOT_WINDOW + 1:].min(),
        )
        if clean[i] < neighbors_min:
            pivot = float(clean[i])
    if pivot is None:
        pivot = float(clean.tail(SWING_LOOKBACK).min())
    return pivot


def _bandwidth_percentile(close: pd.Series) -> float | None:
    """Current Bollinger bandwidth's percentile within its trailing 6 months."""
    clean = close.dropna().reset_index(drop=True)
    if len(clean) < BANDWIDTH_WINDOW + 10:
        return None
    mid = clean.rolling(BANDWIDTH_WINDOW).mean()
    std = clean.rolling(BANDWIDTH_WINDOW).std()
    bandwidth = (BANDWIDTH_STD_MULT * std) / mid
    bw = bandwidth.dropna().tail(BANDWIDTH_LOOKBACK)
    if len(bw) < 10:
        return None
    current = bw.iloc[-1]
    return float((bw <= current).mean())
