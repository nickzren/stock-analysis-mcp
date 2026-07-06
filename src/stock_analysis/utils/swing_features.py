"""Swing feature extraction shared by the trade card and short-term technicals."""

from __future__ import annotations

from datetime import date
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

    core_scalars = (
        last_close, prior_close, prior_day_high, prior_day_low,
        high_20d_prior, low_20d_prior,
    )
    if any(pd.isna(v) for v in core_scalars):
        return None

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


SHORT_TERM_COMPRESSION_PCTILE = 0.25
GAP_EPSILON = 0.001  # |gap_pct| below this is "no gap"


def build_short_term_block(
    df: pd.DataFrame | None,
    *,
    today: date,
    session: str,
) -> dict[str, Any] | None:
    """Daily-derived short-term block for get_technicals (spec: step 2).

    Pure: `today` and `session` are injected. Returns None below 21 rows,
    mirroring compute_setup_features.
    """
    if df is None or len(df) < 21:
        return None

    open_ = pd.to_numeric(df["open"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    levels = {
        "prior_day_high": _clean(high.iloc[-2]),
        "prior_day_low": _clean(low.iloc[-2]),
        "prior_5d_high": _clean(high.iloc[-6:-1].max()),
        "prior_5d_low": _clean(low.iloc[-6:-1].min()),
        "prior_20d_high": _clean(high.iloc[-21:-1].max()),
        "prior_20d_low": _clean(low.iloc[-21:-1].min()),
        "swing_low": _most_recent_pivot_low(low),
        "swing_high": _most_recent_pivot_high(high),
    }

    prior_close = _clean(close.iloc[-2])
    open_today = _clean(open_.iloc[-1])
    low_today = _clean(low.iloc[-1])
    high_today = _clean(high.iloc[-1])
    gap_pct: float | None = None
    direction: str | None = None
    filled: bool | None = None
    if prior_close and open_today is not None:
        gap_pct = (open_today - prior_close) / prior_close
        if abs(gap_pct) >= GAP_EPSILON:
            direction = "up" if gap_pct > 0 else "down"
            if direction == "up" and low_today is not None:
                filled = low_today <= prior_close
            elif direction == "down" and high_today is not None:
                filled = high_today >= prior_close

    prior_volume = volume.iloc[-21:-1]
    avg_volume_20d = float(prior_volume.mean()) if not prior_volume.isna().all() else None
    last_volume = _clean(volume.iloc[-1])
    rvol_value = (
        last_volume / avg_volume_20d
        if last_volume is not None and avg_volume_20d
        else None
    )
    last_bar_date = _bar_date(df)
    rvol_basis = (
        "partial_day"
        if last_bar_date == today and session == "regular"
        else "full_day"
    )

    bandwidth_pctile = _bandwidth_percentile(close)
    return {
        "levels": {k: safe_round(v, 2) for k, v in levels.items()},
        "gap": {
            "gap_pct": safe_round(gap_pct, 4),
            "direction": direction,
            "filled": filled,
            "prior_close": safe_round(prior_close, 2),
        },
        "rvol": {"value": safe_round(rvol_value, 2), "basis": rvol_basis},
        "bandwidth_pctile_6m": safe_round(bandwidth_pctile, 2),
        "compression": (
            bandwidth_pctile is not None
            and bandwidth_pctile <= SHORT_TERM_COMPRESSION_PCTILE
        ),
    }


def measured_move_target(
    trigger: float | None,
    base_low: float | None,
) -> float | None:
    """Breakout measured move: base depth projected above the level."""
    if trigger is None or base_low is None or trigger <= base_low:
        return None
    return trigger + (trigger - base_low)


def _most_recent_pivot_high(high: pd.Series) -> float | None:
    """Strict-pivot mirror of _most_recent_pivot_low."""
    clean = high.dropna().reset_index(drop=True)
    if len(clean) < 2 * SWING_PIVOT_WINDOW + 1:
        return None
    start = max(SWING_PIVOT_WINDOW, len(clean) - SWING_LOOKBACK)
    pivot: float | None = None
    for i in range(start, len(clean) - SWING_PIVOT_WINDOW):
        window = clean[i - SWING_PIVOT_WINDOW: i + SWING_PIVOT_WINDOW + 1]
        neighbors_max = max(
            window.iloc[:SWING_PIVOT_WINDOW].max(),
            window.iloc[SWING_PIVOT_WINDOW + 1:].max(),
        )
        if clean[i] > neighbors_max:
            pivot = float(clean[i])
    if pivot is None:
        pivot = float(clean.tail(SWING_LOOKBACK).max())
    return pivot


def _clean(value: Any) -> float | None:
    """NaN-safe float conversion."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _bar_date(df: pd.DataFrame) -> date | None:
    raw = str(df["date"].iloc[-1])[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None
