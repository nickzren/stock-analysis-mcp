"""Phase-1 scan screen: one daily fetch worth of data, the SHARED detectors.

No-false-negative contract (spec amendment): the payload carries every
detection and gate input the full card's detectors read. The only
approximation is rsi.bullish_divergence=False, which feeds quality grading
only and cannot affect promotion.
"""

from __future__ import annotations

import operator
from typing import Any

import pandas as pd

from stock_analysis.tools.analyze.gates import (
    check_liquidity,
    is_falling_knife_technicals,
)
from stock_analysis.tools.trade_setup.setups import detect_setup
from stock_analysis.utils.helpers import safe_last_float, safe_round
from stock_analysis.utils.indicators import (
    calculate_atr,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
    calculate_sma_slope,
    days_since_extreme,
    weekly_return_zscore,
)
from stock_analysis.utils.swing_features import compute_setup_features
from stock_analysis.utils.validators import check_rule_expr

MIN_SCREEN_BARS = 21  # matches compute_setup_features' floor


def build_screen_payload(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or len(df) < MIN_SCREEN_BARS:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    current_price = safe_last_float(close)
    sma_20 = safe_last_float(calculate_sma(close, 20))
    sma_50 = safe_last_float(calculate_sma(close, 50))
    sma_200_series = calculate_sma(close, 200)
    sma_200 = safe_last_float(sma_200_series)

    week_52_high = float(high.max()) if not high.isna().all() else None
    week_52_low = float(low.min()) if not low.isna().all() else None
    position_in_range = (
        (current_price - week_52_low) / (week_52_high - week_52_low)
        if current_price and week_52_high and week_52_low and week_52_high != week_52_low
        else None
    )

    current_volume = safe_last_float(volume)
    avg_volume_20d = float(volume.tail(20).mean()) if len(volume) >= 20 else None
    volume_ratio = (
        current_volume / avg_volume_20d
        if current_volume is not None and avg_volume_20d
        else None
    )

    return {
        "current_price": safe_round(current_price, 2),
        "moving_averages": {
            "sma_20": safe_round(sma_20, 2),
            "sma_50": safe_round(sma_50, 2),
            "sma_200": safe_round(sma_200, 2),
            "sma_200_slope_pct_per_day": safe_round(
                calculate_sma_slope(sma_200_series, slope_window=20), 6
            ),
            "rules": {
                "above_sma20": {"triggered": check_rule_expr(current_price, sma_20, operator.gt)},
                "above_sma50": {"triggered": check_rule_expr(current_price, sma_50, operator.gt)},
                "above_sma200": {"triggered": check_rule_expr(current_price, sma_200, operator.gt)},
                "golden_cross": {"triggered": check_rule_expr(sma_50, sma_200, operator.gt)},
                "death_cross": {"triggered": check_rule_expr(sma_50, sma_200, operator.lt)},
            },
        },
        "rsi": {
            "value": safe_round(safe_last_float(calculate_rsi(close, 14)), 1),
            "bullish_divergence": False,  # quality-only; cannot affect promotion
        },
        "atr": {"value": safe_round(safe_last_float(calculate_atr(high, low, close, 14)), 2)},
        "returns": {
            "return_3m": safe_round(calculate_returns(close, 63), 4),
            "return_1w_zscore": safe_round(weekly_return_zscore(close), 2),  # amendment
        },
        "price_position": {
            "week_52_high": safe_round(week_52_high, 2),
            "week_52_low": safe_round(week_52_low, 2),
            "position_in_range": safe_round(position_in_range, 4),
            "days_since_52w_high": days_since_extreme(high, kind="high"),
        },
        "volume": {"ratio": safe_round(volume_ratio, 2)},
    }


def screen_symbol(df: pd.DataFrame | None) -> dict[str, Any]:
    """Phase-1 verdict from one daily frame, using the shared detectors/gates."""
    payload = build_screen_payload(df)
    features = compute_setup_features(df)
    if payload is None or features is None or payload["current_price"] is None:
        return {
            "promote": False, "action_hint": "wait_for_data", "setup_type": None,
            "trigger_price": None, "last_close": None,
            "blocker_ids": ["data_quality_critical"],
        }

    blocker_ids: list[str] = []
    knife, _ = is_falling_knife_technicals(payload)
    if knife:
        blocker_ids.append("falling_knife")

    dollar_volume = _avg_dollar_volume(df)
    weak, missing, liq_blockers = check_liquidity(
        {"liquidity": {"avg_dollar_volume": dollar_volume}}
    )
    blocker_ids.extend(b["id"] for b in liq_blockers)

    if knife or weak or missing:
        return {
            "promote": False, "action_hint": "avoid", "setup_type": None,
            "trigger_price": None, "last_close": payload["current_price"],
            "blocker_ids": blocker_ids,
        }

    setup = detect_setup(payload, features, float(payload["current_price"]))
    if setup is None:
        return {
            "promote": False, "action_hint": "no_setup", "setup_type": None,
            "trigger_price": None, "last_close": payload["current_price"],
            "blocker_ids": [],
        }
    return {
        "promote": True, "action_hint": "candidate",
        "setup_type": setup["type"], "trigger_price": setup["trigger_price"],
        "last_close": payload["current_price"], "blocker_ids": [],
    }


def _avg_dollar_volume(df: pd.DataFrame | None) -> float | None:
    """Direct per-bar dollar average over the prior 20 bars (owner note)."""
    if df is None:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    window = (close * volume).iloc[-21:-1]
    if window.isna().all():
        return None
    value = float(window.mean())
    return None if pd.isna(value) else value
