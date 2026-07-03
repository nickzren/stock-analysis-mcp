"""Swing setup detection: qualifier + trigger + stop + primary target per setup."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stock_analysis.tools.trade_setup.setup_rules import (
    BREAKOUT_BANDWIDTH_PCTILE_MAX,
    BREAKOUT_MAX_BELOW_HIGH,
    BREAKOUT_STOP_LEVEL_BUFFER,
    BREAKOUT_VOLUME_RATIO_MIN,
    MEANREV_RSI_MAX,
    MEANREV_STOP_ATR_MULT,
    MEANREV_SUPPORT_SMA200_FACTOR,
    MEANREV_ZSCORE_MAX,
    PULLBACK_MAX_OFF_HIGH,
    PULLBACK_MIN_OFF_HIGH,
    PULLBACK_RSI_MAX,
    PULLBACK_RSI_MIN,
    SETUP_PRIORITY,
    STOP_ATR_MULT,
)

_QUALITY_BY_FACTORS = {3: "A", 2: "B"}


def detect_setup(
    technicals_data: dict[str, Any],
    features: dict[str, Any],
    actionable_price: float,
) -> dict[str, Any] | None:
    """Return the highest-priority qualifying setup, or None."""
    detectors: dict[str, Callable[..., dict[str, Any] | None]] = {
        "pullback_in_uptrend": _detect_pullback,
        "breakout": _detect_breakout,
        "oversold_mean_reversion": _detect_mean_reversion,
    }
    for name in SETUP_PRIORITY:
        setup = detectors[name](technicals_data, features, actionable_price)
        if setup is not None:
            return setup
    return None


def _grade(factors: list[bool]) -> str:
    return _QUALITY_BY_FACTORS.get(sum(1 for f in factors if f), "C")


def _rule(technicals_data: dict[str, Any], name: str) -> bool | None:
    rules = (technicals_data.get("moving_averages") or {}).get("rules") or {}
    return (rules.get(name) or {}).get("triggered")


def _atr(technicals_data: dict[str, Any]) -> float | None:
    return (technicals_data.get("atr") or {}).get("value")


def _pct_off_high(actionable_price: float, high_20d_prior: float | None) -> float | None:
    if high_20d_prior is None or high_20d_prior <= 0:
        return None
    return (high_20d_prior - actionable_price) / high_20d_prior


def _detect_pullback(
    technicals_data: dict[str, Any],
    features: dict[str, Any],
    actionable_price: float,
) -> dict[str, Any] | None:
    ma = technicals_data.get("moving_averages") or {}
    rsi_value = (technicals_data.get("rsi") or {}).get("value")
    slope = ma.get("sma_200_slope_pct_per_day")
    off_high = _pct_off_high(actionable_price, features.get("high_20d_prior"))

    uptrend = (
        _rule(technicals_data, "above_sma200") is True
        and _rule(technicals_data, "golden_cross") is True
        and slope is not None
        and slope >= 0
    )
    in_band = (
        off_high is not None
        and PULLBACK_MIN_OFF_HIGH <= off_high <= PULLBACK_MAX_OFF_HIGH
    )
    rsi_ok = rsi_value is not None and PULLBACK_RSI_MIN <= rsi_value <= PULLBACK_RSI_MAX
    if not (uptrend and in_band and rsi_ok):
        return None

    trigger_price = features.get("prior_day_high")
    if trigger_price is None:
        return None

    swing_low = features.get("swing_low")
    atr = _atr(technicals_data)
    if swing_low is not None and swing_low < trigger_price:
        stop_price, stop_basis = float(swing_low), "swing_low"
    elif atr is not None:
        stop_price, stop_basis = trigger_price - STOP_ATR_MULT * atr, "atr"
    else:
        return None
    if stop_price >= trigger_price:
        return None

    high_20d = features.get("high_20d_prior")
    target_primary = (
        {"price": float(high_20d), "basis": "prior_20d_high"}
        if high_20d is not None and high_20d > trigger_price
        else None
    )

    quality = _grade([
        slope is not None and slope > 0,
        rsi_value is not None and 40.0 <= rsi_value <= 50.0,
        _rule(technicals_data, "above_sma50") is True,
    ])
    return {
        "type": "pullback_in_uptrend",
        "quality": quality,
        "thesis": [
            "Uptrend intact (price > SMA200, SMA50 > SMA200)",
            f"Pulled back {off_high:.1%} from 20d high with RSI {rsi_value:.0f}",
        ],
        "invalidation": [
            "Close below the pullback swing low",
            "SMA50 crossing below SMA200",
        ],
        "trigger_price": round(float(trigger_price), 2),
        "trigger_satisfied": actionable_price > trigger_price,
        "trigger_condition": f"price reclaims prior-day high {trigger_price:.2f}",
        "stop_price": round(float(stop_price), 2),
        "stop_basis": stop_basis,
        "target_primary": target_primary,
    }


def _detect_breakout(
    technicals_data: dict[str, Any],
    features: dict[str, Any],
    actionable_price: float,
) -> dict[str, Any] | None:
    high_20d = features.get("high_20d_prior")
    off_high = _pct_off_high(actionable_price, high_20d)
    bandwidth_pctile = features.get("bandwidth_pctile_6m")
    volume_ratio = features.get("last_volume_ratio")

    near_high = off_high is not None and off_high <= BREAKOUT_MAX_BELOW_HIGH
    compressed = (
        bandwidth_pctile is not None and bandwidth_pctile <= BREAKOUT_BANDWIDTH_PCTILE_MAX
    )
    if not (near_high and compressed and high_20d is not None):
        return None

    trigger_price = float(high_20d)
    stop_price = round(trigger_price * (1 - BREAKOUT_STOP_LEVEL_BUFFER), 2)
    if stop_price >= trigger_price:
        atr = _atr(technicals_data)
        if atr is None:
            return None
        stop_price = round(trigger_price - STOP_ATR_MULT * atr, 2)
        if stop_price >= trigger_price:
            return None

    volume_ok = volume_ratio is not None and volume_ratio >= BREAKOUT_VOLUME_RATIO_MIN
    quality = _grade([
        bandwidth_pctile is not None and bandwidth_pctile <= 0.10,
        volume_ratio is not None and volume_ratio >= 2.0,
        _rule(technicals_data, "above_sma200") is True
        and _rule(technicals_data, "golden_cross") is True,
    ])
    return {
        "type": "breakout",
        "quality": quality,
        "thesis": [
            f"Within {BREAKOUT_MAX_BELOW_HIGH:.0%} of the 20d high after volatility compression",
            f"Bollinger bandwidth in the {bandwidth_pctile:.0%} percentile of its 6-month range",
        ],
        "invalidation": [
            "Close back below the breakout level",
            "Breakout on below-average volume",
        ],
        "trigger_price": round(trigger_price, 2),
        "trigger_satisfied": bool(actionable_price > trigger_price and volume_ok),
        "trigger_condition": (
            f"price clears 20d high {trigger_price:.2f} with volume ratio >= "
            f"{BREAKOUT_VOLUME_RATIO_MIN}"
        ),
        "stop_price": stop_price,
        "stop_basis": "below_breakout_level",
        "target_primary": None,  # blue sky above a 20d-high break: 1R fallback
    }


def _detect_mean_reversion(
    technicals_data: dict[str, Any],
    features: dict[str, Any],
    actionable_price: float,
) -> dict[str, Any] | None:
    rsi = technicals_data.get("rsi") or {}
    rsi_value = rsi.get("value")
    zscore = (technicals_data.get("returns") or {}).get("return_1w_zscore")
    sma_200 = (technicals_data.get("moving_averages") or {}).get("sma_200")

    oversold = (rsi_value is not None and rsi_value < MEANREV_RSI_MAX) or (
        zscore is not None and zscore < MEANREV_ZSCORE_MAX
    )
    above_support = (
        sma_200 is not None
        and actionable_price >= MEANREV_SUPPORT_SMA200_FACTOR * sma_200
    )
    if not (oversold and above_support):
        return None

    trigger_price = features.get("prior_day_high")
    atr = _atr(technicals_data)
    if trigger_price is None or atr is None:
        return None
    stop_price = round(float(trigger_price) - MEANREV_STOP_ATR_MULT * atr, 2)
    if stop_price >= trigger_price:
        return None

    sma_20 = (technicals_data.get("moving_averages") or {}).get("sma_20")
    target_primary = (
        {"price": float(sma_20), "basis": "sma_20"}
        if sma_20 is not None and sma_20 > trigger_price
        else None
    )

    pos_in_range = (technicals_data.get("price_position") or {}).get("position_in_range")
    quality = _grade([
        rsi.get("bullish_divergence") is True,
        zscore is not None and zscore < -2.5,
        pos_in_range is not None and pos_in_range > 0.2,
    ])
    rsi_text = f"RSI {rsi_value:.0f}" if rsi_value is not None else "1w return z-score extreme"
    return {
        "type": "oversold_mean_reversion",
        "quality": quality,
        "thesis": [
            f"Oversold ({rsi_text}) above long-term support",
            "Reversion target at the 20-day mean",
        ],
        "invalidation": [
            "Close below the ATR stop",
            f"Loss of long-term support (price < {MEANREV_SUPPORT_SMA200_FACTOR:.0%} of SMA200)",
        ],
        "trigger_price": round(float(trigger_price), 2),
        "trigger_satisfied": actionable_price > trigger_price,
        "trigger_condition": f"first close above prior-day high {trigger_price:.2f}",
        "stop_price": stop_price,
        "stop_basis": "atr",
        "target_primary": target_primary,
    }
