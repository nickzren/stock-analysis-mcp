"""Phase-1 scan screen: one daily fetch worth of data, the SHARED detectors.

No-false-negative contract (spec amendment): the payload carries every
detection and gate input the full card's detectors read. The only
approximation is rsi.bullish_divergence=False, which feeds quality grading
only and cannot affect promotion. `volume.ratio` is included for payload
parity with technicals, but detectors read `features.last_volume_ratio`
(from compute_setup_features), not this field.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Any

import pandas as pd

from stock_analysis.utils.freshness import (
    FRESHNESS_GATE_IDS,
    daily_leg_freshness,
    freshness_blockers,
)
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
from stock_analysis.utils.liquidity import RISK_METRICS_MIN_BARS, mean_dollar_volume
from stock_analysis.utils.swing_features import compute_setup_features, daily_actionable_price
from stock_analysis.utils.validators import check_rule_expr
from stock_analysis.utils.valuation_basis import (
    classify_support,
    sizing_unavailable_reason,
)

MIN_SCREEN_BARS = 21  # the payload floor; matches compute_setup_features' floor
# Freshness gates carry the leg they fire or go unevaluated on ("stale_data@daily").
QUOTE_LEG = frozenset(f"{gate_id}@quote" for gate_id in FRESHNESS_GATE_IDS)
DAILY_LEG = frozenset(f"{gate_id}@daily" for gate_id in FRESHNESS_GATE_IDS)


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
    window_mean = volume.tail(20).mean()
    avg_volume_20d = float(window_mean) if len(volume) >= 20 and pd.notna(window_mean) else None
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


def screen_symbol(df: pd.DataFrame | None, *, now: datetime, session: str) -> dict[str, Any]:
    """Stage 1: evaluate every gate the daily frame allows, then resolve rank-first like the card (D-11)."""
    # Lazy: tools.* imports here would close a cycle via
    # tools/__init__ -> watchlist -> scan_screen.
    from stock_analysis.tools.analyze.gates import (
        SCREEN_UNEVALUATED,
        check_liquidity,
        is_falling_knife_technicals,
    )
    from stock_analysis.tools.trade_setup.setups import detect_setup

    payload = build_screen_payload(df)
    features = compute_setup_features(df)
    blockers: set[str] = set()
    unevaluated = set(SCREEN_UNEVALUATED) | (QUOTE_LEG if session == "regular" else set())

    # Data: the card's floor (off-session the card prices off the daily close) and its daily freshness leg.
    if (
        df is None or len(df) < RISK_METRICS_MIN_BARS or features is None
        or (session != "regular" and daily_actionable_price(features) is None)
    ):
        blockers.add("data_quality_critical")
    freshness = daily_leg_freshness(df, session=session, now=now)
    blockers |= {f"{b['id']}@daily" for b in freshness_blockers(freshness)}
    data_veto = bool(blockers)

    knife = False
    if payload is None:
        unevaluated.add("falling_knife")
    else:
        knife, _ = is_falling_knife_technicals(payload)
        if knife:
            blockers.add("falling_knife")

    # WP4 (D5): instrument and currency come with the daily history. Unsupported caps;
    # unknown is unverified, so stage 1 leaves it to the card.
    attrs = df.attrs if df is not None else {}
    support = classify_support(
        quote_type=attrs.get("instrument_type"), currency=attrs.get("currency"), symbol=attrs.get("symbol")
    )
    capped = False
    for gate_id, dimension in (("unsupported_instrument", "instrument"), ("unsupported_currency", "currency")):
        status = support[dimension]["status"]
        if status == "unsupported":
            blockers.add(gate_id)
            capped = True
        elif status == "unverified":
            unevaluated.add(gate_id)

    # Liquidity on the card's basis, and only for supported names (a non-USD price makes no dollar claim).
    weak = missing = False
    if sizing_unavailable_reason(support) is None:
        weak, missing, liq_blockers = check_liquidity(
            {"liquidity": {"avg_dollar_volume": _card_liquidity(df)}}
        )
        blockers |= {b["id"] for b in liq_blockers}

    setup: dict[str, Any] | None = None
    if data_veto:
        action = "wait_for_data"
    elif knife or weak or missing:
        action = "avoid"
    else:
        assert payload is not None and features is not None
        setup = detect_setup(payload, features, float(payload["current_price"]))
        action = "no_setup" if setup is None else ("watch" if capped else "candidate")
    return {
        "promote": action == "candidate", "action_hint": action,
        "setup_type": setup["type"] if setup else None,
        "trigger_price": setup["trigger_price"] if setup else None,
        "trigger_satisfied": bool(setup["trigger_satisfied"]) if setup else False,
        "last_close": payload["current_price"] if payload else None,
        "blocker_ids": sorted(blockers), "unevaluated": sorted(unevaluated),
    }


def _card_liquidity(df: pd.DataFrame | None) -> float | None:
    """risk_metrics' liquidity.avg_dollar_volume on the same frame; None under its bar floor."""
    if df is None or len(df) < RISK_METRICS_MIN_BARS:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    return safe_round(mean_dollar_volume(volume, safe_last_float(close)), 2)
