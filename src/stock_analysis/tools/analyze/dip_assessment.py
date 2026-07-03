"""Dip assessment: depth, oversold metrics, bounce potential, entry timing."""

from typing import Any

from stock_analysis.tools.analyze.gates import (
    FALLING_KNIFE_SCORE_THRESHOLD,
    falling_knife_assessment,
)
from stock_analysis.utils.helpers import round_or_none

# Static dip-type / recommendation lookups used by build_dip_assessment.
_DIP_TYPE_EXPLANATIONS: dict[str, str] = {
    "falling_knife": "Severe decline with broken trends - high risk of further downside",
    "extended_decline": "Prolonged weakness but not in freefall",
    "healthy_pullback": "Normal retracement in an uptrend - favorable for dip buying",
    "mixed_signals": "Conflicting trend signals - proceed with caution",
    "undetermined": "Insufficient data to classify",
}

_BUY_RECOMMENDATION_RATIONALES: dict[str, str] = {
    "strong_buy_the_dip": "Oversold pullback in uptrend - high probability bounce",
    "buy_the_dip": "Good entry point in established trend",
    "cautious_accumulation": "Acceptable entry but use smaller position size",
    "small_speculative_position": "High risk but extremely oversold - small position only",
    "do_not_catch_falling_knife": "Trend is broken - wait for stabilization",
    "wait_for_better_setup": "Conditions not favorable for dip buying yet",
}


def _support_status(distance_pct: float | None) -> str | None:
    """Classify a support level by its percent-distance from current price."""
    if distance_pct is None:
        return None
    if distance_pct > 0:
        return "breached" if distance_pct <= 0.01 else "broken"
    if abs(distance_pct) <= 0.01:
        return "tested"
    return "above"


def _build_oversold_composite(
    rsi: float | None,
    return_1w_zscore: float | None,
    distance_to_sma50_atr: float | None,
    position_in_range: float | None,
) -> dict[str, Any]:
    """Build oversold composite score from de-duplicated buckets."""
    oversold_notes: list[str] = []
    rsi_score = 0.0
    if rsi is not None:
        if rsi < 25:
            rsi_score = 2.0
        elif rsi < 30:
            rsi_score = 1.5
        elif rsi < 35:
            rsi_score = 1.0
    elif return_1w_zscore is None:
        oversold_notes.append("momentum_missing")

    z_score = 0.0
    if return_1w_zscore is not None:
        if return_1w_zscore <= -2.0:
            z_score = 1.5
        elif return_1w_zscore <= -1.5:
            z_score = 1.0
        elif return_1w_zscore <= -1.0:
            z_score = 0.5

    momentum_score = max(rsi_score, z_score)

    trend_deviation = 0.0
    if distance_to_sma50_atr is not None:
        if distance_to_sma50_atr <= -2.0:
            trend_deviation = 2.0
        elif distance_to_sma50_atr <= -1.0:
            trend_deviation = 1.0
    else:
        oversold_notes.append("trend_deviation_missing")

    range_position = 0.0
    if position_in_range is not None:
        if position_in_range <= 0.05:
            range_position = 1.0
        elif position_in_range <= 0.15:
            range_position = 0.5
    else:
        oversold_notes.append("range_position_missing")

    optional_band = 0.0

    oversold_raw = momentum_score + trend_deviation + range_position + optional_band
    oversold_composite_score = min(5.0, oversold_raw)
    if oversold_composite_score >= 4.0:
        oversold_composite_level = "extreme"
    elif oversold_composite_score >= 2.0:
        oversold_composite_level = "oversold"
    elif oversold_composite_score >= 1.0:
        oversold_composite_level = "mild"
    else:
        oversold_composite_level = "not_oversold"

    return {
        "score": round(oversold_composite_score, 2),
        "level": oversold_composite_level,
        "components": {
            "momentum": round(momentum_score, 2),
            "trend_deviation": round(trend_deviation, 2),
            "range_position": round(range_position, 2),
            "optional_band": round(optional_band, 2),
        },
        "cap": 5.0,
        "notes": oversold_notes or [],
    }


def _build_dip_depth(
    from_52w_high: float | None,
    from_52w_low: float | None,
    from_3m_high: float | None,
    from_6m_high: float | None,
    days_since_52w_high: int | None,
    days_since_52w_low: int | None,
    risk_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Build dip depth metrics with severity classification.

    Severity bands (based on distance from 52w high):
    - none: >= -2% (within 2% of high)
    - shallow: > -10%
    - moderate: > -25%
    - deep: > -40%
    - extreme: <= -40%
    """
    # Primary basis: from_52w_high
    # Fallback: max_drawdown_1y from risk_data
    dd_data = risk_data.get("drawdown", {})
    max_drawdown_1y = dd_data.get("max_1y")

    # Determine which value to use for severity
    severity_value: float | None = None
    severity_basis: str | None = None

    if from_52w_high is not None:
        severity_value = from_52w_high
        severity_basis = "from_52w_high"
    elif max_drawdown_1y is not None:
        severity_value = max_drawdown_1y
        severity_basis = "max_drawdown_1y"

    # Classify severity
    severity: str
    if severity_value is None:
        severity = "unknown"
    elif severity_value >= -0.02:
        severity = "none"
    elif severity_value > -0.10:
        severity = "shallow"
    elif severity_value > -0.25:
        severity = "moderate"
    elif severity_value > -0.40:
        severity = "deep"
    else:
        severity = "extreme"

    low_set_today = days_since_52w_low == 0 if days_since_52w_low is not None else None
    high_set_today = days_since_52w_high == 0 if days_since_52w_high is not None else None

    return {
        "from_52w_high": from_52w_high,
        "from_52w_low": from_52w_low,
        "from_3m_high": from_3m_high,
        "from_6m_high": from_6m_high,
        "days_since_52w_high": days_since_52w_high,
        "days_since_52w_low": days_since_52w_low,
        "low_set_today": low_set_today,
        "high_set_today": high_set_today,
        "severity": severity,
        "severity_basis": severity_basis,
    }


def _classify_dip(
    bullish_signals: list[str],
    bearish_signals: list[str],
    return_3m: float | None,
    position_in_range: float | None,
    sma_200_slope: float | None,
    days_since_52w_high: int | None,
    sma_50: float | None,
    sma_200: float | None,
    current_price: float | None,
    from_3m_high: float | None,
) -> tuple[str, list[str]]:
    """Classify the current decline as pullback, falling-knife, etc."""
    dip_signals: list[str] = []

    falling_knife_score, knife_reasons = falling_knife_assessment(
        death_cross="death_cross" in bearish_signals,
        below_sma200="price_below_sma200" in bearish_signals,
        return_3m=return_3m,
        position_in_range=position_in_range,
        sma_200_slope=sma_200_slope,
        days_since_52w_high=days_since_52w_high,
    )
    dip_signals.extend(knife_reasons)

    pullback_score = 0
    if "golden_cross" in bullish_signals or (sma_50 and sma_200 and sma_50 > sma_200):
        pullback_score += 2
        dip_signals.append("uptrend_intact")
    if sma_200_slope is not None and sma_200_slope > 0:
        pullback_score += 1
        dip_signals.append("sma200_uptrend")
    if sma_200 and current_price and current_price > sma_200:
        pullback_score += 1
        dip_signals.append("above_sma200")
    if return_3m is not None and return_3m > 0:
        pullback_score += 1
        dip_signals.append("positive_3m_trend")
    if from_3m_high is not None and from_3m_high > -0.15:
        pullback_score += 1
        dip_signals.append("near_3m_high")

    if falling_knife_score >= FALLING_KNIFE_SCORE_THRESHOLD:
        dip_type = "falling_knife"
    elif falling_knife_score >= 2 and pullback_score < 2:
        dip_type = "extended_decline"
    elif pullback_score >= 2 and falling_knife_score <= 1:
        dip_type = "healthy_pullback"
    elif pullback_score >= 1:
        dip_type = "mixed_signals"
    else:
        dip_type = "undetermined"

    return dip_type, dip_signals


def _analyze_volume(
    volume_ratio: float | None,
    return_1w: float | None,
) -> tuple[str, str | None]:
    """Interpret recent volume relative to the 30-day average."""
    if volume_ratio is None:
        return "neutral", None
    if volume_ratio > 2.0:
        if return_1w is not None and return_1w < -0.05:
            return (
                "potential_capitulation",
                "High volume on decline may indicate capitulation selling",
            )
        if return_1w is not None and return_1w > 0.03:
            return (
                "accumulation",
                "High volume on advance suggests institutional buying",
            )
        return "elevated", "Elevated volume - watch for directional confirmation"
    if volume_ratio > 1.5:
        return "above_average", "Above average interest"
    if volume_ratio >= 0.9:
        return "normal", "Normal volume"
    if volume_ratio >= 0.5:
        return "below_average", "Below average interest"
    return "low_conviction", "Low volume - moves may lack conviction"


def _build_support_levels(
    current_price: float | None,
    low_1m: float | None,
    sma_50: float | None,
    sma_200: float | None,
    week_52_low: float | None,
) -> list[dict[str, Any]]:
    """Build candidate support levels with distances and statuses; sorted closest first."""
    price_basis = "close"
    support_levels: list[dict[str, Any]] = []

    if low_1m and current_price:
        distance = (low_1m - current_price) / current_price
        support_levels.append({
            "level": round(low_1m, 2),
            "type": "1m_low",
            "distance_pct": round(distance, 4),
            "strength": "weak",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    if sma_50 and current_price and current_price > sma_50:
        distance = (sma_50 - current_price) / current_price
        support_levels.append({
            "level": round(sma_50, 2),
            "type": "sma_50",
            "distance_pct": round(distance, 4),
            "strength": "medium",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    if sma_200 and current_price:
        distance = (sma_200 - current_price) / current_price
        support_levels.append({
            "level": round(sma_200, 2),
            "type": "sma_200",
            "distance_pct": round(distance, 4),
            "strength": "strong",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    if week_52_low and current_price:
        distance = (week_52_low - current_price) / current_price
        support_levels.append({
            "level": round(week_52_low, 2),
            "type": "52w_low",
            "distance_pct": round(distance, 4),
            "strength": "critical",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    support_levels.sort(key=lambda x: abs(x["distance_pct"]))
    return support_levels


def build_dip_assessment(
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
    fund_data: dict[str, Any],
    market_context: dict[str, Any],
    signals: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Assess whether current price action represents a buying opportunity for dip buyers.

    Analyzes:
    - Dip classification (pullback vs falling knife)
    - Oversold conditions (RSI, distance from MAs)
    - Support levels and bounce potential
    - Volume patterns (capitulation/accumulation)
    - Market context (buying dips in uptrending market vs downtrending)
    """
    ma = tech_data.get("moving_averages", {})
    rsi_data = tech_data.get("rsi", {})
    price_pos = tech_data.get("price_position", {})
    returns = tech_data.get("returns", {})
    volume = tech_data.get("volume", {})
    macd_data = tech_data.get("macd", {})
    price_action = tech_data.get("price_action", {})
    current_price = tech_data.get("current_price")
    atr_data = risk_data.get("atr", {})

    # Extract values
    sma_50 = ma.get("sma_50")
    sma_200 = ma.get("sma_200")
    sma_200_slope = ma.get("sma_200_slope_pct_per_day")
    rsi = rsi_data.get("value")
    rsi_divergence = rsi_data.get("bullish_divergence")
    week_52_low = price_pos.get("week_52_low")
    from_52w_high = price_pos.get("from_52w_high")
    from_52w_low = price_pos.get("from_52w_low")
    from_3m_high = price_pos.get("from_3m_high")
    from_6m_high = price_pos.get("from_6m_high")
    days_since_52w_high = price_pos.get("days_since_52w_high")
    days_since_52w_low = price_pos.get("days_since_52w_low")
    low_1m = price_pos.get("low_1m")
    position_in_range = price_pos.get("position_in_range")
    return_1w = returns.get("return_1w")
    return_1w_zscore = returns.get("return_1w_zscore")
    return_3m = returns.get("return_3m")
    volume_ratio = volume.get("ratio")
    atr_val = atr_data.get("value")
    atr_pct = atr_data.get("as_pct_of_price")

    bullish_signals = signals.get("bullish", [])
    bearish_signals = signals.get("bearish", [])

    # --- DIP CLASSIFICATION ---
    dip_type, dip_signals = _classify_dip(
        bullish_signals=bullish_signals,
        bearish_signals=bearish_signals,
        return_3m=return_3m,
        position_in_range=position_in_range,
        sma_200_slope=sma_200_slope,
        days_since_52w_high=days_since_52w_high,
        sma_50=sma_50,
        sma_200=sma_200,
        current_price=current_price,
        from_3m_high=from_3m_high,
    )

    # --- OVERSOLD METRICS ---
    oversold_indicators: list[str] = []
    oversold_score = 0

    # RSI oversold
    rsi_status: str = "neutral"
    if rsi is not None:
        if rsi < 30:
            rsi_status = "oversold"
            oversold_score += 2
            oversold_indicators.append(f"rsi_oversold_{rsi:.0f}")
        elif rsi < 40:
            rsi_status = "approaching_oversold"
            oversold_score += 1
            oversold_indicators.append(f"rsi_low_{rsi:.0f}")
        elif rsi > 70:
            rsi_status = "overbought"
            oversold_indicators.append(f"rsi_overbought_{rsi:.0f}")

    # Distance from moving averages (mean reversion potential)
    price_vs_sma20 = ma.get("price_vs_sma20")
    price_vs_sma50 = ma.get("price_vs_sma50")
    price_vs_sma200 = ma.get("price_vs_sma200")

    distance_to_sma50_atr: float | None = None
    if atr_val and sma_50 and current_price:
        distance_to_sma50_atr = (current_price - sma_50) / atr_val

    if price_vs_sma20 is not None and price_vs_sma20 < -0.10:
        oversold_score += 1
        oversold_indicators.append(f"extended_below_sma20_{price_vs_sma20:.1%}")
    if price_vs_sma50 is not None and price_vs_sma50 < -0.15:
        oversold_score += 1
        oversold_indicators.append(f"extended_below_sma50_{price_vs_sma50:.1%}")
    if price_vs_sma200 is not None and price_vs_sma200 < -0.20:
        oversold_score += 1
        oversold_indicators.append(f"extended_below_sma200_{price_vs_sma200:.1%}")
    if distance_to_sma50_atr is not None and distance_to_sma50_atr < -2.0:
        oversold_score += 1
        oversold_indicators.append(f"below_sma50_{abs(distance_to_sma50_atr):.1f}atr")

    # Position in 52-week range
    if position_in_range is not None and position_in_range < 0.20:
        oversold_score += 1
        oversold_indicators.append("bottom_20pct_of_range")

    if return_1w_zscore is not None and return_1w_zscore <= -1.5:
        oversold_score += 1
        oversold_indicators.append(f"1w_return_zscore_{return_1w_zscore:.1f}")

    oversold_level: str
    if oversold_score >= 4:
        oversold_level = "extremely_oversold"
    elif oversold_score >= 2:
        oversold_level = "oversold"
    elif oversold_score >= 1:
        oversold_level = "mildly_oversold"
    else:
        oversold_level = "not_oversold"

    # Oversold composite (de-dup correlated signals)
    oversold_composite = _build_oversold_composite(
        rsi,
        return_1w_zscore,
        distance_to_sma50_atr,
        position_in_range,
    )
    oversold_composite_level = oversold_composite["level"]

    # --- SUPPORT LEVELS ---
    support_levels = _build_support_levels(
        current_price=current_price,
        low_1m=low_1m,
        sma_50=sma_50,
        sma_200=sma_200,
        week_52_low=week_52_low,
    )

    # --- VOLUME ANALYSIS ---
    volume_signal, volume_interpretation = _analyze_volume(volume_ratio, return_1w)

    # --- BOUNCE POTENTIAL ---
    # Calculate likelihood of a bounce based on combined factors
    bounce_score = 0
    bounce_factors: list[str] = []

    # Oversold conditions favor bounce
    if oversold_composite_level == "extreme":
        bounce_score += 2
        bounce_factors.append("oversold_composite_extreme")
    elif oversold_composite_level == "oversold":
        bounce_score += 1
        bounce_factors.append("oversold_composite_oversold")

    # Healthy pullback in uptrend
    if dip_type == "healthy_pullback":
        bounce_score += 2
        bounce_factors.append("pullback_in_uptrend")
    elif dip_type == "falling_knife":
        bounce_score -= 2
        bounce_factors.append("falling_knife_risk")

    # Near support
    if (
        support_levels
        and abs(support_levels[0]["distance_pct"]) < 0.03
        and support_levels[0]["strength"] in ("strong", "critical")
    ):
        bounce_score += 1
        bounce_factors.append(f"near_{support_levels[0]['type']}")

    # Capitulation volume
    if volume_signal == "potential_capitulation":
        bounce_score += 1
        bounce_factors.append("capitulation_volume")

    # Market context - buying dips in bull market more favorable
    spy_bullish = market_context.get("spy_above_200d", False)
    if spy_bullish:
        bounce_score += 1
        bounce_factors.append("bullish_market_context")
    else:
        bounce_score -= 1
        bounce_factors.append("bearish_market_context")

    bounce_potential: str
    if bounce_score >= 4:
        bounce_potential = "high"
    elif bounce_score >= 2:
        bounce_potential = "moderate"
    elif bounce_score >= 0:
        bounce_potential = "low"
    else:
        bounce_potential = "very_low"

    # --- ENTRY TIMING ---
    entry_signals: list[dict[str, str]] = []
    wait_for: list[str] = []

    # Immediate entry signals
    if rsi_status == "oversold" and dip_type != "falling_knife":
        entry_signals.append({
            "signal": "rsi_oversold",
            "action": "consider_small_position",
            "rationale": "RSI indicates oversold but confirm with price action",
        })

    if rsi_divergence is True:
        entry_signals.append({
            "signal": "rsi_bullish_divergence",
            "action": "starter_position",
            "rationale": "Price made lower low while RSI improved",
        })

    if volume_signal == "potential_capitulation":
        entry_signals.append({
            "signal": "capitulation_volume",
            "action": "watch_for_reversal",
            "rationale": "High volume selling may exhaust sellers",
        })

    if dip_type == "healthy_pullback" and oversold_level in ("oversold", "mildly_oversold"):
        entry_signals.append({
            "signal": "pullback_in_uptrend",
            "action": "accumulate",
            "rationale": "Healthy pullback to buy in established uptrend",
        })

    if price_action.get("higher_closes_2d") is True:
        entry_signals.append({
            "signal": "two_higher_closes",
            "action": "starter_position",
            "rationale": "Short-term momentum stabilizing",
        })

    if price_action.get("break_5d_high") is True:
        entry_signals.append({
            "signal": "break_5d_high",
            "action": "add_position",
            "rationale": "Price reclaimed recent high",
        })

    if price_vs_sma20 is not None and price_vs_sma20 > 0:
        entry_signals.append({
            "signal": "close_above_sma20",
            "action": "add_position",
            "rationale": "Reclaimed short-term trend",
        })

    if macd_data.get("histogram_rising_3d") is True:
        entry_signals.append({
            "signal": "macd_histogram_rising",
            "action": "add_position",
            "rationale": "Momentum improving for 3 sessions",
        })

    # Wait signals
    if dip_type == "falling_knife":
        wait_for.append("price_stabilization_above_support")
        if rsi_divergence is not True:
            wait_for.append("rsi_bullish_divergence")
        wait_for.append("volume_dry_up")

    if sma_200 and current_price and current_price < sma_200:
        wait_for.append(f"reclaim_sma200_at_{sma_200:.2f}")

    if rsi is not None and rsi > 50 and dip_type != "healthy_pullback":
        wait_for.append("deeper_pullback_or_rsi_oversold")

    if price_action.get("higher_closes_2d") is not True:
        wait_for.append("two_higher_closes")
    if price_action.get("break_5d_high") is not True:
        wait_for.append("break_5d_high")
    if price_vs_sma20 is not None and price_vs_sma20 <= 0:
        wait_for.append("close_above_sma20")
    if macd_data.get("histogram_rising_3d") is not True:
        wait_for.append("macd_histogram_rising_3d")

    # --- DIP CONFIDENCE ---
    confidence_score = 0
    missing_metrics: list[str] = []

    def _score_metric(value: Any, name: str) -> None:
        nonlocal confidence_score
        if value is None:
            missing_metrics.append(name)
        else:
            confidence_score += 1

    _score_metric(rsi, "rsi")
    _score_metric(price_vs_sma200, "price_vs_sma200")
    _score_metric(return_3m, "return_3m")
    _score_metric(volume_ratio, "volume_ratio")
    _score_metric(atr_pct, "atr_pct")

    if dip_type in ("healthy_pullback", "falling_knife"):
        confidence_score += 1
    elif dip_type == "mixed_signals":
        confidence_score -= 1

    if oversold_level in ("oversold", "extremely_oversold"):
        confidence_score += 1
    elif oversold_level == "not_oversold":
        confidence_score -= 1

    if confidence_score >= 5:
        confidence_level = "high"
    elif confidence_score >= 3:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # --- OVERALL ASSESSMENT ---
    # Combine all factors for buy-the-dip recommendation
    dip_quality: str
    if dip_type == "healthy_pullback" and oversold_level in ("oversold", "extremely_oversold"):
        dip_quality = "excellent"
    elif dip_type == "healthy_pullback" and oversold_level == "mildly_oversold":
        dip_quality = "good"
    elif dip_type == "mixed_signals" and oversold_level in ("oversold", "extremely_oversold"):
        dip_quality = "fair"
    elif dip_type == "extended_decline" and oversold_level == "extremely_oversold":
        dip_quality = "speculative"
    elif dip_type == "falling_knife":
        dip_quality = "avoid"
    else:
        dip_quality = "wait"

    # Buy recommendation
    buy_recommendation: str
    if dip_quality == "excellent":
        buy_recommendation = "strong_buy_the_dip"
    elif dip_quality == "good":
        buy_recommendation = "buy_the_dip"
    elif dip_quality == "fair":
        buy_recommendation = "cautious_accumulation"
    elif dip_quality == "speculative":
        buy_recommendation = "small_speculative_position"
    elif dip_quality == "avoid":
        buy_recommendation = "do_not_catch_falling_knife"
    else:
        buy_recommendation = "wait_for_better_setup"

    return {
        "dip_classification": {
            "type": dip_type,
            "signals": dip_signals,
            "explanation": _DIP_TYPE_EXPLANATIONS.get(dip_type, ""),
        },
        "dip_depth": _build_dip_depth(
            from_52w_high,
            from_52w_low,
            from_3m_high,
            from_6m_high,
            days_since_52w_high,
            days_since_52w_low,
            risk_data,
        ),
        "oversold_metrics": {
            "level": oversold_level,
            "score": oversold_score,
            "rsi_status": rsi_status,
            "rsi_value": rsi,
            "indicators": oversold_indicators,
            "distance_from_sma20": price_vs_sma20,
            "distance_from_sma50": price_vs_sma50,
            "distance_from_sma200": price_vs_sma200,
            "distance_from_sma50_atr": round_or_none(distance_to_sma50_atr, 2),
            "return_1w_zscore": return_1w_zscore,
            "sma200_slope_pct_per_day": sma_200_slope,
            "position_in_52w_range": position_in_range,
            "oversold_composite": oversold_composite,
        },
        "support_levels": support_levels[:4],  # Top 4 closest supports
        "volume_analysis": {
            "signal": volume_signal,
            "ratio": volume_ratio,
            "interpretation": volume_interpretation,
        },
        "bounce_potential": {
            "rating": bounce_potential,
            "score": bounce_score,
            "factors": bounce_factors,
        },
        "entry_timing": {
            "signals": entry_signals,
            "wait_for": wait_for,
        },
        "dip_confidence": {
            "level": confidence_level,
            "score": confidence_score,
            "missing": missing_metrics or [],
        },
        "assessment": {
            "dip_quality": dip_quality,
            "recommendation": buy_recommendation,
            "rationale": _BUY_RECOMMENDATION_RATIONALES.get(buy_recommendation, ""),
        },
        "method": "dip_assessment_v2",
    }


def align_dip_assessment_with_action(
    dip_assessment: dict[str, Any] | None,
    mismatch_status: str | None,
    can_start_now: bool,
    add_only_if: list[str] | None = None,
) -> dict[str, Any] | None:
    """Reframe dip guidance when a dislocation starter is acceptable now.

    The dip block is an entry-timing lens, not the master action decision.
    If the broader analysis says a small starter is acceptable now, avoid
    rendering a flat contradiction like "do not catch falling knife".
    """
    if not dip_assessment:
        return dip_assessment
    if mismatch_status != "price_broken_more_than_business" or not can_start_now:
        return dip_assessment

    assessment = dip_assessment.get("assessment")
    if not isinstance(assessment, dict):
        return dip_assessment

    recommendation = assessment.get("recommendation")
    if recommendation not in {
        "do_not_catch_falling_knife",
        "wait_for_better_setup",
    }:
        assessment["scope"] = "entry_timing_only"
        assessment["timing_only"] = True
        return dip_assessment

    assessment["scope"] = "entry_timing_only"
    assessment["timing_only"] = True
    assessment["dip_quality"] = "starter_only"
    assessment["recommendation"] = "starter_only_wait_for_stabilization"
    assessment["rationale"] = (
        "Starter position acceptable for a price-vs-business dislocation, "
        "but trend is still broken. Wait for stabilization before adding."
    )
    assessment["add_only_if"] = list(add_only_if or [])[:4]
    return dip_assessment
