"""Shared hard-gate checks used by the decision card and the trade-setup card."""

from __future__ import annotations

from typing import Any

from stock_analysis.tools.analyze.constants import CRITICAL_ANALYZE_TOOLS

# Liquidity threshold below which we block new buys for small accounts.
WEAK_LIQUIDITY_DOLLARS_PER_DAY = 1_000_000.0

# Days before earnings during which entries are blocked.
EARNINGS_BLACKOUT_DAYS = 5

# Falling-knife rubric threshold (same rubric as dip assessment).
FALLING_KNIFE_SCORE_THRESHOLD = 4


def check_earnings_blackout(events_data: dict[str, Any]) -> tuple[bool, dict[str, str] | None]:
    earnings = events_data.get("earnings") or {}
    days_until = earnings.get("days_until")
    fired = (
        isinstance(days_until, (int, float))
        and 0 <= days_until <= EARNINGS_BLACKOUT_DAYS
    )
    if not fired or not isinstance(days_until, (int, float)):
        return False, None
    return True, {
        "id": "earnings_blackout",
        "reason": f"earnings in {int(days_until)} days (<= {EARNINGS_BLACKOUT_DAYS})",
    }


def check_liquidity(risk_data: dict[str, Any]) -> tuple[bool, bool, list[dict[str, str]]]:
    liquidity = risk_data.get("liquidity") or {}
    avg_dollar_volume = liquidity.get("avg_dollar_volume")
    weak = (
        isinstance(avg_dollar_volume, (int, float))
        and avg_dollar_volume < WEAK_LIQUIDITY_DOLLARS_PER_DAY
    )
    missing = avg_dollar_volume is None
    blockers: list[dict[str, str]] = []
    if weak and isinstance(avg_dollar_volume, (int, float)):
        blockers.append({
            "id": "weak_liquidity",
            "reason": (
                f"avg daily dollar volume "
                f"~${avg_dollar_volume / 1_000_000:.2f}M < "
                f"${WEAK_LIQUIDITY_DOLLARS_PER_DAY / 1_000_000:.0f}M threshold"
            ),
        })
    if missing:
        blockers.append({
            "id": "liquidity_missing",
            "reason": "avg_dollar_volume unavailable — cannot confirm tradability for small account",
        })
    return weak, missing, blockers


def check_data_quality(
    data_quality: dict[str, Any],
    critical_tools: frozenset[str] = CRITICAL_ANALYZE_TOOLS,
) -> tuple[bool, dict[str, str] | None]:
    fund_status = data_quality.get("fundamentals_status")
    missing_critical = data_quality.get("missing_critical") or []
    tool_failures = data_quality.get("tool_failures") or []
    critical_tool_failed = any(
        tf.get("tool") in critical_tools for tf in tool_failures if isinstance(tf, dict)
    )
    fired = fund_status == "missing" or bool(missing_critical) or critical_tool_failed
    if not fired:
        return False, None
    reasons: list[str] = []
    if fund_status == "missing":
        reasons.append("fundamentals_status=missing")
    if missing_critical:
        reasons.append(f"missing_critical={','.join(str(x) for x in missing_critical)}")
    if critical_tool_failed:
        failed_names = sorted({
            str(tf.get("tool"))
            for tf in tool_failures
            if isinstance(tf, dict) and tf.get("tool") in critical_tools
        })
        reasons.append(f"critical_tool_failed={','.join(failed_names)}")
    return True, {
        "id": "data_quality_critical",
        "reason": "critical data unavailable: " + "; ".join(reasons),
    }


def falling_knife_assessment(
    *,
    death_cross: bool | None,
    below_sma200: bool | None,
    return_3m: float | None,
    position_in_range: float | None,
    sma_200_slope: float | None,
    days_since_52w_high: int | None,
) -> tuple[int, list[str]]:
    """Score falling-knife conditions. Identical rubric to dip assessment.

    A knife requires evidence of an actual decline: price below SMA200, or a
    3-month return past the significant-decline tier (< -20%). Structure-lag
    artifacts alone (persistent death cross, stale 52w high, a barely-negative
    long SMA slope) must not qualify a recovering chart.
    """
    declining = below_sma200 is True or (
        return_3m is not None and return_3m < -0.20
    )
    if not declining:
        return 0, []
    score = 0
    reasons: list[str] = []
    if death_cross:
        score += 2
        reasons.append("death_cross_active")
    if below_sma200:
        score += 1
        reasons.append("below_sma200")
    if return_3m is not None and return_3m < -0.30:
        score += 2
        reasons.append("severe_3m_decline")
    elif return_3m is not None and return_3m < -0.20:
        score += 1
        reasons.append("significant_3m_decline")
    if position_in_range is not None and position_in_range < 0.10:
        score += 1
        reasons.append("near_52w_low")
    if sma_200_slope is not None and sma_200_slope < 0:
        score += 1
        reasons.append("sma200_downtrend")
    if days_since_52w_high is not None and days_since_52w_high > 180:
        score += 1
        reasons.append("stale_52w_high")
    return score, reasons


def is_falling_knife_technicals(technicals_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate the falling-knife rubric directly from a technicals payload."""
    ma = technicals_data.get("moving_averages") or {}
    rules = ma.get("rules") or {}
    pos = technicals_data.get("price_position") or {}
    returns = technicals_data.get("returns") or {}

    death_cross = (rules.get("death_cross") or {}).get("triggered")
    above = (rules.get("above_sma200") or {}).get("triggered")
    below_sma200 = None if above is None else not above

    score, reasons = falling_knife_assessment(
        death_cross=death_cross,
        below_sma200=below_sma200,
        return_3m=returns.get("return_3m"),
        position_in_range=pos.get("position_in_range"),
        sma_200_slope=ma.get("sma_200_slope_pct_per_day"),
        days_since_52w_high=pos.get("days_since_52w_high"),
    )
    return score >= FALLING_KNIFE_SCORE_THRESHOLD, reasons
