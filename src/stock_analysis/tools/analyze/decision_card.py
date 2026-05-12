"""Decision card: a compact, gated "what do I do now?" block.

Consolidates the scattered outputs (`policy_action`, `decision_modes`, `action_zones`,
events, data_quality, dip_assessment, burn_metrics) into a single small-account-friendly
block. Applies hard pre-buy gates *before* surfacing any bullish or starter action so
a single blocking condition (e.g., earnings within 5 days, falling knife, missing
runway on an unprofitable burner) overrides the headline action.

This is additive — it does not replace the existing fields, just synthesizes them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

# Liquidity threshold below which we block new buys for small accounts.
# Sized so that a $5K position is well under 1% of typical daily volume.
_WEAK_LIQUIDITY_DOLLARS_PER_DAY = 1_000_000.0

# Number of days before earnings during which we block new buys.
# Earnings move stocks ±5-15% on average; entering inside this window is gambling.
_EARNINGS_BLACKOUT_DAYS = 5

# Default look-ahead for "next_review.date" when no earnings date is near.
_DEFAULT_REVIEW_HORIZON_DAYS = 30


def _compute_hard_gates(
    events_data: dict[str, Any],
    data_quality: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
    fundamentals_summary: dict[str, Any],
    risk_data: dict[str, Any],
) -> dict[str, Any]:
    """Collect blocking conditions that must override any bullish recommendation."""
    earnings = events_data.get("earnings") or {}
    days_until_earnings = earnings.get("days_until")
    earnings_blackout = (
        isinstance(days_until_earnings, (int, float))
        and 0 <= days_until_earnings <= _EARNINGS_BLACKOUT_DAYS
    )

    fund_status = data_quality.get("fundamentals_status")
    data_quality_critical = fund_status == "missing"

    dip_type: str | None = None
    if dip_assessment:
        dip_type = (dip_assessment.get("dip_classification") or {}).get("type")
    falling_knife = dip_type == "falling_knife"

    burn = fundamentals_summary.get("burn_metrics") or {}
    burn_status = burn.get("status")
    valuation_summary = fundamentals_summary.get("valuation") or {}
    is_unprofitable = valuation_summary.get("valuation_note") == "pe_not_meaningful"
    missing_runway = is_unprofitable and burn_status == "unavailable"

    liquidity = risk_data.get("liquidity") or {}
    avg_dollar_volume = liquidity.get("avg_dollar_volume")
    weak_liquidity = (
        isinstance(avg_dollar_volume, (int, float))
        and avg_dollar_volume < _WEAK_LIQUIDITY_DOLLARS_PER_DAY
    )

    blocking: list[dict[str, str]] = []
    if earnings_blackout and isinstance(days_until_earnings, (int, float)):
        blocking.append({
            "id": "earnings_blackout",
            "reason": f"earnings in {int(days_until_earnings)} days (<= {_EARNINGS_BLACKOUT_DAYS})",
        })
    if data_quality_critical:
        blocking.append({
            "id": "data_quality_critical",
            "reason": "fundamentals data unavailable; cannot assess business quality",
        })
    if falling_knife:
        blocking.append({
            "id": "falling_knife",
            "reason": "severe decline with broken trends — wait for stabilization",
        })
    if missing_runway:
        blocking.append({
            "id": "missing_runway",
            "reason": "unprofitable company with no usable runway data — uninvestable for sizing",
        })
    if weak_liquidity and isinstance(avg_dollar_volume, (int, float)):
        blocking.append({
            "id": "weak_liquidity",
            "reason": (
                f"avg daily dollar volume "
                f"~${avg_dollar_volume / 1_000_000:.2f}M < "
                f"${_WEAK_LIQUIDITY_DOLLARS_PER_DAY / 1_000_000:.0f}M threshold"
            ),
        })

    return {
        "any_blocking": bool(blocking),
        "blocking": blocking,
        "checks": {
            "earnings_blackout": earnings_blackout,
            "data_quality_critical": data_quality_critical,
            "falling_knife": falling_knife,
            "missing_runway": missing_runway,
            "weak_liquidity": weak_liquidity,
        },
    }


# Map verbose policy_action.mid_term values to compact action_now codes used by
# the decision card. Falls back to the raw value for anything unrecognized.
_MID_TERM_TO_ACTION: dict[str, str] = {
    "buy": "buy",
    "hold_or_add": "hold",
    "hold": "hold",
    "hold_or_reduce": "hold_or_reduce",
    "speculative_small_position": "starter",
    "wait_for_entry": "wait",
    "avoid": "avoid",
    "insufficient_data": "insufficient_data",
}


def _resolve_action_now(
    policy_action: dict[str, Any],
    hard_gates: dict[str, Any],
) -> tuple[str, list[str]]:
    """Pick the compact action_now code, applying hard gates as a pre-filter."""
    rationale: list[str] = []
    mid_term = policy_action.get("mid_term") or "insufficient_data"
    base_action = _MID_TERM_TO_ACTION.get(mid_term, mid_term)
    rationale.append(f"policy_action.mid_term={mid_term} -> {base_action}")

    # Hard gates only override buy/starter — not hold/reduce/sell paths. A user
    # already holding the position needs different guidance than someone
    # considering a new entry, but at the analysis layer we don't know which.
    # Conservative override: bullish-leaning actions become "wait_for_data" so
    # the user re-evaluates after the gate clears.
    if hard_gates["any_blocking"] and base_action in ("buy", "starter"):
        gate_ids = ",".join(g["id"] for g in hard_gates["blocking"])
        rationale.append(f"hard_gates fired ({gate_ids}); downgrading to wait_for_data")
        return "wait_for_data", rationale

    return base_action, rationale


def _build_sizing(
    action_zones: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Extract sizing into a flat block; expose fractional shares alongside whole."""
    position_sizing = action_zones.get("position_sizing_range") or {}
    current_price = summary.get("current_price")

    starter_pct = position_sizing.get("starter_pct")
    full_pct_cap = position_sizing.get("max_pct")

    dollars_block = position_sizing.get("dollars_for_account") or {}
    starter_dollars = dollars_block.get("min") if dollars_block else None
    full_dollars = dollars_block.get("max") if dollars_block else None

    shares_block = position_sizing.get("shares_range") or {}
    whole_shares = shares_block.get("min") if shares_block else None
    at_price = shares_block.get("at_price") or current_price

    fractional_shares: float | None = None
    if (
        starter_dollars is not None
        and at_price is not None
        and isinstance(at_price, (int, float))
        and at_price > 0
    ):
        fractional_shares = round(starter_dollars / at_price, 4)

    return {
        "starter_pct": starter_pct,
        "full_pct_cap": full_pct_cap,
        "starter_dollars": starter_dollars,
        "full_dollars": full_dollars,
        "whole_shares": whole_shares,
        "fractional_shares": fractional_shares,
        "at_price": at_price,
    }


def _build_entry(
    action_zones: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the entry block (zone, price, buy band)."""
    levels = action_zones.get("levels") or {}
    return {
        "current_zone": action_zones.get("current_zone"),
        "current_price": summary.get("current_price"),
        "buy_zone_low": levels.get("strong_buy_below"),
        "buy_zone_high": levels.get("accumulate_near"),
    }


def _build_exit(action_zones: dict[str, Any]) -> dict[str, Any]:
    """Build the exit block (stop / invalidation level)."""
    levels = action_zones.get("levels") or {}
    basis = action_zones.get("basis") or {}
    return {
        "stop_or_invalidation": levels.get("stop_loss"),
        "stop_basis": basis.get("stop_loss"),
    }


def _build_conditions(
    policy_action: dict[str, Any],
    hard_gates: dict[str, Any],
    dislocation_framework: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compress conditions: add_only_if, do_not_buy_if (hard gates), reduce_if, monitor."""
    do_not_buy_if = [g["reason"] for g in hard_gates["blocking"]]

    add_only_if: list[str] = []
    if dislocation_framework:
        action_block = dislocation_framework.get("action") or {}
        framework_add_only_if = action_block.get("add_only_if") or []
        if isinstance(framework_add_only_if, list):
            add_only_if = [str(x) for x in framework_add_only_if]

    reduce_if = list(policy_action.get("conditions_to_downgrade") or [])
    monitor = list(policy_action.get("conditions_to_upgrade") or [])

    return {
        "add_only_if": add_only_if,
        "do_not_buy_if": do_not_buy_if,
        "reduce_if": reduce_if,
        "monitor": monitor,
    }


def _build_next_review(events_data: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Pick the next review date: next earnings if within ~90 days, else default."""
    earnings = events_data.get("earnings") or {}
    next_earnings = earnings.get("next_date")
    days_until_earnings = earnings.get("days_until")

    if (
        next_earnings
        and isinstance(days_until_earnings, (int, float))
        and 0 <= days_until_earnings <= 90
    ):
        return {
            "date": str(next_earnings),
            "reason": f"next earnings (~{int(days_until_earnings)} days)",
        }

    anchor = now or datetime.now()
    review_date = anchor + timedelta(days=_DEFAULT_REVIEW_HORIZON_DAYS)
    return {
        "date": review_date.date().isoformat(),
        "reason": f"default {_DEFAULT_REVIEW_HORIZON_DAYS}-day re-evaluation",
    }


def build_decision_card(
    *,
    summary: dict[str, Any],
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    policy_action: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    risk_data: dict[str, Any],
    events_data: dict[str, Any],
    data_quality: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
    dislocation_framework: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the compact small-account decision card.

    Schema:
        action_now: buy | starter | hold | hold_or_reduce | wait | wait_for_data |
                    avoid | insufficient_data
        rationale: short bulleted reasons
        hard_gates: blocking conditions that override bullish recommendations
        sizing: starter/full pct + dollars + whole/fractional shares
        entry: current zone, price, buy band
        exit: stop / invalidation level
        conditions: add_only_if / do_not_buy_if / reduce_if / monitor
        next_review: date + reason
        confidence: from verdict
        horizon_fit: mid/long-term action labels from policy_action
    """
    hard_gates = _compute_hard_gates(
        events_data=events_data,
        data_quality=data_quality,
        dip_assessment=dip_assessment,
        fundamentals_summary=fundamentals_summary,
        risk_data=risk_data,
    )
    action_now, rationale = _resolve_action_now(policy_action, hard_gates)

    return {
        "action_now": action_now,
        "rationale": rationale,
        "hard_gates": hard_gates,
        "sizing": _build_sizing(action_zones, summary),
        "entry": _build_entry(action_zones, summary),
        "exit": _build_exit(action_zones),
        "conditions": _build_conditions(policy_action, hard_gates, dislocation_framework),
        "next_review": _build_next_review(events_data),
        "confidence": verdict.get("confidence", "low"),
        "horizon_fit": {
            "mid_term": policy_action.get("mid_term"),
            "long_term": policy_action.get("long_term"),
        },
    }
