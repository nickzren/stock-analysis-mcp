"""Decision card: a compact, gated "what do I do now?" block.

Consolidates the scattered outputs (`policy_action`, `decision_modes`, `action_zones`,
events, data_quality, technicals, burn_metrics) into a single small-account-friendly
block. Applies hard pre-buy gates *before* surfacing any bullish or starter action so
a single blocking condition (e.g., earnings within 5 days, falling knife, missing
runway on an unprofitable burner) overrides the headline action.

This is additive — it does not replace the existing fields, just synthesizes them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from stock_analysis.tools.analyze.gates import (
    ANALYZE_GATE_IDS,
    GATES,
    check_data_quality,
    check_earnings_blackout,
    check_earnings_unverifiable,
    check_liquidity,
    is_falling_knife_technicals,
    merge_blockers,
)
from stock_analysis.utils.valuation_basis import sizing_unavailable_reason, support_blockers

# Default look-ahead for "next_review.date" when no earnings date is near.
_DEFAULT_REVIEW_HORIZON_DAYS = 30

_MISSING_RUNWAY = {"id": "missing_runway",
                   "reason": "unprofitable company with no usable runway data — uninvestable for sizing"}
_FALLING_KNIFE = {"id": "falling_knife", "reason": "severe decline with broken trends — wait for stabilization"}
_ANALYZE_ORDER = {gate_id: i for i, gate_id in enumerate(ANALYZE_GATE_IDS)}


def _compute_hard_gates(
    events_data: dict[str, Any],
    data_quality: dict[str, Any],
    technicals_data: dict[str, Any] | None,
    fundamentals_summary: dict[str, Any],
    risk_data: dict[str, Any],
    support: dict[str, Any] | None = None,
    freshness_blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Every gate analyze evaluates, once each, in disclosure order (gates.ANALYZE_GATE_IDS).

    The knife reads the technicals for every instrument (D-5a). ``support`` (WP4, D5): an
    unsupported or unverified instrument/currency blocks new money, and a non-USD price makes
    no dollar-liquidity claim. ``freshness_blockers`` come from the price freshness legs.
    """
    burn = fundamentals_summary.get("burn_metrics") or {}
    valuation_summary = fundamentals_summary.get("valuation") or {}
    missing_runway = (valuation_summary.get("valuation_note") == "pe_not_meaningful"
                      and burn.get("status") == "unavailable")
    knife, _ = is_falling_knife_technicals(technicals_data or {})
    liquidity_applies = support is None or sizing_unavailable_reason(support) is None
    candidates = [
        *(freshness_blockers or []),
        check_data_quality(data_quality)[1],
        _MISSING_RUNWAY if missing_runway else None,
        _FALLING_KNIFE if knife else None,
        *(check_liquidity(risk_data)[2] if liquidity_applies else []),
        *(support_blockers(support) if support is not None else []),
        check_earnings_blackout(events_data)[1],
        check_earnings_unverifiable(events_data, data_quality.get("tool_failures"))[1],
    ]
    merged = merge_blockers(b for b in candidates if b is not None)
    blocking = sorted(merged, key=lambda b: _ANALYZE_ORDER[b["id"]])  # KeyError: an id outside the registry
    fired = {b["id"] for b in blocking}
    return {"any_blocking": bool(blocking), "blocking": blocking,
            "checks": {gate_id: gate_id in fired for gate_id in ANALYZE_GATE_IDS}}


# Map verbose policy_action.mid_term values to compact action_now codes used by
# the decision card. Falls back to the raw value for anything unrecognized.
#
# `hold_or_add` is preserved as its own code (not collapsed to `hold`) because it
# implies new-money is still appropriate for existing holders — and hard gates
# should treat it as buy-adjacent for the override.
_MID_TERM_TO_ACTION: dict[str, str] = {
    "buy": "buy",
    "hold_or_add": "hold_or_add",
    "hold": "hold",
    "hold_or_reduce": "hold_or_reduce",
    "speculative_small_position": "starter",
    "wait_for_entry": "wait",
    "avoid": "avoid",
    "insufficient_data": "insufficient_data",
}

# Actions that allow new money in (subject to hard gates).
_NEW_MONEY_ACTIONS = frozenset({"buy", "starter", "hold_or_add"})
_SUPPORT_GATE_IDS = frozenset(g for g, gate in GATES.items() if gate.gate_class == "support")


def _resolve_action_now(
    policy_action: dict[str, Any],
    hard_gates: dict[str, Any],
) -> tuple[str, list[str]]:
    """Pick the compact action_now code, applying hard gates as a pre-filter."""
    rationale: list[str] = []
    mid_term = policy_action.get("mid_term") or "insufficient_data"
    base_action = _MID_TERM_TO_ACTION.get(mid_term, mid_term)
    rationale.append(f"policy_action.mid_term={mid_term} -> {base_action}")

    # Hard gates override new-money actions only. Hold/reduce/avoid/sell pass
    # through because a user already holding the position needs different
    # guidance than someone considering a new entry, and the analysis layer
    # doesn't know which. Conservative override: actions that admit new money
    # become "wait_for_data" so the user re-evaluates after the gate clears.
    if hard_gates["any_blocking"] and base_action in _NEW_MONEY_ACTIONS:
        gate_ids = ",".join(g["id"] for g in hard_gates["blocking"])
        if any(g["id"] in _SUPPORT_GATE_IDS for g in hard_gates["blocking"]):
            # A permanent boundary, not missing data (WP4, D5).
            rationale.append(f"hard_gates fired ({gate_ids}); unsupported, capping at watch")
            return "watch", rationale
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
        "unavailable_reason": position_sizing.get("unavailable_reason"),
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
    """Compress conditions: buy_only_if, add_only_if, do_not_buy_if, reduce_if, monitor.

    `dislocation_framework.action` distinguishes between:
      - `buy_only_if`: conditions to lift before any new buy (when can_start_now=False)
      - `add_only_if`: conditions to lift before adding more (when can_start_now=True)
    Both surface here so the user sees the right guidance without inspecting the framework.
    """
    do_not_buy_if = [g["reason"] for g in hard_gates["blocking"]]

    add_only_if: list[str] = []
    buy_only_if: list[str] = []
    if dislocation_framework:
        action_block = dislocation_framework.get("action") or {}
        framework_add_only_if = action_block.get("add_only_if") or []
        framework_buy_only_if = action_block.get("buy_only_if") or []
        if isinstance(framework_add_only_if, list):
            add_only_if = [str(x) for x in framework_add_only_if]
        if isinstance(framework_buy_only_if, list):
            buy_only_if = [str(x) for x in framework_buy_only_if]

    reduce_if = list(policy_action.get("conditions_to_downgrade") or [])
    monitor = list(policy_action.get("conditions_to_upgrade") or [])

    return {
        "buy_only_if": buy_only_if,
        "add_only_if": add_only_if,
        "do_not_buy_if": do_not_buy_if,
        "reduce_if": reduce_if,
        "monitor": monitor,
    }


def _build_next_review(events_data: dict[str, Any], now: datetime) -> dict[str, Any]:
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

    review_date = now + timedelta(days=_DEFAULT_REVIEW_HORIZON_DAYS)
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
    technicals_data: dict[str, Any] | None,
    dislocation_framework: dict[str, Any] | None,
    now: datetime,
    freshness_blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the compact small-account decision card.

    Schema:
        action_now: buy | starter | hold | hold_or_add | hold_or_reduce | wait |
                    wait_for_data | avoid | insufficient_data
        rationale: short bulleted reasons
        hard_gates: blocking conditions that override new-money recommendations.
                    Checks: one boolean per gates.ANALYZE_GATE_IDS.
        sizing: starter/full pct + dollars + whole/fractional shares
        entry: current zone, price, buy band
        exit: stop / invalidation level
        conditions: buy_only_if / add_only_if / do_not_buy_if / reduce_if / monitor
        next_review: date + reason
        confidence: from verdict
        horizon_fit: mid/long-term action labels from policy_action
    """
    hard_gates = _compute_hard_gates(
        events_data=events_data,
        data_quality=data_quality,
        technicals_data=technicals_data,
        fundamentals_summary=fundamentals_summary,
        risk_data=risk_data,
        support=summary.get("support"),
        freshness_blockers=freshness_blockers,
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
        "next_review": _build_next_review(events_data, now),
        "confidence": verdict.get("confidence", "low"),
        "horizon_fit": {
            "mid_term": policy_action.get("mid_term"),
            "long_term": policy_action.get("long_term"),
        },
    }
