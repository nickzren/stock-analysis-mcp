"""Decision card: analyze's canonical decision in a compact "what do I do now?" block.

evaluate_gates computes every analyze hard gate once (hard_gates) and the ids of the applicable
gates it could not compute; decision.decide() resolves them against the uncapped candidate
policy; build_decision_card assembles decide()'s output with hard_gates, sizing, entry, exit,
conditions and next_review. The older fields (`policy_action`, `decision_modes`, ...) stay
alongside it until WP2.3.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from stock_analysis.tools.analyze.gates import (
    ANALYZE_GATE_IDS,
    UNEVALUATED,
    check_data_quality,
    check_earnings_blackout,
    check_earnings_unverifiable,
    check_liquidity,
    check_near_52w_high,
    check_thesis_broken,
    earnings_window,
    events_tool_failed,
    format_day,
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
    *,
    thesis_blocker: dict[str, Any] | None = None,
    zone_blocker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every gate analyze evaluates, once each, in disclosure order (gates.ANALYZE_GATE_IDS).

    The knife reads the technicals for every instrument (D-5a). ``support`` (WP4, D5): an
    unsupported or unverified instrument/currency blocks new money, and a non-USD price makes
    no dollar-liquidity claim. ``freshness_blockers`` come from the price freshness legs; the
    thesis and zone blockers from gates.check_thesis_broken and check_near_52w_high.
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
        thesis_blocker,
        *(support_blockers(support) if support is not None else []),
        check_earnings_blackout(events_data)[1],
        check_earnings_unverifiable(events_data, data_quality.get("tool_failures"))[1],
        zone_blocker,
    ]
    merged = merge_blockers(b for b in candidates if b is not None)
    blocking = sorted(merged, key=lambda b: _ANALYZE_ORDER[b["id"]])  # KeyError: an id outside the registry
    fired = {b["id"] for b in blocking}
    return {"any_blocking": bool(blocking), "blocking": blocking,
            "checks": {gate_id: gate_id in fired for gate_id in ANALYZE_GATE_IDS}}


def evaluate_gates(
    *,
    events_data: dict[str, Any],
    data_quality: dict[str, Any],
    technicals_data: dict[str, Any] | None,
    fundamentals_summary: dict[str, Any],
    risk_data: dict[str, Any],
    support: dict[str, Any] | None,
    freshness_blockers: list[dict[str, Any]] | None,
    thesis_signals: dict[str, Any],
    zone: str | None,
) -> tuple[dict[str, Any], frozenset[str]]:
    """(hard_gates, the ids of applicable analyze gates that were not computed) (plan §4, R2.6(f))."""
    fundamentals_status = data_quality.get("fundamentals_status")
    thesis_status, thesis_blocker = check_thesis_broken(fundamentals_status, thesis_signals)
    zone_status, zone_blocker = check_near_52w_high(zone)
    hard_gates = _compute_hard_gates(
        events_data=events_data, data_quality=data_quality, technicals_data=technicals_data,
        fundamentals_summary=fundamentals_summary, risk_data=risk_data, support=support,
        freshness_blockers=freshness_blockers, thesis_blocker=thesis_blocker, zone_blocker=zone_blocker,
    )
    not_computed = {
        "falling_knife": not technicals_data,
        "missing_runway": fundamentals_status == "missing",
        "thesis_broken": thesis_status == UNEVALUATED,
        "earnings_blackout": events_tool_failed(data_quality.get("tool_failures")),
        "near_52w_high": zone_status == UNEVALUATED,
    }
    return hard_gates, frozenset(gate for gate, missing in not_computed.items() if missing)


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
    """Pick the next review date: inside an active earnings interval, the first session after it (never
    on or before today, RA.17); else next earnings if within ~90 days; else default."""
    earnings = events_data.get("earnings") or {}
    today = now.date()
    window = earnings_window(earnings, today)
    active = any(start <= today <= end for start, end in window.intervals)
    cleared = window.clear_session(today) if active else None
    if cleared is not None:
        day, through = cleared
        return {"date": day.isoformat(), "reason": f"after earnings ({format_day(through)})"}

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
    events_data: dict[str, Any],
    dislocation_framework: dict[str, Any] | None,
    now: datetime,
    hard_gates: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Build the compact small-account decision card around decide()'s decision.

    Schema:
        horizon, new_money, if_held, long_term: the canonical decision (decision.decide)
        hard_gates: every fired gate (blocking) and one boolean per gates.ANALYZE_GATE_IDS (checks)
        action_now: alias of new_money.action (decision.NEW_MONEY_ACTIONS), kept one release
        rationale: the emitted mid-term policy code, the new_money action and its deciding gate
        sizing: starter/full pct + dollars + whole/fractional shares
        entry: current zone, price, buy band
        exit: stop / invalidation level
        conditions: buy_only_if / add_only_if / do_not_buy_if / reduce_if / monitor
        next_review: date + reason
        confidence: from verdict
        horizon_fit: mid/long-term action labels from policy_action
    """
    new_money = decision["new_money"]
    rationale = [f"policy_action.mid_term={policy_action.get('mid_term')} -> {new_money['action']}"]
    if new_money["decided_by"]:
        rationale.append(f"decided_by={new_money['decided_by']}")

    return {
        **decision,
        "hard_gates": hard_gates,
        "action_now": new_money["action"],
        "rationale": rationale,
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
