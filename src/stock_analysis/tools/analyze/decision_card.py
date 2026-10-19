"""Decision card: analyze's canonical decision in a compact "what do I do now?" block.

evaluate_gates computes every analyze hard gate once (hard_gates) and the ids of the applicable
gates it could not compute; decision.decide() resolves them against the uncapped candidate
policy; build_conditions lists the conditions that bind each horizon; build_decision_card
assembles decide()'s output with hard_gates, why, sizing, entry, exit, conditions and
next_review. build_diagnostics emits the pre-gate candidate (full tier only).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from math import isfinite
from numbers import Real
from typing import Any

from stock_analysis.tools.analyze.action_zones import zone_after_caps
from stock_analysis.tools.analyze.conditions import CandidateInputs, Factor
from stock_analysis.tools.analyze.decision import (
    ENTRY_ACTIONS,
    LONG_TERM,
    MID_TERM,
    entry_allowed,
    map_candidate,
    starter_only,
    tighter,
)
from stock_analysis.tools.analyze.executive_summary import policy_codes
from stock_analysis.tools.analyze.gates import (
    ANALYZE_GATE_IDS,
    GATES,
    UNEVALUATED,
    check_data_quality,
    check_earnings_blackout,
    check_earnings_unverifiable,
    check_liquidity,
    check_near_52w_high,
    check_thesis_broken,
    clears_when_for,
    earnings_window,
    events_tool_failed,
    falling_knife_status,
    format_day,
    is_falling_knife_technicals,
    liquidity_applicability_unverified,
    merge_blockers,
    stale_data_unevaluated,
)
from stock_analysis.tools.analyze.reassessment import permanent_support_gate
from stock_analysis.tools.analyze.verdict import build_horizon_fit
from stock_analysis.utils.valuation_basis import sizing_unavailable_reason, support_blockers

PRE_GATE_ROLE = "pre_gate_input_not_advice"

# Default look-ahead for "next_review.date" when no earnings date is near.
_DEFAULT_REVIEW_HORIZON_DAYS = 30

_MISSING_RUNWAY = {"id": "missing_runway",
                   "reason": "unprofitable company with no usable runway data — uninvestable for sizing"}
_THESIS_UNVERIFIABLE = {"id": "thesis_unverifiable",
                        "reason": "required business or cash-runway evidence unavailable, or applicability unverified"}
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
    thesis_unverifiable: bool = False,
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
        _THESIS_UNVERIFIABLE if thesis_unverifiable else None,
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
    price_zone: str | None,
) -> tuple[dict[str, Any], frozenset[str]]:
    """(hard_gates, the ids of applicable analyze gates that were not computed) (plan §4, R2.6(f))."""
    fundamentals_status = data_quality.get("fundamentals_status")
    thesis_status, thesis_blocker = check_thesis_broken(fundamentals_status, thesis_signals)
    zone_status, zone_blocker = check_near_52w_high(price_zone)
    hard_gates = _compute_hard_gates(
        events_data=events_data, data_quality=data_quality, technicals_data=technicals_data,
        fundamentals_summary=fundamentals_summary, risk_data=risk_data, support=support,
        freshness_blockers=freshness_blockers, thesis_blocker=thesis_blocker, zone_blocker=zone_blocker,
        thesis_unverifiable=thesis_status == UNEVALUATED,
    )
    not_computed = {
        "falling_knife": falling_knife_status(technicals_data) == UNEVALUATED,
        "stale_data": stale_data_unevaluated(
            data_quality.get("freshness"), fired=hard_gates["checks"]["stale_data"]),
        "missing_runway": fundamentals_status in ("missing", "unverified"),
        "thesis_broken": thesis_status == UNEVALUATED,
        "earnings_blackout": events_tool_failed(data_quality.get("tool_failures")),
        "near_52w_high": zone_status == UNEVALUATED,
        "weak_liquidity": (hard_gates["checks"]["liquidity_missing"]
                           or liquidity_applicability_unverified(support)),
    }
    return hard_gates, frozenset(gate for gate, missing in not_computed.items() if missing)


def adapt_card_sizing(
    sizing_result: dict[str, Any],
    new_money: dict[str, Any],
) -> dict[str, Any]:
    """Apply entry visibility and retain legacy fields from one computed size."""
    action = new_money["action"]
    if not entry_allowed(action):
        blockers = ", ".join(blocker["id"] for blocker in new_money.get("blocked_by", []))
        reason = f"new_money is {action}" + (f"; blocked by: {blockers}" if blockers else "")
        return {"sizing": None, "sizing_status": {"status": "not_allowed", "reason": reason}}
    canonical = sizing_result["sizing"]
    if canonical is None:
        return sizing_result
    starter = canonical["starter"]
    full = None if starter_only(action) else canonical["full"]
    sizing = {
        **canonical,
        "full": full,
        "starter_pct": starter["allocation_cap_pct"],
        "full_pct_cap": full["allocation_cap_pct"] if full else None,
        "starter_dollars": starter["position_dollars"],
        "full_dollars": full["position_dollars"] if full else None,
        "whole_shares": starter["whole_shares"],
        "fractional_shares": None,
        "at_price": canonical["assumptions"]["entry_price"],
        "unavailable_reason": None,
    }
    return {**sizing_result, "sizing": sizing}


def _build_entry(
    action_zones: dict[str, Any],
    summary: dict[str, Any],
    new_money_action: str,
) -> dict[str, Any] | None:
    """The buy band (price, band edges) iff new_money admits an entry (plan §7)."""
    if not entry_allowed(new_money_action):
        return None
    levels = action_zones.get("levels") or {}
    low, high = levels.get("strong_buy_below"), levels.get("accumulate_near")
    reason = None
    if low is None or high is None:
        reason = "missing"
    elif any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0
             for value in (low, high)):
        reason = "invalid"
    elif low > high:
        reason = "inverted"
    return {
        "current_price": summary.get("current_price"),
        "buy_zone_low": None if reason else low,
        "buy_zone_high": None if reason else high,
        "entry_band_unavailable": {"reason": reason} if reason else None,
    }


def _build_exit(action_zones: dict[str, Any]) -> dict[str, Any]:
    """Build the exit block (stop / invalidation level)."""
    levels = action_zones.get("levels") or {}
    basis = action_zones.get("basis") or {}
    return {
        "stop_or_invalidation": levels.get("stop_loss"),
        "stop_basis": basis.get("stop_loss"),
    }


def candidate_codes(inputs: CandidateInputs, overrides: Mapping[str, Any]) -> tuple[str, str]:
    """The pre-gate (mid_term, long_term) codes with ``overrides`` applied to what the candidate read."""
    fit_inputs = {**inputs.fit, **{k: v for k, v in overrides.items() if k in inputs.fit}}
    fit = build_horizon_fit(**fit_inputs)
    valuation_gate = overrides.get("valuation_gate", inputs.valuation_gate)
    zone = zone_after_caps(inputs.uncapped_zone, extreme=fit_inputs["risk_label"] == "extreme",
                           valuation_gate=valuation_gate, dip_type=inputs.dip_type)
    mid, long, _ = policy_codes(
        mid_term_fit=fit["mid_term"], long_term_fit=fit["long_term"], long_term_gates=fit["long_term_gates"],
        tilt=inputs.tilt, zone=zone, valuation_gate=valuation_gate,
        is_unprofitable=overrides.get("is_unprofitable", inputs.is_unprofitable), dislocation=inputs.dislocation,
    )
    return mid, long


def build_conditions(
    *,
    decision: dict[str, Any],
    candidate: Mapping[str, Any],
    factors: list[Factor],
    inputs: CandidateInputs | None,
    downgrade: list[str],
    support: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Only binding conditions, per horizon (plan §7e [FIX BL-E1, BL-E2]).

    A horizon lists the clears_when of each fired non-support gate tighter than map(candidate), then each factor
    whose cleared state alone moves map(candidate) into the target: an entry action when the candidate does not
    admit new money, buy under a starter (starter -> full). A permanent support gate empties the horizon.
    ``candidate`` is the uncapped pre-gate policy decide() read; ``inputs`` may be None only without factors.
    """
    replayed = [(factor, candidate_codes(inputs, factor.overrides)) for factor in factors] if inputs else []
    binding: dict[str, list[str] | None] = {}
    for index, horizon, new_money in ((0, MID_TERM, decision["new_money"]),
                                      (1, LONG_TERM, decision[LONG_TERM]["new_money"])):
        if any(permanent_support_gate(b["id"], support) for b in new_money["blocked_by"]):
            binding[horizon] = None
            continue
        mapped = map_candidate(horizon, candidate[horizon])
        gates = [clears_when_for(b) for b in new_money["blocked_by"]
                 if b["class"] != "support" and tighter(GATES[b["id"]].outcome[horizon], mapped)]
        target = (ENTRY_ACTIONS if not entry_allowed(mapped)
                  else frozenset({"buy"}) if starter_only(new_money["action"]) else frozenset())
        cleared = [text for factor, codes in replayed if map_candidate(horizon, codes[index]) in target
                   for text in factor.texts]
        binding[horizon] = list(dict.fromkeys(gates + cleared))
    mid = binding[MID_TERM] or []
    upgrade = list(dict.fromkeys(text for factor in factors for text in factor.texts))
    return {
        MID_TERM: {
            "buy_only_if": mid,
            "add_only_if": list(mid) if decision["if_held"]["action"] == "hold" else [],
            "do_not_buy_if": [b["reason"] for b in decision["new_money"]["blocked_by"]],
            "reduce_if": list(downgrade),
            "monitor": [] if binding[MID_TERM] is None else [text for text in upgrade if text not in mid],
        },
        LONG_TERM: {"buy_only_if": binding[LONG_TERM] or []},
    }


def build_why(decision: dict[str, Any], verdict: dict[str, Any], price_zone: str | None) -> list[str]:
    """decision_card.why (BL-E3): the deciding gate, the mid-term evidence, the other fired gates. A-class text
    (mid-term): never a gate reason or a pre-gate code."""
    new_money = decision["new_money"]
    why = [f"decided by {new_money['decided_by']}"] if new_money["decided_by"] else []
    evidence = (f"mid-term fit {(verdict.get('horizon_fit') or {}).get('mid_term') or 'unknown'}, "
                f"tilt {verdict.get('tilt') or 'missing'}")
    why.append(evidence + (f", price zone {price_zone}" if price_zone else ""))
    others = [b["id"] for b in new_money["blocked_by"] if b["id"] != new_money["decided_by"]]
    if others:
        why.append("other gates fired: " + ", ".join(others))
    return why


def _exit_triggers(dislocation_framework: dict[str, Any] | None, data_quality: dict[str, Any]) -> list[str]:
    """decision_card.exit_triggers (BL-E3): the thesis stop triggers, none while fundamentals are missing."""
    if data_quality.get("fundamentals_status") == "missing":
        return []
    thesis = (dislocation_framework or {}).get("thesis_integrity") or {}
    return list(thesis.get("sell_without_attachment_if") or [])


def _data_flags(data_quality: dict[str, Any], action_zones: dict[str, Any]) -> list[str]:
    """decision_card.data_flags (BL-E3): fundamentals status, staleness and zone warnings, in that order."""
    status = data_quality.get("fundamentals_status")
    flags = [f"fundamentals_{status}"] if status not in (None, "available") else []
    flags += [*(data_quality.get("staleness_warnings") or []), *(action_zones.get("zone_warnings") or [])]
    return list(dict.fromkeys(flags))


def _build_next_review(events_data: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Pick the next review date: inside an active earnings interval, the first session after it, or the
    default when that interval is undatable (never on or before today, RA.17); else next earnings if within
    ~90 days; else default."""
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
        not active
        and next_earnings
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
        "reason": f"default {_DEFAULT_REVIEW_HORIZON_DAYS}-day re-evaluation"
        + (" (earnings window undatable)" if active else ""),
    }


def build_decision_card(
    *,
    summary: dict[str, Any],
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    events_data: dict[str, Any],
    dislocation_framework: dict[str, Any] | None,
    now: datetime,
    hard_gates: dict[str, Any],
    decision: dict[str, Any],
    conditions: dict[str, Any],
    data_quality: dict[str, Any],
    why: list[str],
    sizing_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact small-account decision card around decide()'s decision.

    Schema:
        horizon, new_money, if_held, long_term: the canonical decision (decision.decide); long_term.new_money
            carries its own conditions (buy_only_if)
        hard_gates: every fired gate (blocking) and one boolean per gates.ANALYZE_GATE_IDS (checks)
        action_now: alias of new_money.action (decision.NEW_MONEY_ACTIONS), kept one release
        why, exit_triggers, data_flags: the deciding evidence, the thesis stop triggers, the data caveats
        sizing: one validated result for mid-term new money; a starter hides the full size
        sizing_status: availability and reason, present even when sizing is null
        entry: price and buy band, iff new_money is buy or starter
        exit: stop / invalidation level
        conditions: mid-term buy_only_if / add_only_if / do_not_buy_if / reduce_if / monitor
        next_review: date + reason
        confidence: from verdict
    """
    new_money = decision["new_money"]
    long_term = decision[LONG_TERM]
    if sizing_result is None:
        sizing_result = adapt_card_sizing({
            "sizing": None, "sizing_status": {"status": "unavailable", "reason": "sizing_inputs_unavailable"},
        }, new_money)
    return {
        **decision,
        LONG_TERM: {**long_term, "new_money": {**long_term["new_money"], "conditions": conditions[LONG_TERM]}},
        "hard_gates": hard_gates,
        "action_now": new_money["action"],
        "why": why,
        "exit_triggers": _exit_triggers(dislocation_framework, data_quality),
        "data_flags": _data_flags(data_quality, action_zones),
        "sizing": sizing_result["sizing"],
        "sizing_status": sizing_result["sizing_status"],
        "entry": _build_entry(action_zones, summary, new_money["action"]),
        "exit": _build_exit(action_zones),
        "conditions": conditions[MID_TERM],
        "next_review": _build_next_review(events_data, now),
        "confidence": verdict.get("confidence", "low"),
    }


def build_diagnostics(pre_gate_policy: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """diagnostics (full tier only, P class; plan §2, §7a): the emitted, support-capped candidate next to the
    decision that supersedes it, and the decision trace (was decision_card.rationale)."""
    new_money = decision["new_money"]
    trace = [f"pre_gate_policy.mid_term={pre_gate_policy['mid_term']} -> {new_money['action']}"]
    if new_money["decided_by"]:
        trace.append(f"decided_by={new_money['decided_by']}")
    return {
        "pre_gate_policy": {
            **pre_gate_policy,
            "role": PRE_GATE_ROLE,
            "superseded_by": {"new_money": new_money["action"], "if_held": decision["if_held"]["action"],
                              "long_term": decision[LONG_TERM]["new_money"]["action"],
                              "blocked_by": [b["id"] for b in new_money["blocked_by"]]},
        },
        "decision_trace": trace,
    }
