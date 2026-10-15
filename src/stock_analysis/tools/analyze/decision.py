"""The canonical analyze decision (plan §2-§6): one pure decide()."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import date
from typing import Any

from stock_analysis.tools.analyze.gates import (
    BUSINESS_BROKEN,
    GATES,
    NOT_APPLICABLE,
    RUNWAY_CRITICAL,
    EarningsWindow,
    disclose,
    disclosure_order,
)
from stock_analysis.tools.analyze.reassessment import earliest_reassessment

MID_TERM, LONG_TERM = "mid_term", "long_term"

# The new-money vocabulary, loosest first: plan §0 #2's tightness order read right to left.
NEW_MONEY_ACTIONS = ("buy", "starter", "wait", "wait_for_data", "insufficient_data", "watch", "avoid")
_TIGHTNESS = {action: rank for rank, action in enumerate(NEW_MONEY_ACTIONS)}
ENTRY_ACTIONS = frozenset({"buy", "starter"})  # the one new-money code set (plan §7 entry_allowed)

CANDIDATE_ACTIONS: dict[str, dict[str, str]] = {  # plan §2 [D-2]: every pre-gate policy code
    MID_TERM: {
        "buy": "buy", "hold_or_add": "starter", "speculative_small_position": "starter",
        "hold": "wait", "hold_or_reduce": "wait", "wait_for_entry": "wait",
        "avoid": "avoid", "insufficient_data": "insufficient_data",
    },
    LONG_TERM: {
        "accumulate": "buy", "hold_or_add": "starter", "small_position_with_stops": "starter",
        "speculative_with_thesis": "starter", "speculative_only": "starter",
        "hold": "wait", "hold_size_conservatively": "wait", "wait_or_avoid": "wait",
        "avoid_core_position": "avoid", "insufficient_data": "insufficient_data",
    },
}
ENTRY_CANDIDATE_CODES = frozenset(code for table in CANDIDATE_ACTIONS.values()
                                  for code, action in table.items() if action in ENTRY_ACTIONS)

# Every code that means "put money in" on any analyze surface (annex output_contract §0), and the starter-sized
# ones; the dip caps and the output-contract lints read them (plan §7).
ENTRY_CODES = frozenset({
    "buy", "starter", "hold_or_add", "accumulate", "speculative_small_position", "speculative_with_thesis",
    "speculative_only", "small_position_with_stops", "build_position", "starter_position_only", "starter_then_add",
    "strong_buy_the_dip", "buy_the_dip", "cautious_accumulation", "small_speculative_position",
    "starter_only_wait_for_stabilization", "consider_small_position", "starter_position", "add_position",
    "strong_buy", "small_starter", "starter_only",
})
STARTER_ONLY_CODES = frozenset({
    "starter", "speculative_small_position", "small_position_with_stops", "starter_position_only",
    "small_speculative_position", "starter_only_wait_for_stabilization", "consider_small_position",
    "starter_position", "small_starter", "starter_only",
})
FULL_ENTRY_CODES = ENTRY_CODES - STARTER_ONLY_CODES

IF_HELD_ACTIONS = ("hold", "reduce", "review_exit", "insufficient_data")
EVIDENCE_BASES = (RUNWAY_CRITICAL, BUSINESS_BROKEN, "tilt_bearish")  # required for reduce and review_exit
NON_EVIDENCE_BASES = ("candidate_admits_new_money", "price_broken_more_than_business", "no_reduction_evidence",
                      "tilt_missing", "candidate_insufficient")
CORE = {"ok": "buy", "caution": "wait", "unknown": "insufficient_data", "avoid": "avoid"}  # plan §3 [D-3]


def map_candidate(horizon: str, code: Any) -> str:
    """The new-money action for a pre-gate policy code; an unknown code raises (D-2)."""
    try:
        return CANDIDATE_ACTIONS[horizon][code]
    except (KeyError, TypeError):
        raise ValueError(f"unknown {horizon} candidate code: {code!r}") from None


def most_restrictive(actions: Iterable[str]) -> str:
    return max(actions, key=lambda action: _TIGHTNESS[action])


def tighter(action: str, than: str) -> bool:
    return _TIGHTNESS[action] > _TIGHTNESS[than]


def entry_allowed(action: str) -> bool:
    return action in ENTRY_ACTIONS


def if_held(*, thesis: tuple[str, ...], candidate_action: str, dislocation: bool, tilt: str | None) -> dict[str, str]:
    """Mid-term holder guidance, first match (plan §3). Reads evidence only: never gates, zone, dip or support."""
    rules = (
        (RUNWAY_CRITICAL in thesis, "review_exit", RUNWAY_CRITICAL),
        (BUSINESS_BROKEN in thesis, "review_exit", BUSINESS_BROKEN),
        (candidate_action in ENTRY_ACTIONS, "hold", "candidate_admits_new_money"),
        (dislocation, "hold", "price_broken_more_than_business"),
        (tilt is None, "insufficient_data", "tilt_missing"),
        (candidate_action == "insufficient_data", "insufficient_data", "candidate_insufficient"),
        (tilt == "bearish", "reduce", "tilt_bearish"),
    )
    action, basis = next(((a, b) for hit, a, b in rules if hit), ("hold", "no_reduction_evidence"))
    return {"action": action, "basis": basis}


def core_action(long_term_fit: str, long_term_action: str) -> str:
    """Core follows the canonical long-term decision under its own cap (plan §3 [D-3])."""
    return most_restrictive((CORE[long_term_fit], long_term_action))


def _new_money(horizon: str, mapped: str, blocking: Sequence[dict[str, Any]], unevaluated: Collection[str],
               window: EarningsWindow, today: date, support: Mapping[str, Any] | None) -> dict[str, Any]:
    order = disclosure_order(horizon)
    outcome = {b["id"]: GATES[b["id"]].outcome[horizon] for b in blocking}
    fired = sorted((b for b in blocking if outcome[b["id"]] != NOT_APPLICABLE), key=lambda b: order.index(b["id"]))
    blocked_by = [disclose(b, horizon) for b in fired]
    action = most_restrictive([mapped, *(outcome[b["id"]] for b in fired)])
    decided_by = None if action == mapped else next(b["id"] for b in fired if outcome[b["id"]] == action)
    return {
        "action": action,
        "blocked_by": blocked_by,
        "decided_by": decided_by,
        "unevaluated": [{"id": g, "reason": GATES[g].unevaluated_when[horizon]} for g in order if g in unevaluated],
        "earliest_reassessment": earliest_reassessment(blocked_by=blocked_by, admits=mapped in ENTRY_ACTIONS,
                                                       window=window, today=today, support=support),
    }


def decide(
    *,
    candidate: Mapping[str, Any],
    blocking: Sequence[dict[str, Any]],
    unevaluated: Collection[str],
    thesis: tuple[str, ...],
    dislocation: bool,
    tilt: str | None,
    window: EarningsWindow,
    today: date,
    support: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """new_money = most_restrictive(map(candidate), every applicable fired gate's outcome), per horizon (D-5).

    ``candidate`` is the pre-gate policy (no knife override, no support cap); ``blocking`` the fired analyze
    gates (unique ids); ``unevaluated`` the ids of applicable gates not computed. Gates never reach if_held;
    add_allowed follows the gated mid-term action.
    """
    mid_mapped = map_candidate(MID_TERM, candidate.get(MID_TERM))
    long_mapped = map_candidate(LONG_TERM, candidate.get(LONG_TERM))
    new_money = _new_money(MID_TERM, mid_mapped, blocking, unevaluated, window, today, support)
    held = if_held(thesis=thesis, candidate_action=mid_mapped, dislocation=dislocation, tilt=tilt)
    return {
        "horizon": MID_TERM,
        "new_money": new_money,
        "if_held": {"action": held["action"], "add_allowed": entry_allowed(new_money["action"]),
                    "basis": held["basis"]},
        "long_term": {"horizon": LONG_TERM,
                      "new_money": _new_money(LONG_TERM, long_mapped, blocking, unevaluated, window, today, support)},
    }
