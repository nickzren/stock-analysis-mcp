"""WP2.2: the decision core (plan §2, §4): decide(), the candidate mapping and the analyze combination.

Annex: docs/plans/step2-v2-research/gate_contract.md and policy_holder.md, "Corrected proposal", as amended
by plan §0 #1 (most-restrictive in analyze; GC-R3, R4, R5, R13 rewritten) and §4 R2.6. The expected tables are
literals from the plan, not read back from the registry. Pure runs call decide() through dm.decide_on.
"""

from itertools import combinations, product
from typing import Any

import pytest

from stock_analysis.tools.analyze.decision import (
    CANDIDATE_ACTIONS,
    ENTRY_CANDIDATE_CODES,
    map_candidate,
)
from stock_analysis.tools.analyze.executive_summary import build_policy_action
from stock_analysis.tools.analyze.gates import ANALYZE_GATE_IDS, GATES, NOT_APPLICABLE, gate_rank
from tests.support import decision_matrix as dm

WFD = "wait_for_data"
ORDER = ("buy", "starter", "wait", WFD, "insufficient_data", "watch", "avoid")  # plan §0 #2, loosest first
MID_MAP = {"buy": "buy", "hold_or_add": "starter", "speculative_small_position": "starter", "hold": "wait",
           "hold_or_reduce": "wait", "wait_for_entry": "wait", "avoid": "avoid",
           "insufficient_data": "insufficient_data"}
LONG_MAP = {"accumulate": "buy", "hold_or_add": "starter", "small_position_with_stops": "starter",
            "speculative_with_thesis": "starter", "speculative_only": "starter", "hold": "wait",
            "hold_size_conservatively": "wait", "wait_or_avoid": "wait", "avoid_core_position": "avoid",
            "insufficient_data": "insufficient_data"}
MAPS = {"mid_term": MID_MAP, "long_term": LONG_MAP}
OUTCOME = {  # gate: (mid_term, long_term), the plan §4 registry's analyze columns [D-5a, D-5c]
    "stale_data": (WFD, WFD), "freshness_unverifiable": (WFD, WFD), "data_quality_critical": (WFD, WFD),
    "missing_runway": (WFD, WFD), "liquidity_missing": (WFD, WFD), "falling_knife": ("avoid", "avoid"),
    "weak_liquidity": ("avoid", "avoid"), "thesis_broken": ("avoid", "avoid"),
    "unsupported_instrument": ("watch", "watch"), "unsupported_currency": ("watch", "watch"),
    "earnings_blackout": ("wait", "wait"), "earnings_unverifiable": ("wait", "wait"),
    "near_52w_high": ("wait", NOT_APPLICABLE),
}
CLASS_EFFECT = {  # analyze class and effect; liquidity_missing is data in analyze (D-5c)
    "stale_data": ("data", "veto"), "freshness_unverifiable": ("data", "veto"), "data_quality_critical": ("data", "veto"),
    "missing_runway": ("data", "veto"), "liquidity_missing": ("data", "veto"), "falling_knife": ("price_risk", "veto"),
    "weak_liquidity": ("liquidity", "veto"), "thesis_broken": ("thesis", "cap"),
    "unsupported_instrument": ("support", "cap"), "unsupported_currency": ("support", "cap"),
    "earnings_blackout": ("event", "cap"), "earnings_unverifiable": ("event", "cap"), "near_52w_high": ("zone", "cap"),
}
RESOLUTION = {  # (map(candidate), one gate's outcome) -> new_money, most-restrictive [D-5]
    **{(base, outcome): outcome for base in ("buy", "starter") for outcome in (WFD, "avoid", "watch", "wait")},
    ("wait", WFD): WFD, ("wait", "avoid"): "avoid", ("wait", "watch"): "watch", ("wait", "wait"): "wait",
    ("insufficient_data", WFD): "insufficient_data", ("insufficient_data", "avoid"): "avoid",
    ("insufficient_data", "watch"): "watch", ("insufficient_data", "wait"): "insufficient_data",
    **{("avoid", outcome): "avoid" for outcome in (WFD, "avoid", "watch", "wait")},
}


def blockers(*gate_ids: str) -> list[dict[str, Any]]:
    """Fired blockers as _compute_hard_gates emits them (a thesis_broken blocker names its failing component)."""
    return [{"id": g, "reason": f"{g} fired", **({"codes": ["business_broken"]} if g == "thesis_broken" else {})}
            for g in gate_ids]


def run(mid: str = "hold_or_add", long: str = "accumulate", gates: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    return dm.decide_on({"mid_term": mid, "long_term": long}, blockers(*gates))


def ids(entries: list[dict[str, Any]]) -> list[str]:
    return [e["id"] for e in entries]


# --- PH-RF10 [D-2] --------------------------------------------------------------------------------


def test_every_candidate_code_maps_to_one_new_money_action() -> None:
    """PH-RF10: the 8 mid-term and 10 long-term pre-gate codes each map to one new-money action; an unknown
    code raises (today decision_card passes it through). PD D-2."""
    assert CANDIDATE_ACTIONS == MAPS
    assert {a for table in MAPS.values() for a in table.values()} <= set(ORDER)
    assert {"buy", "hold_or_add", "speculative_small_position", "accumulate", "small_position_with_stops",
            "speculative_with_thesis", "speculative_only"} == ENTRY_CANDIDATE_CODES
    for horizon, code in [("mid_term", "accumulate"), ("long_term", "buy"), ("mid_term", "watch"),
                          ("long_term", "watch"), ("mid_term", None), ("mid_term", "ok")]:
        with pytest.raises(ValueError, match=f"^unknown {horizon} candidate code: "):
            map_candidate(horizon, code)
    with pytest.raises(ValueError, match="^unknown long_term candidate code: 'ok'$"):
        run(long="ok")


def test_the_mapping_covers_every_code_the_policy_emits() -> None:
    """PH-RF10: over every fit, tilt, zone, valuation gate, profitability and dislocation input,
    build_policy_action emits exactly the mapped codes, so a new policy code fails here first. PD D-2."""
    fundamentals = {"growth": {"revenue_yoy": 0.12}, "profitability": {"net_margin": 0.2, "fcf_positive": True},
                    "health": {"net_cash_positive": True, "debt_to_equity": 0.5},
                    "burn_metrics": {"status": "not_applicable"}}
    deep = {"dip_depth": {"from_52w_high": -0.45, "severity": "deep"}}
    fits = ("ok", "caution", "avoid", "unknown")
    seen: dict[str, set[str]] = {"mid_term": set(), "long_term": set()}
    for mid, long, tilt, zone, gate, unprofitable, dislocation in product(
            fits, fits, ("bullish", "neutral", "bearish", None),
            ("strong_buy", "accumulate", "hold_bullish", "hold_neutral", "reduce", "undetermined", None),
            ("attractive", "neutral", "headwind", "unknown", "not_applicable"), (False, True), (False, True)):
        verdict = {"tilt": tilt, "horizon_fit": {"mid_term": mid, "long_term": long},
                   "decomposed": {"business_quality": "strong"}}
        zones = {"current_zone": zone, "valuation_assessment": {"gate": gate, "is_unprofitable": unprofitable}}
        out = build_policy_action(verdict=verdict, action_zones=zones, decomposed=verdict["decomposed"],
                                  dip_assessment=deep if dislocation else None,
                                  fundamentals_summary=fundamentals if dislocation else {})
        for horizon in seen:
            seen[horizon].add(out[horizon])

    assert seen == {h: set(MAPS[h]) for h in MAPS}


# --- GC-R2 analyze columns [D-5a, D-5b] ---------------------------------------------------------------


@pytest.mark.parametrize("gate_id", list(OUTCOME))
def test_gate_outcome_per_horizon_analyze(gate_id: str) -> None:
    """GC-R2 (analyze columns): one fired gate gives its registry outcome at each horizon it applies to,
    with its analyze class and effect; long-term is gated like mid-term except near_52w_high. PD D-5a, D-5b."""
    result = run("buy", "accumulate", [gate_id])
    mid_outcome, long_outcome = OUTCOME[gate_id]
    mid, long = result["new_money"], result["long_term"]["new_money"]

    assert tuple(OUTCOME) == ANALYZE_GATE_IDS
    assert (mid["action"], mid["decided_by"], ids(mid["blocked_by"])) == (mid_outcome, gate_id, [gate_id])
    assert (mid["blocked_by"][0]["class"], mid["blocked_by"][0]["effect"]) == CLASS_EFFECT[gate_id]
    if long_outcome == NOT_APPLICABLE:
        assert (long["action"], long["decided_by"], long["blocked_by"]) == ("buy", None, [])
    else:
        assert (long["action"], long["decided_by"], ids(long["blocked_by"])) == (long_outcome, gate_id, [gate_id])


# --- GC-R3, rewritten (§0 #1) [D-5] ------------------------------------------------------------------


@pytest.mark.parametrize("horizon", ["mid_term", "long_term"])
def test_resolution_table(horizon: str) -> None:
    """GC-R3: every (candidate code, applicable gate) cell of the analyze resolution: most_restrictive of the
    mapped candidate and the gate's outcome; decided_by is the gate iff it tightened the candidate. PD D-5."""
    index = 0 if horizon == "mid_term" else 1
    for code, gate_id in product(MAPS[horizon], OUTCOME):
        outcome = OUTCOME[gate_id][index]
        if outcome == NOT_APPLICABLE:
            continue
        codes = (code, "insufficient_data") if horizon == "mid_term" else ("insufficient_data", code)
        result = run(*codes, gates=[gate_id])
        decision = result["new_money"] if horizon == "mid_term" else result["long_term"]["new_money"]
        expected = RESOLUTION[(MAPS[horizon][code], outcome)]
        assert decision["action"] == expected, (code, gate_id)
        assert decision["decided_by"] == (None if expected == MAPS[horizon][code] else gate_id), (code, gate_id)


PLAN_TABLE = {  # plan §4 "Combination rule" table, analyze column: (mid code, gates) -> new_money
    "data_on_avoid": ("avoid", ["stale_data"], "avoid"),
    "data_on_insufficient": ("insufficient_data", ["data_quality_critical"], "insufficient_data"),
    "support_on_wait": ("hold", ["unsupported_instrument"], "watch"),
    "support_on_insufficient": ("insufficient_data", ["unsupported_currency"], "watch"),
    "data_and_support": ("hold_or_add", ["stale_data", "unsupported_instrument"], "watch"),
    "risk_metrics_outage": ("hold_or_add", ["data_quality_critical", "liquidity_missing"], WFD),
    "fundamentals_failed_empty_risk": ("insufficient_data", ["data_quality_critical", "liquidity_missing"],
                                       "insufficient_data"),
    "stale_and_knife": ("hold_or_add", ["stale_data", "falling_knife"], "avoid"),
    "case_g": ("avoid", ["falling_knife", "unsupported_currency"], "avoid"),
    "event_on_wait": ("wait_for_entry", ["earnings_blackout"], "wait"),
    "knife_on_wait": ("wait_for_entry", ["falling_knife"], "avoid"),
}


PLAN_DECIDED_BY = {"data_and_support": "unsupported_instrument", "stale_and_knife": "falling_knife"}


@pytest.mark.parametrize("case", PLAN_TABLE)
def test_plan_combination_table(case: str) -> None:
    """GC-R3: the analyze column of plan §4's combination table; where two gates fire, decided_by is the
    highest-ranked gate whose outcome is the shown action (plan §4:232). PD D-5."""
    code, gates, expected = PLAN_TABLE[case]
    new_money = run(code, gates=gates)["new_money"]

    assert new_money["action"] == expected
    if case in PLAN_DECIDED_BY:
        assert new_money["decided_by"] == PLAN_DECIDED_BY[case]


# --- GC-R4, rewritten [D-5]: analyze most-restrictive vs the card's rank-first, R2.6(e) ----------------

SHARED = [g for g in ANALYZE_GATE_IDS if GATES[g].outcome["swing"] != NOT_APPLICABLE]
UNREACHABLE = {frozenset(pair) for pair in product(("weak_liquidity", "liquidity_missing"),
                                                   ("unsupported_instrument", "unsupported_currency"))}
UNREACHABLE.add(frozenset(("weak_liquidity", "liquidity_missing")))
DATA = ("stale_data", "freshness_unverifiable", "data_quality_critical")
PINNED_DIFFERENCES = {  # plan §4 R2.6(e)
    *(frozenset((d, g)) for d in DATA for g in ("falling_knife", "weak_liquidity")),  # avoid vs wait_for_data
    *(frozenset((d, g)) for d in DATA for g in ("unsupported_instrument", "unsupported_currency")),  # watch vs wfd
    frozenset({"liquidity_missing"}),  # wait_for_data vs avoid (D-5c)
    frozenset({"liquidity_missing", "earnings_blackout"}), frozenset({"liquidity_missing", "earnings_unverifiable"}),
}


def card_rank_first(gate_ids: tuple[str, ...]) -> str:
    top = min(gate_ids, key=lambda g: (gate_rank(g, "swing"), list(GATES).index(g)))
    outcome = GATES[top].outcome["swing"]
    return "wait" if outcome == "watch" and GATES[top].gate_class == "event" else outcome  # R2.6(d): event watch


def test_shared_precedence_pairs() -> None:
    """GC-R4: over every single gate and reachable pair shared by analyze and the card, analyze is
    order-independent and discloses in rank order, and it differs from the card's rank-first outcome exactly
    on the pinned R2.6(e) sets. PD D-5."""
    assert SHARED == ["stale_data", "freshness_unverifiable", "data_quality_critical", "liquidity_missing",
                      "falling_knife", "weak_liquidity", "unsupported_instrument", "unsupported_currency",
                      "earnings_blackout", "earnings_unverifiable"]
    differ = set()
    for size in (1, 2):
        for gates in combinations(SHARED, size):
            if frozenset(gates) in UNREACHABLE:
                continue
            analyze = run("hold_or_add", gates=gates)["new_money"]
            assert run("hold_or_add", gates=gates[::-1])["new_money"] == analyze, gates
            assert ids(analyze["blocked_by"]) == [g for g in ANALYZE_GATE_IDS if g in gates], gates
            if analyze["action"] != card_rank_first(gates):
                differ.add(frozenset(gates))

    assert differ == PINNED_DIFFERENCES


# --- GC-R5, rewritten [D-5] ----------------------------------------------------------------------------


def test_a_gate_tightens_any_base_and_never_loosens() -> None:
    """GC-R5: for every mid-term code and pair of analyze gates, adding a gate never loosens the action, and a
    gate tightens a non-admitting candidate (support on wait gives watch). PD D-5 (plan §4:228-230)."""
    for code, (g1, g2) in product(MID_MAP, combinations([*ANALYZE_GATE_IDS, None], 2)):
        first = [g for g in (g1,) if g]
        before = run(code, gates=first)["new_money"]["action"]
        after = run(code, gates=[*first, *(g for g in (g2,) if g)])["new_money"]["action"]
        assert ORDER.index(after) >= ORDER.index(before) >= ORDER.index(MID_MAP[code]), (code, g1, g2)
        assert after == max([MID_MAP[code], *(OUTCOME[g][0] for g in (g1, g2) if g)], key=ORDER.index), (code, g1, g2)
