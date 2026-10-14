"""WP2.2: holder guidance, the thesis and price-extension caps and the core rule (plan §3, §4).

Annex: docs/plans/step2-v2-research/policy_holder.md, "Corrected proposal" §3-§10, as amended by plan §0 #7
(no long-term holder read in 2a: the long-term if_held columns are dropped), #8, #17 (near_52w_high), #24
(gates leave if_held.action and basis alone; add_allowed follows the gated new money) and #25. Unit grids call
decide() through dm.decide_on over the real build_policy_action's codes.
"""

import inspect
from itertools import product
from typing import Any

import pytest

from stock_analysis.tools.analyze import decision as dc
from stock_analysis.tools.analyze.action_zones import (
    _apply_valuation_gate_to_zone,
    apply_dip_gates_to_action_zones,
)
from stock_analysis.tools.analyze.dislocation_framework import classify_dislocation_opportunity
from stock_analysis.tools.analyze.executive_summary import build_policy_action
from stock_analysis.tools.analyze.gates import (
    ANALYZE_GATE_IDS,
    BUSINESS_BROKEN,
    GATES,
    RUNWAY_CRITICAL,
    THESIS_CLEARS_WHEN,
    check_near_52w_high,
    check_thesis_broken,
    clears_when_for,
)
from tests.support import decision_matrix as dm

BB, RC = BUSINESS_BROKEN, RUNWAY_CRITICAL
ENTRY = ("buy", "starter")
DEEP = {"from_52w_high": -0.45, "severity": "deep"}
DISLOCATION_FUNDAMENTALS = {"growth": {"revenue_yoy": 0.12}, "profitability": {"net_margin": 0.2, "fcf_positive": True},
                            "health": {"net_cash_positive": True, "debt_to_equity": 0.5},
                            "burn_metrics": {"status": "not_applicable"}}
MID_FITS = ("ok", "caution", "avoid", "unknown")
TILTS = ("bullish", "neutral", "bearish", None)
ZONES = ("accumulate", "hold_bullish", "reduce")
THESES: tuple[tuple[str, ...], ...] = ((), (BB,), (RC,), (BB, RC))


def candidate(mid: str, tilt: str | None, zone: str, dislocation: bool, knife: bool) -> dict[str, str]:
    """The real build_policy_action's codes (quality strong, valuation attractive, long fit ok)."""
    verdict = {"tilt": tilt, "horizon_fit": {"mid_term": mid, "long_term": "ok"},
               "decomposed": {"business_quality": "strong"}}
    zones = {"current_zone": zone, "valuation_assessment": {"gate": "attractive", "is_unprofitable": False}}
    dip: dict[str, Any] = {"dip_depth": DEEP} if dislocation else {}
    if knife:
        dip["dip_classification"] = {"type": "falling_knife"}
    action = build_policy_action(verdict=verdict, action_zones=zones, decomposed=verdict["decomposed"],
                                 dip_assessment=dip or None, fundamentals_summary=DISLOCATION_FUNDAMENTALS)
    return {"mid_term": action["mid_term"], "long_term": action["long_term"]}


def blockers(gate_ids: list[str], thesis: tuple[str, ...]) -> list[dict[str, Any]]:
    """Fired blockers in disclosure order; the thesis gate fires iff the thesis evidence is broken."""
    fired = [{"id": g, "reason": "r"} for g in gate_ids if g != "thesis_broken"]
    if thesis:
        fired.append({"id": "thesis_broken", "reason": "r", "codes": list(thesis)})
    return sorted(fired, key=lambda b: ANALYZE_GATE_IDS.index(b["id"]))


def run_decide(cand: dict[str, str], gate_ids: list[str] | tuple[str, ...] = (), thesis: tuple[str, ...] = (),
               dislocation: bool = False, tilt: str | None = "neutral") -> dict[str, Any]:
    return dm.decide_on(cand, blockers(list(gate_ids), thesis), thesis=thesis, dislocation=dislocation, tilt=tilt)


def grid() -> Any:
    return product(MID_FITS, TILTS, ZONES, (False, True))


def held(decision: dict[str, Any]) -> tuple[str, str]:
    return decision["if_held"]["action"], decision["if_held"]["basis"]


def tightness(action: str) -> int:
    return dc.NEW_MONEY_ACTIONS.index(action)


# --- PH-RF2: the knife (PH-RF1's grid lands with the knife-override removal) ----------------------


def test_knife_never_loosens_new_money() -> None:
    """PH-RF2: adding the knife never loosens new money at either horizon. RF."""
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        plain = run_decide(candidate(mid, tilt, zone, disl, knife=False), [], thesis, disl, tilt)
        knifed = run_decide(candidate(mid, tilt, zone, disl, knife=True), ["falling_knife"], thesis, disl, tilt)
        for decision in (lambda d: d["new_money"], lambda d: d["long_term"]["new_money"]):
            assert tightness(decision(knifed)["action"]) >= tightness(decision(plain)["action"]), (mid, tilt, zone)


# --- PH-RF4 [D-1a, D-1b, D-1d]: the mid-term if_held table ----------------------------------------------

MID_TABLE = [  # (pre-gate mid code, tilt, dislocation) -> (action, basis); annex §6 truth table, plan §3 rule order
    ("buy", "bullish", False, ("hold", "candidate_admits_new_money")),
    ("buy", "bullish", True, ("hold", "candidate_admits_new_money")),
    ("hold_or_add", "bullish", False, ("hold", "candidate_admits_new_money")),
    ("hold_or_add", "bullish", True, ("hold", "candidate_admits_new_money")),
    ("speculative_small_position", "bullish", False, ("hold", "candidate_admits_new_money")),
    *((("speculative_small_position", t, True, ("hold", "candidate_admits_new_money"))) for t in TILTS),
    ("hold", "neutral", False, ("hold", "no_reduction_evidence")),
    ("hold", "neutral", True, ("hold", "price_broken_more_than_business")),
    ("hold_or_reduce", "bearish", False, ("reduce", "tilt_bearish")),
    ("hold_or_reduce", "bearish", True, ("hold", "price_broken_more_than_business")),
    ("wait_for_entry", "neutral", False, ("hold", "no_reduction_evidence")),
    ("wait_for_entry", "bearish", False, ("reduce", "tilt_bearish")),
    ("avoid", "bullish", False, ("hold", "no_reduction_evidence")),
    ("avoid", "neutral", False, ("hold", "no_reduction_evidence")),
    ("avoid", "bearish", False, ("reduce", "tilt_bearish")),
    ("avoid", None, False, ("insufficient_data", "tilt_missing")),
    *((("avoid", t, True, ("hold", "price_broken_more_than_business"))) for t in TILTS),
    *((("insufficient_data", t, False, ("insufficient_data", "candidate_insufficient")))
      for t in ("bullish", "neutral", "bearish")),
    ("insufficient_data", None, False, ("insufficient_data", "tilt_missing")),
    *((("insufficient_data", t, True, ("hold", "price_broken_more_than_business"))) for t in TILTS),
]


def if_held(code: str, tilt: str | None, dislocation: bool, thesis: tuple[str, ...] = ()) -> tuple[str, str]:
    got = dc.if_held(thesis=thesis, candidate_action=dc.map_candidate("mid_term", code), dislocation=dislocation,
                     tilt=tilt)
    return got["action"], got["basis"]


@pytest.mark.parametrize(("code", "tilt", "dislocation", "expected"), MID_TABLE)
def test_mid_if_held_row(code: str, tilt: str | None, dislocation: bool, expected: tuple[str, str]) -> None:
    """PH-RF4: each row of the mid-term if_held table (first match: admits, dislocation, tilt missing,
    insufficient candidate, bearish tilt, otherwise hold). PD D-1a, D-1b, D-1d."""
    assert if_held(code, tilt, dislocation) == expected


@pytest.mark.parametrize("thesis", [(BB,), (RC,), (BB, RC)])
def test_thesis_rows(thesis: tuple[str, ...]) -> None:
    """PH-RF4: a broken thesis gives review_exit on every row; the runway basis wins when both fire. PD D-1c."""
    for code, tilt, dislocation, _ in MID_TABLE:
        assert if_held(code, tilt, dislocation, thesis) == ("review_exit", RC if RC in thesis else BB)


def test_table_covers_every_reachable_candidate() -> None:
    """PH-RF4: every (mid code, tilt, dislocation) the real policy produces over the grid is a table row.
    PD D-1a, D-1b, D-1d."""
    rows = {(code, tilt, disl) for code, tilt, disl, _ in MID_TABLE}
    reached = {(candidate(mid, tilt, zone, disl, knife=False)["mid_term"], tilt, disl)
               for mid, tilt, zone, disl in grid()}

    assert reached <= rows


# --- PH-RF6 [D-1a], PH-RF7 ------------------------------------------------------------------------------


def test_reduce_requires_an_evidence_basis() -> None:
    """PH-RF6: reduce or review_exit iff the basis is an evidence basis; every action and basis is in the
    plan §2 vocabularies. PD D-1a."""
    assert dc.IF_HELD_ACTIONS == ("hold", "reduce", "review_exit", "insufficient_data")
    assert dc.EVIDENCE_BASES == ("runway_critical", "business_broken", "tilt_bearish")
    assert dc.NON_EVIDENCE_BASES == ("candidate_admits_new_money", "price_broken_more_than_business",
                                     "no_reduction_evidence", "tilt_missing", "candidate_insufficient")
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        got = run_decide(candidate(mid, tilt, zone, disl, knife=False), [], thesis, disl, tilt)["if_held"]
        assert got["action"] in dc.IF_HELD_ACTIONS
        assert got["basis"] in dc.EVIDENCE_BASES + dc.NON_EVIDENCE_BASES
        assert (got["action"] in ("reduce", "review_exit")) is (got["basis"] in dc.EVIDENCE_BASES)


def test_same_horizon_coherence() -> None:
    """PH-RF7: reduce or review_exit never stands next to new_money buy or starter, and add_allowed is exactly
    new_money in {buy, starter}, under every single gate. RF."""
    gate_sets = [[], *([g] for g in ANALYZE_GATE_IDS if g != "thesis_broken")]
    for (mid, tilt, zone, disl), thesis, gate_ids in product(grid(), THESES, gate_sets):
        decision = run_decide(candidate(mid, tilt, zone, disl, knife=False), gate_ids, thesis, disl, tilt)
        new_money, holder = decision["new_money"]["action"], decision["if_held"]
        if holder["action"] in ("reduce", "review_exit"):
            assert new_money not in ENTRY, (mid, tilt, zone, disl, thesis, gate_ids)
        assert holder["add_allowed"] is (new_money in ENTRY)


# --- PH-RF8 = GC-R6 (§0 #24): gates never change if_held ---------------------------------------------------


@pytest.mark.parametrize("gate_id", ANALYZE_GATE_IDS)
def test_gates_never_change_if_held(gate_id: str) -> None:
    """PH-RF8 / GC-R6: with the evidence fixed, firing any analyze gate alone or with any other leaves if_held's
    action and basis unchanged; add_allowed follows the gated new_money (R2.6(h)). RF."""
    others = [(), *((g,) for g in ANALYZE_GATE_IDS if g != gate_id)]
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        cand = candidate(mid, tilt, zone, disl, knife=False)
        base = held(run_decide(cand, [], thesis, disl, tilt))
        for other in others:
            gated = run_decide(cand, [gate_id, *other], thesis, disl, tilt)
            assert held(gated) == base, (gate_id, other, mid, tilt, zone, disl, thesis)
            assert gated["if_held"]["add_allowed"] is (gated["new_money"]["action"] in ENTRY)


def test_if_held_reads_only_its_evidence() -> None:
    """PH-RF8: if_held's inputs are the thesis, the mapped candidate, the dislocation flag and tilt: never gates,
    zone, dip or support (plan §3). G."""
    assert set(inspect.signature(dc.if_held).parameters) == {"thesis", "candidate_action", "dislocation", "tilt"}


# --- PH-RF11 [D-4]: near_52w_high, mid-term new money only (Codex boundary 1) ---------------------------


def test_near_52w_high_never_reaches_long_term_core_or_if_held() -> None:
    """PH-RF11: with near_52w_high fired, mid-term new money is never buy or starter, while if_held, the whole
    long_term block and core equal the gate-free run (Codex boundary 1). PD D-4."""
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        cand = candidate(mid, tilt, zone, disl, knife=False)
        capped = run_decide(cand, ["near_52w_high"], thesis, disl, tilt)
        free = run_decide(cand, [], thesis, disl, tilt)
        assert capped["new_money"]["action"] not in ENTRY
        assert held(capped) == held(free)
        assert capped["long_term"] == free["long_term"]
        assert dc.core_action("ok", capped["long_term"]["new_money"]["action"]) == dc.core_action(
            "ok", free["long_term"]["new_money"]["action"])


def test_near_52w_high_status() -> None:
    """PH-RF11: the gate fires on the reduce zone only, and is unevaluated with no zone or an undetermined one.
    PD D-4."""
    assert check_near_52w_high("reduce") == ("fired", {"id": "near_52w_high",
                                                       "reason": "price in the near-52-week-high band"})
    for zone in ("strong_buy", "accumulate", "hold_bullish", "hold_neutral"):
        assert check_near_52w_high(zone) == ("clear", None)
    for zone in (None, "undetermined"):
        assert check_near_52w_high(zone) == ("unevaluated", None)


@pytest.mark.parametrize("zone", ["strong_buy", "accumulate", "hold_bullish", "hold_neutral", "reduce",
                                  "undetermined", None])
def test_zone_caps_never_move_the_gate_zones(zone: str | None) -> None:
    """G-ZONE: the dip and valuation zone caps never move a zone into or out of {reduce, undetermined, None}, so
    current_zone reads the uncapped classification for near_52w_high until WP2.3 emits price_zone. G."""
    special = (None, "undetermined", "reduce")
    for dip_type, risk in product(("falling_knife", "extended_decline", None), ("extreme", "high", None)):
        out = apply_dip_gates_to_action_zones({"current_zone": zone}, {"dip_classification": {"type": dip_type}},
                                              {"classification": risk})["current_zone"]
        assert (out in special) is (zone in special) and (zone not in special or out == zone)
    for gate in ("headwind", "attractive", "neutral", "unknown", "not_applicable"):
        out = _apply_valuation_gate_to_zone(zone, gate)[0]
        assert (out in special) is (zone in special) and (zone not in special or out == zone)


# --- PH-RF12, RF12b [D-1c]: the thesis cap ---------------------------------------------------------------


def test_thesis_caps_both_horizons() -> None:
    """PH-RF12: a broken thesis gives new_money avoid at both horizons and if_held review_exit. PD D-1c."""
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES[1:]):
        decision = run_decide(candidate(mid, tilt, zone, disl, knife=False), [], thesis, disl, tilt)
        assert (decision["new_money"]["action"], decision["long_term"]["new_money"]["action"],
                decision["if_held"]["action"]) == ("avoid", "avoid", "review_exit")


def test_thesis_status() -> None:
    """PH-RF12: thesis_broken is not_applicable for an unsupported instrument, unevaluated with fundamentals
    missing, and otherwise fires with the failing components as codes (predicate order). PD D-1c."""
    broken = {"business_status": "broken", "balance_status": "unsafe"}

    assert check_thesis_broken("missing", broken) == ("unevaluated", None)
    assert check_thesis_broken("not_applicable", broken) == ("not_applicable", None)
    assert check_thesis_broken("available", {"business_status": "mixed", "balance_status": "watch"}) == ("clear", None)
    status, blocker = check_thesis_broken("not_meaningful", broken)
    assert (status, blocker["codes"]) == ("fired", [BB, RC])


@pytest.mark.parametrize(("quality", "status"), [("poor", "fired"), ("strong", "clear")])
def test_codex_case_clears(quality: str, status: str) -> None:
    """PH-RF12b: Codex's case: a broken business with a safe balance sheet and no runway figure fires on the
    business component only, names no runway clause, and clears once the business is intact. PD D-1c."""
    signals = classify_dislocation_opportunity(
        verdict={"decomposed": {"business_quality": quality}}, action_zones={},
        fundamentals_summary={"health": {"net_cash_positive": True, "debt_to_equity": 0.5}, "burn_metrics": {}},
        dip_assessment=None)
    got, blocker = check_thesis_broken("available", signals)

    assert signals["balance_status"] == "safe"
    assert got == status
    if blocker:
        assert blocker["codes"] == [BB]
        assert clears_when_for(blocker) == THESIS_CLEARS_WHEN[BB]
        assert "runway" not in clears_when_for(blocker) and "runway" not in blocker["reason"]


def test_clears_when_names_only_failing_components() -> None:
    """PH-RF12b: clears_when lists the failing components only; both together are the registry's string.
    PD D-1c."""
    both = check_thesis_broken("available", {"business_status": "broken", "balance_status": "unsafe"})[1]
    runway = check_thesis_broken("available", {"business_status": "mixed", "balance_status": "unsafe"})[1]

    assert clears_when_for(both) == GATES["thesis_broken"].clears_when
    assert clears_when_for(runway) == THESIS_CLEARS_WHEN[RC] == "cash runway no longer below 4 quarters"


# --- PH-RF13 [D-1b], PH-RF14 --------------------------------------------------------------------------------


@pytest.mark.parametrize(("mid", "code", "basis"), [("ok", "hold_or_reduce", "price_broken_more_than_business"),
                                                    ("caution", "speculative_small_position",
                                                     "candidate_admits_new_money"),
                                                    ("avoid", "avoid", "price_broken_more_than_business")])
def test_dislocation_holder_holds(mid: str, code: str, basis: str) -> None:
    """PH-RF13: a dislocation candidate holder holds with a bearish tilt at every mid fit; the caution row's
    basis is candidate_admits_new_money (plan §3 rule 3 before rule 4; the annex text says price_broken).
    PD D-1b."""
    cand = candidate(mid, "bearish", "accumulate", True, knife=False)

    assert cand["mid_term"] == code
    assert if_held(cand["mid_term"], "bearish", True) == ("hold", basis)


def test_mid_if_held_is_monotone_in_mid_fit() -> None:
    """PH-RF13: going ok -> caution -> avoid never gives a softer if_held with the other evidence fixed. PD D-1b."""
    order = {"hold": 0, "insufficient_data": 1, "reduce": 2, "review_exit": 3}
    for tilt, zone, disl, thesis in product(TILTS, ZONES, (False, True), THESES):
        seq = [order[if_held(candidate(m, tilt, zone, disl, False)["mid_term"], tilt, disl, thesis)[0]]
               for m in ("ok", "caution", "avoid")]
        assert seq == sorted(seq), (tilt, zone, disl, thesis, seq)


def test_unknown_mid_fit_with_bearish_tilt() -> None:
    """PH-RF14: an unknown mid fit with a bearish tilt gives insufficient_data (candidate_insufficient), not
    reduce: rule 6 precedes rule 7. RF."""
    cand = candidate("unknown", "bearish", "accumulate", False, knife=False)

    assert cand["mid_term"] == "insufficient_data"
    assert if_held(cand["mid_term"], "bearish", False) == ("insufficient_data", "candidate_insufficient")


# --- PH-RF18 [D-3]: core's own cap ---------------------------------------------------------------------------


def test_core_own_cap() -> None:
    """PH-RF18: core = most_restrictive(CORE[long fit], long_term.new_money): caution + speculative_only gives
    wait, ok + headwind (hold_size_conservatively) wait (C11), and ok + null tilt + neutral valuation
    insufficient_data (A2). PD D-3."""
    verdict = {"tilt": None, "horizon_fit": {"mid_term": "ok", "long_term": "ok"}, "decomposed": {}}
    zones = {"current_zone": "hold_bullish", "valuation_assessment": {"gate": "neutral", "is_unprofitable": False}}
    a2 = build_policy_action(verdict=verdict, action_zones=zones, decomposed={}, fundamentals_summary={})

    assert dc.CORE == {"ok": "buy", "caution": "wait", "unknown": "insufficient_data", "avoid": "avoid"}
    assert dc.core_action("caution", dc.map_candidate("long_term", "speculative_only")) == "wait"
    assert dc.core_action("ok", dc.map_candidate("long_term", "hold_size_conservatively")) == "wait"
    assert a2["long_term"] == "insufficient_data"
    assert dc.core_action("ok", dc.map_candidate("long_term", a2["long_term"])) == "insufficient_data"
    for fit, action in product(dc.CORE, dc.NEW_MONEY_ACTIONS):
        assert dc.core_action(fit, action) == max((dc.CORE[fit], action), key=tightness)
