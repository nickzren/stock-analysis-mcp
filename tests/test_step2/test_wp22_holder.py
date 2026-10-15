"""WP2.2: holder guidance, the thesis and price-extension caps and the core rule (plan §3, §4).

Annex: docs/plans/step2-v2-research/policy_holder.md, "Corrected proposal" §3-§10, as amended by plan §0 #7
(no long-term holder read in 2a: the long-term if_held columns are dropped), #8, #17 (near_52w_high), #24
(gates leave if_held.action and basis alone; add_allowed follows the gated new money) and #25. Unit grids call
decide() through dm.decide_on; analyze runs go through tests/support/decision_matrix.py, which pins the clock.
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
UNEVALUABLE = ("missing_runway", "falling_knife", "thesis_broken", "earnings_blackout", "near_52w_high")  # R2.6(f)
A7_CAUTION = dm.Scenario("PH A7 caution", dm.at_price(dm.STRONG, 118.0), dm.QUALITY, dm.EXTREME)
E_INTACT = dm.Scenario("PH-RF12b intact twin", dm.STRONG,
                       {**dm.POOR_BUT_BULLISH, "cash_flow": {"rules": {"positive_fcf": {"triggered": True}}},
                        "financial_health": {"debt_to_equity": 0.3, "rules": {"low_debt": {"triggered": True}}}},
                       dm.MODERATE)


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


def fire(gate_ids: list[str]) -> list[dict[str, Any]]:
    """Fired blockers in disclosure order, whatever the evidence: the thesis gate names the business component."""
    return [{"id": g, "reason": "r", **({"codes": [BB]} if g == "thesis_broken" else {})}
            for g in sorted(gate_ids, key=ANALYZE_GATE_IDS.index)]


def run_decide(cand: dict[str, str], gate_ids: list[str] | tuple[str, ...] = (), thesis: tuple[str, ...] = (),
               dislocation: bool = False, tilt: str | None = "neutral") -> dict[str, Any]:
    return dm.decide_on(cand, blockers(list(gate_ids), thesis), thesis=thesis, dislocation=dislocation, tilt=tilt)


def grid() -> Any:
    return product(MID_FITS, TILTS, ZONES, (False, True))


def held(decision: dict[str, Any]) -> tuple[str, str]:
    return decision["if_held"]["action"], decision["if_held"]["basis"]


def ids(blocked: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blocked]


def tightness(action: str) -> int:
    return dc.NEW_MONEY_ACTIONS.index(action)


# --- PH-RF1, RF2, RF3: the knife ---------------------------------------------------------------------


def test_knife_never_changes_if_held() -> None:
    """PH-RF1: over the policy grid and every thesis, the knife (gate on, override gone) leaves if_held's action
    and basis unchanged (mid-term only, plan §0 #7). RF."""
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        plain = run_decide(candidate(mid, tilt, zone, disl, knife=False), [], thesis, disl, tilt)
        knifed = run_decide(candidate(mid, tilt, zone, disl, knife=True), ["falling_knife"], thesis, disl, tilt)
        assert held(plain) == held(knifed), (mid, tilt, zone, disl, thesis)


async def test_knife_never_changes_if_held_through_the_pipeline() -> None:
    """PH-RF1: Codex's probe A (broken, bearish) and B (A + knife) give the same if_held. RF."""
    a, b = await dm.run("A_broken_bearish"), await dm.run("B_broken_bearish_knife")

    assert held(a["decision_card"]) == held(b["decision_card"]) == ("review_exit", "business_broken")


def test_knife_never_loosens_new_money() -> None:
    """PH-RF2: adding the knife never loosens new money at either horizon. RF."""
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        plain = run_decide(candidate(mid, tilt, zone, disl, knife=False), [], thesis, disl, tilt)
        knifed = run_decide(candidate(mid, tilt, zone, disl, knife=True), ["falling_knife"], thesis, disl, tilt)
        for decision in (lambda d: d["new_money"], lambda d: d["long_term"]["new_money"]):
            assert tightness(decision(knifed)["action"]) >= tightness(decision(plain)["action"]), (mid, tilt, zone)


async def test_codex_probe_a_and_b() -> None:
    """PH-RF2 and PH-RF3 (mid decision clauses; the candidate clause is WP2.3): A and B both give new_money avoid
    and if_held review_exit (business_broken); B's blocked_by == [falling_knife, thesis_broken]. PD D-1c."""
    a, b = await dm.run("A_broken_bearish"), await dm.run("B_broken_bearish_knife")

    for result in (a, b):
        card = result["decision_card"]
        assert (card["new_money"]["action"], *held(card)) == ("avoid", "review_exit", "business_broken")
    assert ids(a["decision_card"]["new_money"]["blocked_by"]) == ["thesis_broken"]
    assert ids(b["decision_card"]["new_money"]["blocked_by"]) == ["falling_knife", "thesis_broken"]


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


@pytest.mark.parametrize(("name", "expected"), [("ETF_strong", ("hold", "candidate_admits_new_money")),
                                                ("ETF_bearish", ("reduce", "tilt_bearish"))])
async def test_unsupported_reads_the_uncapped_candidate(name: str, expected: tuple[str, str]) -> None:
    """PH-RF4: an unsupported instrument's if_held reads the uncapped candidate, never the support cap's watch
    (ETF bullish hold_or_add -> hold; ETF bearish avoid -> reduce). PD D-1a/b/d."""
    assert held((await dm.run(name))["decision_card"]) == expected


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


async def test_case_d_holds() -> None:
    """PH-RF6: case D (quality, attractive, poor chart): new_money avoid, if_held hold (no_reduction_evidence):
    a poor setup alone never makes a holder reduce. PD D-1a."""
    card = (await dm.run("D_quality_attractive_poor_chart"))["decision_card"]

    assert (card["new_money"]["action"], *held(card)) == ("avoid", "hold", "no_reduction_evidence")


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


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_coherence_through_the_matrix(name: str) -> None:
    """PH-RF7: coherence on every matrix scenario and tier; the thesis gate fires exactly when the dislocation
    framework's thesis is broken and if_held is review_exit (D-1c). RF."""
    result = await dm.run(name)
    card = result["decision_card"]
    holder, new_money = card["if_held"], card["new_money"]["action"]

    if holder["action"] in ("reduce", "review_exit"):
        assert new_money not in ENTRY
    assert holder["add_allowed"] is (new_money in ENTRY)
    thesis_status = (result["dislocation_framework"].get("thesis_integrity") or {}).get("status")
    assert ("thesis_broken" in dm.gate_ids(result)) is (thesis_status == "broken") is (holder["action"] == "review_exit")
    for tier, shown in dm.tiers(result).items():
        assert shown["decision_card"]["if_held"] == holder, tier


# --- PH-RF8 = GC-R6 (§0 #24): gates never change if_held ---------------------------------------------------


@pytest.mark.parametrize("gate_id", ANALYZE_GATE_IDS)
def test_gates_never_change_if_held(gate_id: str) -> None:
    """PH-RF8 / GC-R6: with the evidence fixed (θ at every value, () included), firing any analyze gate alone or
    with any other, with or without every other gate that can be unevaluated listed, leaves if_held's action and
    basis equal to the gate-free run; the thesis gate fires with codes [business_broken] whatever θ is. With the
    thesis gate fired iff θ is broken, coded θ (up to three gates), it leaves them equal to the run with no other
    gate. add_allowed follows the gated new_money on every run (R2.6(h)). RF."""
    others = [(), *((g,) for g in ANALYZE_GATE_IDS if g != gate_id)]
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        cand = candidate(mid, tilt, zone, disl, knife=False)
        base = held(dm.decide_on(cand, [], thesis=thesis, dislocation=disl, tilt=tilt))
        consistent = held(run_decide(cand, [], thesis, disl, tilt))
        for other in others:
            fired = [gate_id, *other]
            gated = dm.decide_on(cand, fire(fired), thesis=thesis, dislocation=disl, tilt=tilt)
            listed = dm.decide_on(cand, fire(fired), unevaluated=frozenset(UNEVALUABLE).difference(fired),
                                  thesis=thesis, dislocation=disl, tilt=tilt)
            consistent_run = run_decide(cand, fired, thesis, disl, tilt)
            assert held(gated) == held(listed) == base, (gate_id, other, mid, tilt, zone, disl, thesis)
            assert held(consistent_run) == consistent, (gate_id, other, mid, tilt, zone, disl, thesis)
            for decision in (gated, listed, consistent_run):
                assert decision["if_held"]["add_allowed"] is (decision["new_money"]["action"] in ENTRY)


@pytest.mark.parametrize("gate_id", UNEVALUABLE)
def test_unevaluated_gates_never_change_if_held(gate_id: str) -> None:
    """PH-RF8 / GC-R6 for the unevaluated status (R2.6(f), (h)): with the evidence fixed, listing any analyze gate
    that can be unevaluated, alone or with any other, leaves if_held's action and basis equal to the run with none
    listed; add_allowed follows new_money. RF."""
    others = [(), *((g,) for g in UNEVALUABLE if g != gate_id)]
    for (mid, tilt, zone, disl), thesis in product(grid(), THESES):
        cand = candidate(mid, tilt, zone, disl, knife=False)
        base = held(dm.decide_on(cand, [], thesis=thesis, dislocation=disl, tilt=tilt))
        for other in others:
            listed = dm.decide_on(cand, [], unevaluated=frozenset({gate_id, *other}), thesis=thesis,
                                  dislocation=disl, tilt=tilt)
            assert held(listed) == base, (gate_id, other, mid, tilt, zone, disl, thesis)
            assert listed["if_held"]["add_allowed"] is (listed["new_money"]["action"] in ENTRY)


@pytest.mark.parametrize(("name", "expected"), [("S3_plain_avoid", ("reduce", "tilt_bearish")),
                                                ("D_quality_attractive_poor_chart", ("hold", "no_reduction_evidence")),
                                                ("T3_plain_wait_weak_setup", ("hold", "no_reduction_evidence"))])
async def test_an_events_failure_never_changes_if_held(name: str, expected: tuple[str, str]) -> None:
    """PH-RF8 / GC-R6 through analyze: with the events tool failing (earnings_blackout unevaluated), S3, D and T3
    keep the if_held of the run with events. RF."""
    present, failed = (await dm.run(name))["decision_card"], (await dm.run(name, events=None))["decision_card"]

    assert held(present) == held(failed) == expected
    assert "earnings_blackout" in ids(failed["new_money"]["unevaluated"])


@pytest.mark.parametrize(("plain", "gated"), [("S1_hold_or_add_clean", "S4_earnings_gate_on_hold_or_add"),
                                              ("S1_hold_or_add_clean", "S9_weak_liquidity"),
                                              ("C1_headwind_hold_or_add", "C2_headwind_earnings_gate"),
                                              ("D_quality_attractive_poor_chart", "D2_quality_poor_chart_earnings_3d"),
                                              ("S3_plain_avoid", "S5_earnings_gate_on_avoid")])
async def test_gate_pairs_through_the_pipeline(plain: str, gated: str) -> None:
    """PH-RF8 / GC-R6 through analyze: a gated twin keeps its plain twin's if_held action and basis. RF."""
    p, g = (await dm.run(plain))["decision_card"], (await dm.run(gated))["decision_card"]

    assert held(p) == held(g)
    assert g["hard_gates"]["blocking"]
    assert g["if_held"]["add_allowed"] is (g["new_money"]["action"] in ENTRY)


def test_if_held_reads_only_its_evidence() -> None:
    """PH-RF8: if_held's inputs are the thesis, the mapped candidate, the dislocation flag and tilt: never gates,
    zone, dip or support (plan §3). RF."""
    assert set(inspect.signature(dc.if_held).parameters) == {"thesis", "candidate_action", "dislocation", "tilt"}


# --- PH-RF11 [D-4]: near_52w_high, mid-term new money only (Codex boundary 1) ---------------------------


@pytest.mark.parametrize(("scenario", "long_term", "core"), [
    (dm.SCENARIOS["A7_near_52w_high_bullish"], "buy", "buy"), (A7_CAUTION, "starter", "wait")], ids=["ok", "caution"])
async def test_a7_caps_mid_term_new_money_only(scenario: dm.Scenario, long_term: str, core: str) -> None:
    """PH-RF11: A7 (reduce zone, bullish; ok fit hold_or_add, caution speculative_small_position) gives new_money
    wait [near_52w_high], if_held hold (candidate_admits_new_money) with add_allowed false, and leaves
    long_term.new_money and core alone. PD D-4."""
    result = await dm.run(scenario)
    card = result["decision_card"]
    long = card["long_term"]["new_money"]

    assert result["action_zones"]["price_zone"] == "near_52w_high"
    assert (card["new_money"]["action"], ids(card["new_money"]["blocked_by"]), card["new_money"]["decided_by"]) == (
        "wait", ["near_52w_high"], "near_52w_high")
    assert card["if_held"] == {"action": "hold", "add_allowed": False, "basis": "candidate_admits_new_money"}
    assert (long["action"], long["blocked_by"], long["decided_by"], long["unevaluated"]) == (long_term, [], None, [])
    assert result["decision_modes"]["core"]["action"] == core


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
    """PH-RF11: the gate fires on price_zone near_52w_high only (the uncapped reduce classification; WP2.3, OC-M8),
    and is unevaluated with no price_zone or an undetermined one. PD D-4."""
    assert check_near_52w_high("near_52w_high") == ("fired", {"id": "near_52w_high",
                                                              "reason": "price in the near-52-week-high band"})
    for zone in ("near_52w_low", "below_sma200", "above_sma50", "at_or_below_sma50"):
        assert check_near_52w_high(zone) == ("clear", None)
    for zone in (None, "undetermined"):
        assert check_near_52w_high(zone) == ("unevaluated", None)


async def test_an_undetermined_zone_leaves_near_52w_high_unevaluated_at_mid_term_only() -> None:
    """PH-RF11 / GC-R10: no price leaves the zone undetermined, so near_52w_high is listed unevaluated at mid term
    (registry reason) and not at long term, where it does not apply. PD D-4."""
    card = (await dm.run(dm.Scenario("PH-RF11 no zone", None, dm.PROFITABLE, dm.CALM)))["decision_card"]
    reason = GATES["near_52w_high"].unevaluated_when["mid_term"]

    assert {"id": "near_52w_high", "reason": reason} in card["new_money"]["unevaluated"]
    assert "near_52w_high" not in ids(card["long_term"]["new_money"]["unevaluated"])


@pytest.mark.parametrize("zone", ["strong_buy", "accumulate", "hold_bullish", "hold_neutral", "reduce",
                                  "undetermined", None])
def test_zone_caps_never_move_the_gate_zones(zone: str | None) -> None:
    """G-ZONE: the dip and valuation zone caps never move a zone into or out of {reduce, undetermined, None}, so
    near_52w_high reading price_zone (WP2.3) fires exactly where the capped label it read in WP2.2 did. G."""
    special = (None, "undetermined", "reduce")
    for dip_type, risk in product(("falling_knife", "extended_decline", None), ("extreme", "high", None)):
        out = apply_dip_gates_to_action_zones({"current_zone": zone}, {"dip_classification": {"type": dip_type}},
                                              {"classification": risk})["current_zone"]
        assert (out in special) is (zone in special) and (zone not in special or out == zone)
    for gate in ("headwind", "attractive", "neutral", "unknown", "not_applicable"):
        out = _apply_valuation_gate_to_zone(zone, gate)[0]
        assert (out in special) is (zone in special) and (zone not in special or out == zone)


# --- PH-RF12, RF12b [D-1c]: the thesis cap ---------------------------------------------------------------


async def test_case_e() -> None:
    """PH-RF12: case E (poor business, bullish chart) gives new_money avoid [thesis_broken] decided by it, with the
    plan's reason (C22) on the blocker and in do_not_buy_if, if_held review_exit (business_broken) and
    long_term.new_money avoid [thesis_broken]. PD D-1c."""
    card = (await dm.run("E_poor_business_bullish_chart"))["decision_card"]
    long = card["long_term"]["new_money"]
    reason = "thesis broken: business broken (business_status broken)"

    assert (card["new_money"]["action"], ids(card["new_money"]["blocked_by"]), card["new_money"]["decided_by"]) == (
        "avoid", ["thesis_broken"], "thesis_broken")
    assert card["new_money"]["blocked_by"][0]["reason"] == card["hard_gates"]["blocking"][0]["reason"] == reason
    assert card["conditions"]["do_not_buy_if"] == [reason]
    assert held(card) == ("review_exit", "business_broken")
    assert (long["action"], ids(long["blocked_by"])) == ("avoid", ["thesis_broken"])


async def test_burner_runway_2q() -> None:
    """PH-RF12: a 2-quarter burner gives avoid at both horizons, if_held review_exit (runway_critical), the
    blocker's codes [runway_critical] and core avoid; it closes the p5 inversion. PD D-1c."""
    result = await dm.run("H_burner_runway_2q")
    card = result["decision_card"]

    assert (card["new_money"]["action"], card["long_term"]["new_money"]["action"]) == ("avoid", "avoid")
    assert held(card) == ("review_exit", "runway_critical")
    assert card["new_money"]["blocked_by"][0]["codes"] == ["runway_critical"]
    assert result["decision_modes"]["core"]["action"] == "avoid"


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
    assert blocker["reason"] == ("thesis broken: business broken (business_status broken) and cash runway below "
                                 "4 quarters (balance sheet unsafe)")


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


async def test_case_e_and_its_intact_twin() -> None:
    """PH-RF12b through analyze: E (business broken, no burn, runway unknown, balance not unsafe) fires on the
    business component with no runway clause, in clears_when or in the reassessment's outstanding; its twin
    with an intact business carries no thesis gate and admits a starter. PD D-1c."""
    broken, intact = await dm.run("E_poor_business_bullish_chart"), await dm.run(E_INTACT)
    blocker = broken["decision_card"]["new_money"]["blocked_by"][0]
    # E's candidate admits (hold_or_add), so the reassessment lists the thesis condition
    outstanding = broken["decision_card"]["new_money"]["earliest_reassessment"]["outstanding"]

    for result in (broken, intact):
        assert result["fundamentals_summary"]["burn_metrics"] is None
        assert result["dislocation_framework"]["balance_sheet_safety"]["status"] != "unsafe"
    assert (blocker["id"], blocker["codes"]) == ("thesis_broken", [BB])
    assert outstanding == [{"id": "thesis_broken", "clears_when": THESIS_CLEARS_WHEN[BB]}]
    assert intact["dislocation_framework"]["business_integrity"]["status"] == "intact"
    assert (intact["decision_card"]["new_money"]["action"], dm.gate_ids(intact)) == ("starter", [])


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


# --- PH-RF15 to RF18 [D-3]: core follows the canonical long-term decision ----------------------------------


async def test_core_direction_1_case_d() -> None:
    """PH-RF15: case D: new_money avoid, long_term.new_money buy, core buy: the mid-term chart never decides
    core; balanced avoid holds in both vocabularies. PD D-3."""
    result = await dm.run("D_quality_attractive_poor_chart")
    card = result["decision_card"]

    assert (card["new_money"]["action"], card["long_term"]["new_money"]["action"],
            result["decision_modes"]["core"]["action"]) == ("avoid", "buy", "buy")
    assert result["decision_modes"]["balanced"]["action"] == "avoid"


@pytest.mark.parametrize(("name", "long_term", "gates"), [
    ("D2_quality_poor_chart_earnings_3d", "wait", ["earnings_blackout"]),
    ("C_quality_knife_dislocation", "avoid", ["falling_knife"]),
    ("G_unsupported_currency_knife", "avoid", ["falling_knife", "unsupported_currency"])])
async def test_core_follows_the_gated_long_term(name: str, long_term: str, gates: list[str]) -> None:
    """PH-RF16: gates reach core through long_term.new_money (D2 wait, C and G avoid) while mid new money is
    avoid. PD D-3, D-5b."""
    result = await dm.run(name)
    card = result["decision_card"]

    assert (card["new_money"]["action"], card["long_term"]["new_money"]["action"],
            result["decision_modes"]["core"]["action"]) == ("avoid", long_term, long_term)
    assert ids(card["long_term"]["new_money"]["blocked_by"]) == gates


@pytest.mark.parametrize(("name", "new_money", "long_term"), [("E2_mixed_fcf_bullish_chart", "starter", "starter"),
                                                              ("C1_headwind_hold_or_add", "starter", "wait")])
async def test_core_direction_2(name: str, new_money: str, long_term: str) -> None:
    """PH-RF17 (core clauses; balanced is WP2.3): E2 (caution fit) and C1 (headwind) keep mid new money starter
    and core wait; C1's long term is wait (neither horizon caps the other). PD D-3."""
    result = await dm.run(name)
    card = result["decision_card"]

    assert (card["new_money"]["action"], card["long_term"]["new_money"]["action"],
            result["decision_modes"]["core"]["action"]) == (new_money, long_term, "wait")


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


@pytest.mark.parametrize(("name", "core"), [("F3_quality_neutral_valuation_poor_chart", "wait"),
                                            ("C1_headwind_hold_or_add", "wait"), ("T1_extreme_risk_bullish", "wait")])
async def test_core_own_cap_through_the_pipeline(name: str, core: str) -> None:
    """PH-RF18 through analyze: F3 (neutral valuation), C1 (headwind) and T1 (caution fit) give core wait where
    0609fde gave build_position or not_core_yet. PD D-3."""
    assert (await dm.run(name))["decision_modes"]["core"]["action"] == core
