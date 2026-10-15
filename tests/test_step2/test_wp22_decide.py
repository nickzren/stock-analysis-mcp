"""WP2.2: the decision core (plan §2, §4): decide(), the candidate mapping and the analyze combination.

Annex: docs/plans/step2-v2-research/gate_contract.md and policy_holder.md, "Corrected proposal", as amended
by plan §0 #1 (most-restrictive in analyze; GC-R3, R4, R5, R13 rewritten) and §4 R2.6. The expected tables are
literals from the plan, not read back from the registry. Analyze runs go through
tests/support/decision_matrix.py, which pins the clock; pure runs call decide() through dm.decide_on.
"""

import importlib
from itertools import combinations, product
from typing import Any

import pytest

from stock_analysis.tools.analyze.decision import (
    CANDIDATE_ACTIONS,
    ENTRY_CANDIDATE_CODES,
    map_candidate,
)
from stock_analysis.tools.analyze.executive_summary import build_policy_action
from stock_analysis.tools.analyze.gates import (
    ANALYZE_GATE_IDS,
    GATES,
    NOT_APPLICABLE,
    THESIS_CLEARS_WHEN,
    gate_rank,
)
from tests.support import decision_matrix as dm
from tests.support import freshness_frames as fr
from tests.support import swing_matrix as sm
from tests.test_step2 import test_characterize_gate_contract as gc
from tests.test_step2 import test_characterize_policy_holder as ph
from tests.test_step2.test_characterize_freshness import history
from tests.test_tools.test_trade_card import build_card

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
risk_module = importlib.import_module("stock_analysis.tools.risk_metrics")

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
STALE_HISTORY = {"last_row_date": "2026-10-08", "last_valid_date": "2026-10-08", "last_row_valid": True,
                 "completed_valid_date": "2026-10-08"}  # one session behind the matrix clock


def blockers(*gate_ids: str) -> list[dict[str, Any]]:
    """Fired blockers as _compute_hard_gates emits them (a thesis_broken blocker names its failing component)."""
    return [{"id": g, "reason": f"{g} fired", **({"codes": ["business_broken"]} if g == "thesis_broken" else {})}
            for g in gate_ids]


def run(mid: str = "hold_or_add", long: str = "accumulate", gates: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    return dm.decide_on({"mid_term": mid, "long_term": long}, blockers(*gates))


def ids(entries: list[dict[str, Any]]) -> list[str]:
    return [e["id"] for e in entries]


def stale(scenario: dm.Scenario) -> dm.Scenario:
    return dm.Scenario(scenario.source, {**(scenario.technicals or {}), "daily_history": STALE_HISTORY},
                       scenario.fundamentals, scenario.risk, events=scenario.events, summary=scenario.summary)


def horizons(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    card = result["decision_card"]
    return {"mid_term": card["new_money"], "long_term": card["long_term"]["new_money"]}


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


PIPELINE_GATE_CASES: dict[str, tuple[dm.Scenario, tuple[str, str | None], tuple[str, str | None]]] = {
    "stale_data": (dm.Scenario("GC-R2", {**dm.STRONG, "daily_history": STALE_HISTORY}, dm.PROFITABLE, dm.CALM),
                   (WFD, "stale_data"), (WFD, "stale_data")),
    "freshness_unverifiable": (dm.Scenario("GC-R2", {**dm.STRONG, "daily_history": dm.OMIT}, dm.PROFITABLE,
                                           dm.CALM), (WFD, "freshness_unverifiable"), (WFD, "freshness_unverifiable")),
    "data_quality_critical": (dm.Scenario("GC-R2", dm.STRONG, dm.PROFITABLE, None),
                              (WFD, "data_quality_critical"), (WFD, "data_quality_critical")),
    "missing_runway": (dm.Scenario("GC-R2", dm.WEAK, ph.UNKNOWN_RUNWAY_SNAPSHOT, dm.MODERATE),
                       (WFD, "missing_runway"), (WFD, "missing_runway")),
    "liquidity_missing": (dm.Scenario("GC-R2", dm.STRONG, dm.PROFITABLE,
                                      {**dm.CALM, "liquidity": {"avg_dollar_volume": None}}),
                          (WFD, "liquidity_missing"), (WFD, "liquidity_missing")),
    "falling_knife": (dm.SCENARIOS["C_quality_knife_dislocation"], ("avoid", None), ("avoid", "falling_knife")),
    "falling_knife_unsupported": (gc.ANALYZE_PIPELINE_ROWS["etf_crash"][0], ("avoid", None),
                                  ("avoid", "falling_knife")),
    "weak_liquidity": (dm.SCENARIOS["S9_weak_liquidity"], ("avoid", "weak_liquidity"), ("avoid", "weak_liquidity")),
    "thesis_broken": (dm.SCENARIOS["H_burner_runway_2q"], ("avoid", "thesis_broken"), ("avoid", "thesis_broken")),
    "unsupported_instrument": (dm.SCENARIOS["ETF_strong"], ("watch", "unsupported_instrument"),
                               ("watch", "unsupported_instrument")),
    "unsupported_currency": (dm.SCENARIOS["GBp_strong"], ("watch", "unsupported_currency"),
                             ("watch", "unsupported_currency")),
    "earnings_blackout": (dm.SCENARIOS["S4_earnings_gate_on_hold_or_add"], ("wait", "earnings_blackout"),
                          ("wait", "earnings_blackout")),
    "earnings_unverifiable": (dm.Scenario("GC-R2", dm.STRONG, dm.PROFITABLE, dm.CALM, events=sm.UNVERIFIABLE_EVENTS),
                              ("wait", "earnings_unverifiable"), ("wait", "earnings_unverifiable")),
    "near_52w_high": (dm.SCENARIOS["A7_near_52w_high_bullish"], ("wait", "near_52w_high"), ("buy", None)),
}


@pytest.mark.parametrize("case", PIPELINE_GATE_CASES)
async def test_gate_outcome_per_horizon_analyze_pipeline(case: str) -> None:
    """GC-R2 (analyze columns) through analyze: each gate's action and decided_by per horizon, and the gate is
    disclosed exactly where it applies. Closes S-C6, S-C7 and BL-A4's analyze half (WP2.1 carry-forward).
    PD D-5a, D-5b."""
    scenario, mid, long = PIPELINE_GATE_CASES[case]
    gate_id = case.removesuffix("_unsupported")
    shown = horizons(await dm.run(scenario))

    for horizon, expected in (("mid_term", mid), ("long_term", long)):
        decision = shown[horizon]
        assert (decision["action"], decision["decided_by"]) == expected, horizon
        assert (gate_id in ids(decision["blocked_by"])) is (GATES[gate_id].outcome[horizon] != NOT_APPLICABLE), horizon


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


PLAN_DECIDED_BY = {"data_and_support": "unsupported_instrument", "risk_metrics_outage": "data_quality_critical",
                   "stale_and_knife": "falling_knife"}


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


ANCHORS = {  # R2.6(e) rows on real runs: (analyze scenario, horizon, action, card case, card action)
    "data+support": (stale(dm.SCENARIOS["ETF_strong"]), "mid_term", "watch", "stale+etf", WFD),
    "data+price_risk": (stale(dm.SCENARIOS["C_quality_knife_dislocation"]), "long_term", "avoid", "stale+knife", WFD),
    "data+liquidity": (stale(dm.SCENARIOS["S9_weak_liquidity"]), "mid_term", "avoid", "stale+weak_liq", WFD),
    "data_on_avoid": (stale(dm.SCENARIOS["S3_plain_avoid"]), "mid_term", "avoid", "stale_only", WFD),
    "liquidity_missing": (PIPELINE_GATE_CASES["liquidity_missing"][0], "mid_term", WFD, "liq_missing", "avoid"),
}


@pytest.mark.parametrize("name", ANCHORS)
async def test_pinned_differences_on_real_runs(name: str) -> None:
    """GC-R4: each pinned R2.6(e) difference on a real analyze run and a real card. PD D-5."""
    scenario, horizon, action, card_case, card_action = ANCHORS[name]

    assert horizons(await dm.run(scenario))[horizon]["action"] == action
    assert sm.card_case(card_case)["action"] == card_action


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


# --- GC-R10, analyze half ---------------------------------------------------------------------------------

UNEVALUATED_CASES: dict[str, tuple[dm.Scenario, list[str], list[str]]] = {
    "technicals_failed": (dm.Scenario("GC-R10", None, dm.PROFITABLE, dm.CALM),
                          ["falling_knife", "near_52w_high"], ["falling_knife"]),
    "fundamentals_failed": (dm.SCENARIOS["S8_fundamentals_failed"], ["missing_runway", "thesis_broken"],
                            ["missing_runway", "thesis_broken"]),
    "events_failed": (dm.Scenario("GC-R10", dm.STRONG, dm.PROFITABLE, dm.CALM, events=None),
                      ["earnings_blackout"], ["earnings_blackout"]),
    "us_sink": (dm.Scenario("GC-R11", dm.KNIFE, None, None, events=None),
                ["missing_runway", "thesis_broken", "earnings_blackout"],
                ["missing_runway", "thesis_broken", "earnings_blackout"]),
    "etf_sink": (dm.Scenario("GC-R11", dm.KNIFE, None, None, events=None, summary=dm.ETF),
                 ["earnings_blackout"], ["earnings_blackout"]),
    "clean": (dm.SCENARIOS["S1_hold_or_add_clean"], [], []),
    "etf": (dm.SCENARIOS["ETF_strong"], [], []),
}


@pytest.mark.parametrize("name", UNEVALUATED_CASES)
async def test_unevaluated_lists(name: str) -> None:
    """GC-R10 (analyze half): applicable gates that were not computed are listed per horizon with the registry
    reason, in rank order, never alongside a fired one; not_applicable gates (missing_runway and thesis_broken
    for an ETF, near_52w_high at long term) are never listed. RF: R2.6(f), plan §4:208-212."""
    scenario, mid, long = UNEVALUATED_CASES[name]
    shown = horizons(await dm.run(scenario))

    for horizon, expected in (("mid_term", mid), ("long_term", long)):
        decision = shown[horizon]
        assert decision["unevaluated"] == [{"id": g, "reason": GATES[g].unevaluated_when[horizon]} for g in expected]
        assert not set(ids(decision["unevaluated"])) & set(ids(decision["blocked_by"]))


# --- GC-R13, rewritten [D-5]; PH-RF21 [D-5] ---------------------------------------------------------------


async def test_summary_failure_outcomes() -> None:
    """GC-R13: with stock_summary down, analyze gives watch at both horizons (data + support, most-restrictive)
    decided by unsupported_instrument, and the card watch [unsupported_*] (Q-GC2 (i)). PD D-5."""
    failed = {"error": True, "error_type": "data_unavailable", "message": "upstream"}
    shown = horizons(await dm.run(dm.Scenario("GC-R13", dm.STRONG, dm.PROFITABLE, dm.CALM, summary=failed)))
    card = build_card(summary_data={}, tool_failures=[{"tool": "stock_summary", "error": "x"}])

    for decision in shown.values():
        assert (decision["action"], decision["decided_by"], ids(decision["blocked_by"])) == (
            "watch", "unsupported_instrument",
            ["data_quality_critical", "unsupported_instrument", "unsupported_currency"])
    assert (card["action"], ids(card["blockers"])) == ("watch", ["unsupported_instrument", "unsupported_currency"])


async def test_case_g_is_avoid_at_both_horizons() -> None:
    """PH-RF21: case G (EUR quality crash + knife) gives avoid at both horizons (most-restrictive; support-first
    would give watch); core follows long_term (avoid). PD D-5."""
    result = await dm.run("G_unsupported_currency_knife")
    shown = horizons(result)

    assert (shown["mid_term"]["action"], shown["mid_term"]["decided_by"]) == ("avoid", None)
    assert (shown["long_term"]["action"], shown["long_term"]["decided_by"]) == ("avoid", "falling_knife")
    for decision in shown.values():
        assert ids(decision["blocked_by"]) == ["falling_knife", "unsupported_currency"]
    assert result["decision_modes"]["core"]["action"] == "avoid"


# --- FR-R-F2 (action) [D-7], FR-R-F14 [D-5c], FR-R-F18 ---------------------------------------------------


async def test_r3_waits_for_data_at_both_horizons() -> None:
    """FR-R-F2 (action clause; the sizing clause is test_wp23_sizing's): R3 (fresh 5m, technicals 2 sessions
    behind) gives wait_for_data at both horizons, decided by stale_data [daily_history_stale]. PD D-7."""
    result = await dm.run(fr.scenario(fr.TRUTH_TABLE["R3"], "R3"))

    for decision in horizons(result).values():
        assert (decision["action"], decision["decided_by"]) == (WFD, "stale_data")
        assert [(b["id"], b["class"], b["effect"], b["codes"]) for b in decision["blocked_by"]] == [
            ("stale_data", "data", "veto", ["daily_history_stale"])]
    assert result["decision_card"]["action_now"] == WFD


async def empty_latest_bar_risk() -> dict[str, Any]:
    """The real risk_metrics over a daily frame whose latest bar is empty."""
    real = risk_module.fetch_history
    risk_module.fetch_history = history({"1d": fr.daily("2026-10-09", None)})
    try:
        risk = await risk_module.risk_metrics("ACME")
    finally:
        risk_module.fetch_history = real
    return {k: v for k, v in risk.items() if k not in ("symbol", "meta")}


@pytest.mark.parametrize(("name", "clean_action"), [("S1_hold_or_add_clean", "starter"),
                                                    ("T3_plain_wait_weak_setup", "wait")])
async def test_an_empty_latest_bar_waits_for_data(name: str, clean_action: str) -> None:
    """FR-R-F14: an empty latest bar (avg_dollar_volume None) gives liquidity_missing, data class in analyze, so
    new money waits for data at both horizons, over a wait candidate too; the card keeps avoid. PD D-5c."""
    risk = await empty_latest_bar_risk()
    clean = horizons(await dm.run(name))["mid_term"]
    shown = horizons(await dm.run(name, risk=risk))
    card = build_card(risk_data=risk)

    assert risk["liquidity"]["avg_dollar_volume"] is None
    assert clean["action"] == clean_action
    for decision in shown.values():
        assert (decision["action"], decision["decided_by"]) == (WFD, "liquidity_missing")
        assert [(b["id"], b["class"]) for b in decision["blocked_by"]] == [("liquidity_missing", "data")]
    assert (card["action"], [(b["id"], b["class"]) for b in card["blockers"]]) == (
        "avoid", [("liquidity_missing", "liquidity")])


@pytest.mark.parametrize("row", ["R3", "R5", "O2"])
async def test_freshness_never_moves_scores_or_the_candidate(row: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-R-F18: the freshness gate changes new_money only: score, tilt and the pre-gate policy codes equal the
    run with the gate disabled (the Step 2 boundary). RF."""
    gated = await dm.run(fr.scenario(fr.TRUTH_TABLE[row], row))
    real = orchestrator.check_price_freshness
    monkeypatch.setattr(orchestrator, "check_price_freshness", lambda **kw: ([], real(**kw)[1]))
    ungated = await dm.run(fr.scenario(fr.TRUTH_TABLE[row], row))

    def boundary(result: dict[str, Any]) -> tuple[Any, ...]:
        policy = result["diagnostics"]["pre_gate_policy"]
        return (result["verdict"]["score"], result["verdict"]["tilt"], policy["mid_term"], policy["long_term"])

    assert boundary(gated) == boundary(ungated) == (boundary(gated)[0], boundary(gated)[1], "hold_or_add", "accumulate")
    assert (gated["decision_card"]["new_money"]["action"], ungated["decision_card"]["new_money"]["action"]) == (
        WFD, "starter")


# --- The card carries one decision (plan §2; D-13 alias) ----------------------------------------------------

CARD_KEYS = ["horizon", "new_money", "if_held", "long_term", "hard_gates", "action_now"]
NEW_MONEY_KEYS = ["action", "blocked_by", "decided_by", "unevaluated", "earliest_reassessment"]


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_the_card_carries_one_decision(name: str) -> None:
    """Card shape: the decision_card at every tier: horizon, new_money, if_held, long_term in plan §2's
    vocabularies (long_term.new_money also carries its conditions, WP2.3 §7e); action_now aliases
    new_money.action and the snapshot's decision_action_now follows it (D-13);
    hard_gates lists every fired gate (blocking = new_money.blocked_by without class and effect; long term drops
    the not_applicable near_52w_high) and checks every analyze gate in disclosure order. RF (plan §2)."""
    result = await dm.run(name)
    card = result["decision_card"]
    mid, long = card["new_money"], card["long_term"]["new_money"]

    assert list(card)[:len(CARD_KEYS)] == CARD_KEYS
    assert (card["horizon"], list(card["long_term"]), card["long_term"]["horizon"]) == (
        "mid_term", ["horizon", "new_money"], "long_term")
    assert (list(mid), list(long)) == (NEW_MONEY_KEYS, [*NEW_MONEY_KEYS, "conditions"])  # WP2.3: §2, §7e
    for decision in (mid, long):
        assert decision["action"] in ORDER
        assert decision["decided_by"] is None or decision["decided_by"] in ids(decision["blocked_by"])
    assert list(card["if_held"]) == ["action", "add_allowed", "basis"]
    assert card["action_now"] == mid["action"] == result["watchlist_snapshot"]["decision_action_now"]
    assert [{k: v for k, v in b.items() if k not in ("class", "effect")} for b in mid["blocked_by"]] == card[
        "hard_gates"]["blocking"]
    assert ids(long["blocked_by"]) == [g for g in ids(mid["blocked_by"]) if g != "near_52w_high"]
    assert card["hard_gates"]["checks"] == {g: g in ids(mid["blocked_by"]) for g in ANALYZE_GATE_IDS}
    assert list(card["hard_gates"]["checks"]) == list(ANALYZE_GATE_IDS)
    assert result["watchlist_snapshot"]["hard_gate_ids"] == sorted(ids(mid["blocked_by"]))
    for tier, shown in dm.tiers(result).items():
        assert {k: shown["decision_card"][k] for k in CARD_KEYS} == {k: card[k] for k in CARD_KEYS}, tier


@pytest.mark.parametrize(("name", "trace"), [
    ("S1_hold_or_add_clean", ["pre_gate_policy.mid_term=hold_or_add -> starter"]),
    ("S4_earnings_gate_on_hold_or_add", ["pre_gate_policy.mid_term=hold_or_add -> wait", "decided_by=earnings_blackout"])])
async def test_the_decision_trace_names_the_emitted_code_and_the_deciding_gate(name: str, trace: list[str]) -> None:
    """Card shape (C8): diagnostics.decision_trace (the card's rationale until WP2.3, plan §2) is the emitted mid code
    and new_money's action, plus a decided_by line only when a gate decided (S1 none; S4 earnings_blackout); the
    card has no rationale. RF (plan §2)."""
    result = await dm.run(name)

    assert "rationale" not in result["decision_card"]
    assert result["diagnostics"]["decision_trace"] == trace


# --- Interim 2.2 surfaces (flipped by WP2.3) -----------------------------------------------------------------
# Until WP2.3 the derived surfaces read the emitted, knife-free policy (WP2.2 integrated plan §5); WP2.3 derives
# them from new_money. The snapshot keeps reading the pre-gate candidate until WP2.4 (plan §0 #22).

KNIFE_BOUNCE = {  # still a falling knife (below a falling SMA200, -25% over 3m, 52w high 200 days ago), bouncing
    "current_price": 80.0,
    "moving_averages": {"sma_20": 77.6, "sma_50": 78.4, "sma_200": 89.6, "sma_200_slope_pct_per_day": -0.001,
                        "rules": {"above_sma20": {"triggered": True}, "above_sma50": {"triggered": True},
                                  "above_sma200": {"triggered": False}, "golden_cross": {"triggered": False},
                                  "death_cross": {"triggered": False}}},
    "rsi": {"value": 28.0, "rules": {"oversold": {"triggered": True}, "overbought": {"triggered": False}}},
    "macd": {"rules": {"bullish_cross": {"triggered": True}, "bearish_cross": {"triggered": False}}},
    "returns": {"return_1w": 0.03, "return_1w_zscore": 0.5, "return_1m": 0.05, "return_3m": -0.25, "return_1y": -0.1},
    "price_position": {"position_in_range": 0.3, "week_52_low": 72.0, "week_52_high": 120.0, "from_52w_high": -0.33,
                       "days_since_52w_high": 200},
}
KNIFE_NOT_DISLOCATION = dm.Scenario(  # the whole-WP review's knife_evidence.py inputs
    "WP2.2 review knife", KNIFE_BOUNCE,
    {**dm.QUALITY, "analyst_coverage": dm.PROFITABLE["analyst_coverage"],
     "growth": {"revenue_yoy": 0.30, "rules": {"high_growth": {"triggered": True}}},
     "profitability": {"net_margin": 0.25, "rules": {"profitable": {"triggered": True}}}},
    dm.CALM, news=dm.NEWS_VENDOR_RICH, ownership=dm.INSIDER_BUYING)


async def test_a_non_dislocation_knife_shows_no_admitting_surface() -> None:
    """IS-CH1 (flipped by WP2.3): a falling knife that is not a dislocation candidate (mismatch unclear) and whose
    knife-free candidate admits new money (bullish tilt, caution fit) gives new_money avoid [falling_knife], and
    every derived surface follows it: no top-level policy_action or card.horizon_fit; balanced avoid, speculative
    and core avoid with capped_by; the summary's new-money segment "avoid; blocked by falling_knife" with no size
    and no "small starter position"; dislocation can_start_now false / buy_now null; no card sizing, buy band or
    zone band; the knife's clears_when binds at both horizons (plan §7e). speculative_small_position survives only
    as the full tier's diagnostics.pre_gate_policy.mid_term, superseded by new_money. G from WP2.3 (PD D-9, D-3);
    the snapshot clause is test_the_snapshot_keeps_the_knife_free_candidate_until_wp2_4."""
    result = await dm.run(KNIFE_NOT_DISLOCATION)
    card = result["decision_card"]
    dislocation = result["dislocation_framework"]
    policy = result["diagnostics"]["pre_gate_policy"]
    modes = result["decision_modes"]
    knife = GATES["falling_knife"].clears_when

    assert (result["dip_assessment"]["dip_classification"]["type"], dislocation["mismatch_verdict"]["status"]) == (
        "falling_knife", "unclear")
    assert (result["verdict"]["tilt"], result["verdict"]["horizon_fit"]["mid_term"]) == ("bullish", "caution")
    assert (card["new_money"]["action"], ids(card["new_money"]["blocked_by"]), card["new_money"]["decided_by"]) == (
        "avoid", ["falling_knife"], "falling_knife")
    assert ("policy_action" in result, "horizon_fit" in card) == (False, False)
    assert (policy["mid_term"], policy["superseded_by"]["new_money"]) == ("speculative_small_position", "avoid")
    assert {m: (mode["action"], mode.get("capped_by")) for m, mode in modes.items()} == {
        "core": ("avoid", "long_term_avoid"), "balanced": ("avoid", None), "speculative": ("avoid", "new_money_avoid")}
    assert result["executive_summary"].startswith(
        f"Decision (mid-term, new money): avoid; blocked by falling_knife; no reassessment date: re-run analyze "
        f"after {knife}. If held: hold. Long-term (1-5y): avoid; blocked by falling_knife;")
    assert "size ~" not in result["executive_summary"] and "small starter" not in result["executive_summary"]
    assert (dislocation["action"]["can_start_now"], dislocation["action"]["buy_now"]) == (False, None)
    assert (card["sizing"], card["entry"], result["action_zones"]["position_sizing_range"]) == (None, None, None)
    assert (card["conditions"]["buy_only_if"], card["conditions"]["add_only_if"],
            card["long_term"]["new_money"]["conditions"]["buy_only_if"]) == ([knife], [knife], [knife])


async def test_the_snapshot_keeps_the_knife_free_candidate_until_wp2_4() -> None:
    """IS-CH1b: the 2.0.0 snapshot's action_mid_term reads diagnostics.pre_gate_policy (plan §0 #22), so it shows
    the knife-free speculative_small_position while new_money is avoid. CH: flipped by WP2.4 (snapshot 3.0.0
    new_money_action). Keep the live MCP on v0.2.0 until then."""
    result = await dm.run(KNIFE_NOT_DISLOCATION)

    assert result["decision_card"]["new_money"]["action"] == "avoid"
    assert result["watchlist_snapshot"]["action_mid_term"] == "speculative_small_position"


async def test_a_thesis_capped_row_offers_no_starter_and_no_add() -> None:
    """IS-CH2 (flipped by WP2.3): H (2-quarter burner) gives new_money avoid (thesis_broken) and if_held
    review_exit, and the derived surfaces follow: dislocation can_start_now false / buy_now null; add_only_if
    empty on the card and the framework (if_held is not hold, §7e); buy_only_if names only the binding thesis
    component (runway, plan §4) on the card, the framework, long_term.new_money.conditions and core's
    required_before_full_size; the upgrade items that do not bind are monitored. G from WP2.3 (PD D-9, §7e)."""
    result = await dm.run("H_burner_runway_2q")
    card = result["decision_card"]
    action = result["dislocation_framework"]["action"]
    runway = [THESIS_CLEARS_WHEN["runway_critical"]]

    assert (card["new_money"]["action"], card["new_money"]["decided_by"], card["if_held"]["action"]) == (
        "avoid", "thesis_broken", "review_exit")
    assert (action["can_start_now"], action["buy_now"]) == (False, None)
    assert card["conditions"]["add_only_if"] == action["add_only_if"] == []
    assert card["conditions"]["buy_only_if"] == action["buy_only_if"] == runway
    assert card["long_term"]["new_money"]["conditions"] == {"buy_only_if": runway}
    assert result["decision_modes"]["core"]["required_before_full_size"] == runway
    assert card["conditions"]["monitor"] == ["profitMargins > 0 for 2 consecutive quarters",
                                             "freeCashflow > 0 for 2 consecutive quarters", "free_cash_flow_ttm >= 0"]
