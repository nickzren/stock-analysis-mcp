"""WP2.3 conditions: each horizon lists only what binds it (plan §7e [FIX BL-E1, BL-E2], §0 #15, #25).

A horizon lists the clears_when of each fired non-support gate tighter than map(candidate), plus each candidate
factor whose cleared state alone moves map(candidate) into the target (an entry action when the candidate does not
admit new money; buy under a starter). decision_card.conditions is mid-term; long_term.new_money.conditions is
long-term; core's required_before_full_size mirrors long-term; balanced, speculative, dislocation and dip mirror
mid-term. Annex: output_contract.md §5 "Corrected proposal" (its U3 cells replaced by §7e)."""

import importlib
import itertools
import json
from typing import Any

import pytest

from stock_analysis.tools.analyze import action_zones as az
from stock_analysis.tools.analyze.action_zones import (
    _apply_valuation_gate_to_zone,
    apply_dip_gates_to_action_zones,
    zone_after_caps,
)
from stock_analysis.tools.analyze.conditions import (
    CandidateInputs,
    Factor,
    candidate_factors,
    risk_text,
    valuation_headwind_texts,
)
from stock_analysis.tools.analyze.decision_card import build_conditions, candidate_codes
from stock_analysis.tools.analyze.gates import GATES, THESIS_CLEARS_WHEN
from stock_analysis.tools.analyze.reassessment import PERMANENT_CLEARS_WHEN
from stock_analysis.tools.analyze.signals import classify_risk_regime
from stock_analysis.tools.analyze.verdict import build_horizon_fit
from tests.support import decision_matrix as dm
from tests.support import output_contract as oc

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

PE = "P/E <= 50x (currently headwind)"
PEG = "PEG <= 3.0 (currently headwind)"
EARN = "no earnings date or window within 5 days"
TU = "applicability verified and required business and cash-runway checks evaluable"
LIQ = "1-year average daily dollar volume >= $1M"
KNIFE = "falling-knife rubric score below 4"
NEAR = "price leaves the near-52-week-high band"
BB = THESIS_CLEARS_WHEN["business_broken"]
RC = "cash runway no longer below 4 quarters"
DE, REV, FCF = "debt_to_equity < 1.5x", "revenue_yoy > -20%", "free_cash_flow_ttm >= 0"
PM, FC = "profitMargins > 0 for 2 consecutive quarters", "freeCashflow > 0 for 2 consecutive quarters"
VOL60 = "risk_regime <= high (volatility < 60%)"
DD35 = "risk_regime <= medium (max_drawdown_1y > -35%)"
DD50 = "risk_regime <= high (max_drawdown_1y > -50%)"
BETA18 = "risk_regime <= high (beta < 1.8)"

DRAWDOWN_EXTREME = {**dm.CALM, "volatility": {"annualized": 0.30}, "drawdown": {"max_1y": -0.55}}
BETA_EXTREME = {**dm.CALM, "beta": {"value": 2.0}}
PEG_HEADWIND = {**dm.PROFITABLE, "valuation": {"pe_trailing": 30.0, "trailing_eps": 3.0, "peg_ratio": 3.5},
                "yield_metrics": {}}
LOCAL = {
    "T1_drawdown": dm.Scenario("RF-E2 T1 drawdown", dm.STRONG, dm.QUALITY, DRAWDOWN_EXTREME),
    "T1_beta": dm.Scenario("RF-E2 T1 beta", dm.STRONG, dm.QUALITY, BETA_EXTREME),
    "S2_drawdown": dm.Scenario("RF-E2 mid starter->full", dm.at_price(dm.STRONG, 89.0), dm.QUALITY, DRAWDOWN_EXTREME),
    "S2_headwind": dm.Scenario("RF-E1 zone-capped headwind", dm.at_price(dm.STRONG, 89.0), dm.HEADWIND, dm.CALM),
    "C1_peg": dm.Scenario("RF-E1 PEG", dm.STRONG, PEG_HEADWIND, dm.CALM),
}


async def run(name: str) -> dict[str, Any]:
    return await dm.run(LOCAL.get(name) or name)


def lists(result: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    """(mid buy_only_if, mid add_only_if, monitor, long buy_only_if)."""
    card = result["decision_card"]
    conditions = card["conditions"]
    return (conditions["buy_only_if"], conditions["add_only_if"], conditions["monitor"],
            card["long_term"]["new_money"]["conditions"]["buy_only_if"])


# --- RF-E1 [FIX BL-E1] (plan §7e, retargeted to long term by §0 #25) ---------------------------------------


async def test_headwind_condition_binds_at_long_term_only() -> None:
    """RF-E1 (C1, valuation headwind): mid-term is already a starter that clearing the headwind would not lift to
    buy, so the valuation condition is not in the mid-term lists (monitored only); it binds the long-term wait
    and core's conditionals, quoting the assessment's own P/E threshold, never "P/E < 25x". RF."""
    result = await dm.run("C1_headwind_hold_or_add", fundamentals={
        **dm.HEADWIND, "financial_health": {"debt_to_equity": 0.5}})
    for tier, shown in dm.tiers(result).items():
        card = shown["decision_card"]
        assert card["new_money"]["action"] == "starter", tier
        assert (card["conditions"]["buy_only_if"], card["conditions"]["add_only_if"],
                card["conditions"]["monitor"]) == ([], [], [PE]), tier
        assert card["long_term"]["new_money"]["conditions"] == {"buy_only_if": [PE]}, tier
    core = result["decision_modes"]["core"]
    assert (core["action"], core["required_before_full_size"]) == ("wait", [PE])
    assert "25x" not in json.dumps(result)


@pytest.mark.parametrize(("codes", "basis", "expected"), [
    (["valuation_extended_high_pe"], "pe_trailing", [PE]),
    (["valuation_extended_high_peg"], "peg_ratio", [PEG]),
    (["valuation_extended_high_sales_multiple_unprofitable"], "ev_to_sales", ["EV/S <= 10x (currently headwind)"]),
    (["valuation_extended_high_sales_multiple_unprofitable"], "ps_trailing", ["P/S <= 10x (currently headwind)"]),
    (["valuation_extended_high_pe", "valuation_extended_high_peg"], "peg_ratio", [PE, PEG]),
])
def test_headwind_text_quotes_the_assessment_threshold(codes: list[str], basis: str, expected: list[str]) -> None:
    """RF-E1 (unit): one condition per fired headwind metric, naming the metric and the threshold the gate uses.
    RF."""
    assert valuation_headwind_texts({"gate": "headwind", "basis": basis}, codes) == expected


UNPROFITABLE = {"trailing_eps": -1.0}


@pytest.mark.parametrize(("fund", "headwind"), [
    ({"valuation": {"pe_trailing": 50.0}}, False), ({"valuation": {"pe_trailing": 50.1}}, True),
    ({"valuation": {"pe_trailing": 20.0, "peg_ratio": 3.0}}, False),
    ({"valuation": {"pe_trailing": 20.0, "peg_ratio": 3.01}}, True),
    ({"valuation": {**UNPROFITABLE, "ps_trailing": 10.0}, "profitability": {"net_margin": -0.1}}, False),
    ({"valuation": {**UNPROFITABLE, "ps_trailing": 10.1}, "profitability": {"net_margin": -0.1}}, True),
])
def test_the_quoted_thresholds_are_the_gate_boundaries(fund: dict[str, Any], headwind: bool) -> None:
    """RF-E1 (boundaries): the quoted thresholds are the gate's own: P/E 50, PEG 3.0 and an unprofitable P/S 10
    are not headwind, just above them is (action_zones' named constants feed the gate and the text). G."""
    assessment, _ = az._build_valuation_assessment(fund_data=fund, signals={"bullish": [], "bearish": []})
    assert (assessment["gate"] == "headwind") is headwind


async def test_a_peg_headwind_binds_with_its_own_text() -> None:
    """RF-E1 (pipeline): a PEG headwind (which had no text before) binds the long-term wait with "PEG <= 3.0";
    the mid-term starter list stays empty. RF."""
    assert lists(await dm.run(LOCAL["C1_peg"], fundamentals={
        **PEG_HEADWIND, "financial_health": {"debt_to_equity": 0.5}}))[0::3] == ([], [PEG])


@pytest.mark.parametrize("name", ["C3_headwind_plain_avoid", "F3_quality_neutral_valuation_poor_chart"])
async def test_poor_setup_avoids_carry_no_valuation_condition(name: str) -> None:
    """RF-E1: on a poor-setup avoid (C3 headwind, F3 neutral valuation) no valuation condition appears in either
    horizon's lists or in core's conditionals (clearing the headwind alone admits nothing); C3 monitors it. RF."""
    result = await dm.run(name)
    buy, add, monitor, long = lists(result)
    listed = buy + add + long + result["decision_modes"]["core"]["required_before_full_size"]
    assert not [t for t in listed if t.startswith(("P/E", "PEG", "EV/S", "P/S"))]
    assert monitor == ([PE] if name.startswith("C3") else [])


async def test_a_zone_capping_headwind_binds_starter_to_full_at_mid() -> None:
    """RF-E1: a headwind that caps the zone (accumulate -> hold_neutral) holds a bullish ok-fit candidate at
    hold_or_add (starter); clearing it alone gives buy, so it is the starter -> full condition at mid term and
    binds the long-term wait. Known debt makes the thesis assessable, so the holder also receives the add condition. RF."""
    result = await dm.run(LOCAL["S2_headwind"], fundamentals={
        **dm.HEADWIND, "financial_health": {"debt_to_equity": 0.5}})
    assert result["decision_card"]["if_held"]["basis"] == "candidate_admits_new_money"
    assert lists(result) == ([PE], [PE], [], [PE])


# --- RF-E2 [FIX BL-E2] ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("level", "factors", "expected"), [
    ("extreme", {"volatility": "extreme"}, VOL60),
    ("extreme", {"drawdown": "extreme"}, DD50),
    ("extreme", {"beta": "extreme"}, BETA18),
    ("extreme", {"volatility": "extreme", "beta": "extreme"}, "risk_regime <= high (volatility < 60% and beta < 1.8)"),
    ("high", {"drawdown": "high", "volatility": "medium"}, DD35),
    ("high", None, "risk_regime <= medium"),
])
def test_risk_text_names_the_regime_driver(level: str, factors: dict[str, str] | None, expected: str) -> None:
    """RF-E2 (unit): the risk condition names every factor at the regime's level, with the next-lower regime's
    threshold (volatility, drawdown, beta), and no parenthetical when no factor is known. RF."""
    assert risk_text(level, factors) == expected


@pytest.mark.parametrize(("inputs", "factors"), [
    ((0.60, None, None), {"volatility": "extreme"}), ((0.599, None, None), {"volatility": "high"}),
    ((None, -0.50, None), {"drawdown": "extreme"}), ((None, -0.499, None), {"drawdown": "high"}),
    ((None, None, 1.8), {"beta": "extreme"}), ((None, None, 1.79), {"beta": "high"}),
])
def test_the_driver_thresholds_are_the_regime_boundaries(inputs: tuple[Any, ...], factors: dict[str, str]) -> None:
    """RF-E2 (boundaries): the quoted thresholds are classify_risk_regime's boundaries (signals' named
    constants). G."""
    regime = classify_risk_regime(*inputs)
    assert {k: v for k, v in regime["factors"].items() if v} == factors


@pytest.mark.parametrize(("name", "expected"), [
    ("T1_drawdown", ([], [], [DD50], [DD50])), ("T1_beta", ([], [], [BETA18], [BETA18])),
    ("S2_drawdown", ([DD50], [DD50], [], [DD50])),
    ("S6_knife_dislocation", ([], [], [DD35], [KNIFE])),
])
async def test_binding_risk_condition_names_its_driver(name: str, expected: tuple[list[str], ...]) -> None:
    """RF-E2 (pipeline): a drawdown- or beta-driven extreme regime binds with its own driver (T1 at long term and
    on core, monitored at mid term where it cannot lift the starter to buy; S2's starter -> full at mid term); S6's
    high regime is driven by its drawdown, not its 35% volatility. RF."""
    result = await run(name)
    assert lists(result) == expected
    assert result["decision_modes"]["core"]["required_before_full_size"] == expected[3]


@pytest.mark.parametrize("name", [*dm.SCENARIOS, *LOCAL])
async def test_no_volatility_condition_without_a_volatility_driver(name: str) -> None:
    """RF-E2: no condition on the card, the modes or the dislocation says "volatility <" unless the volatility
    factor is at the regime's level. RF."""
    result = await run(name)
    regime = result["risk_summary"].get("risk_regime") or {}
    volatility = (regime.get("factors") or {}).get("volatility")
    action = result["dislocation_framework"]["action"]
    texts = [*itertools.chain(*lists(result)), *result["decision_modes"]["core"]["required_before_full_size"],
             *result["decision_modes"]["balanced"]["add_on_only_if"],
             *action.get("buy_only_if", []), *action.get("add_only_if", [])]
    assert not [t for t in texts if "volatility <" in t] or volatility == regime.get("classification")


# --- OC-RF7: the conditions truth table, per horizon ----------------------------------------------------------

EMPTY: tuple[list[str], ...] = ([], [], [], [])
TRUTH_TABLE = {  # (mid buy_only_if, mid add_only_if, monitor, long buy_only_if)
    "S1_hold_or_add_clean": ([TU], [], [], [TU]),
    "S2_buy_clean": ([TU], [], [], [TU]),
    "S3_plain_avoid": ([], [], [], [TU]),
    "T3_plain_wait_weak_setup": ([TU], [], [], [TU]),
    "V1_vendor_rich_plain_avoid": ([], [], [], [TU]),
    "S4_earnings_gate_on_hold_or_add": ([TU, EARN], [], [], [TU, EARN]),
    "S5_earnings_gate_on_avoid": ([], [], [], [TU, EARN]),
    "S6_knife_dislocation": ([], [], [DD35], [KNIFE]),
    "S8_fundamentals_failed": ([], [], ["revenue data available for valuation"], []),
    "S9_weak_liquidity": ([TU, LIQ], [], [], [TU, LIQ]),
    "T1_extreme_risk_bullish": ([], [], [VOL60], [VOL60]),
    "C1_headwind_hold_or_add": ([TU], [], [PE], [TU, PE]),
    "C2_headwind_earnings_gate": ([TU, EARN], [], [PE], [TU, PE]),
    "C3_headwind_plain_avoid": ([], [], [PE], [TU]),
    "A_broken_bearish": ([], [], [DE, REV], []),
    "B_broken_bearish_knife": ([], [], [DE, REV], []),
    "C_quality_knife_dislocation": ([], [], [], [KNIFE]),
    "D2_quality_poor_chart_earnings_3d": ([], [], [], [EARN]),
    "E_poor_business_bullish_chart": ([BB], [], [DE], [DE]),
    "E2_mixed_fcf_bullish_chart": ([], [], [FCF], []),
    "A7_near_52w_high_bullish": ([TU, NEAR], [], [], [TU]),
    "H_burner_runway_2q": ([RC], [], [PM, FC, FCF], [RC]),
    "S2_headwind": ([TU], [], [PE], [TU, PE]),
    "T5_dislocation_starter": ([], [], [VOL60], [VOL60]),
    "T6_dislocation_starter_extreme": ([VOL60], [VOL60], [], [VOL60]),
    "T1_near_52w_high": ([NEAR], [NEAR], [VOL60], [VOL60]),
    "T1_crash_risk": ([], [], [DD35], []),
}


@pytest.mark.parametrize("name", [*dm.SCENARIOS, "S2_headwind"])
async def test_conditions_truth_table(name: str) -> None:
    """OC-RF7 (plan §7e, §0 #15, #25): each row's mid-term buy/add, monitor and long-term lists (every other matrix
    row lists nothing); each horizon's target follows its own action (T1 near its high: mid wait, long starter; T1
    crash risk: mid starter, long buy); do_not_buy_if is new_money's blocker reasons; reduce_if the candidate's
    downgrade conditions; add_only_if equals buy_only_if iff if_held is hold; monitor never repeats a listed
    condition; a buy lists nothing at either horizon; every value is a predicate (R-LEX-3) at every tier; core
    mirrors long-term, balanced, speculative, dislocation and dip mirror the card's mid-term lists. RF."""
    result = await run(name)
    card = result["decision_card"]
    new_money, held, conditions = card["new_money"], card["if_held"], card["conditions"]
    pre_gate = result["diagnostics"]["pre_gate_policy"]
    modes = result["decision_modes"]

    assert lists(result) == TRUTH_TABLE.get(name, EMPTY)
    if name in {"S4_earnings_gate_on_hold_or_add", "S9_weak_liquidity", "C2_headwind_earnings_gate",
                "A7_near_52w_high_bullish", "S2_headwind"}:
        assert (held["action"], held["basis"]) == ("insufficient_data", "thesis_unknown")
    assert conditions["do_not_buy_if"] == [b["reason"] for b in new_money["blocked_by"]]
    assert conditions["reduce_if"] == pre_gate["conditions_to_downgrade"]
    assert conditions["add_only_if"] == (conditions["buy_only_if"] if held["action"] == "hold" else [])
    assert not set(conditions["monitor"]) & set(conditions["buy_only_if"] + conditions["add_only_if"])
    assert set(conditions["monitor"]) <= set(pre_gate["conditions_to_upgrade"])
    if new_money["action"] == "buy":
        assert conditions["buy_only_if"] == conditions["add_only_if"] == conditions["do_not_buy_if"] == []
    if card["long_term"]["new_money"]["action"] == "buy":
        assert lists(result)[3] == modes["core"]["required_before_full_size"] == []
    for tier, shown in dm.tiers(result).items():
        assert oc.conditional_violations(shown) == [], tier
    assert modes["core"]["required_before_full_size"] == card["long_term"]["new_money"]["conditions"]["buy_only_if"]
    assert modes["balanced"]["add_on_only_if"] == modes["speculative"]["add_on_only_if"] == conditions["add_only_if"]
    action = result["dislocation_framework"]["action"]
    if result["dislocation_framework"].get("status") != "not_applicable":
        assert (action["buy_only_if"], action["add_only_if"]) == (conditions["buy_only_if"], conditions["add_only_if"])
    dip = (result["dip_assessment"] or {}).get("assessment") or {}
    if "add_only_if" in dip:
        assert dip["add_only_if"] == conditions["add_only_if"]


def _inputs(fit: dict[str, Any], *, tilt: str = "bullish", valuation_gate: str = "attractive",
            is_unprofitable: bool = False) -> CandidateInputs:
    """A synthetic candidate's inputs (an accumulate zone, no dislocation, no dip type)."""
    fit = {"business_quality_status": "evaluated", "signals": {"bullish": [], "bearish": []}, **fit}
    return CandidateInputs(fit=fit, tilt=tilt, valuation_gate=valuation_gate, is_unprofitable=is_unprofitable,
                           dislocation=False, uncapped_zone="accumulate", dip_type=None)


def _units(policy: dict[str, str], blocking: list[dict[str, Any]], support: dict[str, Any] | None = None,
           factors: tuple[Factor, ...] = (), **kwargs: Any) -> tuple[list[str], list[str], list[str]]:
    decision = dm.decide_on(policy, blocking, support=support, **kwargs)
    built = build_conditions(decision=decision, candidate={"long_term": "insufficient_data", **policy},
                             factors=list(factors), inputs=_inputs({"setup_label": "strong", "risk_label": "high",
                                                                    "business_quality_label": "strong"}),
                             downgrade=[], support=support)
    return built["mid_term"]["buy_only_if"], built["long_term"]["buy_only_if"], built["mid_term"]["monitor"]


BLACKOUT = {"id": "earnings_blackout", "reason": "earnings in 2 days"}
RISK_HIGH = Factor("risk_high", (DD35,), {"risk_label": "medium"})


def test_only_a_tighter_gate_binds() -> None:
    """OC-RF7 (units): a blackout binds a starter or buy candidate (wait is tighter) at both horizons, not a wait
    one; a thesis gate binds a wait (avoid is tighter); the two freshness gates share one clears_when, listed
    once; support gates are never listed, and a permanent one empties the horizon (monitor too: a monitored
    risk factor goes) while an unverified one keeps the other binding gates and the monitored factor. RF."""
    assert _units({"mid_term": "hold_or_add", "long_term": "accumulate"}, [BLACKOUT])[:2] == ([EARN], [EARN])
    assert _units({"mid_term": "hold", "long_term": "hold"}, [BLACKOUT])[:2] == ([], [])
    thesis = {"id": "thesis_broken", "reason": "r", "codes": ["business_broken"]}
    assert _units({"mid_term": "hold", "long_term": "hold"}, [thesis], thesis=("business_broken",))[:2] == ([BB], [BB])
    freshness = [{"id": "stale_data", "reason": "r", "codes": ["daily_stale"]},
                 {"id": "freshness_unverifiable", "reason": "r", "codes": ["daily_latest_bar_invalid"]}]
    assert GATES["stale_data"].clears_when == GATES["freshness_unverifiable"].clears_when
    assert _units({"mid_term": "hold_or_add"}, freshness)[0] == [GATES["stale_data"].clears_when]
    support = {"id": "unsupported_instrument", "reason": "ETF"}
    unverified = {"instrument": {"status": "unverified"}, "currency": {"status": "supported"}}
    unsupported = {"instrument": {"status": "unsupported"}, "currency": {"status": "supported"}}
    buy = {"mid_term": "buy", "long_term": "accumulate"}
    assert _units(buy, [support, BLACKOUT], unverified, (RISK_HIGH,)) == ([EARN], [EARN], [DD35])
    assert _units(buy, [support, BLACKOUT], unsupported, (RISK_HIGH,)) == ([], [], [])


# --- G: invariants of the conditions engine -------------------------------------------------------------------


def test_every_condition_text_is_a_predicate() -> None:
    """G-C1: every text a factor, a gate (mid or long applicable), the thesis components or the permanent support
    gate can emit is a predicate with no advice phrase (R-LEX-3, OC-G1's form). G."""
    texts = {risk_text(level, dict.fromkeys(("volatility", "drawdown", "beta"), level)) for level in ("extreme", "high")}
    texts |= set(valuation_headwind_texts({"basis": "ev_to_sales"}, [
        "valuation_extended_high_pe", "valuation_extended_high_peg", "valuation_extended_high_sales_multiple_unprofitable"]))
    for quality, leverage, gate, reason in itertools.product(("poor", "weak"), ("negative_equity", None),
                                                             ("headwind", "unknown"),
                                                             ("ps_unverified", "ev_nonpositive", "none")):
        for factor in candidate_factors(
                decomposed={"risk": "extreme", "business_quality": quality,
                            "business_quality_status": "evaluated_unprofitable"},
                valuation_assessment={"gate": gate, "reasons": [reason]}, zone_warnings=[],
                risk_regime=None, long_term_gates=["burn_metrics_missing", "low_runway_confidence",
                                                   "severe_revenue_decline", "negative_fcf"],
                leverage_status=leverage):
            texts |= set(factor.texts)
    texts |= {g.clears_when for g in GATES.values() if g.outcome["mid_term"] != "not_applicable"
              or g.outcome["long_term"] != "not_applicable"}
    texts |= {*THESIS_CLEARS_WHEN.values(), PERMANENT_CLEARS_WHEN}
    assert not [t for t in texts if oc.PREDICATE_TOKEN.search(t) or oc.IMPERATIVE.match(t) or oc.advice(t)]


@pytest.mark.parametrize("name", [*dm.SCENARIOS, *LOCAL])
async def test_the_engine_replays_what_the_candidate_read(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """G-C2: the binding engine's inputs reproduce the run: build_horizon_fit(inputs.fit) is the verdict's fit;
    zone_after_caps gives the candidate's zone; candidate_codes with nothing cleared gives the uncapped candidate;
    the factors' texts are the candidate's conditions_to_upgrade. G."""
    captured: dict[str, Any] = {}
    real = orchestrator.build_conditions

    def spy(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs, candidate=dict(kwargs["candidate"]))
        return real(**kwargs)

    monkeypatch.setattr(orchestrator, "build_conditions", spy)
    result = await run(name)
    inputs, pre_gate = captured["inputs"], result["diagnostics"]["pre_gate_policy"]

    assert build_horizon_fit(**inputs.fit) == result["verdict"]["horizon_fit"]
    assert zone_after_caps(inputs.uncapped_zone, extreme=inputs.fit["risk_label"] == "extreme",
                           valuation_gate=inputs.valuation_gate, dip_type=inputs.dip_type) == pre_gate["inputs"]["zone"]
    candidate = captured["candidate"]
    assert candidate_codes(inputs, {}) == (candidate["mid_term"], candidate["long_term"])
    assert [t for f in captured["factors"] for t in f.texts] == pre_gate["conditions_to_upgrade"]


def _cleared(factor_id: str, fit: dict[str, Any], positives: tuple[str, ...] = (), **kwargs: Any) -> tuple[Any, ...]:
    """(candidate codes, the codes with factor_id's own cleared state) over a synthetic candidate."""
    inputs = _inputs(fit, **kwargs)
    factors = candidate_factors(
        decomposed={"risk": fit["risk_label"], "business_quality": fit["business_quality_label"],
                    "business_quality_status": inputs.fit["business_quality_status"],
                    "business_quality_evidence": {"positives": list(positives)}},
        valuation_assessment={"gate": inputs.valuation_gate, "is_unprofitable": inputs.is_unprofitable},
        zone_warnings=[], risk_regime=None, long_term_gates=build_horizon_fit(**inputs.fit)["long_term_gates"],
        leverage_status=None)
    factor = next(f for f in factors if f.id == factor_id)
    return candidate_codes(inputs, {}), candidate_codes(inputs, factor.overrides)


def test_each_factor_replays_its_own_cleared_state() -> None:
    """G-C2 (cleared states, spec-conditions §6 #4-5): clearing unprofitability alone (label mixed, profitable)
    leaves an unprofitable, collapsing, neutral-tilt long term at wait_or_avoid, not a speculative entry; clearing
    negative FCF alone lifts the extreme-risk, declining, cash-burning avoid to caution; business_poor clears to
    moderate when the rubric found a positive (long-term ok) and to mixed when it found none. G."""
    collapsing = {"setup_label": "weak", "business_quality_label": "unprofitable",
                  "business_quality_status": "evaluated_unprofitable", "risk_label": "medium", "revenue_yoy": -0.35,
                  "fcf_ttm": -2e8, "burn_metrics_status": "available"}
    assert _cleared("unprofitable", collapsing, tilt="neutral", valuation_gate="neutral", is_unprofitable=True) == (
        ("wait_for_entry", "avoid_core_position"), ("wait_for_entry", "wait_or_avoid"))
    triple = {"setup_label": "strong", "business_quality_label": "moderate", "risk_label": "extreme",
              "revenue_yoy": -0.25, "fcf_ttm": -1e8}
    assert _cleared("negative_fcf", triple) == (("speculative_small_position", "avoid_core_position"),
                                                ("speculative_small_position", "small_position_with_stops"))
    poor = {"setup_label": "strong", "business_quality_label": "poor", "risk_label": "medium", "revenue_yoy": 0.05,
            "fcf_ttm": 0.0}
    assert _cleared("business_poor", poor, ("net_margin=15.0%",)) == (("buy", "avoid_core_position"),
                                                                     ("buy", "accumulate"))
    assert _cleared("business_poor", poor) == (("buy", "avoid_core_position"), ("buy", "small_position_with_stops"))


@pytest.mark.parametrize(("uncapped", "gate", "dip_type", "extreme"), list(itertools.product(
    ("strong_buy", "accumulate", "hold_bullish", "hold_neutral", "reduce", "undetermined"), ("headwind", "neutral"),
    (None, "falling_knife", "extended_decline"), (True, False))))
def test_zone_after_caps_replays_the_cap_chain(uncapped: str, gate: str, dip_type: str | None, extreme: bool) -> None:
    """G-C5: zone_after_caps equals the real chain (build-time extreme cap, valuation downgrade, dip gates). G."""
    zone = "accumulate" if extreme and uncapped == "strong_buy" else uncapped
    zone, _ = _apply_valuation_gate_to_zone(zone, gate)
    zone = apply_dip_gates_to_action_zones({"current_zone": zone}, {"dip_classification": {"type": dip_type}},
                                           {"classification": "extreme" if extreme else "medium"})["current_zone"]
    assert zone_after_caps(uncapped, extreme=extreme, valuation_gate=gate, dip_type=dip_type) == zone


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_a_headwind_has_a_text(name: str) -> None:
    """G-C6: the valuation gate is headwind iff a headwind code is in zone_warnings, so the headwind factor always
    carries a text. G."""
    zones = (await dm.run(name))["action_zones"]
    codes = {"valuation_extended_high_pe", "valuation_extended_high_peg",
             "valuation_extended_high_sales_multiple_unprofitable"}
    assert (zones["valuation_assessment"]["gate"] == "headwind") is bool(codes & set(zones["zone_warnings"]))
