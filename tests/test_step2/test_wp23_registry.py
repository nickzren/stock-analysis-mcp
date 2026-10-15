"""WP2.3: the closed field-class registry (plan §7 "Field classes") and the lints it drives; the D-text rewrites
(§7c); BL-E3 (why, exit_triggers, data_flags); the restated freshness invariant; FR-R-F12; RA.11.

Annex: docs/plans/step2-v2-research/output_contract.md §1-§2, §10 "Corrected proposal" as amended by plan §0
(#8 long-term vocabulary, #22 interim snapshot: watchlist_snapshot is excluded until WP2.4). The registry and the
lints live in tests/support/output_contract.py.
"""

import importlib
import re
from typing import Any

import pytest

from stock_analysis.prompts.templates import PROMPTS, get_prompt
from stock_analysis.resources.analyze_guide import ANALYZE_RENDERING_GUIDE
from stock_analysis.tools.analyze import action_zones as az
from stock_analysis.tools.analyze import dip_assessment as dip
from stock_analysis.tools.analyze.decision import (
    ENTRY_ACTIONS,
    ENTRY_CANDIDATE_CODES,
    ENTRY_CODES,
    FULL_ENTRY_CODES,
    STARTER_ONLY_CODES,
)
from stock_analysis.tools.analyze.verdict import build_horizon_fit
from tests.support import decision_matrix as dm
from tests.support import freshness_frames as fr
from tests.support import output_contract as oc

server = importlib.import_module("stock_analysis.server")
orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
FRESHNESS_IDS = ("stale_data", "freshness_unverifiable")
BLOCKING_ROWS = [name for name, row in fr.TRUTH_TABLE.items() if row.codes]
REENTRY = re.compile(r"(?i)re-?entr|re-?enter|resum")


async def error_result() -> dict[str, Any]:
    """S2 with stock_summary and technicals failing: the top-level error keys and the missing-price zones."""
    scenario = dm.SCENARIOS["S2_buy_clean"]
    with dm.stubbed(scenario):
        async def fail(symbol: str, **kwargs: Any) -> dict[str, Any]:
            raise TimeoutError("stubbed tool failure")
        orchestrator.stock_summary, orchestrator.technicals = fail, fail
        return await orchestrator.analyze_stock("ACME", account_size=10_000.0, _now=scenario.now)


EXTRA = {"error": error_result, "R13": lambda: dm.run(fr.scenario(fr.TRUTH_TABLE["R13"], "R13"))}

# --- G: the registry itself ----------------------------------------------------------------------------------


def test_the_registry_is_well_formed() -> None:
    """G-REG1: unique globs, known classes and tiers, a horizon on every A and S row (optional on C), P only as
    the full-tier diagnostics.**, no T row under decision_card, and the WP2.4 snapshot exclusion. G."""
    globs = [f.glob for f in oc.FIELD_REGISTRY]
    assert len(globs) == len(set(globs))
    for f in oc.FIELD_REGISTRY:
        assert f.cls in oc.CLASSES and f.tiers and f.tiers <= oc.ALL, f
        assert (f.horizon is not None) is (f.cls in (oc.A, oc.S)) or f.cls == oc.C, f
        assert not f.glob.startswith(("*", ".")), f
    assert [f.glob for f in oc.FIELD_REGISTRY if f.cls == oc.P] == ["diagnostics.**"]
    assert all(f.tiers == oc.FULL for f in oc.FIELD_REGISTRY if f.cls == oc.P)
    assert not [f for f in oc.FIELD_REGISTRY if f.cls == oc.T and f.glob.startswith("decision_card")]
    assert oc.INTERIM_EXCLUDED_ROOTS == ("watchlist_snapshot",)


def test_leaves_and_globs() -> None:
    """G-REG2: an empty container is a leaf at its own path, list items collapse to [*], and globs are
    root-anchored (options_signals.meta.tool is T, meta.tool is D). G."""
    assert oc.leaves({"a": {}, "b": [], "c": [{"d": 1}]}) == [("a", {}), ("b", []), ("c[*].d", 1)]
    assert (oc.classify("options_signals.meta.tool").cls, oc.classify("meta.tool").cls) == (oc.T, oc.D)
    assert oc.classify("decision_modes.core.label").cls == oc.E
    with pytest.raises(oc.UnregisteredPathError):
        oc.classify("policy_action.mid_term")


ANNEX_ENTRY_CODES = {  # annex output_contract §0, 22 codes
    "buy", "starter", "hold_or_add", "accumulate",
    "speculative_small_position", "speculative_with_thesis", "speculative_only", "small_position_with_stops",
    "build_position", "starter_position_only", "starter_then_add",
    "strong_buy_the_dip", "buy_the_dip", "cautious_accumulation", "small_speculative_position",
    "starter_only_wait_for_stabilization",
    "consider_small_position", "starter_position", "add_position",
    "strong_buy", "small_starter", "starter_only",
}
ANNEX_STARTER_ONLY_CODES = {  # annex output_contract §0, 10 codes
    "starter", "speculative_small_position", "small_position_with_stops", "starter_position_only",
    "small_speculative_position", "starter_only_wait_for_stabilization", "consider_small_position",
    "starter_position", "small_starter", "starter_only",
}


def test_the_code_sets() -> None:
    """G-REG3: ENTRY_CODES and STARTER_ONLY_CODES are the annex output_contract §0 literals; ENTRY_CODES is the
    disjoint union of the starter-only and full-entry codes and holds every new-money entry action and entry
    candidate code. G."""
    assert (ENTRY_CODES, STARTER_ONLY_CODES) == (ANNEX_ENTRY_CODES, ANNEX_STARTER_ONLY_CODES)
    assert ENTRY_CODES == STARTER_ONLY_CODES | FULL_ENTRY_CODES and not STARTER_ONLY_CODES & FULL_ENTRY_CODES
    assert ENTRY_ACTIONS | ENTRY_CANDIDATE_CODES <= ENTRY_CODES


# --- OC-RF1 ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [*dm.SCENARIOS, *EXTRA])
async def test_every_leaf_matches_exactly_one_row(name: str) -> None:
    """OC-RF1 (non-snapshot paths, plan §0 #22): every leaf of every matrix scenario (V1's vendor subtrees
    included), an error result and a failed-technicals run (R13) matches exactly one registry row, at a tier the
    row allows. RF."""
    result = await (EXTRA[name]() if name in EXTRA else dm.run(name))
    for tier, shown in dm.tiers(result).items():
        assert oc.registry_violations(shown, tier) == [], tier


async def test_every_row_is_exercised() -> None:
    """OC-RF1b: every registry row matches some leaf over the matrix and the extra results, except the listed
    UNEXERCISED_ROWS. RF."""
    results = [*[await dm.run(name) for name in dm.SCENARIOS], *[await make() for make in EXTRA.values()]]
    used = {oc.classify(path).glob for result in results for shown in dm.tiers(result).values()
            for path, _ in oc.contract_leaves(shown)}
    assert {f.glob for f in oc.FIELD_REGISTRY} - used == set(oc.UNEXERCISED_ROWS)


# --- OC-RF4 ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_actionable_codes_follow_their_horizon(name: str) -> None:
    """OC-RF4 (plan §0 #8, #21, #23): an A value with an entry code or advice phrase needs its horizon's
    predicate; a full-entry code in a mid-term field needs new_money buy; a starter_only label needs a starter;
    every capped_by names the current action of its horizon. RF."""
    result = await dm.run(name)
    for tier, shown in dm.tiers(result).items():
        assert oc.actionable_violations(shown, result["decision_card"]) == [], tier


async def test_a_capped_dip_names_its_cap() -> None:
    """OC-RF4 (capped_by is not vacuous): S3's dip recommendation and both entry signals are capped to watch by
    new_money avoid. RF."""
    dip_assessment = (await dm.run("S3_plain_avoid"))["dip_assessment"]
    assert (dip_assessment["assessment"]["recommendation"], dip_assessment["assessment"]["capped_by"]) == (
        "watch", "new_money_avoid")
    assert {(s["action"], s["capped_by"]) for s in dip_assessment["entry_timing"]["signals"]} == {
        ("watch", "new_money_avoid")}


# --- OC-RF10: the D lexicon (plan §7c) ---------------------------------------------------------------------

@pytest.mark.parametrize("name", [*dm.SCENARIOS, "extreme_starter"])
async def test_diagnostics_carry_no_advice(name: str) -> None:
    """OC-RF10: no D leaf at any tier (verdict reasons, horizon drivers, zone_warnings, the band rationale, the
    dip rationales and explanation, ...) carries an entry code or advice phrase (R-LEX-2). "extreme_starter" is T1
    with the starter-dip chart, so the extreme-regime strings show. PD D-9."""
    result = await (dm.run("T1_extreme_risk_bullish", technicals=dm.STARTER_DIP) if name == "extreme_starter"
                    else dm.run(name))
    for tier, shown in dm.tiers(result).items():
        assert oc.diagnostic_violations(shown) == [], tier


FIT_CASES = {  # build_horizon_fit branch -> rewritten reason (plan §7c: verdict.py's four sizing reasons)
    "extreme": ({"setup_label": "strong", "risk_label": "extreme"}, "mid_term: extreme risk regime"),
    "high": ({"setup_label": "strong", "risk_label": "high"}, "mid_term: setup favorable, high risk regime"),
    "standard": ({"setup_label": "strong", "risk_label": "medium"}, "mid_term: setup favorable"),
    "mixed": ({"setup_label": "strong", "risk_label": "medium", "business_quality_label": "mixed"},
              "long_term: mixed fundamentals"),
}


@pytest.mark.parametrize("case", FIT_CASES)
def test_horizon_fit_reasons_are_diagnostic(case: str) -> None:
    """OC-RF10 (unit): each rewritten horizon-fit reason, and no reason carries advice. PD D-9."""
    kwargs, reason = FIT_CASES[case]
    fit = build_horizon_fit(**{"business_quality_label": "strong", "business_quality_status": "available",
                               "signals": {"bullish": [], "bearish": []}, **kwargs})
    assert reason in fit["reasons"]
    assert [r for r in fit["reasons"] if oc.advice(r)] == []


def test_band_rationales_and_regime_warnings_name_the_regime() -> None:
    """OC-RF10 (unit): the band rationale codes and the zone_warnings regime entries are regime names. PD D-9."""
    assert {r: az._base_sizing_range(r)[2]["rationale"] for r in ("extreme", "high", "medium", "low", None)} == {
        "extreme": "extreme_risk_regime", "high": "high_risk_regime", "medium": "medium_risk_regime",
        "low": "low_risk_regime", None: "low_risk_regime"}
    for extreme, high, expected in ((True, False, ["extreme_risk_regime"]), (False, True, ["high_risk_regime"])):
        warnings: list[str] = []
        az._append_regime_warnings(warnings, extreme, high)
        assert warnings == expected


def test_dip_text_is_diagnostic() -> None:
    """OC-RF10 (unit): every dip explanation and recommendation rationale, and the dislocation starter override's
    rationale (plan §7c), carry no entry code or advice phrase. PD D-9."""
    texts = [*dip._DIP_TYPE_EXPLANATIONS.values(), *dip._BUY_RECOMMENDATION_RATIONALES.values()]
    overridden = dip.align_dip_assessment_with_action(
        {"assessment": {"recommendation": "do_not_catch_falling_knife"}}, "price_broken_more_than_business",
        starter_only=True)
    assert overridden is not None
    assert overridden["assessment"]["rationale"] == (
        "Price looks more broken than the business, but the trend is still broken: wait for stabilization.")
    assert [t for t in [*texts, overridden["assessment"]["rationale"]] if oc.advice(t)] == []


# --- The freshness invariant, restated (plan §10 WP2.3) ------------------------------------------------------

@pytest.mark.parametrize("name", BLOCKING_ROWS)
async def test_no_entry_wording_under_a_freshness_gate(name: str) -> None:
    """FI (plan §10, non-snapshot paths): when stale_data or freshness_unverifiable fires, no A, S, N or C field at
    any tier carries an entry code or advice phrase (S: no value at all); P and T are exempt. RF."""
    result = await dm.run(fr.scenario(fr.TRUTH_TABLE[name], name))
    assert {b["id"] for b in result["decision_card"]["new_money"]["blocked_by"]} & set(FRESHNESS_IDS)
    for tier, shown in dm.tiers(result).items():
        assert oc.freshness_violations(shown) == [], tier


# --- FR-R-F12 ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["R3", "R6", "O2", "R13", "R14", "O8"])
async def test_freshness_blockers_reach_every_tier(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-R-F12 (plan §0 #4): the freshness ids and their codes reach new_money and long_term at every tier, with
    data_quality.freshness; if_held's action and basis equal the same run without the gate. RF (red on 0609fde,
    which has no freshness id and no freshness decision key; WP2.1/WP2.2 already satisfy it)."""
    row = fr.TRUTH_TABLE[name]
    scenario = fr.scenario(row, name)
    gated = await dm.run(scenario)
    real = orchestrator.check_price_freshness
    monkeypatch.setattr(orchestrator, "check_price_freshness", lambda **kw: ([], real(**kw)[1]))
    ungated = await dm.run(scenario)

    def fresh(blocked: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
        return [(b["id"], b["codes"]) for b in blocked if b["id"] in FRESHNESS_IDS]

    expected = fresh(gated["decision_card"]["new_money"]["blocked_by"])
    assert sorted(c for _, codes in expected for c in codes) == sorted(row.codes)
    for tier, shown in dm.tiers(gated).items():
        card = shown["decision_card"]
        assert fresh(card["new_money"]["blocked_by"]) == fresh(card["long_term"]["new_money"]["blocked_by"]) == (
            expected), tier
        assert shown["data_quality"]["freshness"]["codes"] == list(row.codes), tier
    held, unheld = gated["decision_card"]["if_held"], ungated["decision_card"]["if_held"]
    assert (held["action"], held["basis"]) == (unheld["action"], unheld["basis"])
    assert held["add_allowed"] is False


# --- RA.11 -------------------------------------------------------------------------------------------------

def test_guide_prompts_and_tool_description_have_no_reentry_wording() -> None:
    """RA.11 (plan §0 #11: the trade-card cases are dropped): no re-entry or resume wording in the rendering
    guide, any prompt or the analyze tool description; the analyze tiers are test_wp22_reassessment's RA.11 pin.
    G (no such wording on 0609fde either; mutation: a "resume" in the guide)."""
    args = {"symbol": "ACME", "cost_basis": "80", "purchase_date": "2025-01-02"}
    texts = [ANALYZE_RENDERING_GUIDE, *(repr(get_prompt(n, args)) for n in PROMPTS),
             server.analyze.__doc__ or ""]
    assert [t[:40] for t in texts if REENTRY.search(t)] == []


# --- BL-E3: why, exit_triggers, data_flags (plan §2; WP2.2 deferral) -----------------------------------------

def _flags(result: dict[str, Any]) -> list[str]:
    data_quality, zones = result["data_quality"], result["action_zones"]
    status = data_quality["fundamentals_status"]
    return list(dict.fromkeys([*([] if status in (None, "available") else [f"fundamentals_{status}"]),
                               *data_quality["staleness_warnings"], *zones["zone_warnings"]]))


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_the_card_explains_itself(name: str) -> None:
    """BL-E3: at every tier the card's why is the deciding gate, the mid-term evidence (fit, tilt, price zone) and
    the other fired gates; exit_triggers are the thesis stop triggers (none while fundamentals are missing);
    data_flags are the fundamentals status, staleness and zone warnings; balanced.why opens with the card's why.
    why carries no gate reason or pre-gate code (R-LEX-2 strength), exit_triggers are predicates (R-LEX-3),
    data_flags are diagnostic. RF."""
    result = await dm.run(name)
    new_money, verdict = result["decision_card"]["new_money"], result["verdict"]
    others = [b["id"] for b in new_money["blocked_by"] if b["id"] != new_money["decided_by"]]
    zone = result["action_zones"].get("price_zone")
    evidence = (f"mid-term fit {verdict['horizon_fit']['mid_term']}, tilt {verdict['tilt'] or 'missing'}"
                + (f", price zone {zone}" if zone else ""))
    why = [*([f"decided by {new_money['decided_by']}"] if new_money["decided_by"] else []), evidence,
           *([f"other gates fired: {', '.join(others)}"] if others else [])]
    assert result["decision_modes"]["balanced"]["why"][:len(why)] == why
    thesis = result["dislocation_framework"].get("thesis_integrity") or {}
    missing = result["data_quality"]["fundamentals_status"] == "missing"
    for tier, shown in dm.tiers(result).items():
        card = shown["decision_card"]
        assert card["why"] == why, tier
        assert card["exit_triggers"] == ([] if missing else thesis.get("sell_without_attachment_if", [])), tier
        assert card["data_flags"] == _flags(result), tier
        assert not [line for line in card["why"] if oc.advice(line)], tier
        assert not [t for t in card["exit_triggers"] if oc.PREDICATE_TOKEN.search(t) or oc.IMPERATIVE.match(t)], tier
        assert not [flag for flag in card["data_flags"] if oc.advice(flag)], tier


async def test_a_missing_price_has_no_zone() -> None:
    """OC-RF6 and BL-E3 (missing price, plan §1.2): with stock_summary and technicals failing, action_zones.price_zone
    is null (not undetermined) and why carries no price-zone part, at every tier that shows them. RF."""
    result = await error_result()
    assert result["summary"]["current_price"] is None
    for tier, shown in dm.tiers(result).items():
        if tier != "decision":
            assert shown["action_zones"]["price_zone"] is None, tier
        assert shown["decision_card"]["why"] == [
            "decided by unsupported_instrument", "mid-term fit unknown, tilt missing",
            "other gates fired: freshness_unverifiable, data_quality_critical, unsupported_currency"], tier


STALE = {"R3": {}, "R3_fundamentals_failed": {"fundamentals": None}}  # the R3 freshness run, with overrides


@pytest.mark.parametrize(("name", "field", "expected"), [
    ("S4_earnings_gate_on_hold_or_add", "why",
     ["decided by earnings_blackout", "mid-term fit ok, tilt bullish, price zone above_sma50"]),
    ("S5_earnings_gate_on_avoid", "why",
     ["mid-term fit avoid, tilt bearish, price zone below_sma200", "other gates fired: earnings_blackout"]),
    ("A7_near_52w_high_bullish", "why",
     ["decided by near_52w_high", "mid-term fit ok, tilt bullish, price zone near_52w_high"]),
    ("H_burner_runway_2q", "exit_triggers",
     ["cash_runway < 4 quarters (dilutive raise imminent)",
      "net margin worsens 3 consecutive quarters (path to profitability not progressing)",
      "revenue_yoy < -30% for 2 quarters (growth thesis invalidated)"]),
    ("S7_etf_below_sma200", "exit_triggers", []),
    ("S8_fundamentals_failed", "exit_triggers", []),
    ("T1_extreme_risk_bullish", "data_flags", ["extreme_risk_regime", "zone_capped_extreme_risk"]),
    ("S8_fundamentals_failed", "data_flags", ["fundamentals_missing", "valuation_gate_unknown_no_metrics"]),
    ("H_burner_runway_2q", "data_flags", ["fundamentals_not_meaningful"]),
    ("S1_hold_or_add_clean", "data_flags", []),
    ("R3", "data_flags", ["price_stale_2_sessions"]),
    ("R3_fundamentals_failed", "data_flags",
     ["fundamentals_missing", "price_stale_2_sessions", "valuation_gate_unknown_no_metrics"]),
])
async def test_pinned_explanations(name: str, field: str, expected: list[str]) -> None:
    """BL-E3 (pinned values): S4 decided by the blackout; S5's blackout fired but did not decide; A7's price
    zone; H's three runway triggers; none on an ETF or with fundamentals missing; T1's regime flags; S8's
    missing fundamentals; H's not-meaningful fundamentals; none on a clean row; R3's stale price; R3 with
    fundamentals failing lists the fundamentals status, then staleness, then zone warnings (plan §1.5). RF."""
    run = dm.run(fr.scenario(fr.TRUTH_TABLE["R3"], "R3"), **STALE[name]) if name in STALE else dm.run(name)
    assert (await run)["decision_card"][field] == expected
