"""WP2.3: the closed field-class registry (plan §7 "Field classes") and the lints it drives; the D-text rewrites
(§7c); BL-E3 (why, exit_triggers, data_flags); the restated freshness invariant; FR-R-F12; RA.11.

Annex: docs/plans/step2-v2-research/output_contract.md §1-§2, §10 "Corrected proposal" as amended by plan §0
(#8 long-term vocabulary, #22 interim snapshot: watchlist_snapshot is excluded until WP2.4). The registry and the
lints live in tests/support/output_contract.py.
"""

import importlib
import re

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
FRESHNESS_IDS = ("stale_data", "freshness_unverifiable")
BLOCKING_ROWS = [name for name, row in fr.TRUTH_TABLE.items() if row.codes]
REENTRY = re.compile(r"(?i)re-?entr|re-?enter|resum")

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


def test_the_code_sets() -> None:
    """G-REG3: ENTRY_CODES is the disjoint union of the starter-only and full-entry codes and holds every
    new-money entry action and entry candidate code (annex output_contract §0). G."""
    assert ENTRY_CODES == STARTER_ONLY_CODES | FULL_ENTRY_CODES and not STARTER_ONLY_CODES & FULL_ENTRY_CODES
    assert ENTRY_ACTIONS | ENTRY_CANDIDATE_CODES <= ENTRY_CODES


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


# --- RA.11 -------------------------------------------------------------------------------------------------

def test_guide_prompts_and_tool_description_have_no_reentry_wording() -> None:
    """RA.11 (plan §0 #11: the trade-card cases are dropped): no re-entry or resume wording in the rendering
    guide, any prompt or the analyze tool description; the analyze tiers are test_wp22_reassessment's RA.11 pin.
    G (no such wording on 0609fde either; mutation: a "resume" in the guide)."""
    args = {"symbol": "ACME", "cost_basis": "80", "purchase_date": "2025-01-02"}
    texts = [ANALYZE_RENDERING_GUIDE, *(repr(get_prompt(n, args)) for n in PROMPTS),
             server.analyze.__doc__ or ""]
    assert [t[:40] for t in texts if REENTRY.search(t)] == []
