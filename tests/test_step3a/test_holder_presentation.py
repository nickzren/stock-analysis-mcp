"""Long-term ownership prose distinguishes the policy, suitability and additions."""

import copy

import pytest

from stock_analysis import SCHEMA_VERSION
from stock_analysis.tools.analyze.executive_summary import build_executive_summary
from stock_analysis.utils.provenance import build_meta
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm
from tests.support import output_contract as oc


def test_holder_response_schema_is_8():
    assert SCHEMA_VERSION == build_meta("analyze_stock")["schema_version"] == "8"


@pytest.mark.parametrize("name", list(hm.STOCK_CASES))
async def test_summary_distinguishes_long_term_new_money_and_ownership(name):
    result = await hm.run_stock(name)
    long_term = result["decision_card"]["long_term"]
    segment = oc.segments(result["executive_summary"])[2]
    assert segment.startswith("Long-term (1-5y): new money: " + long_term["new_money"]["action"].replace("_", " "))
    assert "; if held: " + long_term["if_held"]["action"].replace("_", " ") in segment
    assert "Position:" not in segment and "planning stop" not in segment and "risk budget:" not in segment
    for tier, payload in dm.tiers(result).items():
        if tier != "decision":
            assert payload["executive_summary"] == result["executive_summary"]


@pytest.mark.parametrize("fit", ["avoid", "caution", "unknown"])
async def test_hold_keeps_the_actual_long_term_fit_reasons(fit):
    result = await hm.run_stock("sound_bullish")
    verdict = copy.deepcopy(result["verdict"])
    verdict["horizon_fit"].update(long_term=fit, reasons=["mid_term: unrelated", "long_term: extreme risk",
                                                       "long_term: revenue decline"])
    summary = build_executive_summary(result["summary"], result["technicals_summary"],
                                     result["fundamentals_summary"], result["risk_summary"],
                                     verdict, result["decision_card"])
    segment = oc.segments(summary)[2]
    assert "if held: hold" in segment
    assert "no demonstrated thesis failure" in segment and "not an endorsement" in segment
    assert f"suitability: {fit}" in segment
    assert "extreme risk" in segment and "revenue decline" in segment
    assert "unrelated" not in segment


@pytest.mark.parametrize("name", ["etf_spy", "etf_vti", "preferred"])
async def test_summary_not_assessed_is_a_coverage_limit(name):
    result = await hm.run_stock(name)
    segment = oc.segments(result["executive_summary"])[2]
    assert "if held: not assessed" in segment
    assert "company rules do not apply" in segment
    assert "missing" not in segment and "cash runway" not in segment
    assert "insufficient data" not in segment and "suitability:" not in segment
    assert "long_term" in result["verdict"]["horizon_fit"]  # Structured legacy fit is not rewritten.


@pytest.mark.parametrize("name,reason", [
    ("critical_runway", "runway critical"), ("broken_bullish", "business broken"),
    ("fundamentals_failed", "business not evaluated"), ("classification_unknown", "applicability unverified"),
])
async def test_holder_segment_keeps_the_actual_basis_and_missing_checks(name, reason):
    segment = oc.segments((await hm.run_stock(name))["executive_summary"])[2]
    assert reason in segment
    if name in {"critical_runway", "broken_bullish"}:
        assert "not an immediate sell instruction" in segment


async def test_observed_hold_with_avoid_fit_retains_all_three_real_reasons():
    result = await hm.run_stock("hold_with_avoid_fit")
    segment = oc.segments(result["executive_summary"])[2]
    assert "if held: hold (no demonstrated thesis failure); not an endorsement" in segment
    assert "suitability: avoid (extreme_risk + revenue_decline_40pct + negative_fcf)" in segment
    assert result["decision_card"]["long_term"]["new_money"]["action"] == "avoid"


async def test_every_tier_preserves_the_existing_fit_and_actual_reasons():
    result = await hm.run_stock("hold_with_avoid_fit")
    for tier, payload in dm.tiers(result).items():
        assert payload["verdict"]["horizon_fit"] == result["verdict"]["horizon_fit"], tier
        assert payload["decision_card"]["long_term"]["if_held"]["action"] == "hold"
