"""One stock assessment feeds the holder and personal context, without feedback."""
import copy
from unittest.mock import AsyncMock, patch

import pytest

from stock_analysis.tools import position
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm


@pytest.mark.asyncio
async def test_private_analysis_retains_fetched_context_without_public_leak():
    case = hm.SOUND
    with dm.stubbed(case):
        public = await dm.orchestrator.analyze_stock("ACME", _now=case.now)
        response, context = await dm.orchestrator._analyze_with_context("ACME", _now=case.now)
    public.pop("meta")
    response.pop("meta")
    public["data_quality"].pop("tool_timings", None)
    response["data_quality"].pop("tool_timings", None)
    assert public == response
    assert context["technicals"]["macd"] == case.payloads()["technicals"]["macd"]
    assert context["quote_type"] == "EQUITY"
    assert "technicals" not in response
    assert "quote_type" not in response["summary"]


@pytest.mark.asyncio
@pytest.mark.parametrize("horizon", ["mid_term", "long_term"])
async def test_position_uses_one_analysis_and_verbatim_horizon_without_personal_feedback(horizon):
    case = hm.SOUND
    with dm.stubbed(case):
        analysis, context = await dm.orchestrator._analyze_with_context("ACME", _now=case.now)
    original = copy.deepcopy(analysis)
    shared = AsyncMock(return_value=(analysis, context))
    with patch.object(position, "_analyze_with_context", shared):
        result = await position.analyze_position(
            "ACME", 80, "2026-09-01", 20, horizon=horizon,
            portfolio_value=10000, max_holding_pct=10, _now=case.now,
        )
    shared.assert_awaited_once_with("ACME", _now=case.now)
    card = analysis["decision_card"]
    chosen = card if horizon == "mid_term" else card["long_term"]
    assert result["position_card"]["holder_assessment"] == chosen["if_held"]
    assert result["position_card"]["new_money"] == chosen["new_money"]
    assert result["position_card"]["review_status"] == "review_allocation"
    assert result["position"]["current_price"] == analysis["summary"]["current_price"]
    assert analysis == original
    assert "sizing" not in result["position_card"]
    assert not {"sell_signals", "current_signals", "support_levels"} & result.keys()
    assert set(result["tax"]) == {"status", "reason"}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["hold", "reduce", "review_exit", "insufficient_data", "not_assessed"])
@pytest.mark.parametrize("allocation,extra,review", [
    ("not_requested", {}, "not_requested"),
    ("not_evaluated", {"max_holding_pct": 10}, "not_evaluated"),
    ("within_limit", {"shares": 10, "portfolio_value": 10000, "max_holding_pct": 10}, "no_allocation_trigger"),
    ("above_limit", {"shares": 20, "portfolio_value": 10000, "max_holding_pct": 10}, "review_allocation"),
])
async def test_allocation_is_independent_of_every_holder_action(action, allocation, extra, review):
    """P07: the holder vocabulary cannot become a second allocation decision layer."""
    case = hm.SOUND
    with dm.stubbed(case):
        analysis, context = await dm.orchestrator._analyze_with_context("ACME", _now=case.now)
    analysis["decision_card"]["if_held"]["action"] = action
    with patch.object(position, "_analyze_with_context", AsyncMock(return_value=(analysis, context))):
        result = await position.analyze_position("ACME", _now=case.now, **extra)
    card = result["position_card"]
    assert card["holder_assessment"]["action"] == action
    assert card["concentration_check"]["status"] == allocation
    assert card["review_status"] == review


@pytest.mark.asyncio
@pytest.mark.parametrize("add_allowed", [False, True])
@pytest.mark.parametrize("extra,phrase", [
    ({"shares": 20, "portfolio_value": 10000, "max_holding_pct": 10}, "ceiling"),
    ({"max_holding_pct": 10}, "have not been checked"),
])
async def test_addition_context_keeps_the_allocation_concern(add_allowed, extra, phrase):
    case = hm.SOUND
    with dm.stubbed(case):
        analysis, context = await dm.orchestrator._analyze_with_context("ACME", _now=case.now)
    analysis["decision_card"]["if_held"]["add_allowed"] = add_allowed
    with patch.object(position, "_analyze_with_context", AsyncMock(return_value=(analysis, context))):
        result = await position.analyze_position("ACME", _now=case.now, **extra)
    assert result["position_card"]["holder_assessment"]["add_allowed"] is add_allowed
    assert phrase in result["position_card"]["addition_context"]


@pytest.mark.asyncio
async def test_raw_macd_is_not_reconstructed_from_bullish_false():
    case = hm.SOUND
    with dm.stubbed(case):
        analysis, context = await dm.orchestrator._analyze_with_context("ACME", _now=case.now)
    context["technicals"]["macd"] = {"rules": {"bullish_cross": {"triggered": False}}}
    with patch.object(position, "_analyze_with_context", AsyncMock(return_value=(analysis, context))):
        result = await position.analyze_position("ACME", _now=case.now)
    tech = result["diagnostics"]["technicals"]
    assert tech["flags"]["macd_bearish"] is None
    assert "macd_bearish" not in tech["current_signals"]["bearish"]
    assert "active_count" not in tech["flags"]
    assert "not holder stops" in tech["note"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", dm.TOOLS)
async def test_source_failure_keeps_structured_business_and_personal_answers(failed):
    case = hm.SOUND
    with dm.stubbed(case), patch.object(dm.orchestrator, failed, AsyncMock(side_effect=TimeoutError("offline failure"))):
        result = await position.analyze_position("ACME", shares=20, portfolio_value=10000,
                                                max_holding_pct=10, horizon="long_term", _now=case.now)
    assert result["position_card"]["holder_assessment"]["action"] in {"hold", "review_exit", "insufficient_data", "not_assessed"}
    if failed == "fundamentals_snapshot":
        assert result["position_card"]["holder_assessment"]["action"] == "insufficient_data"
        assert result["position_card"]["review_status"] == "review_allocation"
    if failed in {"stock_summary", "technicals"}:
        assert result["position_card"]["review_status"] == "not_evaluated"


@pytest.mark.asyncio
async def test_selected_horizon_fit_reasons_are_retained_beside_hold():
    case = hm.STOCK_CASES["hold_with_avoid_fit"]
    with dm.stubbed(case):
        analysis = await dm.orchestrator.analyze_stock("ACME", _now=case.now)
        result = await position.analyze_position("ACME", horizon="long_term", _now=case.now)
    assert result["position_card"]["holder_assessment"]["action"] == "hold"
    assert result["suitability"]["fit"] == "avoid"
    assert result["suitability"]["reasons"] == [
        reason for reason in analysis["verdict"]["horizon_fit"]["reasons"] if reason.startswith("long_term:")
    ]


@pytest.mark.asyncio
async def test_foreign_technical_reference_levels_keep_their_quote_currency():
    case = hm.STOCK_CASES["eur_stock"]
    with dm.stubbed(case):
        result = await position.analyze_position("ACME", cost_basis=80, shares=20, _now=case.now)
    assert result["position"]["current_value"] is None
    assert result["position"]["current_price"] is None
    assert result["diagnostics"]["technicals"]["quote_currency"] == "EUR"
    assert result["diagnostics"]["technicals"]["reference_levels"]["sma_50"] == case.technicals["moving_averages"]["sma_50"]
