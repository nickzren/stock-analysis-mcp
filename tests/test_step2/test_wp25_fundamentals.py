"""Unknown thesis evidence never masquerades as recovery; raw runway drives the cap."""

import copy
import importlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from stock_analysis.tools.analyze.decision import if_held
from stock_analysis.tools.analyze.dislocation_framework import classify_dislocation_opportunity
from tests.support import decision_matrix as dm

fundamentals = importlib.import_module("stock_analysis.tools.fundamentals")
diff_analysis = importlib.import_module("stock_analysis.tools.diff_analysis")

INFO = {
    "quoteType": "EQUITY", "currency": "USD", "financialCurrency": "USD", "currentPrice": 100.,
    "marketCap": 1e10, "trailingEps": -1., "priceToSalesTrailing12Months": 4., "totalRevenue": 1e9,
    "revenueGrowth": .3, "profitMargins": -.2, "freeCashflow": -2e8, "operatingCashflow": -1.6e8,
    "totalCash": 1e8, "debtToEquity": 10., "totalDebt": 0.,
}


async def produced(monkeypatch: pytest.MonkeyPatch, info: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setattr(fundamentals, "fetch_info", AsyncMock(return_value=info))
    monkeypatch.setattr(fundamentals, "fetch_ticker", AsyncMock(side_effect=TimeoutError("fixture statements absent")))
    result = await fundamentals.fundamentals_snapshot("ACME")
    assert not result.get("error")
    return result


async def analyze_fund(payload: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
    return await dm.run(dm.Scenario("WP25 thesis evidence", dm.STRONG, payload, dm.CALM), **kwargs)


async def changes(monkeypatch: pytest.MonkeyPatch, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setattr(diff_analysis, "analyze_stock", AsyncMock(return_value=new))
    return await diff_analysis.what_changed("ACME", previous_snapshot=old["watchlist_snapshot"])


def assert_unknown(result: dict[str, Any]) -> None:
    card = result["decision_card"]
    assert card["if_held"]["action"] == "insufficient_data"
    assert card["if_held"]["basis"] == "thesis_unknown"
    for money in (card["new_money"], card["long_term"]["new_money"]):
        assert "thesis_broken" in {gate["id"] for gate in money["unevaluated"]}
    assert not card["hard_gates"]["checks"]["thesis_broken"]


@pytest.mark.parametrize("missing", ["totalCash", "cashflows", "business"])
async def test_partial_real_fundamentals_are_unknown_not_recovered(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    original = await analyze_fund(await produced(monkeypatch, INFO))
    incomplete = dict(INFO)
    for key in {"totalCash": ["totalCash"], "cashflows": ["freeCashflow", "operatingCashflow"],
                "business": ["profitMargins", "trailingEps", "freeCashflow", "operatingCashflow"]}[missing]:
        incomplete.pop(key)
    unknown = await analyze_fund(await produced(monkeypatch, incomplete))
    assert_unknown(unknown)
    lost = await changes(monkeypatch, original, unknown)
    assert "thesis_broken" in lost["changes"]["hard_gates"]["not_evaluated"]
    assert "thesis_broken" not in lost["changes"]["hard_gates"]["cleared"]
    recovered = await changes(monkeypatch, unknown, original)
    assert "thesis_broken" in recovered["changes"]["hard_gates"]["new"]
    assert original["decision_card"]["if_held"]["action"] == "review_exit"


@pytest.mark.parametrize("source", ["summary_failed", "type_absent"])
async def test_unverified_instrument_is_unknown_not_inapplicable(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    original = await analyze_fund(dm.BURNER)
    summary = {key: value for key, value in dm.EQUITY.items() if key != "quote_type"}
    if source == "summary_failed":
        scenario = dm.Scenario("summary failed", dm.STRONG, dm.BURNER, dm.CALM)
        with dm.stubbed(scenario):
            monkeypatch.setattr(dm.orchestrator, "stock_summary", AsyncMock(side_effect=TimeoutError("summary failed")))
            unknown = await dm.orchestrator.analyze_stock("ACME", _now=scenario.now)
    else:
        unknown = await analyze_fund(dm.BURNER, summary=summary)
    assert_unknown(unknown)
    assert unknown["data_quality"]["fundamentals_status"] == "unverified"
    assert unknown["fundamentals_summary"]["status"] == "unverified"
    assert unknown["dislocation_framework"]["status"] == "unverified"
    assert unknown["decision_context"]["thesis_checkpoints"]["hold_thesis"] == "unknown"
    change = await changes(monkeypatch, original, unknown)
    assert "thesis_broken" not in change["changes"]["hard_gates"]["cleared"]


@pytest.mark.parametrize("summary", [{**dm.EQUITY, "currency": None}, dm.GBP_EQUITY])
async def test_currency_does_not_erase_established_company_risk(summary: dict[str, Any]) -> None:
    result = await analyze_fund(dm.BURNER, summary=summary)
    assert result["decision_card"]["if_held"] == {"action": "review_exit", "basis": "runway_critical", "add_allowed": False}
    assert result["decision_card"]["hard_gates"]["checks"]["thesis_broken"]


async def test_known_unsupported_instrument_keeps_existing_holder_behavior() -> None:
    result = await analyze_fund(dm.BURNER, summary=dm.ETF)
    assert result["data_quality"]["fundamentals_status"] == "not_applicable"
    assert result["decision_card"]["if_held"]["action"] == "hold"
    assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]


@pytest.mark.parametrize("quality", ["strong", "unprofitable"])
async def test_known_nonburner_requires_no_numeric_runway(monkeypatch: pytest.MonkeyPatch, quality: str) -> None:
    info = {**INFO, "freeCashflow": 2e8, "operatingCashflow": 1.6e8}
    info.pop("totalCash")
    if quality == "strong":
        info.update(profitMargins=.2, trailingEps=1., trailingPE=10.)
    result = await analyze_fund(await produced(monkeypatch, info))
    assert result["decision_card"]["if_held"]["action"] == "hold"
    assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]


@pytest.mark.parametrize("kind", ["fcf", "ocf", "both"])
@pytest.mark.parametrize("runway", [3.94, 3.96, 3.99, 4., 4.01])
async def test_exact_runway_boundary_preserves_rounded_presentation(kind: str, runway: float) -> None:
    payload = copy.deepcopy(dm.BURNER)
    payload["financial_health"]["cash_and_st_investments"] = runway * 5e7
    payload["cash_flow"]["free_cash_flow_ttm"] = -2e8 if kind != "ocf" else None
    payload["cash_flow"]["operating_cf_ttm"] = -2e8 if kind != "fcf" else None
    result = await analyze_fund(payload)
    assert result["fundamentals_summary"]["burn_metrics"]["cash_runway_quarters"] == round(runway, 1)
    assert result["decision_card"]["hard_gates"]["checks"]["thesis_broken"] is (runway < 4)
    assert result["decision_card"]["if_held"]["action"] == ("review_exit" if runway < 4 else "hold")
    assert result["dislocation_framework"]["balance_sheet_safety"]["checks"]["runway"] == ("fail" if runway < 4 else "watch")


def test_known_failure_precedes_unknown_holder_evidence() -> None:
    for basis in ("business_broken", "runway_critical"):
        held = if_held(thesis=(basis,), candidate_action="buy", dislocation=False, tilt="bullish", thesis_evaluable=False)
        assert held == {"action": "review_exit", "basis": basis}


def test_classifier_keeps_partial_components_distinct() -> None:
    signals = classify_dislocation_opportunity({"decomposed": {"business_quality_status": "data_missing"}}, {},
        {"burn_metrics": {"status": "available", "cash_runway_quarters": 2}}, None)
    assert signals["business_evaluable"] is False
    assert signals["runway_evaluable"] is True
    assert signals["balance_status"] == "unsafe"


@pytest.mark.parametrize("payload", [None, {"growth": {"revenue_yoy": .3}}])
async def test_missing_business_does_not_invent_speculative_exit_thesis(payload: Any) -> None:
    result = await analyze_fund(payload)
    checkpoints = result["decision_context"]["thesis_checkpoints"]
    assert checkpoints["hold_thesis"] == "unknown"
    assert checkpoints["thesis_stop_triggers"] is None
    framework = result["dislocation_framework"]
    assert framework["thesis_integrity"]["status"] == "unknown"
    assert framework["action"]["sell_without_attachment_if"] == []
    assert "still alive" not in framework["thesis_integrity"]["summary"]


async def test_zero_cash_is_known_critical_not_missing() -> None:
    payload = copy.deepcopy(dm.BURNER)
    payload["financial_health"]["cash_and_st_investments"] = 0.
    result = await analyze_fund(payload)
    assert result["fundamentals_summary"]["burn_metrics"]["cash_runway_quarters"] == 0.
    assert result["decision_card"]["if_held"]["basis"] == "runway_critical"
    assert "thesis_broken" in dm.gate_ids(result)
    assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]


async def test_public_runway_does_not_leak_unrounded_ratio_into_sizing() -> None:
    payload = copy.deepcopy(dm.BURNER)
    payload["financial_health"]["cash_and_st_investments"] = 3.96 * 5e7
    payload["cash_flow"]["currency"] = "USD"
    result = await analyze_fund(payload, summary={**dm.EQUITY, "financial_currency": "USD"})
    burn = result["fundamentals_summary"]["burn_metrics"]
    assert burn["cash_runway_quarters"] == 4.
    assert burn["dilution_analysis"]["raise_needed_for_2y_runway"] == 2e8
    assert result["decision_card"]["new_money"]["action"] == "avoid"
    assert result["decision_card"]["sizing"] is None


@pytest.mark.parametrize("missing", ["debtToEquity", "freeCashflow"])
async def test_partial_business_rubric_cannot_clear_known_broken_thesis(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    info = {**INFO, "profitMargins": .02, "trailingEps": .5, "trailingPE": 40., "debtToEquity": 300.,
            "freeCashflow": 0., "operatingCashflow": 1e8}
    original = await analyze_fund(await produced(monkeypatch, info))
    assert original["decision_card"]["if_held"]["basis"] == "business_broken"
    partial = dict(info)
    partial.pop(missing)
    unknown = await analyze_fund(await produced(monkeypatch, partial))
    assert_unknown(unknown)
    assert unknown["decision_context"]["thesis_checkpoints"]["hold_thesis"] == "unknown"
    assert unknown["decision_context"]["thesis_checkpoints"]["thesis_stop_triggers"] is None
    assert unknown["dislocation_framework"]["business_integrity"]["status"] == "unknown"
    # The approved thesis-data cap now blocks the previously admitted starter.
    assert unknown["decision_card"]["new_money"]["action"] == "wait_for_data"
    assert "thesis_unverifiable" in dm.gate_ids(unknown)
    lost = await changes(monkeypatch, original, unknown)
    assert "thesis_broken" in lost["changes"]["hard_gates"]["not_evaluated"]
    assert "thesis_broken" not in lost["changes"]["hard_gates"]["cleared"]
    recovered = await changes(monkeypatch, unknown, original)
    assert "thesis_broken" in recovered["changes"]["hard_gates"]["new"]


@pytest.mark.parametrize("verified", [{"freeCashflow": 1e8}, {"debtToEquity": 100.}])
async def test_partial_business_inputs_can_still_prove_no_broken_business(monkeypatch: pytest.MonkeyPatch, verified: dict[str, Any]) -> None:
    info = {**INFO, "profitMargins": .02, "trailingEps": .5, "trailingPE": 40.}
    info.pop("freeCashflow")
    info.pop("debtToEquity")
    info.update(verified)
    result = await analyze_fund(await produced(monkeypatch, info))
    assert result["decision_card"]["if_held"]["action"] == "hold"
    assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]
