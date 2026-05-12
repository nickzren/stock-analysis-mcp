"""Tests for token-efficient analyze detail projections."""

import json

import pytest

from stock_analysis.server import analyze
from stock_analysis.tools.analyze.projection import (
    project_analyze_result,
)


def _full_result() -> dict:
    """Representative full analyze payload for projection tests."""
    return {
        "meta": {"tool": "analyze_stock", "duration_ms": 1.0},
        "symbol": "TEST",
        "summary": {
            "name": "Test Corp",
            "current_price": 100.0,
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 1_000_000_000,
            "currency": "USD",
            "extra_summary_field": "full-only",
        },
        "decision_card": {"action_now": "buy", "hard_gates": {"blocking": []}},
        "verdict": {
            "score": 0.42,
            "tilt": "bullish",
            "confidence": "moderate",
            "components": {"technicals": 0.5},
            "pros": ["positive fcf"],
            "cons": ["high valuation"],
            "weights_used": {"technicals": 1.0},
            "inputs_used": {"rsi": 55},
            "confidence_path": {"current_blockers": ["coverage"]},
        },
        "data_quality": {
            "completeness": 0.8,
            "missing_critical": [],
            "warnings": ["earnings_within_7_days"],
            "tool_timings": {"stock_summary": 1.2},
            "component_freshness": {"news": {"status": "fresh"}},
        },
        "watchlist_snapshot": {"snapshot_hash": "abc", "decision_action_now": "buy"},
        "executive_summary": "Test summary.",
        "section_summaries": {"risk": "Moderate risk."},
        "action_zones": {
            "current_zone": "accumulate",
            "levels": {"stop_loss": 90.0},
            "stop_calculation": {"stop_distance_pct": 0.1},
            "valuation_assessment": {"gate": "neutral"},
            "position_sizing_range": {"starter_pct": 3.0},
            "distance_labels": {"stop_loss": "10% below"},
        },
        "events_summary": {
            "next_catalyst": {"type": "earnings"},
            "days_to_earnings": 30,
            "earnings": {"extra": "full-only"},
        },
        "news_summary": {
            "article_count": 2,
            "sentiment": {"overall": "positive"},
            "headlines": ["Headline"],
            "catalyst_intelligence": {"bullish": []},
            "summary": "News summary.",
            "articles": [{"title": "full-only"}],
        },
        "risk_summary": {
            "risk_regime": {"classification": "moderate"},
            "annualized_volatility": 0.3,
            "beta": 1.1,
            "max_drawdown_1y": -0.2,
            "atr_pct": 0.03,
            "summary": "Risk summary.",
            "liquidity": {"avg_dollar_volume": 100_000_000},
        },
        "fundamentals_summary": {
            "valuation": {"pe_trailing": 20},
            "growth": {"revenue_yoy": 0.2},
            "profitability": {"net_margin": 0.1},
            "health": {"debt_to_equity": 0.3},
            "cash_flow": {"free_cash_flow_label": "$100M TTM"},
            "burn_metrics": {"status": "not_applicable"},
            "analyst_estimates": {"eps_next_year": 5.0},
            "summary": "Fundamentals summary.",
            "valuation_history": [{"period": "full-only"}],
        },
        "analyst_coverage": {"rating": "buy"},
        "short_interest": {"days_to_cover": 2.0},
        "ownership": {"institutional_pct": 0.7},
        "sector_comparison": {"pe_percentile": 45},
        "relative_performance": {"vs_spy_1m": 0.02},
        "company_profile": {"description": "Long profile"},
        "policy_action": {"mid_term": "buy"},
        "decision_modes": {"balanced": {"action": "starter"}},
        "dislocation_framework": {"action": {"buy_only_if": []}},
        "dip_assessment": {"dip_classification": {"type": "healthy_pullback"}},
        "decision_context": {"top_triggers": [{"id": "risk"}]},
        "ownership_flow": {"transactions": [{"shares": 1000}]},
        "options_signals": {"put_call_ratio": {"volume_based": 0.8}},
        "signals": {"bullish": ["macd_bullish"]},
        "governance": {"risk": "low"},
        "quality": {"score": "good"},
        "valuation_context": {"note": "full-only"},
        "market_context": {"spy_trend": "up"},
    }


def _size(payload: dict) -> int:
    return len(json.dumps(payload, separators=(",", ":"), default=str).encode())


def test_full_projection_preserves_complete_payload() -> None:
    result = _full_result()

    assert project_analyze_result(result, "full") is result


def test_decision_projection_uses_slim_allowlists() -> None:
    projected = project_analyze_result(_full_result(), "decision")

    assert list(projected) == [
        "meta",
        "symbol",
        "summary",
        "decision_card",
        "verdict",
        "data_quality",
        "watchlist_snapshot",
    ]
    assert "extra_summary_field" not in projected["summary"]
    assert "weights_used" not in projected["verdict"]
    assert "tool_timings" not in projected["data_quality"]


def test_decision_projection_preserves_tool_failures_for_diagnostics() -> None:
    result = _full_result()
    result["error"] = True
    result["data_quality"]["tool_failures"] = [
        {"tool": "stock_summary", "error_type": "invalid_symbol", "message": "No data"}
    ]

    decision = project_analyze_result(result, "decision")
    standard = project_analyze_result(result, "standard")

    assert decision["data_quality"]["tool_failures"] == result["data_quality"]["tool_failures"]
    assert standard["data_quality"]["tool_failures"] == result["data_quality"]["tool_failures"]


def test_standard_projection_includes_investor_blocks_and_omits_full_only_blocks() -> None:
    projected = project_analyze_result(_full_result(), "standard")

    assert projected["analyst_coverage"] == {"rating": "buy"}
    assert projected["short_interest"] == {"days_to_cover": 2.0}
    assert projected["ownership"] == {"institutional_pct": 0.7}
    assert projected["sector_comparison"] == {"pe_percentile": 45}
    assert projected["relative_performance"] == {"vs_spy_1m": 0.02}
    assert "dip_assessment" not in projected
    assert "decision_context" not in projected
    assert "ownership_flow" not in projected
    assert "company_profile" not in projected
    assert "signals" not in projected


def test_standard_nested_blocks_use_schema_accurate_allowlists() -> None:
    projected = project_analyze_result(_full_result(), "standard")

    assert set(projected["events_summary"]) == {"next_catalyst", "days_to_earnings"}
    assert set(projected["risk_summary"]) == {
        "risk_regime",
        "annualized_volatility",
        "beta",
        "max_drawdown_1y",
        "atr_pct",
        "summary",
    }
    assert set(projected["fundamentals_summary"]) == {
        "valuation",
        "growth",
        "profitability",
        "health",
        "cash_flow",
        "burn_metrics",
        "analyst_estimates",
        "summary",
    }
    assert "stop_calculation" in projected["action_zones"]


def test_projection_size_ordering() -> None:
    result = _full_result()

    decision = project_analyze_result(result, "decision")
    standard = project_analyze_result(result, "standard")
    full = project_analyze_result(result, "full")

    assert _size(decision) < _size(standard) < _size(full)


@pytest.mark.asyncio
async def test_server_analyze_defaults_to_standard_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_analyze_stock(*args, **kwargs) -> dict:
        return _full_result()

    monkeypatch.setattr("stock_analysis.server.analyze_stock", fake_analyze_stock)

    parsed = json.loads(await analyze("TEST"))

    assert "executive_summary" in parsed
    assert "decision_context" not in parsed
    assert "decision_card" in parsed


@pytest.mark.asyncio
async def test_server_analyze_full_detail_preserves_full_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_analyze_stock(*args, **kwargs) -> dict:
        return _full_result()

    monkeypatch.setattr("stock_analysis.server.analyze_stock", fake_analyze_stock)

    parsed = json.loads(await analyze("TEST", detail="full"))

    assert "decision_context" in parsed
    assert "dip_assessment" in parsed
    assert parsed["company_profile"]["description"] == "Long profile"


@pytest.mark.asyncio
async def test_server_analyze_snapshot_hash_is_stable_across_detail_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_analyze_stock(*args, **kwargs) -> dict:
        return _full_result()

    monkeypatch.setattr("stock_analysis.server.analyze_stock", fake_analyze_stock)

    standard = json.loads(await analyze("TEST"))
    full = json.loads(await analyze("TEST", detail="full"))

    assert standard["watchlist_snapshot"]["snapshot_hash"] == full["watchlist_snapshot"]["snapshot_hash"]


@pytest.mark.asyncio
async def test_server_analyze_invalid_detail_returns_error_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_if_called(*args, **kwargs) -> dict:
        raise AssertionError("analyze_stock should not run for invalid detail")

    monkeypatch.setattr("stock_analysis.server.analyze_stock", fail_if_called)

    parsed = json.loads(await analyze("TEST", detail="tiny"))

    assert parsed["error"] is True
    assert parsed["error_type"] == "invalid_parameters"
    assert "Invalid detail" in parsed["message"]
