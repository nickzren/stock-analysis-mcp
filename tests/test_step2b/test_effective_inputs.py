"""Effective sizing inputs retain precision and validate before analysis I/O."""

from unittest.mock import AsyncMock, patch

import pytest

from stock_analysis.tools.analyze import orchestrator
from stock_analysis.tools.analyze.investor_profile import resolve_investor_profile
from tests.support import decision_matrix as dm


@pytest.mark.parametrize(("name", "risk", "cap"), [
    ("core", 0.5, 8.0), ("balanced", 1.0, 10.0), ("speculative", 1.5, 6.0),
])
def test_profile_defaults_and_their_origins_are_preserved(name, risk, cap):
    result = resolve_investor_profile(name)
    assert result["risk_per_trade_pct"] == risk
    assert result["max_position_pct"] == cap
    assert result["limit_origins"] == {"risk_per_trade_pct": "profile", "max_position_pct": "profile"}


@pytest.mark.parametrize(("overrides", "risk", "cap", "origins"), [
    ({"risk_per_trade_pct": "0.001"}, 0.001, 10.0, ("caller", "profile")),
    ({"max_position_pct": "0.06"}, 1.0, 0.06, ("profile", "caller")),
    ({"risk_per_trade_pct": "0.001", "max_position_pct": "0.06"}, 0.001, 0.06, ("caller", "caller")),
    ({"risk_per_trade_pct": None, "max_position_pct": None}, 1.0, 10.0, ("profile", "profile")),
])
def test_explicit_limits_do_not_round_or_replace_each_other(overrides, risk, cap, origins):
    result = resolve_investor_profile(account_size="10000.12345", **overrides)
    assert result["account_size"] == 10000.12345
    assert result["risk_per_trade_pct"] == risk
    assert result["risk_per_trade_decimal"] == risk / 100
    assert result["max_position_pct"] == cap
    assert tuple(result["limit_origins"].values()) == origins


@pytest.mark.parametrize("invalid", [
    {"account_size": True}, {"risk_per_trade_pct": float("nan")},
    {"max_position_pct": 101}, {"risk_per_trade_pct": "1_000"}, {"profile": "invalid"},
])
async def test_bad_direct_analyze_inputs_never_start_search_or_provider_reads(monkeypatch, invalid):
    search = AsyncMock(side_effect=AssertionError("lookup started before validation"))
    provider = AsyncMock()
    monkeypatch.setattr(orchestrator, "symbol_search", search)
    monkeypatch.setattr(orchestrator, "stock_summary", provider)
    with pytest.raises(ValueError, match=next(iter(invalid))):
        await orchestrator.analyze_stock("Acme Company", **invalid)
    search.assert_not_awaited()
    provider.assert_not_awaited()


async def test_tiny_budget_is_forwarded_to_risk_metrics_without_default_substitution():
    case = dm.SCENARIOS["S2_buy_clean"]
    risk = AsyncMock(return_value=case.risk)
    with dm.stubbed(case), patch.object(orchestrator, "risk_metrics", risk):
        await orchestrator.analyze_stock("ACME", risk_per_trade_pct=0.001, _now=case.now)
    assert risk.await_args.kwargs["risk_per_trade"] == 0.00001
