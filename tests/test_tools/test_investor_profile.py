"""Tests for investor profile helpers and prompt alignment."""

from stock_analysis.prompts.templates import get_prompt
from stock_analysis.tools.analyze.investor_profile import resolve_investor_profile


def test_resolve_investor_profile_aliases_and_overrides() -> None:
    profile = resolve_investor_profile(
        profile="small_account",
        account_size=3_000,
        risk_per_trade_pct=1.0,
        max_position_pct=5.0,
    )

    assert profile["profile"] == "speculative"
    assert profile["account_size"] == 3000
    assert profile["small_account"] is True
    assert profile["risk_per_trade_pct"] == 1.0
    assert profile["max_position_pct"] == 5.0


def test_full_analysis_prompt_uses_analyze_tool_first() -> None:
    prompt = get_prompt("full_analysis", {"symbol": "NVDA"})

    assert prompt is not None
    content = prompt["messages"][0]["content"]
    assert 'analyze("NVDA")' in content
    assert "Do not manually reconstruct the report" in content
    assert "decision_modes.balanced" in content
    assert "get_stock_summary" not in content
