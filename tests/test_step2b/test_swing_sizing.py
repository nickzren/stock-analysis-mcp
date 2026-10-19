"""Whole-share swing sizing and origin disclosure through public entrypoints."""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp import Client

from stock_analysis import server
from stock_analysis.tools import watchlist
from stock_analysis.tools.trade_setup import orchestrator as swing
from stock_analysis.tools.trade_setup.plan import build_plan
from tests.test_tools.test_trade_plan import NOW, make_setup


def plan(**overrides):
    values = {"account_size": 10_000.0, "risk_per_trade_pct": 1.0, "max_position_pct": 8.25}
    values.update(overrides)
    return build_plan(
        make_setup(trigger_price=100.0, stop_price=97.0, trigger_satisfied=True),
        action="trade_now", session="regular", now=NOW, actionable_price=100.0, **values,
    )


def test_swing_amounts_use_whole_shares_and_loss_alias():
    result = plan()
    assert result["shares"] == 8
    assert result["fractional_shares"] is None
    assert result["position_dollars"] == 800
    assert result["planned_loss_at_stop"] == result["max_loss_dollars"] == 24
    assert result["position_pct"] == 8
    size = result["sizing"]
    assert size["horizon"] == "swing"
    assert size["full"]["allocation_cap_pct"] == 8.25
    assert size["full"]["position_pct"] == result["position_pct"]
    assert result["sizing_status"]["status"] == "available"
    assert "Gaps and slippage can make the actual loss larger." in size["loss_disclosure"]


def test_zero_whole_shares_has_no_affordable_order_prose():
    result = plan(account_size=1000, max_position_pct=8)
    assert result["shares"] == result["position_dollars"] == result["planned_loss_at_stop"] == 0
    assert result["sizing"]["full"]["quantity_status"] == "below_one_whole_share"
    assert result["entry"]["condition_text"] == "The setup qualifies, but no whole-share position fits this budget."
    assert result["entry"]["type"] == "market"


def test_no_account_keeps_percentage_only_and_no_concrete_amounts():
    result = plan(account_size=None)
    assert result["sizing_status"]["status"] == "percentage_only"
    assert result["position_pct"] == result["sizing"]["full"]["allocation_cap_pct"] == 8.25
    assert all(result[field] is None for field in (
        "shares", "fractional_shares", "position_dollars", "planned_loss_at_stop", "max_loss_dollars",
    ))


def test_swing_uses_displayed_stop_and_preserves_existing_geometry():
    result = build_plan(
        make_setup(trigger_price=100, stop_price=97.004, trigger_satisfied=True),
        action="trade_now", session="regular", now=NOW, actionable_price=105.001,
        account_size=10_000, risk_per_trade_pct=1, max_position_pct=10,
    )
    size = result["sizing"]
    assert size["assumptions"]["entry_price"] == 105.001
    assert size["assumptions"]["stop_price"] == result["stop"]["price"] == 97
    shares = result["shares"]
    assert Decimal(str(result["position_dollars"])) == shares * Decimal("105.001")
    assert Decimal(str(result["planned_loss_at_stop"])) == shares * Decimal("8.001")
    assert result["stop"]["distance_pct"] == round((105.001 - 97.004) / 105.001, 4)
    assert result["targets"][-1]["price"] == round(105.001 + 2 * (105.001 - 97.004), 2)


@pytest.fixture
def capture_card(monkeypatch):
    for name in ("stock_summary", "technicals", "risk_metrics", "events_calendar"):
        monkeypatch.setattr(swing, name, AsyncMock(return_value={}))
    monkeypatch.setattr(swing, "fetch_history", AsyncMock(return_value=None))
    card = Mock(return_value={})
    monkeypatch.setattr(swing, "build_trade_setup_card", card)
    return card


@pytest.mark.parametrize("values,origins", [
    ({}, {"risk_per_trade_pct": "default", "max_position_pct": "default"}),
    ({"risk_per_trade_pct": 1.0}, {"risk_per_trade_pct": "caller", "max_position_pct": "default"}),
    ({"max_position_pct": "10"}, {"risk_per_trade_pct": "default", "max_position_pct": "caller"}),
    ({"risk_per_trade_pct": "1", "max_position_pct": 10.0},
     {"risk_per_trade_pct": "caller", "max_position_pct": "caller"}),
])
@pytest.mark.parametrize("transport", ["direct", "mcp"])
async def test_swing_omitted_and_explicit_equal_defaults_keep_distinct_origins(capture_card, values, origins, transport):
    if transport == "mcp":
        async with Client(server.mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            properties = tools["analyze_trade_setup"].inputSchema["properties"]
            assert properties["risk_per_trade_pct"]["default"] == 1
            assert properties["max_position_pct"]["default"] == 10
            result = await client.call_tool("analyze_trade_setup", {"symbol": "ACME", **values})
            assert not result.is_error
    else:
        await swing.analyze_trade_setup("ACME", **values)
    assert capture_card.call_args.kwargs["limit_origins"] == origins


@pytest.mark.parametrize("values,origins", [
    ({}, {"risk_per_trade_pct": "default", "max_position_pct": "default"}),
    ({"risk_per_trade_pct": "1", "max_position_pct": 10},
     {"risk_per_trade_pct": "caller", "max_position_pct": "caller"}),
])
@pytest.mark.parametrize("transport", ["direct", "mcp"])
async def test_scan_preserves_origins_when_promoting(capture_card, monkeypatch, tmp_path, values, origins, transport):
    monkeypatch.setattr(watchlist, "resolve_data_dir", lambda: tmp_path)
    monkeypatch.setattr(watchlist, "load_watchlist", Mock(return_value=({"ACME": {}}, [])))
    monkeypatch.setattr(watchlist, "load_scan_state", Mock(return_value=({}, [])))
    monkeypatch.setattr(watchlist, "save_scan_state", Mock())
    monkeypatch.setattr(watchlist, "_screen_one", AsyncMock(return_value={"promote": True}))
    capture_card.return_value = {"action": "no_setup"}
    if transport == "mcp":
        async with Client(server.mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            properties = tools["scan_watchlist"].inputSchema["properties"]
            assert properties["risk_per_trade_pct"]["default"] == 1
            assert properties["max_position_pct"]["default"] == 10
            result = await client.call_tool("scan_watchlist", values)
            assert not result.is_error
    else:
        await watchlist.scan_watchlist(data_dir=tmp_path, **values)
    assert capture_card.call_args.kwargs["limit_origins"] == origins


def test_swing_has_no_invented_regime_or_starter_limits():
    result = plan(account_size=1000, risk_per_trade_pct=100, max_position_pct=100)
    size = result["sizing"]
    assert size["starter"] is None
    assert all(size["limits"][key] is None for key in (
        "regime_max_pct", "regime_starter_pct", "preset_starter_pct",
    ))
    assert size["full"]["binding_constraints"] == ["caller_max_position"]
    assert result["shares"] == 10
