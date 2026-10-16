"""WP2b.0: current MCP coercion and application validation, defects included.

The handler-capture cases test the real registered tools and FastMCP client, but
replace analysis below the wrapper. Acceptance here does not mean the complete
application accepts a value. These are before pins, not the intended 2b contract.
"""

import json
import math
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

from stock_analysis import server
from stock_analysis.tools.analyze import orchestrator
from stock_analysis.tools.analyze.investor_profile import resolve_investor_profile
from stock_analysis.tools.trade_setup.orchestrator import validate_sizing_params

TOOLS = ("analyze", "analyze_trade_setup", "scan_watchlist")
FIELDS = ("account_size", "risk_per_trade_pct", "max_position_pct")


def arguments(tool, **values):
    return {**({} if tool == "scan_watchlist" else {"symbol": "ACME"}), **values}


@pytest.fixture
def capture_handler(monkeypatch):
    handler = AsyncMock(return_value={"captured": True})
    for name in ("analyze_stock", "trade_setup_analysis", "watchlist_scan"):
        monkeypatch.setattr(server, name, handler)
    monkeypatch.setattr(server, "project_analyze_result", lambda result, detail: result)
    return handler


async def test_published_numeric_schemas_have_no_bounds_and_do_not_advertise_strings():
    """Capture the three tools' numeric schema fragments through the real client."""
    nullable = {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None}
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    for name in TOOLS:
        schema = tools[name].inputSchema
        numeric = {field: schema["properties"][field] for field in FIELDS}
        expected = dict.fromkeys(FIELDS, nullable) if name == "analyze" else {
            "account_size": nullable,
            "risk_per_trade_pct": {"type": "number", "default": 1.0},
            "max_position_pct": {"type": "number", "default": 10.0},
        }
        assert numeric == expected, name
        assert not set(FIELDS).intersection(schema.get("required", []))


@pytest.mark.parametrize("tool", TOOLS)
async def test_framework_coercion_before_application_validation(tool, capture_handler):
    """Known defects: booleans/nonfinite/unbounded values reach the application."""
    cases = [
        (1.5, 1.5), ("1.5", 1.5), (True, 1.0), (False, 0.0),
        (" +1e-3 ", 0.001), ("1_000", 1000.0),
        ("nan", math.nan), ("inf", math.inf), ("1e309", math.inf),
        ("1e-400", 0.0), (0, 0.0), (101, 101.0),
    ]
    async with Client(server.mcp) as client:
        for raw, expected in cases:
            capture_handler.reset_mock()
            result = await client.call_tool(
                tool, arguments(tool, **dict.fromkeys(FIELDS, raw)), raise_on_error=False,
            )
            assert not result.is_error, (tool, raw, result.content)
            capture_handler.assert_awaited_once()
            for field in FIELDS:
                actual = capture_handler.await_args.kwargs[field]
                assert type(actual) is float, (tool, raw, field)
                if math.isnan(expected):
                    assert math.isnan(actual), (tool, raw, field)
                else:
                    assert actual == expected, (tool, raw, field)


@pytest.mark.parametrize("tool", TOOLS)
async def test_framework_rejects_unparseable_values_before_handler(tool, capture_handler):
    async with Client(server.mcp) as client:
        for raw in ("", "true", "false", "1%", "$1", "0x10", {}, []):
            capture_handler.reset_mock()
            result = await client.call_tool(
                tool, arguments(tool, risk_per_trade_pct=raw), raise_on_error=False,
            )
            assert result.is_error, (tool, raw)
            assert "risk_per_trade_pct" in result.content[0].text
            capture_handler.assert_not_awaited()


@pytest.mark.parametrize("tool", TOOLS)
async def test_omission_and_null_have_tool_specific_defaults(tool, capture_handler):
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, arguments(tool), raise_on_error=False)
        assert not result.is_error
        values = capture_handler.await_args.kwargs
        assert {field: values[field] for field in FIELDS} == (
            dict.fromkeys(FIELDS) if tool == "analyze" else {
                "account_size": None, "risk_per_trade_pct": 1.0, "max_position_pct": 10.0,
            }
        )
        for field in FIELDS:
            capture_handler.reset_mock()
            result = await client.call_tool(
                tool, arguments(tool, **{field: None}), raise_on_error=False,
            )
            nullable = tool == "analyze" or field == "account_size"
            assert result.is_error is not nullable
            if nullable:
                capture_handler.assert_awaited_once()
                assert capture_handler.await_args.kwargs[field] is None
            else:
                capture_handler.assert_not_awaited()
                assert field in result.content[0].text


@pytest.mark.parametrize("invalid", [{"risk_per_trade_pct": "0"}, {"profile": "invalid"}])
async def test_real_analyze_resolves_company_before_rejecting_invalid_inputs(monkeypatch, invalid):
    """Known ordering defect: lookup happens before application input validation."""
    search = AsyncMock(return_value={"exact_match": "ACME"})
    monkeypatch.setattr(orchestrator, "symbol_search", search)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "analyze", {"symbol": "Acme Company", **invalid}, raise_on_error=False,
        )
    assert not result.is_error  # application JSON error, not framework tool error
    payload = json.loads(result.content[0].text)
    assert payload["error_type"] == "invalid_parameters"
    assert next(iter(invalid)) in payload["message"]
    search.assert_awaited_once_with(query="Acme Company", limit=5)


def test_direct_validators_differ_from_framework_coercion():
    """Analyze admits nonfinite/over-100 limits; swing rejects them, both admit True."""
    for raw in (math.nan, math.inf, 101.0):
        profile = resolve_investor_profile(risk_per_trade_pct=raw)
        value = profile["risk_per_trade_pct"]
        assert math.isnan(value) if math.isnan(raw) else value == raw
        assert "risk_per_trade_pct" in validate_sizing_params(None, raw, 10.0)
    assert resolve_investor_profile(risk_per_trade_pct=True)["risk_per_trade_pct"] == 1
    assert validate_sizing_params(True, True, True) is None
    with pytest.raises(TypeError):
        resolve_investor_profile(risk_per_trade_pct="1.5")
    with pytest.raises(TypeError):
        validate_sizing_params(None, "1.5", 10.0)
