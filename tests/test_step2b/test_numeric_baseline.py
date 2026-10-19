"""Approved sizing input contract through the real MCP client and direct calls."""

import json
import math
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp import Client
from jsonschema import Draft202012Validator
from pydantic import TypeAdapter, ValidationError

from stock_analysis import server
from stock_analysis.tools import watchlist
from stock_analysis.tools.analyze import orchestrator
from stock_analysis.tools.analyze.investor_profile import resolve_investor_profile
from stock_analysis.tools.trade_setup import orchestrator as swing
from stock_analysis.tools.trade_setup.orchestrator import validate_sizing_params
from stock_analysis.utils.sizing_inputs import AccountSize, PositionPercent, RiskPercent

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


async def test_published_numeric_schemas_advertise_decimal_strings_and_valid_bounds():
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    for name in TOOLS:
        schema = tools[name].inputSchema
        Draft202012Validator.check_schema(schema)
        assert not set(FIELDS).intersection(schema.get("required", []))
        for field in FIELDS:
            entry = schema["properties"][field]
            variants = entry["anyOf"]
            numeric = next(item for item in variants if item.get("type") == "number")
            text = next(item for item in variants if item.get("type") == "string")
            assert numeric["exclusiveMinimum"] == 0
            assert "pattern" in text
            validator = Draft202012Validator(entry)
            for raw in (1.5, "1.5", " +1e-3 ", ".5", "1."):
                assert validator.is_valid(raw), (name, field, raw)
            for raw in (True, False, 0, -1, "true", "nan", "1_000", "$1", "1%", "0x10"):
                assert not validator.is_valid(raw), (name, field, raw)
            assert "gt" not in numeric and "le" not in numeric
            if field != "account_size":
                assert numeric["maximum"] == 100
            nullable = name == "analyze" or field == "account_size"
            assert any(item.get("type") == "null" for item in variants) is nullable


@pytest.mark.parametrize("tool", TOOLS)
async def test_framework_normalizes_valid_numbers_and_decimal_strings(tool, capture_handler):
    cases = [(1.5, 1.5), ("1.5", 1.5), (" +1e-3 ", 0.001), (".5", 0.5), ("1.", 1.0)]
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
                assert actual == expected, (tool, raw, field)


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("field", FIELDS)
async def test_framework_rejects_invalid_before_handler(tool, field, capture_handler):
    invalid = [True, False, "nan", "inf", "-inf", "1e309", "1e-400", "1_000", 0, -1]
    if field != "account_size":
        invalid.extend([100.001, "101"])
    async with Client(server.mcp) as client:
        for raw in invalid:
            capture_handler.reset_mock()
            result = await client.call_tool(
                tool, arguments(tool, **{field: raw}), raise_on_error=False,
            )
            assert result.is_error, (tool, field, raw)
            assert field in result.content[0].text
            capture_handler.assert_not_awaited()


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
async def test_real_analyze_rejects_invalid_inputs_before_company_search(monkeypatch, invalid):
    search = AsyncMock(side_effect=AssertionError("invalid input reached search"))
    monkeypatch.setattr(orchestrator, "symbol_search", search)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "analyze", {"symbol": "Acme Company", **invalid}, raise_on_error=False,
        )
    if "profile" in invalid:
        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["error_type"] == "invalid_parameters"
        assert "profile" in payload["message"]
    else:
        assert result.is_error
        assert "risk_per_trade_pct" in result.content[0].text
    search.assert_not_awaited()


def test_direct_validators_match_numeric_contract():
    for raw in (True, False, math.nan, math.inf, 101.0, "nan", "1e309", "1e-400", "1_000"):
        with pytest.raises(ValueError, match="risk_per_trade_pct"):
            resolve_investor_profile(risk_per_trade_pct=raw)
        assert "risk_per_trade_pct" in validate_sizing_params(None, raw, 10.0)
    assert resolve_investor_profile(risk_per_trade_pct="1.5")["risk_per_trade_pct"] == 1.5
    assert validate_sizing_params("10000", "1.5", "10") is None


@pytest.mark.parametrize("tool", TOOLS)
async def test_invalid_mcp_input_never_reaches_search_provider_or_watchlist(monkeypatch, tool):
    readers = []
    for module, names in ((orchestrator, ("symbol_search", "stock_summary")),
                          (swing, ("stock_summary", "fetch_history")),
                          (watchlist, ("fetch_history",))):
        for name in names:
            reader = AsyncMock(side_effect=AssertionError("invalid input reached provider"))
            monkeypatch.setattr(module, name, reader)
            readers.append(reader)
    for name in ("resolve_data_dir", "load_watchlist", "load_scan_state", "save_scan_state"):
        reader = Mock(side_effect=AssertionError("invalid input reached watchlist"))
        monkeypatch.setattr(watchlist, name, reader)
        readers.append(reader)
    async with Client(server.mcp) as client:
        for field in FIELDS:
            for raw in (True, "nan", "1e-400", "1_000", 0):
                result = await client.call_tool(
                    tool, arguments(tool, **{field: raw}), raise_on_error=False,
                )
                assert result.is_error, (tool, field, raw)
                assert field in result.content[0].text
    for reader in readers:
        reader.assert_not_called()


@pytest.mark.parametrize("alias", [AccountSize, RiskPercent, PositionPercent])
def test_nonfinite_values_received_by_boundary_are_rejected(alias):
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            TypeAdapter(alias).validate_python(value)


async def test_client_nonfinite_to_null_conversion_is_not_server_validation(capture_handler):
    # Nonfinite numbers are not JSON numbers. This client's serializer loses the
    # original value; the server must preserve its intentional-null semantics.
    async with Client(server.mcp) as client:
        for value in (math.nan, math.inf, -math.inf):
            capture_handler.reset_mock()
            result = await client.call_tool(
                "analyze", arguments("analyze", risk_per_trade_pct=value), raise_on_error=False,
            )
            assert not result.is_error
            capture_handler.assert_awaited_once()
            assert capture_handler.await_args.kwargs["risk_per_trade_pct"] is None
