"""Observed v0.4.0 position boundary, including defects; not approved future policy.

The real MCP schema, coercion, handler and JSON serializer run here. Only the
technicals source and the position clock are replaced. In particular, accepting
booleans, negative holdings, nonfinite strings and future dates is legacy behavior
scheduled for replacement by the approved design, not a requirement to preserve it.
Starting Step 3a.2 still needs separate authorization.
"""

import json
import math
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from jsonschema import Draft202012Validator

from stock_analysis import server
from stock_analysis.tools import position

ARGUMENTS = {
    "symbol": " acme ",
    "cost_basis": 80,
    "purchase_date": "2026-10-01",
    "shares": 2,
}


@pytest.fixture
def technical_source(monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is None, "legacy position uses a host-local clock"
            return cls(2026, 10, 19, 12)

    source = AsyncMock(return_value={"current_price": 100.0})
    monkeypatch.setattr(position, "technicals", source)
    monkeypatch.setattr(position, "datetime", FixedClock)
    return source


async def call_position(arguments):
    async with Client(server.mcp) as client:
        return await client.call_tool("analyze_my_position", arguments, raise_on_error=False)


def accepted_payload(result, technical_source):
    assert not result.is_error, result.content
    payload = json.loads(result.content[0].text)
    assert not payload.get("error"), payload
    technical_source.assert_awaited_once_with("ACME")
    assert payload["symbol"] == "ACME"
    return payload


async def test_published_position_schema_requires_cost_and_date_despite_runtime_coercion():
    async with Client(server.mcp) as client:
        tool = next(tool for tool in await client.list_tools() if tool.name == "analyze_my_position")
    schema = tool.inputSchema
    Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["symbol", "cost_basis", "purchase_date"]
    assert set(schema["properties"]) == {"symbol", "cost_basis", "purchase_date", "shares"}
    assert schema["properties"]["cost_basis"] == {"type": "number"}
    assert schema["properties"]["shares"] == {
        "anyOf": [{"type": "number"}, {"type": "null"}], "default": None,
    }
    validator = Draft202012Validator(schema)
    assert validator.is_valid(ARGUMENTS)
    assert not validator.is_valid({**ARGUMENTS, "cost_basis": "80"})
    assert not validator.is_valid({**ARGUMENTS, "cost_basis": True})
    assert validator.is_valid({**ARGUMENTS, "cost_basis": -80, "shares": -2})


@pytest.mark.parametrize("field", ["cost_basis", "purchase_date"])
@pytest.mark.parametrize("mode", ["omitted", "null"])
async def test_missing_required_context_is_framework_error_before_data(field, mode, technical_source):
    arguments = dict(ARGUMENTS)
    if mode == "omitted":
        del arguments[field]
    else:
        arguments[field] = None
    result = await call_position(arguments)
    assert result.is_error
    assert field in result.content[0].text
    assert ("missing_argument" if mode == "omitted" else "_type") in result.content[0].text
    technical_source.assert_not_awaited()


@pytest.mark.parametrize("field,raw", [
    ("cost_basis", "not-a-number"), ("cost_basis", []), ("cost_basis", {}),
    ("shares", "not-a-number"), ("shares", []), ("shares", {}),
    ("purchase_date", 20261001), ("purchase_date", True),
])
async def test_bad_types_are_framework_errors_before_data(field, raw, technical_source):
    result = await call_position({**ARGUMENTS, field: raw})
    assert result.is_error
    assert field in result.content[0].text
    technical_source.assert_not_awaited()


@pytest.mark.parametrize("raw", ["2026-02-30", "2025-02-29", "2026/10/01", "", "2026-10-01T00:00:00"])
async def test_bad_date_syntax_is_application_error_before_data(raw, technical_source):
    result = await call_position({**ARGUMENTS, "purchase_date": raw})
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["error"] is True
    assert payload["error_type"] == "invalid_parameters"
    assert payload["message"] == f"Invalid purchase_date format: {raw}. Expected YYYY-MM-DD"
    technical_source.assert_not_awaited()


@pytest.mark.parametrize("raw,cost,gain,gain_dollars", [
    pytest.param(80, 80.0, 0.25, 40.0, id="number"),
    pytest.param("80.5", 80.5, 0.2422, 39.0, id="decimal-string"),
    pytest.param("8e1", 80.0, 0.25, 40.0, id="scientific-string"),
    pytest.param(True, 1.0, 99.0, 198.0, id="boolean-true-coerced"),
    pytest.param(False, 0.0, None, 200.0, id="boolean-false-coerced"),
    pytest.param(-80, -80.0, -2.25, 360.0, id="negative-cost-accepted"),
    pytest.param(0, 0.0, None, 200.0, id="zero-cost"),
    pytest.param("1e-400", 0.0, None, 200.0, id="underflow-becomes-zero"),
    pytest.param("nan", None, None, None, id="nan-sanitized"),
    pytest.param("inf", None, None, None, id="infinity-sanitized"),
    pytest.param("-inf", None, None, None, id="negative-infinity-sanitized"),
    pytest.param("1e309", None, None, None, id="overflow-sanitized"),
])
async def test_legacy_cost_acceptance_and_sanitization(raw, cost, gain, gain_dollars, technical_source):
    payload = accepted_payload(await call_position({**ARGUMENTS, "cost_basis": raw}), technical_source)
    assert payload["position"] == {
        "cost_basis": cost, "current_price": 100.0, "shares": 2.0,
        "current_value": 200.0, "gain_loss": gain, "gain_loss_dollars": gain_dollars,
    }


@pytest.mark.parametrize("raw,shares,value,gain_dollars", [
    pytest.param(2.5, 2.5, 250.0, 50.0, id="fractional-shares"),
    pytest.param("2.5", 2.5, 250.0, 50.0, id="decimal-string"),
    pytest.param("2e0", 2.0, 200.0, 40.0, id="scientific-string"),
    pytest.param(True, 1.0, 100.0, 20.0, id="boolean-true-coerced"),
    pytest.param(False, 0.0, 0.0, 0.0, id="boolean-false-coerced"),
    pytest.param(-2, -2.0, -200.0, -40.0, id="negative-shares-accepted"),
    pytest.param(0, 0.0, 0.0, 0.0, id="zero-shares-accepted"),
    pytest.param("1e-400", 0.0, 0.0, 0.0, id="underflow-becomes-zero"),
    pytest.param("nan", None, None, None, id="nan-sanitized"),
    pytest.param("inf", None, None, None, id="infinity-sanitized"),
    pytest.param("-inf", None, None, None, id="negative-infinity-sanitized"),
    pytest.param("1e309", None, None, None, id="overflow-sanitized"),
    pytest.param(1e308, 1e308, None, None, id="finite-product-overflow-sanitized"),
])
async def test_legacy_share_acceptance_and_sanitization(raw, shares, value, gain_dollars, technical_source):
    payload = accepted_payload(await call_position({**ARGUMENTS, "shares": raw}), technical_source)
    assert payload["position"] == {
        "cost_basis": 80.0, "current_price": 100.0, "shares": shares,
        "current_value": value, "gain_loss": 0.25, "gain_loss_dollars": gain_dollars,
    }


@pytest.mark.parametrize("mode", ["omitted", "null"])
async def test_optional_shares_omit_only_dollar_calculations(mode, technical_source):
    arguments = dict(ARGUMENTS)
    if mode == "omitted":
        del arguments["shares"]
    else:
        arguments["shares"] = None
    payload = accepted_payload(await call_position(arguments), technical_source)
    assert payload["position"] == {
        "cost_basis": 80.0, "current_price": 100.0, "shares": None,
        "current_value": None, "gain_loss": 0.25, "gain_loss_dollars": None,
    }


@pytest.mark.parametrize("field", ["cost_basis", "shares"])
@pytest.mark.parametrize("raw", [math.nan, math.inf, -math.inf])
async def test_client_nonfinite_numbers_arrive_as_null_not_text(field, raw, technical_source):
    # This client's JSON conversion loses the value before the server sees it.
    result = await call_position({**ARGUMENTS, field: raw})
    if field == "cost_basis":
        assert result.is_error
        assert "cost_basis" in result.content[0].text
        assert "float_type" in result.content[0].text
        technical_source.assert_not_awaited()
    else:
        payload = accepted_payload(result, technical_source)
        assert payload["position"]["shares"] is None
        assert payload["position"]["current_value"] is None
        assert payload["position"]["gain_loss_dollars"] is None


@pytest.mark.parametrize("raw,days,long_term,until_long_term", [
    ("2026-10-20", -1, False, 367),
    ("2026-10-19", 0, False, 366),
    ("2026-2-3", 258, False, 108),
    ("2025-10-19", 365, False, 1),
    ("2025-10-18", 366, True, None),
])
async def test_legacy_future_unpadded_and_tax_day_dates(raw, days, long_term, until_long_term, technical_source):
    payload = accepted_payload(await call_position({**ARGUMENTS, "purchase_date": raw}), technical_source)
    assert payload["holding"] == {
        "purchase_date": raw, "days_held": days, "is_long_term": long_term,
        "days_to_long_term": until_long_term,
    }
    assert payload["tax"]["type"] == ("long_term" if long_term else "short_term")
    assert payload["tax"]["rate_if_sold_now"] == (0.20 if long_term else 0.37)
    assert payload["tax"]["rate_if_long_term"] == 0.20


async def test_no_personal_context_is_rejected_before_data(technical_source):
    arguments = {"symbol": "ACME"}
    async with Client(server.mcp) as client:
        tool = next(tool for tool in await client.list_tools() if tool.name == "analyze_my_position")
        assert not Draft202012Validator(tool.inputSchema).is_valid(arguments)
        result = await client.call_tool("analyze_my_position", arguments, raise_on_error=False)
    assert result.is_error
    assert "cost_basis" in result.content[0].text
    assert "purchase_date" in result.content[0].text
    assert result.content[0].text.count("type=missing_argument") == 2
    technical_source.assert_not_awaited()


@pytest.mark.parametrize("field,raw", [
    ("horizon", "long_term"), ("portfolio_value", 10_000), ("max_holding_pct", 10),
])
async def test_future_context_fields_are_rejected_not_silently_ignored(field, raw, technical_source):
    arguments = {**ARGUMENTS, field: raw}
    async with Client(server.mcp) as client:
        tool = next(tool for tool in await client.list_tools() if tool.name == "analyze_my_position")
        assert tool.inputSchema["additionalProperties"] is False
        assert field not in tool.inputSchema["properties"]
        assert not Draft202012Validator(tool.inputSchema).is_valid(arguments)
        result = await client.call_tool("analyze_my_position", arguments, raise_on_error=False)
    assert result.is_error
    assert field in result.content[0].text
    assert "unexpected_keyword_argument" in result.content[0].text
    technical_source.assert_not_awaited()
