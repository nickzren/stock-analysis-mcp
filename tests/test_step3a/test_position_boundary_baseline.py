"""Registered MCP input contract for optional, validated position context.

The client, published schema and serializer are real; all eight market-data tools
are stubbed. The immutable before files retain the superseded v0.4.0 behavior.
"""

import copy
import json
import math
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from jsonschema import Draft202012Validator

from stock_analysis import server
from stock_analysis.tools import position
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm

ARGUMENTS = {"symbol": " acme ", "cost_basis": 80, "purchase_date": "2026-10-01", "shares": 2}
NUMERIC_FIELDS = ("cost_basis", "shares", "portfolio_value", "max_holding_pct")


@pytest.fixture
def analysis_source(monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            clock = hm.POSITION_CLOCK
            return clock.astimezone(tz) if tz is not None else clock.replace(tzinfo=None)

    real = position._analyze_with_context
    analyses = []

    async def capture(*args, **kwargs):
        result, context = await real(*args, **kwargs)
        analyses.append(copy.deepcopy(result))
        return result, context

    source = AsyncMock(side_effect=capture)
    source.analyses = analyses
    monkeypatch.setattr(position, "_analyze_with_context", source)
    monkeypatch.setattr(position, "datetime", FixedClock)
    with dm.stubbed(hm.SOUND):
        yield source


async def call_position(arguments):
    async with Client(server.mcp) as client:
        return await client.call_tool("analyze_my_position", arguments, raise_on_error=False)


def accepted_payload(result, source):
    assert not result.is_error, result.content
    payload = json.loads(result.content[0].text)
    assert not payload.get("error"), payload
    source.assert_awaited_once()
    assert payload["symbol"] == "ACME"
    return payload


def assert_rejected(result, source):
    if not result.is_error:
        payload = json.loads(result.content[0].text)
        assert payload["error"] is True
        assert payload["error_type"] == "invalid_parameters"
    source.assert_not_awaited()


async def test_published_schema_has_optional_position_context_and_decimal_strings():
    async with Client(server.mcp) as client:
        tool = next(tool for tool in await client.list_tools() if tool.name == "analyze_my_position")
    schema = tool.inputSchema
    Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["symbol"]
    assert set(schema["properties"]) == {
        "symbol", "cost_basis", "purchase_date", "shares", "horizon", "portfolio_value", "max_holding_pct"}
    assert schema["additionalProperties"] is False
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"symbol": "ACME"})
    for field in NUMERIC_FIELDS:
        for value in (1.5, "1.5", "1.5e0", None):
            assert validator.is_valid({**ARGUMENTS, field: value}), (field, value)
        assert not validator.is_valid({**ARGUMENTS, field: True}), field
        assert not validator.is_valid({**ARGUMENTS, field: "nan"}), field
    assert not validator.is_valid({**ARGUMENTS, "horizon": "swing"})


@pytest.mark.parametrize("field", ["cost_basis", "purchase_date", "shares", "portfolio_value", "max_holding_pct"])
@pytest.mark.parametrize("mode", ["omitted", "null"])
async def test_context_is_optional(field, mode, analysis_source):
    arguments = dict(ARGUMENTS)
    if mode == "omitted":
        arguments.pop(field, None)
    else:
        arguments[field] = None
    payload = accepted_payload(await call_position(arguments), analysis_source)
    assert payload["position_card"]["holder_assessment"]["action"] == "hold"
    if field in ("cost_basis", "shares"):
        assert payload["position"][field] is None
        assert payload["position"]["gain_loss_dollars"] is None


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize("raw", [True, False, "nan", "inf", "-inf", "1e309", "1e-400", -1, [], {}])
async def test_invalid_numeric_context_is_rejected_before_analysis(field, raw, analysis_source):
    assert_rejected(await call_position({**ARGUMENTS, field: raw}), analysis_source)


@pytest.mark.parametrize("field,raw", [
    ("shares", 0), ("portfolio_value", 0), ("max_holding_pct", 0), ("max_holding_pct", 100.01),
    ("purchase_date", 20261001), ("purchase_date", True), ("horizon", "swing"), ("horizon", None),
])
async def test_invalid_ranges_and_types_are_rejected_before_analysis(field, raw, analysis_source):
    assert_rejected(await call_position({**ARGUMENTS, field: raw}), analysis_source)


@pytest.mark.parametrize("raw", ["2026-02-30", "2025-02-29", "2026/10/01", "", "2026-10-01T00:00:00",
                                  "2026-2-3", "2026-10-10"])
async def test_invalid_or_future_et_date_is_rejected_before_analysis(raw, analysis_source):
    result = await call_position({**ARGUMENTS, "purchase_date": raw})
    assert_rejected(result, analysis_source)


@pytest.mark.parametrize("raw,cost,gain,gain_dollars", [
    (80, 80.0, .25, 40.0), ("80.5", 80.5, .2422, 39.0), ("8e1", 80.0, .25, 40.0),
    (0, 0.0, None, 200.0), ("0e10", 0.0, None, 200.0),
])
async def test_numeric_cost_and_decimal_strings_have_defined_zero_handling(raw, cost, gain, gain_dollars,
                                                                         analysis_source):
    body = accepted_payload(await call_position({**ARGUMENTS, "cost_basis": raw}), analysis_source)["position"]
    assert body["cost_basis"] == cost
    assert body["gain_loss"] == gain
    assert body["gain_loss_dollars"] == gain_dollars
    assert body["current_value"] == 200


@pytest.mark.parametrize("raw", [2.5, "2.5", "2.5e0"])
async def test_fractional_owned_shares_are_not_rounded_to_entry_whole_shares(raw, analysis_source):
    body = accepted_payload(await call_position({**ARGUMENTS, "shares": raw}), analysis_source)["position"]
    assert body["shares"] == 2.5
    assert body["current_value"] == 250
    assert body["gain_loss_dollars"] == 50


@pytest.mark.parametrize("raw", [10_000, "10000", "1e4"])
async def test_numeric_and_text_portfolio_context_uses_explicit_ceiling(raw, analysis_source):
    body = accepted_payload(await call_position({**ARGUMENTS, "portfolio_value": raw,
                                                 "max_holding_pct": "1.5"}), analysis_source)
    assert body["position_card"]["concentration_check"]["status"] == "above_limit"
    assert body["position_card"]["review_status"] == "review_allocation"


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize("raw", [math.nan, math.inf, -math.inf])
async def test_client_nonfinite_numbers_arrive_as_null_not_nonfinite_text(field, raw, analysis_source):
    # This client's serializer loses these values before the handler sees them.
    payload = accepted_payload(await call_position({**ARGUMENTS, field: raw}), analysis_source)
    assert payload["position"][field] is None


async def test_finite_arithmetic_overflow_is_explicit_partial_context(analysis_source):
    payload = accepted_payload(await call_position({**ARGUMENTS, "shares": 1e308,
                                                    "portfolio_value": 1e308, "max_holding_pct": 10}),
                               analysis_source)
    assert payload["position"]["current_value"] is None
    assert payload["position_card"]["context_status"]["status"] == "partial"
    assert payload["position_card"]["review_status"] == "not_evaluated"
    assert "numeric_range_unavailable" in json.dumps(payload["position"]["calculation_reasons"])
    assert payload["position_card"]["holder_assessment"]["action"] == "hold"


async def test_no_personal_context_has_useful_stock_assessment_without_defaults(analysis_source):
    payload = accepted_payload(await call_position({"symbol": "ACME"}), analysis_source)
    assert payload["position_card"]["horizon"] == "mid_term"
    assert payload["position_card"]["holder_assessment"]["action"] == "hold"
    assert payload["position_card"]["context_status"]["status"] == "not_supplied"
    assert payload["position_card"]["review_status"] == "not_requested"
    assert payload["holding"] == {"purchase_date": None, "days_held": None}
    for field in ("cost_basis", "shares", "current_value", "gain_loss", "gain_loss_dollars"):
        assert payload["position"][field] is None


async def test_client_context_changes_cannot_feed_analysis_or_persist_personal_data(analysis_source):
    data_dir = Path(os.environ["STOCK_ANALYSIS_DATA_DIR"])
    original = {p.relative_to(data_dir): p.read_bytes() for p in data_dir.rglob("*") if p.is_file()}
    requests = [{"symbol": ARGUMENTS["symbol"]}, ARGUMENTS,
                {**ARGUMENTS, "shares": "0.5", "cost_basis": 0, "purchase_date": "2024-02-29",
                 "horizon": "long_term", "portfolio_value": 50_000, "max_holding_pct": .01}]
    for arguments in requests:
        result = await call_position(arguments)
        assert not result.is_error
        assert not json.loads(result.content[0].text).get("error")
    assert analysis_source.await_count == 3
    assert analysis_source.await_args_list[0] == analysis_source.await_args_list[1] == analysis_source.await_args_list[2]
    records = analysis_source.analyses
    for record in records[1:]:
        for field in ("decision_card", "watchlist_snapshot", "action_zones", "decision_modes", "verdict"):
            assert record[field] == records[0][field], field
        assert record["watchlist_snapshot"]["snapshot_version"] == "3.0.0"
    assert {p.relative_to(data_dir): p.read_bytes() for p in data_dir.rglob("*") if p.is_file()} == original
