"""C.1: preserve error-response classifications through aggregation."""

import json
from typing import Any

import pytest

from stock_analysis.tools.stock_summary import stock_summary
from stock_analysis.tools.technicals import technicals
from stock_analysis.tools.trade_setup import orchestrator as trade
from stock_analysis.utils.provenance import SymbolNotFoundError
from tests.support import decision_matrix as dm
from tests.support.transport import FakeSession, TransportFixture


async def _run_analyze(
    monkeypatch: pytest.MonkeyPatch,
    summary: dict[str, Any] | Exception,
    technical: dict[str, Any] | Exception,
) -> dict[str, Any]:
    def tool(value: dict[str, Any] | Exception) -> Any:
        async def read(symbol: str, **kwargs: Any) -> dict[str, Any]:
            if isinstance(value, Exception):
                raise value
            return value
        return read

    with dm.stubbed(dm.SCENARIOS["S2_buy_clean"]), monkeypatch.context() as patch:
        patch.setattr(dm.orchestrator, "stock_summary", tool(summary))
        patch.setattr(dm.orchestrator, "technicals", tool(technical))
        return await dm.orchestrator.analyze_stock("ACME", _now=dm.FRIDAY_EVENING)


def error(kind: str | None, message: str) -> dict[str, Any]:
    result: dict[str, Any] = {"error": True, "message": message}
    if kind is not None:
        result["error_type"] = kind
    return result


@pytest.mark.parametrize("message", [
    "Failed to fetch data: no data returned while parsing provider payload",
    "Failed to fetch data: internal parser encountered invalid symbol field",
    "Failed to fetch data: incomplete yfinance info after invalid crumb",
])
@pytest.mark.parametrize("kind", ["data_unavailable", "ValueError", "SymbolNotFoundError", None])
async def test_structured_failures_ignore_message_hints(
    monkeypatch: pytest.MonkeyPatch, kind: str | None, message: str,
) -> None:
    result = await _run_analyze(monkeypatch, error(kind, message), error(kind, message))
    assert result["error"] is True
    assert result["error_type"] == "data_unavailable"
    for failure in result["data_quality"]["tool_failures"]:
        assert set(failure) == {"tool", "error", "message", "duration_ms"}


@pytest.mark.parametrize("reverse", [False, True])
async def test_mixed_failures_do_not_report_an_invalid_symbol(
    monkeypatch: pytest.MonkeyPatch, reverse: bool,
) -> None:
    failures = [error("invalid_symbol", "Invalid symbol"), error("data_unavailable", "no data returned during parsing")]
    if reverse:
        failures.reverse()
    result = await _run_analyze(monkeypatch, *failures)
    assert result["error_type"] == "data_unavailable"


@pytest.mark.parametrize(("summary", "technical", "expected"), [
    (error("invalid_symbol", "not found"), error("invalid_symbol", "not found"), "invalid_symbol"),
    (ValueError("no data returned"), RuntimeError("possibly delisted"), "invalid_symbol"),
    (SymbolNotFoundError("gone"), SymbolNotFoundError("gone"), "invalid_symbol"),
    (RuntimeError("read failed"), RuntimeError("read failed"), "data_unavailable"),
    (type("invalid_symbol", (Exception,), {})("read failed"), type("invalid_symbol", (Exception,), {})("read failed"), "data_unavailable"),
])
async def test_exception_origin_and_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch, summary: Any, technical: Any, expected: str,
) -> None:
    result = await _run_analyze(monkeypatch, summary, technical)
    assert result["error_type"] == expected


async def test_one_failed_price_source_does_not_enter_the_top_level_error_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = await _run_analyze(
        monkeypatch, error("data_unavailable", "invalid symbol field in payload"), dm.STRONG,
    )
    assert not result.get("error")


@pytest.mark.parametrize(("summary_type", "technical_type", "expected"), [
    ("invalid_symbol", "invalid_symbol", "invalid_symbol"),
    ("invalid_symbol", "data_unavailable", "data_unavailable"),
    ("data_unavailable", "invalid_symbol", "data_unavailable"),
    ("data_unavailable", "data_unavailable", "data_unavailable"),
])
async def test_trade_setup_keeps_structured_unanimous_invalid_checks(
    monkeypatch: pytest.MonkeyPatch, summary_type: str, technical_type: str, expected: str,
) -> None:
    async def summary(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return error(summary_type, "no data returned while parsing")

    async def technical(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return error(technical_type, "invalid symbol field during parsing")

    async def empty(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    async def no_history(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(trade, "stock_summary", summary)
    monkeypatch.setattr(trade, "technicals", technical)
    monkeypatch.setattr(trade, "risk_metrics", empty)
    monkeypatch.setattr(trade, "events_calendar", empty)
    monkeypatch.setattr(trade, "fetch_history", no_history)
    result = await trade.analyze_trade_setup("ACME", _now=dm.FRIDAY_EVENING)
    assert result["error_type"] == expected


async def test_real_unknown_symbol_keeps_the_analyze_invalid_symbol_label(
    monkeypatch: pytest.MonkeyPatch, yf_transport: FakeSession,
) -> None:
    yf_transport.route("calendar", "http_404_json")
    yf_transport.route("quote_v7", "empty_result")
    yf_transport._queues["chart"] = [TransportFixture(404, json.dumps({
        "chart": {"result": None, "error": {"code": "Not Found", "description": "No data found"}},
    }), "application/json")]
    with dm.stubbed(dm.SCENARIOS["S2_buy_clean"]), monkeypatch.context() as patch:
        patch.setattr(dm.orchestrator, "stock_summary", stock_summary)
        patch.setattr(dm.orchestrator, "technicals", technicals)
        result = await dm.orchestrator.analyze_stock("EXMP", _now=dm.FRIDAY_EVENING)
    assert result["error_type"] == "invalid_symbol"
    assert {f["error"] for f in result["data_quality"]["tool_failures"]} == {"invalid_symbol"}
    assert not yf_transport.unrouted
