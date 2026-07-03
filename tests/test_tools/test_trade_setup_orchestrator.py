"""Orchestrator tests with all fetches monkeypatched (no network)."""

from datetime import datetime
from typing import Any

import pandas as pd
import pytest
import pytz

import stock_analysis.tools.trade_setup.orchestrator as orch
from tests.test_tools.test_setup_detection import make_technicals

ET = pytz.timezone("America/New_York")
NOW_EVENING = ET.localize(datetime(2026, 3, 10, 18, 30))  # Tuesday after hours


def daily_df() -> pd.DataFrame:
    n = 60
    closes = [90.0 + i * 0.2 for i in range(n)]  # gentle uptrend
    dates = pd.date_range("2025-12-15", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    dates[-1] = "2026-03-10"  # EOD-fresh for NOW_EVENING
    return pd.DataFrame({
        "date": dates,
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
        "volume": [1_000_000.0] * n,
    })


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_summary(symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "name": "Test Co", "currency": "USD",
                "current_price": 101.8}

    async def fake_technicals(symbol: str) -> dict[str, Any]:
        return make_technicals()

    async def fake_risk(symbol: str) -> dict[str, Any]:
        return {"liquidity": {"avg_dollar_volume": 50_000_000}}

    async def fake_events(symbol: str) -> dict[str, Any]:
        return {"earnings": {"days_until": 40, "next_date": "2026-04-19"}}

    async def fake_history(params: Any) -> pd.DataFrame:
        if params.interval == "1d":
            return daily_df()
        raise ValueError("no intraday data off-hours")

    monkeypatch.setattr(orch, "stock_summary", fake_summary)
    monkeypatch.setattr(orch, "technicals", fake_technicals)
    monkeypatch.setattr(orch, "risk_metrics", fake_risk)
    monkeypatch.setattr(orch, "events_calendar", fake_events)
    monkeypatch.setattr(orch, "fetch_history", fake_history)
    monkeypatch.setattr(orch, "get_market_state", lambda: {"state": "after_hours"})


@pytest.mark.asyncio
async def test_off_hours_card_reaches_enter_or_watch_states(patched: None) -> None:
    result = await orch.analyze_trade_setup("TEST", _now=NOW_EVENING)
    assert result.get("error") is None
    assert result["action"] in {"trade_now", "enter_on_trigger", "watch", "no_setup"}
    assert result["action"] != "trade_now"  # session invariant off-hours
    assert result["freshness"]["session"] == "after_hours"
    assert result["meta"]["tool"] == "analyze_trade_setup"


@pytest.mark.asyncio
async def test_probe_failure_off_hours_still_works(patched: None) -> None:
    result = await orch.analyze_trade_setup("TEST", _now=NOW_EVENING)
    assert result["freshness"]["basis"] == "bar_timestamp"  # daily EOD check
    assert result["freshness"]["stale"] is False


@pytest.mark.asyncio
async def test_all_core_tools_failing_returns_error(patched: None,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(symbol: str) -> dict[str, Any]:
        return {"error": True, "error_type": "invalid_symbol", "message": "bad"}

    monkeypatch.setattr(orch, "stock_summary", boom)
    monkeypatch.setattr(orch, "technicals", boom)
    result = await orch.analyze_trade_setup("NOPE", _now=NOW_EVENING)
    assert result["error"] is True


@pytest.mark.asyncio
async def test_technicals_failure_alone_degrades_to_wait_for_data(
    patched: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(symbol: str) -> dict[str, Any]:
        return {"error": True, "error_type": "fetch_error", "message": "boom"}

    monkeypatch.setattr(orch, "technicals", boom)
    result = await orch.analyze_trade_setup("TEST", _now=NOW_EVENING)
    assert result["action"] == "wait_for_data"
    assert any(b["id"] == "data_quality_critical" for b in result["blockers"])
