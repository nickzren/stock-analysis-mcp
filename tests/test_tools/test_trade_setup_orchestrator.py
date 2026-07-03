"""Orchestrator tests with all fetches monkeypatched (no network)."""

import math
from datetime import datetime
from typing import Any

import pandas as pd
import pytest
import pytz

import stock_analysis.tools.trade_setup.orchestrator as orch
from tests.test_tools.test_setup_detection import make_technicals

ET = pytz.timezone("America/New_York")
NOW_EVENING = ET.localize(datetime(2026, 3, 10, 18, 30))  # Tuesday after hours
NOW_REGULAR = ET.localize(datetime(2026, 3, 10, 10, 30))  # Tuesday regular session


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


def breakout_daily_df() -> pd.DataFrame:
    """60 daily bars: a noisy early uptrend (keeps historical bandwidth wide),
    a tight 24-bar consolidation (compresses current bandwidth into the
    breakout detector's <=25th percentile band), then a final bar that clears
    the 20d high on 2x volume (breakout trigger_satisfied)."""
    n = 60
    base: list[float] = []
    price = 85.0
    for i in range(35):
        price += 0.3 + 1.5 * math.sin(i * 0.9)
        base.append(round(price, 2))
    consolidation = [base[-1] + 0.02 * (i % 2) for i in range(24)]
    breakout_close = consolidation[-1] + 1.0
    closes = base + consolidation + [breakout_close]
    highs = [c + 0.1 for c in closes[:-1]] + [breakout_close + 0.1]
    lows = [c - 0.1 for c in closes[:-1]] + [breakout_close - 0.1]
    volumes = [1_000_000.0] * (n - 2) + [1_000_000.0, 2_000_000.0]  # last-bar ratio 2.0

    dates = pd.date_range("2025-12-15", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    dates[-1] = "2026-03-10"
    return pd.DataFrame({
        "date": dates,
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": volumes,
    })


def breakout_probe_df() -> pd.DataFrame:
    """5m intraday bar, timestamped 5 minutes before NOW_REGULAR (within the
    15m freshness ceiling), with a close above the breakout trigger."""
    return pd.DataFrame({
        "date": ["2026-03-10T10:25:00-0400"],
        "open": [96.5], "high": [96.9], "low": [96.4], "close": [96.7],
        "volume": [500_000.0],
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


@pytest.mark.asyncio
async def test_regular_session_fresh_satisfied_trigger_is_trade_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: regular session, fresh 5m probe above the breakout trigger
    with qualifying volume -> trade_now with a market entry."""
    async def fake_summary(symbol: str) -> dict[str, Any]:
        return {"symbol": symbol, "name": "Test Co", "currency": "USD",
                "current_price": 96.7}

    breakout_technicals = make_technicals(
        rsi={"value": 65.0, "bullish_divergence": False},  # outside pullback/meanrev RSI bands
        atr={"value": 1.0, "value_pct": 0.01},
        price_position={"position_in_range": 0.8, "days_since_52w_high": 2},
        returns={"return_3m": 0.10, "return_1w_zscore": 0.5},
        volume={"ratio": 2.0},
    )

    async def fake_technicals(symbol: str) -> dict[str, Any]:
        return breakout_technicals

    async def fake_risk(symbol: str) -> dict[str, Any]:
        return {"liquidity": {"avg_dollar_volume": 50_000_000}}

    async def fake_events(symbol: str) -> dict[str, Any]:
        return {"earnings": {"days_until": 40, "next_date": "2026-04-19"}}

    async def fake_history(params: Any) -> pd.DataFrame:
        if params.interval == "1d":
            return breakout_daily_df()
        return breakout_probe_df()

    monkeypatch.setattr(orch, "stock_summary", fake_summary)
    monkeypatch.setattr(orch, "technicals", fake_technicals)
    monkeypatch.setattr(orch, "risk_metrics", fake_risk)
    monkeypatch.setattr(orch, "events_calendar", fake_events)
    monkeypatch.setattr(orch, "fetch_history", fake_history)
    monkeypatch.setattr(orch, "get_market_state", lambda: {"state": "regular"})

    result = await orch.analyze_trade_setup("TEST", _now=NOW_REGULAR)

    assert result.get("error") is None
    assert result["action"] == "trade_now"
    assert result["plan"]["entry"]["type"] == "market"
    assert result["freshness"]["basis"] == "bar_timestamp"
    assert result["freshness"]["stale"] is False


@pytest.mark.asyncio
async def test_transport_failures_report_data_unavailable(
    patched: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def transport_error(symbol: str) -> dict[str, Any]:
        return {"error": True, "error_type": "data_unavailable",
                "message": "Failed to fetch data: connection refused"}

    monkeypatch.setattr(orch, "stock_summary", transport_error)
    monkeypatch.setattr(orch, "technicals", transport_error)
    result = await orch.analyze_trade_setup("AAPL", _now=NOW_EVENING)
    assert result["error"] is True
    assert result["error_type"] == "data_unavailable"
    assert "AAPL" in result["message"]
    failures = result["data_quality"]["tool_failures"]
    assert {tf["tool"] for tf in failures} == {"stock_summary", "technicals"}


@pytest.mark.asyncio
async def test_unanimous_invalid_symbol_stays_invalid(
    patched: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def invalid(symbol: str) -> dict[str, Any]:
        return {"error": True, "error_type": "invalid_symbol", "message": "Invalid symbol"}

    monkeypatch.setattr(orch, "stock_summary", invalid)
    monkeypatch.setattr(orch, "technicals", invalid)
    result = await orch.analyze_trade_setup("NOPE", _now=NOW_EVENING)
    assert result["error"] is True
    assert result["error_type"] == "invalid_symbol"


@pytest.mark.asyncio
async def test_mixed_failures_prefer_data_unavailable(
    patched: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def invalid(symbol: str) -> dict[str, Any]:
        return {"error": True, "error_type": "invalid_symbol", "message": "Invalid symbol"}

    async def raises(symbol: str) -> dict[str, Any]:
        raise RuntimeError("socket closed")

    monkeypatch.setattr(orch, "stock_summary", invalid)
    monkeypatch.setattr(orch, "technicals", raises)
    result = await orch.analyze_trade_setup("AAPL", _now=NOW_EVENING)
    assert result["error"] is True
    assert result["error_type"] == "data_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(("kwargs", "bad_param"), [
    ({"account_size": 0.0}, "account_size"),
    ({"account_size": -5000.0}, "account_size"),
    ({"risk_per_trade_pct": 0.0}, "risk_per_trade_pct"),
    ({"risk_per_trade_pct": -1.0}, "risk_per_trade_pct"),
    ({"risk_per_trade_pct": 150.0}, "risk_per_trade_pct"),
    ({"max_position_pct": 0.0}, "max_position_pct"),
    ({"max_position_pct": 101.0}, "max_position_pct"),
])
async def test_out_of_bounds_sizing_inputs_rejected(
    patched: None, kwargs: dict[str, float], bad_param: str,
) -> None:
    result = await orch.analyze_trade_setup("TEST", _now=NOW_EVENING, **kwargs)
    assert result["error"] is True
    assert result["error_type"] == "invalid_parameters"
    assert bad_param in result["message"]
