"""Wiring tests: timeframe param, back-compat pin, block presence."""

import importlib
from datetime import datetime
from typing import Any

import pandas as pd
import pytest
import pytz

# NOTE: tools/__init__.py re-exports the `technicals` FUNCTION under the same
# name as its submodule, so `import stock_analysis.tools.technicals as ...`
# binds the function. importlib resolves the real module.
tech_mod = importlib.import_module("stock_analysis.tools.technicals")

ET = pytz.timezone("America/New_York")
NOW = ET.localize(datetime(2026, 3, 10, 11, 30))

PRE_STEP2_KEYS = {
    "meta", "data_provenance", "symbol", "current_price", "moving_averages",
    "rsi", "macd", "atr", "price_position", "returns", "volume", "bollinger",
    "obv", "fibonacci", "price_action",
}


def daily_df(n: int = 60) -> pd.DataFrame:
    closes = [90.0 + i * 0.2 for i in range(n)]
    dates = pd.date_range("2025-12-15", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    dates[-1] = "2026-03-10"
    return pd.DataFrame({"date": dates, "open": closes,
                         "high": [c + 1 for c in closes],
                         "low": [c - 1 for c in closes],
                         "close": closes, "volume": [1_000_000.0] * n})


def intraday_5m() -> pd.DataFrame:
    return pd.DataFrame({"date": ["2026-03-10T11:25:00-0400"], "open": [101.5],
                         "high": [102.0], "low": [101.0], "close": [101.8],
                         "volume": [50_000.0]})


def hourly_1h(n: int = 60) -> pd.DataFrame:
    closes = [100.0 + 0.1 * i for i in range(n)]
    dates = pd.date_range("2026-02-01 10:00", periods=n, freq="h",
                          tz="America/New_York").strftime("%Y-%m-%dT%H:%M:%S%z")
    return pd.DataFrame({"date": list(dates), "open": closes,
                         "high": [c + 0.5 for c in closes],
                         "low": [c - 0.5 for c in closes],
                         "close": closes, "volume": [10_000.0] * n})


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_history(params: Any) -> pd.DataFrame:
        if params.interval == "1d":
            return daily_df()
        if params.interval == "5m":
            return intraday_5m()
        return hourly_1h()

    monkeypatch.setattr(tech_mod, "fetch_history", fake_history)
    monkeypatch.setattr(tech_mod, "get_market_state", lambda: {"state": "regular"})


@pytest.mark.asyncio
async def test_default_response_is_backward_compatible(patched: None) -> None:
    result = await tech_mod.technicals("TEST", _now=NOW)
    assert set(result.keys()) >= PRE_STEP2_KEYS
    new_keys = set(result.keys()) - PRE_STEP2_KEYS
    assert new_keys == {"short_term"}  # exactly one additive key
    assert "intraday" not in result


@pytest.mark.asyncio
async def test_short_term_block_populated(patched: None) -> None:
    result = await tech_mod.technicals("TEST", _now=NOW)
    st = result["short_term"]
    assert st["levels"]["prior_day_high"] is not None
    assert st["rvol"]["basis"] == "partial_day"  # bar==today, session regular


@pytest.mark.asyncio
async def test_swing_mode_adds_intraday(patched: None) -> None:
    result = await tech_mod.technicals("TEST", timeframe="swing", _now=NOW)
    assert "intraday" in result
    assert result["intraday"]["vwap"] is not None
    assert result["intraday"]["freshness"]["basis"] == "bar_timestamp"


@pytest.mark.asyncio
async def test_swing_intraday_fetch_failure_degrades_with_warnings(
    patched: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_intraday(params: Any) -> pd.DataFrame:
        if params.interval == "1d":
            return daily_df()
        raise ValueError("no intraday")

    monkeypatch.setattr(tech_mod, "fetch_history", fail_intraday)
    result = await tech_mod.technicals("TEST", timeframe="swing", _now=NOW)
    assert result["short_term"] is not None  # daily payload intact
    intraday = result["intraday"]
    assert intraday["vwap"] is None
    ids = {w["id"] for w in intraday["warnings"]}
    assert "intraday_unavailable" in ids and "hourly_unavailable" in ids


@pytest.mark.asyncio
async def test_invalid_timeframe_rejected_before_fetch(
    patched: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def explode(params: Any) -> pd.DataFrame:
        raise AssertionError("fetch must not run")

    monkeypatch.setattr(tech_mod, "fetch_history", explode)
    result = await tech_mod.technicals("TEST", timeframe="daily")
    assert result["error"] is True
    assert result["error_type"] == "invalid_parameters"
    assert "timeframe" in result["message"]
