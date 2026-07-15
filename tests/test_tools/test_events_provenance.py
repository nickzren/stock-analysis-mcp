"""Red-first regressions R1/R2/R3/R4/R13/R17: earnings provenance + day math."""

from datetime import datetime

import pandas as pd
import pytest
import pytz

from stock_analysis.tools import events as events_mod

_ET = pytz.timezone("America/New_York")
NOW = _ET.localize(datetime(2026, 7, 15, 14, 30))  # Wed, regular hours


class FakeTicker:
    def __init__(self, calendar=None, earnings_dates=None,
                 calendar_raises=False, earnings_dates_raises=False):
        self._calendar = calendar
        self._earnings_dates = earnings_dates
        self._calendar_raises = calendar_raises
        self._earnings_dates_raises = earnings_dates_raises

    @property
    def calendar(self):
        if self._calendar_raises:
            raise ConnectionError("transport down")
        return self._calendar

    @property
    def earnings_dates(self):
        if self._earnings_dates_raises:
            raise ConnectionError("transport down")
        return self._earnings_dates

    @property
    def splits(self):
        return pd.Series(dtype=float)


def _patch(monkeypatch, ticker, info=None, info_raises=False):
    async def fake_fetch_ticker(symbol):
        return ticker

    async def fake_fetch_info(symbol):
        if info_raises:
            raise ConnectionError("info down")
        return info or {}

    monkeypatch.setattr(events_mod, "fetch_ticker", fake_fetch_ticker)
    monkeypatch.setattr(events_mod, "fetch_info", fake_fetch_info)


async def _earnings(monkeypatch, ticker, **kw):
    _patch(monkeypatch, ticker, **kw)
    result = await events_mod.events_calendar("TEST", _now=NOW)
    assert not result.get("error"), result
    return result["earnings"]


@pytest.mark.asyncio
async def test_r1_same_day_earnings_is_day_zero(monkeypatch):
    ticker = FakeTicker(calendar={"Earnings Date": [NOW.date()]})
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] == "2026-07-15"
    assert earnings["days_until"] == 0  # OLD: -1 (floor of negative timedelta)


@pytest.mark.asyncio
async def test_r2_transport_failures_populate_sources_failed(monkeypatch):
    ticker = FakeTicker(calendar_raises=True, earnings_dates_raises=True)
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] is None
    assert earnings["sources_failed"] == ["calendar", "earnings_dates"]


@pytest.mark.asyncio
async def test_r3_verified_empty_has_no_failed_sources(monkeypatch):
    ticker = FakeTicker(calendar={}, earnings_dates=None)
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] is None
    assert earnings["sources_failed"] == []


@pytest.mark.asyncio
async def test_r4_t5_vs_t6_boundary_via_earnings_dates(monkeypatch):
    # T-5: 2026-07-20; T-6: 2026-07-21. earnings_dates is the resolving source.
    for target, expected_days in (("2026-07-20", 5), ("2026-07-21", 6)):
        idx = pd.DatetimeIndex([pd.Timestamp(f"{target} 16:30", tz="America/New_York")])
        df = pd.DataFrame({"EPS Estimate": [1.0]}, index=idx)
        ticker = FakeTicker(calendar={}, earnings_dates=df)
        earnings = await _earnings(monkeypatch, ticker)
        assert earnings["next_date"] == target
        assert earnings["days_until"] == expected_days  # OLD: one less (wall-clock floor)


@pytest.mark.asyncio
async def test_r13_dataframe_calendar_still_supported(monkeypatch):
    cal = pd.DataFrame({0: [pd.Timestamp("2026-07-20")]}, index=["Earnings Date"])
    ticker = FakeTicker(calendar=cal)
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] == "2026-07-20"
    assert earnings["sources_failed"] == []


@pytest.mark.asyncio
async def test_r17_invalid_info_timestamp_marks_info_failed(monkeypatch):
    ticker = FakeTicker(calendar={}, earnings_dates=None)
    earnings = await _earnings(
        monkeypatch, ticker, info={"earningsTimestamp": "not-a-number"},
    )
    assert earnings["next_date"] is None
    assert "info" in earnings["sources_failed"]


@pytest.mark.asyncio
async def test_info_fetch_raise_marks_info_failed(monkeypatch):
    ticker = FakeTicker(calendar={}, earnings_dates=None)
    earnings = await _earnings(monkeypatch, ticker, info_raises=True)
    assert earnings["sources_failed"] == ["info"]
