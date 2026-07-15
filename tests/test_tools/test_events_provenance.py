"""Red-first regressions R1/R2/R3/R4/R13/R17/R18/R19 + P1/P2/P3 post-merge fixes:
earnings provenance, day math, strict/ET-normalized date parsing."""

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


async def _events(monkeypatch, ticker, **kw):
    _patch(monkeypatch, ticker, **kw)
    result = await events_mod.events_calendar("TEST", _now=NOW)
    assert not result.get("error"), result
    return result


async def _earnings(monkeypatch, ticker, **kw):
    result = await _events(monkeypatch, ticker, **kw)
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


@pytest.mark.asyncio
async def test_r18_naive_earnings_timestamp_same_day_is_day_zero(monkeypatch):
    # Naive index entries (no tzinfo) must be treated as exchange-local (ET)
    # wall time. Localizing as UTC first shifts a pre-~04:00 ET wall time back
    # one calendar day, so 00:30 on earnings day would resolve to yesterday
    # and be filtered out by the `row_date >= cutoff` check below.
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-15 00:30")])  # naive
    df = pd.DataFrame({"EPS Estimate": [1.0]}, index=idx)
    ticker = FakeTicker(calendar={}, earnings_dates=df)
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] == "2026-07-15"  # OLD: None (shifted to 07-14, filtered out)
    assert earnings["days_until"] == 0


@pytest.mark.asyncio
async def test_r19_naive_earnings_timestamp_t5_boundary(monkeypatch):
    idx = pd.DatetimeIndex([pd.Timestamp("2026-07-20 16:30")])  # naive
    df = pd.DataFrame({"EPS Estimate": [1.0]}, index=idx)
    ticker = FakeTicker(calendar={}, earnings_dates=df)
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] == "2026-07-20"
    assert earnings["days_until"] == 5


@pytest.mark.asyncio
async def test_p1_invalid_dict_calendar_value_marks_calendar_failed(monkeypatch):
    # OLD: _format_date's string branch passed "not-a-date" through unchanged
    # on parse failure, so next_date came back truthy with status "available"
    # and sources_failed empty -- a garbage date reached callers with no
    # blocker. Present-but-uninterpretable must read as a source failure.
    ticker = FakeTicker(calendar={"Earnings Date": ["not-a-date"]})
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] is None  # OLD: "not-a-date"
    assert earnings["next_date_status"] == "unavailable"  # OLD: "available"
    assert earnings["sources_failed"] == ["calendar"]  # OLD: []


@pytest.mark.asyncio
async def test_p1_invalid_dataframe_calendar_value_marks_calendar_failed(monkeypatch):
    cal = pd.DataFrame({0: ["garbage"]}, index=["Earnings Date"])
    ticker = FakeTicker(calendar=cal)
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] is None  # OLD: "garbage"
    assert earnings["next_date_status"] == "unavailable"  # OLD: "available"
    assert earnings["sources_failed"] == ["calendar"]  # OLD: []


@pytest.mark.asyncio
async def test_p2_mixed_earnings_dates_rows_marks_earnings_dates_failed(monkeypatch):
    # OLD: the earnings_dates branch only marked the source failed when
    # EVERY row was unparseable (rows_unparseable == rows_seen). A valid
    # PAST row alongside one malformed row left next_date None with no
    # failure recorded -- indistinguishable from a verified-empty result,
    # even though the malformed row could have been the future earnings.
    idx = pd.Index(
        [pd.Timestamp("2026-01-15 16:30", tz="America/New_York"), "garbage-index"],
        dtype=object,
    )
    df = pd.DataFrame({"EPS Estimate": [1.0, 1.0]}, index=idx)
    ticker = FakeTicker(calendar={}, earnings_dates=df)
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] is None
    assert earnings["sources_failed"] == ["earnings_dates"]  # OLD: []


@pytest.mark.asyncio
async def test_p3_tz_aware_calendar_value_normalizes_to_et_date(monkeypatch):
    # OLD: _format_date formatted tz-aware values with a bare strftime on
    # the value's own (non-ET) wall clock, so a UTC 00:30 timestamp just
    # after ET midnight rollover reported the UTC date/day count instead
    # of the ET one -- shifting blackout boundaries by a day.
    ticker = FakeTicker(calendar={"Earnings Date": [pd.Timestamp("2026-07-16 00:30", tz="UTC")]})
    earnings = await _earnings(monkeypatch, ticker)
    assert earnings["next_date"] == "2026-07-15"  # OLD: "2026-07-16"
    assert earnings["days_until"] == 0  # OLD: 1


@pytest.mark.asyncio
async def test_p1_invalid_dividend_ex_date_no_longer_passthrough(monkeypatch):
    # Dividends ripple: _build_dividends also calls _format_date, so the
    # same strict-parsing fix must apply to the Ex-Dividend Date field.
    ticker = FakeTicker(calendar={"Ex-Dividend Date": "not-a-date"})
    result = await _events(monkeypatch, ticker)
    assert result["dividends"]["ex_date"] is None  # OLD: "not-a-date"
