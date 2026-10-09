"""Red-first regressions R1/R2/R3/R4/R17/R18/R19: earnings provenance and day
math, fed through the provider reads (WP2). Legacy Ticker.calendar shapes
(DataFrame calendars, NaT or string dates) no longer reach the resolver: the
provider validates calendarEvents and the earnings page first (WP1)."""

from datetime import datetime
from typing import Any

import pytest
import pytz

from stock_analysis.data.provider import SourceRead
from stock_analysis.tools import events as events_mod
from tests.support.fake_source import FakeSource
from tests.support.fake_ticker import FakeTicker

_ET = pytz.timezone("America/New_York")
NOW = _ET.localize(datetime(2026, 7, 15, 14, 30))  # Wed, regular hours

NO_CALENDAR = FakeSource.empty("calendar", {})
NO_PAGE = FakeSource.empty("earnings_page", [])


def _calendar_at(when: datetime) -> FakeSource:
    return FakeSource.ok("calendar", {"earnings": {"earningsDate": [int(when.timestamp())]}})


def _page_at(when: datetime) -> FakeSource:
    row = {"timestamp": when, "eps_estimate": 1.0, "reported_eps": None, "surprise_pct": None}
    return FakeSource(SourceRead("earnings_page", "ok", payload=[row]))


async def _earnings(
    monkeypatch: pytest.MonkeyPatch,
    cal: FakeSource = NO_CALENDAR,
    pg: FakeSource = NO_PAGE,
    info: dict[str, Any] | None = None,
    info_raises: bool = False,
) -> dict[str, Any]:
    async def fake_fetch_ticker(symbol: str) -> FakeTicker:
        return FakeTicker(symbol, splits=None)

    async def fake_fetch_info(symbol: str) -> dict[str, Any]:
        if info_raises:
            raise ConnectionError("info down")
        return info or {}

    monkeypatch.setattr(events_mod, "fetch_ticker", fake_fetch_ticker)
    monkeypatch.setattr(events_mod, "fetch_info", fake_fetch_info)
    monkeypatch.setattr(events_mod, "read_calendar_events", cal)
    monkeypatch.setattr(events_mod, "read_earnings_page", pg)
    result = await events_mod.events_calendar("TEST", _now=NOW)
    assert not result.get("error"), result
    return result["earnings"]


async def test_r1_same_day_earnings_is_day_zero(monkeypatch):
    earnings = await _earnings(monkeypatch, _calendar_at(_ET.localize(datetime(2026, 7, 15, 16, 0))))
    assert earnings["next_date"] == "2026-07-15"
    assert earnings["days_until"] == 0  # OLD: -1 (floor of negative timedelta)


async def test_r2_transport_failures_populate_sources_failed(monkeypatch):
    earnings = await _earnings(
        monkeypatch, FakeSource.failed("calendar"), FakeSource.failed("earnings_page")
    )
    assert earnings["next_date"] is None
    assert earnings["sources_failed"] == ["calendar", "earnings_dates"]


async def test_r3_verified_empty_has_no_failed_sources(monkeypatch):
    earnings = await _earnings(monkeypatch)
    assert earnings["next_date"] is None
    assert earnings["sources_failed"] == []
    assert earnings["verification"] == "verified_none"


@pytest.mark.parametrize(("target", "expected_days"), [("2026-07-20", 5), ("2026-07-21", 6)])
async def test_r4_t5_vs_t6_boundary_via_earnings_dates(monkeypatch, target, expected_days):
    when = _ET.localize(datetime.fromisoformat(f"{target} 16:30"))
    earnings = await _earnings(monkeypatch, pg=_page_at(when))
    assert earnings["next_date"] == target
    assert earnings["days_until"] == expected_days  # OLD: one less (wall-clock floor)


async def test_r17_invalid_info_timestamp_marks_info_failed(monkeypatch):
    earnings = await _earnings(monkeypatch, info={"earningsTimestamp": "not-a-number"})
    assert earnings["next_date"] is None
    assert "info" in earnings["sources_failed"]


async def test_info_fetch_raise_marks_info_failed(monkeypatch):
    earnings = await _earnings(monkeypatch, info_raises=True)
    assert earnings["sources_failed"] == ["info"]


async def test_r18_naive_earnings_timestamp_same_day_is_day_zero(monkeypatch):
    # Naive timestamps are ET wall time. Localizing as UTC first would shift a
    # pre-~04:00 ET wall time back one calendar day, so 00:30 on earnings day
    # would resolve to yesterday and be filtered out.
    earnings = await _earnings(monkeypatch, pg=_page_at(datetime(2026, 7, 15, 0, 30)))
    assert earnings["next_date"] == "2026-07-15"  # OLD: None (shifted to 07-14, filtered out)
    assert earnings["days_until"] == 0


async def test_r19_naive_earnings_timestamp_t5_boundary(monkeypatch):
    earnings = await _earnings(monkeypatch, pg=_page_at(datetime(2026, 7, 20, 16, 30)))
    assert earnings["next_date"] == "2026-07-20"
    assert earnings["days_until"] == 5
