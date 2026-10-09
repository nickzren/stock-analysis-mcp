"""Step 1 WP2: earnings dates, windows, time zones and verification (R2.1-R2.10).

Drives the public events_calendar tool with provider reads replaced by
FakeSource, so each source's verified state is explicit. Today is 2026-10-08.
"""

from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import pytest
import pytz

import stock_analysis.tools.events as events
from stock_analysis.data.provider import SourceRead
from tests.support.fake_source import FakeSource
from tests.support.fake_ticker import FakeTicker
from tests.test_tools.test_trade_card import blocker_ids, build_card

ET = pytz.timezone("America/New_York")
NOW = ET.localize(datetime(2026, 10, 8, 12, 0))


def epoch(day: str, hour: int = 16) -> int:
    return int(ET.localize(datetime.fromisoformat(day).replace(hour=hour)).timestamp())


def calendar(*days: str, is_estimate: bool | None = None, **extra: Any) -> FakeSource:
    earnings: dict[str, Any] = {"earningsDate": [epoch(d) for d in days]}
    if is_estimate is not None:
        earnings["isEarningsDateEstimate"] = is_estimate
    events_payload = {"earnings": earnings, **extra} if days else {**extra}
    return FakeSource(SourceRead("calendar", "ok" if days or extra else "empty", payload=events_payload))


def page(*days: str, reported: dict[str, float] | None = None) -> FakeSource:
    reported = reported or {}
    rows = [
        {
            "timestamp": ET.localize(datetime.fromisoformat(d).replace(hour=16)),
            "eps_estimate": 1.0,
            "reported_eps": reported.get(d),
            "surprise_pct": None,
        }
        for d in days
    ]
    return FakeSource(SourceRead("earnings_page", "ok" if rows else "empty", payload=rows))


def window_info(start: str, end: str, *, is_estimate: bool | None = True) -> dict[str, Any]:
    info: dict[str, Any] = {"earningsTimestampStart": epoch(start), "earningsTimestampEnd": epoch(end)}
    if is_estimate is not None:
        info["isEarningsDateEstimate"] = is_estimate
    return info


FAILED_CALENDAR = FakeSource.failed("calendar")
FAILED_PAGE = FakeSource.failed("earnings_page")
UNVERIFIED_PAGE = FakeSource.unverified("earnings_page")
NO_PAGE_DATES = page("2026-07-29", reported={"2026-07-29": 1.1})
STALE_CALENDAR = calendar("2026-07-01")
NO_CALENDAR_DATES = calendar()


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    async def run(
        cal: FakeSource,
        pg: FakeSource,
        info: dict[str, Any] | None = None,
        *,
        info_fails: bool = False,
        now: datetime = NOW,
    ) -> dict[str, Any]:
        async def fake_fetch_ticker(symbol: str) -> FakeTicker:
            return FakeTicker(symbol, splits=None)

        async def fake_fetch_info(symbol: str) -> dict[str, Any]:
            if info_fails:
                raise ConnectionError("info down")
            return info or {}

        monkeypatch.setattr(events, "fetch_ticker", fake_fetch_ticker)
        monkeypatch.setattr(events, "fetch_info", fake_fetch_info)
        monkeypatch.setattr(events, "read_calendar_events", cal)
        monkeypatch.setattr(events, "read_earnings_page", pg)
        return await events.events_calendar("EXMP", _now=now)

    return run


class TestTruthTable:
    """R2.7: every truth-table row."""

    @pytest.mark.parametrize(
        ("row", "cal", "pg", "info", "info_fails", "verification", "next_date"),
        [
            ("1 point", calendar("2026-10-20"), NO_PAGE_DATES, {}, False, "verified_date", "2026-10-20"),
            ("1 confirmed window", calendar("2026-10-20", "2026-10-24", is_estimate=True), NO_PAGE_DATES,
             window_info("2026-10-20", "2026-10-24"), False, "verified_date", "2026-10-20"),
            ("1b assumed future window", calendar("2026-10-20", "2026-10-24"), NO_PAGE_DATES, {}, False,
             "unverifiable", "2026-10-20"),
            ("2a confirmed active window", calendar("2026-10-07", "2026-10-15", is_estimate=True), NO_PAGE_DATES,
             window_info("2026-10-07", "2026-10-15"), False, "verified_date", "2026-10-08"),
            ("2b assumed active window", calendar("2026-10-07", "2026-10-15"), NO_PAGE_DATES, {}, False,
             "unverifiable", "2026-10-08"),
            ("3", FAILED_CALENDAR, page("2026-10-20"), {}, False, "verified_date", "2026-10-20"),
            ("4", FAILED_CALENDAR, NO_PAGE_DATES, {}, False, "unverifiable", None),
            ("5", STALE_CALENDAR, page("2026-10-09"), {}, False, "verified_date", "2026-10-09"),
            ("6", STALE_CALENDAR, NO_PAGE_DATES, {}, False, "unverifiable", None),
            ("7", NO_CALENDAR_DATES, NO_PAGE_DATES, {}, False, "verified_none", None),
            ("8", NO_CALENDAR_DATES, UNVERIFIED_PAGE, {}, False, "unverifiable", None),
            ("9", calendar("2026-10-22"), page("2026-10-20"), {}, False, "verified_date", "2026-10-20"),
            ("10", FAILED_CALENDAR, FAILED_PAGE, {}, True, "unverifiable", None),
        ],
    )
    async def test_row(
        self,
        run: Callable[..., Any],
        row: str,
        cal: FakeSource,
        pg: FakeSource,
        info: dict[str, Any],
        info_fails: bool,
        verification: str,
        next_date: str | None,
    ) -> None:
        earnings = (await run(cal, pg, info, info_fails=info_fails))["earnings"]

        assert (earnings["verification"], earnings["next_date"]) == (verification, next_date)

    async def test_row_9_discloses_the_disagreement(self, run: Callable[..., Any]) -> None:
        earnings = (await run(calendar("2026-10-22"), page("2026-10-20")))["earnings"]

        assert earnings["candidate_dates"] == {"calendar": "2026-10-22", "earnings_dates": "2026-10-20"}

    @pytest.mark.parametrize(
        ("verification", "action", "unverifiable"),
        [("verified_date", "trade_now", False), ("verified_none", "trade_now", False),
         ("unverifiable", "watch", True)],
    )
    def test_card_caps_only_unverifiable(self, verification: str, action: str, unverifiable: bool) -> None:
        earnings = {"verification": verification, "next_date": None, "days_until": None, "sources_failed": []}

        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert card["action"] == action
        assert ("earnings_unverifiable" in blocker_ids(card)) is unverifiable


class TestStaleCalendar:
    async def test_r2_1_stale_calendar_falls_through_to_the_page(self, run: Callable[..., Any]) -> None:
        earnings = (await run(STALE_CALENDAR, page("2026-10-09")))["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert (earnings["next_date"], earnings["days_until"]) == ("2026-10-09", 1)
        assert earnings["sources_stale"] == ["calendar"]
        assert "earnings_blackout" in blocker_ids(card)

    async def test_r2_2_stale_calendar_alone_is_unverifiable(self, run: Callable[..., Any]) -> None:
        earnings = (await run(STALE_CALENDAR, NO_PAGE_DATES))["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert (earnings["verification"], earnings["days_until"]) == ("unverifiable", None)
        assert card["action"] == "watch"
        assert "earnings_unverifiable" in blocker_ids(card)


class TestWindows:
    @pytest.mark.parametrize(
        ("cal", "info", "verification", "basis"),
        [
            (calendar("2026-10-07", "2026-10-15", is_estimate=True), window_info("2026-10-07", "2026-10-15"),
             "verified_date", "confirmed"),
            (calendar("2026-10-07", "2026-10-15"), {}, "unverifiable", "assumed"),
        ],
    )
    async def test_r2_4_window_covering_today_is_active(
        self, run: Callable[..., Any], cal: FakeSource, info: dict[str, Any], verification: str, basis: str
    ) -> None:
        earnings = (await run(cal, NO_PAGE_DATES, info))["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["days_until"] == 0  # not "Oct 15, 7 days"
        assert earnings["next_window"]["window_active"] is True
        assert (earnings["verification"], earnings["next_window"]["window_basis"]) == (verification, basis)
        assert card["action"] == "watch"
        assert "earnings_blackout" in blocker_ids(card)

    async def test_r2_5_future_window_counts_to_its_start(self, run: Callable[..., Any]) -> None:
        earnings = (await run(calendar("2026-10-20", "2026-10-24"), NO_PAGE_DATES))["earnings"]

        assert (earnings["next_date"], earnings["days_until"]) == ("2026-10-20", 12)
        assert (earnings["next_window"]["start"], earnings["next_window"]["end"]) == ("2026-10-20", "2026-10-24")
        assert earnings["next_window"]["window_active"] is False

    @pytest.mark.parametrize(
        "cal",
        [calendar("2026-10-24", "2026-10-20"), calendar("2026-10-20", "2026-10-22", "2026-10-24")],
        ids=["reversed", "three_entries"],
    )
    async def test_r2_5_malformed_window_fails_the_calendar(self, run: Callable[..., Any], cal: FakeSource) -> None:
        earnings = (await run(cal, NO_PAGE_DATES))["earnings"]

        assert earnings["sources_failed"] == ["calendar"]
        assert earnings["verification"] == "unverifiable"

    @pytest.mark.parametrize(
        ("cal", "info", "basis", "conflict"),
        [
            (calendar("2026-10-20", "2026-10-24", is_estimate=True), window_info("2026-10-20", "2026-10-24"),
             "confirmed", False),
            (calendar("2026-10-20", "2026-10-24"), {}, "assumed", False),
            (calendar("2026-10-20", "2026-10-24", is_estimate=True), {}, "assumed", False),
            (calendar("2026-10-20", "2026-10-24", is_estimate=False), {}, "assumed", True),
            (calendar("2026-10-20", "2026-10-24", is_estimate=True), window_info("2026-10-21", "2026-10-24"),
             "assumed", True),
        ],
        ids=["consistent", "absent", "range_absent", "contradicts_estimate_flag", "contradicts_range"],
    )
    async def test_r2_10_window_basis(
        self, run: Callable[..., Any], cal: FakeSource, info: dict[str, Any], basis: str, conflict: bool
    ) -> None:
        earnings = (await run(cal, NO_PAGE_DATES, info))["earnings"]
        window = earnings["next_window"]

        assert (window["window_basis"], window["metadata_conflict"]) == (basis, conflict)
        assert (earnings["verification"] == "verified_date") is (basis == "confirmed")

    async def test_r2_10_independent_date_resolves_an_assumed_window(self, run: Callable[..., Any]) -> None:
        earnings = (await run(calendar("2026-10-07", "2026-10-15"), page("2026-10-14")))["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["verification"] == "verified_date"
        assert earnings["next_window"]["window_basis"] == "assumed"  # still disclosed
        assert earnings["days_until"] == 0  # the active window still drives the blackout
        assert "earnings_unverifiable" not in blocker_ids(card)
        assert "earnings_blackout" in blocker_ids(card)


class TestTimeZones:
    @pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
    async def test_r2_6_calendar_epochs_convert_in_et(
        self, run: Callable[..., Any], host_tz: Callable[[str], None], tz: str
    ) -> None:
        host_tz(tz)
        # 2026-10-09T00:30Z is 2026-10-08 20:30 ET.
        payload = {"earnings": {"earningsDate": [1791505800]}}
        cal = FakeSource(SourceRead("calendar", "ok", payload=payload))

        earnings = (await run(cal, NO_PAGE_DATES, now=ET.localize(datetime(2026, 10, 6, 12, 0))))["earnings"]

        assert earnings["next_date"] == "2026-10-08"

    @pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
    async def test_dividend_epochs_are_utc_dates(
        self, run: Callable[..., Any], host_tz: Callable[[str], None], tz: str
    ) -> None:
        """Yahoo sends dividend dates as UTC midnight (KO 1789430400 = 2026-09-15)."""
        host_tz(tz)
        cal = calendar(exDividendDate=1789430400, dividendDate=1790726400)

        dividends = (await run(cal, NO_PAGE_DATES))["dividends"]

        assert (dividends["ex_date"], dividends["pay_date"]) == ("2026-09-15", "2026-09-30")

    async def test_r2_9_same_day_earnings_is_day_zero(self, run: Callable[..., Any]) -> None:
        earnings = (await run(calendar("2026-10-08"), NO_PAGE_DATES))["earnings"]

        assert (earnings["next_date"], earnings["days_until"]) == ("2026-10-08", 0)

    async def test_r2_9_naive_page_timestamps_are_et(self, run: Callable[..., Any]) -> None:
        rows = [{"timestamp": datetime(2026, 10, 8, 23, 0), "eps_estimate": 1.0, "reported_eps": None,
                 "surprise_pct": None}]
        pg = FakeSource(SourceRead("earnings_page", "ok", payload=rows))

        earnings = (await run(NO_CALENDAR_DATES, pg))["earnings"]

        assert (earnings["next_date"], earnings["days_until"]) == ("2026-10-08", 0)


class TestHistory:
    async def test_r2_8_history_has_only_reported_rows_newest_first(self, run: Callable[..., Any]) -> None:
        pg = page(
            "2027-01-27", "2026-10-28", "2026-04-29", "2026-07-29",
            reported={"2026-07-29": 1.2, "2026-04-29": 0.9},
        )

        earnings = (await run(NO_CALENDAR_DATES, pg))["earnings"]

        assert [h["date"] for h in earnings["history"]] == ["2026-07-29", "2026-04-29"]
        assert earnings["beat_rate"] == 0.5


class TestInfo:
    async def test_info_timestamp_is_a_source(self, run: Callable[..., Any]) -> None:
        info = {"earningsTimestamp": epoch("2026-10-20"), "isEarningsDateEstimate": True}

        earnings = (await run(NO_CALENDAR_DATES, NO_PAGE_DATES, info))["earnings"]

        assert (earnings["next_date"], earnings["next_date_source"]) == ("2026-10-20", "info_timestamp")
        assert earnings["next_date_is_estimate"] is True

    async def test_invalid_info_timestamp_marks_info_failed(self, run: Callable[..., Any]) -> None:
        earnings = (await run(NO_CALENDAR_DATES, NO_PAGE_DATES, {"earningsTimestamp": "soon"}))["earnings"]

        assert earnings["sources_failed"] == ["info"]
        assert earnings["verification"] == "unverifiable"


def test_today_is_the_fixture_date() -> None:
    assert NOW.date() == date(2026, 10, 8)
