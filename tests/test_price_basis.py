"""WP5: price selection, provenance and staleness (R5.1-R5.3)."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
import pytz

from stock_analysis.utils.market_calendar import last_completed_session, trading_days_between
from stock_analysis.utils.price_basis import (
    latest_intraday_bar,
    latest_session_close,
    quote_staleness,
    select_price,
    session_staleness,
    to_utc,
)

ET = pytz.timezone("America/New_York")


def at(y: int, m: int, d: int, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return ET.localize(datetime(y, m, d, hh, mm, ss))


def daily(dates: list[str], closes: list[float]) -> pd.DataFrame:
    df = pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes,
                       "close": closes, "volume": [1.0] * len(dates)})
    df.attrs["fetched_at"] = "2026-10-09T20:00:00Z"
    return df


class TestCalendar:
    @pytest.mark.parametrize(
        ("now", "expected"),
        [
            (at(2026, 10, 12, 8), date(2026, 10, 9)),     # Monday pre-market -> Friday
            (at(2026, 10, 12, 11), date(2026, 10, 9)),    # Monday regular: today still forming
            (at(2026, 10, 12, 16), date(2026, 10, 12)),   # Monday at the close
            (at(2026, 10, 10, 12), date(2026, 10, 9)),    # Saturday
            (at(2026, 11, 26, 12), date(2026, 11, 25)),   # Thanksgiving
            (at(2026, 11, 27, 12, 59), date(2026, 11, 25)),  # early close, still open
            (at(2026, 11, 27, 13), date(2026, 11, 27)),   # early close, closed
        ],
    )
    def test_last_completed_session(self, now: datetime, expected: date) -> None:
        assert last_completed_session(now) == expected

    def test_trading_days_between_skips_weekends_and_holidays(self) -> None:
        assert trading_days_between(date(2026, 10, 9), date(2026, 10, 13)) == 2
        assert trading_days_between(date(2026, 11, 25), date(2026, 11, 30)) == 2  # Thu 26 closed
        assert trading_days_between(date(2026, 10, 13), date(2026, 10, 9)) == 0


class TestR53SessionStaleness:
    def test_friday_close_on_monday_before_the_open_is_current(self) -> None:
        s = session_staleness(date(2026, 10, 9), at(2026, 10, 12, 8))
        assert s == {"unit": "sessions", "expected_session": "2026-10-09",
                     "sessions_behind": 0, "stale": False}

    def test_friday_close_on_tuesday_before_the_open_is_one_session_stale(self) -> None:
        s = session_staleness(date(2026, 10, 9), at(2026, 10, 13, 8))
        assert (s["sessions_behind"], s["stale"], s["expected_session"]) == (1, True, "2026-10-12")

    def test_holidays_come_from_the_calendar(self) -> None:
        assert session_staleness(date(2026, 11, 25), at(2026, 11, 27, 8))["sessions_behind"] == 0
        assert session_staleness(date(2026, 11, 25), at(2026, 11, 30, 8))["sessions_behind"] == 1

    def test_utc_now_is_read_in_market_time(self) -> None:
        # 17:00 UTC is 13:00 in New York: Monday's session has not closed yet.
        s = session_staleness(date(2026, 10, 9), datetime(2026, 10, 12, 17, tzinfo=UTC))
        assert (s["expected_session"], s["sessions_behind"]) == ("2026-10-09", 0)


class TestR52UtcAges:
    def test_equivalent_instants_parse_identically(self) -> None:
        assert to_utc("2026-10-09T10:00:00-04:00") == to_utc("2026-10-09T14:00:00Z")
        assert to_utc("2026-10-09T10:00:00-0400") == to_utc(1791554400)
        assert to_utc("2026-10-09T14:00:00") == to_utc("2026-10-09T14:00:00+00:00")  # naive = UTC

    @pytest.mark.parametrize("value", [None, "", "not a date", True, float("nan"), [1]])
    def test_unparseable_is_none(self, value: object) -> None:
        assert to_utc(value) is None

    def test_quote_ages_match_across_offsets(self) -> None:
        now = at(2026, 10, 9, 11)  # 15:00 UTC
        et = quote_staleness(to_utc("2026-10-09T10:00:00-04:00"), now)  # type: ignore[arg-type]
        utc = quote_staleness(to_utc("2026-10-09T14:00:00Z"), now)  # type: ignore[arg-type]
        assert et == utc
        assert et["age_hours"] == 1.0


class TestQuoteStaleness:
    def test_regular_session_uses_the_15_minute_ceiling(self) -> None:
        now = at(2026, 10, 9, 11)
        assert quote_staleness(now.astimezone(UTC) - timedelta(minutes=15), now)["stale"] is False
        assert quote_staleness(now.astimezone(UTC) - timedelta(minutes=15, seconds=1), now)["stale"] is True

    def test_off_hours_compares_the_quote_session_with_the_last_completed_one(self) -> None:
        friday_close = at(2026, 10, 9, 16).astimezone(UTC)
        assert quote_staleness(friday_close, at(2026, 10, 12, 8))["stale"] is False
        assert quote_staleness(friday_close, at(2026, 10, 13, 8))["stale"] is True
        # Monday 20:30 in New York is already Tuesday in UTC; Tuesday has since closed.
        monday_evening = at(2026, 10, 12, 20, 30).astimezone(UTC)
        assert quote_staleness(monday_evening, at(2026, 10, 13, 17))["stale"] is True

    def test_off_hours_a_quote_from_before_the_close_is_stale(self) -> None:
        mid_session = at(2026, 10, 12, 14, 30).astimezone(UTC)
        assert quote_staleness(mid_session, at(2026, 10, 12, 18))["stale"] is True
        assert quote_staleness(mid_session, at(2026, 10, 13, 8))["stale"] is True
        early_close_day = at(2026, 11, 27, 11).astimezone(UTC)
        assert quote_staleness(early_close_day, at(2026, 11, 27, 14))["stale"] is True
        assert quote_staleness(at(2026, 11, 27, 13).astimezone(UTC), at(2026, 11, 27, 14))["stale"] is False

    def test_fetching_an_old_quote_after_the_close_never_makes_it_current(self) -> None:
        now = at(2026, 10, 13, 18)
        info = {"price": 100.0, "quote_time": (now.astimezone(UTC) - timedelta(hours=31)).isoformat(),
                "fetched_at": at(2026, 10, 13, 17).astimezone(UTC).isoformat(), "cache_hit": True}
        _, prov = select_price(now=now, info_quote=info)
        assert (prov["staleness"]["age_hours"], prov["staleness"]["stale"]) == (31.0, True)
        assert prov["fetched_at"] == info["fetched_at"]  # still disclosed

    def test_future_timestamp_beyond_skew_is_stale(self) -> None:
        now = at(2026, 10, 9, 11)
        s = quote_staleness(now.astimezone(UTC) + timedelta(seconds=6), now)
        assert (s["stale"], s["reason"]) == (True, "future_timestamp")


class TestLatestSessionClose:
    def test_forming_bar_is_skipped_during_the_session(self) -> None:
        df = daily(["2026-10-08", "2026-10-09"], [80.0, 85.0])
        got = latest_session_close(df, at(2026, 10, 9, 11))
        assert got == {"close": 80.0, "session_date": "2026-10-08",
                       "fetched_at": "2026-10-09T20:00:00Z", "cache_hit": False}

    def test_completed_session_is_used_after_the_close(self) -> None:
        df = daily(["2026-10-08", "2026-10-09"], [80.0, 85.0])
        assert latest_session_close(df, at(2026, 10, 9, 16, 5))["session_date"] == "2026-10-09"  # type: ignore[index]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
    def test_latest_valid_close_wins(self, bad: float) -> None:
        df = daily(["2026-10-07", "2026-10-08"], [80.0, bad])
        got = latest_session_close(df, at(2026, 10, 9, 8))
        assert got is not None
        assert (got["close"], got["session_date"]) == (80.0, "2026-10-07")

    def test_no_valid_close_is_none(self) -> None:
        assert latest_session_close(daily(["2026-10-08"], [float("nan")]), at(2026, 10, 9, 8)) is None
        assert latest_session_close(None, at(2026, 10, 9, 8)) is None


class TestLatestIntradayBar:
    def test_reads_the_latest_valid_bar_in_utc(self) -> None:
        df = pd.DataFrame({"date": ["2026-10-09T10:55:00-0400", "2026-10-09T11:00:00-0400"],
                           "close": [81.0, float("nan")]})
        df.attrs["fetched_at"] = "2026-10-09T15:01:00Z"
        assert latest_intraday_bar(df) == {"close": 81.0, "bar_time": "2026-10-09T14:55:00+00:00",
                                           "fetched_at": "2026-10-09T15:01:00Z", "cache_hit": False}

    def test_nothing_valid_is_none(self) -> None:
        assert latest_intraday_bar(pd.DataFrame({"date": ["x"], "close": [1.0]})) is None
        assert latest_intraday_bar(None) is None


SESSION_CLOSE = {"close": 80.0, "session_date": "2026-10-08", "fetched_at": "2026-10-09T15:00:00Z",
                 "cache_hit": False}
CACHED_INFO = {"price": 100.0, "quote_time": "2026-10-09T11:00:00Z", "fetched_at": "2026-10-09T11:00:05Z",
               "cache_hit": True}


def intraday(now: datetime, *, seconds_old: float, close: float = 82.0) -> dict[str, object]:
    bar_time = (now.astimezone(UTC) - timedelta(seconds=seconds_old)).isoformat()
    return {"close": close, "bar_time": bar_time, "fetched_at": now.astimezone(UTC).isoformat(),
            "cache_hit": False}


class TestR51SelectPrice:
    def test_cached_info_never_beats_the_daily_close(self) -> None:
        price, prov = select_price(now=at(2026, 10, 9, 8), session_close=SESSION_CLOSE, info_quote=CACHED_INFO)
        assert price == 80.0
        assert prov == {
            "price_source": "daily_close", "price_as_of": "2026-10-08", "as_of_kind": "session_close",
            "as_of_precision": "date", "fetched_at": "2026-10-09T15:00:00Z", "cache_hit": False,
            "fallback_reason": None,
            "staleness": {"unit": "sessions", "expected_session": "2026-10-08",
                          "sessions_behind": 0, "stale": False},
        }

    def test_fresh_intraday_bar_wins_in_the_regular_session(self) -> None:
        now = at(2026, 10, 9, 11)
        price, prov = select_price(now=now, intraday_quote=intraday(now, seconds_old=300),
                                   session_close=SESSION_CLOSE, info_quote=CACHED_INFO)
        assert price == 82.0
        assert (prov["price_source"], prov["as_of_kind"], prov["as_of_precision"]) == (
            "intraday_quote", "bar_time", "second")
        assert prov["price_as_of"] == "2026-10-09T14:55:00+00:00"
        assert prov["staleness"] == {"unit": "hours", "age_hours": 0.08, "stale": False}

    @pytest.mark.parametrize(
        ("seconds_old", "used"),
        [(15 * 60, True), (15 * 60 + 1, False), (-5, True), (-6, False)],
        ids=["at_ceiling", "past_ceiling", "skew_ok", "future"],
    )
    def test_freshness_is_measured_from_the_bar_time(self, seconds_old: float, used: bool) -> None:
        now = at(2026, 10, 9, 11)
        _, prov = select_price(now=now, intraday_quote=intraday(now, seconds_old=seconds_old),
                               session_close=SESSION_CLOSE)
        assert (prov["price_source"] == "intraday_quote") is used

    @pytest.mark.parametrize(
        ("quote", "reason"),
        [(None, "intraday_quote_unavailable"), ("stale", "intraday_quote_stale"),
         ("future", "intraday_future_timestamp")],
    )
    def test_a_rejected_intraday_bar_is_disclosed(self, quote: str | None, reason: str) -> None:
        now = at(2026, 10, 9, 11)
        bar = {"stale": intraday(now, seconds_old=16 * 60), "future": intraday(now, seconds_old=-60),
               None: None}[quote]
        _, prov = select_price(now=now, intraday_quote=bar, session_close=SESSION_CLOSE)
        assert (prov["price_source"], prov["fallback_reason"]) == ("daily_close", reason)

    def test_a_bar_newer_than_the_start_clock_is_judged_by_its_fetch_time(self) -> None:
        now = at(2026, 10, 9, 10, 59, 54)  # analyze started before the 11:00 bar existed
        bar = {"close": 82.0, "bar_time": "2026-10-09T15:00:00+00:00", "fetched_at": "2026-10-09T15:00:03Z",
               "cache_hit": False}
        price, prov = select_price(now=now, intraday_quote=bar, session_close=SESSION_CLOSE)
        assert (price, prov["price_source"], prov["staleness"]["stale"]) == (82.0, "intraday_quote", False)

    def test_outside_regular_hours_the_completed_close_is_preferred(self) -> None:
        now = at(2026, 10, 9, 16, 5)
        _, prov = select_price(now=now, intraday_quote=intraday(now, seconds_old=60), session_close=SESSION_CLOSE)
        assert prov["price_source"] == "daily_close"

    def test_info_is_a_labelled_fallback(self) -> None:
        price, prov = select_price(now=at(2026, 10, 9, 8), info_quote=CACHED_INFO)
        assert price == 100.0
        assert prov["price_source"] == "info_quote_fallback"
        assert prov["fallback_reason"] == "no_valid_daily_close"
        assert (prov["price_as_of"], prov["as_of_kind"]) == ("2026-10-09T11:00:00+00:00", "quote_time")
        assert (prov["fetched_at"], prov["cache_hit"]) == ("2026-10-09T11:00:05Z", True)

    def test_fallback_quote_newer_than_the_start_clock_is_judged_by_its_fetch_time(self) -> None:
        now = datetime(2026, 10, 9, 15, 0, 0, tzinfo=UTC)  # 11:00 ET, regular session
        info = {"price": 100.0, "quote_time": "2026-10-09T15:00:06Z", "fetched_at": "2026-10-09T15:00:07Z",
                "cache_hit": False}
        _, prov = select_price(now=now, info_quote=info)
        assert prov["staleness"] == {"unit": "hours", "age_hours": 0.0, "stale": False}

    def test_fallback_without_a_quote_time_is_unverifiable(self) -> None:
        _, prov = select_price(now=at(2026, 10, 9, 8), info_quote={**CACHED_INFO, "quote_time": None})
        assert (prov["price_as_of"], prov["as_of_kind"]) == (None, None)
        assert prov["staleness"] == {"unit": "hours", "age_hours": None, "stale": True, "reason": "no_quote_time"}

    def test_no_price_at_all(self) -> None:
        price, prov = select_price(now=at(2026, 10, 9, 8), info_quote={"price": None})
        assert price is None
        assert (prov["price_source"], prov["fallback_reason"]) == (None, "no_price")
