"""Deterministic tests for the trade-setup freshness gate."""

from datetime import date, datetime

import pandas as pd
import pytz

from stock_analysis.tools.trade_setup.freshness import (
    build_freshness,
    freshness_blockers,
    most_recent_expected_trading_day,
)

ET = pytz.timezone("America/New_York")


def intraday_df(last_ts: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-03-10T10:00:00-0400", last_ts],
        "open": [100.0, 100.5], "high": [101.0, 101.5],
        "low": [99.0, 100.0], "close": [100.5, 101.0],
        "volume": [1_000.0, 1_100.0],
    })


def daily_df(last_date: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-03-06", last_date],
        "open": [100.0, 101.0], "high": [101.0, 102.0],
        "low": [99.0, 100.0], "close": [100.5, 101.5],
        "volume": [1_000.0, 1_100.0],
    })


class TestRegularSession:
    NOW = ET.localize(datetime(2026, 3, 10, 10, 30))  # Tuesday, regular hours

    def test_fresh_bar_is_not_stale(self) -> None:
        f = build_freshness(
            intraday_df=intraday_df("2026-03-10T10:25:00-0400"),
            daily_df=None, session="regular", now=self.NOW,
        )
        assert f["basis"] == "bar_timestamp"
        assert f["stale"] is False
        assert f["quote_age_seconds"] == 300
        assert freshness_blockers(f) == []

    def test_old_bar_is_stale(self) -> None:
        f = build_freshness(
            intraday_df=intraday_df("2026-03-10T10:00:00-0400"),
            daily_df=None, session="regular", now=self.NOW,
        )
        assert f["stale"] is True
        assert freshness_blockers(f)[0]["id"] == "stale_data"

    def test_missing_probe_is_unverifiable(self) -> None:
        f = build_freshness(intraday_df=None, daily_df=daily_df("2026-03-10"),
                            session="regular", now=self.NOW)
        assert f["basis"] == "unverifiable"
        assert f["stale"] is True
        assert freshness_blockers(f)[0]["id"] == "freshness_unverifiable"

    def test_misclassified_holiday_prior_day_bars_are_stale(self) -> None:
        # Session classifier says "regular" but newest probe bar is a full day old.
        f = build_freshness(
            intraday_df=intraday_df("2026-03-09T15:55:00-0400"),
            daily_df=None, session="regular", now=self.NOW,
        )
        assert f["stale"] is True


class TestOffHours:
    NOW_EVENING = ET.localize(datetime(2026, 3, 10, 18, 30))  # Tuesday after hours
    NOW_PREMARKET = ET.localize(datetime(2026, 3, 10, 8, 0))  # Tuesday pre-market
    NOW_SUNDAY = ET.localize(datetime(2026, 3, 8, 12, 0))     # Sunday

    def test_eod_fresh_daily_bar_not_stale(self) -> None:
        f = build_freshness(intraday_df=None, daily_df=daily_df("2026-03-10"),
                            session="after_hours", now=self.NOW_EVENING)
        assert f["stale"] is False
        assert f["as_of"] == "2026-03-10"
        assert f["quote_age_seconds"] is None

    def test_premarket_expects_prior_trading_day(self) -> None:
        f = build_freshness(intraday_df=None, daily_df=daily_df("2026-03-09"),
                            session="pre_market", now=self.NOW_PREMARKET)
        assert f["stale"] is False

    def test_weekend_expects_friday(self) -> None:
        f = build_freshness(intraday_df=None, daily_df=daily_df("2026-03-06"),
                            session="closed", now=self.NOW_SUNDAY)
        assert f["stale"] is False

    def test_old_daily_bar_is_stale(self) -> None:
        f = build_freshness(intraday_df=None, daily_df=daily_df("2026-03-06"),
                            session="after_hours", now=self.NOW_EVENING)
        assert f["stale"] is True

    def test_no_daily_data_is_unverifiable(self) -> None:
        f = build_freshness(intraday_df=None, daily_df=None,
                            session="closed", now=self.NOW_SUNDAY)
        assert f["basis"] == "unverifiable"


class TestExpectedTradingDay:
    def test_regular_tuesday_is_same_day(self) -> None:
        now = ET.localize(datetime(2026, 3, 10, 10, 30))
        assert most_recent_expected_trading_day(now, "regular") == date(2026, 3, 10)

    def test_premarket_is_prior_weekday(self) -> None:
        now = ET.localize(datetime(2026, 3, 10, 8, 0))
        assert most_recent_expected_trading_day(now, "pre_market") == date(2026, 3, 9)

    def test_monday_premarket_is_friday(self) -> None:
        now = ET.localize(datetime(2026, 3, 9, 8, 0))
        assert most_recent_expected_trading_day(now, "pre_market") == date(2026, 3, 6)
