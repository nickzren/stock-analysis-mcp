"""Deterministic tests for the trade-setup freshness gate."""

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
import pytz

from stock_analysis.tools.trade_setup.freshness import (
    FUTURE_SKEW_TOLERANCE_SECONDS,
    build_freshness,
    freshness_blockers,
    most_recent_expected_trading_day,
)
from stock_analysis.utils.market_calendar import previous_trading_day

ET = pytz.timezone("America/New_York")


def intraday_df(last_ts: str, close: float = 101.0) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-03-10T10:00:00-0400", last_ts],
        "open": [100.0, 100.5], "high": [101.0, 101.5],
        "low": [99.0, 100.0], "close": [100.5, close],
        "volume": [1_000.0, 1_100.0],
    })


def daily_df(last_date: str, close: float = 101.5) -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2026-03-06", last_date],
        "open": [100.0, 101.0], "high": [101.0, 102.0],
        "low": [99.0, 100.0], "close": [100.5, close],
        "volume": [1_000.0, 1_100.0],
    })


def current_daily_df(now: datetime, close: float = 101.5) -> pd.DataFrame:
    """Daily frame whose last bar satisfies the regular-session daily leg for `now`."""
    return daily_df(previous_trading_day(now.date()).isoformat(), close=close)


class TestRegularSession:
    NOW = ET.localize(datetime(2026, 3, 10, 10, 30))  # Tuesday, regular hours

    def test_fresh_bar_is_not_stale(self) -> None:
        f = build_freshness(
            intraday_df=intraday_df("2026-03-10T10:25:00-0400"),
            daily_df=current_daily_df(self.NOW), session="regular", now=self.NOW,
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


class TestHolidayAwareWalkBack:
    def test_monday_after_holiday_friday_expects_thursday(self) -> None:
        now = ET.localize(datetime(2026, 7, 6, 8, 0))  # Mon pre_market
        assert most_recent_expected_trading_day(now, "pre_market") == date(2026, 7, 2)

    def test_eod_fresh_across_holiday_weekend(self) -> None:
        # Monday 2026-07-06 pre_market with newest daily bar Thu 07-02:
        # previously stale (expected Fri 07-03); now fresh.
        now = ET.localize(datetime(2026, 7, 6, 8, 0))
        f = build_freshness(intraday_df=None, daily_df=daily_df("2026-07-02"),
                            session="pre_market", now=now)
        assert f["stale"] is False


class TestNanCloseGate:
    def test_valid_timestamp_nan_close_is_unverifiable(self) -> None:
        import numpy as np
        df = intraday_df("2026-03-10T10:25:00-0400")
        df.loc[df.index[-1], "close"] = np.nan
        f = build_freshness(intraday_df=df, daily_df=None,
                            session="regular", now=TestRegularSession.NOW)
        assert f["basis"] == "unverifiable"
        assert f["stale"] is True
        assert freshness_blockers(f)[0]["id"] == "freshness_unverifiable"

    def test_finite_close_path_unchanged(self) -> None:
        f = build_freshness(intraday_df=intraday_df("2026-03-10T10:25:00-0400"),
                            daily_df=current_daily_df(TestRegularSession.NOW),
                            session="regular", now=TestRegularSession.NOW)
        assert f["basis"] == "bar_timestamp" and f["stale"] is False


class TestFreshnessCoherence:
    """Two-leg regular-session gate: fresh intraday AND current daily frame
    (R5), future-skew and non-finite-close guards (R6/R7/R14/R15/R16)."""

    NOW = ET.localize(datetime(2026, 3, 10, 10, 30))          # Tuesday, regular hours
    NOW_EVENING = ET.localize(datetime(2026, 3, 10, 18, 30))  # Tuesday, after hours

    def test_r5_fresh_probe_stale_daily_is_stale(self) -> None:
        fresh_intraday = intraday_df(
            (self.NOW - timedelta(seconds=60)).isoformat(), close=100.0,
        )
        old_daily = daily_df((self.NOW.date() - timedelta(days=7)).isoformat())  # ~5 sessions back
        fr = build_freshness(intraday_df=fresh_intraday, daily_df=old_daily,
                             session="regular", now=self.NOW)
        assert fr["stale"] is True                    # OLD: False (daily never checked)
        assert fr["reason_code"] == "stale_daily"
        assert fr["daily_expected_date"] == previous_trading_day(self.NOW.date()).isoformat()

    def test_r6_future_bar_timestamp_unverifiable(self) -> None:
        df = intraday_df((self.NOW + timedelta(minutes=10)).isoformat(), close=100.0)
        fr = build_freshness(intraday_df=df, daily_df=current_daily_df(self.NOW),
                             session="regular", now=self.NOW)
        assert fr["basis"] == "unverifiable"           # OLD: fresh, age clamped to 0
        assert fr["reason_code"] == "future_timestamp"

    @pytest.mark.parametrize(("skew", "expect_basis"), [
        (FUTURE_SKEW_TOLERANCE_SECONDS, "bar_timestamp"),
        (FUTURE_SKEW_TOLERANCE_SECONDS + 1, "unverifiable"),
    ])
    def test_r14_skew_boundary(self, skew: int, expect_basis: str) -> None:
        df = intraday_df((self.NOW + timedelta(seconds=skew)).isoformat(), close=100.0)
        fr = build_freshness(intraday_df=df, daily_df=current_daily_df(self.NOW),
                             session="regular", now=self.NOW)
        assert fr["basis"] == expect_basis
        if expect_basis == "bar_timestamp":
            assert fr["quote_age_seconds"] == 0

    def test_r7_inf_close_unverifiable(self) -> None:
        df = intraday_df((self.NOW - timedelta(seconds=60)).isoformat(), close=float("inf"))
        fr = build_freshness(intraday_df=df, daily_df=current_daily_df(self.NOW),
                             session="regular", now=self.NOW)
        assert fr["basis"] == "unverifiable"           # OLD: passes (isna-only check)
        assert fr["reason_code"] == "nonfinite_close"

    @pytest.mark.parametrize("bad_close", [float("nan"), float("inf"), float("-inf")])
    def test_r15_nonfinite_daily_close(self, bad_close: float) -> None:
        daily = current_daily_df(self.NOW, close=bad_close)
        fresh_intraday = intraday_df(
            (self.NOW - timedelta(seconds=60)).isoformat(), close=100.0,
        )
        regular = build_freshness(intraday_df=fresh_intraday, daily_df=daily,
                                  session="regular", now=self.NOW)
        assert regular["stale"] is True and regular["reason_code"] == "stale_daily"
        off = build_freshness(intraday_df=None, daily_df=daily,
                              session="closed", now=self.NOW_EVENING)
        assert off["basis"] == "unverifiable" and off["reason_code"] == "daily_unusable"

    def test_r16_future_daily_date(self) -> None:
        daily = daily_df((self.NOW.date() + timedelta(days=1)).isoformat())
        fresh_intraday = intraday_df(
            (self.NOW - timedelta(seconds=60)).isoformat(), close=100.0,
        )
        regular = build_freshness(intraday_df=fresh_intraday, daily_df=daily,
                                  session="regular", now=self.NOW)
        assert regular["stale"] is True
        off = build_freshness(intraday_df=None, daily_df=daily,
                              session="closed", now=self.NOW_EVENING)
        assert off["basis"] == "unverifiable"
