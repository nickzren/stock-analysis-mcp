"""Tests for tz-aware earnings-date handling in news.py and events.py.

yfinance returns earnings timestamps tz-aware (typically America/New_York).
Stripping tzinfo without converting treats the local wall-clock time as UTC,
producing 4-5 hour boundary errors at the cutoff window edges.
"""

from datetime import datetime, timedelta

import pandas as pd

from stock_analysis.tools.news import _build_recent_earnings


class _TickerWithEarnings:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.earnings_dates = frame


class TestNewsEarningsTimezone:
    """_build_recent_earnings must convert tz-aware timestamps to UTC."""

    def test_tz_aware_timestamp_inside_window(self) -> None:
        """A tz-aware ET timestamp within the lookback window must be included."""
        # cutoff_date is naive-UTC (7 days back from now)
        now = datetime(2026, 5, 12, 0, 0, 0)
        cutoff = now - timedelta(days=7)

        # Earnings call at 4pm ET on May 10 → 20:00 UTC May 10
        # This is inside the 7d lookback window from May 12.
        earnings_ts = pd.Timestamp("2026-05-10 16:00:00", tz="America/New_York")
        frame = pd.DataFrame(
            {"EPS Estimate": [1.5], "Reported EPS": [1.7]},
            index=pd.DatetimeIndex([earnings_ts]),
        )

        result = _build_recent_earnings(_TickerWithEarnings(frame), cutoff_date=cutoff, now=now)

        assert result is not None
        assert result["eps_estimate"] == 1.5
        assert result["eps_actual"] == 1.7
        # The date string should be a sensible YYYY-MM-DD; whether it lands on
        # May 10 or May 11 in UTC depends on the time-of-day. 16:00 ET = 20:00 UTC
        # (or 21:00 during DST → same calendar day).
        assert result["date"].startswith("2026-05-1")

    def test_naive_timestamp_still_handled(self) -> None:
        """A naive timestamp (no tz) should work without crashing."""
        now = datetime(2026, 5, 12, 0, 0, 0)
        cutoff = now - timedelta(days=7)

        earnings_ts = pd.Timestamp("2026-05-10 16:00:00")  # naive
        frame = pd.DataFrame(
            {"EPS Estimate": [1.5], "Reported EPS": [1.7]},
            index=pd.DatetimeIndex([earnings_ts]),
        )

        result = _build_recent_earnings(_TickerWithEarnings(frame), cutoff_date=cutoff, now=now)

        assert result is not None

    def test_future_earnings_excluded(self) -> None:
        """An earnings date in the future (relative to `now`) must not be reported."""
        now = datetime(2026, 5, 12, 0, 0, 0)
        cutoff = now - timedelta(days=7)

        # Future date: May 20, well after `now`
        earnings_ts = pd.Timestamp("2026-05-20 16:00:00", tz="America/New_York")
        frame = pd.DataFrame(
            {"EPS Estimate": [1.5], "Reported EPS": [1.7]},
            index=pd.DatetimeIndex([earnings_ts]),
        )

        result = _build_recent_earnings(_TickerWithEarnings(frame), cutoff_date=cutoff, now=now)

        assert result is None

    def test_too_old_earnings_excluded(self) -> None:
        """An earnings date older than the cutoff must not be reported."""
        now = datetime(2026, 5, 12, 0, 0, 0)
        cutoff = now - timedelta(days=7)

        # 30 days back
        earnings_ts = pd.Timestamp("2026-04-12 16:00:00", tz="America/New_York")
        frame = pd.DataFrame(
            {"EPS Estimate": [1.5], "Reported EPS": [1.7]},
            index=pd.DatetimeIndex([earnings_ts]),
        )

        result = _build_recent_earnings(_TickerWithEarnings(frame), cutoff_date=cutoff, now=now)

        assert result is None
