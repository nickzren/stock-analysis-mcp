"""Tests for market state detection."""

from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest
import pytz

from stock_analysis.data.yfinance_client import get_market_state


class TestMarketState:
    """Tests for market state detection."""

    def test_market_state_method(self) -> None:
        """Test market state includes a recognized method."""
        state = get_market_state()
        assert state["method"] in (
            "calendar_static", "clock_only_no_holidays_fallback",
        )

    def test_market_state_has_checked_at(self) -> None:
        """Test market state includes timestamp."""
        state = get_market_state()
        assert "checked_at" in state

    @pytest.mark.parametrize(
        ("mock_now", "expected_state"),
        [
            (datetime(2024, 1, 6, 10, 0, 0), "closed"),
            (datetime(2024, 1, 7, 14, 0, 0), "closed"),
            (datetime(2024, 1, 8, 7, 0, 0), "pre_market"),
            (datetime(2024, 1, 8, 11, 0, 0), "regular"),
            (datetime(2024, 1, 8, 17, 0, 0), "after_hours"),
            (datetime(2024, 1, 8, 21, 0, 0), "closed"),
            (datetime(2024, 1, 8, 3, 0, 0), "closed"),
            (datetime(2024, 1, 8, 9, 30, 0), "regular"),
            (datetime(2024, 1, 8, 16, 0, 0), "after_hours"),
        ],
        ids=[
            "saturday_closed",
            "sunday_closed",
            "pre_market",
            "regular_hours",
            "after_hours",
            "night_closed",
            "early_morning_closed",
            "open_boundary",
            "close_boundary",
        ],
    )
    @patch("stock_analysis.data.yfinance_client.datetime")
    def test_market_state_by_clock(
        self,
        mock_datetime: Any,
        mock_now: datetime,
        expected_state: str,
    ) -> None:
        """Test market state from weekday and clock time."""
        eastern = pytz.timezone("America/New_York")
        mock_datetime.now.return_value = eastern.localize(mock_now)

        state = get_market_state()
        assert state["state"] == expected_state


class TestCalendarAwareSession:
    def test_full_holiday_is_closed(self) -> None:
        from datetime import datetime

        import pytz

        from stock_analysis.data.cache_manager import classify_session
        et = pytz.timezone("America/New_York")
        assert classify_session(et.localize(datetime(2026, 7, 3, 11, 30))) == "closed"

    def test_get_market_state_reports_method(self) -> None:
        from stock_analysis.data.yfinance_client import get_market_state
        state = get_market_state()
        assert state["method"] in (
            "calendar_static", "clock_only_no_holidays_fallback",
        )
