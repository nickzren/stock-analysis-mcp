"""Tests for market state detection."""

from datetime import datetime
from unittest.mock import patch

import pytest
import pytz

from stock_mcp.data.yfinance_client import get_market_state


class TestMarketState:
    """Tests for market state detection."""

    def test_market_state_method(self) -> None:
        """Test market state includes method."""
        state = get_market_state()
        assert state["method"] == "clock_only_no_holidays"

    def test_market_state_has_checked_at(self) -> None:
        """Test market state includes timestamp."""
        state = get_market_state()
        assert "checked_at" in state

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_closed_weekend_saturday(self, mock_datetime: any) -> None:
        """Test market is closed on Saturday."""
        eastern = pytz.timezone("America/New_York")
        # Saturday at 10 AM
        mock_now = eastern.localize(datetime(2024, 1, 6, 10, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "closed"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_closed_weekend_sunday(self, mock_datetime: any) -> None:
        """Test market is closed on Sunday."""
        eastern = pytz.timezone("America/New_York")
        # Sunday at 2 PM
        mock_now = eastern.localize(datetime(2024, 1, 7, 14, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "closed"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_pre_market(self, mock_datetime: any) -> None:
        """Test pre-market hours (4 AM - 9:30 AM)."""
        eastern = pytz.timezone("America/New_York")
        # Monday at 7 AM
        mock_now = eastern.localize(datetime(2024, 1, 8, 7, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "pre_market"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_regular_hours(self, mock_datetime: any) -> None:
        """Test regular hours (9:30 AM - 4 PM)."""
        eastern = pytz.timezone("America/New_York")
        # Monday at 11 AM
        mock_now = eastern.localize(datetime(2024, 1, 8, 11, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "regular"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_after_hours(self, mock_datetime: any) -> None:
        """Test after hours (4 PM - 8 PM)."""
        eastern = pytz.timezone("America/New_York")
        # Monday at 5 PM
        mock_now = eastern.localize(datetime(2024, 1, 8, 17, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "after_hours"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_closed_night(self, mock_datetime: any) -> None:
        """Test market is closed late night (after 8 PM)."""
        eastern = pytz.timezone("America/New_York")
        # Monday at 9 PM
        mock_now = eastern.localize(datetime(2024, 1, 8, 21, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "closed"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_closed_early_morning(self, mock_datetime: any) -> None:
        """Test market is closed early morning (before 4 AM)."""
        eastern = pytz.timezone("America/New_York")
        # Monday at 3 AM
        mock_now = eastern.localize(datetime(2024, 1, 8, 3, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "closed"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_open_boundary(self, mock_datetime: any) -> None:
        """Test market open boundary (exactly 9:30 AM)."""
        eastern = pytz.timezone("America/New_York")
        # Monday at exactly 9:30 AM
        mock_now = eastern.localize(datetime(2024, 1, 8, 9, 30, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "regular"

    @patch("stock_mcp.data.yfinance_client.datetime")
    def test_market_close_boundary(self, mock_datetime: any) -> None:
        """Test market close boundary (exactly 4 PM)."""
        eastern = pytz.timezone("America/New_York")
        # Monday at exactly 4 PM
        mock_now = eastern.localize(datetime(2024, 1, 8, 16, 0, 0))
        mock_datetime.now.return_value = mock_now

        state = get_market_state()
        assert state["state"] == "after_hours"
