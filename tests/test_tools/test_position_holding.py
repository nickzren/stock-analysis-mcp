"""Boundary tests for the long-term capital-gains holding period.

IRS rule (26 USC §1222): long-term capital gains require the asset to be held
**more than** one year. A position held for exactly 365 days is still
short-term; only at 366+ days does it qualify as long-term.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from stock_analysis.tools.position import _build_holding


def _holding_for_days_ago(days: int) -> dict:
    """Build a holding dict for a position purchased exactly `days` ago."""
    # Use a fixed "now" so the test is deterministic regardless of clock.
    fixed_now = datetime(2026, 5, 12)
    purchase_dt = fixed_now - timedelta(days=days)
    purchase_date = purchase_dt.strftime("%Y-%m-%d")
    with patch("stock_analysis.tools.position.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        return _build_holding(purchase_date=purchase_date, purchase_dt=purchase_dt)


class TestHoldingPeriodBoundary:
    """The 365/366 day boundary for short-term vs long-term capital gains."""

    def test_364_days_is_short_term(self) -> None:
        h = _holding_for_days_ago(364)
        assert h["days_held"] == 364
        assert h["is_long_term"] is False
        assert h["days_to_long_term"] == 2  # 366 - 364 = 2

    def test_365_days_is_short_term(self) -> None:
        """Exactly 365 days is still short-term (must hold >365)."""
        h = _holding_for_days_ago(365)
        assert h["days_held"] == 365
        assert h["is_long_term"] is False
        assert h["days_to_long_term"] == 1

    def test_366_days_is_long_term(self) -> None:
        """At 366 days the holding crosses into long-term."""
        h = _holding_for_days_ago(366)
        assert h["days_held"] == 366
        assert h["is_long_term"] is True
        assert h["days_to_long_term"] is None

    def test_500_days_is_long_term(self) -> None:
        h = _holding_for_days_ago(500)
        assert h["is_long_term"] is True
        assert h["days_to_long_term"] is None
