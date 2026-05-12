"""Tests for _analyze_dividend_history streak/CAGR partial-year handling."""

from unittest.mock import patch

import pandas as pd

from stock_analysis.tools.fundamentals import _analyze_dividend_history


class _FakeTicker:
    """Minimal ticker stub exposing `.dividends` as a pandas Series."""

    def __init__(self, dividends: pd.Series) -> None:
        self.dividends = dividends


def _build_dividends(yearly: dict[int, list[tuple[int, float]]]) -> pd.Series:
    """Build a dividends Series from {year: [(month, amount), ...]}."""
    rows: list[tuple[pd.Timestamp, float]] = []
    for year, payments in yearly.items():
        for month, amount in payments:
            rows.append((pd.Timestamp(year=year, month=month, day=15), amount))
    if not rows:
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    index = pd.DatetimeIndex([r[0] for r in rows])
    values = [r[1] for r in rows]
    return pd.Series(values, index=index)


class TestPartialYearHandling:
    """Dividend streak/CAGR must ignore the current (in-progress) calendar year."""

    @patch("stock_analysis.tools.fundamentals.datetime")
    def test_streak_ignores_partial_current_year(self, mock_datetime) -> None:
        """A mid-year partial total must not falsely break a multi-year streak."""
        mock_datetime.now.return_value = pd.Timestamp("2026-05-15").to_pydatetime()

        # 2022: $4.00, 2023: $4.50, 2024: $5.00, 2025: $5.50 (4 complete years rising),
        # 2026: $1.50 partial (Q1 only)
        dividends = _build_dividends({
            2022: [(3, 1.0), (6, 1.0), (9, 1.0), (12, 1.0)],
            2023: [(3, 1.1), (6, 1.1), (9, 1.1), (12, 1.2)],
            2024: [(3, 1.2), (6, 1.2), (9, 1.3), (12, 1.3)],
            2025: [(3, 1.3), (6, 1.4), (9, 1.4), (12, 1.4)],
            2026: [(3, 1.5)],  # partial — Q1 only
        })

        result = _analyze_dividend_history(
            _FakeTicker(dividends),
            info={"payoutRatio": 0.4},
        )

        assert result is not None
        # Streak must be 3 (2025>2024>2023>2022), NOT 0 (which would be the bug
        # where partial 2026 < 2025 breaks the streak)
        assert result["dividend_streak"] == 3
        # The partial year is still listed but tagged
        annual = {entry["year"]: entry for entry in result["annual_dividends"]}
        assert annual[2026].get("partial") is True
        assert annual[2025].get("partial") is None or annual[2025].get("partial") is False

    @patch("stock_analysis.tools.fundamentals.datetime")
    def test_cagr_uses_complete_years_only(self, mock_datetime) -> None:
        """CAGR baseline must skip the partial current year so it's anchored on a full year."""
        mock_datetime.now.return_value = pd.Timestamp("2026-05-15").to_pydatetime()

        # 5y: 2021..2025 complete; 2026 partial.
        dividends = _build_dividends({
            2021: [(6, 4.0)],
            2022: [(6, 4.4)],
            2023: [(6, 4.8)],
            2024: [(6, 5.3)],
            2025: [(6, 5.8)],
            2026: [(3, 1.5)],  # partial
        })

        result = _analyze_dividend_history(
            _FakeTicker(dividends),
            info={"payoutRatio": 0.5},
        )

        assert result is not None
        # cagr_1y should compare 2025 to 2024 (both complete), giving roughly
        # (5.8/5.3) - 1 = ~0.094, NOT (1.5/5.8) which would be -0.74.
        cagr_1y = result["cagr_1y"]
        assert cagr_1y is not None
        assert cagr_1y > 0  # complete-year CAGR must be positive given the data

    @patch("stock_analysis.tools.fundamentals.datetime")
    def test_no_partial_year_when_latest_is_prior_year(self, mock_datetime) -> None:
        """When the latest payment is from a prior calendar year, no partial flag."""
        mock_datetime.now.return_value = pd.Timestamp("2026-05-15").to_pydatetime()

        dividends = _build_dividends({
            2024: [(6, 4.0), (12, 4.0)],
            2025: [(6, 4.5), (12, 4.5)],
        })

        result = _analyze_dividend_history(
            _FakeTicker(dividends),
            info={"payoutRatio": 0.5},
        )

        assert result is not None
        annual = {entry["year"]: entry for entry in result["annual_dividends"]}
        # Neither year should have partial flag (2026 has no payments at all)
        assert "partial" not in annual.get(2025, {}) or annual[2025].get("partial") is False
        assert result["dividend_streak"] == 1  # 2025 > 2024
