"""Tests for OHLCV standardization."""

import pandas as pd
import pytest

from stock_mcp.utils.ohlcv import df_to_csv, df_to_rows, standardize_ohlcv


class TestStandardizeOhlcv:
    """Tests for standardize_ohlcv function."""

    def test_standardize_basic(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test basic standardization."""
        result = standardize_ohlcv(sample_ohlcv_df, adjusted=True)

        # Check column order and names
        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]

        # Check data integrity
        assert len(result) == 10
        assert result["close"].iloc[0] == 100.5

    def test_standardize_removes_adj_close(
        self, sample_ohlcv_df_with_adj_close: pd.DataFrame
    ) -> None:
        """Test that Adj Close column is removed."""
        result = standardize_ohlcv(sample_ohlcv_df_with_adj_close, adjusted=False)

        assert "adj close" not in result.columns
        assert "Adj Close" not in result.columns
        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_standardize_handles_multi_index(self) -> None:
        """Test handling of multi-index columns from yf.download."""
        # Create multi-index DataFrame like yf.download returns for multiple tickers
        arrays = [
            ["Open", "High", "Low", "Close", "Volume"],
            ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)

        df = pd.DataFrame(
            [[100, 101, 99, 100.5, 1000000]],
            index=pd.DatetimeIndex(["2024-01-01"]),
            columns=index,
        )

        result = standardize_ohlcv(df, adjusted=True)
        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_standardize_fills_missing_columns(self) -> None:
        """Test that missing columns are filled with NaN."""
        # DataFrame missing volume
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
            }
        ).set_index("Date")

        result = standardize_ohlcv(df, adjusted=True)

        # Should have all 6 columns
        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]
        # Volume should be NaN
        assert result["volume"].isna().all()

    def test_standardize_date_format_daily(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test date formatting for daily data."""
        result = standardize_ohlcv(sample_ohlcv_df, adjusted=True)

        # Daily data should be YYYY-MM-DD
        assert result["date"].iloc[0] == "2024-01-01"

    def test_standardize_date_format_intraday(self) -> None:
        """Test date formatting for intraday data."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01 09:30", periods=5, freq="5min"),
                "Open": [100, 101, 102, 101, 103],
                "High": [101, 102, 103, 102, 104],
                "Low": [99, 100, 101, 100, 102],
                "Close": [100.5, 101.5, 102.5, 101.5, 103.5],
                "Volume": [1000] * 5,
            }
        ).set_index("Date")

        result = standardize_ohlcv(df, adjusted=True)

        # Intraday data should have time component
        assert "T" in result["date"].iloc[0]
        assert result["date"].iloc[0] == "2024-01-01T09:30:00"

    def test_standardize_tz_aware_dates(self) -> None:
        """Test handling of timezone-aware dates."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range(
                    "2024-01-01 09:30", periods=5, freq="5min", tz="America/New_York"
                ),
                "Open": [100, 101, 102, 101, 103],
                "High": [101, 102, 103, 102, 104],
                "Low": [99, 100, 101, 100, 102],
                "Close": [100.5, 101.5, 102.5, 101.5, 103.5],
                "Volume": [1000] * 5,
            }
        ).set_index("Date")

        result = standardize_ohlcv(df, adjusted=True)

        # Should handle tz-aware dates
        assert len(result) == 5
        assert "date" in result.columns


class TestDfToRows:
    """Tests for df_to_rows function."""

    def test_df_to_rows_lowercase_keys(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test that keys are lowercase."""
        standardized = standardize_ohlcv(sample_ohlcv_df, adjusted=True)
        rows = df_to_rows(standardized)

        assert len(rows) == 10
        for row in rows:
            assert all(k.islower() for k in row.keys())
            assert set(row.keys()) == {"date", "open", "high", "low", "close", "volume"}


class TestDfToCsv:
    """Tests for df_to_csv function."""

    def test_df_to_csv_roundtrip(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test CSV roundtrip."""
        standardized = standardize_ohlcv(sample_ohlcv_df, adjusted=True)
        csv_text = df_to_csv(standardized)

        # Parse back
        from io import StringIO

        parsed = pd.read_csv(StringIO(csv_text))

        assert list(parsed.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(parsed) == 10
