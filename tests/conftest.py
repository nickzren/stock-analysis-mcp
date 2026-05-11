"""Pytest configuration and fixtures."""

import pandas as pd
import pytest

from stock_analysis.utils.validators import FetchParams


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Sample OHLCV DataFrame for testing."""
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Open": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5],
            "High": [101.0, 102.5, 103.0, 102.5, 104.5, 105.5, 105.0, 106.5, 107.0, 106.5],
            "Low": [99.5, 100.5, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 105.0, 104.5],
            "Close": [100.5, 102.0, 101.5, 102.0, 104.0, 103.5, 104.5, 106.0, 105.5, 106.0],
            "Volume": [1000000] * 10,
        }
    ).set_index("Date")


@pytest.fixture
def sample_ohlcv_df_with_adj_close(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Sample OHLCV DataFrame with Adj Close column."""
    df = sample_ohlcv_df.copy()
    df.insert(
        df.columns.get_loc("Volume"),
        "Adj Close",
        [100.0, 101.5, 101.0, 101.5, 103.5, 103.0, 104.0, 105.5, 105.0, 105.5],
    )
    return df


@pytest.fixture
def sample_price_series() -> pd.Series:
    """Sample price series for indicator testing."""
    return pd.Series(
        [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
         107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
         114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0]
    )


@pytest.fixture
def sample_returns_series(sample_price_series: pd.Series) -> pd.Series:
    """Sample returns series for indicator testing."""
    return sample_price_series.pct_change().dropna()


@pytest.fixture
def default_fetch_params() -> FetchParams:
    """Canonical 1y daily adjusted AAPL fetch params."""
    return FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=True)
