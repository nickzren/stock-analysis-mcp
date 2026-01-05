"""Tests for technical indicators."""

import math

import pandas as pd
import pytest

from stock_mcp.utils.indicators import (
    calculate_atr,
    calculate_beta,
    calculate_current_drawdown,
    calculate_ema,
    calculate_macd,
    calculate_max_drawdown,
    calculate_pairwise_correlations,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
    calculate_var,
    calculate_volatility,
)


class TestSMA:
    """Tests for SMA calculation."""

    def test_sma_basic(self, sample_price_series: pd.Series) -> None:
        """Test basic SMA calculation."""
        sma = calculate_sma(sample_price_series, 5)

        # SMA should have NaN for first (period-1) values
        assert sma.iloc[:4].isna().all()

        # Check a known value
        # SMA of first 5 values: (100 + 101 + 102 + 101.5 + 103) / 5 = 101.5
        assert abs(sma.iloc[4] - 101.5) < 0.01

    def test_sma_insufficient_data(self) -> None:
        """Test SMA with insufficient data."""
        prices = pd.Series([100, 101, 102])
        sma = calculate_sma(prices, 5)

        # All values should be NaN
        assert sma.isna().all()


class TestEMA:
    """Tests for EMA calculation."""

    def test_ema_basic(self, sample_price_series: pd.Series) -> None:
        """Test basic EMA calculation."""
        ema = calculate_ema(sample_price_series, 5)

        # EMA should have values starting from period
        assert not pd.isna(ema.iloc[4])

        # EMA should be smoother than SMA
        sma = calculate_sma(sample_price_series, 5)
        assert ema.std() <= sma.std() * 1.1  # Allow small tolerance


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_range(self, sample_price_series: pd.Series) -> None:
        """Test RSI stays in 0-100 range."""
        rsi = calculate_rsi(sample_price_series, 14)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_uptrend(self) -> None:
        """Test RSI in strong uptrend."""
        # Consistently rising prices
        prices = pd.Series([100 + i for i in range(30)])
        rsi = calculate_rsi(prices, 14)

        # RSI should be high (overbought territory)
        assert rsi.iloc[-1] > 70

    def test_rsi_downtrend(self) -> None:
        """Test RSI in strong downtrend."""
        # Consistently falling prices
        prices = pd.Series([100 - i for i in range(30)])
        rsi = calculate_rsi(prices, 14)

        # RSI should be low (oversold territory)
        assert rsi.iloc[-1] < 30


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_components(self, sample_price_series: pd.Series) -> None:
        """Test MACD returns all components."""
        macd = calculate_macd(sample_price_series, 12, 26, 9)

        assert "macd_line" in macd
        assert "signal_line" in macd
        assert "histogram" in macd

        # Histogram should equal MACD - Signal
        valid_idx = ~(macd["histogram"].isna())
        diff = macd["macd_line"][valid_idx] - macd["signal_line"][valid_idx]
        assert (abs(macd["histogram"][valid_idx] - diff) < 0.0001).all()


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_basic(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test basic ATR calculation."""
        high = sample_ohlcv_df["High"]
        low = sample_ohlcv_df["Low"]
        close = sample_ohlcv_df["Close"]

        atr = calculate_atr(high, low, close, 5)

        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_atr_increases_with_volatility(self) -> None:
        """Test ATR increases with higher volatility."""
        # Low volatility
        low_vol = pd.DataFrame(
            {
                "high": [101, 101, 101, 101, 101, 101, 101, 101, 101, 101],
                "low": [99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
                "close": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            }
        )

        # High volatility
        high_vol = pd.DataFrame(
            {
                "high": [110, 110, 110, 110, 110, 110, 110, 110, 110, 110],
                "low": [90, 90, 90, 90, 90, 90, 90, 90, 90, 90],
                "close": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            }
        )

        atr_low = calculate_atr(low_vol["high"], low_vol["low"], low_vol["close"], 5)
        atr_high = calculate_atr(high_vol["high"], high_vol["low"], high_vol["close"], 5)

        assert atr_high.iloc[-1] > atr_low.iloc[-1]


class TestReturns:
    """Tests for returns calculation."""

    def test_returns_basic(self, sample_price_series: pd.Series) -> None:
        """Test basic returns calculation."""
        ret = calculate_returns(sample_price_series, 5)

        assert ret is not None
        # 5-day return from index -6 to -1
        expected = (sample_price_series.iloc[-1] - sample_price_series.iloc[-6]) / sample_price_series.iloc[-6]
        assert abs(ret - expected) < 0.0001

    def test_returns_insufficient_data(self) -> None:
        """Test returns with insufficient data."""
        prices = pd.Series([100, 101, 102])
        ret = calculate_returns(prices, 5)

        assert ret is None


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_annualized(self, sample_returns_series: pd.Series) -> None:
        """Test annualized volatility calculation."""
        vol = calculate_volatility(sample_returns_series, annualize=True)

        assert vol is not None
        # Annualized vol should be larger than daily
        daily_vol = calculate_volatility(sample_returns_series, annualize=False)
        assert vol > daily_vol

    def test_volatility_annualization_factor(self, sample_returns_series: pd.Series) -> None:
        """Test that annualization uses sqrt(252)."""
        vol_ann = calculate_volatility(sample_returns_series, annualize=True)
        vol_daily = calculate_volatility(sample_returns_series, annualize=False)

        expected_ann = vol_daily * math.sqrt(252)
        assert abs(vol_ann - expected_ann) < 0.0001


class TestDrawdown:
    """Tests for drawdown calculations."""

    def test_max_drawdown_basic(self) -> None:
        """Test max drawdown calculation."""
        # Price goes up then crashes
        prices = pd.Series([100, 110, 120, 100, 90, 95])
        dd = calculate_max_drawdown(prices)

        # Max drawdown from 120 to 90 = -25%
        assert dd is not None
        assert abs(dd - (-0.25)) < 0.01

    def test_current_drawdown(self) -> None:
        """Test current drawdown calculation."""
        prices = pd.Series([100, 110, 120, 115, 118])
        dd, days = calculate_current_drawdown(prices)

        # Current at 118, peak at 120 = -1.67%
        assert dd is not None
        assert abs(dd - (-2 / 120)) < 0.01


class TestBeta:
    """Tests for beta calculation."""

    def test_beta_perfect_correlation(self) -> None:
        """Test beta with perfect correlation."""
        # Same returns = beta of 1
        returns = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.02] * 50,
            index=pd.date_range("2024-01-01", periods=250, freq="D"),
        )

        result = calculate_beta(returns, returns, min_overlap=100)

        assert result["value"] is not None
        assert abs(result["value"] - 1.0) < 0.01

    def test_beta_insufficient_overlap(self) -> None:
        """Test beta with insufficient overlap."""
        returns1 = pd.Series(
            [0.01] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="D"),
        )
        returns2 = pd.Series(
            [0.01] * 50,
            index=pd.date_range("2024-06-01", periods=50, freq="D"),
        )

        result = calculate_beta(returns1, returns2, min_overlap=200)

        assert result["value"] is None
        assert "insufficient_overlap" in result.get("warning", "")


class TestVaR:
    """Tests for VaR calculation."""

    def test_var_positive(self) -> None:
        """Test VaR is positive (represents loss)."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02, -0.03] * 20)
        var = calculate_var(returns, 0.95)

        assert var is not None
        assert var > 0  # Loss should be positive

    def test_var_insufficient_data(self) -> None:
        """Test VaR with insufficient data."""
        returns = pd.Series([0.01, -0.02, 0.015])
        var = calculate_var(returns, 0.95)

        assert var is None


class TestPairwiseCorrelations:
    """Tests for pairwise correlation calculation."""

    def test_pairwise_correlations_basic(self) -> None:
        """Test basic pairwise correlations."""
        returns = {
            "A": pd.Series(
                [0.01, -0.02, 0.015, -0.01, 0.02] * 30,
                index=pd.date_range("2024-01-01", periods=150, freq="D"),
            ),
            "B": pd.Series(
                [0.01, -0.02, 0.015, -0.01, 0.02] * 30,
                index=pd.date_range("2024-01-01", periods=150, freq="D"),
            ),
            "C": pd.Series(
                [-0.01, 0.02, -0.015, 0.01, -0.02] * 30,
                index=pd.date_range("2024-01-01", periods=150, freq="D"),
            ),
        }

        result = calculate_pairwise_correlations(returns, min_overlap=100)

        assert len(result["pairs"]) == 3  # A-B, A-C, B-C

        # A and B should be perfectly correlated
        ab_pair = next(p for p in result["pairs"] if set(p["symbols"]) == {"A", "B"})
        assert ab_pair["correlation"] is not None
        assert abs(ab_pair["correlation"] - 1.0) < 0.01

    def test_pairwise_high_correlation_uses_abs(self) -> None:
        """Test that high correlation detection uses absolute value."""
        returns = {
            "A": pd.Series(
                [0.01, -0.02, 0.015, -0.01, 0.02] * 30,
                index=pd.date_range("2024-01-01", periods=150, freq="D"),
            ),
            "B": pd.Series(
                [-0.01, 0.02, -0.015, 0.01, -0.02] * 30,  # Perfectly negatively correlated
                index=pd.date_range("2024-01-01", periods=150, freq="D"),
            ),
        }

        result = calculate_pairwise_correlations(returns, min_overlap=100)

        # Should detect high correlation even though it's negative
        assert len(result["high_correlation_pairs"]) == 1
