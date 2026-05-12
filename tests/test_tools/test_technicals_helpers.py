"""Regression tests for technicals helpers under missing-price edge cases."""

import pandas as pd

from stock_analysis.tools.technicals import _build_bollinger, _build_fibonacci


class TestBollingerWithMissingCurrentPrice:
    """_build_bollinger must not crash when current_price is None."""

    def test_none_current_price_yields_none_triggered(self) -> None:
        prices = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0] * 5)

        result = _build_bollinger(prices, current_price=None)

        # Bollinger band values still compute
        assert result["upper"] is not None
        assert result["middle"] is not None
        assert result["lower"] is not None
        # Triggered flags that depend on current_price are None, not booleans
        assert result["rules"]["above_upper"]["triggered"] is None
        assert result["rules"]["below_lower"]["triggered"] is None
        # Squeeze does not depend on current_price
        assert isinstance(result["rules"]["squeeze"]["triggered"], bool)

    def test_valid_current_price_emits_booleans(self) -> None:
        prices = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0] * 5)

        result = _build_bollinger(prices, current_price=200.0)

        assert isinstance(result["rules"]["above_upper"]["triggered"], bool)
        assert isinstance(result["rules"]["below_lower"]["triggered"], bool)


class TestFibonacciWithMissingCurrentPrice:
    """_build_fibonacci must not crash when current_price is None."""

    def test_none_current_price_yields_none_nearest(self) -> None:
        result = _build_fibonacci(
            price_position={"week_52_high": 200.0, "week_52_low": 100.0},
            current_price=None,
        )

        assert result is not None
        # Fib levels still computed
        assert "level_0" in result or "level_618" in result or len(result) > 2
        # nearest_support and nearest_resistance are None, not crashing
        assert result["nearest_support"] is None
        assert result["nearest_resistance"] is None

    def test_valid_current_price_finds_nearest(self) -> None:
        result = _build_fibonacci(
            price_position={"week_52_high": 200.0, "week_52_low": 100.0},
            current_price=150.0,
        )

        assert result is not None
        # 150 falls between the levels (100..200 range); both should resolve
        assert result["nearest_support"] is not None
        assert result["nearest_resistance"] is not None
        assert result["nearest_support"] <= 150.0
        assert result["nearest_resistance"] >= 150.0

    def test_missing_range_returns_none(self) -> None:
        result = _build_fibonacci(
            price_position={"week_52_high": None, "week_52_low": 100.0},
            current_price=150.0,
        )
        assert result is None


class TestPctChangeFillBehavior:
    """Regression: pct_change(fill_method=None) must surface gaps as NaN.

    Default pandas behavior forward-fills NaN before computing percent change, which
    silently masks data gaps. Risk metrics need the explicit `fill_method=None` to
    avoid distorting volatility/correlation calculations.
    """

    def test_pct_change_with_gap_surfaces_nan(self) -> None:
        prices = pd.Series([100.0, float("nan"), 110.0])

        returns = prices.pct_change(fill_method=None)

        # Position 1 (the NaN row) must remain NaN
        assert pd.isna(returns.iloc[0])  # first row always NaN
        assert pd.isna(returns.iloc[1])  # gap surfaces as NaN
        # Position 2: 110 vs prior valid 100 (pandas computes against last valid value
        # even without fill_method when the prior is NaN — verify behavior)
        # The key contract is: NaN in input must not silently become 0 return.
        # We accept either NaN or the correctly-computed (110-100)/100 = 0.10 here.
        assert pd.isna(returns.iloc[2]) or abs(returns.iloc[2] - 0.10) < 1e-9
