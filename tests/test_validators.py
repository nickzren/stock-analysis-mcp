"""Tests for validators and FetchParams."""

import operator

import pytest

from stock_mcp.utils.validators import (
    VALID_INTERVALS,
    VALID_PERIODS,
    FetchParams,
    check_rule,
    check_rule_expr,
)


class TestFetchParams:
    """Tests for FetchParams dataclass."""

    def test_symbol_normalization(self) -> None:
        """Test symbol is normalized to uppercase."""
        params = FetchParams(symbol="aapl", period="1y", interval="1d", adjusted=True)
        assert params.symbol == "AAPL"

    def test_symbol_strips_whitespace(self) -> None:
        """Test symbol strips whitespace."""
        params = FetchParams(symbol="  nvda  ", period="1y", interval="1d", adjusted=True)
        assert params.symbol == "NVDA"

    def test_period_normalization(self) -> None:
        """Test period is normalized to lowercase."""
        params = FetchParams(symbol="AAPL", period="1Y", interval="1d", adjusted=True)
        assert params.period == "1y"

    def test_interval_normalization(self) -> None:
        """Test interval is normalized to lowercase."""
        params = FetchParams(symbol="AAPL", period="1y", interval="1D", adjusted=True)
        assert params.interval == "1d"

    def test_invalid_period_raises(self) -> None:
        """Test invalid period raises ValueError."""
        with pytest.raises(ValueError, match="Invalid period"):
            FetchParams(symbol="AAPL", period="invalid", interval="1d", adjusted=True)

    def test_invalid_interval_raises(self) -> None:
        """Test invalid interval raises ValueError."""
        with pytest.raises(ValueError, match="Invalid interval"):
            FetchParams(symbol="AAPL", period="1y", interval="invalid", adjusted=True)

    def test_all_valid_periods(self) -> None:
        """Test all valid periods are accepted."""
        for period in VALID_PERIODS:
            params = FetchParams(symbol="AAPL", period=period, interval="1d", adjusted=True)
            assert params.period == period

    def test_all_valid_intervals(self) -> None:
        """Test all valid intervals are accepted."""
        for interval in VALID_INTERVALS:
            params = FetchParams(symbol="AAPL", period="1y", interval=interval, adjusted=True)
            assert params.interval == interval

    def test_to_uri_canonical(self) -> None:
        """Test URI generation is canonical."""
        params = FetchParams(symbol="NVDA", period="1y", interval="1d", adjusted=True)
        uri = params.to_uri()

        # URI should be deterministic
        assert uri == params.to_uri()

        # Should contain symbol, period, interval, adjusted
        assert "NVDA" in uri
        assert "1y" in uri
        assert "1d" in uri
        assert uri.endswith("/adjusted")

    def test_to_uri_sorted_params(self) -> None:
        """Test URI params are sorted for cache key stability."""
        params1 = FetchParams(
            symbol="AAPL", period="1y", interval="1d", adjusted=True, tz="America/New_York"
        )
        params2 = FetchParams(
            symbol="AAPL", period="1y", interval="1d", adjusted=True, tz="America/New_York"
        )

        # URIs should be identical
        assert params1.to_uri() == params2.to_uri()

    def test_to_yf_kwargs(self) -> None:
        """Test yfinance kwargs generation."""
        params = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=True)
        kwargs = params.to_yf_kwargs()

        assert kwargs["tickers"] == "AAPL"
        assert kwargs["period"] == "1y"
        assert kwargs["interval"] == "1d"
        assert kwargs["auto_adjust"] is True
        assert kwargs["progress"] is False

    def test_immutable(self) -> None:
        """Test FetchParams is immutable."""
        params = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=True)

        with pytest.raises(AttributeError):
            params.symbol = "NVDA"


class TestCheckRule:
    """Tests for check_rule function."""

    def test_check_rule_true(self) -> None:
        """Test rule that triggers."""
        result = check_rule(0.5, 0.3, operator.gt)
        assert result is True

    def test_check_rule_false(self) -> None:
        """Test rule that doesn't trigger."""
        result = check_rule(0.2, 0.3, operator.gt)
        assert result is False

    def test_check_rule_none_value(self) -> None:
        """Test rule with None value returns None (not False)."""
        result = check_rule(None, 0.3, operator.gt)
        assert result is None

    def test_check_rule_lt_operator(self) -> None:
        """Test rule with less-than operator."""
        result = check_rule(0.2, 0.3, operator.lt)
        assert result is True

    def test_check_rule_ge_operator(self) -> None:
        """Test rule with greater-or-equal operator."""
        result = check_rule(0.3, 0.3, operator.ge)
        assert result is True


class TestCheckRuleExpr:
    """Tests for check_rule_expr function."""

    def test_check_rule_expr_true(self) -> None:
        """Test expression rule that triggers."""
        result = check_rule_expr(100.0, 50.0, operator.gt)
        assert result is True

    def test_check_rule_expr_false(self) -> None:
        """Test expression rule that doesn't trigger."""
        result = check_rule_expr(50.0, 100.0, operator.gt)
        assert result is False

    def test_check_rule_expr_none_first(self) -> None:
        """Test expression rule with None first value returns None."""
        result = check_rule_expr(None, 50.0, operator.gt)
        assert result is None

    def test_check_rule_expr_none_second(self) -> None:
        """Test expression rule with None second value returns None."""
        result = check_rule_expr(100.0, None, operator.gt)
        assert result is None

    def test_check_rule_expr_both_none(self) -> None:
        """Test expression rule with both None values returns None."""
        result = check_rule_expr(None, None, operator.gt)
        assert result is None
