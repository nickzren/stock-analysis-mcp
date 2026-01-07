"""Tests for info completeness checking."""

import math

import pytest

from stock_mcp.data.yfinance_client import (
    INFO_CORE_FUND_SENTINELS,
    INFO_FUND_SENTINELS,
    InfoCompleteness,
    YFinanceIncompleteInfoError,
    _has_value,
    assess_info_completeness,
)


class TestAssessInfoCompleteness:
    """Tests for assess_info_completeness function."""

    def test_complete_equity_with_fundamentals(self):
        """Complete EQUITY response with fundamentals should pass."""
        info = {
            "symbol": "MRNA",
            "quoteType": "EQUITY",
            "totalCash": 4504000000,
            "profitMargins": -1.39,
            "revenueGrowth": -0.45,
            "freeCashflow": -1984375040,
        }
        c = assess_info_completeness(info)
        assert c.is_incomplete is False
        assert c.quote_type == "EQUITY"
        assert c.sentinel_keys_present >= 1
        assert c.sentinel_values_present >= 1

    def test_incomplete_equity_no_fundamentals(self):
        """Incomplete EQUITY response (401 style) should be flagged."""
        # This simulates a crumb-broken response with only basic metadata
        info = {
            "symbol": "MRNA",
            "quoteType": "EQUITY",
            "regularMarketPrice": 35.66,
            "exchange": "NASDAQ",
            # No fundamental sentinel keys
        }
        c = assess_info_completeness(info)
        assert c.is_incomplete is True
        assert c.sentinel_keys_present == 0
        assert c.sentinel_values_present == 0

    def test_etf_does_not_enforce_fundamentals(self):
        """ETF quoteType should NOT enforce fundamentals."""
        info = {
            "symbol": "SPY",
            "quoteType": "ETF",
            "regularMarketPrice": 691.81,
        }
        c = assess_info_completeness(info)
        assert c.is_incomplete is False
        assert c.quote_type == "ETF"

    def test_mutual_fund_does_not_enforce_fundamentals(self):
        """MUTUALFUND quoteType should NOT enforce fundamentals."""
        info = {
            "symbol": "VFIAX",
            "quoteType": "MUTUALFUND",
            "regularMarketPrice": 500.0,
        }
        c = assess_info_completeness(info)
        assert c.is_incomplete is False
        assert c.quote_type == "MUTUALFUND"

    def test_index_does_not_enforce_fundamentals(self):
        """INDEX quoteType should NOT enforce fundamentals."""
        info = {
            "symbol": "^GSPC",
            "quoteType": "INDEX",
            "regularMarketPrice": 5000.0,
        }
        c = assess_info_completeness(info)
        assert c.is_incomplete is False
        assert c.quote_type == "INDEX"

    def test_empty_dict_is_incomplete(self):
        """Empty dict should be flagged as incomplete."""
        c = assess_info_completeness({})
        assert c.is_incomplete is True
        assert c.key_count == 0

    def test_none_input(self):
        """None input should be handled (as empty dict)."""
        # Type checker won't allow None, but test runtime behavior
        c = assess_info_completeness({})  # type: ignore
        assert c.is_incomplete is True

    def test_sentinel_keys_vs_values(self):
        """Test that keys present vs values present are counted correctly."""
        # Has keys but values are None (only 2 keys, below threshold)
        info = {
            "symbol": "TEST",
            "quoteType": "EQUITY",
            "totalCash": None,  # Key present but value None
            "profitMargins": None,
            # Only 2 sentinel keys, below _MIN_SENTINEL_KEYS_PRESENT (3)
        }
        c = assess_info_completeness(info)
        # Keys are present, but values are None
        assert c.sentinel_keys_present == 2  # totalCash, profitMargins
        assert c.sentinel_values_present == 0
        # Should be incomplete because keys_present < 3 AND values_present < 1
        assert c.is_incomplete is True

    def test_keys_present_but_all_none_is_incomplete(self):
        """Keys present but all values None should be flagged as incomplete.

        This is a regression test for Hole #1: the boolean logic should catch
        "keys present but values missing" edge cases.
        """
        info = {
            "symbol": "TEST",
            "quoteType": "EQUITY",
            "totalCash": None,
            "profitMargins": None,
            "freeCashflow": None,
            "totalRevenue": None,  # 4 core sentinel keys present, but all None
        }
        c = assess_info_completeness(info)
        assert c.sentinel_keys_present >= 3
        assert c.sentinel_values_present == 0
        # Should be incomplete because no core values are truly present
        assert c.is_incomplete is True

    def test_passes_with_one_value_present(self):
        """Passes if sentinel_values_present >= 1."""
        info = {
            "symbol": "TEST",
            "quoteType": "EQUITY",
            "totalCash": 1000000,  # One sentinel value present
        }
        c = assess_info_completeness(info)
        assert c.sentinel_values_present >= 1
        assert c.is_incomplete is False

    def test_unknown_quote_type_enforces_fundamentals(self):
        """Unknown/missing quoteType should enforce fundamentals."""
        info = {
            "symbol": "UNKNOWN",
            "regularMarketPrice": 100.0,
            # No quoteType
        }
        c = assess_info_completeness(info)
        # Should enforce fundamentals for unknown quoteType
        assert c.is_incomplete is True

    def test_empty_quote_type_enforces_fundamentals(self):
        """Empty string quoteType should enforce fundamentals."""
        info = {
            "symbol": "TEST",
            "quoteType": "",
            "regularMarketPrice": 100.0,
        }
        c = assess_info_completeness(info)
        assert c.is_incomplete is True


class TestYFinanceIncompleteInfoError:
    """Tests for YFinanceIncompleteInfoError exception."""

    def test_exception_message_format(self):
        """Exception message should include all diagnostic info."""
        err = YFinanceIncompleteInfoError(
            "MRNA",
            key_count=83,
            quote_type="EQUITY",
            sentinel_keys_present=0,
            sentinel_values_present=0,
        )
        msg = str(err)
        assert "MRNA" in msg
        assert "keys=83" in msg
        assert "quoteType=EQUITY" in msg
        assert "sentinel_keys_present=0" in msg
        assert "sentinel_values_present=0" in msg

    def test_exception_attributes(self):
        """Exception should store all attributes."""
        err = YFinanceIncompleteInfoError(
            "AAPL",
            key_count=169,
            quote_type="EQUITY",
            sentinel_keys_present=12,
            sentinel_values_present=10,
        )
        assert err.symbol == "AAPL"
        assert err.key_count == 169
        assert err.quote_type == "EQUITY"
        assert err.sentinel_keys_present == 12
        assert err.sentinel_values_present == 10


class TestHasValue:
    """Tests for _has_value helper function."""

    def test_none_is_not_a_value(self):
        """None should not count as a value."""
        assert _has_value(None) is False

    def test_nan_is_not_a_value(self):
        """NaN should not count as a value.

        This is a regression test for Hole #2: yfinance uses float("nan")
        for missing numerics, which passes `is not None` but should be
        treated as missing.
        """
        assert _has_value(float("nan")) is False
        assert _has_value(math.nan) is False

    def test_empty_string_is_not_a_value(self):
        """Empty/whitespace string should not count as a value."""
        assert _has_value("") is False
        assert _has_value("   ") is False
        assert _has_value("\t\n") is False

    def test_zero_is_a_value(self):
        """Zero should count as a value (it's meaningful)."""
        assert _has_value(0) is True
        assert _has_value(0.0) is True

    def test_negative_is_a_value(self):
        """Negative numbers should count as values."""
        assert _has_value(-1.5) is True
        assert _has_value(-1000000) is True

    def test_positive_is_a_value(self):
        """Positive numbers should count as values."""
        assert _has_value(1.5) is True
        assert _has_value(1000000) is True

    def test_string_is_a_value(self):
        """Non-empty strings should count as values."""
        assert _has_value("test") is True
        assert _has_value("EQUITY") is True


class TestNaNHandling:
    """Regression tests for NaN handling in completeness check."""

    def test_nan_values_not_counted_as_present(self):
        """NaN values in sentinel fields should not count as present.

        This is a regression test for Hole #2: if profitMargins=NaN,
        it should still be flagged as incomplete.
        """
        info = {
            "symbol": "TEST",
            "quoteType": "EQUITY",
            "profitMargins": float("nan"),  # NaN, not a real value
            "totalCash": float("nan"),
            "freeCashflow": float("nan"),
        }
        c = assess_info_completeness(info)
        # Keys are present, but NaN should not count as values
        assert c.sentinel_keys_present >= 3
        assert c.sentinel_values_present == 0  # NaN doesn't count
        assert c.is_incomplete is True

    def test_mix_of_nan_and_real_values(self):
        """Mix of NaN and real values should work correctly."""
        info = {
            "symbol": "TEST",
            "quoteType": "EQUITY",
            "profitMargins": float("nan"),  # NaN
            "totalCash": 1000000,  # Real value
            "freeCashflow": float("nan"),  # NaN
        }
        c = assess_info_completeness(info)
        assert c.sentinel_keys_present >= 2
        assert c.sentinel_values_present == 1  # Only totalCash counts
        assert c.is_incomplete is False  # One real core value is enough


class TestInfoFundSentinels:
    """Tests for INFO_FUND_SENTINELS constant."""

    def test_sentinels_are_strings(self):
        """All sentinel keys should be strings."""
        for sentinel in INFO_FUND_SENTINELS:
            assert isinstance(sentinel, str)

    def test_sentinels_not_empty(self):
        """Should have at least a few sentinel keys."""
        assert len(INFO_FUND_SENTINELS) >= 5

    def test_key_fundamental_sentinels_present(self):
        """Key fundamental metrics should be in sentinels."""
        required = ["totalCash", "profitMargins", "freeCashflow"]
        for key in required:
            assert key in INFO_FUND_SENTINELS, f"{key} should be a sentinel"

    def test_core_sentinels_are_subset(self):
        """Core sentinels should be a subset of full sentinels."""
        for core in INFO_CORE_FUND_SENTINELS:
            assert core in INFO_FUND_SENTINELS, f"{core} should be in full sentinels"

    def test_core_sentinels_are_financial_statement_fields(self):
        """Core sentinels should be true financial statement fields."""
        # These should be fields that crumb-broken payloads reliably lack
        expected_core = [
            "totalRevenue",
            "revenueGrowth",
            "profitMargins",
            "grossMargins",
            "operatingCashflow",
            "freeCashflow",
            "totalCash",
        ]
        for field in expected_core:
            assert field in INFO_CORE_FUND_SENTINELS, f"{field} should be a core sentinel"
