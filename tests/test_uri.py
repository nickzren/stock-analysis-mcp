"""Tests for URI canonicalization."""

import pytest

from stock_mcp.utils.validators import FetchParams


class TestURICanonicalization:
    """Tests for URI generation and canonicalization."""

    def test_uri_roundtrip_stability(self) -> None:
        """Test URI is stable across multiple generations."""
        params = FetchParams(symbol="NVDA", period="1y", interval="1d", adjusted=True)

        uri1 = params.to_uri()
        uri2 = params.to_uri()
        uri3 = params.to_uri()

        assert uri1 == uri2 == uri3

    def test_uri_format_strict(self) -> None:
        """Test URI follows strict format."""
        params = FetchParams(
            symbol="AAPL",
            period="1y",
            interval="1d",
            adjusted=True,
            tz="America/New_York",
        )
        uri = params.to_uri()

        # Check format: price://SYMBOL/PERIOD/INTERVAL/adjusted|unadjusted
        assert uri.startswith("price://")
        assert "/AAPL/" in uri
        assert "/1y/" in uri
        assert "/1d/" in uri
        assert uri.endswith("/adjusted")

    def test_uri_adjusted_true(self) -> None:
        """Test URI with adjusted=true."""
        params = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=True)
        uri = params.to_uri()

        assert uri.endswith("/adjusted")

    def test_uri_adjusted_false(self) -> None:
        """Test URI with adjusted=false."""
        params = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=False)
        uri = params.to_uri()

        assert uri.endswith("/unadjusted")

    def test_uri_different_params_different_uri(self) -> None:
        """Test different params produce different URIs."""
        params1 = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=True)
        params2 = FetchParams(symbol="NVDA", period="1y", interval="1d", adjusted=True)
        params3 = FetchParams(symbol="AAPL", period="6mo", interval="1d", adjusted=True)
        params4 = FetchParams(symbol="AAPL", period="1y", interval="1h", adjusted=True)
        params5 = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=False)

        uris = {params1.to_uri(), params2.to_uri(), params3.to_uri(), params4.to_uri(), params5.to_uri()}

        # All URIs should be unique
        assert len(uris) == 5

    def test_uri_case_normalized(self) -> None:
        """Test URI is case-normalized."""
        params_lower = FetchParams(symbol="aapl", period="1y", interval="1d", adjusted=True)
        params_upper = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=True)

        assert params_lower.to_uri() == params_upper.to_uri()

    def test_uri_whitespace_normalized(self) -> None:
        """Test URI handles whitespace in symbol."""
        params1 = FetchParams(symbol="AAPL", period="1y", interval="1d", adjusted=True)
        params2 = FetchParams(symbol="  AAPL  ", period="1y", interval="1d", adjusted=True)

        assert params1.to_uri() == params2.to_uri()
