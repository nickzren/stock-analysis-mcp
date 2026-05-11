"""Tests for in-memory price resource caching."""

import time

import pandas as pd

from stock_analysis.data.cache import PriceCache
from stock_analysis.utils.validators import FetchParams


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        }
    )


def test_price_cache_roundtrip() -> None:
    cache = PriceCache()
    params = FetchParams(symbol="NVDA", period="1y", interval="1d", adjusted=True)

    uri = cache.store(params, _frame())

    assert cache.exists(uri) is True
    assert cache.get_csv(uri) is not None

    metadata = cache.get_metadata(uri)
    assert metadata is not None
    assert metadata["rows"] == 2
    assert metadata["columns"] == ["date", "open", "high", "low", "close", "volume"]


def test_price_cache_ttl_expiry(default_fetch_params: FetchParams) -> None:
    cache = PriceCache()

    uri = cache.store(default_fetch_params, _frame(), ttl=0)
    time.sleep(0.01)

    assert cache.exists(uri) is False
    assert cache.get(uri) is None
