"""Pytest configuration and fixtures."""

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import curl_cffi.requests
import pandas as pd
import pytest
import requests
import yfinance as yf

import tests.support.transport as transport
from stock_analysis.data import jobs, provider, yfinance_client
from stock_analysis.utils.validators import FetchParams
from tests.support.network_guard import blocked_request, violation_report
from tests.support.transport import FakeSession, stubbed_yfinance_transport

pytest_plugins = ["pytester"]


@pytest.fixture(scope="session", autouse=True)
def _isolated_yfinance_cache(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep yfinance's persistent cookie/tz/ISIN caches out of the developer's cache dir."""
    yf.set_tz_cache_location(str(tmp_path_factory.mktemp("yfinance-cache")))


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Fail any test that sends a request through a curl_cffi or requests session
    (yfinance's transport), or makes an unrouted stub request, even if the error
    is swallowed."""
    violations: list[str] = []
    monkeypatch.setattr(
        curl_cffi.requests.Session, "request", blocked_request("curl_cffi", violations)
    )
    monkeypatch.setattr(requests.Session, "request", blocked_request("requests", violations))
    monkeypatch.setattr(transport, "violation_sink", violations)
    yield violations
    if violations:
        pytest.fail(violation_report(violations), pytrace=False)


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep watchlist/scan state out of the developer's real data directory."""
    monkeypatch.setenv("STOCK_ANALYSIS_DATA_DIR", str(tmp_path / "stock-analysis-data"))


@pytest.fixture(autouse=True)
def _provider_test_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No real retry sleeps; no rate-limit cooldown or cached source reads leaking
    between tests."""

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(provider, "_sleep", no_sleep)
    monkeypatch.setattr(yfinance_client, "_sleep", no_sleep)
    monkeypatch.setattr(yfinance_client, "_rate_limited_until", 0.0)
    provider._source_cache.clear()
    yield
    provider._source_cache.clear()
    # A test that leaves a job blocked must release it itself; this only stops a
    # finished test's permits, locks and in-flight entries leaking into the next.
    jobs._gate.reset()
    jobs.clear_flights()


@pytest.fixture
def host_tz() -> Iterator[Callable[[str], None]]:
    """Set the process time zone for one test (host-local date conversions)."""
    if not hasattr(time, "tzset"):
        # e.g. some uv-managed (python-build-standalone) builds
        pytest.skip("time.tzset is unavailable on this Python build")
    original = os.environ.get("TZ")

    def set_tz(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield set_tz
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.fixture
def yf_transport() -> Iterator[FakeSession]:
    """Real pinned yfinance over a stubbed transport, in the app's error mode."""
    with stubbed_yfinance_transport() as session:
        yield session


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
