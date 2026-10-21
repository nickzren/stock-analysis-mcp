"""C.1: transport isolation and the independent property-read protections."""

from typing import Any

import pandas as pd
import pytest
import yfinance as yf
from yfinance.data import YfData

from stock_analysis.data import provider, yfinance_client
from stock_analysis.data.cache_manager import ticker_cache
from tests.support.fake_ticker import FakeTicker, http_error
from tests.support.transport import stubbed_yfinance_transport

SYMBOL = "EXMP"


@pytest.mark.parametrize("original", [False, True])
def test_transport_starts_unchunked_and_restores_the_previous_mode(
    monkeypatch: pytest.MonkeyPatch, original: bool,
) -> None:
    data = YfData()
    monkeypatch.setattr(data, "fundamentals_use_chunked", original, raising=False)

    with stubbed_yfinance_transport():
        assert data.fundamentals_use_chunked is False
        data.fundamentals_use_chunked = not original

    assert data.fundamentals_use_chunked is original


async def test_failed_property_evicts_its_ticker_and_clears_response_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticker = FakeTicker(SYMBOL)
    ticker_cache.set(SYMBOL, ticker, 3600)
    expired: list[bool] = []
    monkeypatch.setattr(provider, "expire_yfinance_response_cache", lambda: expired.append(True))

    def read(_: Any) -> Any:
        raise http_error(404)

    result = await provider.read_property(ticker, "income", read)

    assert (result.status, result.error_type, result.attempts) == ("failed", "http_404", 1)
    assert ticker_cache.get(SYMBOL) is None
    assert expired == [True]


async def test_property_retry_expires_response_cache_and_uses_a_different_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticker = yf.Ticker(SYMBOL)
    ticker_cache.set(SYMBOL, ticker, 3600)
    seen: list[Any] = []
    expired: list[bool] = []
    monkeypatch.setattr(provider, "expire_yfinance_response_cache", lambda: expired.append(True))

    def read(current: Any) -> list[int]:
        seen.append(current)
        if len(seen) == 1:
            raise http_error(503)
        return [100]

    result = await provider.read_property(ticker, "income", read)

    assert (result.status, result.attempts, result.payload) == ("ok", 2, [100])
    assert len(seen) == 2 and seen[0] is ticker and seen[1] is not ticker
    assert ticker_cache.get(SYMBOL) is None
    assert expired == [True]


async def test_application_earnings_parser_does_not_use_library_html_parsers(
    monkeypatch: pytest.MonkeyPatch, yf_transport: Any,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the application must use its own html.parser earnings read")

    monkeypatch.setattr(yf.Ticker, "get_earnings_dates", forbidden)
    monkeypatch.setattr(pd, "read_html", forbidden)
    yf_transport.route("earnings_page", "valid_table")

    result = await provider.read_earnings_page(SYMBOL)

    assert result.status == "ok"
    assert [row["timestamp"].date().isoformat() for row in result.payload] == [
        "2027-01-27", "2026-10-28", "2026-07-29", "2026-04-29",
    ]
    assert len(yf_transport.calls_to("earnings_page")) == 1


@pytest.mark.parametrize("statuses", [[], [404], [503], [401, 503]])
def test_normal_return_rejects_any_recorded_http_error(statuses: list[int]) -> None:
    payload = {"revenue": 999}

    def suppressed_error() -> dict[str, int]:
        yfinance_client._job_responses.statuses.extend(statuses)
        return payload

    read = yfinance_client.preserving_http_status(suppressed_error)
    if statuses:
        with pytest.raises(yfinance_client.HiddenHTTPError) as error:
            read()
        assert error.value.status == statuses[-1]
        assert yfinance_client.http_status(error.value) == statuses[-1]
    else:
        assert read() is payload
    assert yfinance_client._job_responses.statuses is None
