"""Characterization of the pinned yfinance over a stubbed transport (Step 1 WP0).

These tests pin the LIBRARY's behavior, including the defects Step 1 works around
(marked "Defect"). The app sets ``hide_exceptions=False`` at import (WP1), so the
``yf_transport`` fixture runs in that mode; tests about yfinance's hidden-error
mode opt into it explicitly. When a work package changes how we read one of these
sources, update the affected test there.
"""

import importlib.util
from collections.abc import Callable
from datetime import date

import pytest
import yfinance as yf
from curl_cffi.requests.exceptions import HTTPError
from yfinance.data import YfData
from yfinance.exceptions import YFDataException, YFRateLimitError

from tests.support.transport import FakeSession, stubbed_yfinance_transport

SYMBOL = "EXMP"
EARNINGS_PAGE_URL = "https://finance.yahoo.com/calendar/earnings?symbol=EXMP&offset=0&size=25"


class TestCalendar:
    @pytest.mark.parametrize(
        ("tz", "expected"),
        [("UTC", date(2026, 10, 9)), ("America/New_York", date(2026, 10, 8))],
    )
    def test_dates_follow_host_timezone(
        self,
        yf_transport: FakeSession,
        host_tz: Callable[[str], None],
        tz: str,
        expected: date,
    ) -> None:
        """Defect (WP2): epoch 2026-10-09T00:30Z is truncated in the HOST zone."""
        host_tz(tz)
        yf_transport.route("calendar", "valid_one_epoch")

        assert yf.Ticker(SYMBOL).calendar["Earnings Date"] == [expected]
        assert len(yf_transport.calls_to("calendar")) == 1

    def test_two_epoch_window_returns_both_dates(
        self, yf_transport: FakeSession, host_tz: Callable[[str], None]
    ) -> None:
        host_tz("America/New_York")
        yf_transport.route("calendar", "valid_window")

        assert yf.Ticker(SYMBOL).calendar["Earnings Date"] == [
            date(2026, 10, 27),
            date(2026, 10, 31),
        ]

    def test_valid_calendar_without_earnings_is_empty(self, yf_transport: FakeSession) -> None:
        yf_transport.route("calendar", "valid_no_earnings")

        assert yf.Ticker(SYMBOL).calendar == {}

    @pytest.mark.parametrize("name", ["http_401", "http_503", "http_404_json"])
    def test_hidden_http_error_is_indistinguishable_from_empty_calendar(self, name: str) -> None:
        """Defect (DL#2), worked around in WP1: with errors hidden (the library
        default), a failed read returns the same {} as a valid empty one."""
        with stubbed_yfinance_transport(hide_exceptions=True) as session:
            session.route("calendar", name)

            assert yf.Ticker(SYMBOL).calendar == {}
            # yfinance re-requests once with the other cookie strategy on any status >= 400.
            assert len(session.calls_to("calendar")) == 2

    @pytest.mark.parametrize(
        ("name", "status"), [("http_401", 401), ("http_503", 503), ("http_404_json", 404)]
    )
    def test_http_error_raises_when_not_hidden(self, name: str, status: int) -> None:
        with stubbed_yfinance_transport(hide_exceptions=False) as session:
            session.route("calendar", name)

            with pytest.raises(HTTPError) as raised:
                _ = yf.Ticker(SYMBOL).calendar

        assert raised.value.response.status_code == status

    def test_rate_limit_raises_even_when_hidden(self) -> None:
        with stubbed_yfinance_transport(hide_exceptions=True) as session:
            session.route("calendar", "http_429")

            with pytest.raises(YFRateLimitError):
                _ = yf.Ticker(SYMBOL).calendar

    @pytest.mark.parametrize(
        ("hide", "error"), [(True, YFDataException), (False, KeyError)]
    )
    def test_malformed_response_poisons_ticker_cache(
        self, hide: bool, error: type[Exception]
    ) -> None:
        """Defect (WP1, R2): _calendar is set to {} before parsing, so it sticks."""
        with stubbed_yfinance_transport(hide_exceptions=hide) as session:
            session.route("calendar", "malformed_missing_calendar_events")
            ticker = yf.Ticker(SYMBOL)

            with pytest.raises(error):
                _ = ticker.calendar
            assert ticker.calendar == {}
            assert len(session.calls_to("calendar")) == 1


class TestNews:
    def test_valid_stream_returns_articles(self, yf_transport: FakeSession) -> None:
        yf_transport.route("news", "valid_stream")

        news = yf.Ticker(SYMBOL).get_news()

        assert [item["id"] for item in news] == ["n1"]

    @pytest.mark.parametrize("hide", [True, False])
    @pytest.mark.parametrize("name", ["valid_empty_stream", "http_404_json", "malformed_envelope"])
    def test_errors_look_like_an_empty_feed(self, name: str, hide: bool) -> None:
        """Defect (WP1, R1): HTTP errors and malformed envelopes return [] like a healthy empty feed."""
        with stubbed_yfinance_transport(hide_exceptions=hide) as session:
            session.route("news", name)

            assert yf.Ticker(SYMBOL).get_news() == []

    def test_library_drops_ads_only(self, yf_transport: FakeSession) -> None:
        """Items without a pubDate survive yfinance; our parser rejects them later."""
        yf_transport.route("news", "all_rejected")

        assert [item["id"] for item in yf.Ticker(SYMBOL).get_news()] == ["n2"]

    @pytest.mark.parametrize(("name", "calls"), [("valid_stream", 1), ("valid_empty_stream", 2)])
    def test_only_nonempty_news_is_cached_on_ticker(
        self, yf_transport: FakeSession, name: str, calls: int
    ) -> None:
        yf_transport.route("news", name)
        ticker = yf.Ticker(SYMBOL)

        ticker.get_news()
        ticker.get_news()

        assert len(yf_transport.calls_to("news")) == calls


class TestSearch:
    def test_valid_news(self, yf_transport: FakeSession) -> None:
        yf_transport.route("search", "valid_news")

        news = yf.Search(SYMBOL, news_count=8, raise_errors=True).news

        assert [item["uuid"] for item in news] == ["s1"]

    @pytest.mark.parametrize("hide", [True, False])
    @pytest.mark.parametrize("name", ["valid_empty_news", "missing_news_key", "http_404_json"])
    def test_errors_look_like_empty_news(self, name: str, hide: bool) -> None:
        """Defect (WP1, R1): raise_errors=True does not check HTTP status or the envelope."""
        with stubbed_yfinance_transport(hide_exceptions=hide) as session:
            session.route("search", name)

            assert yf.Search(SYMBOL, news_count=8, raise_errors=True).news == []

    def test_successful_response_is_reused_across_searches(self, yf_transport: FakeSession) -> None:
        """Defect (WP1, R1.6): cache_get has no expiry, so a stable URL is served from cache."""
        yf_transport.route("search", "valid_news")

        yf.Search(SYMBOL, news_count=8, raise_errors=True)
        yf.Search(SYMBOL, news_count=8, raise_errors=True)

        assert len(yf_transport.calls_to("search")) == 1


class TestResponseCache:
    def test_cached_error_response_is_served_again(self, yf_transport: FakeSession) -> None:
        """Defect (WP1, R2): cache_get caches responses regardless of status."""
        yf_transport.route("earnings_page", "error_page")
        data = YfData()

        assert data.cache_get(EARNINGS_PAGE_URL).status_code == 503
        calls = len(yf_transport.calls_to("earnings_page"))
        yf_transport.route("earnings_page", "no_results")

        assert data.cache_get(EARNINGS_PAGE_URL).status_code == 503
        assert len(yf_transport.calls_to("earnings_page")) == calls

    def test_error_page_reaches_fresh_tickers_from_cache(self, yf_transport: FakeSession) -> None:
        yf_transport.route("earnings_page", "error_page")
        assert yf.Ticker(SYMBOL).get_earnings_dates() is None
        calls = len(yf_transport.calls_to("earnings_page"))
        yf_transport.route("earnings_page", "no_results")

        assert yf.Ticker(SYMBOL).get_earnings_dates() is None
        assert len(yf_transport.calls_to("earnings_page")) == calls


class TestEarningsPage:
    @pytest.mark.parametrize("hide", [True, False])
    @pytest.mark.parametrize("name", ["no_results", "error_page"])
    def test_missing_table_returns_none(self, name: str, hide: bool) -> None:
        """Defect (WP1/WP2, D2): an error page and a no-results page both give None."""
        with stubbed_yfinance_transport(hide_exceptions=hide) as session:
            session.route("earnings_page", name)

            assert yf.Ticker(SYMBOL).get_earnings_dates() is None

    def test_valid_table_needs_lxml(self, yf_transport: FakeSession) -> None:
        """Pin both the older missing-parser defect and the newer parsed table."""
        yf_transport.route("earnings_page", "valid_table")

        if importlib.util.find_spec("lxml") is None:
            with pytest.raises(ImportError, match="lxml"):
                yf.Ticker(SYMBOL).get_earnings_dates()
        else:
            rows = yf.Ticker(SYMBOL).get_earnings_dates()
            assert rows is not None
            assert rows.index.strftime("%Y-%m-%d %H:%M %z").tolist() == [
                "2027-01-27 16:00 -0500", "2026-10-28 16:00 -0400",
                "2026-07-29 16:00 -0400", "2026-04-29 16:00 -0400",
            ]
            assert rows["EPS Estimate"].tolist() == [3.4, 3.1, 2.9, 2.7]
            assert rows["Reported EPS"].iloc[:2].isna().all()
            assert rows["Reported EPS"].iloc[2:].tolist() == [3.05, 2.65]
