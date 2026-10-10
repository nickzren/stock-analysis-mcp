"""Step 1 WP1: provider reads, error policy, cache freshness and retries.

Acceptance R1.1-R1.12 and R1.14 (R1.13 lives in test_offline_harness.py). Reads
run through the real pinned yfinance over the stubbed transport unless a test
says otherwise.
"""

import ast
import importlib
import importlib.metadata
import json
import tomllib
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import pytz
import requests
import yfinance as yf
from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError
from curl_cffi.requests.exceptions import DNSError, IncompleteRead, Timeout
from yfinance.config import YfConfig
from yfinance.data import YfData
from yfinance.exceptions import (
    YFDataException,
    YFPricesMissingError,
    YFRateLimitError,
    YFTzMissingError,
)

import stock_analysis.data.cache_manager as cache_manager
import stock_analysis.tools.events as events
import stock_analysis.tools.fundamentals as fundamentals
import stock_analysis.tools.news as news
import stock_analysis.tools.ownership as ownership
import stock_analysis.tools.trade_setup.orchestrator as trade_orchestrator
from stock_analysis.data import provider, yfinance_client
from stock_analysis.data.cache_manager import ticker_cache
from stock_analysis.data.yfinance_client import (
    YFinanceIncompleteInfoError,
    YFinanceUnreadableResponseError,
    _is_retryable_error,
    fetch_history,
    fetch_ticker,
)
from stock_analysis.tools.symbol_search import symbol_search
from stock_analysis.utils.provenance import FetchError, fetch_or_error
from stock_analysis.utils.validators import FetchParams
from tests.support.fake_source import FakeSource
from tests.support.fake_ticker import FakeTicker, http_error
from tests.support.transport import FakeResponse, FakeSession, TransportFixture
from tests.test_tools.test_trade_card import NOW_REGULAR, blocker_ids, build_card

# tools/__init__ re-exports functions named like these modules, shadowing them.
options_signals = importlib.import_module("stock_analysis.tools.options_signals")
analyze_orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

SYMBOL = "EXMP"
REPO = Path(__file__).resolve().parent.parent
ET = pytz.timezone("America/New_York")
VERIFIED_EMPTY_EVENTS = {
    "earnings": {"next_date": None, "days_until": None, "verification": "verified_none", "sources_failed": []}
}
HISTORY_PARAMS = FetchParams(symbol=SYMBOL, period="1y", interval="1d", adjusted=True)


class _Clock:
    """Monotonic clock for cache_manager's TTL caches only (asyncio keeps the real one)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 1_000.0
        monkeypatch.setattr(cache_manager, "time", SimpleNamespace(monotonic=lambda: self.now))

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    return _Clock(monkeypatch)


@pytest.fixture
def events_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """events_calendar with no info and no splits, so only calendar/page matter."""

    async def fake_fetch_ticker(symbol: str) -> FakeTicker:
        return FakeTicker(symbol, splits=None)

    async def fake_fetch_info(symbol: str) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(events, "fetch_ticker", fake_fetch_ticker)
    monkeypatch.setattr(events, "fetch_info", fake_fetch_info)


async def _events(now: datetime = NOW_REGULAR) -> dict[str, Any]:
    return await events.events_calendar(SYMBOL, _now=now)


class _FixedNewsDateTime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> "_FixedNewsDateTime":
        now = cls(2026, 10, 9, 12, 0, tzinfo=UTC)
        return now if tz is not None else now.replace(tzinfo=None)


@pytest.fixture
def news_env(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_earnings(symbol: str) -> tuple[None, bool]:
        return None, False

    monkeypatch.setattr(news, "datetime", _FixedNewsDateTime)
    monkeypatch.setattr(news, "read_earnings_dates", no_earnings)


def _raiser(error: Exception) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        raise error

    return call


class TestEarningsVerification:
    """R1.1-R1.3, end to end: transport -> events_calendar -> trade card."""

    def test_card_fixture_is_otherwise_actionable(self) -> None:
        assert build_card(actionable_price=101.5, events_data=VERIFIED_EMPTY_EVENTS)["action"] == "trade_now"

    @pytest.mark.parametrize("name", ["http_401", "http_503", "http_404_json"])
    async def test_r1_1_failed_calendar_caps_the_card(
        self, yf_transport: FakeSession, events_env: None, name: str
    ) -> None:
        yf_transport.route("calendar", name)
        yf_transport.route("earnings_page", "error_page")

        earnings = (await _events())["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["next_date"] is None
        assert earnings["sources_failed"] == ["calendar", "earnings_dates"]
        assert card["action"] == "watch"
        assert "earnings_unverifiable" in blocker_ids(card)

    async def test_r1_2_future_page_date_resolves_a_failed_calendar(
        self, yf_transport: FakeSession, events_env: None
    ) -> None:
        yf_transport.route("calendar", "http_503")
        yf_transport.route("earnings_page", "valid_table")

        earnings = (await _events())["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["sources_failed"] == ["calendar"]
        assert (earnings["next_date"], earnings["days_until"]) == ("2026-04-29", 50)
        assert earnings["next_date_source"] == "earnings_dates"
        assert "earnings_unverifiable" not in blocker_ids(card)

    async def test_newest_first_page_still_picks_the_nearest_date(
        self, yf_transport: FakeSession, events_env: None
    ) -> None:
        """The page lists rows newest first; the earliest future row must win, so
        an earnings date 3 days out still triggers the blackout."""
        yf_transport.route("calendar", "http_503")
        yf_transport.route("earnings_page", "valid_table")

        earnings = (await _events(ET.localize(datetime(2026, 10, 25, 10, 30))))["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert (earnings["next_date"], earnings["days_until"]) == ("2026-10-28", 3)
        assert card["action"] == "watch"
        assert "earnings_blackout" in blocker_ids(card)

    async def test_r1_3_verified_sources_without_dates_do_not_cap(
        self, yf_transport: FakeSession, events_env: None
    ) -> None:
        yf_transport.route("calendar", "valid_no_earnings")
        yf_transport.route("earnings_page", "past_only_table")

        earnings = (await _events())["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["next_date"] is None
        assert earnings["sources_failed"] == []
        assert card["action"] == "trade_now"

    @pytest.mark.parametrize("page", ["no_results", "malformed_row"])
    async def test_page_that_cannot_prove_absence_does_not_verify(
        self, yf_transport: FakeSession, events_env: None, page: str
    ) -> None:
        """D2: no table (could be an error page) and a page with an unparseable row
        (it could be the nearest date) both fail to verify."""
        yf_transport.route("calendar", "valid_no_earnings")
        yf_transport.route("earnings_page", page)

        earnings = (await _events(ET.localize(datetime(2026, 10, 26, 10, 30))))["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["sources_failed"] == ["earnings_dates"]
        assert card["action"] == "watch"
        assert "earnings_unverifiable" in blocker_ids(card)

    @pytest.mark.parametrize(
        "name",
        [
            "malformed_earnings_field",
            # Falsy malformed values must not be defaulted to an empty list.
            "earnings_date_zero",
            "earnings_date_false",
            "earnings_date_empty_string",
            "earnings_date_object",
        ],
    )
    async def test_malformed_earnings_field_does_not_verify(
        self, yf_transport: FakeSession, events_env: None, name: str
    ) -> None:
        """A malformed nested earnings field must not read as a verified absence."""
        yf_transport.route("calendar", name)
        yf_transport.route("earnings_page", "past_only_table")

        earnings = (await _events())["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["sources_failed"] == ["calendar"]
        assert card["action"] == "watch"
        assert "earnings_unverifiable" in blocker_ids(card)

    @pytest.mark.parametrize(
        "name", ["valid_no_earnings", "earnings_without_dates", "earnings_dates_empty_list", "earnings_dates_null"]
    )
    async def test_missing_earnings_dates_verify_as_absent(
        self, yf_transport: FakeSession, events_env: None, name: str
    ) -> None:
        """Positive controls: an absent or null field, or an empty list, is a real absence."""
        yf_transport.route("calendar", name)
        yf_transport.route("earnings_page", "past_only_table")

        earnings = (await _events())["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert earnings["sources_failed"] == []
        assert card["action"] == "trade_now"

    async def test_calendar_date_flows_through_the_adapter(
        self, yf_transport: FakeSession, events_env: None, host_tz: Callable[[str], None]
    ) -> None:
        host_tz("America/New_York")
        yf_transport.route("calendar", "valid_one_epoch")
        yf_transport.route("earnings_page", "past_only_table")

        earnings = (await _events(ET.localize(datetime(2026, 10, 6, 12, 0))))["earnings"]
        card = build_card(actionable_price=101.5, events_data={"earnings": earnings})

        assert (earnings["next_date"], earnings["days_until"]) == ("2026-10-08", 2)
        assert (earnings["next_date_source"], earnings["sources_failed"]) == ("calendar", [])
        assert "earnings_blackout" in blocker_ids(card)

    async def test_page_columns_map_into_history(self, yf_transport: FakeSession, events_env: None) -> None:
        yf_transport.route("calendar", "valid_no_earnings")
        yf_transport.route("earnings_page", "valid_table")

        history = (await _events())["earnings"]["history"]

        row = next(h for h in history if h["date"] == "2026-07-29")
        assert (row["estimate"], row["actual"]) == (2.9, 3.05)


class TestCalendarDatesThroughTransport:
    """WP2 ends deviation (b): calendar dates no longer follow the host time zone."""

    @pytest.mark.parametrize(("tz", "expected"), [("UTC", date(2026, 10, 9)), ("America/New_York", date(2026, 10, 8))])
    def test_ticker_calendar_property_is_host_local(
        self, yf_transport: FakeSession, host_tz: Callable[[str], None], tz: str, expected: date
    ) -> None:
        """Characterization (R2.6): why events reads raw epochs, not Ticker.calendar."""
        host_tz(tz)
        yf_transport.route("calendar", "valid_one_epoch")

        assert yf.Ticker(SYMBOL).calendar["Earnings Date"] == [expected]

    @pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
    async def test_window_dates_are_et(
        self, yf_transport: FakeSession, events_env: None, host_tz: Callable[[str], None], tz: str
    ) -> None:
        host_tz(tz)
        yf_transport.route("calendar", "valid_window")
        yf_transport.route("earnings_page", "past_only_table")

        earnings = (await _events(ET.localize(datetime(2026, 10, 6, 12, 0))))["earnings"]

        assert (earnings["next_date"], earnings["days_until"]) == ("2026-10-27", 21)
        window = earnings["next_window"]
        assert (window["start"], window["end"], window["is_estimate"]) == ("2026-10-27", "2026-10-31", True)
        # No info range to confirm the two entries are an interval.
        assert (window["window_basis"], earnings["verification"]) == ("assumed", "unverifiable")

    @pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
    async def test_dividend_dates_are_utc_and_earnings_dates_et(
        self, yf_transport: FakeSession, events_env: None, host_tz: Callable[[str], None], tz: str
    ) -> None:
        host_tz(tz)
        yf_transport.route("calendar", "valid_dividends")
        yf_transport.route("earnings_page", "past_only_table")

        result = await _events(ET.localize(datetime(2026, 10, 6, 12, 0)))

        assert (result["dividends"]["ex_date"], result["dividends"]["pay_date"]) == ("2026-09-15", "2026-10-09")
        assert result["earnings"]["next_date"] == "2026-10-08"


class TestSourceReads:
    async def test_r1_4_calendar_recovers_after_a_malformed_response(self, yf_transport: FakeSession) -> None:
        yf_transport.route("calendar", "malformed_missing_calendar_events")
        first = await provider.read_calendar_events(SYMBOL)
        yf_transport.route("calendar", "valid_one_epoch")
        second = await provider.read_calendar_events(SYMBOL)

        assert (first.status, first.error_type, first.attempts) == ("failed", "malformed", 1)
        assert second.status == "ok"
        assert second.payload["earnings"]["earningsDate"] == [1791505800]
        assert len(yf_transport.calls_to("calendar")) == 2

    async def test_r1_5_earnings_page_recovers_after_a_503(self, yf_transport: FakeSession) -> None:
        yf_transport.route("earnings_page", "error_page")
        first = await provider.read_earnings_page(SYMBOL)
        calls = len(yf_transport.calls_to("earnings_page"))
        yf_transport.route("earnings_page", "valid_table")
        second = await provider.read_earnings_page(SYMBOL)

        assert (first.status, first.error_type, first.attempts) == ("failed", "http_503", 3)
        assert second.status == "ok"
        assert len(yf_transport.calls_to("earnings_page")) == calls + 1

    @pytest.mark.parametrize(
        ("name", "status", "error_type"),
        [
            ("valid_one_epoch", "ok", None),
            ("valid_no_earnings", "empty", None),
            ("http_404_json", "failed", "http_404"),
            ("http_429", "failed", "rate_limited"),
            ("malformed_earnings_field", "failed", "malformed"),
            ("malformed_earnings_dates", "failed", "malformed"),
            ("malformed_dividend_date", "failed", "malformed"),
        ],
    )
    async def test_calendar_statuses(
        self, yf_transport: FakeSession, name: str, status: str, error_type: str | None
    ) -> None:
        yf_transport.route("calendar", name)

        read = await provider.read_calendar_events(SYMBOL)

        assert (read.status, read.error_type) == (status, error_type)

    async def test_earnings_page_rows_keep_page_order_in_et(self, yf_transport: FakeSession) -> None:
        yf_transport.route("earnings_page", "valid_table")

        rows = (await provider.read_earnings_page(SYMBOL)).payload

        assert [row["timestamp"].date().isoformat() for row in rows] == [
            "2027-01-27", "2026-10-28", "2026-07-29", "2026-04-29",
        ]
        assert str(rows[1]["timestamp"].tzinfo) == "America/New_York"
        assert rows[1]["timestamp"].hour == 16
        assert (rows[1]["reported_eps"], rows[2]["reported_eps"]) == (None, 3.05)

    @pytest.mark.parametrize(
        ("name", "status", "error_type", "rejected"),
        [
            ("no_results", "unverified", "no_table", 0),
            ("missing_columns", "failed", "malformed", 0),
            ("malformed_row", "failed", "malformed", 1),
        ],
    )
    async def test_earnings_page_without_a_usable_table(
        self, yf_transport: FakeSession, name: str, status: str, error_type: str, rejected: int
    ) -> None:
        yf_transport.route("earnings_page", name)

        read = await provider.read_earnings_page(SYMBOL)

        assert (read.status, read.error_type, read.rejected_count) == (status, error_type, rejected)

    async def test_reads_bypass_the_ticker_and_lru_caches(self, yf_transport: FakeSession) -> None:
        yf_transport.route("search", "valid_news")

        await provider.read_search_news(SYMBOL)

        assert YfData.cache_get.cache_info().currsize == 0


class TestFreshness:
    """R1.6: after an application TTL expires, the next read does fresh I/O."""

    def test_source_ttls(self) -> None:
        assert provider.SOURCE_TTL_SECONDS == {
            "calendar": 3600, "earnings_page": 21600, "news": 900, "search": 900, "symbol_search": 3600,
        }

    @pytest.mark.parametrize(
        ("read", "source", "fixture"),
        [
            (provider.read_calendar_events, "calendar", "valid_one_epoch"),
            (provider.read_earnings_page, "earnings_page", "valid_table"),
            (provider.read_news_feed, "news", "valid_stream"),
            (provider.read_search_news, "search", "valid_news"),
        ],
    )
    async def test_source_expires_exactly_at_its_ttl(
        self, yf_transport: FakeSession, clock: _Clock, read: Any, source: str, fixture: str
    ) -> None:
        yf_transport.route(source, fixture)
        ttl = provider.SOURCE_TTL_SECONDS[source]
        first = await read(SYMBOL)

        clock.advance(ttl - 1)
        assert await read(SYMBOL) is first
        clock.advance(2)
        assert await read(SYMBOL) is not first
        assert len(yf_transport.calls_to(source)) == 2

    async def test_expired_source_returns_updated_data(self, yf_transport: FakeSession, clock: _Clock) -> None:
        yf_transport.route("calendar", "valid_one_epoch")
        await provider.read_calendar_events(SYMBOL)
        yf_transport.route("calendar", "valid_window")

        clock.advance(provider.SOURCE_TTL_SECONDS["calendar"] + 1)
        fresh = await provider.read_calendar_events(SYMBOL)

        assert fresh.payload["earnings"]["earningsDate"] == [1793131200, 1793476800]

    @pytest.mark.parametrize(
        ("source", "name", "read"),
        [
            ("calendar", "http_503", provider.read_calendar_events),
            ("calendar", "malformed_missing_calendar_events", provider.read_calendar_events),
            ("earnings_page", "no_results", provider.read_earnings_page),
        ],
    )
    async def test_failed_and_unverified_reads_are_never_cached(
        self, yf_transport: FakeSession, source: str, name: str, read: Any
    ) -> None:
        yf_transport.route(source, name)
        await read(SYMBOL)
        calls = len(yf_transport.calls_to(source))

        await read(SYMBOL)

        assert provider._source_cache.size() == 0
        assert len(yf_transport.calls_to(source)) > calls

    async def test_statement_read_on_a_replacement_ticker_is_fresh(
        self, yf_transport: FakeSession, clock: _Clock
    ) -> None:
        """A statement read goes through yfinance's day-scoped cache_get LRU; a
        replacement Ticker made after the ticker TTL must not see the old response."""
        ticker_cache.clear()
        yf_transport.route("timeseries", "quarterly_revenue_100")
        first = await fetch_ticker(SYMBOL)
        assert first.quarterly_income_stmt.iloc[0, 0] == 100
        yf_transport.route("timeseries", "quarterly_revenue_200")

        # Without expiry, a new Ticker in the same UTC day reads the cached 100.
        assert yf.Ticker(SYMBOL).quarterly_income_stmt.iloc[0, 0] == 100
        assert await fetch_ticker(SYMBOL) is first  # within the ticker TTL
        clock.advance(cache_manager.get_ttl(cache_manager.DataType.TICKER) + 1)
        replacement = await fetch_ticker(SYMBOL)

        assert replacement is not first
        assert replacement.quarterly_income_stmt.iloc[0, 0] == 200
        assert len(yf_transport.calls_to("timeseries")) == 2
        ticker_cache.clear()

    async def test_info_refetch_expires_the_response_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expired: list[bool] = []
        monkeypatch.setattr(yfinance_client, "expire_yfinance_response_cache", lambda: expired.append(True))
        monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda symbol: FakeTicker(symbol, info={"x": 1}))
        monkeypatch.setattr(yfinance_client, "assess_info_completeness", _complete)

        await yfinance_client._fetch_info_raw(SYMBOL)

        assert expired == [True]


def _complete(info: dict[str, Any]) -> yfinance_client.InfoCompleteness:
    return yfinance_client.InfoCompleteness(False, len(info), "EQUITY", 1, 1)


class TestPolicyAndDependency:
    def test_r1_7_errors_are_not_hidden(self) -> None:
        assert YfConfig.debug.hide_exceptions is False

    def test_r1_7_policy_is_assigned_exactly_once(self) -> None:
        sites = []
        for path in sorted((REPO / "src").rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                targets = node.targets if isinstance(node, ast.Assign) else []
                sites += [
                    (path.relative_to(REPO).as_posix(), node.lineno)
                    for target in targets
                    if isinstance(target, ast.Attribute) and target.attr == "hide_exceptions"
                ]

        assert [file for file, _ in sites] == ["src/stock_analysis/data/yfinance_client.py"]

    def test_r1_12_yfinance_is_in_the_supported_range(self) -> None:
        major = int(importlib.metadata.version("yfinance").split(".")[0])
        dependencies = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]

        assert major == 1
        assert "yfinance>=1.0,<2" in dependencies
        assert callable(YfData.get) and callable(YfData.post)
        assert yf.Ticker(SYMBOL)._data is YfData()
        assert callable(YfData.cache_get.cache_clear)


class TestNewsReads:
    """R1.8 / R1.9 against the stubbed transport."""

    @pytest.mark.parametrize(
        ("name", "status", "error_type"),
        [
            ("http_404_json", "failed", "http_404"),
            ("malformed_envelope", "failed", "malformed"),
            ("valid_empty_stream", "empty", None),
            ("valid_stream", "ok", None),
            ("null_item", "failed", "malformed"),
        ],
    )
    async def test_r1_8_news_feed(
        self, yf_transport: FakeSession, name: str, status: str, error_type: str | None
    ) -> None:
        yf_transport.route("news", name)

        read = await provider.read_news_feed(SYMBOL)

        assert (read.status, read.error_type) == (status, error_type)

    @pytest.mark.parametrize(
        ("name", "status", "error_type"),
        [
            ("http_404_json", "failed", "http_404"),
            ("missing_news_key", "failed", "malformed"),
            ("valid_empty_news", "empty", None),
            ("valid_news", "ok", None),
            ("null_item", "failed", "malformed"),
        ],
    )
    async def test_r1_9_search(
        self, yf_transport: FakeSession, name: str, status: str, error_type: str | None
    ) -> None:
        yf_transport.route("search", name)

        read = await provider.read_search_news(SYMBOL)

        assert (read.status, read.error_type) == (status, error_type)

    @pytest.mark.parametrize(
        ("feed", "search", "status", "source"),
        [
            ("valid_stream", "http_404_json", "ok", "news"),
            ("http_404_json", "valid_news", "ok", "search"),
            ("valid_empty_stream", "valid_news", "ok", "search"),
            ("valid_empty_stream", "valid_empty_news", "empty", "news"),
            ("http_404_json", "http_404_json", "failed", "news"),
            ("valid_empty_stream", "http_404_json", "unverified", "news"),
        ],
    )
    async def test_feed_falls_back_to_search(
        self, yf_transport: FakeSession, feed: str, search: str, status: str, source: str
    ) -> None:
        yf_transport.route("news", feed)
        yf_transport.route("search", search)

        read = await provider.read_news(SYMBOL)

        assert (read.status, read.source) == (status, source)

    async def test_search_items_become_articles(
        self, yf_transport: FakeSession, news_env: None, host_tz: Callable[[str], None]
    ) -> None:
        # A host zone this far east would move a host-local date to 2026-10-09.
        host_tz("Pacific/Kiritimati")
        yf_transport.route("news", "http_404_json")
        yf_transport.route("search", "valid_news")

        result = await news.stock_news(SYMBOL, days=7)

        assert (result["source_status"], result["source"]) == ("ok", "search")
        article = result["articles"][0]
        assert (article["title"], article["provider"]) == ("Example Corp news via search", "Example Wire")
        assert (article["url"], article["date"]) == ("https://example.com/news/s1", "2026-10-08")

    async def test_out_of_range_search_epoch_is_rejected(self, yf_transport: FakeSession, news_env: None) -> None:
        yf_transport.route("news", "http_404_json")
        yf_transport.route("search", "out_of_range_epoch")

        result = await news.stock_news(SYMBOL, days=7)

        assert (result["source_status"], result["rejected_count"], result["article_count"]) == ("ok", 1, 1)

    async def test_all_items_rejected_is_ok_with_a_warning(self, yf_transport: FakeSession, news_env: None) -> None:
        yf_transport.route("news", "all_rejected")  # one ad, one item without a pubDate

        result = await news.stock_news(SYMBOL, days=7)

        assert (result["source_status"], result["rejected_count"]) == ("ok", 2)
        assert result["warnings"] == [
            "all_items_rejected: no feed item was a usable story (ad or missing publish date)"
        ]

    async def test_malformed_search_does_not_read_as_a_quiet_feed(
        self, yf_transport: FakeSession, news_env: None
    ) -> None:
        yf_transport.route("news", "valid_empty_stream")
        yf_transport.route("search", "null_item")

        result = await news.stock_news(SYMBOL, days=7)

        assert result["source_status"] == "unverified"
        assert result["warnings"] == ["News feed could not be verified (malformed)"]

    async def test_failed_feed_does_not_read_as_a_quiet_period(self, yf_transport: FakeSession, news_env: None) -> None:
        yf_transport.route("news", "http_404_json")
        yf_transport.route("search", "http_404_json")

        result = await news.stock_news(SYMBOL, days=7)

        assert result["source_status"] == "failed"
        assert result["warnings"] == ["News feed unavailable (http_404)"]

    def test_items_with_null_fields_are_parsed_safely(self) -> None:
        items = [
            {"content": {"pubDate": "2026-10-08T12:00:00Z", "title": None, "provider": None, "canonicalUrl": None}},
            {"content": None},
            "not-a-dict",
        ]

        articles, rejected = news._parse_articles(items, datetime(2026, 10, 1))  # type: ignore[arg-type]

        assert [(a["title"], a["provider"], a["url"]) for a in articles] == [("", "Unknown", None)]
        assert rejected == 2


class TestAnalyzeSeesNewsHealth:
    """A failed feed is a normal stock_news result, so analyze must surface it."""

    @pytest.fixture
    def analyze_tools(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "name": "Example", "current_price": 100.0, "currency": "USD"}

        for name in ("technicals", "fundamentals_snapshot", "risk_metrics", "events_calendar",
                     "ownership_analysis", "options_signals"):
            monkeypatch.setattr(analyze_orchestrator, name, empty)
        monkeypatch.setattr(analyze_orchestrator, "stock_summary", summary)

        def with_news(status: str) -> None:
            async def stock_news(symbol: str, **kwargs: Any) -> dict[str, Any]:
                return {"symbol": symbol, "source_status": status, "warnings": [f"news {status}"],
                        "articles": [], "article_count": 0}

            monkeypatch.setattr(analyze_orchestrator, "stock_news", stock_news)

        return with_news

    @pytest.mark.parametrize(
        ("status", "failure", "covered", "gap"),
        [("failed", True, False, False), ("unverified", False, False, True), ("ok", False, True, False)],
    )
    async def test_status_reaches_tool_failures_and_coverage(
        self, analyze_tools: Callable[[str], None], status: str, failure: bool, covered: bool, gap: bool
    ) -> None:
        analyze_tools(status)

        result = await analyze_orchestrator.analyze_stock(SYMBOL)

        failures = [f for f in result["data_quality"]["tool_failures"] if f["tool"] == "stock_news"]
        assert bool(failures) is failure
        assert result["verdict"]["coverage"]["news"]["fetched"] is covered
        assert ("news_unverified" in result["data_quality"]["data_gaps"]) is gap


class TestRetries:
    """R1.10."""

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (http_error(401), (True, 1)),
            (http_error(404), (False, 0)),
            (http_error(429), (True, 3)),
            (http_error(500), (True, 3)),
            (http_error(503), (True, 3)),
            (YFRateLimitError(), (True, 3)),
            (Timeout("timed out"), (True, 3)),
            (CurlConnectionError("refused"), (True, 3)),
            (DNSError("could not resolve host"), (True, 3)),
            (IncompleteRead("partial transfer"), (True, 3)),
            (YFinanceUnreadableResponseError("html body"), (True, 3)),
            (requests.HTTPError(response=SimpleNamespace(status_code=503)), (True, 3)),
            (YFinanceIncompleteInfoError(SYMBOL, key_count=1, quote_type=None,
                                         sentinel_keys_present=0, sentinel_values_present=0), (True, 2)),
            (ValueError("bad input"), (False, 0)),
        ],
    )
    def test_classification(self, error: Exception, expected: tuple[bool, int]) -> None:
        assert _is_retryable_error(error) == expected

    @pytest.mark.parametrize(
        ("name", "attempts"),
        # 5xx/429 stop at three attempts: a fourth backoff would exceed the read's 4 s budget.
        [("http_401", 2), ("http_503", 3), ("http_404_json", 1), ("http_429", 3)],
    )
    async def test_provider_attempt_counts(self, yf_transport: FakeSession, name: str, attempts: int) -> None:
        yf_transport.route("calendar", name)

        assert (await provider.read_calendar_events(SYMBOL)).attempts == attempts

    async def test_retryable_error_then_success_is_ok(self) -> None:
        outcomes: list[Any] = [Timeout("timed out"), FakeResponse(TransportFixture(200, '{"news": []}', "application/json"), "u")]

        def fetch() -> Any:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        read = await provider._read("search", "search:retry", fetch, provider._interpret_search)

        assert (read.status, read.attempts) == ("empty", 2)

    async def test_client_retry_classifies_curl_errors(self) -> None:
        outcomes: list[Any] = [http_error(503), "ok"]

        def call() -> Any:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        result = await yfinance_client._retry_with_backoff("probe", call)

        assert (result.result, result.attempts) == ("ok", 2)

    async def test_each_retry_does_fresh_io(self, yf_transport: FakeSession) -> None:
        """A retry must not replay a response yfinance cached in cache_get."""
        url = "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/EXMP?symbol=EXMP"
        # Two 503s: yfinance re-requests once on >= 400 inside one attempt.
        yf_transport.route("timeseries", "http_503", "http_503", "quarterly_revenue_100")

        def read() -> int:
            response = YfData().cache_get(url)
            response.raise_for_status()
            return int(response.status_code)

        result = await yfinance_client._retry_with_backoff("probe", read)

        assert (result.result, result.attempts) == (200, 2)
        assert len(yf_transport.calls_to("timeseries")) == 3

    async def test_rate_limit_cooldown_delays_other_reads(
        self, yf_transport: FakeSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waits: list[float] = []

        async def record(delay: float) -> None:
            waits.append(delay)

        monkeypatch.setattr(yfinance_client, "_sleep", record)
        yf_transport.route("calendar", "http_429")
        await provider.read_calendar_events(SYMBOL)
        waits.clear()

        yf_transport.route("search", "valid_news")
        await provider.read_search_news(SYMBOL)
        await yfinance_client._retry_with_backoff("probe", lambda: "ok")

        assert len(waits) == 2 and all(wait > 0 for wait in waits)

    @pytest.mark.parametrize(
        "error", [YFPricesMissingError(SYMBOL, ""), YFTzMissingError(SYMBOL), http_error(404)]
    )
    async def test_missing_symbol_history_stays_invalid_symbol(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        """Includes the 404 from yfinance's time-zone probe falling back to quoteSummary."""
        monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda symbol: FakeTicker(symbol, history=_raiser(error)))

        with pytest.raises(FetchError) as raised:
            await fetch_or_error(fetch_history(HISTORY_PARAMS), SYMBOL)

        assert raised.value.response["error_type"] == "invalid_symbol"

    @pytest.mark.parametrize(
        "error", [json.JSONDecodeError("Expecting value", "<html>", 0), YFDataException("Yahoo is down")]
    )
    async def test_unreadable_history_body_is_retried_not_invalid(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        expired: list[bool] = []
        monkeypatch.setattr(yfinance_client, "expire_yfinance_response_cache", lambda: expired.append(True))
        monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda symbol: FakeTicker(symbol, history=_raiser(error)))

        with pytest.raises(FetchError) as raised:
            await fetch_or_error(fetch_history(HISTORY_PARAMS), SYMBOL)

        assert raised.value.response["error_type"] == "data_unavailable"
        # Once per failed attempt in the history read, plus before each retry.
        attempts = yfinance_client._max_retries + 1
        assert len(expired) == attempts + (attempts - 1)


class TestInfoFallback:
    """An info quoteSummary 404 falls back to the v7 quote, as yfinance's hidden mode did."""

    async def test_v7_entry_is_used(self, yf_transport: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda symbol: FakeTicker(symbol, info=http_error(404)))
        yf_transport.route("quote_v7", "etf_entry")

        info, _ = await yfinance_client._fetch_info_raw(SYMBOL)

        assert (info["quoteType"], info["regularMarketPrice"]) == ("ETF", 50.0)
        assert "trailingPE" not in info

    async def test_no_v7_entry_is_invalid_symbol(self, yf_transport: FakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda symbol: FakeTicker(symbol, info=http_error(404)))
        yf_transport.route("quote_v7", "empty_result")

        with pytest.raises(ValueError, match="Invalid symbol"):
            await yfinance_client._fetch_info_raw(SYMBOL)


class TestSymbolSearch:
    @pytest.mark.parametrize("name", ["http_503_html", "http_404_json", "null_quote"])
    async def test_upstream_failure_is_a_structured_error(self, yf_transport: FakeSession, name: str) -> None:
        yf_transport.route("search", name)

        result = await symbol_search("Example Corp")

        assert (result["error"], result["error_type"]) == (True, "data_unavailable")

    async def test_recovers_after_a_failed_search(self, yf_transport: FakeSession) -> None:
        """A failed search is not cached anywhere, so the next call sees recovery."""
        yf_transport.route("search", "http_404_json")
        await symbol_search("Example Corp")
        yf_transport.route("search", "valid_news")

        result = await symbol_search("Example Corp")

        assert [r["symbol"] for r in result["results"]] == ["EXMP"]
        assert YfData.cache_get.cache_info().currsize == 0

    async def test_analyze_explains_a_search_outage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def failed_search(query: str, limit: int = 10) -> dict[str, Any]:
            return {"error": True, "error_type": "data_unavailable", "message": "Symbol search failed upstream"}

        monkeypatch.setattr(analyze_orchestrator, "symbol_search", failed_search)

        with pytest.raises(ValueError, match="symbol search unavailable"):
            await analyze_orchestrator._resolve_symbol_for_analysis("Example Corp")


class TestPropertyReadSites:
    """R1.11 + R3.8: each Ticker property read runs as executor jobs, retries retryable
    errors, becomes a SourceRead, keeps its site's degraded output and evicts the Ticker."""

    async def test_failed_read_is_retried_converted_and_evicts(self) -> None:
        ticker = FakeTicker(SYMBOL, splits=http_error(503))
        ticker_cache.set(SYMBOL, ticker, 3600)

        read = await provider.read_property(ticker, "splits", lambda t: t.splits)

        assert (read.status, read.error_type, read.retryable) == ("failed", "http_503", True)
        assert read.attempts == ticker.reads["splits"] == 3
        assert ticker_cache.get(SYMBOL) is None

    async def test_an_error_status_hidden_behind_a_parse_error_is_retried(self, yf_transport: FakeSession) -> None:
        """yfinance parses a JSON 503 body instead of raising; its status still drives
        the retry, so the healthy second response is used."""
        # yfinance re-requests once on >= 400 inside one attempt.
        yf_transport.route("timeseries", "http_503_json", "http_503_json", "quarterly_revenue_100")
        ticker = yf.Ticker(SYMBOL)

        read = await provider.read_property(ticker, "quarterly_income_stmt", lambda t: t.quarterly_income_stmt)

        assert (read.status, read.attempts) == ("ok", 2)
        assert read.payload.iloc[0, 0] == 100

    @pytest.mark.parametrize(
        ("fixture", "error_type"),
        [("http_404_html", "http_404"), ("unparseable_200", "unreadable_response")],
    )
    async def test_a_parse_error_without_a_retryable_status_fails_once(
        self, yf_transport: FakeSession, fixture: str, error_type: str
    ) -> None:
        yf_transport.route("timeseries", fixture)
        ticker = yf.Ticker(SYMBOL)

        read = await provider.read_property(ticker, "quarterly_income_stmt", lambda t: t.quarterly_income_stmt)

        assert (read.status, read.error_type, read.attempts) == ("failed", error_type, 1)

    async def test_non_retryable_error_fails_once(self) -> None:
        ticker = FakeTicker(SYMBOL, splits=http_error(404))

        read = await provider.read_property(ticker, "splits", lambda t: t.splits)

        assert (read.status, read.error_type, read.attempts) == ("failed", "http_404", 1)

    @pytest.mark.parametrize("value", [None, pd.DataFrame(), pd.Series(dtype=float), ()])
    async def test_empty_value_is_unverified(self, value: Any) -> None:
        read = await provider.read_property(FakeTicker(SYMBOL), "holders", lambda t: value)

        assert (read.status, read.error_type) == ("unverified", "empty_property")

    async def test_fundamentals_reads_map_a_failure_to_none(self) -> None:
        income = pd.DataFrame({pd.Timestamp("2026-06-30"): [1.0]}, index=["Net Income"])
        ticker = FakeTicker(
            SYMBOL,
            get_cashflow=lambda freq=None: pd.DataFrame(),
            quarterly_income_stmt=income,
            history=_raiser(http_error(503)),
            earnings_estimate=pd.DataFrame(),
            revenue_estimate=pd.DataFrame(),
            dividends=pd.Series(dtype=float),
        )
        ticker_cache.set(SYMBOL, ticker, 3600)

        statements = await fundamentals._read_statements(ticker)

        assert statements["price_history_5y"] is None
        assert statements["quarterly_income_stmt"] is income
        assert ticker_cache.get(SYMBOL) is None

    @pytest.mark.parametrize(
        ("call", "degraded"),
        [
            (lambda: fundamentals._compute_valuation_history(None, pd.DataFrame({"Close": [1.0]}),
                                                             {"sharesOutstanding": 1e6}), None),
            (lambda: fundamentals._compute_valuation_history(
                pd.DataFrame({pd.Timestamp("2026-06-30"): [1.0]}, index=["Net Income"]), None,
                {"sharesOutstanding": 1e6}), None),
            (lambda: fundamentals._compute_fundamental_trends(None), None),
            (lambda: fundamentals._build_earnings_estimates(None, pd.DataFrame(), {}), None),
            (lambda: fundamentals._build_earnings_estimates(pd.DataFrame(), None, {}), None),
            (lambda: fundamentals._analyze_dividend_history(None, {}), None),
            (lambda: fundamentals._resolve_cashflow(None, None, 1.0, 2.0, "FY 2025"),
             {"operating_cf_ttm": 1.0, "operating_cf_period": "TTM", "operating_cf_period_end": None,
              "operating_cf_source": "info", "operating_cf_ttm_status": "unavailable", "operating_cf_period_lag": None,
              "free_cash_flow_ttm": 2.0, "free_cash_flow_period": "TTM", "free_cash_flow_period_end": None,
              "free_cash_flow_source": "info", "free_cash_flow_ttm_status": "unavailable",
              "free_cash_flow_period_lag": None}),
            (lambda: events._build_splits(None), {"last_date": None, "last_ratio": None}),
        ],
    )
    def test_failed_input_keeps_the_degraded_output(self, call: Callable[[], Any], degraded: Any) -> None:
        assert call() == degraded

    @pytest.mark.parametrize(
        ("failing", "warning"),
        [
            ("insider_transactions", "insider_transactions_error"),
            ("institutional_holders", "institutional_holders_error"),
            ("major_holders", "major_holders_error"),
        ],
    )
    async def test_ownership_tool_degrades_and_evicts(
        self, monkeypatch: pytest.MonkeyPatch, failing: str, warning: str
    ) -> None:
        attributes: dict[str, Any] = {
            "insider_transactions": pd.DataFrame(),
            "institutional_holders": pd.DataFrame(),
            "major_holders": pd.DataFrame(),
            failing: http_error(503),
        }
        ticker = FakeTicker(SYMBOL, **attributes)
        ticker_cache.set(SYMBOL, ticker, 3600)

        async def fake_fetch_ticker(symbol: str) -> FakeTicker:
            return ticker

        monkeypatch.setattr(ownership, "fetch_ticker", fake_fetch_ticker)

        result = await ownership.ownership_analysis(SYMBOL)

        assert warning in result["warnings"]
        assert ticker_cache.get(SYMBOL) is None

    @pytest.mark.parametrize(
        "attributes",
        [{"options": http_error(503)}, {"options": ("2026-11-20",), "option_chain": _raiser(http_error(503))}],
        ids=["expirations", "chain"],
    )
    async def test_options_tool_reports_a_failed_read_and_evicts(
        self, monkeypatch: pytest.MonkeyPatch, attributes: dict[str, Any]
    ) -> None:
        ticker = FakeTicker(SYMBOL, **attributes)
        ticker_cache.set(SYMBOL, ticker, 3600)

        async def fake_fetch_ticker(symbol: str) -> FakeTicker:
            return ticker

        async def fake_info(symbol: str) -> dict[str, Any]:
            return {"regularMarketPrice": 50.0}

        monkeypatch.setattr(options_signals, "fetch_ticker", fake_fetch_ticker)
        monkeypatch.setattr(options_signals, "_fetch_info_or_empty", fake_info)
        monkeypatch.setattr(options_signals, "_pick_expiration", lambda expirations: expirations[0])

        result = await options_signals.options_signals(SYMBOL)

        assert result["error_type"] == "data_unavailable"
        assert ticker_cache.get(SYMBOL) is None

    @pytest.mark.parametrize(
        "attributes",
        [{"options": http_error(503)}, {"options": ("2026-10-30",), "option_chain": _raiser(http_error(503))}],
    )
    async def test_expected_move_degrades_and_evicts(self, attributes: dict[str, Any]) -> None:
        ticker = FakeTicker(SYMBOL, **attributes)
        ticker_cache.set(SYMBOL, ticker, 3600)

        result = await trade_orchestrator._fetch_expected_move(
            SYMBOL, "2026-10-28", datetime(2026, 10, 20, tzinfo=UTC), 100.0
        )

        assert result is None
        assert ticker_cache.get(SYMBOL) is None

    async def test_eviction_spares_a_newer_ticker(self) -> None:
        failed = FakeTicker(SYMBOL, splits=http_error(503))
        newer = FakeTicker(SYMBOL, splits=pd.Series(dtype=float))
        ticker_cache.set(SYMBOL, newer, 3600)

        await provider.read_property(failed, "splits", lambda t: t.splits)

        assert ticker_cache.get(SYMBOL) is newer
        ticker_cache.clear()

    async def test_successful_read_keeps_the_ticker(self) -> None:
        ticker = FakeTicker(SYMBOL, splits=pd.Series([2.0], index=pd.DatetimeIndex(["2024-06-10"])))
        ticker_cache.set(SYMBOL, ticker, 3600)

        read = await provider.read_property(ticker, "splits", lambda t: t.splits)

        assert read.status == "ok"
        assert ticker_cache.get(SYMBOL) is ticker
        ticker_cache.clear()


class TestFakeSource:
    """R1.14: FakeSource returns each result and drives a tool."""

    @pytest.mark.parametrize(
        ("make_source", "status", "warnings"),
        [
            (lambda: FakeSource.ok("news", [{"content": {"pubDate": "2026-10-08T12:00:00Z", "title": "Up"}}]), "ok", None),
            (lambda: FakeSource.empty("news", []), "empty", ["No news articles found in the past 7 days"]),
            (lambda: FakeSource.failed("news", "http_503"), "failed", ["News feed unavailable (http_503)"]),
            (lambda: FakeSource.unverified("news", "mixed"), "unverified", ["News feed could not be verified (mixed)"]),
        ],
    )
    async def test_each_result_drives_the_news_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        news_env: None,
        make_source: Callable[[], FakeSource],
        status: str,
        warnings: list[str] | None,
    ) -> None:
        source = make_source()
        monkeypatch.setattr(news, "read_news", source)

        result = await news.stock_news(SYMBOL, days=7)

        assert (result["source_status"], result["warnings"]) == (status, warnings)
        assert source.calls == [SYMBOL]

    async def test_drives_events(self, monkeypatch: pytest.MonkeyPatch, events_env: None) -> None:
        monkeypatch.setattr(events, "read_calendar_events", FakeSource.failed("calendar"))
        monkeypatch.setattr(events, "read_earnings_page", FakeSource.empty("earnings_page", []))

        earnings = (await _events())["earnings"]

        assert earnings["sources_failed"] == ["calendar"]
