"""Network reads that must tell a failed fetch apart from a valid empty result.

Step 1 WP1. These sources are read below yfinance's properties, through its shared
session (``YfData``), so the HTTP status and the response envelope can be checked:
quoteSummary ``calendarEvents``, the earnings page, the news feed (``/xhr/ncp``),
Search news and Search quotes (symbol search). They never go through ``YfData.cache_get`` or ``Ticker`` attribute
caches. Each read returns a ``SourceRead``; only ``ok`` and ``empty`` results are
cached, for ``SOURCE_TTL_SECONDS``. Property reads on a cached ``Ticker`` go through
``read_property``, which holds the symbol's lock and evicts the Ticker on failure.

WP3: every attempt is one executor job (``jobs.run_job``) that owns its admission
permit (and, for property reads, the symbol lock) until its thread finishes; a
read runs within one ``jobs.ReadClock``; concurrent reads of one source share a
single in-flight fetch (``jobs.singleflight``).
"""

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeGuard, TypeVar
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError
from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError
from curl_cffi.requests.exceptions import Timeout as CurlTimeout
from yfinance.data import YfData
from yfinance.exceptions import YFDataException

from stock_analysis.data import jobs
from stock_analysis.data.cache_manager import TTLCache
from stock_analysis.data.yfinance_client import (
    _calculate_backoff,
    _is_retryable_error,
    _max_retries,
    evict_ticker,
    expire_yfinance_response_cache,
    http_status,
    is_rate_limit,
    note_rate_limit,
    rate_limit_cooldown_remaining,
    wait_for_rate_limit_cooldown,
)
from stock_analysis.utils.provenance import utcnow_isoformat_z

T = TypeVar("T")
SourceStatus = Literal["ok", "empty", "failed", "unverified"]

_QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
_EARNINGS_PAGE_URL = "https://finance.yahoo.com/calendar/earnings"
_NEWS_URL = "https://finance.yahoo.com/xhr/ncp?queryRef=latestNews&serviceKey=ncp_fin"
_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
_ET = ZoneInfo("America/New_York")

SOURCE_TTL_SECONDS: dict[str, int] = {
    "calendar": 3600,
    "earnings_page": 21600,
    "news": 900,
    "search": 900,
    "symbol_search": 3600,
}

# Total backoff one read may sleep. With the client's 1 s base delay this allows
# three attempts for 5xx/429/transport errors (jitter can't change it). Request
# time is bounded separately: jobs.REQUEST_TIMEOUT_SECONDS per transport request
# and a jobs.ReadClock for the whole read.
_RETRY_BUDGET_SECONDS = 4.0
_NEWS_COUNT = 10

_source_cache = TTLCache()
_sleep = asyncio.sleep  # test seam: retry backoff


@dataclass(frozen=True)
class SourceRead:
    """Outcome of one source read.

    ``empty`` means verified: a 2xx, a valid envelope and an explicitly empty
    collection. ``unverified`` means success can't be proven (e.g. an earnings
    page with no table). ``failed`` covers exceptions, HTTP >= 400, error
    envelopes and malformed payloads.
    """

    source: str
    status: SourceStatus
    payload: Any = None
    error_type: str | None = None
    retryable: bool = False
    attempts: int = 1
    rejected_count: int = 0
    fetched_at: str | None = None

    @property
    def verified(self) -> bool:
        return self.status in ("ok", "empty")


@dataclass(frozen=True)
class _Interpretation:
    status: SourceStatus
    payload: Any = None
    error_type: str | None = None
    rejected_count: int = 0


def _error_type(error: BaseException) -> str:
    if is_rate_limit(error):
        return "rate_limited"
    status = http_status(error)
    if status is not None:
        return f"http_{status}"
    if isinstance(error, CurlTimeout):
        return "timeout"
    if isinstance(error, CurlConnectionError):
        return "connection"
    if isinstance(error, CurlHTTPError):
        return "transport"  # e.g. an incomplete transfer, no HTTP error status
    if isinstance(error, json.JSONDecodeError | YFDataException):
        return "unreadable_response"
    return type(error).__name__


def _property_retryable(error: Exception) -> tuple[bool, int]:
    """Property reads parse the body themselves, so an HTML error page (e.g. a 503)
    surfaces as a JSON/parse error with its HTTP status hidden: retry it."""
    if isinstance(error, json.JSONDecodeError | YFDataException):
        return True, _max_retries
    return _is_retryable_error(error)


@dataclass(frozen=True)
class _Outcome:
    value: Any = None
    attempts: int = 0
    failure: SourceRead | None = None


def _failed(source: str, error_type: str, attempts: int, retryable: bool) -> SourceRead:
    return SourceRead(
        source, "failed", error_type=error_type, retryable=retryable, attempts=attempts,
        fetched_at=utcnow_isoformat_z(),
    )


async def _attempts(
    source: str,
    call: Callable[[], Any],
    retryable: Callable[[Exception], tuple[bool, int]],
    symbol_lock: str | None = None,
) -> _Outcome:
    """Run ``call`` as executor jobs, retrying retryable errors, until the read deadline.

    Backoff and rate-limit cooldown sleeps happen between jobs, so no permit or
    lock is held while sleeping.
    """
    clock = jobs.ReadClock()
    slept = 0.0
    for attempt in range(_max_retries + 1):
        cooldown = rate_limit_cooldown_remaining()
        if cooldown > 0 and cooldown >= clock.remaining():
            return _Outcome(failure=_failed(source, "rate_limited", attempt, True))
        await wait_for_rate_limit_cooldown()
        if attempt:
            expire_yfinance_response_cache()  # a retry must do fresh I/O
        try:
            value = await jobs.run_job(call, clock=clock, symbol_lock=symbol_lock)
            return _Outcome(value=value, attempts=attempt + 1)
        except jobs.JobTimeoutError as e:
            return _Outcome(failure=_failed(source, e.reason, attempt + 1, True))
        except Exception as e:
            can_retry, limit = retryable(e)
            delay = _calculate_backoff(attempt)
            if is_rate_limit(e):
                note_rate_limit(delay)
            exhausted = (
                attempt >= min(_max_retries, limit)
                or slept + delay > _RETRY_BUDGET_SECONDS
                or delay >= clock.remaining()
            )
            if not can_retry or exhausted:
                return _Outcome(failure=_failed(source, _error_type(e), attempt + 1, can_retry))
            slept += delay
            await _sleep(delay)
    raise AssertionError("unreachable: the last attempt always returns")


def _checked(fetch: Callable[[], Any]) -> Any:
    response = fetch()
    response.raise_for_status()
    return response


async def _read(
    source: str,
    cache_key: str,
    fetch: Callable[[], Any],
    interpret: Callable[[Any], _Interpretation],
) -> SourceRead:
    cached = _source_cache.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    try:
        read, _ = await jobs.singleflight(
            ("source", cache_key), lambda: _fetch_source(source, cache_key, fetch, interpret)
        )
    except jobs.JobTimeoutError as e:
        return _failed(source, e.reason, 0, True)  # this caller's scope ended; the fetch goes on
    return read


async def _fetch_source(
    source: str,
    cache_key: str,
    fetch: Callable[[], Any],
    interpret: Callable[[Any], _Interpretation],
) -> SourceRead:
    outcome = await _attempts(source, lambda: _checked(fetch), _is_retryable_error)
    if outcome.failure is not None:
        return outcome.failure
    try:
        interpretation = interpret(outcome.value)
    except Exception:
        interpretation = _Interpretation("failed", error_type="malformed")
    read = SourceRead(
        source,
        interpretation.status,
        payload=interpretation.payload,
        error_type=interpretation.error_type,
        attempts=outcome.attempts,
        rejected_count=interpretation.rejected_count,
        fetched_at=utcnow_isoformat_z(),
    )
    if read.verified:
        _source_cache.set(cache_key, read, SOURCE_TTL_SECONDS[source])
    return read


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    empty = getattr(value, "empty", None)
    if isinstance(empty, bool):
        return empty
    try:
        return len(value) == 0
    except TypeError:
        return False


async def read_property(ticker: Any, source: str, read: Callable[[], T]) -> SourceRead:
    """A network-backed ``Ticker`` property, read as executor jobs (WP3, R3.8).

    Each attempt holds the symbol's lock (yfinance ``Ticker`` state is not
    thread-safe) and its admission permit until the thread finishes. Retryable
    errors, including an error page hidden behind a parse error, are retried.
    ``ok``: a non-empty value. ``unverified``: ``None`` or empty, since a property
    can't prove emptiness. ``failed``: the read failed or missed its deadline;
    the cached Ticker is evicted so its partial state is not reused.
    """
    symbol = _normalize(str(getattr(ticker, "ticker", "") or "")) or None
    outcome = await _attempts(source, read, _property_retryable, symbol_lock=symbol)
    if outcome.failure is not None:
        evict_ticker(ticker)
        return outcome.failure
    status: SourceStatus = "unverified" if _is_empty(outcome.value) else "ok"
    return SourceRead(
        source,
        status,
        payload=outcome.value,
        error_type="empty_property" if status == "unverified" else None,
        attempts=outcome.attempts,
        fetched_at=utcnow_isoformat_z(),
    )


def _dig(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def is_epoch(value: Any) -> TypeGuard[float]:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _interpret_calendar(response: Any) -> _Interpretation:
    summary = _dig(response.json(), "quoteSummary")
    if not isinstance(summary, dict) or summary.get("error"):
        return _Interpretation("failed", error_type="error_envelope")
    result = summary.get("result")
    events = result[0].get("calendarEvents") if isinstance(result, list) and result and isinstance(result[0], dict) else None
    if not isinstance(events, dict):
        return _Interpretation("failed", error_type="malformed")
    # Nested fields must have the expected shape too: a malformed earnings field
    # must not read as a verified absence. Validate raw values before defaulting;
    # only an absent key or null counts as missing.
    earnings = events.get("earnings")
    if earnings is not None and not isinstance(earnings, dict):
        return _Interpretation("failed", error_type="malformed")
    dates = (earnings or {}).get("earningsDate")
    if dates is not None and (not isinstance(dates, list) or not all(is_epoch(d) for d in dates)):
        return _Interpretation("failed", error_type="malformed")
    dividends = [events[key] for key in ("dividendDate", "exDividendDate") if events.get(key) is not None]
    if not all(is_epoch(value) for value in dividends):
        return _Interpretation("failed", error_type="malformed")
    return _Interpretation("ok" if dates or dividends else "empty", payload=events)


def _parse_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _parse_earnings_timestamp(text: str) -> datetime | None:
    """'October 28, 2026 at 4 PM EDT' -> tz-aware ET datetime (yfinance's format)."""
    wall, _, zone = text.rpartition(" ")
    if zone not in ("EDT", "EST"):
        return None
    try:
        return datetime.strptime(wall, "%B %d, %Y at %I %p").replace(tzinfo=_ET)
    except ValueError:
        return None


def _interpret_earnings_page(response: Any) -> _Interpretation:
    table = BeautifulSoup(response.text, "html.parser").find("table")
    if table is None:
        # No table could be an error page or genuinely no data (D2).
        return _Interpretation("unverified", error_type="no_table")
    header = [th.get_text(strip=True) for th in table.find_all("th")]
    columns = ("Earnings Date", "EPS Estimate", "Reported EPS", "Surprise (%)")
    if not all(column in header for column in columns):
        return _Interpretation("failed", error_type="malformed")

    # All or nothing, like yfinance's parse: a dropped row might be the nearest
    # date, so a page with any unparseable row does not verify.
    rows: list[dict[str, Any]] = []
    rejected = 0
    for tr in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        record = dict(zip(header, cells, strict=False)) if len(cells) == len(header) else None
        timestamp = _parse_earnings_timestamp(record["Earnings Date"]) if record else None
        if record is None or timestamp is None:
            rejected += 1
            continue
        rows.append({
            "timestamp": timestamp,
            "eps_estimate": _parse_number(record["EPS Estimate"]),
            "reported_eps": _parse_number(record["Reported EPS"]),
            "surprise_pct": _parse_number(record["Surprise (%)"]),
        })
    if rejected:
        return _Interpretation("failed", error_type="malformed", rejected_count=rejected)
    if not rows:
        return _Interpretation("empty", payload=[])
    return _Interpretation("ok", payload=rows)


def _interpret_news(response: Any) -> _Interpretation:
    stream = _dig(response.json(), "data", "tickerStream", "stream")
    if not isinstance(stream, list):
        return _Interpretation("failed", error_type="malformed")
    if not all(isinstance(item, dict) for item in stream):
        return _Interpretation("failed", error_type="malformed")
    if not stream:
        return _Interpretation("empty", payload=[])
    # Drop ads like yfinance does, but count them: an ads-only stream is not a
    # healthy empty feed.
    items = [item for item in stream if not item.get("ad")]
    return _Interpretation("ok", payload=items, rejected_count=len(stream) - len(items))


def _interpret_search_list(key: str) -> Callable[[Any], _Interpretation]:
    """Search ``news`` or ``quotes``: a list of objects (any other item is malformed)."""

    def interpret(response: Any) -> _Interpretation:
        items = _dig(response.json(), key)
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            return _Interpretation("failed", error_type="malformed")
        return _Interpretation("ok" if items else "empty", payload=items)

    return interpret


_interpret_search = _interpret_search_list("news")


def _normalize(symbol: str) -> str:
    return symbol.upper().strip()


async def read_calendar_events(symbol: str) -> SourceRead:
    """quoteSummary ``calendarEvents`` with raw epochs (payload: the events dict)."""
    sym = _normalize(symbol)
    params = {"modules": "calendarEvents", "corsDomain": "finance.yahoo.com", "formatted": "false", "symbol": sym}
    return await _read(
        "calendar",
        f"calendar:{sym}",
        lambda: YfData().get(_QUOTE_SUMMARY_URL.format(symbol=sym), params=params, timeout=jobs.REQUEST_TIMEOUT_SECONDS),
        _interpret_calendar,
    )


async def read_earnings_page(symbol: str) -> SourceRead:
    """Earnings page rows in page order (newest first), timestamps in ET."""
    sym = _normalize(symbol)
    params = {"symbol": sym, "offset": 0, "size": 25}
    return await _read(
        "earnings_page",
        f"earnings_page:{sym}",
        lambda: YfData().get(_EARNINGS_PAGE_URL, params=params, timeout=jobs.REQUEST_TIMEOUT_SECONDS),
        _interpret_earnings_page,
    )


async def read_news_feed(symbol: str) -> SourceRead:
    """The ``/xhr/ncp`` news stream (payload: story items; ads removed and counted
    in ``rejected_count``)."""
    sym = _normalize(symbol)
    body = {"serviceConfig": {"snippetCount": _NEWS_COUNT, "s": [sym]}}
    return await _read(
        "news",
        f"news:{sym}",
        lambda: YfData().post(_NEWS_URL, body=body, timeout=jobs.REQUEST_TIMEOUT_SECONDS),
        _interpret_news,
    )


async def read_search_news(symbol: str) -> SourceRead:
    """Search ``news`` items (a different schema from the news feed)."""
    sym = _normalize(symbol)
    params = {"q": sym, "quotesCount": 0, "newsCount": _NEWS_COUNT, "enableFuzzyQuery": False, "newsQueryId": "news_cie_vespa"}
    return await _read(
        "search",
        f"search:{sym}",
        lambda: YfData().get(_SEARCH_URL, params=params, timeout=jobs.REQUEST_TIMEOUT_SECONDS),
        _interpret_search,
    )


async def read_symbol_matches(query: str) -> SourceRead:
    """Search ``quotes`` for a ticker or company name (yfinance's ``Search.quotes``
    request, without its cached, unchecked read)."""
    q = query.strip()
    params = {"q": q, "quotesCount": 8, "newsCount": 0, "enableFuzzyQuery": False, "quotesQueryId": "tss_match_phrase_query"}
    return await _read(
        "symbol_search",
        f"symbol_search:{q}",
        lambda: YfData().get(_SEARCH_URL, params=params, timeout=jobs.REQUEST_TIMEOUT_SECONDS),
        _interpret_search_list("quotes"),
    )


async def read_news(symbol: str) -> SourceRead:
    """News feed, falling back to Search unless the feed returned items.

    An ``ok`` result keeps the source that produced it (``news`` or ``search``).
    Otherwise: ``empty`` when both are verified empty, ``failed`` when both
    failed, ``unverified`` for any other mix.
    """
    primary = await read_news_feed(symbol)
    if primary.status == "ok":
        return primary
    fallback = await read_search_news(symbol)
    if fallback.status == "ok":
        return fallback
    statuses = {primary.status, fallback.status}
    status: SourceStatus = (
        "empty" if statuses == {"empty"} else "failed" if statuses == {"failed"} else "unverified"
    )
    return SourceRead(
        "news",
        status,
        payload=[],
        error_type=primary.error_type or fallback.error_type,
        attempts=primary.attempts + fallback.attempts,
        fetched_at=fallback.fetched_at,
    )
