"""Transport-level stub for contract tests against the real pinned yfinance.

Only the HTTP session inside yfinance's ``YfData`` singleton (and its
cookie/crumb acquisition) is replaced. Everything above it -- ``Ticker``
properties, ``Search``, yfinance's error suppression, its ``cache_get`` LRU and
its parsers -- runs for real, so these tests exercise the layer where the
suppression, cache and parser defects live.
"""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

from curl_cffi.requests.exceptions import HTTPError
from yfinance.config import YfConfig
from yfinance.data import YfData

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "transport"

# Fixture source directory -> URL substring. "calendar" matches every quoteSummary
# request (the module is a query param), not only calendarEvents.
ROUTES: dict[str, str] = {
    "calendar": "/v10/finance/quoteSummary/",
    "earnings_page": "finance.yahoo.com/calendar/earnings",
    "news": "/xhr/ncp",
    "search": "/v1/finance/search",
    "timeseries": "/ws/fundamentals-timeseries/",
    "quote_v7": "/v7/finance/quote",
    "options": "/v7/finance/options/",
    "consent_page": "guce.yahoo.com/consent",
    "chart": "/v8/finance/chart/",
}

# Set per test by the autouse guard in tests/conftest.py. Unrouted stub requests
# are reported there as well, so they fail the test at teardown however the stub
# context exits -- even when the error is raised inside and caught outside it.
violation_sink: list[str] | None = None


def source_for_url(url: str) -> str | None:
    for source, fragment in ROUTES.items():
        if fragment in url:
            return source
    return None


@dataclass(frozen=True)
class TransportFixture:
    status: int
    body: str
    content_type: str
    final_url: str | None = None  # where the request ended up after redirects


def load_fixture(source: str, name: str) -> TransportFixture:
    """Load ``tests/fixtures/transport/<source>/<name>.json``.

    The spec holds ``status`` plus exactly one of ``json`` (an object served as
    a JSON body), ``body_file`` (a sibling file) or ``body`` (inline text), and
    optionally ``final_url`` (a redirect target, e.g. Yahoo's consent page).
    """
    path = FIXTURE_ROOT / source / f"{name}.json"
    spec = json.loads(path.read_text())
    final_url = spec.get("final_url")
    if "json" in spec:
        return TransportFixture(spec["status"], json.dumps(spec["json"]), "application/json", final_url)
    if "body_file" in spec:
        body = (path.parent / spec["body_file"]).read_text()
        return TransportFixture(spec["status"], body, spec.get("content_type", "text/html"), final_url)
    return TransportFixture(
        spec["status"], spec.get("body", ""), spec.get("content_type", "text/plain"), final_url
    )


class FakeResponse:
    """The subset of ``curl_cffi.requests.Response`` that yfinance and the recorder read."""

    def __init__(self, fixture: TransportFixture, url: str):
        self.status_code = fixture.status
        self.reason = HTTPStatus(fixture.status).phrase
        self.ok = fixture.status < 400
        self.text = fixture.body
        self.content = fixture.body.encode()
        self.headers = {"content-type": fixture.content_type}
        self.url = url

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise HTTPError(f"HTTP Error {self.status_code}: {self.reason}", 0, self)


class UnroutedRequestError(AssertionError):
    """A stubbed-transport test made a request it did not route."""


@dataclass(frozen=True)
class RecordedCall:
    method: str
    url: str
    source: str
    timeout: float | None = None


class FakeSession:
    """Serves routed fixtures in order; the last fixture of a route repeats.

    An unrouted request raises and is also recorded, in ``unrouted`` and in the
    per-test ``violation_sink``, because app code may swallow the error.
    """

    def __init__(self) -> None:
        self.calls: list[RecordedCall] = []
        self.unrouted: list[str] = []
        self._queues: dict[str, list[TransportFixture]] = {}
        self._holds: dict[str, tuple[threading.Event, threading.Event]] = {}

    def route(self, source: str, *names: str) -> None:
        if source not in ROUTES:
            raise ValueError(f"unknown source {source!r}")
        self._queues[source] = [load_fixture(source, name) for name in names]

    def hold(self, source: str) -> tuple[threading.Event, threading.Event]:
        """Block requests to ``source`` in their worker thread until released.

        Returns (entered, release): ``entered`` is set when a request arrives; set
        ``release`` to let it (and later ones) through. Waits are capped so a
        forgotten release can't hang the suite.
        """
        entered, release = threading.Event(), threading.Event()
        self._holds[source] = (entered, release)
        return entered, release

    def calls_to(self, source: str) -> list[RecordedCall]:
        return [call for call in self.calls if call.source == source]

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._dispatch("GET", url, kwargs.get("timeout"))

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._dispatch("POST", url, kwargs.get("timeout"))

    def _dispatch(self, method: str, url: str, timeout: float | None = None) -> FakeResponse:
        source = source_for_url(url)
        if source in self._holds:
            entered, release = self._holds[source]
            entered.set()
            release.wait(timeout=10)
        queue = self._queues.get(source) if source else None
        if source is None or not queue:
            self.unrouted.append(f"{method} {url}")
            if violation_sink is not None:
                violation_sink.append(f"stub transport: unrouted {method} {url}")
            raise UnroutedRequestError(f"unrouted {method} {url}")
        self.calls.append(RecordedCall(method, url, source, timeout))
        fixture = queue.pop(0) if len(queue) > 1 else queue[0]
        return FakeResponse(fixture, fixture.final_url or url)


@contextmanager
def stubbed_yfinance_transport(*, hide_exceptions: bool | None = None) -> Iterator[FakeSession]:
    """Swap ``YfData``'s session for a ``FakeSession`` and restore it afterwards.

    Clears yfinance's ``cache_get`` LRU on entry and exit. ``hide_exceptions``
    overrides yfinance's global flag for the duration (``None`` keeps the app's
    policy, set at import by ``stock_analysis.data.yfinance_client``) -- a
    test-only toggle, never a production pattern. Does not nest: a nested stub
    would restore the outer fake as the "real" session. Unrouted requests are
    reported through ``violation_sink``, not on exit.
    """
    data = YfData()
    if isinstance(data._session, FakeSession):
        raise RuntimeError("stubbed_yfinance_transport is already active; it does not nest")
    saved = (data._session, data._cookie_strategy, data._cookie, data._crumb)
    saved_hide = YfConfig.debug.hide_exceptions
    session = FakeSession()
    data._session = session
    data._get_cookie_and_crumb = lambda timeout=30: ("test-crumb", "basic")
    YfData.cache_get.cache_clear()
    if hide_exceptions is not None:
        YfConfig.debug.hide_exceptions = hide_exceptions
    try:
        yield session
    finally:
        data._session, data._cookie_strategy, data._cookie, data._crumb = saved
        data.__dict__.pop("_get_cookie_and_crumb", None)
        YfConfig.debug.hide_exceptions = saved_hide
        YfData.cache_get.cache_clear()
