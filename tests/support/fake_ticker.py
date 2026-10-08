"""Application-level stand-in for ``yfinance.Ticker`` in tool-level tests.

This fake does NOT exercise yfinance internals (error suppression, caching,
parsing). For those, use the transport contract harness in
``tests.support.transport``.
"""

from collections import Counter
from typing import Any

from curl_cffi.requests.exceptions import HTTPError

from tests.support.transport import FakeResponse, TransportFixture


class FakeTicker:
    """Serves configured attributes; an exception value is raised on read.

    Callables (e.g. ``history=lambda **kw: df``) are returned as-is so they can
    be called like the real methods. ``reads`` counts attribute accesses.
    """

    def __init__(self, symbol: str = "EXMP", **attributes: Any):
        self.ticker = symbol
        self.reads: Counter[str] = Counter()
        self._attributes = attributes

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name not in self._attributes:
            raise AttributeError(f"FakeTicker has no configured attribute {name!r}")
        self.reads[name] += 1
        value = self._attributes[name]
        if isinstance(value, BaseException):
            raise value
        return value


def http_error(status: int, url: str = "https://example.invalid") -> HTTPError:
    response = FakeResponse(TransportFixture(status, "", "text/plain"), url)
    return HTTPError(f"HTTP Error {status}: {response.reason}", 0, response)
