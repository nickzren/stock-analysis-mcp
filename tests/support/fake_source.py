"""Stand-in for one provider read function in tool-level tests (Step 1 WP1, R1.14).

Patch it over a read the tool calls, e.g.
``monkeypatch.setattr(news, "read_news", FakeSource.failed("news"))``. It skips
the transport entirely; for transport behavior use ``tests.support.transport``.
"""

from typing import Any

from stock_analysis.data.provider import SourceRead


class FakeSource:
    """Returns one fixed ``SourceRead`` and records the symbol of each call."""

    def __init__(self, read: SourceRead) -> None:
        self.read = read
        self.calls: list[str] = []

    async def __call__(self, symbol: str) -> SourceRead:
        self.calls.append(symbol)
        return self.read

    @classmethod
    def ok(cls, source: str, payload: Any) -> "FakeSource":
        return cls(SourceRead(source, "ok", payload=payload))

    @classmethod
    def empty(cls, source: str, payload: Any) -> "FakeSource":
        return cls(SourceRead(source, "empty", payload=payload))

    @classmethod
    def failed(cls, source: str, error_type: str = "http_503") -> "FakeSource":
        return cls(SourceRead(source, "failed", error_type=error_type))

    @classmethod
    def unverified(cls, source: str, error_type: str = "no_table") -> "FakeSource":
        return cls(SourceRead(source, "unverified", error_type=error_type))
