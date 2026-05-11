"""In-memory TTL cache for yfinance data with market-hours awareness."""

import time
from datetime import datetime
from enum import Enum
from typing import Any

import pytz


class DataType(Enum):
    """Data types with associated TTL configurations."""

    PRICE_HISTORY = "price_history"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    EVENTS = "events"
    TICKER = "ticker"


# TTLs in seconds: (market_hours, off_hours)
_TTL_CONFIG: dict[DataType, tuple[int, int]] = {
    DataType.PRICE_HISTORY: (300, 1800),      # 5min / 30min
    DataType.FUNDAMENTALS: (14400, 14400),     # 4h / 4h (quarterly data, no market-hours benefit)
    DataType.NEWS: (1800, 1800),               # 30min / 30min
    DataType.EVENTS: (3600, 3600),             # 1h / 1h
    DataType.TICKER: (3600, 3600),             # 1h / 1h
}


def classify_session(now: datetime) -> str:
    """Classify a US-equities trading session for the given (tz-aware) datetime.

    Returns one of: "closed", "pre_market", "regular", "after_hours".
    Clock-based only — does not account for holidays.
    """
    if now.weekday() >= 5:
        return "closed"
    minutes = now.hour * 60 + now.minute
    if minutes < 4 * 60:
        return "closed"
    if minutes < 9 * 60 + 30:
        return "pre_market"
    if minutes < 16 * 60:
        return "regular"
    if minutes < 20 * 60:
        return "after_hours"
    return "closed"


def is_market_hours() -> bool:
    """Check if US equity markets are currently in regular trading hours."""
    eastern = pytz.timezone("America/New_York")
    return classify_session(datetime.now(eastern)) == "regular"


def get_ttl(data_type: DataType) -> int:
    """Get TTL in seconds for a data type, adjusted for market hours."""
    market_ttl, off_ttl = _TTL_CONFIG[data_type]
    return market_ttl if is_market_hours() else off_ttl


class TTLCache:
    """Simple in-memory TTL cache with per-key expiry."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        """Get value if present and not expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store value with TTL."""
        self._store[key] = (value, time.monotonic() + ttl_seconds)

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def size(self) -> int:
        """Number of entries (including potentially expired)."""
        return len(self._store)


# Global caches per data type
info_cache = TTLCache()
ticker_cache = TTLCache()
