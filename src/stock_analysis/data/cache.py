"""In-memory resource caching for price data."""

import gzip
import hashlib
import os
import threading
import time
from datetime import datetime
from typing import Any

import pandas as pd

from stock_analysis.utils.ohlcv import df_to_csv
from stock_analysis.utils.validators import FetchParams


class PriceCache:
    """
    Cache stores exact CSV text for O(1) deterministic serving.

    Resources only serve cached data for the life of the current process.
    Never fetch live from a resource read.
    """

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = cache_dir
        self._entries: dict[str, tuple[dict[str, Any], float]] = {}
        self._default_ttl = int(os.environ.get("CACHE_TTL", "300"))  # 5 minutes
        self._lock = threading.Lock()

    def store(
        self,
        params: FetchParams,
        df: pd.DataFrame,
        ttl: int | None = None,
    ) -> str:
        """
        Store gzipped CSV + metadata, return canonical URI.

        Note: df should already be standardized (from yfinance_client).

        Args:
            params: Fetch parameters (used to generate URI)
            df: Standardized DataFrame to cache
            ttl: Cache TTL in seconds (default: 300)

        Returns:
            Canonical URI for the cached data
        """
        uri = params.to_uri()

        csv_text = df_to_csv(df)
        csv_bytes = csv_text.encode("utf-8")
        csv_gz = gzip.compress(csv_bytes)

        entry: dict[str, Any] = {
            "csv_gz": csv_gz,
            "encoding": "gzip",
            "size_bytes": len(csv_bytes),
            "compressed_bytes": len(csv_gz),
            "rows": len(df),
            "columns": list(df.columns),
            "hash": hashlib.sha256(csv_bytes).hexdigest()[:16],
            "stored_at": datetime.utcnow().isoformat(),
        }

        expire = ttl if ttl is not None else self._default_ttl
        expires_at = time.monotonic() + expire
        with self._lock:
            self._entries[uri] = (entry, expires_at)

        return uri

    def get(self, uri: str) -> dict[str, Any] | None:
        """
        Get cache entry by URI.

        Args:
            uri: Canonical URI

        Returns:
            Cache entry dict or None if not found
        """
        with self._lock:
            item = self._entries.get(uri)
            if item is None:
                return None

            entry, expires_at = item
            if time.monotonic() > expires_at:
                del self._entries[uri]
                return None

            return entry

    def get_csv(self, uri: str) -> str | None:
        """
        Get decompressed CSV text by URI.

        Args:
            uri: Canonical URI

        Returns:
            CSV text or None if not found
        """
        entry = self.get(uri)
        if not entry:
            return None
        return gzip.decompress(entry["csv_gz"]).decode("utf-8")

    def get_metadata(self, uri: str) -> dict[str, Any] | None:
        """
        Get cache metadata without decompressing data.

        Args:
            uri: Canonical URI

        Returns:
            Metadata dict or None if not found
        """
        entry = self.get(uri)
        if not entry:
            return None
        return {
            "rows": entry["rows"],
            "columns": entry["columns"],
            "size_bytes": entry["size_bytes"],
            "hash": entry["hash"],
            "stored_at": entry["stored_at"],
        }

    def exists(self, uri: str) -> bool:
        """Check if URI exists in cache."""
        return self.get(uri) is not None

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._entries.clear()


# Global instance
price_cache = PriceCache()
