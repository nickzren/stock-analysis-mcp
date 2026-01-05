"""Resource caching for price data."""

import gzip
import hashlib
import os
from datetime import datetime
from typing import Any

import diskcache
import pandas as pd

from stock_mcp.utils.ohlcv import df_to_csv
from stock_mcp.utils.validators import FetchParams


class PriceCache:
    """
    Cache stores exact CSV text for O(1) deterministic serving.

    Resources only serve cached data. Never fetch live.
    """

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.environ.get("CACHE_DIR", ".cache/prices")
        self.cache: diskcache.Cache = diskcache.Cache(cache_dir)
        self._default_ttl = int(os.environ.get("CACHE_TTL", "300"))  # 5 minutes

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
        self.cache.set(uri, entry, expire=expire)

        return uri

    def get(self, uri: str) -> dict[str, Any] | None:
        """
        Get cache entry by URI.

        Args:
            uri: Canonical URI

        Returns:
            Cache entry dict or None if not found
        """
        return self.cache.get(uri)

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
        return uri in self.cache

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()


# Global instance
price_cache = PriceCache()
