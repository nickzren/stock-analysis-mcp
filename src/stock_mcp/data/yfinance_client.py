"""Async yfinance client with bounded concurrency."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import pandas as pd
import pytz
import yfinance as yf

from stock_mcp.utils.ohlcv import standardize_ohlcv
from stock_mcp.utils.validators import FetchParams

# Bounded concurrency for yfinance calls
_max_workers = int(os.environ.get("YF_MAX_WORKERS", "4"))
_executor = ThreadPoolExecutor(max_workers=_max_workers)
_fetch_semaphore = asyncio.Semaphore(_max_workers)

# Shutdown coordination
shutdown_event = asyncio.Event()


class ServerShuttingDownError(Exception):
    """Raised when server is shutting down."""

    pass


async def fetch_history(params: FetchParams) -> pd.DataFrame:
    """
    Fetch price history with bounded concurrency and proper timeout support.

    Standardization happens here (single place) before both preview and cache.

    Args:
        params: Fetch parameters

    Returns:
        Standardized DataFrame with OHLCV data

    Raises:
        ServerShuttingDownError: If server is shutting down
        ValueError: If symbol is invalid or no data returned
    """
    if shutdown_event.is_set():
        raise ServerShuttingDownError("Server is shutting down")

    async with _fetch_semaphore:
        loop = asyncio.get_running_loop()

        def _fetch() -> pd.DataFrame:
            df = yf.download(**params.to_yf_kwargs())
            if df.empty:
                raise ValueError(f"No data returned for {params.symbol}")
            # Standardize here - single place for both preview and cache
            return standardize_ohlcv(df, params.adjusted)

        return await loop.run_in_executor(_executor, _fetch)


async def fetch_info(symbol: str) -> dict[str, Any]:
    """
    Fetch stock info (fundamentals, metadata).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Info dict from yfinance

    Raises:
        ServerShuttingDownError: If server is shutting down
        ValueError: If symbol is invalid
    """
    if shutdown_event.is_set():
        raise ServerShuttingDownError("Server is shutting down")

    async with _fetch_semaphore:
        loop = asyncio.get_running_loop()

        def _fetch() -> dict[str, Any]:
            ticker = yf.Ticker(symbol.upper().strip())
            info = ticker.info
            if not info or info.get("regularMarketPrice") is None:
                # Try to get at least basic info
                if not info:
                    raise ValueError(f"Invalid symbol: {symbol}")
            return info

        return await loop.run_in_executor(_executor, _fetch)


async def fetch_ticker(symbol: str) -> yf.Ticker:
    """
    Get yfinance Ticker object.

    Args:
        symbol: Stock ticker symbol

    Returns:
        yfinance Ticker object for further operations
    """
    if shutdown_event.is_set():
        raise ServerShuttingDownError("Server is shutting down")

    async with _fetch_semaphore:
        loop = asyncio.get_running_loop()

        def _fetch() -> yf.Ticker:
            return yf.Ticker(symbol.upper().strip())

        return await loop.run_in_executor(_executor, _fetch)


def get_market_state(tz: str = "America/New_York") -> dict[str, str]:
    """
    Determine market state. Clock-based only (no holiday calendar).

    Args:
        tz: Timezone (default: America/New_York)

    Returns:
        Dict with state, method, and checked_at timestamp
    """
    eastern = pytz.timezone(tz)
    now = datetime.now(eastern)

    # Weekends
    if now.weekday() >= 5:
        state = "closed"
    else:
        hour, minute = now.hour, now.minute
        time_minutes = hour * 60 + minute

        if time_minutes < 4 * 60:  # Before 4 AM
            state = "closed"
        elif time_minutes < 9 * 60 + 30:  # 4 AM - 9:30 AM
            state = "pre_market"
        elif time_minutes < 16 * 60:  # 9:30 AM - 4 PM
            state = "regular"
        elif time_minutes < 20 * 60:  # 4 PM - 8 PM
            state = "after_hours"
        else:
            state = "closed"

    return {
        "state": state,
        "method": "clock_only_no_holidays",
        "checked_at": now.isoformat(),
    }


async def shutdown_executor() -> None:
    """Cleanup on server shutdown."""
    shutdown_event.set()
    _executor.shutdown(wait=False, cancel_futures=True)
