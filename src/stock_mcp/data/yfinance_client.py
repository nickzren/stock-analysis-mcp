"""Async yfinance client with bounded concurrency and retry logic."""

import asyncio
import logging
import os
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

import pandas as pd
import pytz
import yfinance as yf
from requests.exceptions import HTTPError

from stock_mcp.utils.ohlcv import standardize_ohlcv
from stock_mcp.utils.validators import FetchParams

logger = logging.getLogger(__name__)

# Bounded concurrency for yfinance calls
_max_workers = int(os.environ.get("YF_MAX_WORKERS", "4"))
_executor = ThreadPoolExecutor(max_workers=_max_workers)
_fetch_semaphore = asyncio.Semaphore(_max_workers)

# Retry configuration
_max_retries = int(os.environ.get("YF_MAX_RETRIES", "3"))
_base_delay = float(os.environ.get("YF_BASE_DELAY", "1.0"))  # seconds
_max_delay = float(os.environ.get("YF_MAX_DELAY", "30.0"))  # seconds

# Shutdown coordination
shutdown_event = asyncio.Event()

T = TypeVar("T")


class ServerShuttingDownError(Exception):
    """Raised when server is shutting down."""

    pass


class YFinanceRetryError(Exception):
    """Raised when yfinance fails after all retries."""

    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


class YFinanceIncompleteInfoError(RuntimeError):
    """Raised when yfinance returns incomplete data (e.g., 401 Invalid Crumb)."""

    def __init__(
        self,
        symbol: str,
        *,
        key_count: int,
        quote_type: str | None,
        sentinel_keys_present: int,
        sentinel_values_present: int,
    ):
        msg = (
            f"Incomplete yfinance info for {symbol}: "
            f"keys={key_count}, quoteType={quote_type}, "
            f"sentinel_keys_present={sentinel_keys_present}, "
            f"sentinel_values_present={sentinel_values_present}"
        )
        super().__init__(msg)
        self.symbol = symbol
        self.key_count = key_count
        self.quote_type = quote_type
        self.sentinel_keys_present = sentinel_keys_present
        self.sentinel_values_present = sentinel_values_present


@dataclass(frozen=True)
class InfoCompleteness:
    """Result of info completeness assessment."""

    is_incomplete: bool
    key_count: int
    quote_type: str | None
    sentinel_keys_present: int
    sentinel_values_present: int


# Sentinel keys that indicate a complete fundamental response
# These are very unlikely to be absent in a real full response,
# but will be absent in crumb-broken partial responses
INFO_FUND_SENTINELS: tuple[str, ...] = (
    "totalRevenue",
    "revenueGrowth",
    "profitMargins",
    "grossMargins",
    "operatingCashflow",
    "freeCashflow",
    "totalCash",
    "cashAndShortTermInvestments",
    "ebitda",
    "enterpriseValue",
    "sharesOutstanding",
    "trailingEps",
    "trailingPE",
    "priceToSalesTrailing12Months",
)

# Core financial statement sentinels - true fundamentals that crumb-broken payloads reliably lack
# These are more reliable than market metadata fields (enterpriseValue, sharesOutstanding)
# which may appear in partial payloads
INFO_CORE_FUND_SENTINELS: tuple[str, ...] = (
    "totalRevenue",
    "revenueGrowth",
    "profitMargins",
    "grossMargins",
    "operatingCashflow",
    "freeCashflow",
    "totalCash",
    "cashAndShortTermInvestments",
)

# Thresholds for completeness
# We require at least 1 core value to be truly present (not None/NaN)
_MIN_CORE_VALUES_PRESENT = 1


def _has_value(v: Any) -> bool:
    """
    Check if a value is truly present (not None, NaN, or empty string).

    yfinance often uses float("nan") for missing numerics, which passes
    `is not None` but should be treated as missing.
    """
    import math

    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True

# Singleflight cache for deduplicating concurrent fetch_info calls
# Key: symbol (uppercase), Value: asyncio.Task returning (info, RetryResult)
# Note: Using string annotation to avoid forward reference issue
_info_singleflight: dict[str, "asyncio.Task[tuple[dict[str, Any], RetryResult]]"] = {}
_info_singleflight_lock = asyncio.Lock()


def assess_info_completeness(info: dict[str, Any]) -> InfoCompleteness:
    """
    Assess whether info dict has complete fundamental data.

    Uses core sentinel keys to detect "crumb-broken" partial payloads.
    Skips enforcement for non-EQUITY quoteTypes (ETFs, etc.).

    The key insight is that crumb-broken payloads often have keys present
    but with None/NaN values, or only have "metadata-ish" fields like
    enterpriseValue/sharesOutstanding without true financial statement data.

    We require at least 1 core financial statement value to be truly present.

    Returns:
        InfoCompleteness with diagnostic fields
    """
    if not isinstance(info, dict) or not info:
        return InfoCompleteness(
            is_incomplete=True,
            key_count=0,
            quote_type=None,
            sentinel_keys_present=0,
            sentinel_values_present=0,
        )

    key_count = len(info)

    # Get quoteType - skip enforcement for ETFs, mutual funds, etc.
    quote_type_raw = info.get("quoteType")
    quote_type = str(quote_type_raw).upper() if quote_type_raw is not None else None

    # Only enforce fundamentals for EQUITY or unknown quoteType
    enforce_fundamentals = quote_type in (None, "", "EQUITY")

    # Count sentinel keys present (key exists in dict) - for diagnostics
    sentinel_keys_present = sum(1 for k in INFO_FUND_SENTINELS if k in info)
    # Count sentinel values present using _has_value (handles NaN and empty strings)
    sentinel_values_present = sum(1 for k in INFO_FUND_SENTINELS if _has_value(info.get(k)))

    # Count core values present - this is the primary completeness check
    # Core values are true financial statement fields that crumb-broken payloads reliably lack
    core_values_present = sum(1 for k in INFO_CORE_FUND_SENTINELS if _has_value(info.get(k)))

    # Incomplete if: enforcement required AND insufficient core values
    # This catches both "keys present but values missing" and "only metadata fields present"
    is_incomplete = enforce_fundamentals and core_values_present < _MIN_CORE_VALUES_PRESENT

    return InfoCompleteness(
        is_incomplete=is_incomplete,
        key_count=key_count,
        quote_type=quote_type,
        sentinel_keys_present=sentinel_keys_present,
        sentinel_values_present=sentinel_values_present,
    )


def _is_retryable_error(error: Exception) -> tuple[bool, int]:
    """
    Check if an error is retryable (transient).

    Returns:
        Tuple of (is_retryable, max_retries_for_this_error)
        - max_retries_for_this_error: Use this to limit retries for specific errors
          (e.g., 401 Invalid Crumb rarely recovers with more retries)
    """
    # YFinanceIncompleteInfoError - retry up to 2 times
    if isinstance(error, YFinanceIncompleteInfoError):
        return (True, 2)

    # HTTP 401 "Invalid Crumb" is the main one - retry once then fallback
    if (
        isinstance(error, HTTPError)
        and hasattr(error, "response")
        and error.response is not None
    ):
        status_code = error.response.status_code
        if status_code == 401:
            # 401 Invalid Crumb - retry once, then give up (fallback will handle)
            return (True, 1)
        if status_code == 429:
            # Rate limit - retry with full backoff
            return (True, _max_retries)
        if 500 <= status_code < 600:
            # Server errors - retry with full backoff
            return (True, _max_retries)

    # Also check for string patterns in wrapped errors
    error_str = str(error).lower()

    if "401" in error_str or "invalid crumb" in error_str or "incomplete response" in error_str:
        # Limit 401/incomplete retries - they rarely recover with many retries
        return (True, 2)

    retryable_patterns = [
        "rate limit",
        "too many requests",
        "connection",
        "timeout",
        "temporary",
    ]
    if any(pattern in error_str for pattern in retryable_patterns):
        return (True, _max_retries)

    return (False, 0)


def _calculate_backoff(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter."""
    # Exponential backoff: base_delay * 2^attempt
    delay = _base_delay * (2**attempt)
    # Add jitter (±25%)
    jitter = delay * 0.25 * (2 * random.random() - 1)
    delay = delay + jitter
    # Cap at max delay
    return min(delay, _max_delay)


@dataclass
class RetryAttempt:
    """Record of a single retry attempt for provenance tracking."""

    attempt: int
    ok: bool
    error: str | None = None
    backoff_s: float | None = None
    # For info fetches: sentinel values present (helps debug partial payloads)
    sentinel_values_present: int | None = None


@dataclass
class RetryResult:
    """Result of a retry operation with provenance tracking."""

    result: Any
    attempts: int
    total_backoff_seconds: float
    source: str = "yfinance"
    # Info completeness (only populated for fetch_info calls)
    info_completeness: InfoCompleteness | None = None
    # Track if any attempt failed completeness check and was retried
    incomplete_detected: bool = False
    # Per-attempt trace for debugging (capped at 3 entries)
    retry_trace: list[RetryAttempt] | None = None

    def to_provenance(self) -> dict[str, Any]:
        """Convert to provenance dict for data_provenance field."""
        prov: dict[str, Any] = {
            "source": self.source,
            "attempts": self.attempts,
            "retries_exhausted": False,  # If exhausted, YFinanceRetryError would have been raised
            "fallback_used": False,  # No fallback since Alpha Vantage was removed
            "total_backoff_seconds": self.total_backoff_seconds,
            "incomplete_detected": self.incomplete_detected,
        }
        # Add retry trace if present (for debugging partial payload issues)
        if self.retry_trace:
            prov["retry_trace"] = [
                {
                    "attempt": t.attempt,
                    "ok": t.ok,
                    **({"error": t.error} if t.error else {}),
                    **({"backoff_s": t.backoff_s} if t.backoff_s else {}),
                    **({"sentinel_values_present": t.sentinel_values_present}
                       if t.sentinel_values_present is not None else {}),
                }
                for t in self.retry_trace[-3:]  # Cap at last 3 attempts
            ]
        # Add completeness info if present
        if self.info_completeness is not None:
            prov["info_completeness"] = {
                "is_incomplete": self.info_completeness.is_incomplete,
                "key_count": self.info_completeness.key_count,
                "quote_type": self.info_completeness.quote_type,
                "sentinel_keys_present": self.info_completeness.sentinel_keys_present,
                "sentinel_values_present": self.info_completeness.sentinel_values_present,
                "min_sentinel_keys_present": _MIN_CORE_VALUES_PRESENT,
            }
        return prov


async def _retry_with_backoff(
    operation_name: str,
    sync_func: Callable[[], T],
    max_retries: int = _max_retries,
) -> RetryResult:
    """
    Execute a synchronous function with retry logic.

    Args:
        operation_name: Name for logging (e.g., "fetch_history(AAPL)")
        sync_func: Synchronous function to execute
        max_retries: Maximum number of retry attempts

    Returns:
        RetryResult with result and provenance info

    Raises:
        YFinanceRetryError: If all retries exhausted
        ServerShuttingDownError: If server is shutting down
    """
    last_error: Exception | None = None
    total_backoff: float = 0.0
    effective_max_retries = max_retries
    incomplete_detected = False  # Track if any attempt failed completeness
    retry_trace: list[RetryAttempt] = []  # Per-attempt history for debugging

    for attempt in range(max_retries + 1):
        if shutdown_event.is_set():
            raise ServerShuttingDownError("Server is shutting down")

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(_executor, sync_func)
            # Record successful attempt in trace
            retry_trace.append(RetryAttempt(attempt=attempt + 1, ok=True))
            return RetryResult(
                result=result,
                attempts=attempt + 1,
                total_backoff_seconds=round(total_backoff, 2),
                source="yfinance",
                incomplete_detected=incomplete_detected,
                retry_trace=retry_trace if len(retry_trace) > 1 else None,  # Only include if retries happened
            )
        except Exception as e:
            last_error = e

            # Extract sentinel info for incomplete errors (helps debug partial payloads)
            sentinel_values: int | None = None
            if isinstance(e, YFinanceIncompleteInfoError):
                incomplete_detected = True
                sentinel_values = e.sentinel_values_present

            # Record failed attempt in trace
            error_name = type(e).__name__
            retry_trace.append(RetryAttempt(
                attempt=attempt + 1,
                ok=False,
                error=error_name,
                sentinel_values_present=sentinel_values,
            ))

            # Check if retryable and get max retries for this error type
            is_retryable, error_max_retries = _is_retryable_error(e)

            # Don't retry non-retryable errors
            if not is_retryable:
                raise

            # Use the smaller of: global max_retries or error-specific limit
            effective_max_retries = min(max_retries, error_max_retries)

            # Don't retry if we've exhausted attempts for this error type
            if attempt >= effective_max_retries:
                logger.warning(
                    f"{operation_name}: Failed after {attempt + 1} attempts "
                    f"(limit={effective_max_retries + 1}). Last error: {e}"
                )
                raise YFinanceRetryError(
                    f"Failed after {attempt + 1} attempts: {e}",
                    last_error=last_error,
                ) from e

            # Calculate backoff delay
            delay = _calculate_backoff(attempt)
            total_backoff += delay
            # Update last trace entry with backoff
            retry_trace[-1].backoff_s = round(delay, 2)
            logger.info(
                f"{operation_name}: Attempt {attempt + 1} failed ({e}). "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise YFinanceRetryError(
        f"Failed after {max_retries + 1} attempts",
        last_error=last_error,
    )


async def fetch_history(params: FetchParams) -> pd.DataFrame:
    """
    Fetch price history with bounded concurrency, retry logic, and proper timeout support.

    Standardization happens here (single place) before both preview and cache.

    Args:
        params: Fetch parameters

    Returns:
        Standardized DataFrame with OHLCV data

    Raises:
        ServerShuttingDownError: If server is shutting down
        YFinanceRetryError: If all retries exhausted for retryable errors
        ValueError: If symbol is invalid or no data returned
    """
    if shutdown_event.is_set():
        raise ServerShuttingDownError("Server is shutting down")

    def _fetch() -> pd.DataFrame:
        df = yf.download(**params.to_yf_kwargs())
        if df.empty:
            raise ValueError(f"No data returned for {params.symbol}")
        # Standardize here - single place for both preview and cache
        return standardize_ohlcv(df, params.adjusted)

    async with _fetch_semaphore:
        retry_result = await _retry_with_backoff(
            f"fetch_history({params.symbol})",
            _fetch,
        )
        return retry_result.result


async def fetch_history_with_provenance(params: FetchParams) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fetch price history with retry provenance information.

    Same as fetch_history but returns provenance dict for data_provenance field.

    Returns:
        Tuple of (DataFrame, provenance_dict)
    """
    if shutdown_event.is_set():
        raise ServerShuttingDownError("Server is shutting down")

    def _fetch() -> pd.DataFrame:
        df = yf.download(**params.to_yf_kwargs())
        if df.empty:
            raise ValueError(f"No data returned for {params.symbol}")
        return standardize_ohlcv(df, params.adjusted)

    async with _fetch_semaphore:
        retry_result = await _retry_with_backoff(
            f"fetch_history({params.symbol})",
            _fetch,
        )
        return retry_result.result, retry_result.to_provenance()


async def _fetch_info_raw(symbol: str) -> tuple[dict[str, Any], RetryResult]:
    """
    Raw fetch implementation without singleflight.

    Returns:
        Tuple of (info_dict, retry_result with completeness provenance)
    """
    normalized_symbol = symbol.upper().strip()

    def _fetch() -> dict[str, Any]:
        ticker = yf.Ticker(normalized_symbol)
        info = ticker.info
        if not info:
            raise ValueError(f"Invalid symbol: {symbol}")

        # Check completeness using sentinel keys
        completeness = assess_info_completeness(info)

        if completeness.is_incomplete:
            raise YFinanceIncompleteInfoError(
                normalized_symbol,
                key_count=completeness.key_count,
                quote_type=completeness.quote_type,
                sentinel_keys_present=completeness.sentinel_keys_present,
                sentinel_values_present=completeness.sentinel_values_present,
            )
        return info

    async with _fetch_semaphore:
        retry_result = await _retry_with_backoff(
            f"fetch_info({normalized_symbol})",
            _fetch,
        )

        # Add completeness info to result
        info = retry_result.result
        completeness = assess_info_completeness(info)
        retry_result.info_completeness = completeness

        return info, retry_result


async def _fetch_info_with_singleflight(
    symbol: str,
) -> tuple[dict[str, Any], RetryResult, bool]:
    """
    Fetch info with singleflight deduplication.

    Uses singleflight to deduplicate concurrent requests for the same symbol,
    preventing yfinance crumb collisions when multiple tools call fetch_info
    in parallel (e.g., during analyze_stock).

    The singleflight pattern is leak-proof and cancellation-safe:
    - Cleanup happens in the caller's finally block, not the task
    - Only removes entry if it's still THIS task (prevents stomping)
    - Works correctly even if task is cancelled or throws

    Returns:
        Tuple of (info_dict, retry_result, singleflight_joined)
        - singleflight_joined: True if this caller awaited an existing in-flight task
    """
    if shutdown_event.is_set():
        raise ServerShuttingDownError("Server is shutting down")

    normalized_symbol = symbol.upper().strip()
    singleflight_joined = False

    # Singleflight: deduplicate concurrent requests for the same symbol
    # Acquire lock to check/create task atomically
    async with _info_singleflight_lock:
        task = _info_singleflight.get(normalized_symbol)
        if task is None:
            # First caller - create the task
            task = asyncio.create_task(_fetch_info_raw(normalized_symbol))
            _info_singleflight[normalized_symbol] = task
            logger.debug(f"fetch_info({normalized_symbol}): created singleflight task")
        else:
            singleflight_joined = True
            logger.debug(f"fetch_info({normalized_symbol}): joining existing singleflight")

    # Await outside the lock, cleanup in finally
    # Use asyncio.shield() for joiners to prevent a single cancelled waiter
    # from cancelling the shared underlying fetch task
    try:
        if singleflight_joined:
            # Joiner: shield the shared task so our cancellation doesn't propagate
            info, retry_result = await asyncio.shield(task)
        else:
            # Creator: don't shield - if we're cancelled, cancel the task
            info, retry_result = await task
        return info, retry_result, singleflight_joined
    finally:
        # Cleanup: only remove if entry is still THIS task
        # (prevents stomping if a new task was created after this one failed)
        async with _info_singleflight_lock:
            if _info_singleflight.get(normalized_symbol) is task:
                _info_singleflight.pop(normalized_symbol, None)


async def fetch_info(symbol: str) -> dict[str, Any]:
    """
    Fetch stock info (fundamentals, metadata) with retry logic and singleflight.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Info dict from yfinance

    Raises:
        ServerShuttingDownError: If server is shutting down
        YFinanceRetryError: If all retries exhausted for retryable errors
        YFinanceIncompleteInfoError: If response missing fundamental data
        ValueError: If symbol is invalid
    """
    info, _, _ = await _fetch_info_with_singleflight(symbol)
    return info


async def fetch_info_with_provenance(symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fetch stock info with retry and singleflight provenance information.

    Same as fetch_info but returns provenance dict with completeness info
    and singleflight tracking.

    Returns:
        Tuple of (info_dict, provenance_dict with completeness and singleflight fields)
    """
    info, retry_result, singleflight_joined = await _fetch_info_with_singleflight(symbol)
    provenance = retry_result.to_provenance()
    provenance["singleflight_joined"] = singleflight_joined
    return info, provenance


async def fetch_ticker(symbol: str) -> yf.Ticker:
    """
    Get yfinance Ticker object with retry logic.

    Args:
        symbol: Stock ticker symbol

    Returns:
        yfinance Ticker object for further operations

    Raises:
        ServerShuttingDownError: If server is shutting down
        YFinanceRetryError: If all retries exhausted for retryable errors
    """
    if shutdown_event.is_set():
        raise ServerShuttingDownError("Server is shutting down")

    normalized_symbol = symbol.upper().strip()

    def _fetch() -> yf.Ticker:
        return yf.Ticker(normalized_symbol)

    async with _fetch_semaphore:
        retry_result = await _retry_with_backoff(
            f"fetch_ticker({normalized_symbol})",
            _fetch,
        )
        return retry_result.result


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
