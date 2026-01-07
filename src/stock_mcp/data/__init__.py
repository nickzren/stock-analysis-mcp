"""Data layer for fetching and caching stock data."""

from stock_mcp.data.cache import PriceCache, price_cache
from stock_mcp.data.yfinance_client import (
    INFO_CORE_FUND_SENTINELS,
    INFO_FUND_SENTINELS,
    InfoCompleteness,
    RetryResult,
    YFinanceIncompleteInfoError,
    YFinanceRetryError,
    assess_info_completeness,
    fetch_history,
    fetch_history_with_provenance,
    fetch_info,
    fetch_info_with_provenance,
    fetch_ticker,
    get_market_state,
    shutdown_executor,
)

__all__ = [
    # Cache
    "PriceCache",
    "price_cache",
    # yfinance
    "INFO_CORE_FUND_SENTINELS",
    "INFO_FUND_SENTINELS",
    "InfoCompleteness",
    "RetryResult",
    "YFinanceIncompleteInfoError",
    "YFinanceRetryError",
    "assess_info_completeness",
    "fetch_history",
    "fetch_history_with_provenance",
    "fetch_info",
    "fetch_info_with_provenance",
    "fetch_ticker",
    "get_market_state",
    "shutdown_executor",
]
