"""Stock Analysis MCP Server using FastMCP."""

import json
import logging
import os
from typing import Any

from fastmcp import FastMCP

from stock_mcp import SCHEMA_VERSION, SERVER_VERSION
from stock_mcp.data.cache import price_cache
from stock_mcp.prompts.templates import get_prompt
from stock_mcp.tools import (
    analyze_position,
    analyze_stock,
    data_quality_report,
    events_calendar,
    fundamentals_snapshot,
    portfolio_exposure,
    price_history,
    stock_news,
    stock_summary,
    symbol_search,
    technicals,
)

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP(
    name="stock-analysis",
)


# ============================================================================
# TOOLS
# ============================================================================


@mcp.tool
async def search_symbol(query: str, limit: int = 10) -> str:
    """
    Search for stock symbols by company name or ticker.

    Args:
        query: Search query (company name or ticker symbol)
        limit: Maximum number of results (default: 10)

    Returns:
        JSON with search results and exact match info
    """
    result = await symbol_search(query=query, limit=limit)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def get_stock_summary(symbol: str) -> str:
    """
    Get basic stock information including name, sector, price, and market cap.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)

    Returns:
        JSON with company info, current price, market cap, and dividend yield
    """
    result = await stock_summary(symbol=symbol)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def get_price_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    adjusted: bool = True,
    include_preview: bool = True,
) -> str:
    """
    Fetch historical price data with summary statistics.

    Args:
        symbol: Stock ticker symbol
        period: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Bar interval - 1m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        adjusted: Use split/dividend adjusted prices (default: true)
        include_preview: Include last 5 bars in response (default: true)

    Returns:
        JSON with price summary, preview bars, and resource URI for full data
    """
    result = await price_history(
        symbol=symbol,
        period=period,
        interval=interval,
        adjusted=adjusted,
        include_preview=include_preview,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def get_technicals(symbol: str) -> str:
    """
    Calculate technical indicators for a stock.

    Includes moving averages (SMA 20/50/200, EMA 12/26), RSI, MACD,
    ATR, 52-week position, and multi-period returns.

    Args:
        symbol: Stock ticker symbol

    Returns:
        JSON with technical indicators and rule-based signals
    """
    result = await technicals(symbol=symbol)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def get_fundamentals(symbol: str) -> str:
    """
    Get fundamental financial metrics for a stock.

    Includes valuation ratios (P/E, P/B, PEG), growth rates,
    profitability margins, financial health metrics, and cash flow.

    Args:
        symbol: Stock ticker symbol

    Returns:
        JSON with fundamental metrics and rule-based signals
    """
    result = await fundamentals_snapshot(symbol=symbol)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def get_events(symbol: str) -> str:
    """
    Get upcoming events and historical earnings for a stock.

    Includes next earnings date, earnings history with beat/miss,
    dividend information, and recent stock splits.

    Args:
        symbol: Stock ticker symbol

    Returns:
        JSON with earnings, dividends, and splits information
    """
    result = await events_calendar(symbol=symbol)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def get_news(symbol: str, days: int = 7) -> str:
    """
    Get recent news and earnings for a stock.

    Fetches news from the past week by default. Also includes recent earnings
    report if one occurred within the lookback period.

    Args:
        symbol: Stock ticker symbol
        days: Number of days to look back (default: 7)

    Returns:
        JSON with news articles and recent earnings (if any within period)
    """
    result = await stock_news(symbol=symbol, days=days)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def analyze(symbol: str) -> str:
    """
    Comprehensive stock analysis aggregating multiple data sources.

    Runs technicals, fundamentals, risk metrics, and events analysis
    in parallel and provides bullish/bearish/neutral signals.

    Args:
        symbol: Stock ticker symbol

    Returns:
        JSON with complete analysis and signals
    """
    result = await analyze_stock(symbol=symbol)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def analyze_my_position(
    symbol: str,
    cost_basis: float,
    purchase_date: str,
    shares: float | None = None,
) -> str:
    """
    Analyze an existing position for hold/sell decision.

    Calculates P/L, tax implications (short vs long-term),
    technical sell signals, and support levels.

    Args:
        symbol: Stock ticker symbol
        cost_basis: Your cost basis per share in dollars
        purchase_date: When you bought it (YYYY-MM-DD format)
        shares: Number of shares owned (optional, for dollar calculations)

    Returns:
        JSON with position analysis, tax info, and sell signals
    """
    result = await analyze_position(
        symbol=symbol,
        cost_basis=cost_basis,
        purchase_date=purchase_date,
        shares=shares,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def analyze_portfolio(positions: list[dict[str, Any]]) -> str:
    """
    Analyze portfolio concentration, sector exposure, and correlation risk.

    Args:
        positions: List of positions, each with 'symbol' and 'value' keys.
                   Example: [{"symbol": "AAPL", "value": 10000}, {"symbol": "GOOGL", "value": 5000}]

    Returns:
        JSON with concentration metrics, sector breakdown, correlation matrix, and liquidity analysis
    """
    result = await portfolio_exposure(positions=positions)
    return json.dumps(result, indent=2, default=str)


@mcp.tool
async def check_data_quality(symbols: list[str]) -> str:
    """
    Check data availability and quality for a list of symbols.

    Useful before running analysis to identify any data gaps.

    Args:
        symbols: List of stock ticker symbols to check

    Returns:
        JSON with per-symbol data quality and summary statistics
    """
    result = await data_quality_report(symbols=symbols)
    return json.dumps(result, indent=2, default=str)


# ============================================================================
# RESOURCES
# ============================================================================


@mcp.resource("price://{symbol}/{period}/{interval}/{adjusted}")
def get_cached_price_data(symbol: str, period: str, interval: str, adjusted: str) -> str:
    """
    Get cached price data as CSV.

    Must call get_price_history first to populate the cache.

    Args:
        symbol: Stock ticker symbol
        period: Time period
        interval: Bar interval
        adjusted: 'adjusted' or 'unadjusted'

    Returns:
        CSV data with date,open,high,low,close,volume columns
    """
    from stock_mcp.utils.validators import FetchParams

    try:
        is_adjusted = adjusted.lower() == "adjusted"
        params = FetchParams(
            symbol=symbol,
            period=period,
            interval=interval,
            adjusted=is_adjusted,
        )
        uri = params.to_uri()
        csv_text = price_cache.get_csv(uri)

        if csv_text is None:
            adj_str = "adjusted=true" if is_adjusted else "adjusted=false"
            return f"Resource not cached. Call get_price_history('{symbol}', '{period}', '{interval}', {adj_str}) first."

        return csv_text
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# PROMPTS
# ============================================================================


@mcp.prompt
def full_analysis(symbol: str) -> str:
    """Comprehensive reproducible stock analysis with consistent JSON output."""
    result = get_prompt("full_analysis", {"symbol": symbol})
    if result:
        return result["messages"][0]["content"]
    return f"Analyze {symbol} using all available tools."


@mcp.prompt
def growth_memo(symbol: str) -> str:
    """Generate a growth investment analysis memo for a stock."""
    result = get_prompt("growth_memo", {"symbol": symbol})
    if result:
        return result["messages"][0]["content"]
    return f"Analyze {symbol} as a growth investment using the analyze tool."


@mcp.prompt
def value_memo(symbol: str) -> str:
    """Generate a value investment analysis memo for a stock."""
    result = get_prompt("value_memo", {"symbol": symbol})
    if result:
        return result["messages"][0]["content"]
    return f"Analyze {symbol} as a value investment using the analyze tool."


@mcp.prompt
def position_decision(symbol: str, cost_basis: str, purchase_date: str) -> str:
    """Generate a hold/sell analysis for an existing position."""
    result = get_prompt(
        "position_decision",
        {"symbol": symbol, "cost_basis": cost_basis, "purchase_date": purchase_date},
    )
    if result:
        return result["messages"][0]["content"]
    return f"Analyze position in {symbol} using analyze_my_position."


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    """Run the MCP server."""
    logger.info(f"Starting Stock Analysis MCP Server v{SERVER_VERSION} (schema v{SCHEMA_VERSION})")
    mcp.run()


if __name__ == "__main__":
    main()
