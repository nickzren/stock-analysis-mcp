"""Stock Analysis MCP Server using FastMCP."""

import json
import logging
import math
import os
from typing import Any

from fastmcp import FastMCP

from stock_analysis import SCHEMA_VERSION, SERVER_VERSION
from stock_analysis.data.cache import price_cache
from stock_analysis.prompts.templates import get_prompt
from stock_analysis.resources.analyze_guide import read_analyze_rendering_guide
from stock_analysis.tools import (
    analyze_position,
    analyze_stock,
    compare_stocks,
    data_quality_report,
    events_calendar,
    fundamentals_snapshot,
    options_signals,
    ownership_analysis,
    portfolio_exposure,
    price_history,
    stock_news,
    stock_summary,
    symbol_search,
    technicals,
    what_changed,
)
from stock_analysis.tools import (
    analyze_trade_setup as trade_setup_analysis,
)
from stock_analysis.tools import (
    manage_watchlist as watchlist_manage,
)
from stock_analysis.tools import (
    scan_watchlist as watchlist_scan,
)
from stock_analysis.tools.analyze.projection import normalize_analyze_detail, project_analyze_result
from stock_analysis.utils.provenance import build_error_response

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="stock-analysis",
)


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace NaN/Infinity floats with None so strict JSON parsers accept the result."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _json_response(result: dict[str, Any]) -> str:
    """Serialize a tool result using the server's JSON response format.

    Uses allow_nan=False after sanitizing NaN/Infinity to None so the output is
    strict-JSON valid (non-Python parsers reject `NaN` and `Infinity` literals).
    Defaults to compact JSON to reduce MCP response tokens. Set
    STOCK_ANALYSIS_PRETTY_JSON=1 for local pretty-printed debugging.
    """
    sanitized = _sanitize_for_json(result)
    if os.environ.get("STOCK_ANALYSIS_PRETTY_JSON") == "1":
        return json.dumps(sanitized, indent=2, default=str, allow_nan=False)
    return json.dumps(sanitized, separators=(",", ":"), default=str, allow_nan=False)


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
    return _json_response(result)


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
    return _json_response(result)


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
    return _json_response(result)


@mcp.tool
async def get_technicals(symbol: str, timeframe: str = "position") -> str:
    """
    Calculate technical indicators for a stock.

    Includes moving averages (SMA 20/50/200, EMA 12/26), RSI, MACD, ATR,
    52-week position, multi-period returns, and a `short_term` block
    (prior-day/5d/20d levels, swing pivots, gap, RVOL, compression).

    timeframe="swing" adds an `intraday` block (session VWAP, time-adjusted
    RVOL, hourly trend, daily/hourly alignment) with a freshness sub-block.
    Intraday freshness is disclosure, not gating: failures null the dependent
    fields and add warnings instead of blocking the response.

    Args:
        symbol: Stock ticker symbol
        timeframe: "position" (default) or "swing" (adds intraday features)

    Returns:
        JSON with technical indicators and rule-based signals
    """
    result = await technicals(symbol=symbol, timeframe=timeframe)
    return _json_response(result)


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
    return _json_response(result)


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
    return _json_response(result)


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
    return _json_response(result)


@mcp.tool
async def analyze(
    symbol: str,
    profile: str = "balanced",
    account_size: float | None = None,
    risk_per_trade_pct: float | None = None,
    max_position_pct: float | None = None,
    detail: str = "standard",
) -> str:
    """
    Comprehensive single-stock analysis with optional sizing inputs.

    Args:
        symbol: Stock ticker symbol or company name query
        profile: Investor profile preset (core, balanced, or speculative)
        account_size: Optional account size for dollar sizing
        risk_per_trade_pct: Optional risk budget used for sizing
        max_position_pct: Optional max single-position cap
        detail: Response detail level: standard (default), decision, or full

    Returns:
        JSON analysis payload. For detailed presentation guidance, read the MCP
        resource `stock-analysis://guides/analyze-rendering`.
    """
    try:
        normalized_detail = normalize_analyze_detail(detail)
        result = await analyze_stock(
            symbol=symbol,
            profile=profile,
            account_size=account_size,
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
        )
        result = project_analyze_result(result, normalized_detail)
    except ValueError as e:
        result = build_error_response(
            error_type="invalid_parameters",
            message=str(e),
            symbol=symbol,
        )
    return _json_response(result)


@mcp.tool
async def analyze_trade_setup(
    symbol: str,
    account_size: float | None = None,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 10.0,
) -> str:
    """
    Swing-trade setup card for a single stock (days-to-weeks horizon, long-only).

    Detects pullback_in_uptrend / breakout / oversold_mean_reversion setups and
    returns an action (trade_now, enter_on_trigger, watch, no_setup, avoid,
    wait_for_data) with entry trigger, stop, targets, reward/risk, time stop,
    R-based sizing, and blockers. trade_now requires fresh regular-session data.
    Informational only — not financial advice.

    Rendering rules: report `action` verbatim; when action is not trade_now/
    enter_on_trigger there is no plan — never present entry/stop/target/sizing
    structure for it; do not mix `analyze` fields into this card; label any
    forward-looking levels as conditions to watch, not a plan.

    Args:
        symbol: Stock ticker symbol
        account_size: Optional account size in dollars for share sizing (must be > 0)
        risk_per_trade_pct: Percent of account risked between entry and stop, in (0, 100] (default 1.0)
        max_position_pct: Hard cap on position size as percent of account, in (0, 100] (default 10.0)

    Returns:
        JSON trade-setup card
    """
    result = await trade_setup_analysis(
        symbol=symbol,
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
    )
    return _json_response(result)


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
    return _json_response(result)


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
    return _json_response(result)


@mcp.tool
async def get_ownership(symbol: str) -> str:
    """
    Get insider transactions and institutional ownership for a stock.

    Includes insider buy/sell activity (3/6/12 month aggregates),
    top 5 recent transactions, insider sentiment, and top 10 institutional holders.

    Args:
        symbol: Stock ticker symbol

    Returns:
        JSON with insider activity, institutional holders, and ownership summary
    """
    result = await ownership_analysis(symbol=symbol)
    return _json_response(result)


@mcp.tool
async def get_options_signals(symbol: str) -> str:
    """
    Get options-derived signals for a stock.

    Computes ATM implied volatility, IV/HV ratio, put/call ratios
    (volume and open interest based), and flags unusual options activity.

    Args:
        symbol: Stock ticker symbol

    Returns:
        JSON with implied volatility, put/call ratios, and unusual activity flags
    """
    result = await options_signals(symbol=symbol)
    return _json_response(result)


@mcp.tool
async def compare(symbols: list[str]) -> str:
    """
    Compare 2-5 stocks side by side across key metrics.

    Compares valuation, growth, profitability, risk, technicals, and yield.
    Ranks each stock per metric (direction-aware) and computes composite ranking.

    Args:
        symbols: List of 2-5 stock ticker symbols to compare

    Returns:
        JSON with comparison table, rankings, and per-symbol summaries
    """
    result = await compare_stocks(symbols=symbols)
    return _json_response(result)


@mcp.tool
async def detect_changes(
    symbol: str,
    previous_snapshot: dict[str, Any] | None = None,
) -> str:
    """
    Detect material changes for a stock since a previous snapshot.

    Runs fresh analysis and diffs against previous watchlist_snapshot.
    Identifies price moves >5%, score changes >0.10, tilt/zone/regime shifts,
    and new or removed signals.

    Args:
        symbol: Stock ticker symbol
        previous_snapshot: Previous watchlist_snapshot dict (optional - if omitted, returns current snapshot only)

    Returns:
        JSON with changes summary, key_changes list, and investment impact narrative
    """
    result = await what_changed(symbol=symbol, previous_snapshot=previous_snapshot)
    return _json_response(result)


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
    return _json_response(result)


@mcp.tool
async def manage_watchlist(action: str, symbols: list[str] | None = None) -> str:
    """
    Manage the persisted watchlist (max 25 symbols).

    Storage: $STOCK_ANALYSIS_DATA_DIR, else $XDG_DATA_HOME/stock-analysis,
    else ~/.local/share/stock-analysis.

    Args:
        action: add | remove | list
        symbols: Symbols for add/remove (normalized and deduplicated)

    Returns:
        JSON with the resulting watchlist, count, and warnings
    """
    result = await watchlist_manage(action=action, symbols=symbols)
    return _json_response(result)


@mcp.tool
async def scan_watchlist(
    account_size: float | None = None,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 10.0,
) -> str:
    """
    Scan the persisted watchlist for swing setups (two-phase: cheap daily
    screen, full analyze_trade_setup card only for candidates and
    previously-actionable symbols).

    `changes` lists card transitions since the last scan; `rows` is the full
    current state. Informational only — not financial advice.

    Args:
        account_size: Optional account size in dollars for card sizing (> 0)
        risk_per_trade_pct: Percent of account risked per trade, in (0, 100]
        max_position_pct: Position cap as percent of account, in (0, 100]

    Returns:
        JSON scan report (changes, rows, warnings, errors)
    """
    result = await watchlist_scan(
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
    )
    return _json_response(result)


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
    from stock_analysis.utils.validators import FetchParams

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


@mcp.resource("stock-analysis://guides/analyze-rendering")
def get_analyze_rendering_guide() -> str:
    """
    Get the detailed rendering guide for presenting `analyze` output.

    This guide is intentionally exposed as an on-demand resource instead of
    being embedded in the `analyze` tool description.
    """
    guide, _mime_type = read_analyze_rendering_guide()
    return guide


# ============================================================================
# PROMPTS
# ============================================================================


@mcp.prompt
def full_analysis(symbol: str) -> str:
    """Comprehensive reproducible stock analysis with consistent JSON output."""
    result = get_prompt("full_analysis", {"symbol": symbol})
    if result:
        return result["messages"][0]["content"]
    return f"Analyze {symbol} using the analyze tool."


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
