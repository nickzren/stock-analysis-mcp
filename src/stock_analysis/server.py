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

INSTRUCTIONS = """Stock research from Yahoo Finance. Informational only, not financial advice; no orders.

Start with `analyze`. A bare ticker ("Analyze HOOD") never triggers a question: lead with mid-term (3-12 months). 12 months / one year means mid-term unless long term is explicit. Ask only for two horizons or a period fitting none ("1-2 months"); one matching period ("hold for five years": long term) needs no question.
- swing, days to weeks: `analyze_trade_setup` or `scan_watchlist`;
- mid-term new money (default): `decision_card.new_money`;
- mid-term holding: `decision_card.if_held`; `analyze_my_position` adds P/L and tax, not the decision;
- long-term new money, 1-5 years: `decision_card.long_term.new_money`.
There is no long-term holder decision; label holder guidance mid-term, never substitute the long-term new-money action. Horizons are independent.

Report each new-money action verbatim with its `blocked_by`, and `decision_card.if_held` action verbatim with its `basis`. Adding requires `add_allowed`; a basis is no permission to add. Never upgrade past `blocked_by`: no narrative, zone, dip, rating or news turns it into a buy. List `unevaluated` gates beside entry as not checked, never cleared. Size only from `decision_card.sizing`, mid-term new money only; there is no long-term or holder size, so never attach it to a long-term or holder answer. When null, say "no entry size: new_money is <action>". `earliest_reassessment` means re-run analyze, not a buy date. `diagnostics`, `action_zones.price_zone`, dip and third-party data (analysts, news, ownership, options) are context, not advice.

Rendering: `stock-analysis://guides/analyze-rendering`."""

mcp = FastMCP(name="stock-analysis", instructions=INSTRUCTIONS)


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
    Use when you have a company name or an uncertain ticker and need the exact symbol
    before calling another tool.

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
    Use when you need only basic facts for a stock (name, sector, price, market cap),
    not a decision.

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
    Use when you need raw price bars or return statistics for a period, not a decision.

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
    Use when you need the technical indicators themselves; for a decision call analyze.

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
    Use when you need the fundamental metrics themselves; for a decision call analyze.

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
    Use when you need earnings dates and their verification, dividends or splits on
    their own.

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
    Use when you need recent headlines or a recent earnings report; news is third-party
    context, not advice.

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
    Use when asked about new money, a holding or the long-term view of a stock.
    A bare ticker means this tool.

    `decision_card` (every detail level) holds `decision_card.new_money`
    (mid-term, 3-12 months) and `decision_card.long_term.new_money` (1-5
    years), each with `blocked_by`, and `decision_card.if_held` for a holder
    (mid-term only). A one-year period defaults to mid-term unless long term is
    explicit; there is no long-term holder decision.
    Report each action verbatim and never upgrade past `blocked_by`. Size only
    from `decision_card.sizing` (mid-term new money only, never long-term or
    holder; null: no entry size). `earliest_reassessment` is a date to re-run
    analyze, not a buy date. Rendering guide:
    `stock-analysis://guides/analyze-rendering`.

    Args:
        symbol: Ticker or company name
        profile: core, balanced or speculative;
            changes sizing only, never the decision
        account_size: Account size for dollar sizing
        risk_per_trade_pct: Risk budget for sizing
        max_position_pct: Single-position cap
        detail: standard (default), decision or full

    Returns:
        JSON analysis payload.
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
    Use when the user asks for a swing trade on one stock (days to weeks, long-only).
    Its action is the swing horizon, separate from analyze's mid-term decision.

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
    Use when the user gives a holding's cost basis and purchase date and wants P/L,
    holding period and tax status. It reads technicals only; the hold/reduce decision is
    analyze's `decision_card.if_held`.

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
    Use when the user gives several holdings with values and wants concentration, sector
    and correlation exposure.

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
    Use when you need insider transactions or institutional holders (third-party data,
    context only).

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
    Use when you need implied volatility, put/call ratios or unusual options activity
    (context only).

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
    Use when the user wants 2-5 stocks side by side on metrics; it applies no decision
    gates.

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
    Use when the user wants to know what changed for a stock since a saved analyze
    snapshot. `new_money_action` and `if_held_action` are the mid-term decision and
    `long_term_action` the long-term one; scan_watchlist's saved state holds swing-card
    actions, so a difference between the two is a different horizon, not a
    contradiction.

    Runs fresh analysis and diffs its watchlist_snapshot (snapshot 3.0.0, the same at
    every analyze detail level) against previous_snapshot. Material changes: price moves
    >5%, score changes >0.10, and any change in new_money_action, if_held_action or
    long_term_action, hard gates (a freshness gate, stale_data or
    freshness_unverifiable, firing or clearing included), gates not evaluated (a gate
    whose input failed is reported as not evaluated, never as cleared), price_zone,
    horizon fit, tilt,
    confidence, risk regime, valuation gate, support, a switch to or from the fallback
    price source, or a new or removed signal. Fetch times, cache hits, ages, price_stale
    and the reassessment date never count. A 1.0.0, 1.1.0 or 2.0.0 snapshot is compared
    on the fields it shares with 3.0.0: its actions and zone are not compared (listed in
    not_comparable and named in the summary), and a version warning lists the retired
    fields and the 3.0.0 rule changes that can move a gate.

    Args:
        symbol: Stock ticker symbol
        previous_snapshot: A previous analyze watchlist_snapshot or detect_changes
            current_snapshot for the same symbol (optional - if omitted, returns
            current snapshot only). A snapshot for another symbol, a 3.0.0
            snapshot missing a field, or a wrong-typed field returns an
            invalid_snapshot error; keys outside the snapshot's version are ignored.

    Returns:
        JSON with changes, material_changes, not_comparable, hash_changed,
        warnings and a narrative summary
    """
    result = await what_changed(symbol=symbol, previous_snapshot=previous_snapshot)
    return _json_response(result)


@mcp.tool
async def check_data_quality(symbols: list[str]) -> str:
    """
    Use when you want to check data availability for symbols before analysis.

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
    Use when the user wants to add, remove or list symbols in the saved watchlist
    (max 25).

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
    Use when the user wants the saved watchlist screened for swing setups (days to
    weeks), with changes since the last scan. Rows and the saved scan state carry
    swing-card actions, not analyze's mid-term decision.

    Two-phase: a cheap daily screen, then the full analyze_trade_setup card only for
    candidates and previously-actionable symbols.

    `changes` lists screen/card transitions since the last scan; `rows` is the full
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
    """Holder decision for an existing position: analyze's if_held, with P/L and tax."""
    result = get_prompt(
        "position_decision",
        {"symbol": symbol, "cost_basis": cost_basis, "purchase_date": purchase_date},
    )
    if result:
        return result["messages"][0]["content"]
    return f"Analyze {symbol} with analyze and read decision_card.if_held."


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    """Run the MCP server."""
    logger.info(f"Starting Stock Analysis MCP Server v{SERVER_VERSION} (schema v{SCHEMA_VERSION})")
    mcp.run()


if __name__ == "__main__":
    main()
