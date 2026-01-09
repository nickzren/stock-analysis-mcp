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
    Comprehensive stock analysis for mid/long-term investors.

    Runs technicals, fundamentals, risk metrics, events, and news analysis
    in parallel. Returns verdict with decomposed scores, horizon fit assessment,
    valuation-aware action zones, and multi-factor decision context.

    IMPORTANT RENDERING INSTRUCTIONS - Present analysis in this order:

    1. HEADER: Symbol, price, market cap, sector

    2. EXECUTIVE SUMMARY (from executive_summary field):
       - Render VERBATIM as a blockquote or italicized paragraph
       - This is the TL;DR - a 2-4 sentence narrative summary
       - Do NOT paraphrase or rewrite - use the exact text from the field
       - If FCF is mentioned, it must include value+period via fundamentals_summary.cash_flow.free_cash_flow_label
       - Example: "Moderna shows strong technicals (golden cross, above SMAs, +40% in 1 month)
         but faces severe fundamental headwinds..."

    3. VERDICT SUMMARY:
       - Tilt (bullish/neutral/bearish) + confidence level
       - Decomposed scores: setup (technicals), business_quality, risk regime
       - Horizon fit: mid_term + long_term assessments with reasons

    4. HORIZON DRIVERS (from decision_context.horizon_drivers):
       - Render ONLY if non-empty
       - These are policy gates (not score-based) that affect horizon fit
       - Format: "[long_term] gate: reason" or "[mid_term] gate: reason"
       - Example: "[long_term] burn_metrics_missing: unprofitable with runway unknown"
       - Example: "[mid_term] extreme_risk: volatility > 60%"

    5. SCORE MATH (always render with CONSISTENT 6-decimal precision):
       - Use score_display.formula for pre-formatted output
       - Use score_display.component_breakdown for audit trail
       - Example: "Score: 0.050000 = 0.090909 × 0.550000"
       - Example breakdown: "technicals=0.990×0.545455=0.540000 + risk=-0.990×0.454545=-0.450000"
       - INVARIANT: sum(score_delta) == score_raw (use same precision everywhere)
       - 6 decimals avoids rounding artifacts that make sums appear wrong
       - ALWAYS show component_exclusions when coverage < 1.0:
         ```
         Excluded components:
           - fundamentals: {component_exclusions.fundamentals}
         ```
         Possible exclusion reasons:
           - fundamentals_data_unavailable: no fundamentals data fetched
           - fundamentals_not_meaningful_unprofitable: data present but company unprofitable
           - fundamentals_key_inputs_missing: coverage=true but margin/EPS/PE all null
           - no_fundamental_signals_fired: data available but no thresholds triggered

    6. CONFIDENCE PATH (from verdict.confidence_path):
       - Current blockers: list what's preventing higher confidence
       - Upgrade if: list only UNMET conditions (omit any already satisfied)
       - Downgrade if: what would decrease confidence (guidance cut, etc.)

    7. TOP DRIVERS (from decision_context.top_triggers):
       - Show each trigger with category, direction, reason, and score_delta
       - score_delta = actual contribution to final score (not just weight)
       - BALANCE RULE: If tilt=neutral, show top 2 bearish + top 1 bullish
       - If tilt=bullish, still show top bearish driver for balance
       - Include next_update dates for fundamental triggers
       - DEDUPE: Only show one trigger per category (don't show both risk_regime_extreme AND very_high_volatility)

    8. SIGNALS: Bullish (pros) and Bearish (cons) lists

    9. KEY METRICS TABLE with audit fields:
       - P/E or P/S: show value + source (e.g., "6.2x (computed from mcap/rev)")
       - FCF: use fundamentals_summary.cash_flow.free_cash_flow_label (omit if unavailable)
       - Cash runway: show quarters + basis (e.g., "9.1q (min_fcf_ocf)")
       - Quarterly burn: FCF $X, OCF $Y (from burn_metrics)
       - Beta, volatility, drawdown

    10. ACTION ZONES:
        - Current zone + valuation_assessment.gate
        - Price levels with distances (use action_zones.distance_labels; fallback to price_vs_levels)
        - For stop loss, prefer action_zones.level_vs_current_labels.stop_loss ("X% below current")
        - For unprofitable companies, show P/S-based valuation gate
        - If valuation_gate="unknown", add warning: "(valuation confidence reduced)"

    11. DIP ASSESSMENT (from dip_assessment) - FOR BUY-THE-DIP INVESTORS:
        This section helps dip buyers assess entry timing. Render as follows:

        a) DIP CLASSIFICATION:
           ```
           Dip Type: {dip_classification.type}
           {dip_classification.explanation}
           Signals: {dip_classification.signals}
           ```
           Types: falling_knife (avoid), extended_decline (caution), healthy_pullback (favorable),
                  mixed_signals (uncertain), undetermined

        b) DIP DEPTH:
           ```
           Severity: {dip_depth.severity} (basis: {dip_depth.severity_basis})
           From 52W High: {dip_depth.from_52w_high}
           From 6M High: {dip_depth.from_6m_high}
           From 3M High: {dip_depth.from_3m_high}
           From 52W Low: {dip_depth.from_52w_low}
           Days Since 52W High: {dip_depth.days_since_52w_high} (52W high set today if dip_depth.high_set_today)
           Days Since 52W Low: {dip_depth.days_since_52w_low} (52W low set today if dip_depth.low_set_today)
           ```
           Severity levels: none (>=-2%), shallow (>-10%), moderate (>-25%), deep (>-40%), extreme (<=-40%)
           Render percentages as negative values (e.g., -45.2% from high).
           If dip_depth.low_set_today is true, add: "A new 52-week low was set today (intraday)."

        c) OVERSOLD METRICS:
           ```
           Oversold Composite: {oversold_metrics.oversold_composite.level} (score: {oversold_metrics.oversold_composite.score})
           Components: momentum={oversold_metrics.oversold_composite.components.momentum},
                       trend_deviation={oversold_metrics.oversold_composite.components.trend_deviation},
                       range_position={oversold_metrics.oversold_composite.components.range_position}
           Legacy Oversold: {oversold_metrics.level} (score: {oversold_metrics.score})
           RSI: {oversold_metrics.rsi_value} ({oversold_metrics.rsi_status})
           Distance from SMA20: {oversold_metrics.distance_from_sma20}
           Distance from SMA50: {oversold_metrics.distance_from_sma50}
           Distance from SMA200: {oversold_metrics.distance_from_sma200}
           Distance from SMA50 (ATR): {oversold_metrics.distance_from_sma50_atr}
           1W Return Z-Score: {oversold_metrics.return_1w_zscore}
           SMA200 Slope: {oversold_metrics.sma200_slope_pct_per_day}
           Position in 52W Range: {oversold_metrics.position_in_52w_range}
           ```

        d) SUPPORT LEVELS (render as table):
           | Level | Type | Distance | Strength | Status |
           Show closest 4 support levels

        e) VOLUME ANALYSIS:
           ```
           Volume Signal: {volume_analysis.signal}
           Volume Ratio: {volume_analysis.ratio}x average
           {volume_analysis.interpretation}
           ```
           Signals: low_conviction (<0.5x), below_average (0.5-0.9x), normal (0.9-1.5x),
                    above_average (1.5-2x), elevated/potential_capitulation/accumulation (>2x)

        f) BOUNCE POTENTIAL:
           ```
           Rating: {bounce_potential.rating} (score: {bounce_potential.score})
           Factors: {bounce_potential.factors}
           ```

        g) ENTRY TIMING:
           - Entry Signals: list each with signal, action, rationale
           - Wait For: list conditions that would improve entry

        h) DIP CONFIDENCE:
           ```
           Confidence: {dip_confidence.level} (score: {dip_confidence.score})
           Missing: {dip_confidence.missing}
           ```
           Always render DIP CONFIDENCE. If Missing is empty, render "Missing: none".

        i) OVERALL ASSESSMENT (render prominently):
           ```
           Dip Quality: {assessment.dip_quality}
           Recommendation: {assessment.recommendation}
           {assessment.rationale}
           ```
           Recommendations: strong_buy_the_dip, buy_the_dip, cautious_accumulation,
                           small_speculative_position, do_not_catch_falling_knife, wait_for_better_setup

    12. POSITION SIZING (from action_zones.position_sizing_range):
        - Show range as percentage AND dollars (e.g., "0.5%-3% = $250-$1,500")
        - Show shares range at current price (e.g., "~7-42 shares @ $35.66")
        - Show stop-implied max size if stop distance available

    13. DECISION CONTEXT - Render BY CATEGORY with EXPLICIT status fields:
        Render each category as a separate block (not combined) for scanability.

        a) Fundamentals (include business_quality_evidence for transparency):
           ```
           Fundamentals: {business_quality}
             Status: {status} ({status_explanation if present})
             Evidence: {decomposed.business_quality_evidence}
             Bullish if: {bullish_if[0].condition} ({bullish_if[0].current})
             Bullish if: {bullish_if[1].condition} ({bullish_if[1].current})
             Bearish if: {bearish_if[0].condition} ({bearish_if[0].current})
             Next update: {next_update}
           ```
           For unprofitable companies, always show 2-3 checkpoints with current values inline.

        b) Valuation (show BOTH P/E and P/S status explicitly):
           ```
           Valuation: Gate = {current_gate}
             P/E: {pe_status} ({pe_explanation if present})
             P/S: {ps_status} ({ps_explanation if present})
             Bullish if: {bullish_if}
             Bearish if: {bearish_if}
           ```
           INVARIANT: If gate="unknown", ps_status MUST be "unavailable"

        c) Risk:
           ```
           Risk: {current_regime}
             Bullish if: {bullish_if}
             Bearish if: {bearish_if}
           ```

        d) Technicals:
           ```
           Technicals:
             Bullish if: {bullish_if}
             Bearish if: {bearish_if}
           ```

        e) News:
           ```
           News: {current_sentiment} (confidence: {sentiment_confidence})
             Headline triggers: {headline_triggers}
           ```

        f) Next catalyst: earnings date with days countdown

    14. MARKET CONTEXT (from market_context):
        - SPY trend: above/below 200d SMA
        - Provenance: "SPY as_of={as_of} source={source} adjustment={price_adjustment}"
        - If sanity_warnings is non-empty, show: "Warnings: {sanity_warnings}"

    15. NEWS: Recent headlines if available

    For UNPROFITABLE companies specifically:
    - Show P/S instead of P/E in metrics
    - Show burn_metrics: liquidity, runway, burn rates, dilution risk
    - If burn_metrics.status != "available", show status_reason
    - Emphasize path-to-profitability triggers in fundamentals

    Args:
        symbol: Stock ticker symbol

    Returns:
        JSON with complete analysis - render ALL sections per instructions above
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
