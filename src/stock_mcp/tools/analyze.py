"""Analyze stock aggregator tool."""

import asyncio
from time import perf_counter
from typing import Any

from stock_mcp.tools.events import events_calendar
from stock_mcp.tools.fundamentals import fundamentals_snapshot
from stock_mcp.tools.news import stock_news
from stock_mcp.tools.risk_metrics import risk_metrics
from stock_mcp.tools.stock_summary import stock_summary
from stock_mcp.tools.technicals import technicals
from stock_mcp.utils.provenance import build_meta

TIMEOUT_SECONDS = 10.0


async def analyze_stock(symbol: str) -> dict[str, Any]:
    """
    Aggregate analysis from multiple tools with parallel execution.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Comprehensive analysis with data from all tools
    """
    start_time = perf_counter()
    normalized_symbol = symbol.upper().strip()

    # Define tools to run - names MUST match actual function names
    tool_specs = [
        ("stock_summary", stock_summary(normalized_symbol)),
        ("technicals", technicals(normalized_symbol)),
        ("fundamentals_snapshot", fundamentals_snapshot(normalized_symbol)),
        ("risk_metrics", risk_metrics(normalized_symbol)),
        ("events_calendar", events_calendar(normalized_symbol)),
        ("stock_news", stock_news(normalized_symbol)),
    ]

    async def run_with_timing(
        name: str, coro: Any
    ) -> tuple[str, Any | Exception, float]:
        tool_start = perf_counter()
        try:
            result = await asyncio.wait_for(coro, timeout=TIMEOUT_SECONDS)
            duration = (perf_counter() - tool_start) * 1000
            return (name, result, duration)
        except asyncio.TimeoutError:
            duration = (perf_counter() - tool_start) * 1000
            return (name, TimeoutError(f"exceeded {TIMEOUT_SECONDS}s"), duration)
        except Exception as e:
            duration = (perf_counter() - tool_start) * 1000
            return (name, e, duration)

    results = await asyncio.gather(
        *[run_with_timing(name, coro) for name, coro in tool_specs]
    )

    total_duration = (perf_counter() - start_time) * 1000

    # Process results
    tool_results: dict[str, dict[str, Any] | None] = {}
    tool_failures: list[dict[str, Any]] = []
    tool_timings: dict[str, float] = {}
    data_provenance: dict[str, Any] = {}

    for name, result, duration_ms in results:
        tool_timings[name] = round(duration_ms, 1)

        if isinstance(result, Exception):
            tool_failures.append(
                {
                    "tool": name,
                    "error": type(result).__name__,
                    "message": str(result),
                    "duration_ms": round(duration_ms, 1),
                }
            )
            tool_results[name] = None
        elif isinstance(result, dict) and result.get("error"):
            tool_failures.append(
                {
                    "tool": name,
                    "error": result.get("error_type", "unknown"),
                    "message": result.get("message", ""),
                    "duration_ms": round(duration_ms, 1),
                }
            )
            tool_results[name] = None
        else:
            tool_results[name] = result
            # Collect provenance from successful tools
            if isinstance(result, dict) and "data_provenance" in result:
                data_provenance.update(result["data_provenance"])

    # Build response from available data
    summary_data = tool_results.get("stock_summary") or {}
    tech_data = tool_results.get("technicals") or {}
    fund_data = tool_results.get("fundamentals_snapshot") or {}
    risk_data = tool_results.get("risk_metrics") or {}
    events_data = tool_results.get("events_calendar") or {}
    news_data = tool_results.get("stock_news") or {}

    # Summary section
    summary = {
        "name": summary_data.get("name"),
        "sector": summary_data.get("sector"),
        "market_cap": summary_data.get("market_cap"),
        "current_price": summary_data.get("current_price") or tech_data.get("current_price"),
    }

    # Technicals summary
    ma = tech_data.get("moving_averages", {})
    rsi = tech_data.get("rsi", {})
    macd = tech_data.get("macd", {})
    returns = tech_data.get("returns", {})
    price_pos = tech_data.get("price_position", {})

    technicals_summary = {
        "trend": {
            "above_sma50": _get_rule_triggered(ma, "above_sma50"),
            "above_sma200": _get_rule_triggered(ma, "above_sma200"),
            "golden_cross": _get_rule_triggered(ma, "golden_cross"),
        },
        "momentum": {
            "rsi": rsi.get("value"),
            "rsi_overbought": _get_rule_triggered(rsi, "overbought"),
            "rsi_oversold": _get_rule_triggered(rsi, "oversold"),
            "macd_bullish": _get_rule_triggered(macd, "bullish_cross"),
        },
        "returns": {
            "return_1m": returns.get("return_1m"),
            "return_3m": returns.get("return_3m"),
            "return_1y": returns.get("return_1y"),
        },
        "position_in_52w_range": price_pos.get("position_in_range"),
    }

    # Fundamentals summary
    val = fund_data.get("valuation", {})
    growth = fund_data.get("growth", {})
    profit = fund_data.get("profitability", {})
    health = fund_data.get("financial_health", {})
    cf = fund_data.get("cash_flow", {})

    fundamentals_summary = {
        "valuation": {
            "pe_trailing": val.get("pe_trailing"),
            "peg_ratio": val.get("peg_ratio"),
        },
        "growth": {
            "revenue_yoy": growth.get("revenue_yoy"),
            "eps_yoy": growth.get("eps_yoy"),
        },
        "profitability": {
            "net_margin": profit.get("net_margin"),
            "fcf_positive": _get_rule_triggered(cf, "positive_fcf"),
        },
        "health": {
            "debt_to_equity": health.get("debt_to_equity"),
            "net_cash_positive": _get_rule_triggered(health, "net_cash_positive"),
        },
    }

    # Risk summary
    vol = risk_data.get("volatility", {})
    beta = risk_data.get("beta", {})
    dd = risk_data.get("drawdown", {})
    atr = risk_data.get("atr", {})

    risk_summary = {
        "beta": beta.get("value"),
        "annualized_volatility": vol.get("annualized"),
        "max_drawdown_1y": dd.get("max_1y"),
        "atr_pct": atr.get("as_pct_of_price"),
    }

    # Events summary (next earnings only - recent earnings in news_summary)
    earnings = events_data.get("earnings", {})

    events_summary = {
        "days_to_earnings": earnings.get("days_until"),
    }

    # News summary - brief extract of key info
    articles = news_data.get("articles", [])
    recent_earnings = news_data.get("recent_earnings")

    # Extract just headlines (titles only)
    headlines = [article.get("title") for article in articles[:5] if article.get("title")]

    # Recent earnings highlight (if any)
    earnings_highlight: dict[str, Any] | None = None
    if recent_earnings:
        earnings_highlight = {
            "date": recent_earnings.get("date"),
            "beat_miss": recent_earnings.get("beat_miss"),
            "surprise_pct": recent_earnings.get("surprise_pct"),
        }

    news_summary = {
        "article_count": len(articles),
        "headlines": headlines,
        "recent_earnings": earnings_highlight,
    }

    # Generate signals (all signals, no strategy bias)
    signals = _generate_signals(
        technicals=tech_data,
        fundamentals=fund_data,
        risk=risk_data,
    )

    # Calculate data quality
    total_fields = 20  # Approximate count of key fields
    available_fields = sum(
        1
        for v in [
            summary.get("current_price"),
            technicals_summary["trend"]["above_sma50"],
            technicals_summary["momentum"]["rsi"],
            fundamentals_summary["valuation"]["pe_trailing"],
            risk_summary["beta"],
        ]
        if v is not None
    )
    completeness = available_fields / 5  # Based on 5 key checks

    missing_critical: list[str] = []
    if summary.get("current_price") is None:
        missing_critical.append("current_price")
    if risk_summary["beta"] is None:
        missing_critical.append("beta")

    warnings: list[str] = []
    if earnings.get("days_until") and earnings.get("days_until") <= 7:
        warnings.append("earnings_within_7_days")

    data_quality = {
        "completeness": round(completeness, 2),
        "missing_critical": missing_critical,
        "tool_failures": tool_failures,
        "tool_timings": tool_timings,
        "warnings": warnings,
    }

    return {
        "meta": build_meta("analyze_stock", total_duration),
        "data_provenance": data_provenance,
        "symbol": normalized_symbol,
        "summary": summary,
        "technicals_summary": technicals_summary,
        "fundamentals_summary": fundamentals_summary,
        "risk_summary": risk_summary,
        "events_summary": events_summary,
        "news_summary": news_summary,
        "signals": signals,
        "data_quality": data_quality,
    }


def _get_rule_triggered(data: dict[str, Any], rule_name: str) -> bool | None:
    """Extract triggered value from a rules dict."""
    rules = data.get("rules", {})
    rule = rules.get(rule_name, {})
    return rule.get("triggered")


def _generate_signals(
    technicals: dict[str, Any],
    fundamentals: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, list[str]]:
    """Generate bullish/bearish/neutral signals from all available data."""
    bullish: list[str] = []
    bearish: list[str] = []
    neutral: list[str] = []

    ma = technicals.get("moving_averages", {})
    rsi_data = technicals.get("rsi", {})
    macd_data = technicals.get("macd", {})
    returns = technicals.get("returns", {})

    growth = fundamentals.get("growth", {})
    profit = fundamentals.get("profitability", {})
    health = fundamentals.get("financial_health", {})
    cf = fundamentals.get("cash_flow", {})

    vol = risk.get("volatility", {})
    beta_data = risk.get("beta", {})

    # Technical signals
    if _get_rule_triggered(ma, "above_sma200") is True:
        bullish.append("price_above_sma200")
    elif _get_rule_triggered(ma, "above_sma200") is False:
        bearish.append("price_below_sma200")

    if _get_rule_triggered(ma, "golden_cross") is True:
        bullish.append("golden_cross")
    elif _get_rule_triggered(ma, "death_cross") is True:
        bearish.append("death_cross")

    if _get_rule_triggered(rsi_data, "oversold") is True:
        bullish.append("rsi_oversold")
    elif _get_rule_triggered(rsi_data, "overbought") is True:
        bearish.append("rsi_overbought")

    if _get_rule_triggered(macd_data, "bullish_cross") is True:
        bullish.append("macd_bullish")
    elif _get_rule_triggered(macd_data, "bearish_cross") is True:
        bearish.append("macd_bearish")

    # Returns-based signals
    ret_3m = returns.get("return_3m")
    if ret_3m is not None:
        if ret_3m > 0.15:
            bullish.append("strong_3m_momentum")
        elif ret_3m < -0.15:
            bearish.append("weak_3m_momentum")
        else:
            neutral.append("moderate_3m_momentum")

    # Growth signals
    if _get_rule_triggered(growth, "high_growth") is True:
        bullish.append("high_revenue_growth")
    elif _get_rule_triggered(growth, "positive_revenue_growth") is False:
        bearish.append("negative_revenue_growth")

    # Value/health signals
    if _get_rule_triggered(health, "net_cash_positive") is True:
        bullish.append("net_cash_positive")
    if _get_rule_triggered(health, "low_debt") is True:
        bullish.append("low_debt")

    # Profitability signals
    if _get_rule_triggered(profit, "profitable") is True:
        bullish.append("profitable")
    elif _get_rule_triggered(profit, "profitable") is False:
        bearish.append("unprofitable")

    if _get_rule_triggered(cf, "positive_fcf") is True:
        bullish.append("positive_free_cash_flow")
    elif _get_rule_triggered(cf, "positive_fcf") is False:
        bearish.append("negative_free_cash_flow")

    # Risk signals
    if _get_rule_triggered(vol, "high_volatility") is True:
        bearish.append("high_volatility")

    beta_val = beta_data.get("value")
    if beta_val is not None:
        if beta_val > 1.5:
            neutral.append("very_high_beta")
        elif beta_val < 0.5:
            neutral.append("low_beta_defensive")

    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
    }
