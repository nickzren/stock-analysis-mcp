"""Analyze stock aggregator tool."""

import asyncio
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

from stock_mcp.tools.events import events_calendar
from stock_mcp.tools.fundamentals import fundamentals_snapshot
from stock_mcp.tools.news import stock_news
from stock_mcp.tools.risk_metrics import risk_metrics
from stock_mcp.tools.stock_summary import stock_summary
from stock_mcp.tools.technicals import technicals
from stock_mcp.utils.normalize import build_watchlist_snapshot
from stock_mcp.utils.provenance import build_meta

TIMEOUT_SECONDS = 10.0


def _round_or_none(x: float | None, ndigits: int = 0) -> float | None:
    """Round to ndigits or return None. Handles 0 correctly (unlike truthiness)."""
    if x is None:
        return None
    return round(x, ndigits)


def _is_pos(x: float | None) -> bool:
    """Check if value is positive (not None and > 0)."""
    return x is not None and x > 0


# Volatility regime thresholds
VOLATILITY_REGIME_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.40,
    "high": 0.60,
}


def _format_cashflow_value(value: float | None, currency: str | None = None) -> str | None:
    """Format cash flow values with sign, scale, and currency."""
    if value is None:
        return None
    sign = "+" if value > 0 else "-" if value < 0 else ""
    abs_val = abs(value)

    if abs_val >= 1e9:
        scaled = abs_val / 1e9
        unit = "B"
        decimals = 1
    elif abs_val >= 1e6:
        scaled = abs_val / 1e6
        unit = "M"
        decimals = 0 if abs_val >= 1e8 else 1
    elif abs_val >= 1e3:
        scaled = abs_val / 1e3
        unit = "K"
        decimals = 0
    else:
        scaled = abs_val
        unit = ""
        decimals = 0

    number = f"{scaled:.{decimals}f}{unit}"
    if currency and currency != "USD":
        return f"{sign}{currency} {number}"
    return f"{sign}${number}"


def _format_fcf_label(
    value: float | None,
    period: str | None,
    currency: str | None,
    period_end: str | None = None,
) -> str | None:
    """Format FCF value with period for reporting."""
    value_str = _format_cashflow_value(value, currency)
    if value_str is None:
        return None
    period_label = period or "TTM"
    end_label = f" (end {period_end})" if period_end else ""
    return f"FCF ({period_label}): {value_str}{end_label}"


def _format_level_distance_label(pct: float | None) -> str | None:
    """Format level distance relative to current price."""
    if pct is None:
        return None
    if abs(pct) < 0.0005:
        return "at current"
    direction = "above" if pct > 0 else "below"
    return f"{abs(pct) * 100:.1f}% {direction} current"


def _vol_threshold_for_improvement(
    risk_regime: str | None,
    annualized_vol: float | None,
) -> float | None:
    """Return volatility threshold that improves the current regime."""
    if risk_regime == "extreme":
        return VOLATILITY_REGIME_THRESHOLDS["high"]
    if risk_regime == "high":
        return VOLATILITY_REGIME_THRESHOLDS["medium"]
    if risk_regime == "medium":
        return VOLATILITY_REGIME_THRESHOLDS["low"]
    if annualized_vol is None:
        return None
    if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["high"]:
        return VOLATILITY_REGIME_THRESHOLDS["high"]
    if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["medium"]:
        return VOLATILITY_REGIME_THRESHOLDS["medium"]
    if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["low"]:
        return VOLATILITY_REGIME_THRESHOLDS["low"]
    return None


def _build_oversold_composite(
    rsi: float | None,
    return_1w_zscore: float | None,
    distance_to_sma50_atr: float | None,
    position_in_range: float | None,
) -> dict[str, Any]:
    """Build oversold composite score from de-duplicated buckets."""
    oversold_notes: list[str] = []
    rsi_score = 0.0
    if rsi is not None:
        if rsi < 25:
            rsi_score = 2.0
        elif rsi < 30:
            rsi_score = 1.5
        elif rsi < 35:
            rsi_score = 1.0
    elif return_1w_zscore is None:
        oversold_notes.append("momentum_missing")

    z_score = 0.0
    if return_1w_zscore is not None:
        if return_1w_zscore <= -2.0:
            z_score = 1.5
        elif return_1w_zscore <= -1.5:
            z_score = 1.0
        elif return_1w_zscore <= -1.0:
            z_score = 0.5

    momentum_score = max(rsi_score, z_score)

    trend_deviation = 0.0
    if distance_to_sma50_atr is not None:
        if distance_to_sma50_atr <= -2.0:
            trend_deviation = 2.0
        elif distance_to_sma50_atr <= -1.0:
            trend_deviation = 1.0
    else:
        oversold_notes.append("trend_deviation_missing")

    range_position = 0.0
    if position_in_range is not None:
        if position_in_range <= 0.05:
            range_position = 1.0
        elif position_in_range <= 0.15:
            range_position = 0.5
    else:
        oversold_notes.append("range_position_missing")

    optional_band = 0.0

    oversold_raw = momentum_score + trend_deviation + range_position + optional_band
    oversold_composite_score = min(5.0, oversold_raw)
    if oversold_composite_score >= 4.0:
        oversold_composite_level = "extreme"
    elif oversold_composite_score >= 2.0:
        oversold_composite_level = "oversold"
    elif oversold_composite_score >= 1.0:
        oversold_composite_level = "mild"
    else:
        oversold_composite_level = "not_oversold"

    return {
        "score": round(oversold_composite_score, 2),
        "level": oversold_composite_level,
        "components": {
            "momentum": round(momentum_score, 2),
            "trend_deviation": round(trend_deviation, 2),
            "range_position": round(range_position, 2),
            "optional_band": round(optional_band, 2),
        },
        "cap": 5.0,
        "notes": oversold_notes or [],
    }

# Weights for verdict scoring (mid/long-term investor bias)
# fundamentals > technicals > risk
# Note: news sentiment is shown but not scored (keyword-based sentiment unreliable for mid/long-term)
VERDICT_WEIGHTS = {
    "fundamentals": 0.45,
    "technicals": 0.30,
    "risk": 0.25,
}


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
        except TimeoutError:
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

    # Calculate cash runway for unprofitable companies (quarters of runway)
    # Prefer cash + ST investments for more accurate liquidity
    liquidity = health.get("cash_and_st_investments") or health.get("total_cash")
    fcf_ttm = cf.get("free_cash_flow_ttm")
    operating_cf_ttm = cf.get("operating_cf_ttm")
    market_cap = summary_data.get("market_cap")

    # Compute runway from BOTH FCF and OCF, use conservative min
    runway_quarters_fcf: float | None = None
    runway_quarters_ocf: float | None = None
    quarterly_fcf_burn: float | None = None
    quarterly_ocf_burn: float | None = None

    if fcf_ttm is not None and fcf_ttm < 0:
        quarterly_fcf_burn = abs(fcf_ttm) / 4
        if liquidity is not None and quarterly_fcf_burn > 0:
            runway_quarters_fcf = liquidity / quarterly_fcf_burn

    if operating_cf_ttm is not None and operating_cf_ttm < 0:
        quarterly_ocf_burn = abs(operating_cf_ttm) / 4
        if liquidity is not None and quarterly_ocf_burn > 0:
            runway_quarters_ocf = liquidity / quarterly_ocf_burn

    # Use conservative runway (min of available)
    cash_runway_quarters: float | None = None
    runway_basis: str | None = None
    if runway_quarters_fcf is not None and runway_quarters_ocf is not None:
        cash_runway_quarters = round(min(runway_quarters_fcf, runway_quarters_ocf), 1)
        runway_basis = "min_fcf_ocf"
    elif runway_quarters_fcf is not None:
        cash_runway_quarters = round(runway_quarters_fcf, 1)
        runway_basis = "fcf_only"
    elif runway_quarters_ocf is not None:
        cash_runway_quarters = round(runway_quarters_ocf, 1)
        runway_basis = "ocf_only"

    # Check if company is unprofitable (P/E not meaningful)
    pe_trailing = val.get("pe_trailing")
    net_margin = profit.get("net_margin")
    is_unprofitable_company = (
        pe_trailing is None
        or (net_margin is not None and net_margin < 0)
    )

    # Build valuation dict - include alternate metrics for unprofitable companies
    valuation_summary: dict[str, Any] = {
        "pe_trailing": pe_trailing,
        "peg_ratio": val.get("peg_ratio"),
    }

    # Add P/S, EV/EBITDA, and EV/Sales for unprofitable companies (or always for transparency)
    ps_trailing = val.get("ps_trailing")
    ps_source = val.get("ps_source")  # "direct" or "computed" for auditability
    ps_explanation = val.get("ps_explanation")  # Only present when computed
    ev_to_ebitda = val.get("ev_to_ebitda")
    ev_to_sales = val.get("ev_to_sales")  # Better than P/S when debt/cash material

    if is_unprofitable_company or pe_trailing is None:
        # Essential for unprofitable companies
        valuation_summary["ps_trailing"] = ps_trailing
        valuation_summary["ps_source"] = ps_source  # Audit field: how P/S was derived
        if ps_explanation:
            valuation_summary["ps_explanation"] = ps_explanation  # Show computation basis
        valuation_summary["ev_to_ebitda"] = ev_to_ebitda
        valuation_summary["ev_to_sales"] = ev_to_sales  # EV/Sales for debt/cash aware valuation
        valuation_summary["valuation_note"] = "pe_not_meaningful"

    # Add valuation warning for unprofitable high P/S
    valuation_warnings: list[str] = []
    if is_unprofitable_company and ps_trailing is not None and ps_trailing > 10:
        valuation_warnings.append("unprofitable_high_ps")

    if valuation_warnings:
        valuation_summary["warnings"] = valuation_warnings

    # Add cash runway and dilution risk for unprofitable companies
    # ALWAYS emit burn_metrics for unprofitable companies (with status if data missing)
    burn_metrics: dict[str, Any] | None = None
    if is_unprofitable_company:
        burn_warnings: list[str] = []

        # Determine burn_metrics status
        burn_status: str
        burn_status_reason: str | None = None

        if liquidity is None:
            burn_status = "unavailable"
            burn_status_reason = "missing_liquidity_data"
            burn_warnings.append("liquidity_missing")
        elif fcf_ttm is None and operating_cf_ttm is None:
            burn_status = "unavailable"
            burn_status_reason = "missing_cash_flow_data"
            burn_warnings.append("fcf_and_ocf_missing")
        elif cash_runway_quarters is None:
            # Have liquidity but no negative cash flow (company may be cash flow positive)
            burn_status = "not_applicable"
            burn_status_reason = "cash_flow_not_negative"
        else:
            burn_status = "available"

        # Calculate dilution impact if raise needed
        dilution_analysis: dict[str, Any] | None = None
        target_runway_quarters = 8  # 2 years target runway

        # Use the conservative burn rate for dilution calc
        quarterly_burn_for_dilution = quarterly_fcf_burn or quarterly_ocf_burn

        if (
            cash_runway_quarters is not None
            and cash_runway_quarters < target_runway_quarters
            and market_cap is not None
            and market_cap > 0
            and quarterly_burn_for_dilution is not None
            and quarterly_burn_for_dilution > 0
        ):
            # How much cash needed to reach target runway
            quarters_short = target_runway_quarters - cash_runway_quarters
            raise_needed = quarters_short * quarterly_burn_for_dilution

            # Dilution if raised at current market cap (simplified)
            # Store as decimal (0.123) not percentage (12.3) for schema consistency
            dilution_decimal = raise_needed / market_cap

            # Risk level based on dilution %
            dilution_risk_level: str
            if dilution_decimal > 0.25:
                dilution_risk_level = "severe"
            elif dilution_decimal > 0.15:
                dilution_risk_level = "high"
            elif dilution_decimal > 0.08:
                dilution_risk_level = "moderate"
            else:
                dilution_risk_level = "low"

            dilution_analysis = {
                "raise_needed_for_2y_runway": _round_or_none(raise_needed, 0),
                "dilution_if_raised_today": round(dilution_decimal, 4),  # Decimal, not pct
                "dilution_risk_level": dilution_risk_level,
                "current_market_cap": market_cap,
            }

        # Add warning if only using one burn source
        if runway_basis == "fcf_only":
            burn_warnings.append("using_fcf_only")
        elif runway_basis == "ocf_only":
            burn_warnings.append("using_ocf_only")

        # Determine runway_confidence label
        # high: liquidity + both FCF and OCF present
        # moderate: liquidity + one of FCF/OCF present
        # low: inferred or missing parts
        runway_confidence: str | None = None
        if burn_status == "available":
            has_fcf = fcf_ttm is not None and fcf_ttm < 0
            has_ocf = operating_cf_ttm is not None and operating_cf_ttm < 0
            has_liquidity = liquidity is not None

            if has_liquidity and has_fcf and has_ocf:
                runway_confidence = "high"
            elif has_liquidity and (has_fcf or has_ocf):
                runway_confidence = "moderate"
            else:
                runway_confidence = "low"
        elif burn_status == "not_applicable":
            runway_confidence = None  # Not burning cash
        else:
            runway_confidence = "unknown"

        burn_metrics = {
            "status": burn_status,
            "status_reason": burn_status_reason,
            "liquidity": _round_or_none(liquidity, 0),
            "cash_runway_quarters": cash_runway_quarters,
            "runway_basis": runway_basis,
            "runway_confidence": runway_confidence,
            "quarterly_fcf_burn": _round_or_none(quarterly_fcf_burn, 0),
            "quarterly_ocf_burn": _round_or_none(quarterly_ocf_burn, 0),
            "dilution_analysis": dilution_analysis,
            "warnings": burn_warnings or [],
        }

        # Add dilution risk warning to valuation
        if cash_runway_quarters is not None and cash_runway_quarters < 8:
            if "dilution_risk_elevated" not in valuation_warnings:
                valuation_warnings.append("dilution_risk_elevated")
                valuation_summary["warnings"] = valuation_warnings

    # Get gross_margin for profitability assessment
    # Gross margin is especially important for unprofitable companies to assess path to profitability
    gross_margin = profit.get("gross_margin")
    cash_flow = fund_data.get("cash_flow", {})
    fcf_value = cash_flow.get("free_cash_flow_ttm")
    fcf_period = cash_flow.get("free_cash_flow_period")
    fcf_period_end = cash_flow.get("free_cash_flow_period_end")
    fcf_currency = cash_flow.get("currency")
    fcf_source = cash_flow.get("free_cash_flow_source")
    fcf_label = _format_fcf_label(fcf_value, fcf_period, fcf_currency, fcf_period_end)

    fundamentals_summary = {
        "valuation": valuation_summary,
        "growth": {
            "revenue_yoy": growth.get("revenue_yoy"),
            "eps_yoy": growth.get("eps_yoy"),
        },
        "profitability": {
            "gross_margin": gross_margin,  # Important for path-to-profitability assessment
            "net_margin": net_margin,
            "fcf_positive": _get_rule_triggered(cf, "positive_fcf"),
        },
        "cash_flow": {
            "free_cash_flow_ttm": fcf_value,
            "free_cash_flow_period": fcf_period,
            "free_cash_flow_period_end": fcf_period_end,
            "free_cash_flow_source": fcf_source,
            "currency": fcf_currency,
            "free_cash_flow_label": fcf_label,
        },
        "health": {
            "debt_to_equity": health.get("debt_to_equity"),
            "net_cash_positive": _get_rule_triggered(health, "net_cash_positive"),
        },
        "burn_metrics": burn_metrics,  # Only present for unprofitable companies
    }

    # Risk summary
    vol = risk_data.get("volatility", {})
    beta = risk_data.get("beta", {})
    dd = risk_data.get("drawdown", {})
    atr = risk_data.get("atr", {})

    # Calculate risk regime based on volatility, drawdown, and beta
    risk_regime = _classify_risk_regime(
        annualized_vol=vol.get("annualized"),
        max_drawdown=dd.get("max_1y"),
        beta_val=beta.get("value"),
    )

    risk_summary = {
        "beta": beta.get("value"),
        "annualized_volatility": vol.get("annualized"),
        "max_drawdown_1y": dd.get("max_1y"),
        "atr_pct": atr.get("as_pct_of_price"),
        "risk_regime": risk_regime,
    }

    # Events summary with next catalyst status + reason (always output)
    earnings = events_data.get("earnings", {})
    next_earnings_date = earnings.get("next_date")
    next_earnings_source = earnings.get("next_date_source")
    next_earnings_status = earnings.get("next_date_status", "unavailable")
    next_earnings_reason = earnings.get("next_date_status_reason")

    # Build next_catalyst with explicit status
    next_catalyst: dict[str, Any] = {
        "type": "earnings",
        "status": next_earnings_status,
    }
    if next_earnings_status == "available":
        next_catalyst["date"] = next_earnings_date
        next_catalyst["days_until"] = earnings.get("days_until")
        next_catalyst["source"] = next_earnings_source
    else:
        next_catalyst["reason"] = next_earnings_reason or "calendar_data_unavailable"

    events_summary = {
        "next_catalyst": next_catalyst,
        "days_to_earnings": earnings.get("days_until"),  # Keep for backward compat
    }

    # News summary - brief extract of key info
    articles = news_data.get("articles", [])
    recent_earnings = news_data.get("recent_earnings")
    sentiment_data = news_data.get("sentiment", {})

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

    # Build sentiment summary with recency windows (7d/30d) for investor clarity
    # Always show both windows so investors can weight recent vs longer-term sentiment
    sentiment_summary: dict[str, Any] | None = None
    if sentiment_data:
        sentiment_summary = {
            "overall": sentiment_data.get("overall"),
            "confidence": sentiment_data.get("confidence"),
            "sample_size": len(articles),
            "method": sentiment_data.get("method"),
            # Recency windows - always included when available
            "windows": {
                "7d": {
                    "sentiment": sentiment_data.get("sentiment_7d"),
                    "sample_size": sentiment_data.get("sample_size_7d"),
                    "confidence": sentiment_data.get("confidence_7d"),
                },
                "30d": {
                    "sentiment": sentiment_data.get("sentiment_30d"),
                    "sample_size": sentiment_data.get("sample_size_30d"),
                    "confidence": sentiment_data.get("confidence_30d"),
                },
            },
        }

    news_summary = {
        "article_count": len(articles),
        "headlines": headlines,
        "sentiment": sentiment_summary,
        "recent_earnings": earnings_highlight,
    }

    # Generate signals (all signals, no strategy bias)
    signals = _generate_signals(
        technicals=tech_data,
        fundamentals=fund_data,
        risk=risk_data,
    )

    # Build coverage tracking
    coverage = {
        "price": tool_results.get("technicals") is not None,
        "fundamentals": tool_results.get("fundamentals_snapshot") is not None,
        "risk": tool_results.get("risk_metrics") is not None,
        "news": tool_results.get("stock_news") is not None,
        "events": tool_results.get("events_calendar") is not None,
    }

    # Build verdict (score, tilt, confidence, pros/cons)
    verdict = _build_verdict(
        signals=signals,
        fundamentals_data=fund_data,
        risk_data=risk_data,
        technicals_data=tech_data,
        coverage=coverage,
        risk_regime=risk_regime,
        fundamentals_summary=fundamentals_summary,
    )

    # Build action zones (ATR-based price levels)
    action_zones = _build_action_zones(
        current_price=summary.get("current_price"),
        tech_data=tech_data,
        risk_data=risk_data,
        fund_data=fund_data,
        risk_regime=risk_regime,
        signals=signals,
    )

    # Build relative performance vs benchmark
    relative_performance = _build_relative_performance(
        tech_data=tech_data,
        risk_data=risk_data,
    )

    # Extract market context from risk_data (SPY trend)
    market_context = risk_data.get("market_context", {})

    # Build dip assessment for buy-the-dip investors
    dip_assessment = _build_dip_assessment(
        tech_data=tech_data,
        risk_data=risk_data,
        fund_data=fund_data,
        market_context=market_context,
        signals=signals,
    )

    # Align action zones with dip context and risk regime
    action_zones = _apply_dip_gates_to_action_zones(
        action_zones=action_zones,
        dip_assessment=dip_assessment,
        risk_regime=risk_regime,
    )

    # Calculate data quality
    # For unprofitable companies, check P/S instead of P/E (P/E not meaningful)
    val_summary = fundamentals_summary.get("valuation") or {}
    valuation_metric_available = (
        val_summary.get("pe_trailing") is not None
        or val_summary.get("ps_trailing") is not None
    )
    # Use 'is not None' for numeric/bool fields (False/0.0 are valid data)
    available_fields = sum([
        summary.get("current_price") is not None,
        technicals_summary["trend"]["above_sma50"] is not None,
        technicals_summary["momentum"]["rsi"] is not None,
        valuation_metric_available,  # Already a bool
        risk_summary["beta"] is not None,
    ])
    completeness = available_fields / 5  # Based on 5 key checks

    missing_critical: list[str] = []
    if summary.get("current_price") is None:
        missing_critical.append("current_price")
    if risk_summary["beta"] is None:
        missing_critical.append("beta")

    warnings: list[str] = []
    if earnings.get("days_until") and earnings.get("days_until") <= 7:
        warnings.append("earnings_within_7_days")

    # Determine fundamentals status taxonomy
    # missing = not provided by source
    # not_meaningful = provided but not applicable (e.g., P/E invalid due to losses)
    # available = usable data
    fundamentals_status: str
    fundamentals_status_reason: str | None = None

    fund_tool_failed = tool_results.get("fundamentals_snapshot") is None
    pe_value = val_summary.get("pe_trailing")
    profit_summary = fundamentals_summary.get("profitability") or {}
    net_margin_value = profit_summary.get("net_margin")

    if fund_tool_failed:
        fundamentals_status = "missing"
        fundamentals_status_reason = "data_fetch_failed"
    elif pe_value is None and (net_margin_value is not None and net_margin_value < 0):
        fundamentals_status = "not_meaningful"
        fundamentals_status_reason = "unprofitable_pe_invalid"
    elif pe_value is None:
        fundamentals_status = "not_meaningful"
        fundamentals_status_reason = "pe_unavailable"
    else:
        fundamentals_status = "available"

    # Build data_gaps list for transparency
    data_gaps: list[str] = []

    # Check burn_metrics availability for unprofitable companies
    burn_metrics_data = fundamentals_summary.get("burn_metrics") or {}
    burn_metrics_status: str | None = burn_metrics_data.get("status")
    is_unprofitable_company = val_summary.get("valuation_note") == "pe_not_meaningful"

    if is_unprofitable_company and burn_metrics_status == "unavailable":
        data_gaps.append("burn_metrics_unavailable")
        burn_reason = burn_metrics_data.get("status_reason")
        if burn_reason:
            data_gaps.append(f"burn_metrics_reason:{burn_reason}")

    # Check for sparse fundamentals data
    if fundamentals_status == "not_meaningful":
        data_gaps.append("pe_not_meaningful")

    # === STALENESS TRACKING ===
    # Extract as_of from each component's provenance for freshness tracking
    # This lets investors know when data was last fetched
    component_freshness: dict[str, dict[str, Any]] = {}
    staleness_warnings: list[str] = []
    now = datetime.utcnow()

    # Helper to extract as_of and compute staleness
    def _extract_freshness(prov_key: str, component_name: str, stale_threshold_hours: int) -> None:
        prov = data_provenance.get(prov_key, {})
        as_of_str = prov.get("as_of")
        if as_of_str:
            try:
                # Parse ISO format with or without Z suffix
                as_of_str_clean = as_of_str.replace("Z", "+00:00")
                as_of = datetime.fromisoformat(as_of_str_clean).replace(tzinfo=None)
                age_hours = (now - as_of).total_seconds() / 3600
                component_freshness[component_name] = {
                    "as_of": as_of_str,
                    "age_hours": round(age_hours, 1),
                    "stale": age_hours > stale_threshold_hours,
                }
                if age_hours > stale_threshold_hours:
                    staleness_warnings.append(f"{component_name}_stale_{int(age_hours)}h")
            except (ValueError, TypeError):
                component_freshness[component_name] = {"as_of": as_of_str, "parse_error": True}
        else:
            component_freshness[component_name] = {"as_of": None, "status": "unavailable"}

    # Define staleness thresholds per component (in hours)
    # Price/technicals: 48h (weekend tolerance)
    # Fundamentals: 168h (7 days, since it's quarterly data)
    # Risk: 48h (same as price, uses price data)
    # News: 336h (14 days)
    # Events: 168h (7 days)
    _extract_freshness("price", "price", stale_threshold_hours=48)
    _extract_freshness("fundamentals", "fundamentals", stale_threshold_hours=168)
    _extract_freshness("price", "risk", stale_threshold_hours=48)  # Risk uses price data
    _extract_freshness("news", "news", stale_threshold_hours=336)
    _extract_freshness("events", "events", stale_threshold_hours=168)

    # Also track latest news article date (different from fetch time)
    news_articles = news_data.get("articles", [])
    if news_articles:
        try:
            latest_article_date = news_articles[0].get("date")
            if latest_article_date:
                latest_dt = datetime.strptime(latest_article_date, "%Y-%m-%d")
                news_age_days = (now - latest_dt).days
                component_freshness["news"]["latest_article_date"] = latest_article_date
                component_freshness["news"]["latest_article_age_days"] = news_age_days
                if news_age_days > 14:
                    staleness_warnings.append(f"news_content_stale_{news_age_days}d")
        except (ValueError, KeyError, IndexError):
            pass

    data_quality = {
        "completeness": round(completeness, 2),
        "missing_critical": missing_critical,
        "fundamentals_status": fundamentals_status,
        "fundamentals_status_reason": fundamentals_status_reason,
        "data_gaps": data_gaps or [],
        "component_freshness": component_freshness or {},
        "staleness_warnings": staleness_warnings or [],
        "tool_failures": tool_failures,
        "tool_timings": tool_timings,
        "warnings": warnings,
    }

    # Build decision context (what would change the verdict)
    decision_context = _build_decision_context(
        signals=signals,
        tech_data=tech_data,
        risk_data=risk_data,
        events_data=events_data,
        fund_data=fund_data,
        fundamentals_summary=fundamentals_summary,
        action_zones=action_zones,
        news_data=news_data,
        verdict=verdict,
    )

    # Build policy_action - derived suggestion combining all factors
    # This is the primary "what to do" output for investors
    policy_action = _build_policy_action(
        verdict=verdict,
        action_zones=action_zones,
        decomposed=verdict.get("decomposed"),
        risk_regime=risk_regime,
        dip_assessment=dip_assessment,
    )

    # Build executive summary - narrative TL;DR for investors
    # NOTE: Now deterministic - uses fixed templates and references policy_action
    executive_summary = _build_executive_summary(
        summary=summary,
        technicals_summary=technicals_summary,
        fundamentals_summary=fundamentals_summary,
        risk_summary=risk_summary,
        signals=signals,
        verdict=verdict,
        action_zones=action_zones,
        policy_action=policy_action,  # Source of truth for policy sentence
        news_summary=news_summary,
    )

    # Build raw result first
    raw_result = {
        "meta": build_meta("analyze_stock", total_duration),
        "data_provenance": data_provenance,
        "symbol": normalized_symbol,
        "executive_summary": executive_summary,
        "summary": summary,
        "technicals_summary": technicals_summary,
        "fundamentals_summary": fundamentals_summary,
        "risk_summary": risk_summary,
        "events_summary": events_summary,
        "news_summary": news_summary,
        "signals": signals,
        "verdict": verdict,
        "action_zones": action_zones,
        "policy_action": policy_action,
        "relative_performance": relative_performance,
        "market_context": market_context,
        "dip_assessment": dip_assessment,
        "decision_context": decision_context,
        "data_quality": data_quality,
    }

    # Add watchlist_snapshot for diff-stable comparisons
    # This is a normalized, deterministic view optimized for watchlist diffs
    raw_result["watchlist_snapshot"] = build_watchlist_snapshot(raw_result)

    return raw_result


def _get_rule_triggered(data: dict[str, Any], rule_name: str) -> bool | None:
    """Extract triggered value from a rules dict."""
    rules = data.get("rules", {})
    rule = rules.get(rule_name, {})
    triggered = rule.get("triggered")
    if triggered is None:
        return None
    return bool(triggered)


def _classify_risk_regime(
    annualized_vol: float | None,
    max_drawdown: float | None,
    beta_val: float | None,
) -> dict[str, Any]:
    """
    Classify overall risk regime based on volatility, drawdown, and beta.

    Returns a dict with classification (low/medium/high/extreme), factors, and thresholds.

    Risk thresholds:
    - Volatility: low <25%, medium 25-40%, high 40-60%, extreme >60%
    - Drawdown: low >-20%, medium -20% to -35%, high -35% to -50%, extreme <-50%
    - Beta: low <0.8, medium 0.8-1.2, high 1.2-1.8, extreme >1.8
    """
    factors: dict[str, str | None] = {}
    scores: list[int] = []  # 0=low, 1=medium, 2=high, 3=extreme

    # Classify volatility
    if annualized_vol is not None:
        if annualized_vol < VOLATILITY_REGIME_THRESHOLDS["low"]:
            factors["volatility"] = "low"
            scores.append(0)
        elif annualized_vol < VOLATILITY_REGIME_THRESHOLDS["medium"]:
            factors["volatility"] = "medium"
            scores.append(1)
        elif annualized_vol < VOLATILITY_REGIME_THRESHOLDS["high"]:
            factors["volatility"] = "high"
            scores.append(2)
        else:
            factors["volatility"] = "extreme"
            scores.append(3)
    else:
        factors["volatility"] = None

    # Classify drawdown (note: drawdown is negative)
    if max_drawdown is not None:
        if max_drawdown > -0.20:
            factors["drawdown"] = "low"
            scores.append(0)
        elif max_drawdown > -0.35:
            factors["drawdown"] = "medium"
            scores.append(1)
        elif max_drawdown > -0.50:
            factors["drawdown"] = "high"
            scores.append(2)
        else:
            factors["drawdown"] = "extreme"
            scores.append(3)
    else:
        factors["drawdown"] = None

    # Classify beta
    if beta_val is not None:
        if beta_val < 0.8:
            factors["beta"] = "low"
            scores.append(0)
        elif beta_val < 1.2:
            factors["beta"] = "medium"
            scores.append(1)
        elif beta_val < 1.8:
            factors["beta"] = "high"
            scores.append(2)
        else:
            factors["beta"] = "extreme"
            scores.append(3)
    else:
        factors["beta"] = None

    # Overall classification: use max score (most conservative)
    classification: str | None = None
    if scores:
        max_score = max(scores)
        classification = ["low", "medium", "high", "extreme"][max_score]

    return {
        "classification": classification,
        "factors": factors,
        "thresholds": {
            "volatility": {
                "low": f"<{VOLATILITY_REGIME_THRESHOLDS['low'] * 100:.0f}%",
                "medium": (
                    f"{VOLATILITY_REGIME_THRESHOLDS['low'] * 100:.0f}-"
                    f"{VOLATILITY_REGIME_THRESHOLDS['medium'] * 100:.0f}%"
                ),
                "high": (
                    f"{VOLATILITY_REGIME_THRESHOLDS['medium'] * 100:.0f}-"
                    f"{VOLATILITY_REGIME_THRESHOLDS['high'] * 100:.0f}%"
                ),
                "extreme": f">{VOLATILITY_REGIME_THRESHOLDS['high'] * 100:.0f}%",
            },
            "drawdown": {"low": ">-20%", "medium": "-20% to -35%", "high": "-35% to -50%", "extreme": "<-50%"},
            "beta": {"low": "<0.8", "medium": "0.8-1.2", "high": "1.2-1.8", "extreme": ">1.8"},
        },
    }


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
    annualized_vol = vol.get("annualized")
    if annualized_vol is not None:
        if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["high"]:
            bearish.append("very_high_volatility")
        elif _get_rule_triggered(vol, "high_volatility") is True:
            bearish.append("high_volatility")

    # Drawdown signals
    dd = risk.get("drawdown", {})
    max_dd = dd.get("max_1y")
    if max_dd is not None and max_dd < -0.50:
        bearish.append("deep_drawdown")

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


def _build_verdict(
    signals: dict[str, list[str]],
    fundamentals_data: dict[str, Any],
    risk_data: dict[str, Any],
    technicals_data: dict[str, Any],
    coverage: dict[str, bool],
    risk_regime: dict[str, Any] | None = None,
    fundamentals_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build verdict with score, tilt, confidence, and pros/cons.

    Returns score (None if insufficient coverage), tilt, confidence level,
    coverage status, component scores, decomposed scores (setup/business/risk),
    horizon fit assessment, and pros/cons lists.
    """
    if fundamentals_summary is None:
        fundamentals_summary = {}
    pros: list[str] = []
    cons: list[str] = []

    # Map signals to human-readable pros/cons
    signal_prose = {
        # Bullish signals
        "price_above_sma200": "Trading above 200-day moving average (uptrend)",
        "golden_cross": "Golden cross pattern (bullish trend)",
        "rsi_oversold": "RSI indicates oversold conditions",
        "macd_bullish": "MACD showing bullish momentum",
        "strong_3m_momentum": "Strong 3-month price momentum",
        "high_revenue_growth": "High revenue growth (>20% YoY)",
        "net_cash_positive": "Net cash positive balance sheet",
        "low_debt": "Low debt-to-equity ratio",
        "profitable": "Currently profitable",
        # Bearish signals
        "price_below_sma200": "Trading below 200-day moving average (downtrend)",
        "death_cross": "Death cross pattern (bearish trend)",
        "rsi_overbought": "RSI indicates overbought conditions",
        "macd_bearish": "MACD showing bearish momentum",
        "weak_3m_momentum": "Weak 3-month price momentum",
        "negative_revenue_growth": "Negative revenue growth",
        "unprofitable": "Currently unprofitable",
        "high_volatility": "High price volatility",
        "very_high_volatility": "Very high volatility (>60% annualized)",
        "deep_drawdown": "Deep drawdown (>50% from peak)",
    }

    # Build pros from bullish signals
    for sig in signals.get("bullish", []):
        if sig in ("positive_free_cash_flow", "negative_free_cash_flow"):
            continue
        prose = signal_prose.get(sig, sig.replace("_", " ").title())
        if prose not in pros:  # Deduplicate
            pros.append(prose)

    # Build cons from bearish signals
    for sig in signals.get("bearish", []):
        if sig in ("positive_free_cash_flow", "negative_free_cash_flow"):
            continue
        prose = signal_prose.get(sig, sig.replace("_", " ").title())
        if prose not in cons:  # Deduplicate
            cons.append(prose)

    cash_flow = fundamentals_data.get("cash_flow", {})
    fcf_value = cash_flow.get("free_cash_flow_ttm")
    fcf_period = cash_flow.get("free_cash_flow_period")
    fcf_period_end = cash_flow.get("free_cash_flow_period_end")
    fcf_currency = cash_flow.get("currency")
    fcf_label = _format_fcf_label(fcf_value, fcf_period, fcf_currency, fcf_period_end)
    if fcf_label:
        if fcf_value is not None and fcf_value > 0:
            if fcf_label not in pros:
                pros.append(fcf_label)
        elif fcf_value is not None and fcf_value < 0:
            if fcf_label not in cons:
                cons.append(fcf_label)

    # Add fundamental-specific context (beyond signals)
    val = fundamentals_data.get("valuation", {})
    peg = val.get("peg_ratio")
    if peg is not None:
        if peg < 1.0:
            if "PEG ratio suggests undervaluation" not in pros:
                pros.append("PEG ratio suggests undervaluation")
        elif peg > 2.5:
            if "PEG ratio suggests overvaluation" not in cons:
                cons.append("PEG ratio suggests overvaluation")

    health = fundamentals_data.get("financial_health", {})
    debt_to_equity = health.get("debt_to_equity")
    if debt_to_equity is not None and debt_to_equity > 1.5:
        if "High debt levels (D/E > 1.5)" not in cons:
            cons.append("High debt levels (D/E > 1.5)")

    # Calculate component scores
    # Each category: (pros - cons) / max(total, 1) scaled to weight
    tech_signals = ["price_above_sma200", "price_below_sma200", "golden_cross",
                    "death_cross", "rsi_oversold", "rsi_overbought",
                    "macd_bullish", "macd_bearish", "strong_3m_momentum",
                    "weak_3m_momentum"]
    fund_signals = ["high_revenue_growth", "negative_revenue_growth",
                    "net_cash_positive", "low_debt", "profitable",
                    "unprofitable", "positive_free_cash_flow",
                    "negative_free_cash_flow"]
    risk_signals = ["high_volatility", "very_high_volatility", "deep_drawdown"]

    def calc_component_score(signal_list: list[str]) -> float | None:
        """Calculate normalized score for a signal category."""
        bullish_list = signals.get("bullish", [])
        bearish_list = signals.get("bearish", [])

        pos = sum(1 for s in signal_list if s in bullish_list)
        neg = sum(1 for s in signal_list if s in bearish_list)
        total = pos + neg

        if total == 0:
            return None  # No signals in this category
        return (pos - neg) / total

    tech_score = calc_component_score(tech_signals)
    fund_score = calc_component_score(fund_signals)
    risk_score = calc_component_score(risk_signals)

    # Build decomposed scores for investor clarity
    # setup_score: technicals + momentum (is the chart attractive?)
    # business_quality_score: profitability, FCF, growth, health (is the business good?)
    # risk_assessment: regime + volatility + drawdown (how dangerous is it?)

    # Setup score: purely technical/momentum driven
    setup_label: str | None = None
    if tech_score is not None:
        if tech_score > 0.3:
            setup_label = "strong"
        elif tech_score > 0:
            setup_label = "moderate"
        elif tech_score > -0.3:
            setup_label = "weak"
        else:
            setup_label = "poor"

    # Business quality score: fundamentals-driven
    # Requires profitability/FCF data to be meaningful
    business_quality_label: str | None = None
    business_quality_status: str = "unknown"

    profit = fundamentals_data.get("profitability", {})
    cf = fundamentals_data.get("cash_flow", {})
    health = fundamentals_data.get("financial_health", {})

    net_margin = profit.get("net_margin")
    fcf_positive = cf.get("rules", {}).get("positive_fcf", {}).get("triggered")
    fcf_value = cf.get("free_cash_flow_ttm")
    fcf_period = cf.get("free_cash_flow_period")
    fcf_period_end = cf.get("free_cash_flow_period_end")
    fcf_currency = cf.get("currency")
    fcf_value_label = _format_fcf_label(
        fcf_value,
        fcf_period,
        fcf_currency,
        fcf_period_end,
    )
    low_debt = health.get("rules", {}).get("low_debt", {}).get("triggered")

    # Check if we can assess business quality
    # Use multiple signals to detect unprofitability when data is sparse
    valuation_data = fundamentals_data.get("valuation", {})
    pe_trailing = valuation_data.get("pe_trailing")

    # Check for unprofitability signals from signals list
    bearish_signals = signals.get("bearish", [])
    is_signaled_unprofitable = "unprofitable" in bearish_signals
    has_negative_fcf_signal = "negative_free_cash_flow" in bearish_signals

    # Check valuation_note from fundamentals_summary (set in analyze.py)
    # This catches cases where PE is not meaningful due to losses
    valuation_note = fundamentals_summary.get("valuation", {}).get("valuation_note")
    pe_not_meaningful = valuation_note == "pe_not_meaningful"

    # Get trailing EPS for explicit unprofitability check
    trailing_eps = valuation_data.get("trailing_eps")

    # Get FCF from cash_flow section
    fcf_ttm_val = cf.get("free_cash_flow_ttm")

    # Require explicit signals for "unprofitable" label:
    # 1. trailing_eps <= 0
    # 2. net_margin < 0
    # 3. free_cash_flow_ttm < 0 (only if other signals confirm)
    has_explicit_unprofitable_signal = (
        (net_margin is not None and net_margin < 0)
        or (trailing_eps is not None and trailing_eps <= 0)
        or is_signaled_unprofitable  # From fundamentals rules
        or pe_not_meaningful  # valuation_note indicates PE not meaningful
    )

    if not coverage.get("fundamentals", False):
        business_quality_status = "data_missing"
    elif has_explicit_unprofitable_signal:
        # Explicit signal confirms unprofitable - we HAVE evaluated business quality
        # (it's "unprofitable"), not that we couldn't evaluate it
        business_quality_status = "evaluated_unprofitable"
        business_quality_label = "unprofitable"
    elif fcf_ttm_val is not None and fcf_ttm_val < 0 and has_negative_fcf_signal:
        # Negative FCF confirmed by both value and signal
        # If profitability is positive, call this "mixed" (profitability strong, cash conversion weak)
        business_quality_status = "available"
        if net_margin is not None and net_margin > 0:
            business_quality_label = "mixed"
        else:
            business_quality_label = "weak"
    elif net_margin is None and pe_trailing is None and trailing_eps is None:
        # Insufficient data to assess profitability
        business_quality_status = "data_missing"
        business_quality_label = None
    else:
        business_quality_status = "evaluated"
        # Score based on profitability + FCF + health signals
        biz_positives = sum([
            net_margin is not None and net_margin > 0.10,  # Good margin
            fcf_positive is True,
            low_debt is True,
        ])
        biz_negatives = sum([
            net_margin is not None and net_margin < 0,
            fcf_positive is False,
            low_debt is False and health.get("debt_to_equity") is not None and health.get("debt_to_equity") > 1.5,
        ])

        if biz_positives >= 2 and biz_negatives == 0:
            business_quality_label = "strong"
        elif biz_positives >= 1 and biz_negatives <= 1:
            business_quality_label = "moderate"
        elif biz_negatives >= 2:
            business_quality_label = "poor"
        else:
            business_quality_label = "mixed"

    # Risk assessment from regime
    risk_label: str | None = None
    if risk_regime:
        regime_class = risk_regime.get("classification")
        if regime_class:
            risk_label = regime_class  # low/medium/high/extreme

    # Build business_quality_evidence to show basis for classification
    # This prevents confusion when classification doesn't match user expectations
    business_quality_evidence: dict[str, Any] = {}
    if business_quality_status == "evaluated_unprofitable":
        # Show what triggered "unprofitable" classification
        evidence_reasons: list[str] = []
        if net_margin is not None and net_margin < 0:
            evidence_reasons.append(f"net_margin={net_margin*100:.1f}%")
        if trailing_eps is not None and trailing_eps <= 0:
            evidence_reasons.append(f"trailing_eps=${trailing_eps:.2f}")
        if is_signaled_unprofitable:
            evidence_reasons.append("unprofitable_signal_fired")
        if pe_not_meaningful:
            evidence_reasons.append("pe_not_meaningful")
        business_quality_evidence = {
            "basis": "explicit_unprofitable_signals",
            "triggers": evidence_reasons if evidence_reasons else ["inferred_from_data"],
        }
    elif business_quality_status == "data_missing":
        # Show what data is missing
        missing_inputs: list[str] = []
        if net_margin is None:
            missing_inputs.append("net_margin")
        if pe_trailing is None:
            missing_inputs.append("pe_trailing")
        if trailing_eps is None:
            missing_inputs.append("trailing_eps")
        business_quality_evidence = {
            "basis": "insufficient_data",
            "missing_inputs": missing_inputs,
        }
    elif business_quality_status in ("evaluated", "available"):
        # Show what positives/negatives were counted
        evidence_positives: list[str] = []
        evidence_negatives: list[str] = []
        if net_margin is not None:
            if net_margin > 0.10:
                evidence_positives.append(f"net_margin={net_margin*100:.1f}%")
            elif net_margin < 0:
                evidence_negatives.append(f"net_margin={net_margin*100:.1f}%")
        if fcf_positive is True and fcf_value_label:
            evidence_positives.append(fcf_value_label)
        elif fcf_positive is False and fcf_value_label:
            evidence_negatives.append(fcf_value_label)
        if low_debt is True:
            evidence_positives.append("low_debt")
        elif low_debt is False and health.get("debt_to_equity") is not None:
            dte = health.get("debt_to_equity")
            if dte > 1.5:
                evidence_negatives.append(f"high_debt={dte:.1f}x")
        business_quality_evidence = {
            "basis": "fundamentals_scoring",
            "positives": evidence_positives,
            "negatives": evidence_negatives,
        }

    decomposed_scores = {
        "setup": setup_label,
        "business_quality": business_quality_label,
        "business_quality_status": business_quality_status,
        "business_quality_evidence": business_quality_evidence if business_quality_evidence else None,
        "risk": risk_label,
    }

    # Get revenue_yoy and fcf for horizon fit assessment
    growth_data = fundamentals_data.get("growth", {})
    cf_data = fundamentals_data.get("cash_flow", {})
    revenue_yoy_val = growth_data.get("revenue_yoy")
    fcf_ttm_val = cf_data.get("free_cash_flow_ttm")

    # Get burn_metrics for horizon fit
    burn_metrics_for_horizon = fundamentals_summary.get("burn_metrics") or {}
    burn_metrics_status = burn_metrics_for_horizon.get("status")
    runway_confidence = burn_metrics_for_horizon.get("runway_confidence")

    # Build horizon fit assessment
    # mid_term (3-12 months): can lean on setup + regime-aware sizing
    # long_term (1-5 years): requires business quality + not extreme risk + runway
    horizon_fit = _build_horizon_fit(
        setup_label=setup_label,
        business_quality_label=business_quality_label,
        business_quality_status=business_quality_status,
        risk_label=risk_label,
        signals=signals,
        revenue_yoy=revenue_yoy_val,
        fcf_ttm=fcf_ttm_val,
        burn_metrics_status=burn_metrics_status,
        runway_confidence=runway_confidence,
    )

    # Check minimum coverage for scoring
    required_coverage = ["price", "fundamentals"]
    has_minimum_coverage = all(coverage.get(k, False) for k in required_coverage)

    # Calculate weighted overall score
    score: float | None = None
    components: dict[str, float | None] = {
        "technicals": round(tech_score, 3) if tech_score is not None else None,
        "fundamentals": round(fund_score, 3) if fund_score is not None else None,
        "risk": round(risk_score, 3) if risk_score is not None else None,
    }

    # Explain why components are excluded from scoring
    # This prevents confusion when data is available but not used in scoring
    # Split "data availability" from "scoring usage" for investor clarity
    component_exclusions: dict[str, str] = {}
    if tech_score is None:
        component_exclusions["technicals"] = "no_technical_signals_fired"
    if fund_score is None:
        if not coverage.get("fundamentals", False):
            component_exclusions["fundamentals"] = "fundamentals_data_unavailable"
        elif business_quality_status == "evaluated_unprofitable":
            # Data is present, company evaluated as unprofitable, scored with unprofitable_signals_v1
            # This is NOT an exclusion - fundamentals WAS scored, just using different method
            # Only add to exclusions if no signals triggered
            component_exclusions["fundamentals"] = "no_fundamental_signals_fired"
        elif business_quality_status == "data_missing":
            # Coverage says we have fundamentals, but key inputs (margin, EPS, PE) are all None
            component_exclusions["fundamentals"] = "fundamentals_key_inputs_missing"
        else:
            # We have data and it's meaningful, but no signals triggered thresholds
            component_exclusions["fundamentals"] = "no_fundamental_signals_fired"
    if risk_score is None:
        if not coverage.get("price", False):
            component_exclusions["risk"] = "price_data_unavailable"
        else:
            component_exclusions["risk"] = "no_risk_signals_fired"

    # Track weights actually used (renormalized)
    weights_used: dict[str, float | None] = {}
    score_raw: float | None = None
    coverage_factor: float | None = None

    if has_minimum_coverage:
        # Weight available components
        weighted_sum = 0.0
        total_weight = 0.0

        if tech_score is not None:
            weighted_sum += tech_score * VERDICT_WEIGHTS["technicals"]
            total_weight += VERDICT_WEIGHTS["technicals"]
        if fund_score is not None:
            weighted_sum += fund_score * VERDICT_WEIGHTS["fundamentals"]
            total_weight += VERDICT_WEIGHTS["fundamentals"]
        if risk_score is not None:
            weighted_sum += risk_score * VERDICT_WEIGHTS["risk"]
            total_weight += VERDICT_WEIGHTS["risk"]

        if total_weight > 0:
            # Raw score from available components only (keep full precision)
            # INVARIANT: score_raw == sum(component_score * weight_used) for all components
            score_raw = weighted_sum / total_weight

            # Coverage factor: what fraction of full weights are present
            # Full weights = technicals + fundamentals + risk = 0.30 + 0.45 + 0.25 = 1.00
            full_weight = (
                VERDICT_WEIGHTS["technicals"]
                + VERDICT_WEIGHTS["fundamentals"]
                + VERDICT_WEIGHTS["risk"]
            )
            coverage_factor = total_weight / full_weight

            # Attenuate score by coverage factor to prevent partial-data overconfidence
            # This makes scores from partial data naturally smaller
            score = score_raw * coverage_factor

            # Record renormalized weights (what was actually used after renormalization)
            # Keep full precision - rounding happens only at display time
            if tech_score is not None:
                weights_used["technicals"] = VERDICT_WEIGHTS["technicals"] / total_weight
            if fund_score is not None:
                weights_used["fundamentals"] = VERDICT_WEIGHTS["fundamentals"] / total_weight
            if risk_score is not None:
                weights_used["risk"] = VERDICT_WEIGHTS["risk"] / total_weight

            # Invariant check: sum(weights_used) should ≈ 1.0
            weights_sum = sum(weights_used.values())
            if abs(weights_sum - 1.0) > 0.02:  # Allow small rounding error
                # This shouldn't happen, but log it for debugging
                pass  # weights_used sum check failed

    # Determine tilt and confidence
    tilt: str | None = None
    confidence: str = "low"

    # Build inputs_used for auditability (key numeric inputs that drove scoring)
    inputs_used: dict[str, float | None] = {}

    # Get key values from fundamentals
    vol = fundamentals_data.get("valuation", {})
    growth = fundamentals_data.get("growth", {})
    profit = fundamentals_data.get("profitability", {})
    yield_m = fundamentals_data.get("yield_metrics", {})
    health = fundamentals_data.get("financial_health", {})

    inputs_used["pe_trailing"] = vol.get("pe_trailing")
    inputs_used["peg_ratio"] = vol.get("peg_ratio")
    inputs_used["revenue_yoy"] = growth.get("revenue_yoy")
    inputs_used["net_margin"] = profit.get("net_margin")
    inputs_used["fcf_yield"] = yield_m.get("fcf_yield")
    inputs_used["debt_to_equity"] = health.get("debt_to_equity")

    # Get key values from risk data
    risk_vol = risk_data.get("volatility", {})
    risk_dd = risk_data.get("drawdown", {})
    risk_beta = risk_data.get("beta", {})

    inputs_used["annualized_vol"] = risk_vol.get("annualized")
    inputs_used["max_drawdown_1y"] = risk_dd.get("max_1y")
    inputs_used["beta"] = risk_beta.get("value")

    # Check for confidence-limiting factors
    bearish_list = signals.get("bearish", [])
    is_unprofitable = "unprofitable" in bearish_list
    has_negative_fcf = "negative_free_cash_flow" in bearish_list
    has_extreme_risk = "very_high_volatility" in bearish_list or "deep_drawdown" in bearish_list
    fundamentals_available = coverage.get("fundamentals", False)
    risk_available = coverage.get("risk", False)

    if score is not None:
        if score > 0.2:
            tilt = "bullish"
        elif score < -0.2:
            tilt = "bearish"
        else:
            tilt = "neutral"

        # Confidence based on score magnitude, coverage, AND data quality
        coverage_count = sum(1 for v in coverage.values() if v)

        # Start with score-based confidence
        if abs(score) > 0.5 and coverage_count >= 4:
            confidence = "high"
        elif abs(score) > 0.3 and coverage_count >= 3:
            confidence = "moderate"
        else:
            confidence = "low"

        # Cap confidence if critical data missing or company has red flags
        if not fundamentals_available:
            # Can't be high confidence without fundamentals
            if confidence == "high":
                confidence = "moderate"
        if not risk_available:
            # Can't be high confidence without risk data
            if confidence == "high":
                confidence = "moderate"
        if is_unprofitable or has_negative_fcf:
            # Unprofitable companies should not get high confidence bullish
            if tilt == "bullish" and confidence == "high":
                confidence = "moderate"
        if has_extreme_risk:
            # Extreme risk caps confidence
            if confidence == "high":
                confidence = "moderate"

    # Get additional data for condition-based confidence path
    risk_regime_class = risk_regime.get("classification") if risk_regime else None
    burn_status = (fundamentals_summary.get("burn_metrics") or {}).get("status")
    sma_200_val = technicals_data.get("moving_averages", {}).get("sma_200")
    current_price_val = technicals_data.get("price_info", {}).get("current_price")
    max_dd_val = inputs_used.get("max_drawdown_1y")
    ann_vol_val = inputs_used.get("annualized_vol")

    # Build confidence path (what would upgrade/downgrade confidence)
    confidence_path = _build_confidence_path(
        confidence=confidence,
        tilt=tilt,
        fundamentals_available=fundamentals_available,
        risk_available=risk_available,
        is_unprofitable=is_unprofitable,
        has_negative_fcf=has_negative_fcf,
        has_extreme_risk=has_extreme_risk,
        coverage_count=sum(1 for v in coverage.values() if v),
        risk_regime=risk_regime_class,
        burn_metrics_status=burn_status,
        sma_200=sma_200_val,
        current_price=current_price_val,
        max_drawdown=max_dd_val,
        annualized_vol=ann_vol_val,
    )

    # Build score_display for clean presentation
    # Pre-formatted strings to avoid rendering logic in consumers
    score_display: dict[str, str | None] = {}
    if score is not None and score_raw is not None and coverage_factor is not None:
        score_display["score"] = f"{score:.6f}"
        score_display["score_raw"] = f"{score_raw:.6f}"
        score_display["coverage_factor"] = f"{coverage_factor:.6f}"
        score_display["formula"] = f"{score:.6f} = {score_raw:.6f} × {coverage_factor:.6f}"

        # Build component breakdown for audit
        component_parts: list[str] = []
        if weights_used:
            for comp_name in ["technicals", "fundamentals", "risk"]:
                if comp_name in weights_used and components.get(comp_name) is not None:
                    comp_score = components[comp_name]
                    weight = weights_used[comp_name]
                    delta = comp_score * weight  # type: ignore[operator]
                    component_parts.append(f"{comp_name}={comp_score:.3f}×{weight:.6f}={delta:.6f}")
        score_display["component_breakdown"] = " + ".join(component_parts) if component_parts else None
    else:
        score_display = {"score": None, "score_raw": None, "coverage_factor": None, "formula": None, "component_breakdown": None}

    # Build structured coverage that separates "data availability" from "scoring usage"
    # This prevents confusion when coverage=True but component=None
    # Added: in_score_model (is this component part of the scoring model at all?)
    # Added: scoring_method (which scoring algorithm is used for this component?)
    structured_coverage: dict[str, dict[str, Any]] = {}

    # Map coverage keys to component names for exclusion lookup
    coverage_to_component = {
        "price": "technicals",  # price data feeds technicals + risk
        "fundamentals": "fundamentals",
        "risk": "risk",
        "news": None,  # news not used in scoring
        "events": None,  # events not used in scoring
    }

    # Determine scoring method for fundamentals based on business_quality_status
    fundamentals_scoring_method: str | None = None
    if business_quality_status == "evaluated_unprofitable":
        fundamentals_scoring_method = "unprofitable_signals_v1"
    elif business_quality_status in ("evaluated", "available"):
        fundamentals_scoring_method = "fundamentals_signals_v1"
    elif business_quality_status == "data_missing":
        fundamentals_scoring_method = None  # No method - data missing

    for cov_key, cov_fetched in coverage.items():
        component_name = coverage_to_component.get(cov_key)

        # Determine if this component is in the score model (by design, not by data availability)
        # News and events are NOT in the score model - they're contextual only
        in_score_model = component_name is not None

        # Determine scoring_method for each component
        scoring_method: str | None = None
        if component_name == "technicals":
            scoring_method = "technicals_signals_v1"
        elif component_name == "fundamentals":
            scoring_method = fundamentals_scoring_method
        elif component_name == "risk":
            scoring_method = "risk_signals_v1"
        # news and events: scoring_method = None (not scored)

        # Determine if used in score (only applicable to scored components)
        if not in_score_model:
            # Not a scored component (news, events) - explicitly False, not None
            used_in_score = False
            reason_excluded = "not_in_score_model"
        elif cov_fetched:
            # Data was fetched - check if actually used in score
            component_score = components.get(component_name)
            used_in_score = component_score is not None
            reason_excluded = component_exclusions.get(component_name) if not used_in_score else None
        else:
            # Data was not fetched
            used_in_score = False
            reason_excluded = f"{cov_key}_data_unavailable"

        structured_coverage[cov_key] = {
            "fetched": cov_fetched,
            "in_score_model": in_score_model,
            "scoring_method": scoring_method,
            "used_in_score": used_in_score,
            "reason_excluded": reason_excluded,
        }

    verdict = {
        "score": score,
        "score_raw": score_raw,
        "coverage_factor": coverage_factor,
        "score_display": score_display,
        "tilt": tilt,
        "confidence": confidence,
        "confidence_path": confidence_path,
        "coverage": structured_coverage,
        "components": components,
        "component_exclusions": component_exclusions if component_exclusions else None,
        "decomposed": decomposed_scores,
        "horizon_fit": horizon_fit,
        "weights_full": dict(VERDICT_WEIGHTS),  # Original weights for audit
        "weights_used": weights_used if weights_used else None,
        "inputs_used": inputs_used,
        "pros": pros[:5],  # Top 5
        "cons": cons[:5],  # Top 5
        "method": "weighted_signals_v2",
    }

    # Validate invariants (debug mode only, prevents disagreement between fields)
    _validate_verdict_invariants(verdict)

    return verdict


def _validate_verdict_invariants(verdict: dict[str, Any]) -> None:
    """
    Validate invariants between coverage, weights_used, components, and coverage_factor.

    Invariants enforced:
    1. If coverage.<component>.used_in_score == True, components.<component> must be non-null
    2. If components.<component> is None, coverage.<component>.used_in_score must be False/None
    3. weights_used keys must match components with non-null scores
    4. coverage_factor must equal sum of weights for used components / total weights

    Logs warnings for violations rather than raising (production-safe).
    """
    import logging
    logger = logging.getLogger(__name__)

    coverage = verdict.get("coverage", {})
    components = verdict.get("components", {})
    weights_used = verdict.get("weights_used") or {}
    coverage_factor = verdict.get("coverage_factor")

    # Map coverage keys to component names
    coverage_to_component = {
        "price": "technicals",
        "fundamentals": "fundamentals",
        "risk": "risk",
    }

    violations: list[str] = []

    # Invariant 1 & 2: coverage.used_in_score must agree with components
    for cov_key, comp_name in coverage_to_component.items():
        cov_data = coverage.get(cov_key, {})
        used_in_score = cov_data.get("used_in_score")
        comp_score = components.get(comp_name)

        if used_in_score is True and comp_score is None:
            violations.append(
                f"coverage.{cov_key}.used_in_score=True but components.{comp_name}=None"
            )
        if comp_score is not None and used_in_score is False:
            violations.append(
                f"components.{comp_name}={comp_score} but coverage.{cov_key}.used_in_score=False"
            )

    # Invariant 3: weights_used keys must match non-null component scores
    for comp_name, comp_score in components.items():
        if comp_score is not None:
            if comp_name not in weights_used:
                violations.append(
                    f"components.{comp_name}={comp_score} but {comp_name} not in weights_used"
                )
        else:
            if comp_name in weights_used:
                violations.append(
                    f"components.{comp_name}=None but {comp_name} in weights_used"
                )

    # Invariant 4: coverage_factor consistency (if calculable)
    if coverage_factor is not None and weights_used:
        expected_coverage = sum(weights_used.values())
        # Allow small floating point tolerance
        if abs(expected_coverage - coverage_factor) > 0.0001:
            violations.append(
                f"coverage_factor={coverage_factor} but sum(weights_used)={expected_coverage}"
            )

    if violations:
        for v in violations:
            logger.warning(f"Verdict invariant violation: {v}")


def _build_confidence_path(
    confidence: str,
    tilt: str | None,
    fundamentals_available: bool,
    risk_available: bool,
    is_unprofitable: bool,
    has_negative_fcf: bool,
    has_extreme_risk: bool,
    coverage_count: int,
    risk_regime: str | None = None,
    burn_metrics_status: str | None = None,
    sma_200: float | None = None,
    current_price: float | None = None,
    max_drawdown: float | None = None,
    annualized_vol: float | None = None,
) -> dict[str, Any]:
    """
    Build confidence upgrade/downgrade path.

    Shows what would change the current confidence level up or down.
    Uses condition-based triggers instead of score-based.
    """
    upgrade_conditions: list[dict[str, Any]] = []
    downgrade_conditions: list[dict[str, Any]] = []
    current_blockers: list[str] = []
    note: str | None = None

    # Identify current blockers that cap confidence
    if not fundamentals_available:
        current_blockers.append("fundamentals_missing")
    if not risk_available:
        current_blockers.append("risk_data_missing")
    if is_unprofitable:
        current_blockers.append("unprofitable")
    if has_negative_fcf:
        current_blockers.append("negative_fcf")
    if has_extreme_risk:
        current_blockers.append("extreme_risk")
    if burn_metrics_status == "unavailable" and is_unprofitable:
        current_blockers.append("burn_metrics_missing")

    # Build condition-based upgrade conditions
    if confidence == "low":
        # For low confidence, focus on data availability and risk improvement
        if has_extreme_risk:
            added_specific = False
            if risk_regime == "extreme":
                if max_drawdown is not None and max_drawdown <= -0.50:
                    upgrade_conditions.append({
                        "condition": "drawdown_recovers",
                        "threshold": "max_drawdown_1y > -50%",
                        "current": f"{max_drawdown*100:.0f}%",
                    })
                    added_specific = True
                vol_threshold = _vol_threshold_for_improvement(risk_regime, annualized_vol)
                if annualized_vol is not None and vol_threshold is not None and annualized_vol >= vol_threshold:
                    upgrade_conditions.append({
                        "condition": "volatility_decreases",
                        "threshold": f"annualized_vol < {vol_threshold * 100:.0f}%",
                        "current": f"{annualized_vol*100:.0f}%",
                    })
                    added_specific = True
            if not added_specific:
                upgrade_conditions.append({
                    "condition": "risk_regime_improves",
                    "threshold": "risk_regime <= high",
                    "current": f"risk_regime={risk_regime}",
                })
        if is_unprofitable:
            upgrade_conditions.append({
                "condition": "company_turns_profitable",
                "threshold": "net_margin > 0 for 2 quarters",
            })
        if has_negative_fcf:
            upgrade_conditions.append({
                "condition": "fcf_turns_positive",
                "threshold": "FCF > 0 for 2 quarters",
            })
        if burn_metrics_status == "unavailable" and is_unprofitable:
            upgrade_conditions.append({
                "condition": "burn_metrics_available",
                "threshold": "liquidity and cash flow data populated",
            })
        if not has_extreme_risk:
            vol_threshold = _vol_threshold_for_improvement(risk_regime, annualized_vol)
            if annualized_vol is not None and vol_threshold is not None and annualized_vol >= vol_threshold:
                upgrade_conditions.append({
                    "condition": "volatility_decreases",
                    "threshold": f"annualized_vol < {vol_threshold * 100:.0f}%",
                    "current": f"{annualized_vol*100:.0f}%",
                })

    elif confidence == "moderate":
        # For moderate confidence, focus on quality improvements
        if is_unprofitable:
            upgrade_conditions.append({
                "condition": "achieves_profitability",
                "threshold": "net_margin > 5% sustained",
            })
        if has_extreme_risk:
            upgrade_conditions.append({
                "condition": "risk_normalizes",
                "threshold": "risk_regime <= medium",
                "current": f"risk_regime={risk_regime}",
            })
        if max_drawdown is not None and max_drawdown < -0.35:
            upgrade_conditions.append({
                "condition": "drawdown_recovers",
                "threshold": "max_drawdown_1y > -30%",
                "current": f"{max_drawdown*100:.0f}%",
            })

    elif confidence == "high":
        note = "already at highest level"

    # Build condition-based downgrade conditions
    if confidence == "high":
        downgrade_conditions.append({
            "condition": "earnings_miss_or_guidance_cut",
            "threshold": "EPS misses by >10% or guidance lowered",
        })
        if sma_200 is not None and current_price is not None:
            downgrade_conditions.append({
                "condition": "breaks_sma200",
                "threshold": f"price < ${sma_200:.2f} for 3 sessions",
                "current": f"${current_price:.2f}",
            })
        downgrade_conditions.append({
            "condition": "fcf_turns_negative",
            "threshold": "FCF < 0 for 2 quarters",
        })

    elif confidence == "moderate":
        if sma_200 is not None and current_price is not None:
            downgrade_conditions.append({
                "condition": "breaks_sma200",
                "threshold": f"price < ${sma_200:.2f} by >5%",
                "current": f"${current_price:.2f}",
            })
        if max_drawdown is not None:
            downgrade_conditions.append({
                "condition": "drawdown_worsens",
                "threshold": "max_drawdown_1y < -60%",
                "current": f"{max_drawdown*100:.0f}%",
            })
        downgrade_conditions.append({
            "condition": "risk_regime_worsens",
            "threshold": "risk_regime becomes extreme",
            "current": f"risk_regime={risk_regime}",
        })

    elif confidence == "low":
        # Low confidence can't really downgrade further
        note = "confidence already low"

    return {
        "current": confidence,
        "upgrade_if": upgrade_conditions[:3] if upgrade_conditions else None,
        "downgrade_if": downgrade_conditions[:3] if downgrade_conditions else None,
        "current_blockers": current_blockers if current_blockers else None,
        "note": note,
    }


def _build_horizon_fit(
    setup_label: str | None,
    business_quality_label: str | None,
    business_quality_status: str,
    risk_label: str | None,
    signals: dict[str, list[str]],
    revenue_yoy: float | None = None,
    fcf_ttm: float | None = None,
    burn_metrics_status: str | None = None,
    runway_confidence: str | None = None,
) -> dict[str, Any]:
    """
    Build horizon fit assessment for mid-term and long-term investors.

    mid_term (3-12 months): can lean on setup + regime-aware sizing
    long_term (1-5 years): requires business quality + not extreme risk + runway

    Investment policy gates:
    - Long-term = avoid if: extreme risk AND revenue_yoy < -20% AND fcf < 0
    - Long-term = caution if: unprofitable with declining revenue
    - Long-term = caution if: unprofitable with low runway_confidence
    - Long-term = ok only if: business quality moderate+ AND risk <= high
    - Long-term = data_missing if: unprofitable AND burn_metrics unavailable
    """
    mid_term_fit: str = "unknown"
    long_term_fit: str = "unknown"
    reasons: list[str] = []
    data_gaps: list[str] = []

    has_extreme_risk = risk_label == "extreme"
    has_high_risk = risk_label in ("high", "extreme")
    has_severe_revenue_decline = revenue_yoy is not None and revenue_yoy < -0.20
    has_negative_fcf = fcf_ttm is not None and fcf_ttm < 0
    is_unprofitable = business_quality_label == "unprofitable"
    burn_metrics_missing = burn_metrics_status == "unavailable"

    # Track data gaps for unprofitable companies
    if is_unprofitable and burn_metrics_missing:
        data_gaps.append("burn_metrics_unavailable")

    # Mid-term assessment (3-12 months)
    # Can work with setup alone, but risk regime matters for sizing
    if setup_label is None:
        mid_term_fit = "unknown"
        reasons.append("mid_term: insufficient technical data")
    elif has_extreme_risk:
        mid_term_fit = "caution"
        reasons.append("mid_term: extreme risk requires minimal position size")
    elif setup_label in ("strong", "moderate"):
        if has_high_risk:
            mid_term_fit = "ok"
            reasons.append("mid_term: setup favorable but size conservatively due to high risk")
        else:
            mid_term_fit = "ok"
            reasons.append("mid_term: setup supports position with standard sizing")
    elif setup_label == "weak":
        mid_term_fit = "caution"
        reasons.append("mid_term: weak setup suggests waiting for better entry")
    else:  # poor
        mid_term_fit = "avoid"
        reasons.append("mid_term: poor setup with negative technicals")

    # Long-term assessment (1-5 years)
    # Investment policy gates based on fundamentals + risk + trend
    # Accumulate ALL applicable concerns for 1:1 alignment with horizon_drivers
    long_term_concerns: list[str] = []
    long_term_gates: list[str] = []  # Track which gates fired for drivers

    if business_quality_status == "data_missing":
        long_term_fit = "unknown"
        reasons.append("long_term: fundamentals unavailable")
    else:
        # Collect all applicable concerns - each concern maps to a gate for 1:1 alignment
        if is_unprofitable or business_quality_status == "evaluated_unprofitable":
            long_term_concerns.append("unprofitable")
            long_term_gates.append("unprofitable")  # Explicit gate for unprofitability
            if burn_metrics_missing:
                long_term_concerns.append("burn_metrics_missing")
                long_term_gates.append("burn_metrics_missing")
                data_gaps.append("runway_assessment_blocked")
            elif runway_confidence == "low":
                # Burn metrics available but runway < 2 years = dilution risk
                long_term_concerns.append("low_runway_confidence")
                long_term_gates.append("low_runway_confidence")

        if has_extreme_risk:
            long_term_concerns.append("extreme_risk")
            long_term_gates.append("extreme_risk")

        if has_severe_revenue_decline and revenue_yoy is not None:
            long_term_concerns.append(f"revenue_decline_{abs(revenue_yoy)*100:.0f}pct")
            long_term_gates.append("severe_revenue_decline")

        if has_negative_fcf:
            long_term_concerns.append("negative_fcf")
            long_term_gates.append("negative_fcf")

        # Determine fit level based on severity
        if business_quality_label == "poor":
            long_term_fit = "avoid"
            reasons.append("long_term: poor business quality")
        elif has_extreme_risk and has_severe_revenue_decline and has_negative_fcf:
            # Triple threat = structural issues
            long_term_fit = "avoid"
            reasons.append(f"long_term: {' + '.join(long_term_concerns)}")
        elif is_unprofitable and revenue_yoy is not None and revenue_yoy < -0.30:
            # Unprofitable with collapsing revenue
            long_term_fit = "avoid"
            reasons.append(f"long_term: {' + '.join(long_term_concerns)}")
        elif long_term_concerns:
            # Any concerns = caution
            long_term_fit = "caution"
            reasons.append(f"long_term: {' + '.join(long_term_concerns)}")
        elif business_quality_label in ("strong", "moderate"):
            if has_high_risk:
                long_term_fit = "ok"
                reasons.append("long_term: business quality supports holding despite elevated risk")
            else:
                long_term_fit = "ok"
                reasons.append("long_term: business quality and risk support long-term holding")
        elif business_quality_label == "mixed":
            long_term_fit = "caution"
            reasons.append("long_term: mixed fundamentals suggest smaller position")
        else:
            long_term_fit = "unknown"
            reasons.append("long_term: insufficient data for assessment")

    return {
        "mid_term": mid_term_fit,
        "long_term": long_term_fit,
        "reasons": reasons,
        "data_gaps": data_gaps or [],
        "long_term_gates": long_term_gates or [],  # For horizon_drivers alignment
    }


def _build_action_zones(
    current_price: float | None,
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
    fund_data: dict[str, Any],
    risk_regime: dict[str, Any] | None = None,
    signals: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Build action zones with ATR-based price levels.

    Zones are volatility-adjusted using ATR, not arbitrary percentages.
    Risk regime affects zone interpretation and warnings.
    """
    if current_price is None:
        return {
            "current_zone": None,
            "levels": {},
            "distance_to_levels": {},
            "price_vs_levels": {},
            "distance_labels": {},
            "level_vs_current_labels": {},
            "basis": {},
            "stop_calculation": None,
            "zone_warnings": ["missing_price"],
            "method": "atr_based_v1",
        }

    # Extract risk regime classification
    regime_classification = (
        risk_regime.get("classification") if risk_regime else None
    )
    is_extreme_regime = regime_classification == "extreme"
    is_high_regime = regime_classification in ("high", "extreme")

    # Extract needed values
    ma = tech_data.get("moving_averages", {})
    price_pos = tech_data.get("price_position", {})
    atr_data = risk_data.get("atr", {})

    sma_50 = ma.get("sma_50")
    sma_200 = ma.get("sma_200")
    week_52_low = price_pos.get("week_52_low")
    week_52_high = price_pos.get("week_52_high")
    atr_val = atr_data.get("value")
    atr_pct = atr_data.get("as_pct_of_price")

    zone_warnings: list[str] = []

    # Use ATR for volatility-adjusted zones (default to 2% if missing)
    if atr_pct is not None:
        volatility_band = atr_pct
    else:
        volatility_band = 0.02
        zone_warnings.append("atr_unavailable_using_default")

    # Calculate levels
    levels: dict[str, float | None] = {}
    basis: dict[str, str | None] = {}

    # Strong buy: near 52-week low + 1 ATR buffer
    if week_52_low is not None:
        levels["strong_buy_below"] = round(week_52_low * (1 + volatility_band), 2)
        basis["strong_buy_below"] = "52w_low_plus_1atr"
    else:
        levels["strong_buy_below"] = None
        basis["strong_buy_below"] = None

    # Accumulate: below SMA200 or approaching SMA200
    if sma_200 is not None:
        levels["accumulate_near"] = round(sma_200, 2)
        basis["accumulate_near"] = "sma_200"
    else:
        levels["accumulate_near"] = None
        basis["accumulate_near"] = None

    # Take profit: approaching 52-week high - 1 ATR buffer
    if week_52_high is not None:
        levels["reduce_above"] = round(week_52_high * (1 - volatility_band), 2)
        basis["reduce_above"] = "52w_high_minus_1atr"
    else:
        levels["reduce_above"] = None
        basis["reduce_above"] = None

    # Stop loss: 2 ATR below current price (2.5 ATR for extreme regime)
    stop_multiple_used: float = 2.5 if is_extreme_regime else 2.0
    stop_min_multiple_required: float = 2.0 if is_high_regime else 1.5

    stop_calculation: dict[str, Any] | None = None
    if atr_val is not None:
        stop_price = round(current_price - (atr_val * stop_multiple_used), 2)
        levels["stop_loss"] = stop_price
        basis["stop_loss"] = f"current_minus_{stop_multiple_used}atr"

        stop_distance_pct = round(abs(stop_price - current_price) / current_price, 4)

        stop_calculation = {
            "stop_price": stop_price,
            "stop_distance_pct": stop_distance_pct,
            "atr_pct": round(atr_pct, 4) if atr_pct else None,
            "stop_multiple_used": stop_multiple_used,
            "min_multiple_required": stop_min_multiple_required,
        }
    else:
        levels["stop_loss"] = None
        basis["stop_loss"] = None

    # Calculate distances as percentages from current price
    distance_to_levels: dict[str, float | None] = {}
    for level_name, level_price in levels.items():
        if level_price is not None and current_price > 0:
            distance_to_levels[level_name] = round(
                (level_price - current_price) / current_price, 4
            )
        else:
            distance_to_levels[level_name] = None

    # Calculate price vs level (negative means price below level)
    price_vs_levels: dict[str, float | None] = {}
    for level_name, level_price in levels.items():
        if level_price is not None and level_price != 0:
            price_vs_levels[level_name] = round(
                (current_price / level_price) - 1, 4
            )
        else:
            price_vs_levels[level_name] = None

    # Preformatted distance labels for clearer display (level vs current)
    distance_labels = {
        level_name: _format_level_distance_label(pct)
        for level_name, pct in distance_to_levels.items()
    }

    # Separate field for renderer semantics (stop loss vs other levels)
    level_vs_current_labels = distance_labels.copy()

    # Determine current zone
    current_zone: str | None = None
    strong_buy_level = levels.get("strong_buy_below")
    reduce_level = levels.get("reduce_above")

    if strong_buy_level is not None and current_price <= strong_buy_level:
        current_zone = "strong_buy"
    elif sma_200 and current_price < sma_200:
        current_zone = "accumulate"
    elif reduce_level is not None and current_price >= reduce_level:
        current_zone = "reduce"
    elif sma_50 and current_price > sma_50:
        current_zone = "hold_bullish"
    elif sma_50 and current_price <= sma_50:
        current_zone = "hold_neutral"
    else:
        current_zone = "undetermined"

    # Apply regime-aware zone capping
    # In extreme risk regime, cap "strong_buy" to "accumulate" - be more cautious
    if is_extreme_regime and current_zone == "strong_buy":
        current_zone = "accumulate"
        zone_warnings.append("zone_capped_due_to_extreme_risk")

    # Valuation-aware accumulate gates for long-term investors
    val = fund_data.get("valuation", {})
    yield_m = fund_data.get("yield_metrics", {})
    profit = fund_data.get("profitability", {})

    pe = val.get("pe_trailing")
    peg = val.get("peg_ratio")
    ps = val.get("ps_trailing")
    ev_to_sales = val.get("ev_to_sales")  # Prefer over P/S when debt/cash material
    fcf_yield = yield_m.get("fcf_yield")
    earnings_yield = yield_m.get("earnings_yield")
    net_margin = profit.get("net_margin")

    # Determine if company is unprofitable (use P/S instead of P/E)
    # Use multiple signals for robust detection when data is sparse
    bearish_signals = signals.get("bearish", []) if signals else []
    is_signaled_unprofitable = "unprofitable" in bearish_signals
    has_negative_fcf = "negative_free_cash_flow" in bearish_signals

    # Detect unprofitability from EXPLICIT negative signals only
    # Never classify as unprofitable purely from missing data (that's "fundamentals_missing")
    # 1. Negative net margin (explicit)
    # 2. Signal system detected "unprofitable" (based on EPS or other metrics)
    # 3. Negative trailing EPS (explicit)
    # 4. No P/E + negative FCF (strong inference: negative earnings + burning cash)
    trailing_eps = profit.get("trailing_eps")
    is_unprofitable = False
    if net_margin is not None and net_margin < 0:
        is_unprofitable = True
    elif is_signaled_unprofitable:
        is_unprofitable = True
    elif trailing_eps is not None and trailing_eps <= 0:
        # Explicit negative or zero EPS
        is_unprofitable = True
    elif pe is None and has_negative_fcf:
        # No P/E (likely negative earnings) + negative FCF = strong inference
        is_unprofitable = True
    # NOTE: If pe=None and net_margin=None and trailing_eps=None, we DON'T assume unprofitable
    # That's a data gap, not evidence of losses. Let fundamentals_missing handle it.

    # Valuation gate: determines if valuation supports accumulation
    # attractive = valuation supports adding, neutral = ok to hold, headwind = valuation stretched
    valuation_gate: str = "neutral"
    valuation_gate_reasons: list[str] = []

    # Track which valuation metric drives the gate for auditability
    valuation_basis: str | None = None

    if is_unprofitable:
        # For unprofitable companies, prefer EV/S over P/S
        # EV/S is better when debt/cash position is material (adjusts for net debt)
        sales_multiple: float | None = None
        if ev_to_sales is not None:
            sales_multiple = ev_to_sales
            valuation_basis = "ev_to_sales"
        elif ps is not None:
            sales_multiple = ps
            valuation_basis = "ps_trailing"

        if sales_multiple is not None:
            metric_label = "EV/S" if valuation_basis == "ev_to_sales" else "P/S"
            if sales_multiple < 3:
                valuation_gate = "attractive"
                valuation_gate_reasons.append(
                    f"{metric_label} {sales_multiple:.1f} reasonable for growth"
                )
            elif sales_multiple > 10:
                valuation_gate = "headwind"
                valuation_gate_reasons.append(
                    f"{metric_label} {sales_multiple:.1f} very elevated for unprofitable company"
                )
                zone_warnings.append("valuation_extended_high_sales_multiple_unprofitable")
            else:
                valuation_gate_reasons.append(f"{metric_label} {sales_multiple:.1f} moderate")
        else:
            # Neither EV/S nor P/S available - cannot evaluate valuation
            valuation_gate = "unknown"
            valuation_basis = "unknown"
            valuation_gate_reasons.append("sales_multiple_unavailable_cannot_evaluate")
            zone_warnings.append("valuation_gate_unknown_sales_multiple_missing")
    else:
        # For profitable companies, use FCF yield / earnings yield / PEG
        if fcf_yield is not None and fcf_yield > 0.05:
            valuation_gate = "attractive"
            valuation_basis = "fcf_yield"
            valuation_gate_reasons.append(f"FCF yield {fcf_yield*100:.1f}% attractive")
        elif earnings_yield is not None and earnings_yield > 0.04:
            valuation_gate = "attractive"
            valuation_basis = "earnings_yield"
            valuation_gate_reasons.append(f"Earnings yield {earnings_yield*100:.1f}% reasonable")
        elif peg is not None and peg < 1.0:
            valuation_gate = "attractive"
            valuation_basis = "peg_ratio"
            valuation_gate_reasons.append(f"PEG {peg:.2f} suggests undervaluation")

        # Check for headwinds
        if pe is not None and pe > 50:
            valuation_gate = "headwind"
            valuation_basis = "pe_trailing"
            valuation_gate_reasons.append(f"P/E {pe:.1f} very elevated")
            zone_warnings.append("valuation_extended_high_pe")
        if peg is not None and peg > 3.0:
            if valuation_gate != "headwind":
                valuation_gate = "headwind"
            valuation_basis = "peg_ratio"
            valuation_gate_reasons.append(f"PEG {peg:.2f} suggests overvaluation")
            zone_warnings.append("valuation_extended_high_peg")

        # For neutral gate with profitable company, set basis to primary metric used
        if valuation_gate == "neutral" and valuation_basis is None:
            if pe is not None:
                valuation_basis = "pe_trailing"
            elif peg is not None:
                valuation_basis = "peg_ratio"
            elif fcf_yield is not None:
                valuation_basis = "fcf_yield"
            elif ps is not None:
                valuation_basis = "ps_trailing"

        # If we couldn't assess valuation at all (no P/E, no FCF yield, no earnings yield, no PEG)
        # AND P/S is also missing, gate is unknown
        if (
            valuation_gate == "neutral"
            and pe is None
            and fcf_yield is None
            and earnings_yield is None
            and peg is None
            and ps is None
        ):
            valuation_gate = "unknown"
            valuation_basis = "unknown"
            valuation_gate_reasons.append("insufficient_valuation_data")
            zone_warnings.append("valuation_gate_unknown_no_metrics")

    # Apply valuation gate to zone recommendation
    # If valuation is headwind, downgrade strong_buy to accumulate, accumulate to hold
    if valuation_gate == "headwind":
        if current_zone == "strong_buy":
            current_zone = "accumulate"
            zone_warnings.append("zone_downgraded_valuation_headwind")
        elif current_zone == "accumulate":
            current_zone = "hold_neutral"
            zone_warnings.append("zone_downgraded_valuation_headwind")

    valuation_assessment = {
        "gate": valuation_gate,
        "basis": valuation_basis,  # Which metric drives the gate (ev_to_sales, ps_trailing, pe_trailing, etc.)
        "reasons": valuation_gate_reasons if valuation_gate_reasons else None,
        "is_unprofitable": is_unprofitable,
    }

    # Check for stop_too_tight warning - volatility-aware using ATR multiple
    stop_distance = distance_to_levels.get("stop_loss")
    if stop_distance is not None and atr_pct is not None:
        stop_distance_pct_val = abs(stop_distance)
        if stop_distance_pct_val < (stop_min_multiple_required * atr_pct):
            zone_warnings.append("stop_too_tight")
    elif stop_distance is not None and stop_distance > -0.05:
        # Fallback: fixed 5% if no ATR available
        zone_warnings.append("stop_too_tight")

    # Add regime-specific warnings
    if is_extreme_regime:
        zone_warnings.append("extreme_risk_regime:prefer_small_position")
    elif is_high_regime:
        zone_warnings.append("high_risk_regime:size_conservatively")

    # Position sizing range based on risk regime
    # These are suggested ranges as % of portfolio
    # Default portfolio value for dollar calculations
    default_portfolio_value = 50000.0

    position_sizing_range: dict[str, Any]
    if is_extreme_regime:
        pct_min, pct_max = 0.5, 3.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "extreme_risk_requires_minimal_exposure",
        }
    elif is_high_regime:
        pct_min, pct_max = 2.0, 6.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "high_risk_warrants_conservative_sizing",
        }
    elif regime_classification == "medium":
        pct_min, pct_max = 3.0, 8.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "moderate_risk_standard_sizing",
        }
    else:
        # low risk or unknown
        pct_min, pct_max = 3.0, 10.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "low_risk_allows_larger_positions",
        }

    # Add dollar amounts for default portfolio size
    dollar_min = round(default_portfolio_value * pct_min / 100, 0)
    dollar_max = round(default_portfolio_value * pct_max / 100, 0)
    position_sizing_range["dollars_for_50k"] = {
        "min": dollar_min,
        "max": dollar_max,
        "portfolio_assumption": default_portfolio_value,
    }

    # Add shares range at current price
    if current_price and current_price > 0:
        shares_min = int(dollar_min / current_price)
        shares_max = int(dollar_max / current_price)
        position_sizing_range["shares_range"] = {
            "min": shares_min,
            "max": shares_max,
            "at_price": current_price,
        }

    # Add stop-implied max size (risk 1% of portfolio per trade)
    stop_distance = stop_calculation.get("distance_pct") if stop_calculation else None
    if stop_distance and stop_distance > 0:
        # 1% risk rule: max_position = (portfolio * 0.01) / stop_distance
        risk_pct = 1.0  # Risk 1% of portfolio on a single trade
        stop_implied_pct = round((risk_pct / stop_distance) * 100, 1)
        # Cap at the max_pct from risk regime
        if stop_implied_pct > pct_max:
            stop_implied_pct = pct_max
        stop_implied_dollars = round(default_portfolio_value * stop_implied_pct / 100, 0)
        position_sizing_range["stop_implied_max"] = {
            "pct": stop_implied_pct,
            "dollars_for_50k": stop_implied_dollars,
            "risk_per_trade_pct": risk_pct,
            "stop_distance_pct": round(stop_distance * 100, 1),
        }

    return {
        "current_zone": current_zone,
        "levels": levels,
        "distance_to_levels": distance_to_levels,
        "price_vs_levels": price_vs_levels,
        "distance_labels": distance_labels,
        "level_vs_current_labels": level_vs_current_labels,
        "basis": basis,
        "stop_calculation": stop_calculation,
        "position_sizing_range": position_sizing_range,
        "valuation_assessment": valuation_assessment,
        "zone_warnings": zone_warnings or [],
        "method": "atr_valuation_v2",
    }


def _apply_dip_gates_to_action_zones(
    action_zones: dict[str, Any],
    dip_assessment: dict[str, Any],
    risk_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply dip-aware guards to action zones to avoid conflicting guidance.
    """
    current_zone = action_zones.get("current_zone")
    zone_warnings = list(action_zones.get("zone_warnings") or [])
    dip_type = (dip_assessment.get("dip_classification") or {}).get("type")
    risk_label = risk_regime.get("classification") if risk_regime else None

    def _cap_zone(reason: str) -> None:
        nonlocal current_zone
        if current_zone in ("strong_buy", "accumulate"):
            current_zone = "hold_neutral"
        if reason not in zone_warnings:
            zone_warnings.append(reason)

    if dip_type == "falling_knife":
        _cap_zone("zone_capped_falling_knife")
    elif dip_type == "extended_decline":
        if current_zone == "strong_buy":
            current_zone = "accumulate"
            if "zone_capped_extended_decline" not in zone_warnings:
                zone_warnings.append("zone_capped_extended_decline")

    if risk_label == "extreme":
        _cap_zone("zone_capped_extreme_risk")

    action_zones["current_zone"] = current_zone
    action_zones["zone_warnings"] = zone_warnings
    return action_zones


def _build_dip_depth(
    from_52w_high: float | None,
    from_52w_low: float | None,
    from_3m_high: float | None,
    from_6m_high: float | None,
    days_since_52w_high: int | None,
    days_since_52w_low: int | None,
    risk_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Build dip depth metrics with severity classification.

    Severity bands (based on distance from 52w high):
    - none: >= -2% (within 2% of high)
    - shallow: > -10%
    - moderate: > -25%
    - deep: > -40%
    - extreme: <= -40%
    """
    # Primary basis: from_52w_high
    # Fallback: max_drawdown_1y from risk_data
    dd_data = risk_data.get("drawdown", {})
    max_drawdown_1y = dd_data.get("max_1y")

    # Determine which value to use for severity
    severity_value: float | None = None
    severity_basis: str | None = None

    if from_52w_high is not None:
        severity_value = from_52w_high
        severity_basis = "from_52w_high"
    elif max_drawdown_1y is not None:
        severity_value = max_drawdown_1y
        severity_basis = "max_drawdown_1y"

    # Classify severity
    severity: str
    if severity_value is None:
        severity = "unknown"
    elif severity_value >= -0.02:
        severity = "none"
    elif severity_value > -0.10:
        severity = "shallow"
    elif severity_value > -0.25:
        severity = "moderate"
    elif severity_value > -0.40:
        severity = "deep"
    else:
        severity = "extreme"

    low_set_today = days_since_52w_low == 0 if days_since_52w_low is not None else None
    high_set_today = days_since_52w_high == 0 if days_since_52w_high is not None else None

    return {
        "from_52w_high": from_52w_high,
        "from_52w_low": from_52w_low,
        "from_3m_high": from_3m_high,
        "from_6m_high": from_6m_high,
        "days_since_52w_high": days_since_52w_high,
        "days_since_52w_low": days_since_52w_low,
        "low_set_today": low_set_today,
        "high_set_today": high_set_today,
        "severity": severity,
        "severity_basis": severity_basis,
    }


def _build_dip_assessment(
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
    fund_data: dict[str, Any],
    market_context: dict[str, Any],
    signals: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Assess whether current price action represents a buying opportunity for dip buyers.

    Analyzes:
    - Dip classification (pullback vs falling knife)
    - Oversold conditions (RSI, distance from MAs)
    - Support levels and bounce potential
    - Volume patterns (capitulation/accumulation)
    - Market context (buying dips in uptrending market vs downtrending)
    """
    ma = tech_data.get("moving_averages", {})
    rsi_data = tech_data.get("rsi", {})
    price_pos = tech_data.get("price_position", {})
    returns = tech_data.get("returns", {})
    volume = tech_data.get("volume", {})
    macd_data = tech_data.get("macd", {})
    price_action = tech_data.get("price_action", {})
    current_price = tech_data.get("current_price")
    atr_data = risk_data.get("atr", {})

    # Extract values
    sma_20 = ma.get("sma_20")
    sma_50 = ma.get("sma_50")
    sma_200 = ma.get("sma_200")
    sma_200_slope = ma.get("sma_200_slope_pct_per_day")
    rsi = rsi_data.get("value")
    rsi_divergence = rsi_data.get("bullish_divergence")
    week_52_high = price_pos.get("week_52_high")
    week_52_low = price_pos.get("week_52_low")
    from_52w_high = price_pos.get("from_52w_high")
    from_52w_low = price_pos.get("from_52w_low")
    from_3m_high = price_pos.get("from_3m_high")
    from_6m_high = price_pos.get("from_6m_high")
    days_since_52w_high = price_pos.get("days_since_52w_high")
    days_since_52w_low = price_pos.get("days_since_52w_low")
    low_1m = price_pos.get("low_1m")
    position_in_range = price_pos.get("position_in_range")
    return_1w = returns.get("return_1w")
    return_1w_zscore = returns.get("return_1w_zscore")
    return_1m = returns.get("return_1m")
    return_3m = returns.get("return_3m")
    volume_ratio = volume.get("ratio")
    atr_val = atr_data.get("value")
    atr_pct = atr_data.get("as_pct_of_price")

    bullish_signals = signals.get("bullish", [])
    bearish_signals = signals.get("bearish", [])

    # --- DIP CLASSIFICATION ---
    # Determine if this is a healthy pullback vs falling knife
    dip_type: str = "undetermined"
    dip_signals: list[str] = []

    # Falling knife indicators
    falling_knife_score = 0
    if "death_cross" in bearish_signals:
        falling_knife_score += 2
        dip_signals.append("death_cross_active")
    if "price_below_sma200" in bearish_signals:
        falling_knife_score += 1
        dip_signals.append("below_sma200")
    if return_3m is not None and return_3m < -0.30:
        falling_knife_score += 2
        dip_signals.append("severe_3m_decline")
    elif return_3m is not None and return_3m < -0.20:
        falling_knife_score += 1
        dip_signals.append("significant_3m_decline")
    if position_in_range is not None and position_in_range < 0.10:
        falling_knife_score += 1
        dip_signals.append("near_52w_low")
    if sma_200_slope is not None and sma_200_slope < 0:
        falling_knife_score += 1
        dip_signals.append("sma200_downtrend")
    if days_since_52w_high is not None and days_since_52w_high > 180:
        falling_knife_score += 1
        dip_signals.append("stale_52w_high")

    # Healthy pullback indicators
    pullback_score = 0
    if "golden_cross" in bullish_signals or (sma_50 and sma_200 and sma_50 > sma_200):
        pullback_score += 2
        dip_signals.append("uptrend_intact")
    if sma_200_slope is not None and sma_200_slope > 0:
        pullback_score += 1
        dip_signals.append("sma200_uptrend")
    if sma_200 and current_price and current_price > sma_200:
        pullback_score += 1
        dip_signals.append("above_sma200")
    if return_3m is not None and return_3m > 0:
        pullback_score += 1
        dip_signals.append("positive_3m_trend")
    if from_3m_high is not None and from_3m_high > -0.15:
        pullback_score += 1
        dip_signals.append("near_3m_high")

    # Classify dip
    if falling_knife_score >= 4:
        dip_type = "falling_knife"
    elif falling_knife_score >= 2 and pullback_score < 2:
        dip_type = "extended_decline"
    elif pullback_score >= 2 and falling_knife_score <= 1:
        dip_type = "healthy_pullback"
    elif pullback_score >= 1:
        dip_type = "mixed_signals"
    else:
        dip_type = "undetermined"

    # --- OVERSOLD METRICS ---
    oversold_indicators: list[str] = []
    oversold_score = 0

    # RSI oversold
    rsi_status: str = "neutral"
    if rsi is not None:
        if rsi < 30:
            rsi_status = "oversold"
            oversold_score += 2
            oversold_indicators.append(f"rsi_oversold_{rsi:.0f}")
        elif rsi < 40:
            rsi_status = "approaching_oversold"
            oversold_score += 1
            oversold_indicators.append(f"rsi_low_{rsi:.0f}")
        elif rsi > 70:
            rsi_status = "overbought"
            oversold_indicators.append(f"rsi_overbought_{rsi:.0f}")

    # Distance from moving averages (mean reversion potential)
    price_vs_sma20 = ma.get("price_vs_sma20")
    price_vs_sma50 = ma.get("price_vs_sma50")
    price_vs_sma200 = ma.get("price_vs_sma200")

    distance_to_sma50_atr: float | None = None
    if atr_val and sma_50 and current_price:
        distance_to_sma50_atr = (current_price - sma_50) / atr_val

    if price_vs_sma20 is not None and price_vs_sma20 < -0.10:
        oversold_score += 1
        oversold_indicators.append(f"extended_below_sma20_{price_vs_sma20:.1%}")
    if price_vs_sma50 is not None and price_vs_sma50 < -0.15:
        oversold_score += 1
        oversold_indicators.append(f"extended_below_sma50_{price_vs_sma50:.1%}")
    if price_vs_sma200 is not None and price_vs_sma200 < -0.20:
        oversold_score += 1
        oversold_indicators.append(f"extended_below_sma200_{price_vs_sma200:.1%}")
    if distance_to_sma50_atr is not None and distance_to_sma50_atr < -2.0:
        oversold_score += 1
        oversold_indicators.append(f"below_sma50_{abs(distance_to_sma50_atr):.1f}atr")

    # Position in 52-week range
    if position_in_range is not None and position_in_range < 0.20:
        oversold_score += 1
        oversold_indicators.append(f"bottom_20pct_of_range")

    if return_1w_zscore is not None and return_1w_zscore <= -1.5:
        oversold_score += 1
        oversold_indicators.append(f"1w_return_zscore_{return_1w_zscore:.1f}")

    oversold_level: str
    if oversold_score >= 4:
        oversold_level = "extremely_oversold"
    elif oversold_score >= 2:
        oversold_level = "oversold"
    elif oversold_score >= 1:
        oversold_level = "mildly_oversold"
    else:
        oversold_level = "not_oversold"

    # Oversold composite (de-dup correlated signals)
    oversold_composite = _build_oversold_composite(
        rsi,
        return_1w_zscore,
        distance_to_sma50_atr,
        position_in_range,
    )
    oversold_composite_level = oversold_composite["level"]

    # --- SUPPORT LEVELS ---
    support_levels: list[dict[str, Any]] = []
    price_basis = "close"

    def _support_status(distance_pct: float | None) -> str | None:
        if distance_pct is None:
            return None
        if distance_pct > 0:
            return "breached" if distance_pct <= 0.01 else "broken"
        if abs(distance_pct) <= 0.01:
            return "tested"
        return "above"

    # 1-month low as immediate support
    if low_1m and current_price:
        distance = (low_1m - current_price) / current_price
        support_levels.append({
            "level": round(low_1m, 2),
            "type": "1m_low",
            "distance_pct": round(distance, 4),
            "strength": "weak",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    # SMA 50 as support (if above it)
    if sma_50 and current_price and current_price > sma_50:
        distance = (sma_50 - current_price) / current_price
        support_levels.append({
            "level": round(sma_50, 2),
            "type": "sma_50",
            "distance_pct": round(distance, 4),
            "strength": "medium",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    # SMA 200 as major support
    if sma_200 and current_price:
        distance = (sma_200 - current_price) / current_price
        support_levels.append({
            "level": round(sma_200, 2),
            "type": "sma_200",
            "distance_pct": round(distance, 4),
            "strength": "strong",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    # 52-week low as ultimate support
    if week_52_low and current_price:
        distance = (week_52_low - current_price) / current_price
        support_levels.append({
            "level": round(week_52_low, 2),
            "type": "52w_low",
            "distance_pct": round(distance, 4),
            "strength": "critical",
            "status": _support_status(distance),
            "price_basis": price_basis,
        })

    # Sort by distance (closest first)
    support_levels.sort(key=lambda x: abs(x["distance_pct"]))

    # --- VOLUME ANALYSIS ---
    volume_signal: str = "neutral"
    volume_interpretation: str | None = None

    if volume_ratio is not None:
        if volume_ratio > 2.0:
            # High volume - could be capitulation or breakout
            if return_1w is not None and return_1w < -0.05:
                volume_signal = "potential_capitulation"
                volume_interpretation = "High volume on decline may indicate capitulation selling"
            elif return_1w is not None and return_1w > 0.03:
                volume_signal = "accumulation"
                volume_interpretation = "High volume on advance suggests institutional buying"
            else:
                volume_signal = "elevated"
                volume_interpretation = "Elevated volume - watch for directional confirmation"
        elif volume_ratio > 1.5:
            volume_signal = "above_average"
            volume_interpretation = "Above average interest"
        elif volume_ratio >= 0.9:
            volume_signal = "normal"
            volume_interpretation = "Normal volume"
        elif volume_ratio >= 0.5:
            volume_signal = "below_average"
            volume_interpretation = "Below average interest"
        else:
            volume_signal = "low_conviction"
            volume_interpretation = "Low volume - moves may lack conviction"

    # --- BOUNCE POTENTIAL ---
    # Calculate likelihood of a bounce based on combined factors
    bounce_score = 0
    bounce_factors: list[str] = []

    # Oversold conditions favor bounce
    if oversold_composite_level == "extreme":
        bounce_score += 2
        bounce_factors.append("oversold_composite_extreme")
    elif oversold_composite_level == "oversold":
        bounce_score += 1
        bounce_factors.append("oversold_composite_oversold")

    # Healthy pullback in uptrend
    if dip_type == "healthy_pullback":
        bounce_score += 2
        bounce_factors.append("pullback_in_uptrend")
    elif dip_type == "falling_knife":
        bounce_score -= 2
        bounce_factors.append("falling_knife_risk")

    # Near support
    if support_levels and abs(support_levels[0]["distance_pct"]) < 0.03:
        if support_levels[0]["strength"] in ("strong", "critical"):
            bounce_score += 1
            bounce_factors.append(f"near_{support_levels[0]['type']}")

    # Capitulation volume
    if volume_signal == "potential_capitulation":
        bounce_score += 1
        bounce_factors.append("capitulation_volume")

    # Market context - buying dips in bull market more favorable
    spy_bullish = market_context.get("spy_above_200d", False)
    if spy_bullish:
        bounce_score += 1
        bounce_factors.append("bullish_market_context")
    else:
        bounce_score -= 1
        bounce_factors.append("bearish_market_context")

    bounce_potential: str
    if bounce_score >= 4:
        bounce_potential = "high"
    elif bounce_score >= 2:
        bounce_potential = "moderate"
    elif bounce_score >= 0:
        bounce_potential = "low"
    else:
        bounce_potential = "very_low"

    # --- ENTRY TIMING ---
    entry_signals: list[dict[str, str]] = []
    wait_for: list[str] = []

    # Immediate entry signals
    if rsi_status == "oversold" and dip_type != "falling_knife":
        entry_signals.append({
            "signal": "rsi_oversold",
            "action": "consider_small_position",
            "rationale": "RSI indicates oversold but confirm with price action",
        })

    if rsi_divergence is True:
        entry_signals.append({
            "signal": "rsi_bullish_divergence",
            "action": "starter_position",
            "rationale": "Price made lower low while RSI improved",
        })

    if volume_signal == "potential_capitulation":
        entry_signals.append({
            "signal": "capitulation_volume",
            "action": "watch_for_reversal",
            "rationale": "High volume selling may exhaust sellers",
        })

    if dip_type == "healthy_pullback" and oversold_level in ("oversold", "mildly_oversold"):
        entry_signals.append({
            "signal": "pullback_in_uptrend",
            "action": "accumulate",
            "rationale": "Healthy pullback to buy in established uptrend",
        })

    if price_action.get("higher_closes_2d") is True:
        entry_signals.append({
            "signal": "two_higher_closes",
            "action": "starter_position",
            "rationale": "Short-term momentum stabilizing",
        })

    if price_action.get("break_5d_high") is True:
        entry_signals.append({
            "signal": "break_5d_high",
            "action": "add_position",
            "rationale": "Price reclaimed recent high",
        })

    if price_vs_sma20 is not None and price_vs_sma20 > 0:
        entry_signals.append({
            "signal": "close_above_sma20",
            "action": "add_position",
            "rationale": "Reclaimed short-term trend",
        })

    if macd_data.get("histogram_rising_3d") is True:
        entry_signals.append({
            "signal": "macd_histogram_rising",
            "action": "add_position",
            "rationale": "Momentum improving for 3 sessions",
        })

    # Wait signals
    if dip_type == "falling_knife":
        wait_for.append("price_stabilization_above_support")
        if rsi_divergence is not True:
            wait_for.append("rsi_bullish_divergence")
        wait_for.append("volume_dry_up")

    if sma_200 and current_price and current_price < sma_200:
        wait_for.append(f"reclaim_sma200_at_{sma_200:.2f}")

    if rsi is not None and rsi > 50 and dip_type != "healthy_pullback":
        wait_for.append("deeper_pullback_or_rsi_oversold")

    if price_action.get("higher_closes_2d") is not True:
        wait_for.append("two_higher_closes")
    if price_action.get("break_5d_high") is not True:
        wait_for.append("break_5d_high")
    if price_vs_sma20 is not None and price_vs_sma20 <= 0:
        wait_for.append("close_above_sma20")
    if macd_data.get("histogram_rising_3d") is not True:
        wait_for.append("macd_histogram_rising_3d")

    # --- DIP CONFIDENCE ---
    confidence_score = 0
    missing_metrics: list[str] = []

    def _score_metric(value: Any, name: str) -> None:
        nonlocal confidence_score
        if value is None:
            missing_metrics.append(name)
        else:
            confidence_score += 1

    _score_metric(rsi, "rsi")
    _score_metric(price_vs_sma200, "price_vs_sma200")
    _score_metric(return_3m, "return_3m")
    _score_metric(volume_ratio, "volume_ratio")
    _score_metric(atr_pct, "atr_pct")

    if dip_type in ("healthy_pullback", "falling_knife"):
        confidence_score += 1
    elif dip_type == "mixed_signals":
        confidence_score -= 1

    if oversold_level in ("oversold", "extremely_oversold"):
        confidence_score += 1
    elif oversold_level == "not_oversold":
        confidence_score -= 1

    if confidence_score >= 5:
        confidence_level = "high"
    elif confidence_score >= 3:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # --- OVERALL ASSESSMENT ---
    # Combine all factors for buy-the-dip recommendation
    dip_quality: str
    if dip_type == "healthy_pullback" and oversold_level in ("oversold", "extremely_oversold"):
        dip_quality = "excellent"
    elif dip_type == "healthy_pullback" and oversold_level == "mildly_oversold":
        dip_quality = "good"
    elif dip_type == "mixed_signals" and oversold_level in ("oversold", "extremely_oversold"):
        dip_quality = "fair"
    elif dip_type == "extended_decline" and oversold_level == "extremely_oversold":
        dip_quality = "speculative"
    elif dip_type == "falling_knife":
        dip_quality = "avoid"
    else:
        dip_quality = "wait"

    # Buy recommendation
    buy_recommendation: str
    if dip_quality == "excellent":
        buy_recommendation = "strong_buy_the_dip"
    elif dip_quality == "good":
        buy_recommendation = "buy_the_dip"
    elif dip_quality == "fair":
        buy_recommendation = "cautious_accumulation"
    elif dip_quality == "speculative":
        buy_recommendation = "small_speculative_position"
    elif dip_quality == "avoid":
        buy_recommendation = "do_not_catch_falling_knife"
    else:
        buy_recommendation = "wait_for_better_setup"

    return {
        "dip_classification": {
            "type": dip_type,
            "signals": dip_signals,
            "explanation": {
                "falling_knife": "Severe decline with broken trends - high risk of further downside",
                "extended_decline": "Prolonged weakness but not in freefall",
                "healthy_pullback": "Normal retracement in an uptrend - favorable for dip buying",
                "mixed_signals": "Conflicting trend signals - proceed with caution",
                "undetermined": "Insufficient data to classify",
            }.get(dip_type, ""),
        },
        "dip_depth": _build_dip_depth(
            from_52w_high,
            from_52w_low,
            from_3m_high,
            from_6m_high,
            days_since_52w_high,
            days_since_52w_low,
            risk_data,
        ),
        "oversold_metrics": {
            "level": oversold_level,
            "score": oversold_score,
            "rsi_status": rsi_status,
            "rsi_value": rsi,
            "indicators": oversold_indicators,
            "distance_from_sma20": price_vs_sma20,
            "distance_from_sma50": price_vs_sma50,
            "distance_from_sma200": price_vs_sma200,
            "distance_from_sma50_atr": _round_or_none(distance_to_sma50_atr, 2),
            "return_1w_zscore": return_1w_zscore,
            "sma200_slope_pct_per_day": sma_200_slope,
            "position_in_52w_range": position_in_range,
            "oversold_composite": oversold_composite,
        },
        "support_levels": support_levels[:4],  # Top 4 closest supports
        "volume_analysis": {
            "signal": volume_signal,
            "ratio": volume_ratio,
            "interpretation": volume_interpretation,
        },
        "bounce_potential": {
            "rating": bounce_potential,
            "score": bounce_score,
            "factors": bounce_factors,
        },
        "entry_timing": {
            "signals": entry_signals,
            "wait_for": wait_for,
        },
        "dip_confidence": {
            "level": confidence_level,
            "score": confidence_score,
            "missing": missing_metrics or [],
        },
        "assessment": {
            "dip_quality": dip_quality,
            "recommendation": buy_recommendation,
            "rationale": {
                "strong_buy_the_dip": "Oversold pullback in uptrend - high probability bounce",
                "buy_the_dip": "Good entry point in established trend",
                "cautious_accumulation": "Acceptable entry but use smaller position size",
                "small_speculative_position": "High risk but extremely oversold - small position only",
                "do_not_catch_falling_knife": "Trend is broken - wait for stabilization",
                "wait_for_better_setup": "Conditions not favorable for dip buying yet",
            }.get(buy_recommendation, ""),
        },
        "method": "dip_assessment_v2",
    }


def _build_relative_performance(
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Build relative performance vs benchmark (SPY).

    Uses aligned data from risk_metrics beta calculation.
    """
    returns = tech_data.get("returns", {})
    beta_data = risk_data.get("beta", {})

    # Get stock returns
    stock_return_1y = returns.get("return_1y")
    stock_return_3m = returns.get("return_3m")
    stock_return_1m = returns.get("return_1m")

    # For benchmark returns, we need to estimate from beta relationship
    # or note that we don't have direct benchmark returns here
    # The proper way is to fetch benchmark data separately,
    # but for now we'll note it's not available
    benchmark_return_1y: float | None = None

    # Calculate alpha if we have beta and benchmark return
    # Since we don't have direct benchmark returns in current data,
    # we'll return None for alpha with a note
    alpha_1y: float | None = None
    outperformed_1y: bool | None = None

    warnings: list[str] = []
    if benchmark_return_1y is None:
        warnings.append("benchmark_returns_not_available")

    return {
        "stock_return_1y": stock_return_1y,
        "stock_return_3m": stock_return_3m,
        "stock_return_1m": stock_return_1m,
        "benchmark": "SPY",
        "benchmark_return_1y": benchmark_return_1y,
        "alpha_1y": alpha_1y,
        "outperformed_1y": outperformed_1y,
        "beta": beta_data.get("value"),
        "warnings": warnings or [],
    }


def _build_decision_context(
    signals: dict[str, list[str]],
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
    events_data: dict[str, Any],
    fund_data: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    action_zones: dict[str, Any],
    news_data: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    """
    Build multi-factor decision context explaining what would change the verdict.

    Organized by category with structured top triggers, next update schedules,
    and news catalyst keywords.
    """
    bullish_list = signals.get("bullish", [])
    bearish_list = signals.get("bearish", [])

    # Extract current values for concrete thresholds
    ma = tech_data.get("moving_averages", {})
    rsi_data = tech_data.get("rsi", {})
    current_price = tech_data.get("current_price")
    sma_200 = ma.get("sma_200")
    sma_50 = ma.get("sma_50")
    rsi_value = rsi_data.get("value")

    vol_data = risk_data.get("volatility", {})
    dd_data = risk_data.get("drawdown", {})
    annualized_vol = vol_data.get("annualized")
    max_dd = dd_data.get("max_1y")

    # Get earnings date for next_update
    earnings = events_data.get("earnings", {})
    next_earnings_date = earnings.get("next_date")
    days_until_earnings = earnings.get("days_until")

    # Get valuation assessment
    valuation_assessment = action_zones.get("valuation_assessment", {})
    valuation_gate = valuation_assessment.get("gate")
    is_unprofitable = valuation_assessment.get("is_unprofitable", False)

    # Get tilt from verdict for balancing triggers
    tilt = verdict.get("tilt", "neutral")

    # Get decomposed scores for top triggers
    decomposed = verdict.get("decomposed", {})
    setup_label = decomposed.get("setup")
    business_quality = decomposed.get("business_quality")
    business_quality_status = decomposed.get("business_quality_status", "unknown")
    risk_label = decomposed.get("risk")

    # Get component scores and weights for calculating score_delta
    components = verdict.get("components", {})
    weights_used = verdict.get("weights_used", {})

    # Get valuation metrics for reasons
    val = fund_data.get("valuation", {})
    yield_m = fund_data.get("yield_metrics", {})
    profit = fund_data.get("profitability", {})
    cf = fund_data.get("cash_flow", {})
    growth = fund_data.get("growth", {})
    ps_trailing = val.get("ps_trailing")
    ps_source = val.get("ps_source")
    ev_to_sales = val.get("ev_to_sales")
    ev_to_sales_source = val.get("ev_to_sales_source")
    pe_trailing = val.get("pe_trailing")
    net_margin = profit.get("net_margin")
    fcf = cf.get("free_cash_flow_ttm")
    fcf_period = cf.get("free_cash_flow_period")
    fcf_currency = cf.get("currency")
    fcf_period_end = cf.get("free_cash_flow_period_end")
    fcf_label = _format_fcf_label(
        fcf,
        fcf_period,
        fcf_currency,
        fcf_period_end,
    )
    revenue_yoy = growth.get("revenue_yoy")

    # Get burn_metrics from fundamentals_summary (already computed with liquidity)
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}
    cash_runway_quarters = burn_metrics.get("cash_runway_quarters")
    runway_basis = burn_metrics.get("runway_basis")
    dilution_analysis = burn_metrics.get("dilution_analysis")

    # === STRUCTURED TOP TRIGGERS ===
    # Each trigger includes score_delta = component_score * weight_used
    # This shows actual contribution to final score, not just weight
    top_triggers: list[dict[str, Any]] = []

    # Helper to calculate score_delta from component
    def _calc_score_delta(category: str) -> float | None:
        """Calculate score contribution: component_score * weight_used.

        INVARIANT: sum of all score_deltas == score_raw (no rounding until display)
        """
        comp_score = components.get(category)
        weight = weights_used.get(category)
        if comp_score is not None and weight is not None:
            return comp_score * weight  # No rounding - preserve full precision
        return None

    # Get component scores for display
    risk_component = components.get("risk")
    fund_component = components.get("fundamentals")
    tech_component = components.get("technicals")

    # Add bearish triggers first (most important for risk awareness)
    # Collapse risk regime with supporting details to avoid double-counting
    if risk_label in ("high", "extreme"):
        score_delta = _calc_score_delta("risk")
        # Build detailed reason with supporting factors
        risk_summary = verdict.get("inputs_used", {})
        vol_val = risk_summary.get("annualized_vol")
        dd_val = risk_summary.get("max_drawdown_1y")
        reason_parts = [f"risk_regime={risk_label}"]
        if vol_val is not None:
            reason_parts.append(f"vol={vol_val*100:.0f}%")
        if dd_val is not None:
            reason_parts.append(f"dd={dd_val*100:.0f}%")
        top_triggers.append({
            "id": "elevated_risk_regime",
            "category": "risk",
            "direction": "bearish",
            "reason": " | ".join(reason_parts),
            "component_score": risk_component,
            "weight_used": weights_used.get("risk"),
            "score_delta": score_delta,
        })

    if business_quality in ("poor", "unprofitable"):
        reason_parts = []
        if net_margin is not None and net_margin < 0:
            reason_parts.append(f"unprofitable (margin={net_margin*100:.0f}%)")
        if fcf is not None and fcf < 0:
            reason_parts.append("negative_fcf")
        reason = " and ".join(reason_parts) if reason_parts else f"business_quality={business_quality}"
        score_delta = _calc_score_delta("fundamentals")
        trigger: dict[str, Any] = {
            "id": "weak_business_quality",
            "category": "fundamentals",
            "direction": "bearish",
            "reason": reason,
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": score_delta,
        }
        if next_earnings_date:
            trigger["next_update"] = {"event": "earnings", "date": next_earnings_date}
        top_triggers.append(trigger)

    if valuation_gate == "headwind":
        # Use the same metric that drove the valuation gate decision
        val_basis = valuation_assessment.get("basis")
        if is_unprofitable:
            # For unprofitable, prefer EV/S over P/S (same logic as gate)
            if val_basis == "ev_to_sales" and ev_to_sales is not None:
                reason = f"ev_to_sales={ev_to_sales:.1f}x (unprofitable, debt/cash-adjusted)"
            elif ps_trailing is not None:
                reason = f"ps_trailing={ps_trailing:.1f}x (unprofitable, source={ps_source or 'unknown'})"
            else:
                reason = "valuation_stretched (unprofitable)"
        elif pe_trailing is not None:
            reason = f"pe_trailing={pe_trailing:.1f}x"
        else:
            reason = "valuation_stretched"
        # Valuation uses fundamentals component
        score_delta = _calc_score_delta("fundamentals")
        top_triggers.append({
            "id": "valuation_headwind",
            "category": "valuation",
            "direction": "bearish",
            "reason": reason,
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": score_delta,
        })

    # Revenue decline trigger (severe decline is a major bearish signal)
    if revenue_yoy is not None and revenue_yoy < -0.20:
        score_delta = _calc_score_delta("fundamentals")
        rev_trigger: dict[str, Any] = {
            "id": "severe_revenue_decline",
            "category": "fundamentals",
            "direction": "bearish",
            "reason": f"revenue_yoy={revenue_yoy*100:.0f}% (severe decline)",
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": score_delta,
        }
        if next_earnings_date:
            rev_trigger["next_update"] = {"event": "earnings", "date": next_earnings_date}
        top_triggers.append(rev_trigger)

    # Dilution risk for unprofitable companies with low runway
    if is_unprofitable and cash_runway_quarters is not None and cash_runway_quarters < 8:
        runway_years = round(cash_runway_quarters / 4, 1)
        dilution_reason = f"cash_runway={runway_years}y ({runway_basis or 'fcf'})"
        if dilution_analysis:
            dilution_pct = dilution_analysis.get("dilution_if_raised_today")
            dilution_level = dilution_analysis.get("dilution_risk_level")
            if dilution_pct is not None:
                dilution_reason += f" - {dilution_pct*100:.0f}% dilution if raised ({dilution_level})"
        score_delta = _calc_score_delta("fundamentals")
        top_triggers.append({
            "id": "dilution_risk",
            "category": "fundamentals",
            "direction": "bearish",
            "reason": dilution_reason,
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": score_delta,
        })

    if setup_label in ("weak", "poor"):
        score_delta = _calc_score_delta("technicals")
        top_triggers.append({
            "id": "weak_technical_setup",
            "category": "technicals",
            "direction": "bearish",
            "reason": f"setup={setup_label}",
            "component_score": tech_component,
            "weight_used": weights_used.get("technicals"),
            "score_delta": score_delta,
        })

    # === COLLECT BULLISH TRIGGERS (always, not just when no bearish) ===
    bullish_triggers: list[dict[str, Any]] = []

    if business_quality == "strong":
        score_delta = _calc_score_delta("fundamentals")
        reason = "profitable"
        if fcf_label:
            reason = f"{reason}, {fcf_label}"
        bullish_fund_trigger: dict[str, Any] = {
            "id": "strong_business_quality",
            "category": "fundamentals",
            "direction": "bullish",
            "reason": reason,
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": score_delta,
        }
        if next_earnings_date:
            bullish_fund_trigger["next_update"] = {"event": "earnings", "date": next_earnings_date}
        bullish_triggers.append(bullish_fund_trigger)
    elif business_quality == "moderate":
        score_delta = _calc_score_delta("fundamentals")
        bullish_fund_trigger = {
            "id": "moderate_business_quality",
            "category": "fundamentals",
            "direction": "bullish",
            "reason": "profitable business with some growth",
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": score_delta,
        }
        if next_earnings_date:
            bullish_fund_trigger["next_update"] = {"event": "earnings", "date": next_earnings_date}
        bullish_triggers.append(bullish_fund_trigger)

    if valuation_gate == "attractive":
        # Use the same metric that drove the valuation gate decision
        val_basis = valuation_assessment.get("basis")
        if is_unprofitable:
            # For unprofitable, prefer EV/S over P/S (same logic as gate)
            if val_basis == "ev_to_sales" and ev_to_sales is not None:
                reason = f"ev_to_sales={ev_to_sales:.1f}x reasonable (debt/cash-adjusted)"
            elif ps_trailing is not None:
                reason = f"ps_trailing={ps_trailing:.1f}x reasonable (source={ps_source or 'unknown'})"
            else:
                reason = "valuation_attractive (unprofitable)"
        else:
            fcf_yield = yield_m.get("fcf_yield")
            earnings_yield = yield_m.get("earnings_yield")
            peg_ratio = val.get("peg_ratio")
            if val_basis == "fcf_yield" and fcf_yield is not None:
                reason = f"fcf_yield={fcf_yield*100:.1f}%"
            elif val_basis == "earnings_yield" and earnings_yield is not None:
                reason = f"earnings_yield={earnings_yield*100:.1f}%"
            elif val_basis == "peg_ratio" and peg_ratio is not None:
                reason = f"peg_ratio={peg_ratio:.2f}"
            elif val_basis == "pe_trailing" and pe_trailing is not None:
                reason = f"pe_trailing={pe_trailing:.1f}x"
            else:
                reason = "valuation_attractive"
        score_delta = _calc_score_delta("fundamentals")
        bullish_triggers.append({
            "id": "attractive_valuation",
            "category": "valuation",
            "direction": "bullish",
            "reason": reason,
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": score_delta,
        })

    if setup_label == "strong":
        score_delta = _calc_score_delta("technicals")
        bullish_triggers.append({
            "id": "strong_technical_setup",
            "category": "technicals",
            "direction": "bullish",
            "reason": f"setup={setup_label}",
            "component_score": tech_component,
            "weight_used": weights_used.get("technicals"),
            "score_delta": score_delta,
        })
    elif setup_label == "moderate":
        score_delta = _calc_score_delta("technicals")
        bullish_triggers.append({
            "id": "moderate_technical_setup",
            "category": "technicals",
            "direction": "bullish",
            "reason": f"setup={setup_label} (price above key MAs)",
            "component_score": tech_component,
            "weight_used": weights_used.get("technicals"),
            "score_delta": score_delta,
        })

    if risk_label == "low":
        score_delta = _calc_score_delta("risk")
        bullish_triggers.append({
            "id": "favorable_risk_regime",
            "category": "risk",
            "direction": "bullish",
            "reason": "risk_regime=low",
            "component_score": risk_component,
            "weight_used": weights_used.get("risk"),
            "score_delta": score_delta,
        })
    elif risk_label == "moderate":
        score_delta = _calc_score_delta("risk")
        bullish_triggers.append({
            "id": "acceptable_risk_regime",
            "category": "risk",
            "direction": "bullish",
            "reason": "risk_regime=moderate (manageable)",
            "component_score": risk_component,
            "weight_used": weights_used.get("risk"),
            "score_delta": score_delta,
        })

    # === ADD FALLBACK TRIGGERS ===
    # Only add these if we don't have enough primary bearish triggers
    # Note: vol/drawdown are already included in elevated_risk_regime reason, so don't duplicate
    fallback_triggers: list[dict[str, Any]] = []

    # Check if we already have elevated_risk_regime (which includes vol/dd details)
    has_risk_regime_trigger = any(t.get("id") == "elevated_risk_regime" for t in top_triggers)

    # Get risk metrics for fallback triggers (only used if no regime trigger)
    risk_summary = verdict.get("inputs_used", {})
    annualized_vol = risk_summary.get("annualized_vol")
    max_drawdown = risk_summary.get("max_drawdown_1y")

    # Only add vol/drawdown as separate triggers if risk regime is NOT already a trigger
    # This avoids double-counting the same risk factors
    if not has_risk_regime_trigger:
        if "very_high_volatility" in bearish_list and annualized_vol is not None:
            fallback_triggers.append({
                "id": "very_high_volatility",
                "category": "risk",
                "direction": "bearish",
                "reason": f"volatility={annualized_vol*100:.0f}% (>60%)",
                "component_score": risk_component,
                "weight_used": weights_used.get("risk"),
                "score_delta": _calc_score_delta("risk"),
            })
        elif "high_volatility" in bearish_list and annualized_vol is not None:
            fallback_triggers.append({
                "id": "high_volatility",
                "category": "risk",
                "direction": "bearish",
                "reason": f"volatility={annualized_vol*100:.0f}% (>40%)",
                "component_score": risk_component,
                "weight_used": weights_used.get("risk"),
                "score_delta": _calc_score_delta("risk"),
            })

        if "deep_drawdown" in bearish_list and max_drawdown is not None:
            fallback_triggers.append({
                "id": "deep_drawdown",
                "category": "risk",
                "direction": "bearish",
                "reason": f"drawdown={max_drawdown*100:.0f}% (>35%)",
                "component_score": risk_component,
                "weight_used": weights_used.get("risk"),
                "score_delta": _calc_score_delta("risk"),
            })

    # Burn metrics missing sub-trigger (for unprofitable companies) - always valuable
    burn_status = burn_metrics.get("status")
    if is_unprofitable and burn_status == "unavailable":
        fallback_triggers.append({
            "id": "runway_unknown",
            "category": "fundamentals",
            "direction": "bearish",
            "reason": f"burn_metrics unavailable ({burn_metrics.get('status_reason', 'unknown')})",
            "component_score": fund_component,
            "weight_used": weights_used.get("fundamentals"),
            "score_delta": _calc_score_delta("fundamentals"),
        })

    # Severe revenue decline - always a distinct concern worth showing
    if revenue_yoy is not None and revenue_yoy < -0.20:
        # Only add if not already in top_triggers
        if not any(t.get("id") == "severe_revenue_decline" for t in top_triggers):
            fallback_triggers.append({
                "id": "severe_revenue_decline",
                "category": "fundamentals",
                "direction": "bearish",
                "reason": f"revenue_yoy={revenue_yoy*100:.0f}% (severe decline)",
                "component_score": fund_component,
                "weight_used": weights_used.get("fundamentals"),
                "score_delta": _calc_score_delta("fundamentals"),
            })

    # === BALANCE TRIGGERS BASED ON TILT ===
    # OPTION A: Only include triggers with actual score contribution
    # Triggers from components that didn't fire (score_delta=None) are excluded
    # This keeps Top Drivers = "scoring drivers only" for auditability

    # Filter to only scoring triggers (non-null score_delta)
    scoring_bearish = [
        t for t in top_triggers
        if t.get("direction") == "bearish" and t.get("score_delta") is not None
    ]
    scoring_bullish = [
        t for t in bullish_triggers
        if t.get("score_delta") is not None
    ]

    # Sort bearish by absolute score_delta (most impactful first)
    scoring_bearish.sort(key=lambda x: abs(x.get("score_delta") or 0), reverse=True)

    # Sort bullish by absolute score_delta
    scoring_bullish.sort(key=lambda x: abs(x.get("score_delta") or 0), reverse=True)

    # Reassign for downstream logic
    bearish_triggers = scoring_bearish
    bullish_triggers = scoring_bullish

    # Determine target counts based on tilt
    if tilt == "neutral":
        target_bearish = 2
        target_bullish = 1
    elif tilt == "bullish":
        target_bearish = 1
        target_bullish = 2
    elif tilt == "bearish":
        target_bearish = 2
        target_bullish = 1
    else:
        target_bearish = 2
        target_bullish = 1

    # Add fallback triggers if we don't have enough bearish
    # Avoid duplicates by checking IDs
    # Only include fallbacks with non-null score_delta (Option A: scoring drivers only)
    existing_ids = {t.get("id") for t in bearish_triggers}
    for fallback in fallback_triggers:
        if fallback.get("id") not in existing_ids and fallback.get("score_delta") is not None:
            bearish_triggers.append(fallback)
            existing_ids.add(fallback.get("id"))

    # Re-sort after adding sub-triggers
    bearish_triggers.sort(key=lambda x: abs(x.get("score_delta") or 0), reverse=True)

    # Build final triggers with target counts
    final_triggers: list[dict[str, Any]] = []
    final_triggers.extend(bearish_triggers[:target_bearish])
    final_triggers.extend(bullish_triggers[:target_bullish])

    # Track if we couldn't meet target count
    triggers_incomplete_reason: str | None = None
    if len(bearish_triggers) < target_bearish and len(bullish_triggers) < target_bullish:
        triggers_incomplete_reason = "insufficient_trigger_candidates"
    elif len(bearish_triggers) < target_bearish:
        triggers_incomplete_reason = "insufficient_bearish_candidates"
    elif len(bullish_triggers) < target_bullish:
        triggers_incomplete_reason = "insufficient_bullish_candidates"

    # If we have no triggers at all, just use whatever we collected
    if not final_triggers:
        final_triggers = bearish_triggers + bullish_triggers

    # Replace top_triggers with balanced set
    top_triggers = final_triggers

    # === FUNDAMENTALS CATEGORY ===
    # === MACHINE-CHECKABLE CONDITION BUILDER ===
    # Each condition has structured fields for automated monitoring:
    # - id: unique identifier
    # - data_source: yfinance field path (e.g., "yfinance.info.profitMargins")
    # - operator: <, <=, >, >=, ==, !=
    # - target_value: numeric threshold
    # - current_value: current numeric value (for comparison)
    # - condition: human-readable label
    # - threshold: human-readable threshold description
    # - current: human-readable current value
    # - next_update: when to check again

    def _make_condition(
        id: str,
        condition: str,
        data_source: str,
        operator: str,
        target_value: float | None,
        current_value: float | None,
        threshold_str: str,
        current_str: str,
        next_update: str | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        """Build a machine-checkable condition dict."""
        result: dict[str, Any] = {
            "id": id,
            "condition": condition,
            "data_source": data_source,
            "operator": operator,
            "target_value": target_value,
            "current_value": current_value,
            "threshold": threshold_str,
            "current": current_str,
        }
        if next_update:
            result["next_update"] = next_update
        if priority:
            result["priority"] = priority
        return result

    # For unprofitable companies, ALWAYS include path-to-profitability checkpoints
    # These are thesis-critical and should never be "N/A"
    fundamentals_bullish: list[dict[str, Any]] = []
    fundamentals_bearish: list[dict[str, Any]] = []

    if is_unprofitable:
        # ALWAYS include profitability checkpoint for unprofitable companies
        fundamentals_bullish.append(_make_condition(
            id="returns_to_profitability",
            condition="returns_to_profitability",
            data_source="yfinance.info.profitMargins",
            operator=">",
            target_value=0.0,
            current_value=net_margin,
            threshold_str="net_margin > 0 for 2 consecutive quarters",
            current_str=f"net_margin = {net_margin*100:.1f}%" if net_margin is not None else "net_margin = unknown",
            next_update=next_earnings_date,
            priority="critical",
        ))
        # ALWAYS include FCF checkpoint for unprofitable companies
        fundamentals_bullish.append(_make_condition(
            id="fcf_turns_positive",
            condition="fcf_turns_positive",
            data_source="yfinance.info.freeCashflow",
            operator=">",
            target_value=0.0,
            current_value=fcf,
            threshold_str="FCF > 0 for 2 consecutive quarters",
            current_str=f"FCF = ${fcf/1e6:.0f}M" if fcf is not None else "FCF = unknown",
            next_update=next_earnings_date,
            priority="high",
        ))
        # ALWAYS include revenue growth checkpoint for unprofitable companies
        fundamentals_bullish.append(_make_condition(
            id="revenue_stabilizes",
            condition="revenue_stabilizes",
            data_source="yfinance.info.revenueGrowth",
            operator=">",
            target_value=-0.10,
            current_value=revenue_yoy,
            threshold_str="revenue_yoy > -10%",
            current_str=f"revenue_yoy = {revenue_yoy*100:.0f}%" if revenue_yoy is not None else "revenue_yoy = unknown",
            next_update=next_earnings_date,
            priority="medium",
        ))
        # Add bearish conditions
        fundamentals_bearish.append(_make_condition(
            id="revenue_collapses",
            condition="revenue_collapses",
            data_source="yfinance.info.revenueGrowth",
            operator="<",
            target_value=-0.30,
            current_value=revenue_yoy,
            threshold_str="revenue_yoy < -30%",
            current_str=f"revenue_yoy = {revenue_yoy*100:.0f}%" if revenue_yoy is not None else "revenue_yoy = unknown",
            next_update=next_earnings_date,
        ))
    else:
        # Profitable company checkpoints
        if "unprofitable" in bearish_list:
            fundamentals_bullish.append({
                "condition": "returns_to_profitability",
                "threshold": "net_margin > 0 for 2 consecutive quarters",
                "next_update": next_earnings_date,
            })
        if "profitable" in bullish_list:
            fundamentals_bearish.append({
                "condition": "earnings_turn_negative",
                "threshold": "net_margin < 0",
                "next_update": next_earnings_date,
            })

        if "negative_free_cash_flow" in bearish_list:
            fundamentals_bullish.append({
                "condition": "fcf_turns_positive",
                "threshold": "FCF > 0 for 2 consecutive quarters",
                "next_update": next_earnings_date,
            })
        if "positive_free_cash_flow" in bullish_list:
            fundamentals_bearish.append({
                "condition": "fcf_turns_negative",
                "threshold": "FCF < 0",
                "next_update": next_earnings_date,
            })

        if "high_growth" in bullish_list:
            fundamentals_bearish.append({
                "condition": "growth_decelerates",
                "threshold": "revenue_yoy < 15%",
                "next_update": next_earnings_date,
            })
        if "declining_growth" in bearish_list:
            fundamentals_bullish.append({
                "condition": "growth_accelerates",
                "threshold": "revenue_yoy > 10%",
                "next_update": next_earnings_date,
            })

    # === VALUATION CATEGORY ===
    valuation_bullish: list[dict[str, Any]] = []
    valuation_bearish: list[dict[str, Any]] = []

    # Get valuation metrics for thresholds (reuse val/yield_m from above)
    fcf_yield_val = yield_m.get("fcf_yield")
    pe_forward = val.get("pe_forward")
    peg_ratio = val.get("peg_ratio")

    # For unprofitable companies, prefer EV/S over P/S thresholds
    if is_unprofitable:
        # Use the same metric that drives the valuation gate
        val_basis = valuation_assessment.get("basis")
        if val_basis == "ev_to_sales" and ev_to_sales is not None:
            # EV/S preferred - adjusts for debt/cash position
            valuation_bullish.append(_make_condition(
                id="evs_contracts",
                condition="evs_contracts",
                data_source="computed.ev_to_sales",
                operator="<=",
                target_value=3.0,
                current_value=ev_to_sales,
                threshold_str="ev_to_sales <= 3",
                current_str=f"ev_to_sales = {ev_to_sales:.1f}",
            ))
            valuation_bearish.append(_make_condition(
                id="evs_expands",
                condition="evs_expands",
                data_source="computed.ev_to_sales",
                operator=">=",
                target_value=10.0,
                current_value=ev_to_sales,
                threshold_str="ev_to_sales >= 10",
                current_str=f"ev_to_sales = {ev_to_sales:.1f}",
            ))
        elif ps_trailing is not None:
            # Fallback to P/S if EV/S unavailable
            valuation_bullish.append(_make_condition(
                id="ps_contracts",
                condition="ps_contracts",
                data_source="yfinance.info.priceToSalesTrailing12Months",
                operator="<=",
                target_value=3.0,
                current_value=ps_trailing,
                threshold_str="ps_trailing <= 3",
                current_str=f"ps_trailing = {ps_trailing:.1f}",
            ))
            valuation_bearish.append(_make_condition(
                id="ps_expands",
                condition="ps_expands",
                data_source="yfinance.info.priceToSalesTrailing12Months",
                operator=">=",
                target_value=10.0,
                current_value=ps_trailing,
                threshold_str="ps_trailing >= 10",
                current_str=f"ps_trailing = {ps_trailing:.1f}",
            ))
    else:
        # For profitable companies, use P/E and FCF yield
        if valuation_gate == "headwind":
            if pe_forward is not None and pe_forward > 30:
                valuation_bullish.append({
                    "condition": "pe_contracts",
                    "threshold": "forward_pe < 25",
                    "current": f"forward_pe = {pe_forward:.1f}",
                })
            if fcf_yield_val is not None and fcf_yield_val > 0 and fcf_yield_val < 0.03:
                valuation_bullish.append({
                    "condition": "fcf_yield_expands",
                    "threshold": "fcf_yield > 4%",
                    "current": f"fcf_yield = {fcf_yield_val * 100:.1f}%",
                })
        elif valuation_gate == "attractive":
            if pe_forward is not None:
                valuation_bearish.append({
                    "condition": "pe_expands",
                    "threshold": "forward_pe > 30",
                    "current": f"forward_pe = {pe_forward:.1f}",
                })
            if fcf_yield_val is not None and fcf_yield_val > 0:
                valuation_bearish.append({
                    "condition": "fcf_yield_compresses",
                    "threshold": "fcf_yield < 3%",
                    "current": f"fcf_yield = {fcf_yield_val * 100:.1f}%",
                })

        if "high_pe" in bearish_list and pe_forward is not None:
            valuation_bullish.append({
                "condition": "valuation_normalizes",
                "threshold": "forward_pe < 25 or growth accelerates",
                "current": f"forward_pe = {pe_forward:.1f}",
            })
        if "low_peg" in bullish_list and peg_ratio is not None:
            valuation_bearish.append({
                "condition": "peg_expands",
                "threshold": "peg_ratio > 2.0",
                "current": f"peg_ratio = {peg_ratio:.2f}",
            })

    # === NEWS CATEGORY ===
    news_triggers: dict[str, Any] = {
        "headline_triggers": {
            "bullish": [
                "beat expectations",
                "raises guidance",
                "buyback",
                "upgrade",
                "new product",
                "partnership",
                "expansion",
            ],
            "bearish": [
                "misses expectations",
                "lowers guidance",
                "downgrade",
                "investigation",
                "lawsuit",
                "recall",
                "layoffs",
                "restructuring",
            ],
        },
    }

    # Add recent sentiment if available
    sentiment = news_data.get("sentiment", {})
    if sentiment:
        news_triggers["current_sentiment"] = sentiment.get("overall")
        news_triggers["sentiment_confidence"] = sentiment.get("confidence")

    # === RISK CATEGORY ===
    risk_bullish: list[dict[str, Any]] = []
    risk_bearish: list[dict[str, Any]] = []

    # Thresholds aligned with risk_regime boundaries:
    # extreme: >60%, high: 40-60%, medium: 25-40%, low: <25%
    # bullish_if targets the next lower regime boundary
    vol_threshold = _vol_threshold_for_improvement(risk_label, annualized_vol)
    if annualized_vol is not None and vol_threshold is not None and annualized_vol >= vol_threshold:
        risk_bullish.append({
            "condition": "volatility_decreases",
            "threshold": f"annualized_vol < {vol_threshold * 100:.0f}%",
            "current": f"{annualized_vol * 100:.1f}%",
        })

    if max_dd is not None and max_dd <= -0.50:
        risk_bullish.append({
            "condition": "drawdown_recovers",
            "threshold": "max_drawdown_1y > -50%",
            "current": f"{max_dd * 100:.1f}%",
        })

    if "deep_drawdown" in bearish_list and max_dd is not None:
        risk_bearish.append({
            "condition": "drawdown_worsens",
            "threshold": "max_drawdown_1y < -60%",
            "current": f"{max_dd * 100:.1f}%",
        })

    beta_data = risk_data.get("beta", {})
    beta_val = beta_data.get("value")
    if "high_beta" in bearish_list and beta_val is not None:
        risk_bullish.append({
            "condition": "beta_normalizes",
            "threshold": "beta < 1.3",
            "current": f"beta = {beta_val:.2f}",
        })

    # === TECHNICALS CATEGORY ===
    technicals_bullish: list[dict[str, Any]] = []
    technicals_bearish: list[dict[str, Any]] = []

    if "price_below_sma200" in bearish_list and sma_200 is not None:
        technicals_bullish.append({
            "condition": "price_reclaims_sma200",
            "threshold": f"close > ${sma_200:.2f} for 3 sessions",
            "current": f"${current_price:.2f}" if current_price else None,
        })
    if "price_above_sma200" in bullish_list and sma_200 is not None:
        technicals_bearish.append({
            "condition": "price_breaks_sma200",
            "threshold": f"close < ${sma_200:.2f} by >2% for 3 sessions",
            "current": f"${current_price:.2f}" if current_price else None,
        })

    if "death_cross" in bearish_list:
        technicals_bullish.append({
            "condition": "golden_cross_forms",
            "threshold": "SMA50 crosses above SMA200",
            "current": f"SMA50=${sma_50:.2f}, SMA200=${sma_200:.2f}" if sma_50 and sma_200 else None,
        })
    if "golden_cross" in bullish_list:
        technicals_bearish.append({
            "condition": "death_cross_forms",
            "threshold": "SMA50 crosses below SMA200",
            "current": f"SMA50=${sma_50:.2f}, SMA200=${sma_200:.2f}" if sma_50 and sma_200 else None,
        })

    if "rsi_overbought" in bearish_list:
        technicals_bullish.append({
            "condition": "rsi_normalizes",
            "threshold": "RSI < 70",
            "current": f"RSI = {rsi_value:.1f}" if rsi_value else None,
        })
    if "rsi_oversold" in bullish_list:
        technicals_bearish.append({
            "condition": "rsi_fails_to_recover",
            "threshold": "RSI stays < 30 for 10+ sessions",
            "current": f"RSI = {rsi_value:.1f}" if rsi_value else None,
        })

    if "weak_3m_momentum" in bearish_list:
        technicals_bullish.append({
            "condition": "momentum_reverses",
            "threshold": "3m_return > 0%",
        })
    if "strong_3m_momentum" in bullish_list:
        technicals_bearish.append({
            "condition": "momentum_reverses",
            "threshold": "3m_return < 0%",
        })

    # === NEXT CATALYST ===
    next_catalyst: dict[str, Any] | None = None
    if days_until_earnings is not None and days_until_earnings > 0 and next_earnings_date:
        next_catalyst = {
            "event": "earnings",
            "date": next_earnings_date,
            "days_until": days_until_earnings,
        }
    elif next_earnings_date:
        next_catalyst = {
            "event": "earnings",
            "date": next_earnings_date,
        }

    # === THESIS CHECKPOINTS (2-year framework for long-term investors) ===
    thesis_checkpoints = _build_thesis_checkpoints(
        is_unprofitable=is_unprofitable,
        net_margin=net_margin,
        fcf=fcf,
        cash_runway_quarters=cash_runway_quarters,
        revenue_yoy=growth.get("revenue_yoy") if growth else None,
        valuation_gate=valuation_gate,
        risk_label=risk_label,
        business_quality=business_quality,
        next_earnings_date=next_earnings_date,
    )

    # === APPLY PER-CATEGORY LIMITS FOR CLEANER OUTPUT ===
    max_per_category = 2

    def limit_list(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]] | None:
        if not items:
            return None
        return items[:max_items]

    # Build fundamentals status - separating data availability from valuation applicability
    # fundamentals_status: available/missing (did we get fundamental data?)
    # valuation_status: pe_valid/pe_not_meaningful (can we use P/E?)
    fundamentals_fetch_status: str
    fundamentals_status_explanation: str | None = None
    valuation_pe_status: str
    valuation_pe_explanation: str | None = None

    if business_quality_status == "data_missing":
        fundamentals_fetch_status = "missing"
        fundamentals_status_explanation = "Fundamental data unavailable - cannot assess business quality"
        valuation_pe_status = "unavailable"
    elif business_quality_status == "evaluated_unprofitable":
        # Fundamentals ARE available (revenue, margin, FCF) - it's just P/E that's not meaningful
        fundamentals_fetch_status = "available"
        valuation_pe_status = "not_meaningful"
        # Make explanation basis-aware (EV/S preferred over P/S)
        val_basis = valuation_assessment.get("basis")
        if valuation_gate == "unknown":
            valuation_pe_explanation = "P/E not meaningful (unprofitable), no sales multiple available"
        elif val_basis == "ev_to_sales":
            valuation_pe_explanation = "P/E not meaningful (unprofitable), using EV/S instead"
        elif val_basis == "ps_trailing":
            valuation_pe_explanation = "P/E not meaningful (unprofitable), using P/S instead"
        else:
            valuation_pe_explanation = "P/E not meaningful (unprofitable)"
    else:
        fundamentals_fetch_status = "available"
        valuation_pe_status = "valid"

    # EV/S status (preferred for unprofitable companies when debt/cash is material)
    valuation_evs_status: str
    valuation_evs_explanation: str | None = None
    if ev_to_sales is not None:
        valuation_evs_status = "available"
        if ev_to_sales_source == "computed":
            valuation_evs_explanation = "EV/S computed from enterprise_value / revenue_ttm"
    else:
        valuation_evs_status = "unavailable"

    # P/S status (fallback for unprofitable companies if EV/S unavailable)
    valuation_ps_status: str
    valuation_ps_explanation: str | None = None
    if ps_trailing is not None:
        valuation_ps_status = "available"
        if ps_source == "computed":
            valuation_ps_explanation = "P/S computed from market_cap / revenue_ttm"
    else:
        valuation_ps_status = "unavailable"
        if is_unprofitable and valuation_evs_status == "unavailable":
            valuation_ps_explanation = (
                "Neither EV/S nor P/S available - cannot assess valuation for unprofitable company"
            )

    # === HORIZON DRIVERS (policy gates, not score-based) ===
    # These explain why mid_term/long_term are "caution" or "avoid"
    # Separate from top_triggers which are scoring-only
    horizon_fit = verdict.get("horizon_fit", {})
    horizon_drivers: list[dict[str, Any]] = []

    burn_status = burn_metrics.get("status")

    # Use business_quality_status for consistency with _build_horizon_fit
    # This includes both explicit "unprofitable" label AND "evaluated_unprofitable" status
    is_unprofitable_for_horizon = (
        business_quality_status == "evaluated_unprofitable"
        or business_quality == "unprofitable"
    )

    # Long-term horizon drivers - ONLY emit drivers that fired in horizon_fit
    # This ensures 1:1 alignment between horizon_fit reasons and horizon_drivers
    burn_status_reason = burn_metrics.get("status_reason")
    long_term_gates = horizon_fit.get("long_term_gates") or []

    if horizon_fit.get("long_term") in ("caution", "avoid"):
        # Only emit drivers that are in long_term_gates (from horizon_fit)
        # Each gate in horizon_fit.long_term_gates gets a corresponding driver

        if "unprofitable" in long_term_gates:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "unprofitable",
                "reason": "unprofitable - requires thesis on path to profitability",
                "current": f"business_quality={business_quality}",
            })

        if "burn_metrics_missing" in long_term_gates:
            data_gaps_list = ["burn_metrics_unavailable"]
            if burn_status_reason:
                data_gaps_list.append(burn_status_reason)
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "burn_metrics_missing",
                "reason": f"unprofitable with burn metrics unavailable ({burn_status_reason or 'unknown'})",
                "data_gaps": data_gaps_list,
            })

        if "low_runway_confidence" in long_term_gates:
            # Low runway_confidence = runway < 2 years = dilution risk elevated
            runway_conf = burn_metrics.get("runway_confidence")
            runway_quarters = burn_metrics.get("cash_runway_quarters")
            runway_years = round(runway_quarters / 4, 1) if runway_quarters else None
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "low_runway_confidence",
                "reason": f"cash runway {runway_years}y - dilution risk elevated",
                "current": f"runway_confidence={runway_conf}, {runway_quarters:.1f}q" if runway_quarters else f"runway_confidence={runway_conf}",
            })

        if "extreme_risk" in long_term_gates:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "extreme_risk",
                "reason": "extreme risk unsuitable for core holdings",
                "current": f"risk_regime={risk_label}",
            })

        if "severe_revenue_decline" in long_term_gates and revenue_yoy is not None:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "severe_revenue_decline",
                "reason": f"revenue declining {abs(revenue_yoy)*100:.0f}% YoY",
                "current": f"{revenue_yoy*100:.0f}%",
            })

        if "negative_fcf" in long_term_gates and fcf is not None:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "negative_fcf",
                "reason": "negative free cash flow - burning cash",
                "current": f"FCF=${fcf/1e6:.0f}M",
            })

    # Mid-term horizon drivers
    if horizon_fit.get("mid_term") in ("caution", "avoid"):
        if risk_label == "extreme":
            horizon_drivers.append({
                "horizon": "mid_term",
                "direction": "bearish",
                "gate": "extreme_risk",
                "reason": "extreme risk requires minimal position size",
                "current": f"risk_regime={risk_label}",
            })

    return {
        "top_triggers": top_triggers or [],
        "top_triggers_incomplete_reason": triggers_incomplete_reason,
        "horizon_drivers": horizon_drivers or [],
        "fundamentals": {
            "bullish_if": limit_list(fundamentals_bullish, max_per_category),
            "bearish_if": limit_list(fundamentals_bearish, max_per_category),
            "status": fundamentals_fetch_status,  # available/missing (data fetch status)
            "status_explanation": fundamentals_status_explanation,
            "business_quality": business_quality,  # strong/moderate/mixed/poor/unprofitable/weak or None
            "next_update": next_earnings_date,
            "check_frequency": "quarterly_earnings",
        },
        "valuation": {
            "bullish_if": limit_list(valuation_bullish, 1),
            "bearish_if": limit_list(valuation_bearish, 1),
            "current_gate": valuation_gate,
            "basis": valuation_assessment.get("basis"),  # ev_to_sales/ps_trailing/pe_trailing/etc.
            "pe_status": valuation_pe_status,  # valid/not_meaningful/unavailable
            "pe_explanation": valuation_pe_explanation,
            "evs_status": valuation_evs_status,  # available/unavailable (preferred for unprofitable)
            "evs_explanation": valuation_evs_explanation,
            "ps_status": valuation_ps_status,  # available/unavailable (fallback for unprofitable)
            "ps_explanation": valuation_ps_explanation,
            "is_unprofitable": is_unprofitable if is_unprofitable else None,
            "next_update": next_earnings_date,  # Multiples rerate after earnings
            "check_frequency": "quarterly_earnings",
        },
        "news": {
            **news_triggers,
            "check_frequency": "daily_or_weekly",
            "weight_note": "low_weight_unless_high_sample_size",
        },
        "risk": {
            "bullish_if": limit_list(risk_bullish, max_per_category),
            "bearish_if": limit_list(risk_bearish, max_per_category),
            "current_regime": risk_label,
            "check_frequency": "weekly",
            "regime_note": "risk_regime_shifts_slowly",
        },
        "technicals": {
            "bullish_if": limit_list(technicals_bullish, max_per_category),
            "bearish_if": limit_list(technicals_bearish, max_per_category),
            "check_frequency": "weekly_for_long_term",
        },
        "next_catalyst": next_catalyst,
        "thesis_checkpoints": thesis_checkpoints,
    }


def _build_thesis_checkpoints(
    is_unprofitable: bool,
    net_margin: float | None,
    fcf: float | None,
    cash_runway_quarters: float | None,
    revenue_yoy: float | None,
    valuation_gate: str | None,
    risk_label: str | None,
    business_quality: str | None,
    next_earnings_date: str | None = None,
) -> dict[str, Any]:
    """
    Build thesis checkpoints for 2-year investment framework.

    Returns milestones a long-term investor should monitor over the hold period.
    Each checkpoint includes:
    - data_source: which yfinance field to check
    - update_on: when this data updates (quarterly_earnings, weekly, etc.)
    - next_check: optional specific date for next check
    """
    checkpoints: list[dict[str, Any]] = []
    hold_thesis: str | None = None
    review_triggers: list[str] = []
    thesis_stop_triggers: list[dict[str, Any]] = []  # Non-price based exit triggers

    # Determine the investment thesis based on company profile
    if is_unprofitable:
        hold_thesis = "growth_to_profitability"

        # Unprofitable company checkpoints
        checkpoints.append({
            "id": "returns_to_profitability",
            "milestone": "path_to_profitability",
            "target": "achieve positive net margin within 6-8 quarters",
            "current": f"net_margin = {net_margin*100:.1f}%" if net_margin else "unknown",
            "data_source": "yfinance.info.profitMargins",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "critical",
        })

        if fcf is not None and fcf < 0:
            checkpoints.append({
                "id": "fcf_turns_positive",
                "milestone": "fcf_positive",
                "target": "achieve positive FCF within 4-6 quarters",
                "current": "fcf = negative",
                "data_source": "yfinance.info.freeCashflow",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": "high",
            })

        if cash_runway_quarters is not None:
            runway_status = "adequate" if cash_runway_quarters >= 8 else "limited"
            checkpoints.append({
                "id": "maintain_cash_runway",
                "milestone": "maintain_runway",
                "target": "maintain 2+ years cash runway without dilutive raise",
                "current": f"runway = {cash_runway_quarters/4:.1f} years ({runway_status})",
                "data_source": "computed from yfinance.info.{totalCash,freeCashflow,operatingCashflow}",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": "critical" if cash_runway_quarters < 8 else "medium",
            })

        if revenue_yoy is not None:
            # Adaptive growth checkpoint based on current revenue regime
            if revenue_yoy < -0.20:
                # Severe decline: goal is stabilization
                growth_target = "stabilize revenue (YoY > -10% or 2 sequential quarters of improvement)"
                growth_priority = "critical"
            elif revenue_yoy < -0.10:
                # Moderate decline: goal is re-acceleration
                growth_target = "return to positive growth (YoY > 0%)"
                growth_priority = "high"
            elif revenue_yoy < 0.10:
                # Flat/slow: goal is acceleration
                growth_target = "accelerate growth (YoY > 15%)"
                growth_priority = "high"
            else:
                # Already growing: goal is sustaining
                growth_target = "maintain 15%+ revenue growth while improving margins"
                growth_priority = "medium"

            checkpoints.append({
                "id": "revenue_trajectory_improves",
                "milestone": "revenue_trajectory",
                "target": growth_target,
                "current": f"revenue_yoy = {revenue_yoy*100:.1f}%",
                "data_source": "yfinance.info.revenueGrowth",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": growth_priority,
            })

        review_triggers = [
            "equity_raise_announced",
            "growth_decelerates_below_15%",
            "cash_runway_drops_below_4_quarters",
            "key_executive_departure",
        ]

        # Thesis stop triggers for unprofitable companies (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "runway_critical",
                "condition": "cash_runway < 4 quarters",
                "action": "exit_or_size_to_zero",
                "rationale": "dilutive raise imminent",
            },
            {
                "trigger": "profitability_path_broken",
                "condition": "net margin worsens 3 consecutive quarters",
                "action": "reassess_thesis",
                "rationale": "path to profitability not progressing",
            },
            {
                "trigger": "growth_collapsed",
                "condition": "revenue_yoy < -30% for 2 quarters",
                "action": "exit_or_reduce",
                "rationale": "growth thesis invalidated",
            },
        ]

    elif business_quality in ("strong", "moderate"):
        # Profitable company checkpoints
        if valuation_gate == "attractive":
            hold_thesis = "undervalued_quality"
            checkpoints.append({
                "id": "valuation_rerates",
                "milestone": "valuation_rerates",
                "target": "P/E or P/S expands toward sector median",
                "data_source": "yfinance.info.{trailingPE,priceToSalesTrailing12Months}",
                "update_on": "weekly",
                "next_check": None,  # Price-based, check weekly
                "priority": "medium",
            })
        else:
            hold_thesis = "quality_compounder"

        checkpoints.append({
            "id": "earnings_growth_continues",
            "milestone": "earnings_growth",
            "target": "maintain or accelerate earnings growth trajectory",
            "current": f"net_margin = {net_margin*100:.1f}%" if net_margin else "unknown",
            "data_source": "yfinance.info.{earningsGrowth,profitMargins}",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "high",
        })

        if fcf is not None and fcf > 0:
            checkpoints.append({
                "id": "capital_allocation_quality",
                "milestone": "capital_allocation",
                "target": "FCF deployed to buybacks, dividends, or accretive M&A",
                "current": "fcf_positive",
                "data_source": "yfinance.info.freeCashflow + SEC filings",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": "medium",
            })

        review_triggers = [
            "margin_compression_2_consecutive_quarters",
            "guidance_cut",
            "competitive_threat_emerges",
            "valuation_exceeds_historical_range",
        ]

        # Thesis stop triggers for quality compounders (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "earnings_deterioration",
                "condition": "EPS declines 2 consecutive quarters (ex one-time)",
                "action": "reassess_position_size",
                "rationale": "compounder thesis requires earnings growth",
            },
            {
                "trigger": "margin_collapse",
                "condition": "operating margin drops >500bps YoY",
                "action": "investigate_and_reassess",
                "rationale": "competitive position may be weakening",
            },
            {
                "trigger": "capital_allocation_concern",
                "condition": "large dilutive acquisition or debt-funded buyback at peak",
                "action": "reassess_management_quality",
                "rationale": "poor capital allocation destroys long-term value",
            },
        ]

    elif business_quality == "mixed":
        hold_thesis = "turnaround_or_cyclical"
        checkpoints.append({
            "id": "operational_improvement",
            "milestone": "operational_improvement",
            "target": "margin expansion and/or revenue acceleration",
            "data_source": "yfinance.info.{operatingMargins,revenueGrowth}",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "high",
        })
        checkpoints.append({
            "id": "balance_sheet_strengthening",
            "milestone": "balance_sheet_strengthening",
            "target": "debt reduction or cash build",
            "data_source": "yfinance.info.{totalDebt,totalCash}",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "medium",
        })

        review_triggers = [
            "turnaround_thesis_invalidated",
            "industry_downturn_extends",
            "management_credibility_issue",
        ]

        # Thesis stop triggers for turnarounds (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "turnaround_stalled",
                "condition": "no margin improvement after 4 quarters",
                "action": "exit_position",
                "rationale": "turnaround thesis has failed",
            },
            {
                "trigger": "balance_sheet_worsening",
                "condition": "debt/equity increases or liquidity deteriorates",
                "action": "reduce_or_exit",
                "rationale": "financial stress increasing",
            },
        ]

    else:
        # Poor or unknown business quality
        hold_thesis = "speculative"
        checkpoints.append({
            "milestone": "thesis_validation",
            "target": "concrete evidence of business improvement",
            "priority": "critical",
        })

        review_triggers = [
            "no_improvement_after_4_quarters",
            "deteriorating_fundamentals",
        ]

        # Thesis stop triggers for speculative positions (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "thesis_invalidated",
                "condition": "no evidence of improvement after 4 quarters",
                "action": "exit_position",
                "rationale": "speculative thesis requires rapid validation",
            },
        ]

    # Add risk-related checkpoint if relevant
    if risk_label in ("high", "extreme"):
        checkpoints.append({
            "milestone": "risk_normalization",
            "target": "volatility decreases to moderate levels",
            "current": f"risk_regime = {risk_label}",
            "priority": "medium",
        })

    return {
        "hold_thesis": hold_thesis,
        "checkpoints": checkpoints[:4],  # Limit to top 4 checkpoints
        "review_triggers": review_triggers[:4],  # Limit to top 4 triggers
        "thesis_stop_triggers": thesis_stop_triggers[:3] if thesis_stop_triggers else None,
        "review_frequency": "quarterly_after_earnings",
    }


# === EXECUTIVE SUMMARY HELPERS ===

# Deterministic mappings: internal enums → investor-friendly wording
_ACTION_HUMANIZE: dict[str, str] = {
    # Long-term actions
    "avoid_core_position": "avoid as a core holding",
    "hold_watch": "hold and watch",
    "accumulate": "accumulate",
    "strong_buy": "strong buy",
    # Mid-term actions
    "wait_for_entry": "wait for a better entry",
    "consider_entry": "consider entry",
    "hold": "hold",
    "reduce": "reduce",
    "exit": "exit",
}

# Gate ordering (fixed order for diff stability)
_GATE_ORDER: list[str] = [
    "unprofitable",
    "negative_fcf",
    "severe_revenue_decline",
    "burn_metrics_missing",
    "extreme_risk",
    "low_runway_confidence",
    "high_debt",
]


def _humanize_action(action: str) -> str:
    """Convert internal action enum to investor-friendly wording."""
    return _ACTION_HUMANIZE.get(action, action.replace("_", " "))


def _humanize_gates_with_numbers(
    gates: list[str],
    rev_yoy: float | None,
    net_margin: float | None,
    quarterly_burn: float | None,
    runway_years: float | None,
    vol: float | None,
    dd: float | None,
) -> str:
    """
    Convert gate list to investor-friendly string with numeric anchors.

    Fixed order ensures diff stability. Embeds the actual numbers so investors
    see the "why" immediately.
    """
    # Sort gates by fixed order (unknown gates go to end)
    def gate_sort_key(g: str) -> int:
        try:
            return _GATE_ORDER.index(g)
        except ValueError:
            return len(_GATE_ORDER)

    sorted_gates = sorted(gates, key=gate_sort_key)

    # Build humanized gate strings with numbers
    humanized: list[str] = []
    for gate in sorted_gates[:4]:  # Max 4 gates
        if gate == "unprofitable":
            if net_margin is not None:
                humanized.append(f"unprofitable (margin {net_margin * 100:.0f}%)")
            else:
                humanized.append("unprofitable")
        elif gate == "negative_fcf":
            if quarterly_burn is not None and quarterly_burn > 0:
                burn_str = f"burning cash (~${quarterly_burn / 1_000_000:.0f}M/q"
                # Add runway if available (time dimension investors care about)
                if runway_years is not None:
                    burn_str += f"; runway ~{runway_years:.1f}y"
                burn_str += ")"
                humanized.append(burn_str)
            else:
                humanized.append("negative FCF")
        elif gate == "severe_revenue_decline":
            if rev_yoy is not None:
                humanized.append(f"revenue {rev_yoy * 100:+.0f}% YoY")
            else:
                humanized.append("severe revenue decline")
        elif gate == "extreme_risk":
            risk_parts = []
            if vol is not None:
                risk_parts.append(f"vol {vol * 100:.0f}%")
            if dd is not None:
                risk_parts.append(f"dd {dd * 100:.0f}%")
            if risk_parts:
                humanized.append(f"extreme risk ({', '.join(risk_parts)})")
            else:
                humanized.append("extreme risk")
        elif gate == "burn_metrics_missing":
            humanized.append("burn metrics unavailable")
        elif gate == "low_runway_confidence":
            if runway_years is not None:
                humanized.append(f"runway ~{runway_years:.1f}y (low confidence)")
            else:
                humanized.append("low runway confidence")
        elif gate == "high_debt":
            humanized.append("high debt")
        else:
            # Unknown gate - just humanize the underscore
            humanized.append(gate.replace("_", " "))

    return "; ".join(humanized) if humanized else "multiple concerns"


def _build_executive_summary(
    summary: dict[str, Any],
    technicals_summary: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    risk_summary: dict[str, Any],
    signals: dict[str, list[str]],
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    policy_action: dict[str, Any],
    news_summary: dict[str, Any] | None = None,
) -> str:
    """
    Build a MATERIALITY-FIRST deterministic executive summary from structured fields.

    DETERMINISM CONTRACT:
    - Lead sentence determined by materiality priority (policy gates > risk > fundamentals > technicals)
    - Fixed gate ordering for diff stability
    - Fixed rounding: percents 0-1dp, money in $M, runway 1dp
    - Humanized action names (avoid_core_position → "avoid as a core holding")
    - Numeric anchors embedded in gates (unprofitable → "unprofitable (margin -140%)")
    - Always includes: lead topic, counterbalancing factor, risk (if elevated), policy

    MATERIALITY PRIORITY (lead sentence topic):
    1. If long_term == "avoid" → lead with WHY (policy gates with numbers)
    2. Else if risk_regime in {extreme, high} → lead with RISK
    3. Else if fundamentals are weak/unprofitable → lead with FUNDAMENTALS
    4. Else → lead with TECHNICALS

    This ensures the summary highlights what actually matters for investment decisions.

    Returns:
        A 3-4 sentence narrative summary string.
    """
    name = summary.get("name", "This stock")
    # Use short name (first word or ticker-like)
    short_name = name.split(",")[0].split()[0] if name else "This stock"

    # === EXTRACT STRUCTURED FIELDS ===
    # Technicals
    trend = technicals_summary.get("trend", {})
    momentum = technicals_summary.get("momentum", {})
    returns = technicals_summary.get("returns", {})
    decomposed = verdict.get("decomposed", {})
    setup = decomposed.get("setup", "neutral")  # strong/weak/neutral

    # Fundamentals
    valuation = fundamentals_summary.get("valuation", {})
    profitability = fundamentals_summary.get("profitability", {})
    growth = fundamentals_summary.get("growth", {})
    cash_flow_summary = fundamentals_summary.get("cash_flow", {})
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}
    is_unprofitable = valuation.get("valuation_note") == "pe_not_meaningful"
    business_quality = decomposed.get("business_quality", "unknown")

    # Risk
    risk_regime = decomposed.get("risk", "unknown")  # extreme/high/medium/low
    vol = risk_summary.get("annualized_volatility")
    dd = risk_summary.get("max_drawdown_1y")

    # Policy (from policy_action - the source of truth)
    mid_term_action = policy_action.get("mid_term", "unknown")
    long_term_action = policy_action.get("long_term", "unknown")
    position_sizing = action_zones.get("position_sizing_range", {})
    suggested_pct = position_sizing.get("suggested_pct_range", [])

    # Horizon fit for materiality determination
    horizon_fit = verdict.get("horizon_fit", {})
    long_term_fit = horizon_fit.get("long_term", "unknown")
    long_term_gates = horizon_fit.get("long_term_gates") or []

    # === BUILD REUSABLE CLAUSES ===

    # Technicals clause
    tech_parts: list[str] = []
    if trend.get("above_sma200"):
        tech_parts.append("above SMA200")
    if trend.get("golden_cross"):
        tech_parts.append("golden cross")
    if momentum.get("macd_bullish"):
        tech_parts.append("MACD bullish")

    return_1m = returns.get("return_1m")
    return_3m = returns.get("return_3m")
    # Always include 1M if significant (>=10%), and 3M if available and significant (>=10%)
    # This avoids "1M pop" bias by showing both timeframes
    if return_1m is not None and abs(return_1m) >= 0.10:
        sign = "+" if return_1m >= 0 else ""
        tech_parts.append(f"1M {sign}{return_1m * 100:.1f}%")
    if return_3m is not None and abs(return_3m) >= 0.10:
        sign = "+" if return_3m >= 0 else ""
        tech_parts.append(f"3M {sign}{return_3m * 100:.1f}%")

    tech_str = ", ".join(tech_parts[:5]) if tech_parts else ""

    # Fundamentals clause parts
    fund_parts: list[str] = []
    rev_yoy = growth.get("revenue_yoy")
    if rev_yoy is not None:
        sign = "" if rev_yoy < 0 else "+"
        fund_parts.append(f"revenue_yoy {sign}{rev_yoy * 100:.1f}%")

    net_margin = profitability.get("net_margin")
    if net_margin is not None:
        fund_parts.append(f"net_margin {net_margin * 100:.1f}%")

    fcf_value = cash_flow_summary.get("free_cash_flow_ttm")
    fcf_period = cash_flow_summary.get("free_cash_flow_period")
    fcf_currency = cash_flow_summary.get("currency")
    fcf_period_end = cash_flow_summary.get("free_cash_flow_period_end")
    fcf_label = _format_fcf_label(
        fcf_value,
        fcf_period,
        fcf_currency,
        fcf_period_end,
    )
    if fcf_label:
        fund_parts.append(fcf_label)
    quarterly_burn = burn_metrics.get("quarterly_fcf_burn")
    if quarterly_burn is not None and quarterly_burn > 0:
        annual_fcf = -quarterly_burn * 4
        fund_parts.append(f"FCF ${annual_fcf / 1_000_000:.0f}M")

    if quarterly_burn is not None and quarterly_burn > 0:
        fund_parts.append(f"burn ~${quarterly_burn / 1_000_000:.0f}M/quarter")

    runway_quarters = burn_metrics.get("cash_runway_quarters")
    runway_years = runway_quarters / 4 if runway_quarters else None
    if runway_years is not None:
        fund_parts.append(f"runway ~{runway_years:.1f} years")

    fund_str = "; ".join(fund_parts[:4]) if fund_parts else ""

    # Risk clause
    risk_details: list[str] = []
    if vol is not None:
        risk_details.append(f"vol {vol * 100:.1f}%")
    if dd is not None:
        risk_details.append(f"dd {dd * 100:.1f}%")
    risk_str = ", ".join(risk_details) if risk_details else ""

    # Policy clause (humanized actions)
    mid_term_human = _humanize_action(mid_term_action)
    long_term_human = _humanize_action(long_term_action)
    policy_parts: list[str] = [f"mid-term {mid_term_human}"]
    if suggested_pct and len(suggested_pct) >= 2:
        sizing_str = f"{suggested_pct[0]:.1f}-{suggested_pct[1]:.1f}%"
        policy_parts.append(f"size ~{sizing_str}")
    policy_clause = "Policy: " + "; ".join(policy_parts) + "."

    # === DETERMINE LEAD TOPIC (MATERIALITY-FIRST) ===
    def _determine_lead_topic() -> str:
        # Priority 1: If long-term is "avoid", lead with policy gates
        if long_term_fit == "avoid":
            return "policy"
        # Priority 2: If risk is elevated, lead with risk
        if risk_regime in ("extreme", "high"):
            return "risk"
        # Priority 3: If fundamentals are weak/unprofitable, lead with fundamentals
        if business_quality in ("unprofitable", "poor", "weak") or is_unprofitable:
            return "fundamentals"
        # Priority 4: Default to technicals
        return "technicals"

    lead_topic = _determine_lead_topic()

    # === BUILD SENTENCES BASED ON LEAD TOPIC ===
    sentences: list[str] = []

    if lead_topic == "policy":
        # Lead: Why it's avoid (use humanized gates with numeric anchors)
        gates_str = _humanize_gates_with_numbers(
            gates=long_term_gates,
            rev_yoy=rev_yoy,
            net_margin=net_margin,
            quarterly_burn=quarterly_burn,
            runway_years=runway_years,
            vol=vol,
            dd=dd,
        )
        sentences.append(f"{short_name} is {long_term_human} long-term ({gates_str}).")

        # Counterbalance: mention technicals if strong
        if setup == "strong" and tech_str:
            sentences.append(f"Technicals are strong ({tech_str}).")
        elif tech_str:
            sentences.append(f"Technicals: {tech_str}.")

    elif lead_topic == "risk":
        # Lead: Risk regime
        if risk_str:
            sentences.append(f"{short_name} has {risk_regime} risk ({risk_str}).")
        else:
            sentences.append(f"{short_name} has {risk_regime} risk.")

        # Fundamentals
        if fund_str:
            if business_quality == "unprofitable" or is_unprofitable:
                sentences.append(f"Fundamentals: unprofitable ({fund_str}).")
            elif business_quality in ("poor", "weak"):
                sentences.append(f"Fundamentals are {business_quality} ({fund_str}).")
            else:
                sentences.append(f"Fundamentals: {fund_str}.")

        # Counterbalance: technicals if strong
        if setup == "strong" and tech_str:
            sentences.append(f"Technicals are strong ({tech_str}).")

    elif lead_topic == "fundamentals":
        # Lead: Fundamentals weakness
        if business_quality == "unprofitable" or is_unprofitable:
            if fund_str:
                sentences.append(f"{short_name} is unprofitable ({fund_str}).")
            else:
                sentences.append(f"{short_name} is unprofitable.")
        elif business_quality in ("poor", "weak"):
            if fund_str:
                sentences.append(f"{short_name} has {business_quality} fundamentals ({fund_str}).")
            else:
                sentences.append(f"{short_name} has {business_quality} fundamentals.")
        else:
            sentences.append(f"{short_name} fundamentals: {fund_str}.")

        # Risk if elevated
        if risk_regime in ("extreme", "high") and risk_str:
            sentences.append(f"Risk is {risk_regime} ({risk_str}).")

        # Counterbalance: technicals if strong
        if setup == "strong" and tech_str:
            sentences.append(f"Technicals are strong ({tech_str}).")

    else:  # technicals (default)
        # Lead: Technical setup
        if setup == "strong" and tech_str:
            sentences.append(f"{short_name} has a strong technical setup ({tech_str}).")
        elif setup == "weak":
            sentences.append(f"{short_name} has a weak technical setup.")
        else:
            if tech_str:
                sentences.append(f"{short_name} shows neutral technicals ({tech_str}).")
            else:
                sentences.append(f"{short_name} shows neutral technicals.")

        # Fundamentals
        if fund_str:
            if business_quality in ("good", "strong"):
                sentences.append(f"Fundamentals are solid ({fund_str}).")
            elif business_quality in ("unprofitable", "poor", "weak") or is_unprofitable:
                sentences.append(f"Fundamentals are weak ({fund_str}).")
            else:
                sentences.append(f"Fundamentals: {fund_str}.")

        # Risk if elevated
        if risk_regime in ("extreme", "high") and risk_str:
            sentences.append(f"Risk is {risk_regime} ({risk_str}).")

    # Always end with policy clause
    sentences.append(policy_clause)

    return " ".join(sentences)


def _build_policy_action(
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    decomposed: dict[str, Any] | None = None,
    risk_regime: dict[str, Any] | None = None,
    dip_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build policy_action - the primary "what to do" decision output.

    This is designed to be the first thing an investor reads.
    Combines: tilt, horizon_fit, zone, valuation_gate, position_sizing.
    Includes conditions_to_upgrade for actionable next steps.

    Returns:
        Dict with mid_term/long_term actions, rationale, and upgrade conditions.
    """
    tilt = verdict.get("tilt", "neutral")
    horizon_fit = verdict.get("horizon_fit", {})
    mid_term_fit = horizon_fit.get("mid_term", "unknown")
    long_term_fit = horizon_fit.get("long_term", "unknown")
    long_term_gates = horizon_fit.get("long_term_gates") or []

    zone = action_zones.get("current_zone")
    valuation_assessment = action_zones.get("valuation_assessment", {})
    valuation_gate = valuation_assessment.get("gate")
    valuation_basis = valuation_assessment.get("basis")
    is_unprofitable = valuation_assessment.get("is_unprofitable", False)
    position_sizing = action_zones.get("position_sizing_range", {})
    dip_type = None
    if dip_assessment:
        dip_type = (dip_assessment.get("dip_classification") or {}).get("type")

    # Get decomposed scores for conditions
    business_quality = decomposed.get("business_quality") if decomposed else None
    business_quality_status = decomposed.get("business_quality_status") if decomposed else None
    risk_label = decomposed.get("risk") if decomposed else None

    # Rationale accumulator - structured for auditability
    rationale: list[str] = []

    # === MID-TERM ACTION (3-12 months) ===
    mid_term_action: str
    if mid_term_fit == "avoid":
        mid_term_action = "avoid"
        rationale.append("mid_term_fit=avoid")
    elif mid_term_fit == "caution":
        if tilt == "bullish":
            mid_term_action = "speculative_small_position"
            rationale.append("mid_term_fit=caution, tilt=bullish")
        else:
            mid_term_action = "wait_for_entry"
            rationale.append(f"mid_term_fit=caution, tilt={tilt}")
    elif mid_term_fit == "ok":
        if tilt == "bullish" and zone in ("strong_buy", "accumulate"):
            mid_term_action = "buy"
            rationale.append(f"mid_term=ok, tilt=bullish, zone={zone}")
        elif tilt == "bullish":
            mid_term_action = "hold_or_add"
            rationale.append(f"mid_term=ok, tilt=bullish, zone={zone}")
        elif tilt == "neutral":
            mid_term_action = "hold"
            rationale.append(f"mid_term=ok, tilt=neutral, zone={zone}")
        else:  # bearish
            mid_term_action = "hold_or_reduce"
            rationale.append(f"mid_term=ok, tilt=bearish, zone={zone}")
    else:  # unknown
        mid_term_action = "insufficient_data"
        rationale.append("mid_term_fit=unknown")

    if dip_type == "falling_knife":
        mid_term_action = "wait_for_entry"
        rationale.append("dip_type=falling_knife")

    # === LONG-TERM ACTION (1-5 years) ===
    long_term_action: str
    if long_term_fit == "avoid":
        long_term_action = "avoid_core_position"
        rationale.append(f"long_term_fit=avoid, gates={long_term_gates}")
    elif long_term_fit == "caution":
        if is_unprofitable:
            if valuation_gate == "attractive":
                long_term_action = "speculative_with_thesis"
                rationale.append("long_term=caution, unprofitable, valuation=attractive")
            else:
                long_term_action = "speculative_only"
                rationale.append("long_term=caution, unprofitable")
        else:
            if tilt == "bullish":
                long_term_action = "small_position_with_stops"
                rationale.append("long_term=caution, tilt=bullish")
            else:
                long_term_action = "wait_or_avoid"
                rationale.append(f"long_term=caution, tilt={tilt}")
    elif long_term_fit == "ok":
        if valuation_gate == "attractive":
            long_term_action = "accumulate"
            rationale.append(f"long_term=ok, valuation=attractive, zone={zone}")
        elif valuation_gate == "headwind":
            long_term_action = "hold_size_conservatively"
            rationale.append(f"long_term=ok, valuation=headwind")
        else:
            if tilt == "bullish":
                long_term_action = "hold_or_add"
            else:
                long_term_action = "hold"
            rationale.append(f"long_term=ok, tilt={tilt}, zone={zone}")
    else:  # unknown
        long_term_action = "insufficient_data"
        rationale.append("long_term_fit=unknown")

    # === SIZING GUIDANCE ===
    sizing_guidance: dict[str, Any] = {}
    min_pct = position_sizing.get("min_pct")
    max_pct = position_sizing.get("max_pct")
    if min_pct is not None and max_pct is not None:
        sizing_guidance["min_pct"] = min_pct
        sizing_guidance["max_pct"] = max_pct
        sizing_guidance["note"] = f"{min_pct:.1f}%-{max_pct:.1f}% of portfolio"

    # === CONDITIONS TO UPGRADE ===
    # What would need to change for a better assessment?
    conditions_to_upgrade: list[str] = []

    # Risk-based conditions (aligned with risk_regime boundaries)
    # extreme: >60%, high: 40-60%, medium: 25-40%, low: <25%
    if risk_label == "extreme":
        conditions_to_upgrade.append(
            f"risk_regime <= high (volatility < {VOLATILITY_REGIME_THRESHOLDS['high'] * 100:.0f}%)"
        )
    elif risk_label == "high":
        conditions_to_upgrade.append(
            f"risk_regime <= medium (volatility < {VOLATILITY_REGIME_THRESHOLDS['medium'] * 100:.0f}%)"
        )

    # Profitability conditions for unprofitable companies
    if is_unprofitable or business_quality_status == "evaluated_unprofitable":
        conditions_to_upgrade.append("profitMargins > 0 for 2 consecutive quarters")
        conditions_to_upgrade.append("freeCashflow > 0 for 2 consecutive quarters")

    # Valuation conditions
    if valuation_gate == "headwind":
        if valuation_basis == "ev_to_sales":
            conditions_to_upgrade.append("EV/S < 6x (currently headwind)")
        elif valuation_basis == "ps_trailing":
            conditions_to_upgrade.append("P/S < 6x (currently headwind)")
        elif valuation_basis == "pe_trailing":
            conditions_to_upgrade.append("P/E < 25x (currently headwind)")
    elif valuation_gate == "unknown":
        conditions_to_upgrade.append("revenue data available for valuation")

    # Business quality conditions
    if business_quality == "poor":
        conditions_to_upgrade.append("net_margin > 0%")
        conditions_to_upgrade.append("debt_to_equity < 1.5x")
    elif business_quality == "weak":
        conditions_to_upgrade.append("net_margin > 5%")

    # Horizon gate conditions
    if "burn_metrics_missing" in long_term_gates:
        conditions_to_upgrade.append("cash runway data available")
    if "low_runway_confidence" in long_term_gates:
        conditions_to_upgrade.append("cash runway > 8 quarters")
    if "severe_revenue_decline" in long_term_gates:
        conditions_to_upgrade.append("revenue_yoy > -20%")

    # === CONDITIONS TO DOWNGRADE ===
    # What would make this worse?
    conditions_to_downgrade: list[str] = []

    if risk_label in ("low", "medium"):
        conditions_to_downgrade.append(
            f"volatility spike > {VOLATILITY_REGIME_THRESHOLDS['medium'] * 100:.0f}%"
        )
    if not is_unprofitable:
        conditions_to_downgrade.append("guidance cut or profit warning")
        conditions_to_downgrade.append("net_margin turns negative")
    if valuation_gate == "attractive":
        conditions_to_downgrade.append("multiple expansion without earnings growth")

    return {
        "mid_term": mid_term_action,
        "long_term": long_term_action,
        "sizing_guidance": sizing_guidance if sizing_guidance else None,
        "rationale": rationale,
        "valuation_gate": valuation_gate,
        "valuation_basis": valuation_basis,
        "is_unprofitable": is_unprofitable if is_unprofitable else None,
        "conditions_to_upgrade": conditions_to_upgrade or [],
        "conditions_to_downgrade": conditions_to_downgrade or [],
    }
