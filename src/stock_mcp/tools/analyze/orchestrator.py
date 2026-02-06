"""Analyze stock aggregator tool."""

import asyncio
from datetime import datetime
from time import perf_counter
from typing import Any

from stock_mcp.reference_data import build_sector_comparison
from stock_mcp.tools.analyze.action_zones import apply_dip_gates_to_action_zones, build_action_zones
from stock_mcp.tools.analyze.decision_context import (
    build_decision_context,
    build_relative_performance,
)
from stock_mcp.tools.analyze.dip_assessment import build_dip_assessment
from stock_mcp.tools.analyze.executive_summary import build_executive_summary, build_policy_action
from stock_mcp.tools.analyze.signals import (
    TIMEOUT_SECONDS,
    classify_risk_regime,
    generate_signals,
    get_rule_triggered,
)
from stock_mcp.tools.analyze.summaries import (
    build_analyst_summary_text,
    build_dip_summary_text,
    build_fundamentals_summary_text,
    build_governance_summary_text,
    build_news_summary_text,
    build_ownership_summary_text,
    build_risk_summary_text,
    build_short_interest_summary_text,
    build_technicals_summary_text,
    build_valuation_context_summary_text,
)
from stock_mcp.tools.analyze.verdict import build_verdict
from stock_mcp.tools.events import events_calendar
from stock_mcp.tools.fundamentals import fundamentals_snapshot
from stock_mcp.tools.news import stock_news
from stock_mcp.tools.options_signals import options_signals
from stock_mcp.tools.ownership import ownership_analysis
from stock_mcp.tools.risk_metrics import risk_metrics
from stock_mcp.tools.stock_summary import stock_summary
from stock_mcp.tools.technicals import technicals
from stock_mcp.utils.helpers import first_sentences, format_fcf_label, round_or_none
from stock_mcp.utils.normalize import build_watchlist_snapshot
from stock_mcp.utils.provenance import build_meta


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

    tool_specs = [
        ("stock_summary", stock_summary(normalized_symbol)),
        ("technicals", technicals(normalized_symbol)),
        ("fundamentals_snapshot", fundamentals_snapshot(normalized_symbol)),
        ("risk_metrics", risk_metrics(normalized_symbol)),
        ("events_calendar", events_calendar(normalized_symbol)),
        ("stock_news", stock_news(normalized_symbol)),
        ("ownership_analysis", ownership_analysis(normalized_symbol)),
        ("options_signals", options_signals(normalized_symbol)),
    ]

    async def run_with_timing(
        name: str, coro: Any
    ) -> tuple[str, Any | Exception, float]:
        tool_start = perf_counter()
        result: Any | Exception
        try:
            result = await asyncio.wait_for(coro, timeout=TIMEOUT_SECONDS)
        except TimeoutError:
            result = TimeoutError(f"exceeded {TIMEOUT_SECONDS}s")
        except Exception as e:
            result = e
        duration_ms = (perf_counter() - tool_start) * 1000
        return (name, result, duration_ms)

    results = await asyncio.gather(
        *[run_with_timing(name, coro) for name, coro in tool_specs]
    )

    total_duration = (perf_counter() - start_time) * 1000

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
            if isinstance(result, dict) and "data_provenance" in result:
                data_provenance.update(result["data_provenance"])

    summary_data = tool_results.get("stock_summary") or {}
    tech_data = tool_results.get("technicals") or {}
    fund_data = tool_results.get("fundamentals_snapshot") or {}
    risk_data = tool_results.get("risk_metrics") or {}
    events_data = tool_results.get("events_calendar") or {}
    news_data = tool_results.get("stock_news") or {}
    ownership_data = tool_results.get("ownership_analysis") or {}
    options_data = tool_results.get("options_signals") or {}

    # Summary section
    summary = {
        "name": summary_data.get("name"),
        "sector": summary_data.get("sector"),
        "industry": summary_data.get("industry"),
        "market_cap": summary_data.get("market_cap"),
        "currency": summary_data.get("currency"),
        "current_price": summary_data.get("current_price") or tech_data.get("current_price"),
    }

    company_profile = {
        "description": first_sentences(summary_data.get("description"), max_sentences=2),
        "employees": summary_data.get("employees"),
        "industry": summary_data.get("industry"),
        "website": summary_data.get("website"),
    }

    # Technicals summary
    ma = tech_data.get("moving_averages", {})
    rsi = tech_data.get("rsi", {})
    macd = tech_data.get("macd", {})
    returns = tech_data.get("returns", {})
    price_pos = tech_data.get("price_position", {})

    bollinger = tech_data.get("bollinger", {})
    obv = tech_data.get("obv", {})
    fibonacci = tech_data.get("fibonacci", {})

    technicals_summary = {
        "trend": {
            "above_sma50": get_rule_triggered(ma, "above_sma50"),
            "above_sma200": get_rule_triggered(ma, "above_sma200"),
            "golden_cross": get_rule_triggered(ma, "golden_cross"),
            "death_cross": get_rule_triggered(ma, "death_cross"),
        },
        "momentum": {
            "rsi": rsi.get("value"),
            "rsi_overbought": get_rule_triggered(rsi, "overbought"),
            "rsi_oversold": get_rule_triggered(rsi, "oversold"),
            "macd_bullish": get_rule_triggered(macd, "bullish_cross"),
        },
        "returns": {
            "return_1m": returns.get("return_1m"),
            "return_3m": returns.get("return_3m"),
            "return_1y": returns.get("return_1y"),
        },
        "position_in_52w_range": price_pos.get("position_in_range"),
        "bollinger": {
            "pct_b": bollinger.get("pct_b"),
            "bandwidth": bollinger.get("bandwidth"),
            "above_upper": get_rule_triggered(bollinger, "above_upper"),
            "below_lower": get_rule_triggered(bollinger, "below_lower"),
            "squeeze": get_rule_triggered(bollinger, "squeeze"),
        } if bollinger else None,
        "obv_trend": obv.get("trend"),
        "fibonacci_nearest_support": fibonacci.get("nearest_support"),
        "fibonacci_nearest_resistance": fibonacci.get("nearest_resistance"),
    }
    technicals_summary["summary"] = build_technicals_summary_text(technicals_summary)

    # Fundamentals summary
    val = fund_data.get("valuation", {})
    growth = fund_data.get("growth", {})
    profit = fund_data.get("profitability", {})
    health = fund_data.get("financial_health", {})
    cf = fund_data.get("cash_flow", {})

    # Cash runway (prefer cash + ST investments for liquidity)
    liquidity = health.get("cash_and_st_investments") or health.get("total_cash")
    fcf_ttm = cf.get("free_cash_flow_ttm")
    operating_cf_ttm = cf.get("operating_cf_ttm")
    market_cap = summary_data.get("market_cap")

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

    # Conservative runway: min of available measures
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

    fundamentals_fetched = tool_results.get("fundamentals_snapshot") is not None

    pe_trailing = val.get("pe_trailing")
    trailing_eps = val.get("trailing_eps")
    net_margin = profit.get("net_margin")
    is_unprofitable_company = fundamentals_fetched and (
        (net_margin is not None and net_margin < 0)
        or (trailing_eps is not None and trailing_eps <= 0)
    )

    valuation_summary: dict[str, Any] = {
        "pe_trailing": pe_trailing,
        "peg_ratio": val.get("peg_ratio"),
    }

    ps_trailing = val.get("ps_trailing")
    ps_source = val.get("ps_source")
    ps_explanation = val.get("ps_explanation")
    ev_to_ebitda = val.get("ev_to_ebitda")
    ev_to_sales = val.get("ev_to_sales")

    if fundamentals_fetched and (is_unprofitable_company or pe_trailing is None):
        valuation_summary["ps_trailing"] = ps_trailing
        valuation_summary["ps_source"] = ps_source
        if ps_explanation:
            valuation_summary["ps_explanation"] = ps_explanation
        valuation_summary["ev_to_ebitda"] = ev_to_ebitda
        valuation_summary["ev_to_sales"] = ev_to_sales
        if is_unprofitable_company:
            valuation_summary["valuation_note"] = "pe_not_meaningful"

    valuation_warnings: list[str] = []
    if is_unprofitable_company and ps_trailing is not None and ps_trailing > 10:
        valuation_warnings.append("unprofitable_high_ps")

    if valuation_warnings:
        valuation_summary["warnings"] = valuation_warnings

    burn_metrics: dict[str, Any] | None = None
    if is_unprofitable_company:
        burn_warnings: list[str] = []
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

        dilution_analysis: dict[str, Any] | None = None
        target_runway_quarters = 8  # 2 years
        quarterly_burn_for_dilution = quarterly_fcf_burn or quarterly_ocf_burn

        if (
            cash_runway_quarters is not None
            and cash_runway_quarters < target_runway_quarters
            and market_cap is not None
            and market_cap > 0
            and quarterly_burn_for_dilution is not None
            and quarterly_burn_for_dilution > 0
        ):
            quarters_short = target_runway_quarters - cash_runway_quarters
            raise_needed = quarters_short * quarterly_burn_for_dilution
            dilution_decimal = raise_needed / market_cap

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
                "raise_needed_for_2y_runway": round_or_none(raise_needed, 0),
                "dilution_if_raised_today": round(dilution_decimal, 4),
                "dilution_risk_level": dilution_risk_level,
                "current_market_cap": market_cap,
            }

        if runway_basis == "fcf_only":
            burn_warnings.append("using_fcf_only")
        elif runway_basis == "ocf_only":
            burn_warnings.append("using_ocf_only")

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
            "liquidity": round_or_none(liquidity, 0),
            "cash_runway_quarters": cash_runway_quarters,
            "runway_basis": runway_basis,
            "runway_confidence": runway_confidence,
            "quarterly_fcf_burn": round_or_none(quarterly_fcf_burn, 0),
            "quarterly_ocf_burn": round_or_none(quarterly_ocf_burn, 0),
            "dilution_analysis": dilution_analysis,
            "warnings": burn_warnings or [],
        }

        if cash_runway_quarters is not None and cash_runway_quarters < 8:
            if "dilution_risk_elevated" not in valuation_warnings:
                valuation_warnings.append("dilution_risk_elevated")
                valuation_summary["warnings"] = valuation_warnings

    gross_margin = profit.get("gross_margin")
    fcf_value = cf.get("free_cash_flow_ttm")
    fcf_period = cf.get("free_cash_flow_period")
    fcf_period_end = cf.get("free_cash_flow_period_end")
    fcf_currency = cf.get("currency")
    fcf_source = cf.get("free_cash_flow_source")
    fcf_label = format_fcf_label(fcf_value, fcf_period, fcf_currency, fcf_period_end)

    fundamentals_summary = {
        "valuation": valuation_summary,
        "growth": {
            "revenue_yoy": growth.get("revenue_yoy"),
            "eps_yoy": growth.get("eps_yoy"),
        },
        "profitability": {
            "gross_margin": gross_margin,
            "net_margin": net_margin,
            "fcf_positive": get_rule_triggered(cf, "positive_fcf"),
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
            "net_cash_positive": get_rule_triggered(health, "net_cash_positive"),
        },
        "burn_metrics": burn_metrics,
        "valuation_history": fund_data.get("valuation_history"),
        "fundamental_trends": fund_data.get("fundamental_trends"),
        "analyst_estimates": fund_data.get("analyst_estimates"),
        "dividend_analysis": fund_data.get("dividend_analysis"),
    }
    fundamentals_summary["summary"] = (
        build_fundamentals_summary_text(fundamentals_summary)
        if fundamentals_fetched
        else "Fundamentals unavailable (data fetch failed)."
    )

    analyst_coverage = fund_data.get("analyst_coverage") or {}
    short_interest = fund_data.get("short_interest") or {}
    ownership = fund_data.get("ownership") or {}
    governance = fund_data.get("governance") or {}
    quality = fund_data.get("quality") or {}
    valuation_context = fund_data.get("valuation_context") or {}

    # Risk summary
    vol = risk_data.get("volatility", {})
    beta = risk_data.get("beta", {})
    dd = risk_data.get("drawdown", {})
    atr = risk_data.get("atr", {})

    risk_regime = classify_risk_regime(
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
    risk_summary["summary"] = build_risk_summary_text(risk_summary)

    # Events summary
    earnings = events_data.get("earnings", {})
    next_earnings_date = earnings.get("next_date")
    next_earnings_source = earnings.get("next_date_source")
    next_earnings_status = earnings.get("next_date_status", "unavailable")
    next_earnings_reason = earnings.get("next_date_status_reason")

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
        "days_to_earnings": earnings.get("days_until"),
    }

    # News summary
    articles = news_data.get("articles", [])
    recent_earnings = news_data.get("recent_earnings")
    sentiment_data = news_data.get("sentiment", {})

    headlines = [article.get("title") for article in articles[:5] if article.get("title")]

    earnings_highlight: dict[str, Any] | None = None
    if recent_earnings:
        earnings_highlight = {
            "date": recent_earnings.get("date"),
            "beat_miss": recent_earnings.get("beat_miss"),
            "surprise_pct": recent_earnings.get("surprise_pct"),
        }

    sentiment_summary: dict[str, Any] | None = None
    if sentiment_data:
        sentiment_summary = {
            "overall": sentiment_data.get("overall"),
            "confidence": sentiment_data.get("confidence"),
            "sample_size": len(articles),
            "method": sentiment_data.get("method"),
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
    news_summary["summary"] = build_news_summary_text(news_summary)

    # Signals
    signals = generate_signals(
        technicals=tech_data,
        fundamentals=fund_data,
        risk=risk_data,
    )

    # Enrich signals from ownership, options, and bollinger
    insider_sentiment = ownership_data.get("insider_activity", {}).get("sentiment")
    if insider_sentiment == "buying":
        signals["bullish"].append("insider_buying")
    elif insider_sentiment == "selling":
        signals["bearish"].append("insider_selling")

    pc_ratio = options_data.get("put_call_ratio", {}).get("volume_based")
    if pc_ratio is not None and pc_ratio > 1.5:
        signals["bearish"].append("high_put_call_ratio")

    atm_iv = options_data.get("implied_volatility", {}).get("atm_avg_iv")
    if atm_iv is not None and atm_iv > 0.50:
        signals["neutral"].append("elevated_implied_volatility")

    if get_rule_triggered(bollinger, "below_lower"):
        signals["bullish"].append("below_bollinger_lower")
    elif get_rule_triggered(bollinger, "above_upper"):
        signals["bearish"].append("above_bollinger_upper")

    if get_rule_triggered(bollinger, "squeeze"):
        signals["neutral"].append("bollinger_squeeze")

    sector_comparison = build_sector_comparison(
        sector=summary_data.get("sector"),
        pe=pe_trailing,
        ps=ps_trailing,
        net_margin=net_margin,
        revenue_growth=growth.get("revenue_yoy"),
        debt_to_equity=health.get("debt_to_equity"),
    )

    coverage = {
        "price": tool_results.get("technicals") is not None,
        "fundamentals": tool_results.get("fundamentals_snapshot") is not None,
        "risk": tool_results.get("risk_metrics") is not None,
        "news": tool_results.get("stock_news") is not None,
        "events": tool_results.get("events_calendar") is not None,
        "ownership": tool_results.get("ownership_analysis") is not None,
        "options": tool_results.get("options_signals") is not None,
    }

    verdict = build_verdict(
        signals=signals,
        fundamentals_data=fund_data,
        risk_data=risk_data,
        technicals_data=tech_data,
        coverage=coverage,
        risk_regime=risk_regime,
        fundamentals_summary=fundamentals_summary,
    )

    action_zones = build_action_zones(
        current_price=summary.get("current_price"),
        tech_data=tech_data,
        risk_data=risk_data,
        fund_data=fund_data,
        risk_regime=risk_regime,
        signals=signals,
    )

    relative_performance = build_relative_performance(
        tech_data=tech_data,
        risk_data=risk_data,
    )

    market_context = risk_data.get("market_context", {})

    dip_assessment = build_dip_assessment(
        tech_data=tech_data,
        risk_data=risk_data,
        fund_data=fund_data,
        market_context=market_context,
        signals=signals,
    )
    if dip_assessment:
        dip_assessment["summary"] = build_dip_summary_text(dip_assessment)

    action_zones = apply_dip_gates_to_action_zones(
        action_zones=action_zones,
        dip_assessment=dip_assessment,
        risk_regime=risk_regime,
    )

    section_summaries = {
        "technicals": technicals_summary.get("summary"),
        "fundamentals": fundamentals_summary.get("summary"),
        "risk": risk_summary.get("summary"),
        "ownership": build_ownership_summary_text(ownership),
        "short_interest": build_short_interest_summary_text(short_interest),
        "analyst_view": build_analyst_summary_text(
            analyst_coverage,
            summary.get("current_price"),
            summary.get("currency"),
        ),
        "governance": build_governance_summary_text(governance),
        "valuation_context": build_valuation_context_summary_text(valuation_context),
        "dip_quality": dip_assessment.get("summary") if dip_assessment else None,
        "news": news_summary.get("summary"),
    }

    # Data quality
    val_summary = fundamentals_summary.get("valuation") or {}
    valuation_metric_available = (
        val_summary.get("pe_trailing") is not None
        or val_summary.get("ps_trailing") is not None
    )
    available_fields = sum([
        summary.get("current_price") is not None,
        technicals_summary["trend"]["above_sma50"] is not None,
        technicals_summary["momentum"]["rsi"] is not None,
        valuation_metric_available,
        risk_summary["beta"] is not None,
    ])
    completeness = available_fields / 5

    missing_critical: list[str] = []
    if summary.get("current_price") is None:
        missing_critical.append("current_price")
    if risk_summary["beta"] is None:
        missing_critical.append("beta")

    warnings: list[str] = []
    if earnings.get("days_until") and earnings.get("days_until") <= 7:
        warnings.append("earnings_within_7_days")

    # Fundamentals status: missing | not_meaningful | available
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

    data_gaps: list[str] = []

    burn_metrics_data = fundamentals_summary.get("burn_metrics") or {}
    burn_metrics_status: str | None = burn_metrics_data.get("status")

    if is_unprofitable_company and burn_metrics_status == "unavailable":
        data_gaps.append("burn_metrics_unavailable")
        burn_reason = burn_metrics_data.get("status_reason")
        if burn_reason:
            data_gaps.append(f"burn_metrics_reason:{burn_reason}")

    if fundamentals_status == "not_meaningful":
        data_gaps.append("pe_not_meaningful")

    # Staleness tracking
    component_freshness: dict[str, dict[str, Any]] = {}
    staleness_warnings: list[str] = []
    now = datetime.utcnow()

    def _extract_freshness(prov_key: str, component_name: str, stale_threshold_hours: int) -> None:
        prov = data_provenance.get(prov_key, {})
        as_of_str = prov.get("as_of")
        if as_of_str:
            try:
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

    # Staleness thresholds: price/risk 48h, fundamentals/events 168h, news 336h
    _extract_freshness("price", "price", stale_threshold_hours=48)
    _extract_freshness("fundamentals", "fundamentals", stale_threshold_hours=168)
    _extract_freshness("price", "risk", stale_threshold_hours=48)
    _extract_freshness("news", "news", stale_threshold_hours=336)
    _extract_freshness("events", "events", stale_threshold_hours=168)

    if articles:
        try:
            latest_article_date = articles[0].get("date")
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

    decision_context = build_decision_context(
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

    policy_action = build_policy_action(
        verdict=verdict,
        action_zones=action_zones,
        decomposed=verdict.get("decomposed"),
        risk_regime=risk_regime,
        dip_assessment=dip_assessment,
    )

    executive_summary = build_executive_summary(
        summary=summary,
        technicals_summary=technicals_summary,
        fundamentals_summary=fundamentals_summary,
        risk_summary=risk_summary,
        signals=signals,
        verdict=verdict,
        action_zones=action_zones,
        policy_action=policy_action,
        news_summary=news_summary,
    )

    # Top-level error signaling for invalid/unanalyzable symbols.
    # Keep partial payload for diagnostics, but mark as error for callers.
    critical_tools = {"stock_summary", "technicals", "fundamentals_snapshot", "risk_metrics"}
    critical_failures = [f for f in tool_failures if f.get("tool") in critical_tools]
    critical_failed_tools = {f.get("tool") for f in critical_failures}

    has_core_price = summary.get("current_price") is not None
    missing_core_market_data = (
        not has_core_price
        and "stock_summary" in critical_failed_tools
        and "technicals" in critical_failed_tools
    )

    invalid_hints = (
        "invalid symbol",
        "quote not found",
        "possibly delisted",
        "no data returned",
        "no timezone found",
        "incomplete yfinance info",
    )
    invalid_like_failures = bool(critical_failures) and all(
        failure.get("error") == "invalid_symbol"
        or any(hint in str(failure.get("message", "")).lower() for hint in invalid_hints)
        for failure in critical_failures
    )

    top_level_error = missing_core_market_data
    top_level_error_type: str | None = None
    top_level_error_message: str | None = None
    if top_level_error:
        if invalid_like_failures:
            top_level_error_type = "invalid_symbol"
            top_level_error_message = (
                f"Unable to analyze {normalized_symbol}: symbol appears invalid or unavailable."
            )
        else:
            top_level_error_type = "data_unavailable"
            top_level_error_message = (
                f"Unable to analyze {normalized_symbol}: missing core market data from upstream tools."
            )

    raw_result = {
        "meta": build_meta("analyze_stock", total_duration),
        "data_provenance": data_provenance,
        "symbol": normalized_symbol,
        "executive_summary": executive_summary,
        "summary": summary,
        "company_profile": company_profile,
        "section_summaries": section_summaries,
        "analyst_coverage": analyst_coverage,
        "short_interest": short_interest,
        "ownership": ownership,
        "governance": governance,
        "quality": quality,
        "valuation_context": valuation_context,
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
        "sector_comparison": sector_comparison,
        "ownership_flow": ownership_data or None,
        "options_signals": options_data or None,
        "data_quality": data_quality,
    }

    if top_level_error:
        raw_result["error"] = True
        raw_result["error_type"] = top_level_error_type
        raw_result["message"] = top_level_error_message

    raw_result["watchlist_snapshot"] = build_watchlist_snapshot(raw_result)

    return raw_result
