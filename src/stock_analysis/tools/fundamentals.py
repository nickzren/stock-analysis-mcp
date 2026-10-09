"""Fundamentals snapshot tool."""

import asyncio
import logging
import operator
import statistics
from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from stock_analysis.data.provider import read_property
from stock_analysis.data.yfinance_client import fetch_info, fetch_ticker
from stock_analysis.utils.helpers import (
    current_price_from_info,
    safe_float,
    safe_int,
    safe_round,
    safe_str,
)
from stock_analysis.utils.provenance import (
    FetchError,
    build_meta,
    build_provenance,
    fetch_or_error,
    utcnow_isoformat_z,
)
from stock_analysis.utils.validators import check_rule
from stock_analysis.utils.valuation_basis import (
    PROVIDER,
    UNVERIFIED,
    UNVERIFIED_CURRENCY,
    VERIFIED,
    currency_basis,
    dividend_yields,
    metric_entry,
    share_units_basis,
    usable_metric,
)

logger = logging.getLogger(__name__)

# Cash flow row labels (normalized for matching)
_FCF_LABELS: tuple[str, ...] = ("freecashflow",)
_OCF_LABELS: tuple[str, ...] = ("operatingcashflow",)
_CAPEX_LABELS: tuple[str, ...] = ("capitalexpenditure", "capitalexpenditures")


def _build_valuation(info: dict[str, Any]) -> dict[str, Any]:
    """Build the valuation section from yfinance info.

    WP4: each ratio is kept only on a verified basis (``metric_status``): ratios
    mixing price and statement data need one currency; P/E needs a positive EPS
    of the same horizon; EV-based ratios need EV > 0 to be scored.
    """
    price = current_price_from_info(info)
    currency = currency_basis(info.get("currency"), info.get("financialCurrency"))
    status: dict[str, dict[str, Any]] = {}

    pe_trailing = _pe(info, "trailingPE", "trailingEps", currency, status, "pe_trailing")
    pe_forward = _pe(info, "forwardPE", "forwardEps", currency, status, "pe_forward")

    # P/S: try direct field first, then compute from market_cap / revenue
    ps_trailing = safe_float(info.get("priceToSalesTrailing12Months"))
    ps_source: str | None = None
    if ps_trailing is not None:
        ps_source = "direct"
    else:
        market_cap_val = safe_float(info.get("marketCap"))
        revenue_ttm = safe_float(info.get("totalRevenue"))
        if market_cap_val is not None and revenue_ttm is not None and revenue_ttm > 0:
            ps_trailing = market_cap_val / revenue_ttm
            ps_source = "computed"
    if ps_trailing is not None and currency != VERIFIED:
        ps_trailing = None
        status["ps_trailing"] = metric_entry(currency, False, currency)
    elif ps_trailing is not None:
        status["ps_trailing"] = metric_entry(VERIFIED, ps_trailing > 0, None if ps_trailing > 0 else "ps_nonpositive")

    ps_explanation: str | None = None
    if ps_source == "computed":
        ps_explanation = "P/S computed from market_cap / revenue_ttm (priceToSalesTrailing12Months unavailable)"

    pb_ratio = safe_float(info.get("priceToBook"))
    if pb_ratio is not None:
        pb_basis = currency if currency != VERIFIED else share_units_basis(info, price)
        if pb_basis != VERIFIED:
            pb_ratio = None
            status["pb_ratio"] = metric_entry(pb_basis, False, pb_basis)
        else:
            status["pb_ratio"] = metric_entry(VERIFIED, pb_ratio > 0, None if pb_ratio > 0 else "pb_nonpositive")

    # Provider PEG: kept whatever the growth figures say; only PEG > 0 is scored.
    peg_ratio = safe_float(info.get("pegRatio"))
    if peg_ratio is not None:
        status["peg_ratio"] = metric_entry(PROVIDER, peg_ratio > 0, None if peg_ratio > 0 else "peg_nonpositive")

    # EV can be negative (cash > market cap + debt); EV ratios are scored only when EV > 0.
    enterprise_value = safe_float(info.get("enterpriseValue"))
    revenue_ttm = safe_float(info.get("totalRevenue"))
    ev_to_sales: float | None = None
    ev_to_sales_source: str | None = None
    if enterprise_value is not None and revenue_ttm is not None and revenue_ttm > 0:
        if currency != VERIFIED:
            status["ev_to_sales"] = metric_entry(currency, False, currency)
        else:
            ev_to_sales = enterprise_value / revenue_ttm
            ev_to_sales_source = "computed"
            status["ev_to_sales"] = metric_entry(
                VERIFIED, enterprise_value > 0, None if enterprise_value > 0 else "ev_nonpositive"
            )

    ev_to_ebitda = safe_float(info.get("enterpriseToEbitda"))
    ebitda = safe_float(info.get("ebitda"))
    if ev_to_ebitda is not None:
        if ebitda is not None and ebitda <= 0:
            ev_to_ebitda = None
            status["ev_to_ebitda"] = metric_entry(currency, False, "undefined_ebitda_nonpositive")
        elif currency != VERIFIED:
            ev_to_ebitda = None
            status["ev_to_ebitda"] = metric_entry(currency, False, currency)
        else:
            applicable = ev_to_ebitda > 0 and (enterprise_value is None or enterprise_value > 0)
            status["ev_to_ebitda"] = metric_entry(VERIFIED, applicable, None if applicable else "ev_nonpositive")

    return {
        "pe_trailing": pe_trailing,
        "pe_forward": pe_forward,
        "trailing_eps": safe_float(info.get("trailingEps")),
        "forward_eps": safe_float(info.get("forwardEps")),
        "ps_trailing": safe_round(ps_trailing, 2),
        "ps_source": ps_source,
        "ps_explanation": ps_explanation,
        "pb_ratio": pb_ratio,
        "peg_ratio": peg_ratio,
        "enterprise_value": enterprise_value,
        "ev_to_ebitda": ev_to_ebitda,
        "ev_to_sales": safe_round(ev_to_sales, 2),
        "ev_to_sales_source": ev_to_sales_source,
        "metric_status": status,
    }


def _per_share_basis(info: dict[str, Any], price: float | None) -> str:
    """Basis for a figure mixing statement totals with per-share quote data: one
    currency, then consistent share units."""
    basis = currency_basis(info.get("currency"), info.get("financialCurrency"))
    return basis if basis != VERIFIED else share_units_basis(info, price)


def _pe(
    info: dict[str, Any],
    pe_key: str,
    eps_key: str,
    currency: str,
    status: dict[str, dict[str, Any]],
    name: str,
) -> float | None:
    """A provider P/E judged on its own EPS horizon: undefined when that EPS <= 0,
    unverified when the EPS is unknown or its currency can't be shown to match."""
    pe = safe_float(info.get(pe_key))
    eps = safe_float(info.get(eps_key))
    if pe is None and eps is None:
        return None
    if eps is not None and eps <= 0:
        status[name] = metric_entry(currency, False, "undefined_eps_nonpositive")
        return None
    if pe is None:
        return None  # absent, not unverified
    if currency != VERIFIED:
        status[name] = metric_entry(currency, False, currency)
        return None
    if eps is None or pe <= 0:
        status[name] = metric_entry(UNVERIFIED, False, "eps_basis_unknown")
        return None
    status[name] = metric_entry(VERIFIED, True)
    return pe


def _build_growth(info: dict[str, Any]) -> dict[str, Any]:
    """Build the growth section from yfinance info."""
    revenue_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth"))
    return {
        "revenue_yoy": revenue_growth,
        "revenue_3y_cagr": None,  # Not available from yfinance directly
        "eps_yoy": earnings_growth,
        "eps_3y_cagr": None,  # Not available from yfinance directly
        "rules": {
            "positive_revenue_growth": {
                "triggered": check_rule(revenue_growth, 0, operator.gt),
                "threshold": 0,
            },
            "high_growth": {
                "triggered": check_rule(revenue_growth, 0.20, operator.gt),
                "threshold": 0.20,
            },
        },
    }


def _build_profitability(info: dict[str, Any]) -> dict[str, Any]:
    """Build the profitability section from yfinance info."""
    net_margin = safe_float(info.get("profitMargins"))
    return {
        "gross_margin": safe_float(info.get("grossMargins")),
        "operating_margin": safe_float(info.get("operatingMargins")),
        "net_margin": net_margin,
        "roe": safe_float(info.get("returnOnEquity")),
        "roa": safe_float(info.get("returnOnAssets")),
        "rules": {
            "profitable": {
                "triggered": check_rule(net_margin, 0, operator.gt),
                "threshold": "net_margin > 0",
            },
            "high_margin": {
                "triggered": check_rule(net_margin, 0.15, operator.gt),
                "threshold": 0.15,
            },
        },
    }


def _build_financial_health(info: dict[str, Any]) -> dict[str, Any]:
    """Build the financial_health section from yfinance info."""
    total_cash = safe_float(info.get("totalCash"))
    # Cash + short-term investments (more accurate liquidity for burn calculations)
    cash_and_st_investments = safe_float(info.get("cashAndShortTermInvestments"))
    if cash_and_st_investments is None:
        cash_and_st_investments = total_cash  # Fallback to total cash

    total_debt = safe_float(info.get("totalDebt"))
    net_cash = (
        total_cash - total_debt
        if total_cash is not None and total_debt is not None
        else None
    )
    current_ratio = safe_float(info.get("currentRatio"))
    # Yahoo reports D/E as a percent, always (R4.1).
    debt_to_equity = safe_float(info.get("debtToEquity"))
    if debt_to_equity is not None:
        debt_to_equity = debt_to_equity / 100
    price_to_book = safe_float(info.get("priceToBook"))
    if (debt_to_equity is not None and debt_to_equity < 0) or (
        debt_to_equity is None and price_to_book is not None and price_to_book < 0
    ):
        leverage_status = "negative_equity"  # never low debt (R4.2)
    elif debt_to_equity is None:
        leverage_status = "unavailable"
    else:
        leverage_status = "reported"

    return {
        "total_cash": total_cash,
        "cash_and_st_investments": cash_and_st_investments,
        "total_debt": total_debt,
        "net_cash": net_cash,
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "leverage_status": leverage_status,
        "interest_coverage": None,  # Not directly available
        "rules": {
            "net_cash_positive": {
                "triggered": check_rule(net_cash, 0, operator.gt),
                "threshold": 0,
            },
            "low_debt": {
                "triggered": (
                    False if leverage_status == "negative_equity" else check_rule(debt_to_equity, 0.5, operator.lt)
                ),
                "threshold": 0.5,
            },
            "adequate_liquidity": {
                "triggered": check_rule(current_ratio, 1.0, operator.gt),
                "threshold": 1.0,
            },
        },
    }


def _build_analyst_coverage(info: dict[str, Any], current_price: float | None) -> dict[str, Any]:
    """Build the analyst_coverage section from yfinance info."""
    target_low = safe_float(info.get("targetLowPrice"))
    target_mean = safe_float(info.get("targetMeanPrice"))
    target_high = safe_float(info.get("targetHighPrice"))
    target_median = safe_float(info.get("targetMedianPrice"))

    upside_to_mean_target = (
        (target_mean - current_price) / current_price
        if target_mean is not None and current_price is not None and current_price > 0
        else None
    )
    upside_to_median_target = (
        (target_median - current_price) / current_price
        if target_median is not None and current_price is not None and current_price > 0
        else None
    )

    return {
        "rating": safe_str(info.get("recommendationKey")),
        "rating_score": safe_float(info.get("recommendationMean")),
        "num_analysts": safe_int(info.get("numberOfAnalystOpinions")),
        "price_target_low": target_low,
        "price_target_mean": target_mean,
        "price_target_high": target_high,
        "price_target_median": target_median,
        "upside_to_mean_target": safe_round(upside_to_mean_target, 4),
        "upside_to_median_target": safe_round(upside_to_median_target, 4),
    }


def _build_short_interest(info: dict[str, Any]) -> dict[str, Any]:
    """Build the short_interest section from yfinance info."""
    shares_short = safe_int(info.get("sharesShort"))
    shares_short_prior = safe_int(info.get("sharesShortPriorMonth"))
    short_change_mom = (
        (shares_short - shares_short_prior) / shares_short_prior
        if shares_short is not None and shares_short_prior
        else None
    )
    return {
        "shares_short": shares_short,
        "short_pct_of_float": safe_float(info.get("shortPercentOfFloat")),
        "days_to_cover": safe_float(info.get("shortRatio")),
        "short_change_mom": safe_round(short_change_mom, 4),
        "as_of_date": _format_date_string(info.get("dateShortInterest")),
    }


def _build_ownership(info: dict[str, Any]) -> dict[str, Any]:
    """Build the ownership section from yfinance info."""
    return {
        "insider_pct": safe_float(info.get("heldPercentInsiders")),
        "institutional_pct": safe_float(info.get("heldPercentInstitutions")),
        "float_shares": safe_int(info.get("floatShares")),
    }


def _build_governance(info: dict[str, Any]) -> dict[str, Any]:
    """Build the governance section from yfinance info."""
    return {
        "audit_risk": safe_int(info.get("auditRisk")),
        "board_risk": safe_int(info.get("boardRisk")),
        "compensation_risk": safe_int(info.get("compensationRisk")),
        "shareholder_rights_risk": safe_int(info.get("shareHolderRightsRisk")),
        "overall_risk": safe_int(info.get("overallRisk")),
    }


def _build_quality(info: dict[str, Any]) -> dict[str, Any]:
    """Build the quality section from yfinance info."""
    return {
        "roic": safe_float(info.get("returnOnInvestedCapital") or info.get("returnOnCapitalEmployed")),
        "gross_profit": safe_float(info.get("grossProfits")),
        "ebitda": safe_float(info.get("ebitda")),
        "ebitda_margin": safe_float(info.get("ebitdaMargins")),
        "revenue_per_share": safe_float(info.get("revenuePerShare")),
        "quick_ratio": safe_float(info.get("quickRatio")),
    }


async def fundamentals_snapshot(symbol: str) -> dict[str, Any]:
    """
    Get fundamental financial data for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with valuation, growth, profitability, financial health, cash flow metrics
    """
    start_time = perf_counter()

    try:
        info = await fetch_or_error(fetch_info(symbol), symbol)
    except FetchError as fe:
        return fe.response

    normalized_symbol = symbol.upper().strip()
    current_price = current_price_from_info(info)

    # Fiscal period label (used for cash flow period metadata)
    fiscal_year_end = info.get("lastFiscalYearEnd")
    fiscal_period = None
    if fiscal_year_end:
        try:
            fiscal_date = datetime.fromtimestamp(fiscal_year_end)
            fiscal_period = f"FY {fiscal_date.year}"
        except (ValueError, TypeError, OSError):
            pass

    valuation = _build_valuation(info)
    growth = _build_growth(info)
    profitability = _build_profitability(info)
    financial_health = _build_financial_health(info)

    # Cash Flow
    operating_cf = safe_float(info.get("operatingCashflow"))
    free_cash_flow = safe_float(info.get("freeCashflow"))
    financial_currency = info.get("financialCurrency")

    try:
        ticker_obj = await fetch_ticker(symbol)
    except Exception as e:
        logger.debug("fundamentals ticker fetch failed: %s", e)
        ticker_obj = None

    statements = await _read_statements(ticker_obj) if ticker_obj is not None else {}

    operating_cf, free_cash_flow, fcf_period, fcf_period_end, fcf_source = (
        _resolve_cashflow_ttm(
            statements.get("cashflow_quarterly"),
            statements.get("cashflow_yearly"),
            operating_cf,
            free_cash_flow,
            fiscal_period,
        )
    )
    market_cap = safe_float(info.get("marketCap"))
    revenue = safe_float(info.get("totalRevenue"))

    fcf_margin = (
        free_cash_flow / revenue
        if free_cash_flow is not None and revenue is not None and revenue > 0
        else None
    )

    cash_flow = {
        "operating_cf_ttm": operating_cf,
        "free_cash_flow_ttm": free_cash_flow,
        "free_cash_flow_period": fcf_period,
        "free_cash_flow_period_end": fcf_period_end,
        "free_cash_flow_source": fcf_source,
        "currency": financial_currency,
        "fcf_margin": safe_round(fcf_margin, 4),
        "rules": {
            "positive_fcf": {
                "triggered": check_rule(free_cash_flow, 0, operator.gt),
                "threshold": 0,
            },
        },
    }

    yield_metrics = _build_yield_metrics(info, valuation, free_cash_flow, market_cap, current_price)

    analyst_coverage = _build_analyst_coverage(info, current_price)
    short_interest = _build_short_interest(info)
    ownership = _build_ownership(info)
    governance = _build_governance(info)
    quality = _build_quality(info)

    pe_current = usable_metric(valuation, "pe_trailing")
    ps_current = usable_metric(valuation, "ps_trailing")
    valuation_context_status = "partial" if pe_current is not None or ps_current is not None else "unavailable"

    valuation_context = {
        "pe_current": pe_current,
        "pe_5y_avg": None,
        "pe_percentile_5y": None,
        "ps_current": ps_current,
        "ps_5y_avg": None,
        "history_points": 0,
        "history_span_years": None,
        "status": valuation_context_status,
        "status_reason": "historical_pe_ps_unavailable",
    }

    warnings = []
    if info.get("trailingPE") and not info.get("forwardPE"):
        warnings.append("using_trailing_data")

    # Enrichment: valuation history, trends, estimates, dividend analysis
    income_stmt = statements.get("quarterly_income_stmt")
    valuation_history = _compute_valuation_history(
        income_stmt, statements.get("price_history_5y"), info, valuation
    )
    fundamental_trends = _compute_fundamental_trends(income_stmt)
    analyst_estimates = (
        _build_earnings_estimates(statements.get("earnings_estimate"), statements.get("revenue_estimate"), info)
        if ticker_obj is not None
        else None
    )
    dividend_analysis = _analyze_dividend_history(statements.get("dividends"), info)

    # Wire valuation history stats into valuation_context: *_5y only when the
    # history is sufficient AND actually spans five years (R4.9).
    if valuation_history:
        valuation_context["history_points"] = valuation_history.get("history_points", 0)
        valuation_context["history_span_years"] = valuation_history.get("history_span_years")
        pe_stats = valuation_history.get("pe_stats")
        ps_stats = valuation_history.get("ps_stats")
        span = valuation_history.get("history_span_years") or 0
        if (pe_stats or ps_stats) and span >= 5:
            if pe_stats:
                valuation_context["pe_5y_avg"] = pe_stats.get("mean")
                valuation_context["pe_percentile_5y"] = pe_stats.get("current_percentile")
            if ps_stats:
                valuation_context["ps_5y_avg"] = ps_stats.get("mean")
            valuation_context["status"] = "available"
            valuation_context["status_reason"] = None
        elif pe_stats or ps_stats:
            valuation_context["status_reason"] = "history_span_under_5y"
        elif valuation_history.get("status") != "unavailable":
            valuation_context["status_reason"] = valuation_history.get("status")

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("fundamentals_snapshot", duration_ms),
        "data_provenance": {
            "fundamentals": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
                fiscal_period=fiscal_period,
                warnings=warnings,
            ),
        },
        "symbol": normalized_symbol,
        "valuation": valuation,
        "growth": growth,
        "profitability": profitability,
        "financial_health": financial_health,
        "cash_flow": cash_flow,
        "yield_metrics": yield_metrics,
        "analyst_coverage": analyst_coverage,
        "short_interest": short_interest,
        "ownership": ownership,
        "governance": governance,
        "quality": quality,
        "valuation_context": valuation_context,
        "valuation_history": valuation_history,
        "fundamental_trends": fundamental_trends,
        "analyst_estimates": analyst_estimates,
        "dividend_analysis": dividend_analysis,
    }


async def _read_statements(ticker: Any) -> dict[str, Any]:
    """The Ticker property reads fundamentals needs, each as executor jobs under the
    symbol lock (WP3). A failed read maps to None, as before."""
    reads = {
        "cashflow_quarterly": lambda t: _read_cashflow(t, "quarterly"),
        "cashflow_yearly": lambda t: _read_cashflow(t, None),
        "quarterly_income_stmt": lambda t: t.quarterly_income_stmt,
        "price_history_5y": lambda t: t.history(period="5y", interval="1d", auto_adjust=True),
        "earnings_estimate": lambda t: t.earnings_estimate,
        "revenue_estimate": lambda t: t.revenue_estimate,
        "dividends": lambda t: t.dividends,
    }
    results = await asyncio.gather(*(read_property(ticker, name, read) for name, read in reads.items()))
    for read in results:
        if read.status == "failed":
            logger.debug("fundamentals %s read failed: %s", read.source, read.error_type)
    return {read.source: None if read.status == "failed" else read.payload for read in results}


def _resolve_cashflow_ttm(
    quarterly_cf: pd.DataFrame | None,
    yearly_cf: pd.DataFrame | None,
    operating_cf: float | None,
    free_cash_flow: float | None,
    fiscal_period: str | None,
) -> tuple[float | None, float | None, str | None, str | None, str | None]:
    """Resolve preferred cash flow TTM values from statements, falling back to info."""
    fcf_period: str | None = None
    fcf_period_end: str | None = None
    fcf_source: str | None = None

    if quarterly_cf is not None or yearly_cf is not None:
        try:
            quarterly_cols = (
                _sorted_cashflow_columns(quarterly_cf.columns)
                if quarterly_cf is not None
                else []
            )
            yearly_cols = (
                _sorted_cashflow_columns(yearly_cf.columns)
                if yearly_cf is not None
                else []
            )
            quarterly_period_end = _period_end_from_columns(quarterly_cols)
            yearly_period_end = _period_end_from_columns(yearly_cols)

            fcf_quarterly = _sum_recent_periods(
                _select_cashflow_series(quarterly_cf, _FCF_LABELS),
                quarterly_cols,
                periods=4,
            )
            ocf_quarterly = _sum_recent_periods(
                _select_cashflow_series(quarterly_cf, _OCF_LABELS),
                quarterly_cols,
                periods=4,
            )
            capex_quarterly = _sum_recent_periods(
                _select_cashflow_series(quarterly_cf, _CAPEX_LABELS),
                quarterly_cols,
                periods=4,
            )
            if fcf_quarterly is None and ocf_quarterly is not None and capex_quarterly is not None:
                fcf_quarterly = ocf_quarterly + capex_quarterly

            fcf_yearly = _latest_period_value(
                _select_cashflow_series(yearly_cf, _FCF_LABELS),
                yearly_cols,
            )
            ocf_yearly = _latest_period_value(
                _select_cashflow_series(yearly_cf, _OCF_LABELS),
                yearly_cols,
            )
            capex_yearly = _latest_period_value(
                _select_cashflow_series(yearly_cf, _CAPEX_LABELS),
                yearly_cols,
            )
            if fcf_yearly is None and ocf_yearly is not None and capex_yearly is not None:
                fcf_yearly = ocf_yearly + capex_yearly

            if fcf_quarterly is not None:
                free_cash_flow = fcf_quarterly
                fcf_period = "TTM"
                fcf_period_end = quarterly_period_end
                fcf_source = "cashflow_quarterly"
            elif fcf_yearly is not None:
                free_cash_flow = fcf_yearly
                fcf_period = fiscal_period or "FY"
                fcf_period_end = yearly_period_end
                fcf_source = "cashflow_yearly"

            if ocf_quarterly is not None:
                operating_cf = ocf_quarterly
            elif ocf_yearly is not None:
                operating_cf = ocf_yearly
        except Exception as e:
            logger.debug("cashflow statement parse failed: %s", e)

    if fcf_period is None and free_cash_flow is not None:
        fcf_period = "TTM"
        fcf_source = fcf_source or "info"

    return operating_cf, free_cash_flow, fcf_period, fcf_period_end, fcf_source


def _build_yield_metrics(
    info: dict[str, Any],
    valuation: dict[str, Any],
    free_cash_flow: float | None,
    market_cap: float | None,
    price: float | None,
) -> dict[str, Any]:
    """Build yield and dividend sustainability metrics.

    WP4: FCF yield and the payout ratios mix statement and price data, so they need
    one currency; the forward dividend yield (Q3) is ``dividendRate / price``.
    """
    currency = currency_basis(info.get("currency"), info.get("financialCurrency"))
    currency_mismatch = currency != VERIFIED
    status: dict[str, dict[str, Any]] = {}
    fcf_yield = (
        free_cash_flow / market_cap
        if (
            free_cash_flow is not None
            and market_cap is not None
            and market_cap > 0
            and not currency_mismatch
        )
        else None
    )
    if currency_mismatch and free_cash_flow is not None:
        status["fcf_yield"] = metric_entry(currency, False, currency)
    pe_trailing = usable_metric(valuation, "pe_trailing")
    earnings_yield = (
        1 / pe_trailing
        if pe_trailing is not None and pe_trailing > 0
        else None
    )
    dividends = dividend_yields(info, price)

    dividend_rate = safe_float(info.get("dividendRate"))
    trailing_eps = valuation.get("trailing_eps")
    payout_ratio = (
        dividend_rate / trailing_eps
        if dividend_rate is not None and trailing_eps is not None and trailing_eps > 0 and not currency_mismatch
        else None
    )
    if currency_mismatch and dividend_rate is not None:
        status["dividend_payout_ratio"] = metric_entry(currency, False, currency)
    shares_outstanding = safe_float(info.get("sharesOutstanding"))
    per_share_basis = _per_share_basis(info, price)
    fcf_payout = (
        (dividend_rate * shares_outstanding) / free_cash_flow
        if (
            dividend_rate is not None
            and shares_outstanding is not None
            and free_cash_flow is not None
            and free_cash_flow > 0
            and per_share_basis == VERIFIED
        )
        else None
    )
    if per_share_basis != VERIFIED and dividend_rate is not None and free_cash_flow is not None:
        status["fcf_payout_ratio"] = metric_entry(per_share_basis, False, per_share_basis)
    if dividends["basis"] == UNVERIFIED_CURRENCY:
        status["dividend_yield"] = metric_entry(UNVERIFIED_CURRENCY, False, UNVERIFIED_CURRENCY)
    trailing_rate = safe_float(info.get("trailingAnnualDividendRate"))
    if trailing_rate and dividends["trailing"] is None and price is not None and per_share_basis != VERIFIED:
        status["dividend_yield_trailing"] = metric_entry(per_share_basis, False, per_share_basis)

    yield_warnings: list[str] = []
    is_fcf_negative = free_cash_flow is not None and free_cash_flow <= 0
    if currency_mismatch:
        yield_warnings.append("currency_mismatch")
    if is_fcf_negative:
        yield_warnings.append("negative_fcf")

    is_eps_negative = trailing_eps is not None and trailing_eps <= 0
    if is_eps_negative:
        yield_warnings.append("negative_eps")

    attractive_fcf_triggered: bool | None = None
    if fcf_yield is not None and not is_fcf_negative:
        attractive_fcf_triggered = check_rule(fcf_yield, 0.05, operator.gt)

    sustainable_div_triggered: bool | None = None
    if payout_ratio is not None and not is_eps_negative:
        sustainable_div_triggered = check_rule(payout_ratio, 0.75, operator.lt)

    return {
        "fcf_yield": safe_round(fcf_yield, 4),
        "earnings_yield": safe_round(earnings_yield, 4) if not is_eps_negative else None,
        "dividend_yield": safe_round(dividends["yield"], 4),
        "dividend_yield_basis": dividends["basis"],
        "dividend_yield_trailing": safe_round(dividends["trailing"], 4),
        "dividend_payout_ratio": safe_round(payout_ratio, 4) if not is_eps_negative else None,
        "fcf_payout_ratio": safe_round(fcf_payout, 4) if not is_fcf_negative else None,
        "rules": {
            "attractive_fcf_yield": {
                "triggered": attractive_fcf_triggered,
                "threshold": 0.05,
            },
            "sustainable_dividend": {
                "triggered": sustainable_div_triggered,
                "threshold": 0.75,
            },
        },
        "warnings": yield_warnings if yield_warnings else None,
        "metric_status": status,
    }


_MIN_HISTORY_POINTS = 8  # sufficiency policy for valuation-history stats (WP4)


def _compute_valuation_history(
    income_stmt: pd.DataFrame | None,
    price_hist: pd.DataFrame | None,
    info: dict[str, Any],
    valuation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute historical P/E and P/S from quarterly income statements (None inputs:
    the read failed or returned nothing).

    WP4: entries need the same currency and share basis as current ratios; stats
    need at least ``_MIN_HISTORY_POINTS`` points; ``current_percentile`` ranks the
    current (usable) ratio against the history.
    """
    shares_outstanding = safe_float(info.get("sharesOutstanding"))
    if shares_outstanding is None or shares_outstanding <= 0:
        return None

    if income_stmt is None or income_stmt.empty:
        return None
    if price_hist is None or price_hist.empty or "Close" not in price_hist:
        return None

    basis = currency_basis(info.get("currency"), info.get("financialCurrency"))
    if basis == VERIFIED:
        basis = share_units_basis(info, current_price_from_info(info))
    if basis != VERIFIED:
        return {
            "pe_history": None, "ps_history": None, "pe_stats": None, "ps_stats": None,
            "history_points": 0, "history_span_years": None, "status": basis,
        }

    close_series = pd.to_numeric(price_hist["Close"], errors="coerce").dropna()
    if close_series.empty:
        return None
    close_series.index = pd.to_datetime(close_series.index)
    if getattr(close_series.index, "tz", None) is not None:
        close_series.index = close_series.index.tz_localize(None)

    normalized_idx = _build_normalized_index(income_stmt)

    net_income_label = normalized_idx.get("netincome")
    revenue_label = normalized_idx.get("totalrevenue") or normalized_idx.get("operatingrevenue")

    # Sort columns by date descending
    cols = sorted(income_stmt.columns, reverse=True)

    pe_history: list[dict[str, Any]] = []
    ps_history: list[dict[str, Any]] = []

    # Compute trailing ratios using 4-quarter rolling sums
    for i in range(len(cols) - 3):
        quarter_end = cols[i]
        quarter_label = str(quarter_end.date()) if hasattr(quarter_end, "date") else str(quarter_end)
        trailing_cols = cols[i : i + 4]
        quarter_ts = pd.Timestamp(quarter_end)
        if quarter_ts.tzinfo is not None:
            quarter_ts = quarter_ts.tz_localize(None)
        price_window = close_series.loc[:quarter_ts]
        if price_window.empty:
            continue
        quarter_close = safe_float(price_window.iloc[-1])
        if quarter_close is None or quarter_close <= 0:
            continue

        if net_income_label is not None:
            ni_values = [
                safe_float(income_stmt.loc[net_income_label, c]) for c in trailing_cols
            ]
            if all(v is not None for v in ni_values):
                ni_ttm = sum(ni_values)
                if ni_ttm > 0:
                    eps_ttm = ni_ttm / shares_outstanding
                    pe = safe_round(quarter_close / eps_ttm, 2) if eps_ttm > 0 else None
                    pe_history.append({"quarter": quarter_label, "pe": pe})

        if revenue_label is not None:
            rev_values = [
                safe_float(income_stmt.loc[revenue_label, c]) for c in trailing_cols
            ]
            if all(v is not None for v in rev_values):
                rev_ttm = sum(rev_values)
                if rev_ttm > 0:
                    sales_per_share_ttm = rev_ttm / shares_outstanding
                    ps = (
                        safe_round(quarter_close / sales_per_share_ttm, 2)
                        if sales_per_share_ttm > 0
                        else None
                    )
                    ps_history.append({"quarter": quarter_label, "ps": ps})

    def _compute_stats(
        history: list[dict[str, Any]], key: str, current: float | None,
    ) -> dict[str, Any] | None:
        values = [entry[key] for entry in history if entry.get(key) is not None]
        if len(values) < _MIN_HISTORY_POINTS:
            return None
        percentile = (
            safe_round(sum(1 for v in values if v < current) / len(values) * 100, 1)
            if current is not None
            else None
        )
        return {
            "min": safe_round(min(values), 2),
            "max": safe_round(max(values), 2),
            "mean": safe_round(statistics.mean(values), 2),
            "median": safe_round(statistics.median(values), 2),
            "current_percentile": percentile,
        }

    quarters = sorted({entry["quarter"] for entry in pe_history + ps_history})
    span_years = (
        safe_round((pd.Timestamp(quarters[-1]) - pd.Timestamp(quarters[0])).days / 365.25, 2)
        if len(quarters) > 1
        else (0.0 if quarters else None)
    )
    pe_stats = _compute_stats(pe_history, "pe", usable_metric(valuation, "pe_trailing"))
    ps_stats = _compute_stats(ps_history, "ps", usable_metric(valuation, "ps_trailing"))
    if pe_stats or ps_stats:
        status = "available"
    elif quarters:
        status = "insufficient_history"
    else:
        status = "unavailable"
    return {
        "pe_history": pe_history if pe_history else None,
        "ps_history": ps_history if ps_history else None,
        "pe_stats": pe_stats,
        "ps_stats": ps_stats,
        "history_points": len(quarters),
        "history_span_years": span_years,
        "status": status,
    }


def _compute_fundamental_trends(income_stmt: pd.DataFrame | None) -> dict[str, Any] | None:
    """Compute multi-period fundamental trends from quarterly income statements."""
    if income_stmt is None or income_stmt.empty:
        return None

    normalized_idx = _build_normalized_index(income_stmt)

    metric_keys = {
        "total_revenue": normalized_idx.get("totalrevenue") or normalized_idx.get("operatingrevenue"),
        "gross_profit": normalized_idx.get("grossprofit"),
        "operating_income": normalized_idx.get("operatingincome"),
        "net_income": normalized_idx.get("netincome"),
    }

    # Sort columns descending, take up to 16 quarters (4 years, enough for 3Y CAGR)
    cols = sorted(income_stmt.columns, reverse=True)[:16]
    if len(cols) < 2:
        return None

    quarterly_data: list[dict[str, Any]] = []
    for i, col in enumerate(cols):
        quarter_label = str(col.date()) if hasattr(col, "date") else str(col)
        entry: dict[str, Any] = {"quarter": quarter_label}

        for metric_name, label in metric_keys.items():
            if label is None:
                entry[metric_name] = None
                continue
            val = safe_float(income_stmt.loc[label, col])
            entry[metric_name] = val

            # QoQ change (compare to next column which is previous quarter)
            if i + 1 < len(cols):
                prev_val = safe_float(income_stmt.loc[label, cols[i + 1]])
                if prev_val is not None and prev_val != 0 and val is not None:
                    entry[f"{metric_name}_qoq"] = safe_round((val - prev_val) / abs(prev_val), 4)

            # YoY change (compare to quarter 4 periods ago)
            if i + 4 < len(cols):
                yoy_val = safe_float(income_stmt.loc[label, cols[i + 4]])
                if yoy_val is not None and yoy_val != 0 and val is not None:
                    entry[f"{metric_name}_yoy"] = safe_round((val - yoy_val) / abs(yoy_val), 4)

        quarterly_data.append(entry)

    # Margin trends: compare latest quarter vs 4 quarters ago
    def _margin_trend(numerator_key: str, denominator_key: str) -> str | None:
        if len(quarterly_data) < 5:
            return None
        latest = quarterly_data[0]
        past = quarterly_data[4]
        num_latest = latest.get(numerator_key)
        den_latest = latest.get(denominator_key)
        num_past = past.get(numerator_key)
        den_past = past.get(denominator_key)

        if any(v is None for v in (num_latest, den_latest, num_past, den_past)):
            return None
        if den_latest == 0 or den_past == 0:
            return None

        margin_now = num_latest / den_latest
        margin_then = num_past / den_past
        diff = margin_now - margin_then

        if diff > 0.01:
            return "expanding"
        elif diff < -0.01:
            return "contracting"
        return "stable"

    margin_trend = {
        "gross": _margin_trend("gross_profit", "total_revenue"),
        "operating": _margin_trend("operating_income", "total_revenue"),
        "net": _margin_trend("net_income", "total_revenue"),
    }

    # Revenue CAGR 3Y using a strict 12-quarter lookback
    revenue_cagr_3y = None
    if len(cols) >= 13:
        rev_label = metric_keys.get("total_revenue")
        if rev_label is not None:
            rev_latest = safe_float(income_stmt.loc[rev_label, cols[0]])
            rev_3y_ago = safe_float(income_stmt.loc[rev_label, cols[12]])
            years = 3.0
            if (
                rev_latest is not None
                and rev_3y_ago is not None
                and rev_3y_ago > 0
                and rev_latest > 0
            ):
                revenue_cagr_3y = safe_round((rev_latest / rev_3y_ago) ** (1 / years) - 1, 4)

    # EPS CAGR 3Y using net income proxy with strict 12-quarter lookback
    eps_cagr_3y = None
    if len(cols) >= 13:
        ni_label = metric_keys.get("net_income")
        if ni_label is not None:
            ni_latest = safe_float(income_stmt.loc[ni_label, cols[0]])
            ni_3y_ago = safe_float(income_stmt.loc[ni_label, cols[12]])
            years = 3.0
            if (
                ni_latest is not None
                and ni_3y_ago is not None
                and ni_3y_ago > 0
                and ni_latest > 0
            ):
                eps_cagr_3y = safe_round((ni_latest / ni_3y_ago) ** (1 / years) - 1, 4)

    return {
        "quarterly_data": quarterly_data,
        "margin_trend": margin_trend,
        "revenue_cagr_3y": revenue_cagr_3y,
        "eps_cagr_3y": eps_cagr_3y,
    }


_EARNINGS_ESTIMATE_PERIODS = {
    "0q": "current_quarter",
    "+1q": "next_quarter",
    "0y": "current_year",
    "+1y": "next_year",
}


def _extract_estimate_rows(est_df: Any, key_prefix: str) -> dict[str, dict[str, float | None]]:
    """Pull avg/low/high/num_analysts rows for known periods out of an estimates DataFrame."""
    if est_df is None or est_df.empty:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for row_key, label in _EARNINGS_ESTIMATE_PERIODS.items():
        if row_key in est_df.index:
            row = est_df.loc[row_key]
            out[f"{key_prefix}_{label}"] = {
                "avg": safe_float(row.get("avg")),
                "low": safe_float(row.get("low")),
                "high": safe_float(row.get("high")),
                "num_analysts": safe_float(row.get("numberOfAnalysts")),
            }
    return out


def _build_earnings_estimates(
    eps_estimates: pd.DataFrame | None, revenue_estimates: pd.DataFrame | None, info: dict[str, Any]
) -> dict[str, Any] | None:
    """Analyst earnings and revenue estimates (each None when its read failed), plus a
    forward P/E from the already-fetched info."""
    result: dict[str, Any] = {}
    result.update(_extract_estimate_rows(eps_estimates, "eps"))
    result.update(_extract_estimate_rows(revenue_estimates, "rev"))

    forward_pe: float | None = None
    if isinstance(info, dict) and currency_basis(info.get("currency"), info.get("financialCurrency")) == VERIFIED:
        shares_out = safe_float(info.get("sharesOutstanding"))
        next_year_eps = result.get("eps_next_year", {}).get("avg")
        if (
            next_year_eps is not None
            and next_year_eps > 0
            and shares_out is not None
            and shares_out > 0
        ):
            current_price = current_price_from_info(info)
            if current_price is not None and current_price > 0:
                forward_pe = safe_round(current_price / next_year_eps, 2)

    result["forward_pe_from_estimates"] = forward_pe

    return result if len(result) > 1 else None


def _payout_ratio_score(payout_ratio: float | None) -> float | None:
    """Score the payout ratio safety band (0-100). None when payout_ratio is missing."""
    if payout_ratio is None:
        return None
    if payout_ratio < 0:
        return 0  # Negative earnings
    if payout_ratio <= 0.3:
        return 100
    if payout_ratio <= 0.5:
        return 80
    if payout_ratio <= 0.7:
        return 60
    if payout_ratio <= 0.9:
        return 40
    if payout_ratio <= 1.0:
        return 20
    return 0


def _fcf_coverage_score(
    free_cash_flow: float | None,
    dividend_rate: float | None,
    shares_outstanding: float | None,
) -> float | None:
    """Score FCF coverage of dividends (0-100). None when inputs are missing."""
    if (
        free_cash_flow is None
        or dividend_rate is None
        or shares_outstanding is None
        or shares_outstanding <= 0
    ):
        return None
    total_dividends = dividend_rate * shares_outstanding
    if total_dividends <= 0:
        return None
    fcf_coverage = free_cash_flow / total_dividends
    if fcf_coverage >= 2.0:
        return 100
    if fcf_coverage >= 1.5:
        return 80
    if fcf_coverage >= 1.0:
        return 60
    if fcf_coverage >= 0.5:
        return 30
    return 0


def _analyze_dividend_history(dividends: Any, info: dict[str, Any]) -> dict[str, Any] | None:
    """Analyze dividend payment history, streak, growth, and safety.

    The streak and CAGR calculations intentionally skip the current calendar year
    when it appears partial (i.e., the most recent payment is recent relative to
    today). Otherwise mid-year totals — which include only the dividends paid so
    far — would falsely break a multi-year increase streak.
    """
    if dividends is None or dividends.empty:
        return None

    # Group by year, compute annual totals
    annual: dict[int, float] = {}
    for date, amount in dividends.items():
        year = date.year
        val = safe_float(amount)
        if val is not None and val > 0:
            annual[year] = annual.get(year, 0.0) + val

    if not annual:
        return None

    sorted_years = sorted(annual.keys(), reverse=True)
    current_year = datetime.now().year

    # Determine whether the latest year is a partial (in-progress) calendar year.
    # We treat it as partial only when the latest year equals the current calendar
    # year. Companies whose last payment was in a prior year are not partial.
    latest_year = sorted_years[0]
    latest_year_is_partial = latest_year == current_year

    # "Complete" years are those safe to use for streak/CAGR comparisons.
    complete_years = [y for y in sorted_years if not (y == current_year and latest_year_is_partial)]

    # Last 10 years of annual dividends — include the partial year with a flag.
    annual_dividends: list[dict[str, Any]] = []
    for y in sorted_years[:10]:
        entry: dict[str, Any] = {"year": y, "total": safe_round(annual[y], 4)}
        if y == current_year and latest_year_is_partial:
            entry["partial"] = True
        annual_dividends.append(entry)

    # Consecutive years of increases (dividend streak) — use complete years only.
    streak = 0
    for i in range(len(complete_years) - 1):
        if annual[complete_years[i]] > annual[complete_years[i + 1]]:
            streak += 1
        else:
            break

    # CAGR calculations — use complete years only.
    def _cagr(years_back: int) -> float | None:
        if len(complete_years) <= years_back:
            return None
        latest_y = complete_years[0]
        past_year = complete_years[years_back]
        latest_val = annual[latest_y]
        past_val = annual[past_year]
        actual_years = latest_y - past_year
        if past_val <= 0 or latest_val <= 0 or actual_years <= 0:
            return None
        return safe_round((latest_val / past_val) ** (1 / actual_years) - 1, 4)

    # Safety score (0-100): blend payout-ratio and FCF-coverage bands
    payout_score = _payout_ratio_score(safe_float(info.get("payoutRatio")))
    # FCF (statement) vs dividend per share x shares (quote): one basis only (WP4).
    fcf_score = (
        _fcf_coverage_score(
            safe_float(info.get("freeCashflow")),
            safe_float(info.get("dividendRate")),
            safe_float(info.get("sharesOutstanding")),
        )
        if _per_share_basis(info, current_price_from_info(info)) == VERIFIED
        else None
    )
    scores = [s for s in (payout_score, fcf_score) if s is not None]
    safety_score = safe_round(statistics.mean(scores), 0) if scores else None

    return {
        "annual_dividends": annual_dividends,
        "dividend_streak": streak,
        "cagr_1y": _cagr(1),
        "cagr_3y": _cagr(3),
        "cagr_5y": _cagr(5),
        "safety_score": safety_score,
    }


def _format_date_string(value: Any) -> str | None:
    """Normalize date-like values to YYYY-MM-DD when possible."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value).date().isoformat()
        except (ValueError, OSError, OverflowError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                continue
        return raw
    return None


def _normalize_cashflow_label(label: str) -> str:
    """Normalize cash flow row labels for matching."""
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def _build_normalized_index(df: pd.DataFrame) -> dict[str, Any]:
    """Map normalized row labels to their original dataframe index labels."""
    return {_normalize_cashflow_label(idx): idx for idx in df.index}


def _select_cashflow_series(
    df: pd.DataFrame | None,
    candidates: tuple[str, ...],
) -> pd.Series | None:
    """Return the first matching cashflow series by normalized label."""
    if df is None or df.empty:
        return None
    normalized_index = _build_normalized_index(df)
    for candidate in candidates:
        idx = normalized_index.get(candidate)
        if idx is not None:
            return pd.to_numeric(df.loc[idx], errors="coerce")
    return None


def _sorted_cashflow_columns(columns: pd.Index) -> list[Any]:
    """Sort cashflow columns by date desc when possible; else keep order."""
    cols = list(columns)
    if not cols:
        return cols
    try:
        parsed = pd.to_datetime(cols, errors="coerce")
    except Exception:
        return cols
    if parsed.notna().all():
        return [c for _, c in sorted(zip(parsed, cols, strict=True), reverse=True)]
    return cols


def _sum_recent_periods(
    series: pd.Series | None,
    columns: list[Any],
    *,
    periods: int,
) -> float | None:
    """Sum most recent periods from a series; returns None if insufficient data."""
    if series is None:
        return None
    ordered = series.reindex(columns)
    values = pd.to_numeric(ordered, errors="coerce").dropna()
    if len(values) < periods:
        return None
    return float(values.iloc[:periods].sum())


def _latest_period_value(
    series: pd.Series | None,
    columns: list[Any],
) -> float | None:
    """Return most recent value from a series."""
    if series is None:
        return None
    ordered = series.reindex(columns)
    values = pd.to_numeric(ordered, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _period_end_from_columns(columns: list[Any]) -> str | None:
    """Return ISO date for the most recent period end column."""
    if not columns:
        return None
    try:
        dt = pd.to_datetime(columns[0], errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    return dt.date().isoformat()


def _read_cashflow(ticker: Any, freq: str | None) -> pd.DataFrame | None:
    """Cashflow DataFrame with yfinance fallbacks (blocking; run as a job)."""
    if hasattr(ticker, "get_cashflow"):
        return ticker.get_cashflow(freq=freq) if freq else ticker.get_cashflow()
    if freq == "quarterly":
        return ticker.quarterly_cashflow
    return ticker.cashflow
