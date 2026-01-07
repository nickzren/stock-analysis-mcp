"""Fundamentals snapshot tool."""

import operator
from datetime import datetime
from time import perf_counter
from typing import Any

from stock_mcp.data.yfinance_client import fetch_info
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance
from stock_mcp.utils.validators import check_rule


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
        info = await fetch_info(symbol)
    except ValueError as e:
        return build_error_response(
            error_type="invalid_symbol",
            message=str(e),
            symbol=symbol,
        )
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch data: {e}",
            symbol=symbol,
        )

    normalized_symbol = symbol.upper().strip()

    # Valuation
    # P/S: try direct field first, then compute from market_cap / revenue
    ps_trailing = _safe_float(info.get("priceToSalesTrailing12Months"))
    ps_source: str | None = None
    if ps_trailing is not None:
        ps_source = "direct"
    else:
        # Fallback: compute P/S from market_cap / revenue_ttm
        market_cap_val = _safe_float(info.get("marketCap"))
        revenue_ttm = _safe_float(info.get("totalRevenue"))
        if market_cap_val is not None and revenue_ttm is not None and revenue_ttm > 0:
            ps_trailing = market_cap_val / revenue_ttm
            ps_source = "computed"

    # Build ps_explanation for auditability when computed
    ps_explanation: str | None = None
    if ps_source == "computed":
        ps_explanation = "P/S computed from market_cap / revenue_ttm (priceToSalesTrailing12Months unavailable)"

    # EV/Sales: compute from enterpriseValue / totalRevenue
    # Better than P/S when debt/cash position is material
    enterprise_value = _safe_float(info.get("enterpriseValue"))
    revenue_ttm = _safe_float(info.get("totalRevenue"))
    ev_to_sales: float | None = None
    ev_to_sales_source: str | None = None
    if enterprise_value is not None and revenue_ttm is not None and revenue_ttm > 0:
        ev_to_sales = enterprise_value / revenue_ttm
        ev_to_sales_source = "computed"

    valuation = {
        "pe_trailing": _safe_float(info.get("trailingPE")),
        "pe_forward": _safe_float(info.get("forwardPE")),
        "ps_trailing": _safe_round(ps_trailing, 2),
        "ps_source": ps_source,  # "direct" or "computed" or None
        "ps_explanation": ps_explanation,  # Only present when ps_source="computed"
        "pb_ratio": _safe_float(info.get("priceToBook")),
        "peg_ratio": _safe_float(info.get("pegRatio")),
        "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
        "ev_to_sales": _safe_round(ev_to_sales, 2),
        "ev_to_sales_source": ev_to_sales_source,
    }

    # Growth
    revenue_growth = _safe_float(info.get("revenueGrowth"))
    earnings_growth = _safe_float(info.get("earningsGrowth"))

    growth = {
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

    # Profitability
    gross_margin = _safe_float(info.get("grossMargins"))
    operating_margin = _safe_float(info.get("operatingMargins"))
    net_margin = _safe_float(info.get("profitMargins"))
    roe = _safe_float(info.get("returnOnEquity"))
    roa = _safe_float(info.get("returnOnAssets"))

    profitability = {
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roa": roa,
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

    # Financial Health
    total_cash = _safe_float(info.get("totalCash"))
    # Cash + short-term investments (more accurate liquidity for burn calculations)
    # yfinance doesn't always have this separately, but we can use total cash as proxy
    # Some tickers have cashAndShortTermInvestments
    cash_and_st_investments = _safe_float(info.get("cashAndShortTermInvestments"))
    if cash_and_st_investments is None:
        cash_and_st_investments = total_cash  # Fallback to total cash

    total_debt = _safe_float(info.get("totalDebt"))
    net_cash = (
        total_cash - total_debt
        if total_cash is not None and total_debt is not None
        else None
    )
    current_ratio = _safe_float(info.get("currentRatio"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))
    # Convert D/E from percentage to ratio if needed (yfinance returns as percentage)
    if debt_to_equity is not None and debt_to_equity > 10:
        debt_to_equity = debt_to_equity / 100

    financial_health = {
        "total_cash": total_cash,
        "cash_and_st_investments": cash_and_st_investments,
        "total_debt": total_debt,
        "net_cash": net_cash,
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "interest_coverage": None,  # Not directly available
        "rules": {
            "net_cash_positive": {
                "triggered": check_rule(net_cash, 0, operator.gt),
                "threshold": 0,
            },
            "low_debt": {
                "triggered": check_rule(debt_to_equity, 0.5, operator.lt),
                "threshold": 0.5,
            },
            "adequate_liquidity": {
                "triggered": check_rule(current_ratio, 1.0, operator.gt),
                "threshold": 1.0,
            },
        },
    }

    # Cash Flow
    operating_cf = _safe_float(info.get("operatingCashflow"))
    free_cash_flow = _safe_float(info.get("freeCashflow"))
    market_cap = _safe_float(info.get("marketCap"))
    revenue = _safe_float(info.get("totalRevenue"))

    fcf_margin = (
        free_cash_flow / revenue
        if free_cash_flow is not None and revenue is not None and revenue > 0
        else None
    )

    cash_flow = {
        "operating_cf_ttm": operating_cf,
        "free_cash_flow_ttm": free_cash_flow,
        "fcf_margin": _safe_round(fcf_margin, 4),
        "rules": {
            "positive_fcf": {
                "triggered": check_rule(free_cash_flow, 0, operator.gt),
                "threshold": 0,
            },
        },
    }

    # Yield metrics
    fcf_yield = (
        free_cash_flow / market_cap
        if free_cash_flow is not None and market_cap is not None and market_cap > 0
        else None
    )
    pe_trailing = _safe_float(info.get("trailingPE"))
    earnings_yield = (
        1 / pe_trailing
        if pe_trailing is not None and pe_trailing > 0
        else None
    )
    dividend_yield = _safe_float(info.get("dividendYield"))

    # Dividend sustainability
    dividend_rate = _safe_float(info.get("dividendRate"))
    trailing_eps = _safe_float(info.get("trailingEps"))
    payout_ratio = (
        dividend_rate / trailing_eps
        if dividend_rate is not None and trailing_eps is not None and trailing_eps > 0
        else None
    )
    shares_outstanding = _safe_float(info.get("sharesOutstanding"))
    fcf_payout = (
        (dividend_rate * shares_outstanding) / free_cash_flow
        if (dividend_rate is not None and shares_outstanding is not None
            and free_cash_flow is not None and free_cash_flow > 0)
        else None
    )

    # Build yield metrics warnings
    yield_warnings: list[str] = []

    # FCF yield: still compute if negative, but mark and don't trigger "attractive"
    is_fcf_negative = free_cash_flow is not None and free_cash_flow <= 0
    if is_fcf_negative:
        yield_warnings.append("negative_fcf")

    # Earnings yield: if EPS <= 0, yield is meaningless
    is_eps_negative = trailing_eps is not None and trailing_eps <= 0
    if is_eps_negative:
        yield_warnings.append("negative_eps")

    # Attractive FCF yield rule: None if FCF is negative (not False)
    attractive_fcf_triggered: bool | None = None
    if fcf_yield is not None and not is_fcf_negative:
        attractive_fcf_triggered = check_rule(fcf_yield, 0.05, operator.gt)

    # Sustainable dividend: None if EPS <= 0 (payout ratio meaningless)
    sustainable_div_triggered: bool | None = None
    if payout_ratio is not None and not is_eps_negative:
        sustainable_div_triggered = check_rule(payout_ratio, 0.75, operator.lt)

    yield_metrics = {
        "fcf_yield": _safe_round(fcf_yield, 4),
        "earnings_yield": _safe_round(earnings_yield, 4) if not is_eps_negative else None,
        "dividend_yield": _safe_round(dividend_yield, 4),
        "dividend_payout_ratio": _safe_round(payout_ratio, 4) if not is_eps_negative else None,
        "fcf_payout_ratio": _safe_round(fcf_payout, 4) if not is_fcf_negative else None,
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
    }

    # Build provenance with fiscal period info
    fiscal_year_end = info.get("lastFiscalYearEnd")
    fiscal_period = None
    if fiscal_year_end:
        try:
            fiscal_date = datetime.fromtimestamp(fiscal_year_end)
            fiscal_period = f"FY {fiscal_date.year}"
        except (ValueError, TypeError, OSError):
            pass

    warnings = []
    if info.get("trailingPE") and not info.get("forwardPE"):
        warnings.append("using_trailing_data")

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("fundamentals_snapshot", duration_ms),
        "data_provenance": {
            "fundamentals": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
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
    }


def _safe_float(value: Any) -> float | None:
    """Convert to float or return None."""
    if value is None:
        return None
    try:
        result = float(value)
        # Filter out invalid values
        if result != result:  # NaN check
            return None
        return result
    except (ValueError, TypeError):
        return None


def _safe_round(value: float | None, decimals: int) -> float | None:
    """Round to decimals or return None."""
    if value is None:
        return None
    return round(value, decimals)
