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
    valuation = {
        "pe_trailing": _safe_float(info.get("trailingPE")),
        "pe_forward": _safe_float(info.get("forwardPE")),
        "ps_trailing": _safe_float(info.get("priceToSalesTrailing12Months")),
        "pb_ratio": _safe_float(info.get("priceToBook")),
        "peg_ratio": _safe_float(info.get("pegRatio")),
        "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
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

    # Calculate FCF margin
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
