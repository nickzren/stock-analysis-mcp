"""Descriptive position amounts and the caller's allocation check, without stock policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from datetime import date
from fractions import Fraction
from typing import Any

from stock_analysis.utils.helpers import valid_price
from stock_analysis.utils.position_inputs import PositionInputs

_REVIEW_STATUS = {
    "above_limit": "review_allocation",
    "within_limit": "no_allocation_trigger",
    "not_evaluated": "not_evaluated",
    "not_requested": "not_requested",
}


def _fraction(value: float) -> Fraction:
    return Fraction(str(value))


def _representable(value: Fraction) -> float | None:
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number) or (number == 0 and value != 0):
        return None
    return number


def _unique(reasons: list[str]) -> list[str]:
    return list(dict.fromkeys(reasons))


def _price_context(provenance: dict[str, Any], price: float | None) -> dict[str, Any]:
    as_of = provenance.get("price_as_of")
    if price is None:
        return {"status": "unavailable", "as_of": as_of,
                "qualification": "A verified USD per-share price is unavailable."}
    staleness = provenance.get("staleness") or {}
    current = (
        provenance.get("price_source") in {"daily_close", "intraday_quote"}
        and isinstance(as_of, str) and bool(as_of)
        and not provenance.get("fallback_reason")
        and staleness.get("stale") is False
    )
    if current:
        qualification = f"Price context as of {as_of}; not a live account valuation."
    else:
        qualification = (
            f"Historical price context as of {as_of}; allocation cannot be verified."
            if as_of else "Historical price context; price time is unverified and allocation cannot be verified."
        )
    return {"status": "current" if current else "historical", "as_of": as_of,
            "qualification": qualification}


def build_position_context(
    *, inputs: PositionInputs, summary: dict[str, Any], quote_type: object, today: date,
) -> dict[str, Any]:
    """Calculate only requested personal context from analyze's selected price.

    Exact decimal arithmetic decides allocation boundaries before display rounding.
    A non-representable intermediate value is unavailable, never infinity or zero.
    """
    cost, shares = inputs.cost_basis, inputs.shares
    portfolio, ceiling = inputs.portfolio_value, inputs.max_holding_pct
    provenance = deepcopy(summary.get("price_provenance") or {})
    price_reasons: list[str] = []
    if quote_type not in ("EQUITY", "ETF"):
        price_reasons.append("share_units_unverified")
    currency_support = (summary.get("support") or {}).get("currency") or {}
    if summary.get("currency") != "USD" or currency_support.get("status") != "supported":
        price_reasons.append("usd_price_unverified")
    raw_price = summary.get("current_price")
    price = None if isinstance(raw_price, bool) else valid_price(raw_price)
    if price is None:
        price_reasons.append("price_unavailable")
    if price_reasons:
        price = None
    price_context = _price_context(provenance, price)

    values: dict[str, float | None] = {}
    exact: dict[str, Fraction] = {}
    reasons: dict[str, list[str]] = {"current_price": price_reasons.copy()}

    def calculate(field: str, required: dict[str, float | None], formula: Callable[[], Fraction]) -> None:
        missing = [f"missing_{key}" for key, value in required.items() if value is None]
        reasons[field] = _unique(price_reasons + missing)
        values[field] = None
        if reasons[field]:
            return
        result = formula()
        number = _representable(result)
        if number is None:
            reasons[field] = ["numeric_range_unavailable"]
        else:
            values[field], exact[field] = number, result

    # These lambdas are reached only when each named input and price is available.
    p = _fraction(price) if price is not None else Fraction(0)
    c = _fraction(cost) if cost is not None else Fraction(0)
    q = _fraction(shares) if shares is not None else Fraction(0)
    total = _fraction(portfolio) if portfolio is not None else Fraction(1)
    calculate("current_value", {"shares": shares}, lambda: p * q)
    calculate("gain_loss", {"cost_basis": cost}, lambda: (p - c) / c if c else Fraction(0))
    if cost == 0 and not reasons["gain_loss"]:
        values["gain_loss"] = None
        reasons["gain_loss"] = ["zero_cost_basis"]
    calculate("gain_loss_dollars", {"cost_basis": cost, "shares": shares}, lambda: (p - c) * q)
    calculate("portfolio_weight_pct", {"shares": shares, "portfolio_value": portfolio},
              lambda: 100 * p * q / total)
    # Weight depends on a representable holding value even when cancellation would
    # make the ratio representable by itself.
    if "numeric_range_unavailable" in reasons["current_value"]:
        values["portfolio_weight_pct"] = None
        reasons["portfolio_weight_pct"] = _unique(reasons["portfolio_weight_pct"] + ["numeric_range_unavailable"])
        exact.pop("portfolio_weight_pct", None)

    inconsistent = portfolio is not None and "current_value" in exact and exact["current_value"] > total
    requested: list[str] = []
    if cost is not None:
        requested.append("gain_loss")
    if shares is not None:
        requested.append("current_value")
    if cost is not None and shares is not None:
        requested.append("gain_loss_dollars")
    if portfolio is not None or ceiling is not None:
        requested.append("portfolio_weight_pct")
    context_reasons = _unique([reason for field in requested for reason in reasons[field]])
    historical = price_context["status"] == "historical"
    if requested and historical:
        context_reasons.append("price_not_current")
    if inconsistent:
        context_reasons.insert(0, "holding_exceeds_portfolio")
        context_status = "inconsistent"
    elif context_reasons:
        context_status = "partial"
    elif requested or inputs.purchase_date is not None:
        context_status = "available"
    else:
        context_status = "not_supplied"

    allocation_reasons: list[str] = []
    if ceiling is None:
        allocation_status = "not_requested"
    else:
        allocation_reasons = reasons["portfolio_weight_pct"].copy()
        if historical:
            allocation_reasons.append("price_not_current")
        if inconsistent:
            allocation_reasons.append("holding_exceeds_portfolio")
        if allocation_reasons:
            allocation_status = "not_evaluated"
        else:
            allocation_status = "above_limit" if exact["portfolio_weight_pct"] > _fraction(ceiling) else "within_limit"

    rounded_values: dict[str, float | None] = {}
    for field, value in values.items():
        precision = 4 if field in {"gain_loss", "portfolio_weight_pct"} else 2
        rounded_values[field] = None if value is None else round(value, precision)

    position = {
        "cost_basis": cost,
        "current_price": price,
        "shares": shares,
        **rounded_values,
        "portfolio_value": portfolio,
        "max_holding_pct": ceiling,
        "calculation_reasons": reasons,
        "price_provenance": provenance,
        "price_context": price_context,
    }
    purchase = inputs.purchase_date
    return {
        "position": position,
        "holding": {"purchase_date": purchase.isoformat() if purchase is not None else None,
                    "days_held": (today - purchase).days if purchase is not None else None},
        "context_status": {"status": context_status, "reasons": _unique(context_reasons)},
        "concentration_check": {
            "status": allocation_status,
            "weight_pct": position["portfolio_weight_pct"],
            "ceiling_pct": ceiling,
            "threshold_basis": "user_supplied" if ceiling is not None else None,
            "reasons": _unique(allocation_reasons),
        },
        "review_status": _REVIEW_STATUS[allocation_status],
        "assumptions": [
            "Personal amounts are user-reported USD; no brokerage account was verified.",
            "Cost basis and shares describe the current split-adjusted position; no lot or corporate-action reconciliation was performed.",
            "Portfolio value is the declared total including this holding, not cash or buying power.",
            "Gain/loss is price-only and excludes dividends, fees, tax and realized transactions.",
            "Allocation is checked only against the supplied ceiling; no add capacity or sale amount is calculated.",
        ],
    }
