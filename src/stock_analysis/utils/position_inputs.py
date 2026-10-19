"""Personal position inputs, validated before data access or numeric coercion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import partial
from numbers import Real
from typing import Annotated, Literal, cast

from pydantic import BeforeValidator, Field

from stock_analysis.utils.sizing_inputs import (
    _DECIMAL_PATTERN,
    _DecimalString,
    normalize_positive_number,
)

Horizon = Literal["mid_term", "long_term"]


@dataclass(frozen=True)
class PositionInputs:
    cost_basis: float | None
    purchase_date: date | None
    shares: float | None
    horizon: Horizon
    portfolio_value: float | None
    max_holding_pct: float | None


def _normalize_cost_basis(value: object) -> float:
    """Permit a genuine zero without accepting a positive conversion underflow."""
    message = "cost_basis must be a finite nonnegative number; decimal numeric strings are accepted"
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, Real) and value == 0:
        return 0.0
    try:
        if (
            isinstance(value, str)
            and re.fullmatch(_DECIMAL_PATTERN, value)
            and Decimal(value.strip()).is_zero()
        ):
            return 0.0
    except InvalidOperation as exc:
        raise ValueError(message) from exc
    try:
        return normalize_positive_number(value, field="cost_basis")
    except ValueError as exc:
        raise ValueError(message) from exc


def _normalize_purchase_date(value: object, today: date) -> date:
    message = "purchase_date must be a real YYYY-MM-DD date no later than the request date"
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
        raise ValueError(message)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if parsed > today:
        raise ValueError(message)
    return parsed


def normalize_position_inputs(
    cost_basis: object = None,
    purchase_date: object = None,
    shares: object = None,
    horizon: object = "mid_term",
    portfolio_value: object = None,
    max_holding_pct: object = None,
    *,
    today: date,
) -> PositionInputs:
    """Normalize optional user context without filling missing values."""
    if not isinstance(horizon, str) or horizon not in ("mid_term", "long_term"):
        raise ValueError("horizon must be mid_term or long_term")
    return PositionInputs(
        cost_basis=None if cost_basis is None else _normalize_cost_basis(cost_basis),
        purchase_date=(None if purchase_date is None else _normalize_purchase_date(purchase_date, today)),
        shares=None if shares is None else normalize_positive_number(shares, field="shares"),
        horizon=cast(Horizon, horizon),
        portfolio_value=(None if portfolio_value is None else normalize_positive_number(
            portfolio_value, field="portfolio_value",
        )),
        max_holding_pct=(None if max_holding_pct is None else normalize_positive_number(
            max_holding_pct, field="max_holding_pct", maximum=100,
        )),
    )


CostBasis = Annotated[
    float,
    BeforeValidator(
        _normalize_cost_basis,
        json_schema_input_type=Annotated[float, Field(ge=0)] | _DecimalString,
    ),
]
OwnedShares = Annotated[
    float,
    BeforeValidator(
        partial(normalize_positive_number, field="shares"),
        json_schema_input_type=Annotated[float, Field(gt=0)] | _DecimalString,
    ),
]
PortfolioValue = Annotated[
    float,
    BeforeValidator(
        partial(normalize_positive_number, field="portfolio_value"),
        json_schema_input_type=Annotated[float, Field(gt=0)] | _DecimalString,
    ),
]
HoldingPercent = Annotated[
    float,
    BeforeValidator(
        partial(normalize_positive_number, field="max_holding_pct", maximum=100),
        json_schema_input_type=Annotated[float, Field(gt=0, le=100)] | _DecimalString,
    ),
]
