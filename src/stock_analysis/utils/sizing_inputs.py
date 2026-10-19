"""Shared sizing-input normalization, before framework coercion or data reads."""

from __future__ import annotations

import math
import re
from enum import Enum
from functools import partial
from numbers import Real
from typing import Annotated

from pydantic import BeforeValidator, Field

_DECIMAL_PATTERN = r"^\s*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\s*$"


class SwingSizingDefault(float, Enum):
    """Numeric defaults that remain distinguishable from explicit equal values."""

    RISK = 1.0
    POSITION = 10.0


def normalize_positive_number(
    value: object, *, field: str, maximum: float | None = None,
) -> float:
    """Accept finite positive numbers or decimal strings, without rounding."""
    bounds = "a finite positive number" if maximum is None else f"finite in (0, {maximum:g}]"
    message = f"{field} must be {bounds}; decimal numeric strings are accepted"
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, str):
        if re.fullmatch(_DECIMAL_PATTERN, value) is None:
            raise ValueError(message)
    elif not isinstance(value, Real):
        raise ValueError(message)
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(message)
    if maximum is not None and normalized > maximum:
        raise ValueError(message)
    return normalized


def normalize_sizing_params(
    account_size: object, risk_per_trade_pct: object, max_position_pct: object,
) -> tuple[float | None, float, float]:
    """Normalize swing/scan inputs while preserving their nullable account."""
    account = (
        None if account_size is None
        else normalize_positive_number(account_size, field="account_size")
    )
    risk = normalize_positive_number(risk_per_trade_pct, field="risk_per_trade_pct", maximum=100)
    cap = normalize_positive_number(max_position_pct, field="max_position_pct", maximum=100)
    return account, risk, cap


_DecimalString = Annotated[str, Field(pattern=_DECIMAL_PATTERN)]
_PositiveNumber = Annotated[float, Field(gt=0)]
_PercentageNumber = Annotated[float, Field(gt=0, le=100)]

AccountSize = Annotated[
    float,
    BeforeValidator(
        partial(normalize_positive_number, field="account_size"),
        json_schema_input_type=_PositiveNumber | _DecimalString,
    ),
]
RiskPercent = Annotated[
    float,
    BeforeValidator(
        partial(normalize_positive_number, field="risk_per_trade_pct", maximum=100),
        json_schema_input_type=_PercentageNumber | _DecimalString,
    ),
]
PositionPercent = Annotated[
    float,
    BeforeValidator(
        partial(normalize_positive_number, field="max_position_pct", maximum=100),
        json_schema_input_type=_PercentageNumber | _DecimalString,
    ),
]
