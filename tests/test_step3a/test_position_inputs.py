"""Personal inputs are validated before framework coercion or data reads."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime
from fractions import Fraction
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from stock_analysis.utils.position_inputs import (
    CostBasis,
    HoldingPercent,
    OwnedShares,
    PortfolioValue,
    PositionInputs,
    normalize_position_inputs,
)

TODAY = date(2026, 10, 19)


def test_missing_context_stays_missing_and_inputs_are_frozen():
    inputs = normalize_position_inputs(today=TODAY)
    assert inputs == PositionInputs(None, None, None, "mid_term", None, None)
    with pytest.raises(FrozenInstanceError):
        inputs.shares = 1


def test_normalization_keeps_argument_order_and_fractional_shares():
    inputs = normalize_position_inputs(
        " +80.5 ", "2024-02-29", ".125", "long_term", "1e4", "10.5", today=TODAY,
    )
    assert inputs == PositionInputs(80.5, date(2024, 2, 29), 0.125, "long_term", 10000.0, 10.5)


@pytest.mark.parametrize("raw", [0, 0.0, -0.0, "0", "-0.000", "+0e-400", " 0.0 "])
def test_genuine_zero_cost_is_valid(raw):
    assert normalize_position_inputs(cost_basis=raw, today=TODAY).cost_basis == 0.0


@pytest.mark.parametrize("raw", ["1e-400", "-1e-400", Fraction(1, 10**400)])
def test_cost_conversion_underflow_does_not_become_a_free_position(raw):
    with pytest.raises(ValueError, match="cost_basis"):
        normalize_position_inputs(cost_basis=raw, today=TODAY)


@pytest.mark.parametrize("field", ["cost_basis", "shares", "portfolio_value", "max_holding_pct"])
@pytest.mark.parametrize("raw", [True, False, "nan", "inf", float("nan"), float("inf"),
    -1, "1e309", 10**400, "0_0", "1,000", "0e99999999999999999999", "", [], {}, 1 + 0j])
def test_invalid_numeric_values_name_the_field(field, raw):
    with pytest.raises(ValueError, match=field):
        normalize_position_inputs(**{field: raw}, today=TODAY)


@pytest.mark.parametrize("field", ["shares", "portfolio_value", "max_holding_pct"])
@pytest.mark.parametrize("raw", [0, "0", "1e-400"])
def test_positive_context_fields_reject_zero_and_underflow(field, raw):
    with pytest.raises(ValueError, match=field):
        normalize_position_inputs(**{field: raw}, today=TODAY)


def test_only_holding_percentage_has_a_hundred_percent_ceiling():
    inputs = normalize_position_inputs(1000, shares=1000, portfolio_value=1000,
                                      max_holding_pct="100", today=TODAY)
    assert inputs.max_holding_pct == 100
    for raw in (100.00001, "101"):
        with pytest.raises(ValueError, match="max_holding_pct"):
            normalize_position_inputs(max_holding_pct=raw, today=TODAY)


@pytest.mark.parametrize("raw", ["2026-10-20", "2023-02-29", "2026-13-01", "2026-00-01",
    "2026-1-01", "2026-01-1", "20261019", "2026-10-19T00:00:00", " 2026-10-19 ",
    "0000-01-01", "２０２６-１０-１９", "", True, 20261019, date(2026, 10, 19),
    datetime(2026, 10, 19), []])
def test_dates_require_exact_real_iso_date_not_after_request_date(raw):
    with pytest.raises(ValueError, match="purchase_date"):
        normalize_position_inputs(purchase_date=raw, today=TODAY)


def test_purchase_date_uses_supplied_request_date_and_accepts_leap_day():
    assert normalize_position_inputs(purchase_date="2024-02-29", today=date(2024, 2, 29)).purchase_date == date(2024, 2, 29)
    with pytest.raises(ValueError, match="purchase_date"):
        normalize_position_inputs(purchase_date="2024-03-01", today=date(2024, 2, 29))


@pytest.mark.parametrize("raw", [None, "short_term", "long-term", "MID_TERM", "", True, [], {}])
def test_unknown_horizon_is_rejected(raw):
    with pytest.raises(ValueError, match="horizon"):
        normalize_position_inputs(horizon=raw, today=TODAY)


@pytest.mark.parametrize(("alias", "field", "value"), [
    (CostBasis, "cost_basis", 0), (OwnedShares, "shares", 0.125),
    (PortfolioValue, "portfolio_value", 10000), (HoldingPercent, "max_holding_pct", 100),
])
def test_framework_aliases_accept_decimal_strings_and_reject_booleans_before_coercion(alias, field, value):
    adapter = TypeAdapter(alias)
    assert adapter.validate_python(str(value)) == adapter.validate_python(value) == value
    for raw in (True, False, "nan", "1e309", "1e-400"):
        with pytest.raises(ValidationError, match=field):
            adapter.validate_python(raw)
    variants = adapter.json_schema()["anyOf"]
    assert {variant["type"] for variant in variants} == {"number", "string"}
    assert next(v for v in variants if v["type"] == "string")["pattern"]
    assert get_args(alias)[0] is float
