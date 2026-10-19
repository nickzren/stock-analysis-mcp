"""Personal position arithmetic is descriptive and separate from stock policy."""

from copy import deepcopy
from datetime import date
from itertools import combinations

import pytest

from stock_analysis.tools.position_context import build_position_context
from stock_analysis.utils.position_inputs import PositionInputs

TODAY = date(2026, 10, 19)


def summary(price=100.0):
    return {
        "current_price": price,
        "currency": "USD",
        "support": {
            "currency": {"status": "supported", "code": "USD"},
            "instrument": {"status": "supported"},
        },
        "price_provenance": {
            "price_source": "daily_close", "price_as_of": "2026-10-16",
            "fallback_reason": None, "staleness": {"stale": False},
        },
    }


def context(*, data=None, quote_type="EQUITY", today=TODAY, **kwargs):
    defaults = {"cost_basis": None, "purchase_date": None, "shares": None, "horizon": "mid_term",
                "portfolio_value": None, "max_holding_pct": None}
    return build_position_context(
        inputs=PositionInputs(**(defaults | kwargs)), summary=summary() if data is None else data,
        quote_type=quote_type, today=today,
    )


def test_no_personal_context_does_not_invent_inputs_or_an_allocation_check():
    result = context()
    assert result["context_status"] == {"status": "not_supplied", "reasons": []}
    assert result["review_status"] == "not_requested"
    assert result["concentration_check"]["status"] == "not_requested"
    assert result["holding"] == {"purchase_date": None, "days_held": None}
    for field in ("cost_basis", "shares", "portfolio_value", "current_value", "gain_loss",
                  "gain_loss_dollars", "portfolio_weight_pct", "max_holding_pct"):
        assert result["position"][field] is None


@pytest.mark.parametrize("field", ["cost_basis", "shares", "purchase_date"])
def test_independent_context_does_not_require_unrequested_inputs(field):
    values = {"cost_basis": 80.0, "shares": 0.5, "purchase_date": date(2024, 2, 29)}
    result = context(**{field: values[field]})
    assert result["context_status"] == {"status": "available", "reasons": []}
    assert result["review_status"] == "not_requested"


def test_optional_subsets_null_only_dependent_values():
    values = {
        "cost_basis": 80.0, "shares": 2.5, "portfolio_value": 10000.0,
        "max_holding_pct": 10.0, "purchase_date": date(2026, 10, 1),
    }
    for count in range(len(values) + 1):
        for fields in combinations(values, count):
            supplied = {field: values[field] for field in fields}
            result = context(**supplied)
            position = result["position"]
            assert position["gain_loss"] == (0.25 if "cost_basis" in fields else None)
            assert position["current_value"] == (250.0 if "shares" in fields else None)
            assert position["gain_loss_dollars"] == (
                50.0 if {"shares", "cost_basis"}.issubset(fields) else None
            )
            assert position["portfolio_weight_pct"] == (
                2.5 if {"shares", "portfolio_value"}.issubset(fields) else None
            )
            assert result["holding"]["days_held"] == (18 if "purchase_date" in fields else None)
            if "max_holding_pct" not in fields:
                assert result["review_status"] == "not_requested"
            else:
                expected = "no_allocation_trigger" if {"shares", "portfolio_value"}.issubset(fields) else "not_evaluated"
                assert result["review_status"] == expected


@pytest.mark.parametrize("shares,expected", [(20.0, "above_limit"), (10.0, "within_limit"), (9.0, "within_limit")])
def test_user_ceiling_alone_determines_allocation(shares, expected):
    result = context(shares=shares, portfolio_value=10000, max_holding_pct=10)
    assert result["position"]["portfolio_weight_pct"] == shares
    assert result["concentration_check"]["status"] == expected
    assert result["concentration_check"]["threshold_basis"] == "user_supplied"
    assert result["review_status"] == ("review_allocation" if expected == "above_limit" else "no_allocation_trigger")


def test_no_portfolio_default_can_override_an_explicit_ceiling():
    result = context(shares=15, portfolio_value=10000, max_holding_pct=10)
    assert result["review_status"] == "review_allocation"
    result = context(shares=25, portfolio_value=10000, max_holding_pct=30)
    assert result["review_status"] == "no_allocation_trigger"
    result = context(shares=25, portfolio_value=10000)
    assert result["review_status"] == "not_requested"


def test_unrounded_weight_decides_a_boundary_that_display_rounding_hides():
    result = context(shares=10.000000001, portfolio_value=10000, max_holding_pct=10)
    assert result["position"]["portfolio_weight_pct"] == 10.0
    assert result["concentration_check"]["status"] == "above_limit"


@pytest.mark.parametrize("ceiling", [None, 100.0])
def test_inconsistent_total_is_not_clamped_or_treated_as_a_ceiling_breach(ceiling):
    result = context(shares=100.00000001, portfolio_value=10000, max_holding_pct=ceiling)
    assert result["context_status"] == {"status": "inconsistent", "reasons": ["holding_exceeds_portfolio"]}
    assert result["position"]["portfolio_weight_pct"] == 100.0
    assert result["review_status"] == ("not_requested" if ceiling is None else "not_evaluated")
    assert result["position"]["calculation_reasons"]["portfolio_weight_pct"] == []


def test_zero_cost_is_known_but_does_not_prevent_allocation_or_dollar_difference():
    result = context(cost_basis=0, shares=2.5, portfolio_value=10000, max_holding_pct=10)
    assert result["position"]["gain_loss"] is None
    assert result["position"]["gain_loss_dollars"] == 250.0
    assert result["position"]["calculation_reasons"]["gain_loss"] == ["zero_cost_basis"]
    assert result["context_status"] == {"status": "partial", "reasons": ["zero_cost_basis"]}
    assert result["review_status"] == "no_allocation_trigger"


@pytest.mark.parametrize("quote_type", ["EQUITY", "ETF"])
@pytest.mark.parametrize("rubric_status", ["supported", "unsupported", "unverified"])
def test_share_arithmetic_does_not_reuse_stock_rubric_or_statement_ratio_gates(quote_type, rubric_status):
    data = summary()
    data["support"]["instrument"]["status"] = rubric_status
    data["metric_applicability"] = {"pe": {"status": "unverified"}}
    result = context(data=data, quote_type=quote_type, shares=2.5, cost_basis=80,
                     portfolio_value=10000, max_holding_pct=10)
    assert result["position"]["current_value"] == 250
    assert result["position"]["gain_loss"] == 0.25
    assert result["review_status"] == "no_allocation_trigger"


@pytest.mark.parametrize("quote_type", [None, "OPTION", "FUTURE", "INDEX", "CURRENCY"])
def test_unknown_or_non_share_units_suppress_dependent_calculations(quote_type):
    result = context(quote_type=quote_type, shares=2.5, cost_basis=80, portfolio_value=10000,
                     max_holding_pct=10, purchase_date=date(2026, 10, 1))
    assert result["position"]["current_price"] is None
    for field in ("current_value", "gain_loss", "gain_loss_dollars", "portfolio_weight_pct"):
        assert result["position"][field] is None
        assert "share_units_unverified" in result["position"]["calculation_reasons"][field]
    assert result["holding"]["days_held"] == 18
    assert result["context_status"]["status"] == "partial"
    assert result["review_status"] == "not_evaluated"


@pytest.mark.parametrize("currency,status", [("EUR", "unsupported"), ("GBp", "unsupported"),
                                             (None, "unverified"), ("USD", "unverified")])
def test_currency_never_gets_implicit_fx(currency, status):
    data = summary()
    data["currency"] = currency
    data["support"]["currency"]["status"] = status
    result = context(data=data, shares=2.5, cost_basis=80, portfolio_value=10000, max_holding_pct=10)
    assert result["position"]["current_price"] is None
    assert result["position"]["gain_loss"] is None
    assert result["position"]["current_value"] is None
    assert result["position"]["gain_loss_dollars"] is None
    assert result["position"]["portfolio_weight_pct"] is None
    assert "usd_price_unverified" in result["context_status"]["reasons"]
    assert result["review_status"] == "not_evaluated"


@pytest.mark.parametrize("price", [None, 0, -1, float("inf"), float("nan"), True])
def test_invalid_selected_price_cannot_make_personal_amounts(price):
    result = context(data=summary(price), cost_basis=80, shares=2.5, portfolio_value=10000,
                     max_holding_pct=10)
    assert result["position"]["current_price"] is None
    assert result["position"]["current_value"] is None
    assert "price_unavailable" in result["context_status"]["reasons"]
    assert result["review_status"] == "not_evaluated"


@pytest.mark.parametrize("change", [
    {"price_source": "info_quote_fallback"},
    {"fallback_reason": "intraday_quote_unavailable"},
    {"staleness": {"stale": True}},
    {"staleness": None},
    {"price_as_of": None},
    {"price_source": "unknown"},
])
def test_historical_or_unverified_prices_keep_descriptive_math_but_not_allocation(change):
    data = summary()
    data["price_provenance"].update(change)
    result = context(data=data, cost_basis=80, shares=20, portfolio_value=10000, max_holding_pct=10)
    assert result["position"]["current_value"] == 2000
    assert result["position"]["gain_loss_dollars"] == 400
    assert result["position"]["price_context"]["status"] == "historical"
    assert result["position"]["price_provenance"] == data["price_provenance"]
    assert result["context_status"]["status"] == "partial"
    assert result["review_status"] == "not_evaluated"
    assert "price_not_current" in result["concentration_check"]["reasons"]


def test_date_alone_is_available_despite_unavailable_company_or_quote_data():
    result = context(data={}, quote_type=None, purchase_date=date(2024, 2, 29))
    assert result["context_status"] == {"status": "available", "reasons": []}
    assert result["holding"] == {"purchase_date": "2024-02-29", "days_held": 963}
    assert result["review_status"] == "not_requested"


def test_elapsed_calendar_days_have_no_tax_category_at_year_and_leap_boundaries():
    result = context(today=date(2025, 2, 28), purchase_date=date(2024, 2, 29))
    assert result["holding"] == {"purchase_date": "2024-02-29", "days_held": 365}
    result = context(today=date(2025, 3, 1), purchase_date=date(2024, 2, 29))
    assert result["holding"] == {"purchase_date": "2024-02-29", "days_held": 366}


@pytest.mark.parametrize("shares", [1e308, 1e-308])
def test_unrepresentable_product_is_named_not_infinity_zero_or_inconsistent(shares):
    price = 1e308 if shares > 1 else 1e-308
    result = context(data=summary(price), cost_basis=price, shares=shares,
                     portfolio_value=10000, max_holding_pct=10)
    assert result["position"]["current_value"] is None
    assert result["position"]["gain_loss"] == 0.0
    assert result["position"]["gain_loss_dollars"] == 0.0
    assert result["position"]["portfolio_weight_pct"] is None
    assert result["position"]["calculation_reasons"]["current_value"] == ["numeric_range_unavailable"]
    assert result["context_status"] == {"status": "partial", "reasons": ["numeric_range_unavailable"]}
    assert result["review_status"] == "not_evaluated"


def test_unrepresentable_return_does_not_hide_a_representable_holding():
    result = context(data=summary(1e308), cost_basis=1e-308, shares=1e-308,
                     portfolio_value=10, max_holding_pct=20)
    assert result["position"]["gain_loss"] is None
    assert result["position"]["current_value"] == 1
    assert result["position"]["gain_loss_dollars"] == 1
    assert result["position"]["portfolio_weight_pct"] == 10
    assert result["review_status"] == "no_allocation_trigger"
    assert result["context_status"] == {"status": "partial", "reasons": ["numeric_range_unavailable"]}


def test_unrepresentable_weight_does_not_hide_representable_value():
    result = context(shares=1, portfolio_value=1e-308, max_holding_pct=10)
    assert result["position"]["current_value"] == 100
    assert result["position"]["portfolio_weight_pct"] is None
    assert "numeric_range_unavailable" in result["position"]["calculation_reasons"]["portfolio_weight_pct"]
    # The holding and portfolio total themselves remain representable.
    assert result["context_status"]["status"] == "inconsistent"
    assert "holding_exceeds_portfolio" in result["context_status"]["reasons"]
    assert result["review_status"] == "not_evaluated"


def test_context_never_mutates_the_analysis_or_inputs():
    data = summary()
    original = deepcopy(data)
    result = context(data=data, shares=20, cost_basis=80, portfolio_value=10000, max_holding_pct=10)
    result["position"]["price_provenance"]["staleness"]["stale"] = True
    assert data == original
    assert set(result) == {"position", "holding", "context_status", "concentration_check",
                           "review_status", "assumptions"}
