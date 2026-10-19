"""Exact, pure sizing contract; public adapters are a separate work package."""

from fractions import Fraction

import pytest

from stock_analysis.utils.position_sizing import calculate_position_sizing


def size(**overrides):
    inputs = {
        "entry_price": 100,
        "stop_price": 80,
        "account_size": 10_000,
        "risk_per_trade_pct": 1,
        "max_position_pct": 10,
        "regime_max_pct": 8,
        "regime_starter_pct": 2.5,
        "preset_starter_pct": 3,
    }
    return calculate_position_sizing(**(inputs | overrides))


def scenarios(result):
    assert result["sizing"] is not None
    return result["sizing"]["starter"], result["sizing"]["full"]


@pytest.mark.parametrize(
    ("overrides", "cap", "quantity", "loss"),
    [
        ({}, 5, 5, 100),
        ({"risk_per_trade_pct": 0.25, "risk_origin": "caller"}, 1.25, 1, 20),
        ({"max_position_pct": 2, "max_position_origin": "caller"}, 2, 2, 40),
        ({"max_position_pct": 8, "max_position_origin": "caller"}, 5, 5, 100),
    ],
)
def test_default_and_single_override_worked_examples(overrides, cap, quantity, loss):
    result = size(**overrides)
    starter, full = scenarios(result)
    assert result["sizing_status"] == {"status": "available", "reason": None}
    assert full["allocation_cap_pct"] == Fraction(str(cap))
    assert full["whole_shares"] == quantity
    assert full["position_dollars"] == quantity * 100
    assert full["planned_loss_at_stop"] == loss
    assert starter["whole_shares"] <= quantity


def test_risk_only_example_preserves_limit_origins_and_dollars():
    result = size(risk_per_trade_pct=0.25, risk_origin="caller")
    block = result["sizing"]
    assert block is not None
    assert block["limits"]["risk_per_trade_pct"] == Fraction(1, 4)
    assert block["limits"]["risk_origin"] == "caller"
    assert block["limits"]["max_position_pct"] == 10
    assert block["limits"]["max_position_origin"] == "profile"
    assert block["limits"]["risk_budget_dollars"] == 25
    for scenario in scenarios(result):
        assert scenario["allocation_cap_dollars"] == 125
        assert scenario["whole_shares"] == 1
        assert scenario["position_dollars"] == 100
        assert scenario["planned_loss_at_stop"] == 20
        assert scenario["position_pct"] == 1
        assert scenario["effective_risk_pct"] == Fraction(1, 5)
        assert scenario["binding_constraints"] == ("caller_risk_budget",)
        assert scenario["limitations"] == ("whole_share_granularity",)


def test_small_account_has_independent_starter_and_full_quantity_status():
    result = size(account_size=1000, entry_price=50, stop_price=48)
    starter, full = scenarios(result)
    assert result["sizing_status"]["status"] == "available"
    assert starter["quantity_status"] == "below_one_whole_share"
    assert starter["whole_shares"] == starter["position_dollars"] == 0
    assert starter["planned_loss_at_stop"] == starter["effective_risk_pct"] == 0
    assert full["quantity_status"] == "available"
    assert full["whole_shares"] == 1
    assert full["position_dollars"] == 50
    assert full["planned_loss_at_stop"] == 2


def test_no_account_keeps_exact_ceilings_without_inventing_amounts():
    result = size(account_size=None)
    block = result["sizing"]
    assert block is not None
    assert result["sizing_status"] == {"status": "percentage_only", "reason": None}
    assert block["limits"]["risk_budget_dollars"] is None
    for scenario in scenarios(result):
        assert scenario["quantity_status"] == "percentage_only"
        for field in (
            "allocation_cap_dollars",
            "whole_shares",
            "position_dollars",
            "planned_loss_at_stop",
            "position_pct",
            "effective_risk_pct",
        ):
            assert scenario[field] is None
        assert scenario["ceiling_risk_pct"] == scenario["allocation_cap_pct"] / 5
        assert scenario["limitations"] == ()


@pytest.mark.parametrize(
    ("overrides", "full_bindings", "starter_bindings"),
    [
        ({}, ("profile_risk_budget",), ("regime_starter",)),
        ({"stop_price": 99}, ("regime_maximum",), ("regime_starter",)),
        (
            {"max_position_pct": 2, "max_position_origin": "caller"},
            ("caller_max_position",),
            ("caller_max_position",),
        ),
        ({"preset_starter_pct": 1}, ("profile_risk_budget",), ("preset_starter",)),
        (
            {
                "stop_price": 90,
                "regime_max_pct": 10,
                "regime_starter_pct": 10,
                "preset_starter_pct": 10,
                "max_position_origin": "caller",
                "risk_origin": "default",
            },
            ("default_risk_budget", "caller_max_position", "regime_maximum"),
            (
                "default_risk_budget",
                "caller_max_position",
                "regime_maximum",
                "regime_starter",
                "preset_starter",
            ),
        ),
    ],
)
def test_binding_constraints_preserve_ties_and_origins(overrides, full_bindings, starter_bindings):
    starter, full = scenarios(size(**overrides))
    assert full["binding_constraints"] == full_bindings
    assert starter["binding_constraints"] == starter_bindings


def test_tiny_inputs_are_not_rounded_or_replaced():
    result = size(
        risk_per_trade_pct=0.001, max_position_pct=0.06, entry_price=0.01, stop_price=0.008
    )
    block = result["sizing"]
    assert block is not None
    assert block["limits"]["risk_per_trade_pct"] == Fraction(1, 1000)
    assert block["limits"]["max_position_pct"] == Fraction(3, 50)
    for scenario in scenarios(result):
        assert scenario["allocation_cap_pct"] == Fraction(1, 200)
        assert scenario["whole_shares"] == 50
        assert scenario["position_dollars"] == Fraction(1, 2)
        assert scenario["planned_loss_at_stop"] == Fraction(1, 10)


def test_subcent_money_uses_exact_stated_prices_and_same_quantity():
    result = size(
        account_size=1,
        entry_price=0.003,
        stop_price=0.002,
        risk_per_trade_pct=0.7,
        max_position_pct=100,
        regime_max_pct=100,
        regime_starter_pct=100,
        preset_starter_pct=100,
    )
    starter, full = scenarios(result)
    assert starter == full
    assert full["whole_shares"] == 7
    assert full["position_dollars"] == Fraction(21, 1000)
    assert full["planned_loss_at_stop"] == Fraction(7, 1000)
    assert full["effective_risk_pct"] == Fraction(7, 10)
    assert full["limitations"] == ()


def test_repeating_percent_does_not_lose_a_valid_whole_share():
    result = size(
        account_size=3,
        entry_price=1,
        stop_price=0.97,
        max_position_pct=100,
        regime_max_pct=100,
        regime_starter_pct=100,
        preset_starter_pct=100,
    )
    for scenario in scenarios(result):
        assert scenario["allocation_cap_pct"] == Fraction(100, 3)
        assert scenario["whole_shares"] == 1
        assert scenario["allocation_cap_dollars"] == scenario["position_dollars"] == 1


def test_products_remain_exact_beyond_float_precision():
    entry = 0.12345678901234566
    stop = 0.02
    result = size(
        account_size=1,
        entry_price=entry,
        stop_price=stop,
        risk_per_trade_pct=100,
        max_position_pct=40,
        regime_max_pct=100,
        regime_starter_pct=100,
        preset_starter_pct=100,
    )
    for scenario in scenarios(result):
        assert scenario["whole_shares"] == 3
        assert scenario["position_dollars"] == Fraction("0.37037036703703698")
        assert scenario["planned_loss_at_stop"] == 3 * (Fraction(str(entry)) - Fraction(str(stop)))


@pytest.mark.parametrize(
    "entry,stop", [(0.3, 0.2), (0.01, 0.009), (100, 99.99), (100, 80), (0.12345678901234566, 0.02)]
)
@pytest.mark.parametrize("budget", [0.001, 0.25, 1, 100])
def test_exact_amounts_obey_both_limits(entry, stop, budget):
    result = size(entry_price=entry, stop_price=stop, risk_per_trade_pct=budget)
    block = result["sizing"]
    assert block is not None
    assumptions = block["assumptions"]
    for scenario in scenarios(result):
        quantity = scenario["whole_shares"]
        assert isinstance(quantity, int)
        position = scenario["position_dollars"]
        loss = scenario["planned_loss_at_stop"]
        assert position == quantity * assumptions["entry_price"]
        assert loss == quantity * (assumptions["entry_price"] - assumptions["stop_price"])
        assert position <= scenario["allocation_cap_dollars"]
        assert loss <= block["limits"]["risk_budget_dollars"]
        assert scenario["position_pct"] <= scenario["allocation_cap_pct"]
        assert scenario["effective_risk_pct"] <= block["limits"]["risk_per_trade_pct"]
        assert quantity + 1 > min(
            scenario["allocation_cap_dollars"] / assumptions["entry_price"],
            block["limits"]["risk_budget_dollars"]
            / (assumptions["entry_price"] - assumptions["stop_price"]),
        )


def test_lower_limits_and_wider_stop_never_increase_quantity():
    base = scenarios(size())
    for changed in (
        {"risk_per_trade_pct": 0.2},
        {"max_position_pct": 1},
        {"stop_price": 50},
        {"regime_max_pct": 1},
        {"regime_starter_pct": 0.1},
        {"preset_starter_pct": 0.1},
    ):
        actual = scenarios(size(**changed))
        for baseline, scenario in zip(base, actual, strict=True):
            assert scenario["whole_shares"] <= baseline["whole_shares"]
        assert actual[0]["whole_shares"] <= actual[1]["whole_shares"]


@pytest.mark.parametrize("field", ["entry_price", "stop_price"])
@pytest.mark.parametrize("value", [None, True, False, 0, -1, float("nan"), float("inf"), "100"])
def test_invalid_market_price_returns_explicit_unavailable(field, value):
    result = size(**{field: value})
    assert result == {
        "sizing_status": {"status": "unavailable", "reason": f"{field}_unavailable"},
        "sizing": None,
    }


@pytest.mark.parametrize("stop", [100, 101])
def test_stop_must_be_strictly_below_entry(stop):
    result = size(stop_price=stop)
    assert result["sizing"] is None
    assert result["sizing_status"]["reason"] == "stop_not_below_entry"


@pytest.mark.parametrize("atr", [None, True, 0, -1, float("nan"), float("inf")])
def test_atr_basis_requires_a_finite_positive_atr(atr):
    result = size(stop_basis="atr", atr=atr)
    assert result["sizing"] is None
    assert result["sizing_status"]["reason"] == "atr_unavailable"


def test_provided_stop_needs_no_atr_and_actual_selected_stop_is_authoritative():
    result = size(stop_basis="provided")
    assert result["sizing"] is not None
    actual = size(stop_basis="atr", atr=2, stop_price=95.99)["sizing"]
    assert actual is not None
    assert actual["assumptions"]["stop_price"] == Fraction("95.99")
    assert actual["assumptions"]["atr"] == 2
    assert actual["assumptions"]["stop_distance_fraction"] == Fraction(401, 10_000)


@pytest.mark.parametrize(
    "field",
    [
        "risk_per_trade_pct",
        "max_position_pct",
        "regime_max_pct",
        "regime_starter_pct",
        "preset_starter_pct",
    ],
)
@pytest.mark.parametrize("value", [True, False, 0, -1, 101, None, "1", float("nan"), float("inf")])
def test_invalid_validated_percentage_is_a_caller_error(field, value):
    with pytest.raises(ValueError, match=field):
        size(**{field: value})


@pytest.mark.parametrize("value", [True, False, 0, -1, "100", float("nan"), float("inf")])
def test_invalid_account_is_a_caller_error(value):
    with pytest.raises(ValueError, match="account_size"):
        size(account_size=value)


@pytest.mark.parametrize("field", ["risk_origin", "max_position_origin", "stop_basis"])
def test_invalid_enum_is_a_caller_error(field):
    with pytest.raises(ValueError, match=field):
        size(**{field: "made_up"})
