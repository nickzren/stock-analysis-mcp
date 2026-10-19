"""Numeric wire precision and mode caps adapt the exact sizing calculation."""
from fractions import Fraction

import pytest

from stock_analysis.utils.position_sizing import calculate_position_sizing
from stock_analysis.utils.sizing_output import cap_sizing_scenario, public_sizing


def exact(**overrides):
    return calculate_position_sizing(**({"entry_price": 100, "stop_price": 80, "risk_per_trade_pct": 1,
        "max_position_pct": 10, "regime_max_pct": 8, "regime_starter_pct": 3, "preset_starter_pct": 2.5,
        "account_size": 10_000} | overrides))


def test_wire_numbers_preserve_whole_share_amounts_and_subcent_prices():
    result = public_sizing(exact(entry_price=.0301, stop_price=.01, account_size=1))
    size = result['sizing']
    for name in ('starter', 'full'):
        row = size[name]
        assert row['position_dollars'] == pytest.approx(row['whole_shares'] * .0301)
        assert row['planned_loss_at_stop'] == pytest.approx(row['whole_shares'] * .0201)
        assert row['planned_loss_at_stop'] <= size['limits']['risk_budget_dollars']
        assert row['position_dollars'] <= row['allocation_cap_dollars']
    assert size['horizon'] == 'mid_term'
    assert 'Gaps and slippage' in size['loss_disclosure']


@pytest.mark.parametrize(('stop', 'risk', 'cap'), [(70, .1, Fraction(1, 3)), (89, .01, Fraction(1, 11))])
def test_nonterminating_caps_are_never_rounded_above_the_exact_limit(stop, risk, cap):
    original = exact(entry_price=100, stop_price=stop, risk_per_trade_pct=risk)
    shown = public_sizing(original)['sizing']
    assert 0 < Fraction(str(shown['full']['allocation_cap_pct'])) <= cap
    assert shown['limits']['risk_per_trade_pct'] == risk


def test_unrepresentable_positive_wire_value_is_unavailable_not_zero():
    result = public_sizing(exact(account_size=1e-300, risk_per_trade_pct=1e-100))
    assert result['sizing'] is None
    assert result['sizing_status']['reason'] == 'numeric_range_unavailable'


def test_mode_caps_only_reduce_the_original_scenario():
    original = public_sizing(exact())['sizing']
    smaller = cap_sizing_scenario(original, 'full', 2, 'mode_maximum')
    assert smaller['whole_shares'] == 2
    assert smaller['position_dollars'] == 200
    assert smaller['planned_loss_at_stop'] == 40
    assert smaller['binding_constraints'] == ['mode_maximum']
    assert cap_sizing_scenario(original, 'full', 8, 'mode_maximum') == original['full']
    assert original['full']['whole_shares'] == 5


def test_same_ceiling_does_not_refloor_an_exact_boundary():
    original = public_sizing(exact(entry_price=3, stop_price=2, account_size=300,
                                  risk_per_trade_pct=1/3))['sizing']
    assert cap_sizing_scenario(original, 'full', 100, 'mode_maximum') == original['full']


def test_no_account_mode_caps_stay_percentage_only():
    original = public_sizing(exact(account_size=None))['sizing']
    row = cap_sizing_scenario(original, 'starter', 1, 'mode_starter')
    assert row['allocation_cap_pct'] == 1
    assert row['quantity_status'] == 'percentage_only'
    assert row['whole_shares'] is row['position_dollars'] is row['planned_loss_at_stop'] is None


def test_inexact_monetary_wire_products_are_unavailable_instead_of_rounded():
    result = public_sizing(exact(entry_price=1.2345678901234567, stop_price=.99, account_size=100))
    assert result['sizing'] is None
    assert result['sizing_status']['reason'] == 'numeric_range_unavailable'
