"""Consumer invariants independent of the regenerated after recording."""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from stock_analysis.tools.analyze.investor_profile import resolve_investor_profile
from stock_analysis.tools.analyze.sizing import build_analysis_sizing
from tests.support import decision_matrix as dm
from tests.support import sizing_matrix as sm


def result(entry=100, stop=80, atr=10, **profile):
    return build_analysis_sizing(
        summary={'current_price': entry}, action_zones={'stop_calculation': {'stop_price': stop}},
        investor_profile=resolve_investor_profile(account_size=10_000, **profile),
        risk_data={'atr': {'value': atr}}, risk_regime={'classification': 'medium'},
        decision={'new_money': {'action': 'buy', 'blocked_by': []}})


@pytest.mark.parametrize('scalar', [np.float32, np.float64, np.int64, Decimal])
def test_provider_numeric_scalars_reach_the_same_calculation(scalar):
    assert result(scalar(100), scalar(80), scalar(10)) == result()


@pytest.mark.parametrize(('kwargs', 'reason'), [
    ({'entry': None}, 'entry_price_missing'), ({'entry': False}, 'entry_price_invalid'),
    ({'stop': None}, 'stop_price_missing'), ({'stop': -1}, 'stop_price_invalid'),
    ({'stop': 100}, 'stop_not_below_entry'), ({'atr': None}, 'atr_missing'),
    ({'atr': 0}, 'atr_invalid'), ({'atr': float('nan')}, 'atr_invalid'),
    ({'atr': float('inf')}, 'atr_invalid'),
])
def test_market_input_reasons_distinguish_missing_invalid_and_geometry(kwargs, reason):
    assert result(**kwargs) == {'sizing': None, 'sizing_status': {'status': 'unavailable', 'reason': reason}}


@pytest.mark.parametrize('name', sm.CASES)
async def test_every_quantity_obeys_limits_and_exact_stated_money(name):
    raw = await sm.run_case(name)
    for tier, payload in dm.tiers(raw).items():
        card = payload['decision_card']
        sizing = card['sizing']
        if sizing is None:
            assert card['sizing_status']['status'] in ('not_allowed', 'unavailable')
            continue
        a = sizing['assumptions']
        limits = sizing['limits']
        entry, stop = Decimal(str(a['entry_price'])), Decimal(str(a['stop_price']))
        for key in ('starter', 'full'):
            row = sizing[key]
            if row is None:
                assert card['new_money']['action'] == 'starter'
                continue
            assert row['allocation_cap_pct'] <= limits['max_position_pct']
            assert row['ceiling_risk_pct'] <= limits['risk_per_trade_pct']
            if a['account_size'] is None:
                assert row['whole_shares'] is row['position_dollars'] is row['planned_loss_at_stop'] is None
                continue
            q = row['whole_shares']
            assert Decimal(str(row['position_dollars'])) == q * entry, (name, tier)
            assert Decimal(str(row['planned_loss_at_stop'])) == q * (entry - stop), (name, tier)
            assert row['planned_loss_at_stop'] <= limits['risk_budget_dollars']
            assert row['position_dollars'] <= row['allocation_cap_dollars']
            assert row['position_pct'] <= row['allocation_cap_pct']


def test_smallest_positive_budget_is_not_displayed_as_zero_or_replaced():
    size = result(risk_per_trade_pct=1e-323)['sizing']
    assert size['limits']['risk_per_trade_pct'] == 1e-323
    assert size['limits']['risk_budget_dollars'] > 0
    assert 0 < Fraction(str(size['full']['allocation_cap_pct'])) <= Fraction('5e-323')
    assert size['full']['whole_shares'] == 0
