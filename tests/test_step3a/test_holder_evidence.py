"""The holder contract reuses existing evidence without inventing dates or failures."""

import copy
from dataclasses import replace

import pytest

from stock_analysis.tools.analyze.holder import build_holder_evidence, long_term_if_held
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm
from tests.support.output_contract import registry_violations


@pytest.mark.parametrize('case', hm.STOCK_CASES)
async def test_holder_fields_conform_at_every_detail_level(case):
    result = await hm.run_stock(case)
    for tier, payload in dm.tiers(result).items():
        assert not registry_violations(payload, tier), case


async def test_nonburner_and_profitable_runway_exemptions_are_not_confused():
    nonburner = (await hm.run_stock('nonburner_no_cash'))['decision_card']['long_term']['if_held']
    negative = (await hm.run_stock('confirmed_negative_fcf'))['decision_card']['long_term']['if_held']
    for held in (nonburner, negative):
        assert held['action'] == 'hold'
        assert held['checks']['cash_runway']['state'] == 'not_applicable'
    assert 'not negative' in nonburner['checks']['cash_runway']['reason']
    assert 'does not assert positive' in negative['checks']['cash_runway']['reason']
    assert negative['checks']['business']['evidence']['free_cash_flow_ttm'] < 0
    assert any(f['code'] == 'cash_flow' for f in negative['watch_factors'])


async def test_failed_source_names_the_failed_read_not_missing_company_metrics():
    held = (await hm.run_stock('fundamentals_failed'))['decision_card']['long_term']['if_held']
    for check in held['checks'].values():
        assert check['state'] == 'not_evaluated'
        assert check['reason'] == 'Fundamental source unavailable; required checks could not be run.'
        assert check['evidence'] == {}
    assert held['watch_factors'] == []


async def test_periods_belong_to_individual_inputs_not_composite_or_fetch_time():
    case = hm.STOCK_CASES['critical_runway']
    fund = copy.deepcopy(case.fundamentals)
    fund['cash_flow'].update(free_cash_flow_period='TTM', free_cash_flow_period_end='2026-06-30',
                             free_cash_flow_source='cashflow_quarterly', free_cash_flow_period_lag=0,
                             free_cash_flow_ttm_status='complete',
                             operating_cf_period='annual', operating_cf_period_end='2025-12-31',
                             operating_cf_source='cashflow_yearly', operating_cf_period_lag=1,
                             operating_cf_ttm_status='ttm_incomplete')
    fund['provenance'] = {'as_of': '2099-01-01', 'fiscal_period': '2099-12-31'}
    result = await dm.run(replace(case, fundamentals=fund))
    held = result['decision_card']['long_term']['if_held']
    periods = held['checks']['cash_runway']['periods']
    assert periods['free_cash_flow_ttm'] == {'period': 'TTM', 'period_end': '2026-06-30',
        'source': 'cashflow_quarterly', 'period_lag': 0, 'ttm_status': 'complete'}
    assert periods['operating_cf_ttm'] == {'period': 'annual', 'period_end': '2025-12-31',
        'source': 'cashflow_yearly', 'period_lag': 1, 'ttm_status': 'ttm_incomplete'}
    assert periods['liquidity'] is None
    assert periods['cash_runway_quarters'] is None
    assert held['checks']['business']['periods']['net_margin'] is None
    assert '2099' not in str(held)
    assert 'no composite period' in held['checks']['cash_runway']['periods_reason']


async def test_missing_periods_stay_unknown():
    held = (await hm.run_stock('sound_bullish'))['decision_card']['long_term']['if_held']
    for check in held['checks'].values():
        assert check['periods'] is None
        assert 'Input-specific periods unavailable' in check['periods_reason']


async def test_rounded_four_quarter_runway_does_not_hide_a_known_failure():
    case = hm.STOCK_CASES['critical_runway']
    fund = copy.deepcopy(case.fundamentals)
    fund['financial_health']['cash_and_st_investments'] = 199_500_000
    result = await dm.run(replace(case, fundamentals=fund))
    assert result['fundamentals_summary']['burn_metrics']['cash_runway_quarters'] == 4.0
    held = result['decision_card']['long_term']['if_held']
    assert held['action'] == 'review_exit'
    assert held['basis'] == 'runway_critical'
    assert held['checks']['cash_runway']['evidence']['cash_runway_quarters'] == 3.99


@pytest.mark.parametrize('case,expected', [('net_debt', 'net_debt'), ('negative_equity', 'negative_equity'),
                                        ('watch_runway', 'runway_under_8q')])
async def test_known_concerns_keep_their_evidence_without_becoming_automatic_exits(case, expected):
    held = (await hm.run_stock(case))['decision_card']['long_term']['if_held']
    assert held['action'] == 'hold'
    factor = next(item for item in held['watch_factors'] if item['code'] == expected)
    assert factor['evidence'] and all(value is not None for value in factor['evidence'].values())
    assert 'net_cash_unknown' not in [item['code'] for item in held['watch_factors']]


def test_absent_assessability_flags_are_not_true_defaults():
    contract = build_holder_evidence(support={'instrument': {'status': 'supported'}},
        fundamentals_status='available', fundamentals={}, source={},
        signals={'business_status': 'intact', 'balance_status': 'safe'})
    held = long_term_if_held(contract)
    assert held['action'] == 'insufficient_data'
    assert held['reasons'] == ['business_not_evaluated', 'cash_runway_not_evaluated']
