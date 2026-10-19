"""Compute one mid-term sizing result after the canonical research decision."""

from decimal import Decimal
from math import isfinite
from numbers import Real
from typing import Any, cast

from stock_analysis.tools.analyze.action_zones import _base_sizing_range
from stock_analysis.tools.analyze.decision_card import adapt_card_sizing
from stock_analysis.utils.position_sizing import LimitOrigin, calculate_position_sizing
from stock_analysis.utils.sizing_output import public_sizing


def _market_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if isfinite(number) else None


def build_analysis_sizing(
    *, summary: dict[str, Any], action_zones: dict[str, Any], investor_profile: dict[str, Any],
    risk_data: dict[str, Any], risk_regime: dict[str, Any], decision: dict[str, Any],
) -> dict[str, Any]:
    """Market gaps suppress estimates, never the action, holder answer or snapshot."""
    new_money = decision['new_money']
    if new_money['action'] not in ('buy', 'starter'):
        return adapt_card_sizing({'sizing': None, 'sizing_status': {}}, new_money)
    raw = {
        'entry_price': summary.get('current_price'),
        'stop_price': (action_zones.get('stop_calculation') or {}).get('stop_price'),
        'atr': (risk_data.get('atr') or {}).get('value'),
    }
    values = {name: _market_number(value) for name, value in raw.items()}
    for name, value in values.items():
        if value is None or value <= 0:
            reason = f'{name}_missing' if raw[name] is None else f'{name}_invalid'
            return adapt_card_sizing({'sizing': None, 'sizing_status': {
                'status': 'unavailable', 'reason': reason}}, new_money)
    regime = risk_regime.get('classification')
    if regime == 'moderate':
        regime = 'medium'
    starter, maximum, _ = _base_sizing_range(regime)
    origins = investor_profile['limit_origins']
    exact = calculate_position_sizing(
        entry_price=values['entry_price'], stop_price=values['stop_price'], atr=values['atr'],
        stop_basis='atr', account_size=investor_profile.get('account_size'),
        risk_per_trade_pct=investor_profile['risk_per_trade_pct'],
        max_position_pct=investor_profile['max_position_pct'],
        regime_max_pct=maximum, regime_starter_pct=starter,
        preset_starter_pct=investor_profile['starter_position_pct'],
        risk_origin=cast(LimitOrigin, origins['risk_per_trade_pct']),
        max_position_origin=cast(LimitOrigin, origins['max_position_pct']),
    )
    public = public_sizing(exact, price_provenance=summary.get('price_provenance'))
    return adapt_card_sizing(public, new_money)
