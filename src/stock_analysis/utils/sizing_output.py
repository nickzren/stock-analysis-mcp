"""Adapt exact sizing to numeric JSON; never round a ceiling upward."""

from copy import deepcopy
from fractions import Fraction
from math import isfinite, nextafter
from typing import Any, cast

from stock_analysis.utils.position_sizing import SizingResult

LOSS_DISCLOSURE = "Estimated loss if sold at this stop price. Gaps and slippage can make the actual loss larger."


def _wire(value: Any) -> Any:
    if isinstance(value, Fraction):
        shown = float(value)
        if not isfinite(shown) or value > 0 and shown == 0:
            raise ValueError("numeric_range_unavailable")
        # JSON uses the decimal spelling of a float. Compare that spelling with
        # the exact result, not the float's internal binary approximation.
        if Fraction(str(shown)) > value:
            shown = nextafter(shown, float('-inf'))
        if value > 0 and shown <= 0:
            raise ValueError("numeric_range_unavailable")
        return shown
    if isinstance(value, dict):
        for name in ('position_dollars', 'planned_loss_at_stop'):
            amount = value.get(name)
            if isinstance(amount, Fraction) and Fraction(str(float(amount))) != amount:
                raise ValueError("numeric_range_unavailable")
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    return value


def public_sizing(
    result: SizingResult, *, horizon: str = 'mid_term', price_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert once after quantity calculation; an unrepresentable value is unknown.

    Monetary outputs keep sub-cent precision. Display prose is separate. Values
    outside the JSON float range must not become zero risk or infinity.
    """
    try:
        public = _wire(result)
    except (OverflowError, ValueError):
        return {'sizing_status': {'status': 'unavailable', 'reason': 'numeric_range_unavailable'}, 'sizing': None}
    if public['sizing'] is not None:
        public['sizing']['horizon'] = horizon
        public['sizing']['loss_disclosure'] = LOSS_DISCLOSURE
        if price_provenance is not None:
            public['sizing']['assumptions']['price_provenance'] = deepcopy(price_provenance)
    return cast(dict[str, Any], public)


def cap_sizing_scenario(
    sizing: dict[str, Any], name: str, cap_pct: float, constraint: str,
) -> dict[str, Any] | None:
    """A mode may reduce a scenario; it cannot replace its risk or increase it."""
    original = sizing.get(name)
    if original is None:
        return None
    if cap_pct >= original['allocation_cap_pct']:
        return cast(dict[str, Any], deepcopy(original))
    cap = Fraction(str(cap_pct))
    assumptions = sizing['assumptions']
    entry = Fraction(str(assumptions['entry_price']))
    loss_per_share = entry - Fraction(str(assumptions['stop_price']))
    row = deepcopy(original)
    row.update(allocation_cap_pct=cap, ceiling_risk_pct=cap * loss_per_share / entry,
               binding_constraints=[constraint])
    if assumptions['account_size'] is not None:
        account = Fraction(str(assumptions['account_size']))
        allocation = account * cap / 100
        permitted = allocation / entry
        quantity = min(original['whole_shares'], permitted.numerator // permitted.denominator)
        row.update(allocation_cap_dollars=allocation, whole_shares=quantity,
                   position_dollars=quantity * entry, planned_loss_at_stop=quantity * loss_per_share,
                   position_pct=100 * quantity * entry / account,
                   effective_risk_pct=100 * quantity * loss_per_share / account,
                   quantity_status='available' if quantity else 'below_one_whole_share',
                   limitations=['whole_share_granularity'] if quantity < permitted else [])
    try:
        return cast(dict[str, Any], _wire(row))
    except (OverflowError, ValueError):
        return None
