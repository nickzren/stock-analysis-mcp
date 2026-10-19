"""Pure whole-share sizing with exact arithmetic and no research-action policy.

All numeric results are internal ``Fraction`` values (quantities are integers),
not a public JSON payload. Adapters must handle wire/display precision separately;
rounding a cap before flooring quantity can discard a valid whole share.
Inputs are already normalized numbers. Percentage inputs use percentage points.
"""

from fractions import Fraction
from math import isfinite
from typing import Literal, TypedDict

LimitOrigin = Literal["caller", "profile", "default"]
StopBasis = Literal["provided", "atr"]
Number = int | float


class SizingStatus(TypedDict):
    status: Literal["available", "percentage_only", "unavailable"]
    reason: str | None


class SizingAssumptions(TypedDict):
    entry_price: Fraction
    stop_price: Fraction
    stop_distance_fraction: Fraction
    stop_basis: StopBasis
    atr: Fraction | None
    account_size: Fraction | None


class SizingLimits(TypedDict):
    risk_per_trade_pct: Fraction
    max_position_pct: Fraction
    risk_origin: LimitOrigin
    max_position_origin: LimitOrigin
    regime_max_pct: Fraction
    regime_starter_pct: Fraction
    preset_starter_pct: Fraction
    risk_cap_pct: Fraction
    risk_budget_dollars: Fraction | None


class SizingScenario(TypedDict):
    quantity_status: Literal["available", "percentage_only", "below_one_whole_share"]
    allocation_cap_pct: Fraction
    allocation_cap_dollars: Fraction | None
    ceiling_risk_pct: Fraction
    whole_shares: int | None
    position_dollars: Fraction | None
    planned_loss_at_stop: Fraction | None
    position_pct: Fraction | None
    effective_risk_pct: Fraction | None
    binding_constraints: tuple[str, ...]
    limitations: tuple[str, ...]


class PositionSizing(TypedDict):
    assumptions: SizingAssumptions
    limits: SizingLimits
    starter: SizingScenario
    full: SizingScenario


class SizingResult(TypedDict):
    sizing_status: SizingStatus
    sizing: PositionSizing | None


def _positive_number(value: object) -> Fraction | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0 or isinstance(value, float) and not isfinite(value):
        return None
    return Fraction(str(value))


def _percentage(value: object, name: str) -> Fraction:
    number = _positive_number(value)
    if number is None or number > 100:
        raise ValueError(f"{name} must be a finite number in (0, 100]")
    return number


def _unavailable(reason: str) -> SizingResult:
    return {"sizing_status": {"status": "unavailable", "reason": reason}, "sizing": None}


def _scenario(
    constraints: list[tuple[str, Fraction]],
    entry: Fraction,
    loss_per_share: Fraction,
    account: Fraction | None,
    risk_budget: Fraction | None,
) -> SizingScenario:
    cap = min(value for _, value in constraints)
    result: SizingScenario = {
        "quantity_status": "percentage_only",
        "allocation_cap_pct": cap,
        "allocation_cap_dollars": None,
        "ceiling_risk_pct": cap * loss_per_share / entry,
        "whole_shares": None,
        "position_dollars": None,
        "planned_loss_at_stop": None,
        "position_pct": None,
        "effective_risk_pct": None,
        "binding_constraints": tuple(name for name, value in constraints if value == cap),
        "limitations": (),
    }
    if account is None or risk_budget is None:
        return result

    allocation = account * cap / 100
    permitted_quantity = min(allocation / entry, risk_budget / loss_per_share)
    quantity = permitted_quantity.numerator // permitted_quantity.denominator
    position = quantity * entry
    loss = quantity * loss_per_share
    result["quantity_status"] = "available" if quantity else "below_one_whole_share"
    result["allocation_cap_dollars"] = allocation
    result["whole_shares"] = quantity
    result["position_dollars"] = position
    result["planned_loss_at_stop"] = loss
    result["position_pct"] = 100 * position / account
    result["effective_risk_pct"] = 100 * loss / account
    result["limitations"] = ("whole_share_granularity",) if quantity < permitted_quantity else ()
    return result


def calculate_position_sizing(
    *,
    entry_price: Number | None,
    stop_price: Number | None,
    risk_per_trade_pct: Number,
    max_position_pct: Number,
    regime_max_pct: Number,
    regime_starter_pct: Number,
    preset_starter_pct: Number,
    account_size: Number | None = None,
    risk_origin: LimitOrigin = "profile",
    max_position_origin: LimitOrigin = "profile",
    stop_basis: StopBasis = "provided",
    atr: Number | None = None,
) -> SizingResult:
    """Calculate independent starter/full scenarios from effective limits.

    Caller/configuration invariant failures raise ``ValueError``. Missing or
    invalid market inputs produce explicit unavailability, never a guessed size.
    A supplied stop is authoritative; this function does not reconstruct it from
    ATR. ATR must be valid only when the caller names it as the stop basis.
    """
    risk = _percentage(risk_per_trade_pct, "risk_per_trade_pct")
    maximum = _percentage(max_position_pct, "max_position_pct")
    regime_maximum = _percentage(regime_max_pct, "regime_max_pct")
    regime_starter = _percentage(regime_starter_pct, "regime_starter_pct")
    preset_starter = _percentage(preset_starter_pct, "preset_starter_pct")
    account = _positive_number(account_size)
    if account_size is not None and account is None:
        raise ValueError("account_size must be a finite number greater than zero")
    if risk_origin not in ("caller", "profile", "default"):
        raise ValueError("risk_origin must be caller, profile or default")
    if max_position_origin not in ("caller", "profile", "default"):
        raise ValueError("max_position_origin must be caller, profile or default")
    if stop_basis not in ("provided", "atr"):
        raise ValueError("stop_basis must be provided or atr")

    entry = _positive_number(entry_price)
    if entry is None:
        return _unavailable("entry_price_unavailable")
    stop = _positive_number(stop_price)
    if stop is None:
        return _unavailable("stop_price_unavailable")
    if stop >= entry:
        return _unavailable("stop_not_below_entry")
    selected_atr = _positive_number(atr) if stop_basis == "atr" else None
    if stop_basis == "atr" and selected_atr is None:
        return _unavailable("atr_unavailable")

    distance = (entry - stop) / entry
    risk_cap = risk / distance
    risk_budget = account * risk / 100 if account is not None else None
    full_constraints = [
        (f"{risk_origin}_risk_budget", risk_cap),
        (f"{max_position_origin}_max_position", maximum),
        ("regime_maximum", regime_maximum),
    ]
    starter_constraints = full_constraints + [
        ("regime_starter", regime_starter),
        ("preset_starter", preset_starter),
    ]
    sizing: PositionSizing = {
        "assumptions": {
            "entry_price": entry,
            "stop_price": stop,
            "stop_distance_fraction": distance,
            "stop_basis": stop_basis,
            "atr": selected_atr,
            "account_size": account,
        },
        "limits": {
            "risk_per_trade_pct": risk,
            "max_position_pct": maximum,
            "risk_origin": risk_origin,
            "max_position_origin": max_position_origin,
            "regime_max_pct": regime_maximum,
            "regime_starter_pct": regime_starter,
            "preset_starter_pct": preset_starter,
            "risk_cap_pct": risk_cap,
            "risk_budget_dollars": risk_budget,
        },
        "starter": _scenario(starter_constraints, entry, entry - stop, account, risk_budget),
        "full": _scenario(full_constraints, entry, entry - stop, account, risk_budget),
    }
    return {
        "sizing_status": {
            "status": "available" if account is not None else "percentage_only",
            "reason": None,
        },
        "sizing": sizing,
    }
