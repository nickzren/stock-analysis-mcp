"""Trade plan builder: entry, stop, targets, R-based sizing, time stop."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from stock_analysis.tools.trade_setup.setup_rules import TIME_STOP_TRADING_DAYS
from stock_analysis.utils.market_calendar import add_trading_days
from stock_analysis.utils.position_sizing import LimitOrigin, calculate_position_sizing
from stock_analysis.utils.sizing_output import public_sizing


class PlanInvariantError(ValueError):
    """A generated setup violates plan-geometry invariants (finite, positive,
    stop strictly below entry). Card assembly catches ONLY this and downgrades.
    """


def _finite_positive(name: str, value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanInvariantError(f"{name} not numeric: {value!r}") from exc
    if not math.isfinite(v) or v <= 0:
        raise PlanInvariantError(f"{name} must be finite and positive, got {value!r}")
    return v


def build_plan(
    setup: dict[str, Any],
    *,
    action: str,
    session: str,
    account_size: float | None,
    risk_per_trade_pct: float,
    max_position_pct: float,
    now: datetime,
    actionable_price: float,
    limit_origins: dict[str, LimitOrigin] | None = None,
) -> dict[str, Any]:
    entry_price = _finite_positive("trigger_price", setup["trigger_price"])
    stop_price = _finite_positive("stop_price", setup["stop_price"])
    actionable = _finite_positive("actionable_price", actionable_price)
    if stop_price >= entry_price:
        raise PlanInvariantError(
            f"stop {stop_price} must be strictly below entry {entry_price}"
        )
    anchor = max(entry_price, actionable) if setup["trigger_satisfied"] else entry_price
    risk_per_share = anchor - stop_price
    if round(risk_per_share, 2) < 0.01:
        raise PlanInvariantError(
            f"risk_per_share {risk_per_share} below 0.01 minimum"
        )

    targets = _build_targets(setup, anchor, risk_per_share)

    if action == "trade_now":
        entry = {
            "type": "market",
            "trigger_price": round(entry_price, 2),
            "valid": "good_till_time_stop",
            "condition_text": "Trigger already satisfied — enter at market",
        }
    else:
        entry = {
            "type": "buy_stop",
            "trigger_price": round(entry_price, 2),
            "valid": "good_till_time_stop" if session == "regular" else "next_session",
            "condition_text": setup["trigger_condition"],
        }

    trading_days = TIME_STOP_TRADING_DAYS[setup["type"]]

    origins = limit_origins or {}
    exact = calculate_position_sizing(
        entry_price=anchor,
        stop_price=round(stop_price, 2),
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
        regime_max_pct=100,
        regime_starter_pct=100,
        preset_starter_pct=100,
        account_size=account_size,
        risk_origin=origins.get("risk_per_trade_pct", "caller"),
        max_position_origin=origins.get("max_position_pct", "caller"),
    )
    sized = public_sizing(exact, horizon="swing")
    sizing = sized["sizing"]
    scenario = sizing["full"] if sizing is not None else {}
    if sizing is not None:
        sizing["starter"] = None
        for key in ("regime_max_pct", "regime_starter_pct", "preset_starter_pct"):
            sizing["limits"][key] = None
        scenario["binding_constraints"] = [
            reason for reason in scenario["binding_constraints"] if reason != "regime_maximum"
        ]
    if scenario.get("quantity_status") == "below_one_whole_share":
        entry["condition_text"] = "The setup qualifies, but no whole-share position fits this budget."
    elif sizing is None:
        entry["condition_text"] = "The setup qualifies, but a size estimate is unavailable."

    return {
        "entry": entry,
        "stop": {
            "price": round(stop_price, 2),
            "basis": setup["stop_basis"],
            "distance_pct": round(risk_per_share / anchor, 4),
        },
        "targets": targets,
        "reward_risk": targets[0]["r_multiple"],
        "time_stop": {
            "trading_days": trading_days,
            "date": add_trading_days(now.date(), trading_days).isoformat(),
        },
        "sizing_status": sized["sizing_status"],
        "sizing": sizing,
        "planned_loss_at_stop": scenario.get("planned_loss_at_stop"),
        "max_loss_dollars": scenario.get("planned_loss_at_stop"),
        "shares": scenario.get("whole_shares"),
        "fractional_shares": None,
        "position_dollars": scenario.get("position_dollars"),
        "position_pct": scenario.get("allocation_cap_pct" if account_size is None else "position_pct"),
    }


def _build_targets(
    setup: dict[str, Any],
    anchor: float,
    risk_per_share: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    primary = setup.get("target_primary")
    if primary is not None and primary["price"] > anchor:
        candidates.append({"price": float(primary["price"]), "basis": primary["basis"]})
    else:
        candidates.append({"price": anchor + risk_per_share, "basis": "r_multiple"})
    candidates.append({"price": anchor + 2.0 * risk_per_share, "basis": "r_multiple"})

    targets: list[dict[str, Any]] = []
    seen: set[float] = set()
    for c in sorted(candidates, key=lambda x: x["price"]):
        price = round(c["price"], 2)
        if not math.isfinite(price):
            raise PlanInvariantError(f"target price not finite ({c['basis']})")
        if price in seen or price <= anchor:
            continue
        seen.add(price)
        targets.append({
            "price": price,
            "r_multiple": round((price - anchor) / risk_per_share, 2),
            "basis": c["basis"],
        })
    return targets
