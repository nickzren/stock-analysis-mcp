"""Trade plan builder: entry, stop, targets, R-based sizing, time stop."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from stock_analysis.tools.trade_setup.setup_rules import TIME_STOP_TRADING_DAYS


def add_trading_days(start: date, days: int) -> date:
    d = start
    remaining = days
    while remaining > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            remaining -= 1
    return d


def build_plan(
    setup: dict[str, Any],
    *,
    action: str,
    session: str,
    account_size: float | None,
    risk_per_trade_pct: float,
    max_position_pct: float,
    now: datetime,
) -> dict[str, Any]:
    entry_price = float(setup["trigger_price"])
    stop_price = float(setup["stop_price"])
    risk_per_share = entry_price - stop_price  # > 0 guaranteed by detectors

    targets = _build_targets(setup, entry_price, risk_per_share)

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

    if account_size is None:
        max_loss_dollars = None
        shares = None
        fractional_shares = None
        position_dollars = None
    else:
        risk_dollars = account_size * risk_per_trade_pct / 100.0
        raw_shares = risk_dollars / risk_per_share
        cap_shares = (account_size * max_position_pct / 100.0) / entry_price
        fractional_shares = round(min(raw_shares, cap_shares), 4)
        shares = int(fractional_shares)
        position_dollars = round(fractional_shares * entry_price, 2)
        max_loss_dollars = round(fractional_shares * risk_per_share, 2)

    uncapped_pct = risk_per_trade_pct * entry_price / risk_per_share
    return {
        "entry": entry,
        "stop": {
            "price": round(stop_price, 2),
            "basis": setup["stop_basis"],
            "distance_pct": round(risk_per_share / entry_price, 4),
        },
        "targets": targets,
        "reward_risk": targets[0]["r_multiple"],
        "time_stop": {
            "trading_days": trading_days,
            "date": add_trading_days(now.date(), trading_days).isoformat(),
        },
        "max_loss_dollars": max_loss_dollars,
        "shares": shares,
        "fractional_shares": fractional_shares,
        "position_dollars": position_dollars,
        "position_pct": round(min(uncapped_pct, max_position_pct), 2),
    }


def _build_targets(
    setup: dict[str, Any],
    entry_price: float,
    risk_per_share: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    primary = setup.get("target_primary")
    if primary is not None and primary["price"] > entry_price:
        candidates.append({"price": float(primary["price"]), "basis": primary["basis"]})
    else:
        candidates.append({"price": entry_price + risk_per_share, "basis": "r_multiple"})
    candidates.append({"price": entry_price + 2.0 * risk_per_share, "basis": "r_multiple"})

    targets: list[dict[str, Any]] = []
    seen: set[float] = set()
    for c in sorted(candidates, key=lambda x: x["price"]):
        price = round(c["price"], 2)
        if price in seen or price <= entry_price:
            continue
        seen.add(price)
        targets.append({
            "price": price,
            "r_multiple": round((price - entry_price) / risk_per_share, 2),
            "basis": c["basis"],
        })
    return targets
