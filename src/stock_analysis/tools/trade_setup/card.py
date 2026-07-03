"""Trade-setup card assembly: deterministic action precedence over pure inputs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from stock_analysis.tools.analyze.gates import (
    check_data_quality,
    check_earnings_blackout,
    check_liquidity,
    is_falling_knife_technicals,
)
from stock_analysis.tools.trade_setup.freshness import freshness_blockers
from stock_analysis.tools.trade_setup.plan import add_trading_days, build_plan
from stock_analysis.tools.trade_setup.setup_rules import (
    DEFAULT_REVIEW_TRADING_DAYS,
    PDT_ACCOUNT_MIN,
    TRADE_CRITICAL_TOOLS,
)
from stock_analysis.tools.trade_setup.setups import detect_setup

_ACTIONABLE = frozenset({"trade_now", "enter_on_trigger"})
_CONFIDENCE_BY_QUALITY = {"A": "high", "B": "medium", "C": "low"}

_DELAYED_DATA_NOTE = (
    "Data may be delayed up to 15 minutes — confirm price in your broker before entering."
)
_PDT_NOTE = (
    "Accounts under $25k: a same-day stop-out on a same-day entry counts as a "
    "day trade (PDT rule)."
)


def build_trade_setup_card(
    *,
    symbol: str,
    summary_data: dict[str, Any],
    technicals_data: dict[str, Any],
    risk_data: dict[str, Any],
    events_data: dict[str, Any],
    freshness: dict[str, Any],
    features: dict[str, Any] | None,
    actionable_price: float | None,
    session: str,
    now: datetime,
    account_size: float | None = None,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 10.0,
    tool_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []

    # Rule 1 inputs: freshness + trade-critical data quality.
    blockers.extend(freshness_blockers(freshness))
    dq_fired, dq_blocker = check_data_quality(
        {"tool_failures": tool_failures or []},
        critical_tools=TRADE_CRITICAL_TOOLS,
    )
    if dq_blocker:
        blockers.append(dq_blocker)
    if features is None or actionable_price is None:
        dq_fired = True
        blockers.append({
            "id": "data_quality_critical",
            "reason": "daily price history unavailable — cannot compute setup features",
        })
    wait_for_data = bool(freshness_blockers(freshness)) or dq_fired

    # Rule 2 inputs: structural disqualifiers (evaluable even when rule 1 fired).
    knife = False
    if technicals_data:
        knife, _knife_reasons = is_falling_knife_technicals(technicals_data)
        if knife:
            blockers.append({
                "id": "falling_knife",
                "reason": "severe decline with broken trends — wait for stabilization",
            })
    weak_liquidity, liquidity_missing, liquidity_blockers = check_liquidity(risk_data)
    blockers.extend(liquidity_blockers)

    # Event risk is evaluable regardless of the final action.
    earnings_blackout, earnings_blocker = check_earnings_blackout(events_data)
    if earnings_blocker:
        blockers.append(earnings_blocker)

    setup: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None

    if wait_for_data:
        action = "wait_for_data"
    elif knife or weak_liquidity or liquidity_missing:
        action = "avoid"
    else:
        setup = detect_setup(technicals_data, features, actionable_price)  # type: ignore[arg-type]
        if setup is None:
            action = "no_setup"
        elif earnings_blackout:
            action = "watch"
        elif setup["trigger_satisfied"] and session == "regular" and not freshness["stale"]:
            action = "trade_now"
        elif setup.get("trigger_price") is not None:
            action = "enter_on_trigger"
            if session != "regular":
                blockers.append({
                    "id": "market_closed",
                    "reason": "market not in regular session — trade_now unavailable",
                })
        else:
            action = "watch"

    if action in _ACTIONABLE and setup is not None:
        plan = build_plan(
            setup,
            action=action,
            session=session,
            account_size=account_size,
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
            now=now,
        )

    earnings = events_data.get("earnings") or {}
    days_until = earnings.get("days_until")
    next_date = earnings.get("next_date")

    notes = [_DELAYED_DATA_NOTE]
    if account_size is None or account_size < PDT_ACCOUNT_MIN:
        notes.append(_PDT_NOTE)
    if not events_data:
        notes.append("Earnings calendar unavailable — event risk unverified.")

    return {
        "symbol": symbol,
        "currency": summary_data.get("currency", "USD"),
        "freshness": freshness,
        "action": action,
        "setup": setup,
        "plan": plan,
        "blockers": blockers,
        "event_risk": {
            "earnings_in_days": int(days_until) if isinstance(days_until, (int, float)) else None,
            "next_earnings_date": str(next_date) if next_date else None,
            "expected_move_pct": None,  # roadmap step 5 fills this
        },
        "notes": notes,
        "confidence": (
            _CONFIDENCE_BY_QUALITY.get(setup["quality"], "low")
            if setup is not None and action in _ACTIONABLE
            else "low"
        ),
        "next_review": _next_review(days_until, next_date, plan, now),
    }


def _next_review(
    days_until: float | None,
    next_date: Any,
    plan: dict[str, Any] | None,
    now: datetime,
) -> dict[str, str]:
    if next_date and isinstance(days_until, (int, float)) and 0 <= days_until <= 90:
        return {"date": str(next_date), "reason": f"next earnings (~{int(days_until)} days)"}
    if plan is not None:
        return {"date": plan["time_stop"]["date"], "reason": "time stop"}
    return {
        "date": add_trading_days(now.date(), DEFAULT_REVIEW_TRADING_DAYS).isoformat(),
        "reason": "setup re-check",
    }
