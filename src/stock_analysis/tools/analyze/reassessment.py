"""earliest_reassessment (plan §6): the first NYSE session on which a re-run of analyze could find every
dated blocker lifted. A date to check again, never permission to buy: it never changes an action."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from stock_analysis.tools.analyze.gates import GATES, EarningsWindow, clears_when_for
from stock_analysis.utils.market_calendar import calendar_covers

EARNINGS_REVERIFY = "earnings_reverify"
PERMANENT_CLEARS_WHEN = "never: outside this tool's supported scope"
_EVENT_IDS = frozenset(g for g, gate in GATES.items() if gate.gate_class == "event")
_SUPPORT_SCOPE = {"unsupported_instrument": "instrument", "unsupported_currency": "currency"}


def permanent_support_gate(gate_id: str, support: Mapping[str, Any] | None) -> bool:
    scope = _SUPPORT_SCOPE.get(gate_id)
    block = (support or {}).get(scope) if scope else None
    return isinstance(block, Mapping) and block.get("status") == "unsupported"


def earliest_reassessment(
    *, blocked_by: Sequence[Mapping[str, Any]], admits: bool, window: EarningsWindow, today: date,
    support: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """One horizon's object, or None unless a fired gate binds (blocked_by non-empty and the candidate admits
    new money). ``blocked_by`` is that horizon's disclosure, unique ids in rank order (decide())."""
    if not blocked_by or not admits:
        return None
    permanent = [b["id"] for b in blocked_by if permanent_support_gate(b["id"], support)]
    if permanent:
        return {"date": None, "basis": "unsupported", "earnings_risk_through": None, "session_method": None,
                "outstanding": [{"id": g, "clears_when": PERMANENT_CLEARS_WHEN} for g in permanent]}
    ids = {b["id"] for b in blocked_by}
    outstanding = [{"id": b["id"], "clears_when": clears_when_for(b)} for b in blocked_by if b["id"] not in _EVENT_IDS]
    if ids & _EVENT_IDS:
        outstanding.append({"id": EARNINGS_REVERIFY, "clears_when": GATES["earnings_unverifiable"].clears_when})
    cleared = window.clear_session(today) if "earnings_blackout" in ids and window.datable else None
    if cleared is None:
        return {"date": None, "basis": "conditions_only", "earnings_risk_through": None, "session_method": None,
                "outstanding": outstanding}
    day, through = cleared
    method = "calendar_static" if calendar_covers(today) and calendar_covers(day) else "clock_only_no_holidays_fallback"
    return {"date": day.isoformat(), "basis": "earnings_clear", "earnings_risk_through": through.isoformat(),
            "session_method": method, "outstanding": outstanding}
