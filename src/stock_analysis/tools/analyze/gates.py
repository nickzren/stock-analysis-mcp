"""Shared hard-gate checks used by the decision card and the trade-setup card."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pytz

from stock_analysis.tools.analyze.constants import CRITICAL_ANALYZE_TOOLS
from stock_analysis.utils.freshness import (
    FRESHNESS_CEILING_MINUTES,
    FRESHNESS_UNVERIFIABLE,
    STALE_DATA,
)
from stock_analysis.utils.market_calendar import (
    classify_session,
    last_completed_session,
    session_method,
    trading_days_between,
)

# Liquidity threshold below which we block new buys for small accounts.
WEAK_LIQUIDITY_DOLLARS_PER_DAY = 1_000_000.0

# Days before earnings during which entries are blocked.
EARNINGS_BLACKOUT_DAYS = 5

# Falling-knife rubric threshold (same rubric as dip assessment).
FALLING_KNIFE_SCORE_THRESHOLD = 4


def check_earnings_blackout(events_data: dict[str, Any]) -> tuple[bool, dict[str, str] | None]:
    earnings = events_data.get("earnings") or {}
    days_until = earnings.get("days_until")
    fired = (
        isinstance(days_until, (int, float))
        and 0 <= days_until <= EARNINGS_BLACKOUT_DAYS
    )
    if not fired or not isinstance(days_until, (int, float)):
        return False, None
    return True, {
        "id": "earnings_blackout",
        "reason": f"earnings in {int(days_until)} days (<= {EARNINGS_BLACKOUT_DAYS})",
    }


def check_earnings_unverifiable(
    events_data: dict[str, Any] | None,
    tool_failures: list[dict[str, Any]] | None,
) -> tuple[bool, dict[str, str] | None]:
    """Earnings risk unverified: the events tool failed, or no source verified the next date (B1)."""
    events_failed = any(isinstance(tf, dict) and tf.get("tool") == "events_calendar" for tf in tool_failures or [])
    earnings = (events_data or {}).get("earnings") or {}
    if not events_failed and earnings.get("verification") != "unverifiable":
        return False, None
    cause = _unverifiable_cause(earnings, events_failed)
    return True, {"id": "earnings_unverifiable",
                  "reason": f"earnings calendar unverifiable ({cause}) — event risk unverified"}


def _unverifiable_cause(earnings: dict[str, Any], events_failed: bool) -> str:
    if events_failed:
        return "tool fetch"
    causes = [f"{name} failed" for name in earnings.get("sources_failed") or []]
    causes += [f"{name} stale" for name in earnings.get("sources_stale") or []]
    if any(window.get("window_basis") == "assumed" for window in earnings.get("windows") or []):
        causes.append("estimate window assumed")
    return ", ".join(causes) or "no verified source"


def check_liquidity(risk_data: dict[str, Any]) -> tuple[bool, bool, list[dict[str, str]]]:
    liquidity = risk_data.get("liquidity") or {}
    avg_dollar_volume = liquidity.get("avg_dollar_volume")
    weak = (
        isinstance(avg_dollar_volume, (int, float))
        and avg_dollar_volume < WEAK_LIQUIDITY_DOLLARS_PER_DAY
    )
    missing = avg_dollar_volume is None
    blockers: list[dict[str, str]] = []
    if weak and isinstance(avg_dollar_volume, (int, float)):
        blockers.append({
            "id": "weak_liquidity",
            "reason": (
                f"avg daily dollar volume "
                f"~${avg_dollar_volume / 1_000_000:.2f}M < "
                f"${WEAK_LIQUIDITY_DOLLARS_PER_DAY / 1_000_000:.0f}M threshold"
            ),
        })
    if missing:
        blockers.append({
            "id": "liquidity_missing",
            "reason": "avg_dollar_volume unavailable — cannot confirm tradability for small account",
        })
    return weak, missing, blockers


def check_data_quality(
    data_quality: dict[str, Any],
    critical_tools: frozenset[str] = CRITICAL_ANALYZE_TOOLS,
) -> tuple[bool, dict[str, str] | None]:
    fund_status = data_quality.get("fundamentals_status")
    missing_critical = data_quality.get("missing_critical") or []
    tool_failures = data_quality.get("tool_failures") or []
    critical_tool_failed = any(
        tf.get("tool") in critical_tools for tf in tool_failures if isinstance(tf, dict)
    )
    fired = fund_status == "missing" or bool(missing_critical) or critical_tool_failed
    if not fired:
        return False, None
    reasons: list[str] = []
    if fund_status == "missing":
        reasons.append("fundamentals_status=missing")
    if missing_critical:
        reasons.append(f"missing_critical={','.join(str(x) for x in missing_critical)}")
    if critical_tool_failed:
        failed_names = sorted({
            str(tf.get("tool"))
            for tf in tool_failures
            if isinstance(tf, dict) and tf.get("tool") in critical_tools
        })
        reasons.append(f"critical_tool_failed={','.join(failed_names)}")
    return True, {
        "id": "data_quality_critical",
        "reason": "critical data unavailable: " + "; ".join(reasons),
    }


def falling_knife_assessment(
    *,
    death_cross: bool | None,
    below_sma200: bool | None,
    return_3m: float | None,
    position_in_range: float | None,
    sma_200_slope: float | None,
    days_since_52w_high: int | None,
) -> tuple[int, list[str]]:
    """Score falling-knife conditions. Identical rubric to dip assessment.

    A knife requires evidence of an actual decline: price below SMA200, or a
    3-month return past the significant-decline tier (< -20%). Structure-lag
    artifacts alone (persistent death cross, stale 52w high, a barely-negative
    long SMA slope) must not qualify a recovering chart.
    """
    declining = below_sma200 is True or (
        return_3m is not None and return_3m < -0.20
    )
    if not declining:
        return 0, []
    score = 0
    reasons: list[str] = []
    if death_cross:
        score += 2
        reasons.append("death_cross_active")
    if below_sma200:
        score += 1
        reasons.append("below_sma200")
    if return_3m is not None and return_3m < -0.30:
        score += 2
        reasons.append("severe_3m_decline")
    elif return_3m is not None and return_3m < -0.20:
        score += 1
        reasons.append("significant_3m_decline")
    if position_in_range is not None and position_in_range < 0.10:
        score += 1
        reasons.append("near_52w_low")
    if sma_200_slope is not None and sma_200_slope < 0:
        score += 1
        reasons.append("sma200_downtrend")
    if days_since_52w_high is not None and days_since_52w_high > 180:
        score += 1
        reasons.append("stale_52w_high")
    return score, reasons


def is_falling_knife_technicals(technicals_data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate the falling-knife rubric directly from a technicals payload."""
    ma = technicals_data.get("moving_averages") or {}
    rules = ma.get("rules") or {}
    pos = technicals_data.get("price_position") or {}
    returns = technicals_data.get("returns") or {}

    death_cross = (rules.get("death_cross") or {}).get("triggered")
    above = (rules.get("above_sma200") or {}).get("triggered")
    below_sma200 = None if above is None else not above

    score, reasons = falling_knife_assessment(
        death_cross=death_cross,
        below_sma200=below_sma200,
        return_3m=returns.get("return_3m"),
        position_in_range=pos.get("position_in_range"),
        sma_200_slope=ma.get("sma_200_slope_pct_per_day"),
        days_since_52w_high=pos.get("days_since_52w_high"),
    )
    return score >= FALLING_KNIFE_SCORE_THRESHOLD, reasons


# --- The freshness gate (plan §5, D-7): leg Q (the selected price), leg H (technicals' daily history) ---

FRESHNESS_CODE_IDS: dict[str, str] = {  # plan §5's id table: leg Q codes, then leg H codes in rule order
    "info_quote_fallback": FRESHNESS_UNVERIFIABLE,
    "intraday_quote_unavailable": FRESHNESS_UNVERIFIABLE,
    "intraday_quote_stale": STALE_DATA,
    "intraday_future_timestamp": FRESHNESS_UNVERIFIABLE,
    "daily_close_stale": STALE_DATA,
    "daily_history_unavailable": FRESHNESS_UNVERIFIABLE,
    "daily_latest_bar_invalid": FRESHNESS_UNVERIFIABLE,
    "daily_history_stale": STALE_DATA,
    "daily_future_bar": FRESHNESS_UNVERIFIABLE,
    "daily_bar_after_last_session": FRESHNESS_UNVERIFIABLE,
}
INCLUDES_FORMING_BAR, CALENDAR_UNVERIFIED = "includes_forming_bar", "calendar_unverified"
_INTRADAY_REJECTIONS = frozenset({"intraday_quote_unavailable", "intraday_quote_stale", "intraday_future_timestamp"})
_ET = pytz.timezone("America/New_York")


def _iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _sessions(n: Any) -> str:
    return f"{n} session{'' if n == 1 else 's'}"


def check_price_freshness(
    *, now: datetime, price_provenance: dict[str, Any] | None, daily_history: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(blockers, data_quality.freshness) for analyze's selected price. Pure; never raises.

    No selected price adds no code: that is data_quality_critical's case (current_price missing).
    Flags are disclosure only.
    """
    now_et = now.astimezone(_ET)
    session = classify_session(now_et)
    expected = last_completed_session(now_et)
    today = now_et.date()
    prov = price_provenance if isinstance(price_provenance, dict) else {}
    source = prov.get("price_source")
    behind_q = (prov.get("staleness") or {}).get("sessions_behind")
    codes: list[str] = []
    flags: list[str] = []

    # Leg Q: in the regular session select_price returns an intraday quote only when it is fresh.
    if source == "info_quote_fallback":
        codes.append("info_quote_fallback")
    elif source == "daily_close" and session == "regular":
        rejected = prov.get("fallback_reason")
        codes.append(rejected if rejected in _INTRADAY_REJECTIONS else "intraday_quote_unavailable")
    elif source == "daily_close" and (not isinstance(behind_q, int) or behind_q >= 1):
        codes.append("daily_close_stale")

    # Leg H, every session.
    history = daily_history if isinstance(daily_history, dict) else {}
    last_valid = _iso_date(history.get("last_valid_date"))
    completed = _iso_date(history.get("completed_valid_date"))
    behind_h: int | None = None
    if last_valid is None or not isinstance(history.get("last_row_valid"), bool):
        codes.append("daily_history_unavailable")
        last_valid = completed = None
    else:
        if history["last_row_valid"] is False:
            codes.append("daily_latest_bar_invalid")
        if completed is None:
            codes.append("daily_history_unavailable")
        else:
            behind_h = trading_days_between(completed, expected)
            if behind_h >= 1:
                codes.append("daily_history_stale")
        forming = session == "regular" and last_valid == today
        if last_valid > today:
            codes.append("daily_future_bar")
        elif last_valid > expected and not forming:
            codes.append("daily_bar_after_last_session")
        elif forming:
            flags.append(INCLUDES_FORMING_BAR)
    if session_method(now_et) != "calendar_static":
        flags.append(CALENDAR_UNVERIFIED)

    reasons = {
        "info_quote_fallback": (
            f"price from the info quote fallback (quoted {prov.get('price_as_of') or 'at an unknown time'})"),
        "intraday_quote_unavailable": "no usable intraday quote in the regular session",
        "intraday_quote_stale": f"intraday quote older than {FRESHNESS_CEILING_MINUTES} min",
        "intraday_future_timestamp": "intraday quote timestamped in the future",
        "daily_close_stale": (f"last close {prov.get('price_as_of')} is {_sessions(behind_q)} behind "
                              f"(expected {expected})"),
        "daily_history_unavailable": "daily history unavailable or malformed",
        "daily_latest_bar_invalid": f"latest daily row {history.get('last_row_date')} has no valid close",
        "daily_history_stale": (f"daily history {_sessions(behind_h)} behind "
                                f"(last completed bar {completed}, expected {expected})"),
        "daily_future_bar": f"daily history has a row dated {last_valid}, after today",
        "daily_bar_after_last_session": (f"daily history has a valid row dated {last_valid}, "
                                         f"after the last completed session {expected}"),
    }
    blockers: list[dict[str, Any]] = []
    for gate_id in (STALE_DATA, FRESHNESS_UNVERIFIABLE):
        mine = [code for code in codes if FRESHNESS_CODE_IDS[code] == gate_id]
        if mine:
            blockers.append({"id": gate_id, "reason": "; ".join(reasons[code] for code in mine), "codes": mine})
    disclosure = {
        "session": session,
        "expected_session": expected.isoformat(),
        "price_source": source,
        "daily_last_valid_date": None if last_valid is None else last_valid.isoformat(),
        "daily_completed_session": None if completed is None else completed.isoformat(),
        "daily_sessions_behind": behind_h,
        "codes": codes,
        "flags": flags,
    }
    return blockers, disclosure


# --- The gate registry (plan §4) ---

NOT_APPLICABLE, UNEVALUATED = "not_applicable", "unevaluated"
SURFACES = ("mid_term", "long_term", "swing", "screen")
ANALYZE_SURFACES = frozenset({"mid_term", "long_term"})
CLASS_RANK: dict[str, int] = {
    "data": 1, "price_risk": 2, "liquidity": 2, "thesis": 3, "support": 4,
    "event": 5, "zone": 6, "session": 7, "plan": 8,
}


@dataclass(frozen=True)
class Gate:
    gate_class: str
    effect: str  # veto | cap | conditional
    outcome: Mapping[str, str]  # surface -> action | NOT_APPLICABLE | UNEVALUATED (screen only)
    predicate: str  # documentation; R2.6(b) is enforced by behavior tests
    clears_when: str
    unevaluated_when: Mapping[str, str] = field(default_factory=dict)  # surface -> disclosed reason text
    tool_predicates: Mapping[str, str] = field(default_factory=dict)  # R2.6(b) listed exceptions only
    analyze_class: str | None = None  # R2.6(c) exception: liquidity_missing is data in analyze (D-5c)
    requires_supported: bool = False  # liquidity: not_applicable unless support is supported


def _outcome(mid_term: str, long_term: str, swing: str, screen: str) -> dict[str, str]:
    return {"mid_term": mid_term, "long_term": long_term, "swing": swing, "screen": screen}


_NA, _UE = NOT_APPLICABLE, UNEVALUATED
_WAIT_FOR_DATA = _outcome("wait_for_data", "wait_for_data", "wait_for_data", "wait_for_data")
_AVOID = _outcome("avoid", "avoid", "avoid", "avoid")
_SUPPORT_CAP = _outcome("watch", "watch", "watch", "watch")
_EVENT_CAP = _outcome("wait", "wait", "watch", _UE)
_FUNDAMENTALS_MISSING = "fundamentals missing or failed"
_TECHNICALS_MISSING = "technicals unavailable — falling-knife rubric not computed"
_EVENTS_FAILED = "events calendar unavailable — earnings date not checked"
_QUOTE_LEG = "quote leg in the regular session: stage 1 reads no intraday quote"
_SUPPORT_UNVERIFIED = "support status unverified (history attrs missing): a setup promotes"
_MARKET_DATA_CURRENT = "current market data with a verifiable timestamp: a current price and a current daily history"
_ANALYZE_FRESHNESS = "check_price_freshness: legs Q and H"
_FRESHNESS_TOOL_PREDICATES = {
    "mid_term": _ANALYZE_FRESHNESS, "long_term": _ANALYZE_FRESHNESS,
    "swing": "build_freshness + freshness_blockers",
    "screen": "daily_leg_freshness + freshness_blockers",
}
_ANALYZE_DQ = ("check_data_quality(CRITICAL_ANALYZE_TOOLS): a critical tool failed, current_price missing, "
               "or fundamentals_status missing (beta not required, B3)")
_LIQUIDITY_FLOOR = f"${WEAK_LIQUIDITY_DOLLARS_PER_DAY / 1_000_000:.0f}M"

# Insertion order is the within-rank disclosure order.
GATES: dict[str, Gate] = {
    STALE_DATA: Gate(
        "data", "veto", _WAIT_FOR_DATA,
        predicate="the selected price or the daily history is >= 1 session behind the last completed session",
        clears_when=_MARKET_DATA_CURRENT, unevaluated_when={"screen": _QUOTE_LEG},
        tool_predicates=_FRESHNESS_TOOL_PREDICATES),
    FRESHNESS_UNVERIFIABLE: Gate(
        "data", "veto", _WAIT_FOR_DATA,
        predicate="no verifiable timestamp, or no valid (finite, > 0) price, for the selected price or the daily history",
        clears_when=_MARKET_DATA_CURRENT, unevaluated_when={"screen": _QUOTE_LEG},
        tool_predicates=_FRESHNESS_TOOL_PREDICATES),
    "data_quality_critical": Gate(
        "data", "veto", _WAIT_FOR_DATA,
        predicate="a surface's required inputs are unavailable",
        clears_when="critical tools and fields available",
        tool_predicates={
            "mid_term": _ANALYZE_DQ, "long_term": _ANALYZE_DQ,
            "swing": "TRADE_CRITICAL_TOOLS failed; no features (>= 21 bars); or no actionable price",
            "screen": "fewer than RISK_METRICS_MIN_BARS (50) bars; no features; or, off-session, no actionable daily price",
        }),
    "missing_runway": Gate(
        "data", "veto", _outcome("wait_for_data", "wait_for_data", _NA, _NA),
        predicate="unprofitable (P/E not meaningful) and burn metrics unavailable",
        clears_when="burn metrics available, or the company no longer unprofitable (P/E meaningful)",
        unevaluated_when={"mid_term": _FUNDAMENTALS_MISSING, "long_term": _FUNDAMENTALS_MISSING}),
    "falling_knife": Gate(
        "price_risk", "veto", _AVOID,
        predicate=f"is_falling_knife_technicals: rubric score >= {FALLING_KNIFE_SCORE_THRESHOLD}, every instrument",
        clears_when=f"falling-knife rubric score below {FALLING_KNIFE_SCORE_THRESHOLD}",
        unevaluated_when={"mid_term": _TECHNICALS_MISSING, "long_term": _TECHNICALS_MISSING,
                          "swing": _TECHNICALS_MISSING, "screen": "fewer than 21 bars: no screen payload"}),
    "weak_liquidity": Gate(
        "liquidity", "veto", _AVOID,
        predicate=f"check_liquidity: 1-year mean volume x last close < {_LIQUIDITY_FLOOR}",
        clears_when=f"1-year average daily dollar volume >= {_LIQUIDITY_FLOOR}",
        requires_supported=True),
    "liquidity_missing": Gate(
        "liquidity", "veto", _outcome("wait_for_data", "wait_for_data", "avoid", "avoid"),
        predicate="check_liquidity: average daily dollar volume unavailable",
        clears_when="average daily dollar volume available",
        analyze_class="data", requires_supported=True),
    "thesis_broken": Gate(
        "thesis", "cap", _outcome("avoid", "avoid", _NA, _NA),
        predicate="business_status == broken or balance_status == unsafe (classify_dislocation_opportunity)",
        clears_when=("business_status no longer broken (business_quality not poor; when unknown, growth and "
                     "profitability not both failing) and cash runway no longer below 4 quarters"),
        unevaluated_when={"mid_term": _FUNDAMENTALS_MISSING, "long_term": _FUNDAMENTALS_MISSING}),
    "unsupported_instrument": Gate(
        "support", "cap", _SUPPORT_CAP,
        predicate="support_blockers: quote type unsupported or unverified",
        clears_when="quote type verified as supported", unevaluated_when={"screen": _SUPPORT_UNVERIFIED}),
    "unsupported_currency": Gate(
        "support", "cap", _SUPPORT_CAP,
        predicate="support_blockers: price currency unsupported or unverified",
        clears_when="price currency verified as supported", unevaluated_when={"screen": _SUPPORT_UNVERIFIED}),
    "earnings_blackout": Gate(
        "event", "cap", _EVENT_CAP,
        predicate=f"check_earnings_blackout: 0 <= days_until <= {EARNINGS_BLACKOUT_DAYS}",
        clears_when=f"no earnings date or window within {EARNINGS_BLACKOUT_DAYS} days",
        unevaluated_when={"mid_term": _EVENTS_FAILED, "long_term": _EVENTS_FAILED, "swing": _EVENTS_FAILED}),
    "earnings_unverifiable": Gate(
        "event", "cap", _EVENT_CAP,
        predicate="check_earnings_unverifiable: events tool failed or earnings.verification == unverifiable",
        clears_when="a re-run finds the next earnings verified (verified_date, outside the blackout) or verified_none"),
    "near_52w_high": Gate(
        "zone", "cap", _outcome("wait", _NA, _NA, _NA),
        predicate="uncapped zone classification is reduce (price_zone near_52w_high)",
        clears_when="price leaves the near-52-week-high band",
        unevaluated_when={"mid_term": "price_zone null or undetermined"}),
    "market_closed": Gate(
        "session", "conditional", _outcome(_NA, _NA, "enter_on_trigger", _UE),
        predicate="session != regular on the card's trigger branch",
        clears_when="the regular session is open"),
    "plan_invariant_violation": Gate(
        "plan", "cap", _outcome(_NA, _NA, "watch", _UE),
        predicate="build_plan raises PlanInvariantError (evaluated only while the outcome is actionable)",
        clears_when="plan geometry valid: finite positive prices, stop below entry"),
}


def gate_class_for(gate_id: str, surface: str) -> str:
    gate = GATES[gate_id]
    if gate.analyze_class is not None and surface in ANALYZE_SURFACES:
        return gate.analyze_class
    return gate.gate_class


def gate_rank(gate_id: str, surface: str) -> int:
    return CLASS_RANK[gate_class_for(gate_id, surface)]


def disclosure_order(surface: str) -> tuple[str, ...]:
    """Ids applicable on a surface: rank order, then registry order within a rank."""
    applicable = [g for g, gate in GATES.items() if gate.outcome[surface] != NOT_APPLICABLE]
    return tuple(sorted(applicable, key=lambda g: gate_rank(g, surface)))


# Registered in WP2.1; analyze evaluates them from WP2.2 (decide()), which deletes this set.
REGISTERED_NOT_EVALUATED = frozenset({"thesis_broken", "near_52w_high"})
ANALYZE_GATE_IDS = tuple(g for g in disclosure_order("mid_term") if g not in REGISTERED_NOT_EVALUATED)
SCREEN_UNEVALUATED = frozenset(g for g, gate in GATES.items() if gate.outcome["screen"] == UNEVALUATED)


def merge_blockers(blockers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per id at its first position: distinct reasons joined by '; ', codes unioned in order."""
    merged: dict[str, dict[str, Any]] = {}
    for blocker in blockers:
        first = merged.get(blocker["id"])
        if first is None:
            merged[blocker["id"]] = dict(blocker)
            continue
        if blocker["reason"] not in first["reason"].split("; "):
            first["reason"] = f"{first['reason']}; {blocker['reason']}"
        if blocker.get("codes"):
            first["codes"] = list(dict.fromkeys([*first.get("codes", []), *blocker["codes"]]))
    return list(merged.values())


def disclose(blocker: dict[str, Any], surface: str) -> dict[str, Any]:
    """The blocker with its registry class and effect on one surface."""
    gate_id = blocker["id"]
    return {**blocker, "class": gate_class_for(gate_id, surface), "effect": GATES[gate_id].effect}
