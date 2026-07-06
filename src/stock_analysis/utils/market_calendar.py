"""Vendored NYSE market calendar (2025-2030) and session classification.

Derivation rules (verify against nyse.com/markets/hours-calendars when
extending coverage): full closures are New Year's Day, MLK Day (3rd Mon Jan),
Washington's Birthday (3rd Mon Feb), Good Friday, Memorial Day (last Mon May),
Juneteenth (Jun 19), Independence Day (Jul 4), Labor Day (1st Mon Sep),
Thanksgiving (4th Thu Nov), Christmas (Dec 25). Saturday holidays are observed
the preceding Friday and Sunday holidays the following Monday — EXCEPT
New Year's Day falling on Saturday, which NYSE does not observe (2028).
Early closes (13:00 ET) are Jul 3 (when a weekday trading day), the day after
Thanksgiving, and Dec 24 (when a weekday trading day). Full-holiday precedence
beats early-close logic: a date in FULL_HOLIDAYS never appears in EARLY_CLOSES
(owner correction, 2026-07-06 — e.g. 2026-07-03 and 2027-12-24 are closures).

Outside coverage (<2025 or >2030) classification falls back to the clock-only
weekday ladder, reported via session_method() as
"clock_only_no_holidays_fallback" — never silently calendar-labeled.
Early-close after-hours ending at 17:00 is a documented approximation.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

CALENDAR_FIRST_YEAR = 2025
CALENDAR_LAST_YEAR = 2030

FULL_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026 (Jul 4 Saturday -> observed Fri Jul 3)
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027 (Juneteenth Sat -> Fri Jun 18; Jul 4 Sun -> Mon Jul 5;
    #       Xmas Sat -> Fri Dec 24)
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
    # 2028 (New Year's Saturday -> NOT observed: 9 holidays)
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14), date(2028, 5, 29),
    date(2028, 6, 19), date(2028, 7, 4), date(2028, 9, 4), date(2028, 11, 23),
    date(2028, 12, 25),
    # 2029
    date(2029, 1, 1), date(2029, 1, 15), date(2029, 2, 19), date(2029, 3, 30),
    date(2029, 5, 28), date(2029, 6, 19), date(2029, 7, 4), date(2029, 9, 3),
    date(2029, 11, 22), date(2029, 12, 25),
    # 2030
    date(2030, 1, 1), date(2030, 1, 21), date(2030, 2, 18), date(2030, 4, 19),
    date(2030, 5, 27), date(2030, 6, 19), date(2030, 7, 4), date(2030, 9, 2),
    date(2030, 11, 28), date(2030, 12, 25),
})

EARLY_CLOSES: frozenset[date] = frozenset({
    # 2025
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
    # 2026 (Jul 3 excluded: observed Independence Day closure)
    date(2026, 11, 27), date(2026, 12, 24),
    # 2027 (Jul 3 Saturday; Dec 24 excluded: observed Christmas closure)
    date(2027, 11, 26),
    # 2028 (Dec 24 Sunday)
    date(2028, 7, 3), date(2028, 11, 24),
    # 2029
    date(2029, 7, 3), date(2029, 11, 23), date(2029, 12, 24),
    # 2030
    date(2030, 7, 3), date(2030, 11, 29), date(2030, 12, 24),
})

_EARLY_CLOSE_MINUTE = 13 * 60
_AFTER_HOURS_END_EARLY = 17 * 60


def calendar_covers(d: date) -> bool:
    return CALENDAR_FIRST_YEAR <= d.year <= CALENDAR_LAST_YEAR


def _clock_ladder(minutes: int) -> str:
    if minutes < 4 * 60:
        return "closed"
    if minutes < 9 * 60 + 30:
        return "pre_market"
    if minutes < 16 * 60:
        return "regular"
    if minutes < 20 * 60:
        return "after_hours"
    return "closed"


def classify_session(now: datetime) -> str:
    """US-equities session for a tz-aware datetime. Calendar-aware in coverage."""
    if now.weekday() >= 5:
        return "closed"
    d = now.date()
    minutes = now.hour * 60 + now.minute
    if calendar_covers(d):
        if d in FULL_HOLIDAYS:
            return "closed"
        if d in EARLY_CLOSES:
            if minutes < 4 * 60:
                return "closed"
            if minutes < 9 * 60 + 30:
                return "pre_market"
            if minutes < _EARLY_CLOSE_MINUTE:
                return "regular"
            if minutes < _AFTER_HOURS_END_EARLY:
                return "after_hours"
            return "closed"
    return _clock_ladder(minutes)


def session_method(now: datetime) -> str:
    return (
        "calendar_static"
        if calendar_covers(now.date())
        else "clock_only_no_holidays_fallback"
    )


def regular_session_minutes(d: date) -> int:
    """Length of the regular session in minutes (210 on early-close days)."""
    if calendar_covers(d) and d in EARLY_CLOSES:
        return 210
    return 390


def most_recent_trading_day(now: datetime, session: str) -> date:
    """Most recent day whose daily bar should exist. Holiday-aware in coverage."""
    d = now.date()
    if session == "pre_market" or (session == "closed" and now.hour < 4):
        d = d - timedelta(days=1)
    while d.weekday() >= 5 or (calendar_covers(d) and d in FULL_HOLIDAYS):
        d = d - timedelta(days=1)
    return d
