"""Freshness block builders shared by trade-setup and technicals surfaces.

`as_of` and `quote_age_seconds` derive exclusively from market-data bar
timestamps — never from fetch/provenance time (design doc: Contract Invariant).
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from stock_analysis.utils.market_calendar import (
    most_recent_trading_day,
    previous_trading_day,
)

# Generic freshness property (minutes); trade-setup re-exports it for
# backward compatibility.
FRESHNESS_CEILING_MINUTES = 15

# A bar timestamped further ahead of `now` than this is a data defect
# (clock skew tolerance), not a genuinely fresher quote.
FUTURE_SKEW_TOLERANCE_SECONDS = 5

_UNVERIFIABLE: dict[str, Any] = {
    "as_of": None,
    "basis": "unverifiable",
    "quote_age_seconds": None,
    "stale": True,
}


def most_recent_expected_trading_day(now: datetime, session: str) -> date:
    """Most recent day whose daily bar should exist (holiday-aware in coverage)."""
    return most_recent_trading_day(now, session)


def _daily_state(daily_df: pd.DataFrame | None) -> tuple[date | None, bool]:
    """(last_daily_date, last_close_finite)."""
    last = _last_daily_date(daily_df)
    if daily_df is None or len(daily_df) == 0 or "close" not in daily_df.columns:
        return last, False
    close = pd.to_numeric(daily_df["close"], errors="coerce").iloc[-1]
    finite = not pd.isna(close) and math.isfinite(float(close))
    return last, finite


def _unverifiable(
    session: str, reason_code: str, daily_bar: date | None, daily_expected: date,
) -> dict[str, Any]:
    return {
        **_UNVERIFIABLE,
        "session": session,
        "reason_code": reason_code,
        "daily_bar_date": daily_bar.isoformat() if daily_bar else None,
        "daily_expected_date": daily_expected.isoformat(),
    }


def build_freshness(
    *,
    intraday_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    session: str,
    now: datetime,
) -> dict[str, Any]:
    """Build the freshness block. `now` must be tz-aware (America/New_York).

    Regular session gates on two legs: a fresh, finite-close intraday bar AND
    a daily frame current as of the last expected trading day — a fresh probe
    no longer authorizes setups computed from an arbitrarily stale daily bar.
    """
    daily_bar, daily_close_finite = _daily_state(daily_df)

    if session == "regular":
        expected = previous_trading_day(now.date())
        ts = _last_bar_timestamp_utc(intraday_df)
        if ts is None:
            return _unverifiable(session, "no_bar_timestamp", daily_bar, expected)
        last_close = (
            pd.to_numeric(intraday_df["close"], errors="coerce").iloc[-1]
            if intraday_df is not None and "close" in intraday_df.columns
            and len(intraday_df) > 0
            else None
        )
        if pd.isna(last_close) or not math.isfinite(float(last_close)):
            # A timestamp without a finite close is not a reliable actionable
            # price (contract invariant).
            return _unverifiable(session, "nonfinite_close", daily_bar, expected)
        age_raw = (now.astimezone(UTC) - ts).total_seconds()
        if age_raw < -FUTURE_SKEW_TOLERANCE_SECONDS:
            # Bar timestamped further ahead than clock skew explains.
            return _unverifiable(session, "future_timestamp", daily_bar, expected)
        age = max(0, int(age_raw))
        quote_stale = age > FRESHNESS_CEILING_MINUTES * 60
        daily_ok = (
            daily_bar is not None
            and expected <= daily_bar <= now.date()
            and daily_close_finite
        )
        stale = quote_stale or not daily_ok
        reason_code = None
        if stale:
            reason_code = "stale_daily" if not daily_ok else "stale_quote"
        return {
            "as_of": ts.isoformat(),
            "basis": "bar_timestamp",
            "session": session,
            "quote_age_seconds": age,
            "stale": stale,
            "reason_code": reason_code,
            "daily_bar_date": daily_bar.isoformat() if daily_bar else None,
            "daily_expected_date": expected.isoformat(),
        }

    expected = most_recent_expected_trading_day(now, session)
    if daily_bar is None or not daily_close_finite or daily_bar > now.date():
        return _unverifiable(session, "daily_unusable", daily_bar, expected)
    stale = daily_bar < expected
    return {
        "as_of": daily_bar.isoformat(),
        "basis": "bar_timestamp",
        "session": session,
        "quote_age_seconds": None,
        "stale": stale,
        "reason_code": "stale_daily" if stale else None,
        "daily_bar_date": daily_bar.isoformat(),
        "daily_expected_date": expected.isoformat(),
    }


def freshness_blockers(freshness: dict[str, Any]) -> list[dict[str, str]]:
    if freshness["basis"] == "unverifiable":
        return [{
            "id": "freshness_unverifiable",
            "reason": (
                "no reliable market-data timestamp for the actionable price"
                f" ({freshness.get('reason_code')})"
            ),
        }]
    if freshness["stale"]:
        if freshness["session"] == "regular":
            if freshness.get("reason_code") == "stale_daily":
                reason = (
                    f"daily frame not current (last {freshness['daily_bar_date']}, "
                    f"expected {freshness['daily_expected_date']})"
                )
            else:
                reason = (
                    f"market data is {freshness['quote_age_seconds']}s old "
                    f"(> {FRESHNESS_CEILING_MINUTES}m ceiling)"
                )
        else:
            reason = f"newest daily bar {freshness['as_of']} predates the last expected trading day"
        return [{"id": "stale_data", "reason": reason}]
    return []


def _last_bar_timestamp_utc(df: pd.DataFrame | None) -> datetime | None:
    if df is None or len(df) == 0 or "date" not in df.columns:
        return None
    # Naive timestamps get localized as UTC, which overstates age for ET data —
    # errs toward downgrade, the safe direction.
    ts = pd.to_datetime(df["date"].iloc[-1], utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()  # type: ignore[no-any-return]


def _last_daily_date(df: pd.DataFrame | None) -> date | None:
    if df is None or len(df) == 0 or "date" not in df.columns:
        return None
    raw = str(df["date"].iloc[-1])[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None
