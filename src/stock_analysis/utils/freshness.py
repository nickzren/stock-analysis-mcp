"""Freshness block builders shared by trade-setup and technicals surfaces.

`as_of` and `quote_age_seconds` derive exclusively from market-data bar
timestamps — never from fetch/provenance time (design doc: Contract Invariant).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from stock_analysis.utils.market_calendar import most_recent_trading_day

# Generic freshness property (minutes); trade-setup re-exports it for
# backward compatibility.
FRESHNESS_CEILING_MINUTES = 15

_UNVERIFIABLE: dict[str, Any] = {
    "as_of": None,
    "basis": "unverifiable",
    "quote_age_seconds": None,
    "stale": True,
}


def most_recent_expected_trading_day(now: datetime, session: str) -> date:
    """Most recent day whose daily bar should exist (holiday-aware in coverage)."""
    return most_recent_trading_day(now, session)


def build_freshness(
    *,
    intraday_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    session: str,
    now: datetime,
) -> dict[str, Any]:
    """Build the freshness block. `now` must be tz-aware (America/New_York)."""
    if session == "regular":
        ts = _last_bar_timestamp_utc(intraday_df)
        if ts is None:
            return {**_UNVERIFIABLE, "session": session}
        last_close = (
            pd.to_numeric(intraday_df["close"], errors="coerce").iloc[-1]
            if intraday_df is not None and "close" in intraday_df.columns
            and len(intraday_df) > 0
            else None
        )
        if last_close is None or pd.isna(last_close):
            # A timestamp without a finite close is not a reliable actionable
            # price (contract invariant).
            return {**_UNVERIFIABLE, "session": session}
        age = max(0, int((now.astimezone(UTC) - ts).total_seconds()))
        return {
            "as_of": ts.isoformat(),
            "basis": "bar_timestamp",
            "session": session,
            "quote_age_seconds": age,
            "stale": age > FRESHNESS_CEILING_MINUTES * 60,
        }

    last_daily = _last_daily_date(daily_df)
    if last_daily is None:
        return {**_UNVERIFIABLE, "session": session}
    expected = most_recent_expected_trading_day(now, session)
    return {
        "as_of": last_daily.isoformat(),
        "basis": "bar_timestamp",
        "session": session,
        "quote_age_seconds": None,
        "stale": last_daily < expected,
    }


def freshness_blockers(freshness: dict[str, Any]) -> list[dict[str, str]]:
    if freshness["basis"] == "unverifiable":
        return [{
            "id": "freshness_unverifiable",
            "reason": "no reliable market-data timestamp for the actionable price",
        }]
    if freshness["stale"]:
        if freshness["session"] == "regular":
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
