"""Which price drives analyze's zones, stops, sizing and upside, and how fresh it is (WP5).

Order: a fresh intraday bar in the regular session, else the latest valid close of a
completed session, else the info quote as a labelled fallback (it can sit in the cache
for hours). Daily closes are aged in sessions against the last completed session;
timestamped quotes in hours, in UTC. Staleness here is a disclosure, not a gate.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import pytz

from stock_analysis.utils.freshness import FRESHNESS_CEILING_MINUTES, FUTURE_SKEW_TOLERANCE_SECONDS
from stock_analysis.utils.market_calendar import (
    classify_session,
    last_completed_session,
    trading_days_between,
)

_ET = pytz.timezone("America/New_York")


def to_utc(value: Any) -> datetime | None:
    """A tz-aware UTC datetime from an ISO string (an offset or Z; naive means UTC),
    epoch seconds, or a datetime. None when it can't be read."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None
        try:
            return datetime.fromtimestamp(value, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, (str, datetime)) or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts.to_pydatetime()


def _valid_price(value: Any) -> float | None:
    price = pd.to_numeric(value, errors="coerce")
    if pd.isna(price) or not math.isfinite(float(price)) or float(price) <= 0:
        return None
    return float(price)


def latest_session_close(df: pd.DataFrame | None, now: datetime) -> dict[str, Any] | None:
    """The latest valid close of a completed session. Today's forming bar is skipped
    until its session closes; a missing, non-finite or non-positive close is skipped.
    History is never cached in-app, so ``cache_hit`` is always False."""
    if df is None or len(df) == 0 or "close" not in df.columns or "date" not in df.columns:
        return None
    completed = last_completed_session(now.astimezone(_ET))
    for raw_date, raw_close in zip(reversed(df["date"].tolist()), reversed(df["close"].tolist()), strict=True):
        try:
            session_date = date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            continue
        close = _valid_price(raw_close)
        if session_date <= completed and close is not None:
            return {"close": round(close, 4), "session_date": session_date.isoformat(),
                    "fetched_at": df.attrs.get("fetched_at"), "cache_hit": False}
    return None


def latest_intraday_bar(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """The latest intraday bar with a valid close and a readable timestamp (UTC)."""
    if df is None or len(df) == 0 or "close" not in df.columns or "date" not in df.columns:
        return None
    for raw_date, raw_close in zip(reversed(df["date"].tolist()), reversed(df["close"].tolist()), strict=True):
        ts = to_utc(raw_date)
        close = _valid_price(raw_close)
        if ts is not None and close is not None:
            return {"close": round(close, 4), "bar_time": ts.isoformat(),
                    "fetched_at": df.attrs.get("fetched_at"), "cache_hit": False}
    return None


def session_staleness(session_date: date, now: datetime) -> dict[str, Any]:
    """Sessions between a daily close and the last completed session."""
    expected = last_completed_session(now.astimezone(_ET))
    behind = trading_days_between(session_date, expected)
    return {"unit": "sessions", "expected_session": expected.isoformat(),
            "sessions_behind": behind, "stale": behind > 0}


def quote_staleness(ts: datetime, now: datetime) -> dict[str, Any]:
    """Age of a timestamped quote in hours. In the regular session a quote older than
    the freshness ceiling is stale; otherwise one from before the last completed
    session is."""
    now_et = now.astimezone(_ET)
    age_seconds = (now.astimezone(UTC) - ts).total_seconds()
    staleness: dict[str, Any] = {"unit": "hours", "age_hours": round(age_seconds / 3600, 2)}
    if age_seconds < -FUTURE_SKEW_TOLERANCE_SECONDS:
        return {**staleness, "stale": True, "reason": "future_timestamp"}
    if classify_session(now_et) == "regular":
        stale = age_seconds > FRESHNESS_CEILING_MINUTES * 60
    else:
        stale = ts.astimezone(_ET).date() < last_completed_session(now_et)
    return {**staleness, "stale": stale}


def _provenance(
    source: str | None, as_of: str | None, kind: str | None, block: dict[str, Any],
    staleness: dict[str, Any] | None, fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "price_source": source,
        "price_as_of": as_of,
        "as_of_kind": kind,
        "as_of_precision": None if kind is None else "date" if kind == "session_close" else "second",
        "fetched_at": block.get("fetched_at"),
        "cache_hit": block.get("cache_hit"),
        "fallback_reason": fallback_reason,
        "staleness": staleness,
    }


def select_price(
    *,
    now: datetime,
    intraday_quote: dict[str, Any] | None = None,
    session_close: dict[str, Any] | None = None,
    info_quote: dict[str, Any] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """(price, provenance) for analyze. ``now`` is tz-aware.

    An intraday bar counts only in the regular session, and only when its own
    timestamp is within the freshness ceiling (and not further ahead than clock skew).
    """
    now_utc = now.astimezone(UTC)
    if intraday_quote and classify_session(now.astimezone(_ET)) == "regular":
        ts = to_utc(intraday_quote.get("bar_time"))
        price = _valid_price(intraday_quote.get("close"))
        if ts is not None and price is not None:
            age = (now_utc - ts).total_seconds()
            if -FUTURE_SKEW_TOLERANCE_SECONDS <= age <= FRESHNESS_CEILING_MINUTES * 60:
                return price, _provenance("intraday_quote", ts.isoformat(), "bar_time", intraday_quote,
                                          quote_staleness(ts, now))
    if session_close:
        price = _valid_price(session_close.get("close"))
        session_date = session_close.get("session_date")
        if price is not None and isinstance(session_date, str):
            return price, _provenance("daily_close", session_date, "session_close", session_close,
                                      session_staleness(date.fromisoformat(session_date), now))
    info = info_quote or {}
    price = _valid_price(info.get("price"))
    if price is None:
        return None, _provenance(None, None, None, {}, None, "no_price")
    ts = to_utc(info.get("quote_time"))
    if ts is None:
        staleness = {"unit": "hours", "age_hours": None, "stale": True, "reason": "no_quote_time"}
        return price, _provenance("info_quote_fallback", None, None, info, staleness, "no_valid_daily_close")
    return price, _provenance("info_quote_fallback", ts.isoformat(), "quote_time", info,
                              quote_staleness(ts, now), "no_valid_daily_close")
