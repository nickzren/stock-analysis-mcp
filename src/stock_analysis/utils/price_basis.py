"""Which price drives analyze's zones, stops, sizing and upside, and how fresh it is (WP5).

Order: a fresh intraday bar in the regular session, else the latest valid close of a
completed session, else the info quote as a labelled fallback (it can sit in the cache
for hours). Daily closes are aged in sessions against the last completed session;
timestamped quotes in hours, in UTC. Staleness here is disclosed; analyze's freshness gate
(``tools/analyze/gates.check_price_freshness``) reads this provenance as its quote leg (plan §5).
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd
import pytz

from stock_analysis.utils.freshness import FRESHNESS_CEILING_MINUTES, FUTURE_SKEW_TOLERANCE_SECONDS
from stock_analysis.utils.helpers import valid_price
from stock_analysis.utils.market_calendar import (
    classify_session,
    last_completed_session,
    regular_session_minutes,
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


def last_valid_bar_date(df: pd.DataFrame | None) -> Any:
    """The ``date`` of the newest bar with a valid close, as stored; None when no bar
    has one. An empty vendor session kept by ``keepna`` is not the freshest bar."""
    if df is None or len(df) == 0 or "close" not in df.columns or "date" not in df.columns:
        return None
    for raw_date, raw_close in zip(reversed(df["date"].tolist()), reversed(df["close"].tolist()), strict=True):
        if valid_price(raw_close) is not None:
            return raw_date
    return None


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
        close = valid_price(raw_close)
        if session_date <= completed and close is not None:
            return {"close": round(close, 4), "session_date": session_date.isoformat(),
                    "fetched_at": df.attrs.get("fetched_at"), "cache_hit": False}
    return None


def build_daily_history(df: pd.DataFrame | None, now: datetime) -> dict[str, Any] | None:
    """Technicals' own daily frame, for analyze's freshness leg H (plan §5). None without a frame."""
    if df is None or len(df) == 0 or "close" not in df.columns or "date" not in df.columns:
        return None
    last_valid = last_valid_bar_date(df)
    completed = latest_session_close(df, now)
    return {
        "last_row_date": str(df["date"].iloc[-1])[:10],  # reason text only
        "last_valid_date": None if last_valid is None else str(last_valid)[:10],
        "last_row_valid": valid_price(df["close"].iloc[-1]) is not None,
        "completed_valid_date": None if completed is None else completed["session_date"],
    }


def latest_intraday_bar(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """The latest intraday bar with a valid close and a readable timestamp (UTC)."""
    if df is None or len(df) == 0 or "close" not in df.columns or "date" not in df.columns:
        return None
    for raw_date, raw_close in zip(reversed(df["date"].tolist()), reversed(df["close"].tolist()), strict=True):
        ts = to_utc(raw_date)
        close = valid_price(raw_close)
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


def _session_close(d: date) -> datetime:
    """When the regular session on trading day ``d`` closes (16:00 ET, or 13:00 on
    early-close days)."""
    return _ET.localize(datetime.combine(d, time(9, 30)) + timedelta(minutes=regular_session_minutes(d)))


def quote_staleness(ts: datetime, now: datetime) -> dict[str, Any]:
    """Age of a timestamped quote in hours. In the regular session a quote older than
    the freshness ceiling is stale; otherwise one from before the last completed
    session's close is (when it was fetched never makes it current)."""
    now_et = now.astimezone(_ET)
    age_seconds = (now.astimezone(UTC) - ts).total_seconds()
    staleness: dict[str, Any] = {"unit": "hours", "age_hours": round(age_seconds / 3600, 2)}
    if age_seconds < -FUTURE_SKEW_TOLERANCE_SECONDS:
        return {**staleness, "stale": True, "reason": "future_timestamp"}
    if classify_session(now_et) == "regular":
        stale = age_seconds > FRESHNESS_CEILING_MINUTES * 60
    else:
        stale = ts < _session_close(last_completed_session(now_et))
    return {**staleness, "stale": stale}


def _clock(now: datetime, block: dict[str, Any]) -> datetime:
    """The newest clock reading for a fetched quote: analyze reads ``now`` before the
    fetch, so a later ``fetched_at`` is closer to the truth."""
    fetched = to_utc(block.get("fetched_at"))
    now_utc = now.astimezone(UTC)
    return max(now_utc, fetched) if fetched else now_utc


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
    In the regular session a daily close is itself a fallback, and its
    ``fallback_reason`` says why the intraday bar was not used.
    """
    intraday_rejected: str | None = None
    if classify_session(now.astimezone(_ET)) == "regular":
        intraday_rejected = "intraday_quote_unavailable"
        bar = intraday_quote or {}
        ts = to_utc(bar.get("bar_time"))
        price = valid_price(bar.get("close"))
        if ts is not None and price is not None:
            clock = _clock(now, bar)
            age = (clock - ts).total_seconds()
            if age < -FUTURE_SKEW_TOLERANCE_SECONDS:
                intraday_rejected = "intraday_future_timestamp"
            elif age > FRESHNESS_CEILING_MINUTES * 60:
                intraday_rejected = "intraday_quote_stale"
            else:
                return price, _provenance("intraday_quote", ts.isoformat(), "bar_time", bar,
                                          quote_staleness(ts, clock))
    if session_close:
        price = valid_price(session_close.get("close"))
        session_date = session_close.get("session_date")
        if price is not None and isinstance(session_date, str):
            return price, _provenance("daily_close", session_date, "session_close", session_close,
                                      session_staleness(date.fromisoformat(session_date), now),
                                      intraday_rejected)
    info = info_quote or {}
    price = valid_price(info.get("price"))
    if price is None:
        return None, _provenance(None, None, None, {}, None, "no_price")
    ts = to_utc(info.get("quote_time"))
    if ts is None:
        staleness = {"unit": "hours", "age_hours": None, "stale": True, "reason": "no_quote_time"}
        return price, _provenance("info_quote_fallback", None, None, info, staleness, "no_valid_daily_close")
    return price, _provenance("info_quote_fallback", ts.isoformat(), "quote_time", info,
                              quote_staleness(ts, _clock(now, info)), "no_valid_daily_close")
