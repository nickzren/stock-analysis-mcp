"""Events calendar tool."""

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stock_analysis.data.provider import (
    is_epoch,
    read_attribute,
    read_calendar_events,
    read_earnings_page,
)
from stock_analysis.data.yfinance_client import fetch_info, fetch_ticker
from stock_analysis.utils.helpers import current_price_from_info, safe_float, safe_round
from stock_analysis.utils.provenance import (
    FetchError,
    build_meta,
    build_provenance,
    fetch_or_error,
    utcnow_isoformat_z,
)

_ET = ZoneInfo("America/New_York")


async def events_calendar(symbol: str, _now: datetime | None = None) -> dict[str, Any]:
    """
    Get upcoming events and historical earnings for a symbol.

    Args:
        symbol: Stock ticker symbol
        _now: Test seam for the current ET time; production uses the clock

    Returns:
        Dict with earnings, dividends, and splits information
    """
    start_time = perf_counter()
    normalized_symbol = symbol.upper().strip()
    now = _now or datetime.now(_ET)

    try:
        ticker = await fetch_or_error(fetch_ticker(symbol), symbol)
    except FetchError as fe:
        return fe.response

    info_ok = True
    try:
        info = await fetch_info(symbol)
    except Exception:
        info = {}
        info_ok = False

    (calendar_events, calendar_ok), (earnings_dates, earnings_dates_ok) = await asyncio.gather(
        _read_calendar(normalized_symbol), read_earnings_dates(normalized_symbol)
    )

    today = now.date()
    earnings = _build_earnings(
        _calendar_view(calendar_events, calendar_ok, info, today),
        _page_view(earnings_dates, earnings_dates_ok, today),
        _info_view(info, info_ok, today),
        earnings_dates,
        today,
    )
    dividends = _build_dividends(calendar_events, info)
    splits_info = _build_splits(ticker)
    analyst = _build_analyst(info)

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("events_calendar", duration_ms),
        "data_provenance": {
            "events": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
            ),
        },
        "symbol": normalized_symbol,
        "earnings": earnings,
        "dividends": dividends,
        "splits": splits_info,
        "analyst": analyst,
    }


async def _read_calendar(symbol: str) -> tuple[dict[str, Any], bool]:
    """(quoteSummary calendarEvents payload with raw epochs, ok).

    ok=False when the read failed or its envelope was invalid (WP1).
    """
    read = await read_calendar_events(symbol)
    if not read.verified:
        return {}, False
    return read.payload or {}, True


async def read_earnings_dates(symbol: str) -> tuple[pd.DataFrame | None, bool]:
    """(earnings_dates, ok) in yfinance's ``Ticker.earnings_dates`` shape, page order.

    ok=False when the page failed, was malformed, or had no table: a page without
    a table can't prove there is no data, so it does not verify (D2).
    """
    read = await read_earnings_page(symbol)
    if not read.verified:
        return None, False
    return _earnings_frame(read.payload or []), True


def _earnings_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    index = pd.DatetimeIndex([row["timestamp"] for row in rows], name="Earnings Date")
    return pd.DataFrame(
        {
            "EPS Estimate": [row["eps_estimate"] for row in rows],
            "Reported EPS": [row["reported_eps"] for row in rows],
            "Surprise(%)": [row["surprise_pct"] for row in rows],
        },
        index=index,
    )


def _et_date(epoch: Any) -> date | None:
    """An earnings epoch is a real time; its date is the ET calendar date."""
    if not is_epoch(epoch):
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=UTC).astimezone(_ET).date()
    except (OverflowError, OSError, ValueError):
        return None


def _utc_date(epoch: Any) -> str | None:
    """Yahoo sends dividend dates as UTC midnight, so their date is the UTC date."""
    if not is_epoch(epoch):
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=UTC).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class _Candidate:
    """A possible next earnings date from one source. ``verified`` is False for
    an estimate window whose interval reading is only assumed."""

    source: str
    start: date
    end: date
    verified: bool
    is_estimate: bool | None


@dataclass(frozen=True)
class _SourceView:
    """One source's verdict: ``ok`` (has a future date), ``none`` (verified, no
    future date), ``stale`` (only past dates) or ``failed`` (failed or could
    not be verified)."""

    state: str
    candidate: _Candidate | None = None
    window: dict[str, Any] | None = None


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _info_range(info: dict[str, Any]) -> tuple[date, date] | None:
    start = _et_date(info.get("earningsTimestampStart"))
    end = _et_date(info.get("earningsTimestampEnd"))
    return (start, end) if start and end else None


def _window_basis(start: date, end: date, is_estimate: bool | None, info: dict[str, Any]) -> tuple[str, bool]:
    """(window_basis, metadata_conflict) for a two-date calendar entry (WP2, Q2).

    Confirmed only when the estimate flag says "estimate" and the info range
    matches the entries; absent metadata is assumed; contradicting metadata is
    assumed and flagged.
    """
    info_range = _info_range(info)
    if is_estimate is False or (info_range is not None and info_range != (start, end)):
        return "assumed", True
    if is_estimate is True and info_range == (start, end):
        return "confirmed", False
    return "assumed", False


def _calendar_view(events: dict[str, Any], ok: bool, info: dict[str, Any], today: date) -> _SourceView:
    if not ok:
        return _SourceView("failed")
    earnings = events.get("earnings") or {}
    entries = earnings.get("earningsDate") or []
    if not entries:
        return _SourceView("none")
    if len(entries) > 2 or entries[-1] < entries[0]:
        return _SourceView("failed")
    start, end = _et_date(entries[0]), _et_date(entries[-1])
    if start is None or end is None:
        return _SourceView("failed")
    if end < today:
        return _SourceView("stale")
    is_estimate = _as_bool(earnings.get("isEarningsDateEstimate"))
    if is_estimate is None:
        is_estimate = _as_bool(info.get("isEarningsDateEstimate"))
    if len(entries) == 1:
        return _SourceView("ok", _Candidate("calendar", start, end, True, is_estimate))
    basis, conflict = _window_basis(start, end, is_estimate, info)
    window = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "is_estimate": is_estimate,
        "window_basis": basis,
        "metadata_conflict": conflict,
        "window_active": start <= today <= end,
    }
    return _SourceView("ok", _Candidate("calendar", start, end, basis == "confirmed", is_estimate), window)


def _row_et_date(index: Any) -> date:
    """Earnings-page timestamp -> ET date; naive values are ET wall time."""
    ts = pd.Timestamp(index)
    ts = ts.tz_convert(_ET) if ts.tzinfo is not None else ts.tz_localize(_ET)
    return ts.date()


def _page_view(earnings_dates: pd.DataFrame | None, ok: bool, today: date) -> _SourceView:
    if not ok:
        return _SourceView("failed")
    if earnings_dates is None or earnings_dates.empty:
        return _SourceView("none")
    future = [day for day in map(_row_et_date, earnings_dates.index) if day >= today]
    if not future:
        return _SourceView("none")
    nearest = min(future)  # rows come newest first
    return _SourceView("ok", _Candidate("earnings_dates", nearest, nearest, True, None))


def _info_view(info: dict[str, Any], ok: bool, today: date) -> _SourceView:
    if not ok:
        return _SourceView("failed")
    timestamp = info.get("earningsTimestamp")
    if timestamp is None:
        return _SourceView("none")
    day = _et_date(timestamp)
    if day is None:
        return _SourceView("failed")
    range_start, range_end = _info_range(info) or (day, day)
    is_window = range_start != range_end
    end = max(day, range_end) if is_window else day
    if end < today:
        return _SourceView("none")
    is_estimate = _as_bool(info.get("isEarningsDateEstimate"))
    # Info's own multi-day range is an estimate window, never a verified date.
    return _SourceView("ok", _Candidate("info_timestamp", day, end, not is_window, is_estimate))


def _build_earnings(
    calendar: _SourceView,
    page: _SourceView,
    info: _SourceView,
    earnings_dates: pd.DataFrame | None,
    today: date,
) -> dict[str, Any]:
    """Next earnings date, its verification (WP2 truth table) and history.

    ``next_date`` is the earliest risk date across all candidates, so an
    assumed window still drives the blackout; ``verification`` is
    ``verified_date`` only when a verified candidate exists.
    """
    window, echo = calendar.candidate, info.candidate
    if window and echo and not window.verified and window.start <= echo.start <= window.end:
        # Info and calendarEvents carry the same Yahoo record, so an info date
        # inside an unconfirmed calendar window is not independent evidence.
        info = replace(info, candidate=replace(echo, verified=False))
    views = {"calendar": calendar, "earnings_dates": page, "info": info}
    candidates = [view.candidate for view in views.values() if view.candidate]
    nearest = min(candidates, key=lambda c: max(c.start, today), default=None)
    next_day = max(nearest.start, today) if nearest else None

    if any(c.verified for c in candidates):
        verification = "verified_date"
    elif all(view.state == "none" for view in views.values()):
        verification = "verified_none"
    else:
        verification = "unverifiable"

    earnings_history, beat_count, total_with_data = _build_earnings_history(earnings_dates)
    beat_rate = beat_count / total_with_data if total_with_data > 0 else None

    return {
        "next_date": next_day.isoformat() if next_day else None,
        "next_date_source": nearest.source if nearest else None,
        "next_date_status": "available" if nearest else "unavailable",
        "next_date_status_reason": None if nearest else "calendar_missing_and_no_future_earnings_dates",
        "next_date_is_estimate": nearest.is_estimate if nearest else None,
        "next_window": calendar.window,
        "days_until": (next_day - today).days if next_day else None,
        "verification": verification,
        "candidate_dates": {c.source: c.start.isoformat() for c in candidates if c.verified},
        "sources_failed": [name for name, view in views.items() if view.state == "failed"],
        "sources_stale": [name for name, view in views.items() if view.state == "stale"],
        "history": earnings_history,
        "beat_rate": safe_round(beat_rate, 2),
    }


def _build_earnings_history(earnings_dates: pd.DataFrame | None) -> tuple[list[dict[str, Any]], int, int]:
    """Reported quarters only, newest first (up to 8), plus beat-rate counters."""
    if earnings_dates is None or earnings_dates.empty:
        return [], 0, 0
    reported = [
        (ts, row)
        for ts, row in earnings_dates.iterrows()
        if isinstance(ts, pd.Timestamp) and safe_float(row.get("Reported EPS")) is not None
    ]
    reported.sort(key=lambda item: item[0], reverse=True)

    earnings_history: list[dict[str, Any]] = []
    beat_count = 0
    total_with_data = 0
    for ts, row in reported[:8]:
        estimate = safe_float(row.get("EPS Estimate"))
        actual = safe_float(row.get("Reported EPS"))
        surprise = None
        if estimate is not None and actual is not None and estimate != 0:
            surprise = (actual - estimate) / abs(estimate)
            total_with_data += 1
            if actual > estimate:
                beat_count += 1
        earnings_history.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "estimate": estimate,
                "actual": actual,
                "surprise": safe_round(surprise, 4),
            }
        )
    return earnings_history, beat_count, total_with_data


def _build_dividends(calendar_events: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Build dividend dates and yield fields."""
    # Use pre-fetched info for dividend data
    dividend_amount = safe_float(info.get("lastDividendValue"))
    annual_dividend = safe_float(info.get("dividendRate"))
    dividend_yield = safe_float(info.get("dividendYield"))
    # Convert yield to decimal if it's in percentage form
    if dividend_yield is not None and dividend_yield > 1:
        dividend_yield = dividend_yield / 100

    return {
        "ex_date": _utc_date(calendar_events.get("exDividendDate")),
        "pay_date": _utc_date(calendar_events.get("dividendDate")),
        "amount": dividend_amount,
        "annual": annual_dividend,
        "yield": safe_round(dividend_yield, 4),
    }


def _build_splits(ticker: Any) -> dict[str, str | None]:
    """Build last split metadata."""
    last_split_date = None
    last_split_ratio = None

    try:
        splits = read_attribute(ticker, lambda: ticker.splits)
        if splits is not None and len(splits) > 0:
            last_split = splits.iloc[-1]
            last_split_date = splits.index[-1].strftime("%Y-%m-%d")
            # Format ratio nicely
            ratio_val = float(last_split)
            if ratio_val >= 1:
                last_split_ratio = f"{int(ratio_val)}:1"
            else:
                last_split_ratio = f"1:{int(1/ratio_val)}"
    except Exception:
        pass

    splits_info = {
        "last_date": last_split_date,
        "last_ratio": last_split_ratio,
    }
    return splits_info


def _build_analyst(info: dict[str, Any]) -> dict[str, Any]:
    """Build analyst estimates from prefetched info."""
    analyst: dict[str, Any] = {
        "price_target": None,
        "recommendation": None,
        "num_analysts": None,
    }
    analyst_warning: str | None = None

    current_price = current_price_from_info(info)

    target_mean = safe_float(info.get("targetMeanPrice"))
    target_low = safe_float(info.get("targetLowPrice"))
    target_high = safe_float(info.get("targetHighPrice"))
    num_analysts = info.get("numberOfAnalystOpinions")

    if target_mean is not None:
        upside = (
            (target_mean - current_price) / current_price
            if current_price and current_price > 0
            else None
        )
        analyst["price_target"] = {
            "mean": target_mean,
            "low": target_low,
            "high": target_high,
            "upside": safe_round(upside, 4),
        }
        analyst["num_analysts"] = num_analysts

    rec = info.get("recommendationKey")
    if rec:
        analyst["recommendation"] = str(rec).lower()

    # Only warn if info was empty (fetch failed earlier)
    if not info:
        analyst_warning = "analyst_data_unavailable"

    if analyst_warning:
        analyst["warnings"] = [analyst_warning]

    return analyst
