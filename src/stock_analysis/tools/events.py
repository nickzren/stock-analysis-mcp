"""Events calendar tool."""

import contextlib
from datetime import UTC, date, datetime
from time import perf_counter
from typing import Any

import pandas as pd
import pytz

from stock_analysis.data.yfinance_client import fetch_info, fetch_ticker
from stock_analysis.utils.helpers import current_price_from_info, safe_float, safe_round
from stock_analysis.utils.provenance import (
    FetchError,
    build_meta,
    build_provenance,
    fetch_or_error,
    utcnow_isoformat_z,
)

_ET = pytz.timezone("America/New_York")


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

    calendar, calendar_ok = _fetch_calendar(ticker)
    earnings_dates, earnings_dates_ok = _fetch_earnings_dates(ticker)

    read_failures: set[str] = set()
    if not calendar_ok:
        read_failures.add("calendar")
    if not earnings_dates_ok:
        read_failures.add("earnings_dates")
    if not info_ok:
        read_failures.add("info")

    earnings = _build_earnings(calendar, earnings_dates, info, now, read_failures)
    dividends = _build_dividends(calendar, info)
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


def _fetch_calendar(ticker: Any) -> tuple[Any, bool]:
    """(payload, ok). ok=False only when the read raised."""
    try:
        return ticker.calendar, True
    except Exception:
        return {}, False


def _fetch_earnings_dates(ticker: Any) -> tuple[Any, bool]:
    """(payload, ok). ok=False only when the read raised."""
    try:
        return ticker.earnings_dates, True
    except Exception:
        return None, False


def _build_earnings(
    calendar: Any,
    earnings_dates: Any,
    info: dict[str, Any],
    now: datetime,
    read_failures: set[str],
) -> dict[str, Any]:
    """Build earnings history and next-earnings metadata."""
    earnings_history, beat_count, total_with_data = _build_earnings_history(earnings_dates)
    next_earnings_date, days_until_earnings, earnings_date_source, earnings_date_status, parse_failures = (
        _resolve_next_earnings_date(calendar, earnings_dates, info, now)
    )
    sources_failed = sorted(read_failures | parse_failures)

    earnings_date_status_reason = None
    if next_earnings_date is None:
        earnings_date_status_reason = "calendar_missing_and_no_future_earnings_dates"

    beat_rate = beat_count / total_with_data if total_with_data > 0 else None

    return {
        "next_date": next_earnings_date,
        "next_date_source": earnings_date_source,
        "next_date_status": earnings_date_status,
        "next_date_status_reason": earnings_date_status_reason if earnings_date_status == "unavailable" else None,
        "days_until": days_until_earnings,
        "sources_failed": sources_failed,
        "history": earnings_history,
        "beat_rate": safe_round(beat_rate, 2),
    }


def _build_earnings_history(earnings_dates: Any) -> tuple[list[dict[str, Any]], int, int]:
    """Build recent earnings history plus beat-rate counters."""
    earnings_history: list[dict[str, Any]] = []
    beat_count = 0
    total_with_data = 0

    try:
        if earnings_dates is not None and len(earnings_dates) > 0:
            # earnings_dates is a DataFrame with columns like 'EPS Estimate', 'Reported EPS', etc.
            for date, row in earnings_dates.head(8).iterrows():
                estimate = safe_float(row.get("EPS Estimate"))
                actual = safe_float(row.get("Reported EPS"))

                surprise = None
                if estimate is not None and actual is not None and estimate != 0:
                    surprise = (actual - estimate) / abs(estimate)
                    total_with_data += 1
                    if actual > estimate:
                        beat_count += 1

                # Convert date to string
                if isinstance(date, pd.Timestamp):
                    date_str = date.strftime("%Y-%m-%d")
                else:
                    date_str = str(date)

                earnings_history.append(
                    {
                        "date": date_str,
                        "estimate": estimate,
                        "actual": actual,
                        "surprise": safe_round(surprise, 4),
                    }
                )
    except Exception:
        pass

    return earnings_history, beat_count, total_with_data


def _resolve_next_earnings_date(
    calendar: Any,
    earnings_dates: Any,
    info: dict[str, Any],
    now: datetime,
) -> tuple[str | None, int | None, str | None, str, set[str]]:
    """Resolve the next earnings date from calendar, earnings dates, then info."""
    next_earnings_date = None
    days_until_earnings = None
    earnings_date_source: str | None = None
    earnings_date_status: str = "unavailable"
    parse_failures: set[str] = set()

    # Source 1: Calendar (most reliable when present)
    if isinstance(calendar, dict):
        # Calendar might have 'Earnings Date' as a list
        earnings_date_val = calendar.get("Earnings Date")
        if earnings_date_val:
            if isinstance(earnings_date_val, list) and len(earnings_date_val) > 0:
                next_earnings_date = _format_date(earnings_date_val[0])
            else:
                next_earnings_date = _format_date(earnings_date_val)
            if next_earnings_date:
                earnings_date_source = "calendar"
                earnings_date_status = "available"
            else:
                parse_failures.add("calendar")
    elif isinstance(calendar, pd.DataFrame) and len(calendar) > 0:
        # Some versions return DataFrame
        if "Earnings Date" in calendar.index:
            val = calendar.loc["Earnings Date"].iloc[0]
            next_earnings_date = _format_date(val)
            if next_earnings_date:
                earnings_date_source = "calendar_dataframe"
                earnings_date_status = "available"
            else:
                parse_failures.add("calendar")
    elif calendar is not None and not isinstance(calendar, pd.DataFrame):
        parse_failures.add("calendar")

    # Source 2: Fallback to earnings_dates (future dates), filtered by ET
    # calendar date (via the injected `now`) so same-day earnings survive
    # instead of being floored to "past."
    if next_earnings_date is None:
        if earnings_dates is not None and not isinstance(earnings_dates, pd.DataFrame):
            parse_failures.add("earnings_dates")
        elif earnings_dates is not None and len(earnings_dates) > 0:
            cutoff = now.date()
            rows_seen = 0
            rows_unparseable = 0
            try:
                for idx, _row in earnings_dates.iterrows():
                    rows_seen += 1
                    if isinstance(idx, pd.Timestamp):
                        ts = idx if idx.tzinfo is not None else idx.tz_localize("UTC")
                        row_date = ts.tz_convert("America/New_York").date()
                    else:
                        try:
                            row_date = date.fromisoformat(str(idx)[:10])
                        except ValueError:
                            rows_unparseable += 1
                            continue
                    if row_date >= cutoff:
                        next_earnings_date = row_date.isoformat()
                        earnings_date_source = "earnings_dates"
                        earnings_date_status = "available"
                        break
            except Exception:
                parse_failures.add("earnings_dates")
            if rows_seen > 0 and rows_unparseable == rows_seen:
                parse_failures.add("earnings_dates")

    # Source 3: Fallback to info earningsTimestamp (rare)
    if next_earnings_date is None and info:
        if not isinstance(info, dict):
            parse_failures.add("info")
        else:
            info_earnings = info.get("earningsTimestamp")
            if info_earnings:
                try:
                    dt = datetime.fromtimestamp(info_earnings, tz=UTC)
                    info_date = dt.astimezone(_ET).date()
                    if info_date >= now.date():
                        next_earnings_date = info_date.isoformat()
                        earnings_date_source = "info_timestamp"
                        earnings_date_status = "available"
                except (ValueError, TypeError, OSError, OverflowError):
                    parse_failures.add("info")

    # Compute days until earnings if we have a date
    if next_earnings_date:
        with contextlib.suppress(ValueError):
            days_until_earnings = (date.fromisoformat(next_earnings_date) - now.date()).days

    return (
        next_earnings_date,
        days_until_earnings,
        earnings_date_source,
        earnings_date_status,
        parse_failures,
    )


def _build_dividends(calendar: Any, info: dict[str, Any]) -> dict[str, Any]:
    """Build dividend dates and yield fields."""
    ex_date = None
    pay_date = None

    if isinstance(calendar, dict):
        ex_date = _format_date(calendar.get("Ex-Dividend Date"))
        div_date = calendar.get("Dividend Date")
        if div_date:
            pay_date = _format_date(div_date)

    # Use pre-fetched info for dividend data
    dividend_amount = safe_float(info.get("lastDividendValue"))
    annual_dividend = safe_float(info.get("dividendRate"))
    dividend_yield = safe_float(info.get("dividendYield"))
    # Convert yield to decimal if it's in percentage form
    if dividend_yield is not None and dividend_yield > 1:
        dividend_yield = dividend_yield / 100

    return {
        "ex_date": ex_date,
        "pay_date": pay_date,
        "amount": dividend_amount,
        "annual": annual_dividend,
        "yield": safe_round(dividend_yield, 4),
    }


def _build_splits(ticker: Any) -> dict[str, str | None]:
    """Build last split metadata."""
    last_split_date = None
    last_split_ratio = None

    try:
        splits = ticker.splits
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


def _format_date(value: Any) -> str | None:
    """Format a date value to YYYY-MM-DD string."""
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, str):
        # Try to parse and reformat
        try:
            dt = pd.to_datetime(value)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return value

    return None
