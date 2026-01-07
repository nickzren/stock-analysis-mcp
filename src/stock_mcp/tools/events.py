"""Events calendar tool."""

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd

from stock_mcp.data.yfinance_client import fetch_ticker
from stock_mcp.utils.provenance import build_error_response, build_meta, build_provenance


async def events_calendar(symbol: str) -> dict[str, Any]:
    """
    Get upcoming events and historical earnings for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with earnings, dividends, and splits information
    """
    start_time = perf_counter()
    normalized_symbol = symbol.upper().strip()

    try:
        ticker = await fetch_ticker(symbol)
    except Exception as e:
        return build_error_response(
            error_type="data_unavailable",
            message=f"Failed to fetch data: {e}",
            symbol=symbol,
        )

    # Fetch ticker.info once for reuse (avoids duplicate API calls)
    try:
        info = ticker.info
    except Exception:
        info = {}

    # Get calendar data
    try:
        calendar = ticker.calendar
    except Exception:
        calendar = {}

    # Get earnings history
    earnings_history: list[dict[str, Any]] = []
    beat_count = 0
    total_with_data = 0

    try:
        # Get earnings dates with estimates/actuals
        earnings_dates = ticker.earnings_dates
        if earnings_dates is not None and len(earnings_dates) > 0:
            # earnings_dates is a DataFrame with columns like 'EPS Estimate', 'Reported EPS', etc.
            for date, row in earnings_dates.head(8).iterrows():
                estimate = _safe_float(row.get("EPS Estimate"))
                actual = _safe_float(row.get("Reported EPS"))

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
                        "surprise": _safe_round(surprise, 4),
                    }
                )
    except Exception:
        pass

    # Get next earnings date with multiple fallback sources
    next_earnings_date = None
    days_until_earnings = None
    earnings_date_source: str | None = None
    earnings_date_status: str = "unavailable"
    earnings_date_status_reason: str | None = None

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
    elif isinstance(calendar, pd.DataFrame) and len(calendar) > 0:
        # Some versions return DataFrame
        if "Earnings Date" in calendar.index:
            val = calendar.loc["Earnings Date"].iloc[0]
            next_earnings_date = _format_date(val)
            if next_earnings_date:
                earnings_date_source = "calendar_dataframe"
                earnings_date_status = "available"

    # Source 2: Fallback to earnings_dates (future dates)
    if next_earnings_date is None:
        try:
            earnings_dates = ticker.earnings_dates
            if earnings_dates is not None and len(earnings_dates) > 0:
                now = datetime.now()
                for date, _row in earnings_dates.iterrows():
                    if isinstance(date, pd.Timestamp):
                        dt = date.to_pydatetime().replace(tzinfo=None)
                    else:
                        try:
                            dt = datetime.strptime(str(date)[:10], "%Y-%m-%d")
                        except ValueError:
                            continue
                    # Only use future dates
                    if dt > now:
                        next_earnings_date = _format_date(date)
                        earnings_date_source = "earnings_dates"
                        earnings_date_status = "available"
                        break
        except Exception:
            pass

    # Source 3: Fallback to info earningsQuarterlyGrowth dates (rare)
    # (yfinance sometimes has this)
    if next_earnings_date is None and info:
        # Some tickers have earningsDate in info
        info_earnings = info.get("earningsTimestamp")
        if info_earnings:
            try:
                dt = datetime.fromtimestamp(info_earnings)
                if dt > datetime.now():
                    next_earnings_date = dt.strftime("%Y-%m-%d")
                    earnings_date_source = "info_timestamp"
                    earnings_date_status = "available"
            except (ValueError, TypeError, OSError):
                pass

    # Compute days until earnings if we have a date
    if next_earnings_date:
        try:
            earnings_dt = datetime.strptime(next_earnings_date, "%Y-%m-%d")
            days_until_earnings = (earnings_dt - datetime.now()).days
        except (ValueError, TypeError):
            pass
    else:
        # Provide structured reason for unavailability
        earnings_date_status_reason = "calendar_missing_and_no_future_earnings_dates"

    # Calculate beat rate
    beat_rate = beat_count / total_with_data if total_with_data > 0 else None

    earnings = {
        "next_date": next_earnings_date,
        "next_date_source": earnings_date_source,  # calendar/earnings_dates/info_timestamp
        "next_date_status": earnings_date_status,  # available/unavailable
        "next_date_status_reason": earnings_date_status_reason if earnings_date_status == "unavailable" else None,
        "days_until": days_until_earnings,
        "history": earnings_history,
        "beat_rate": _safe_round(beat_rate, 2),
    }

    # Dividends
    ex_date = None
    pay_date = None
    dividend_amount = None
    annual_dividend = None
    dividend_yield = None

    if isinstance(calendar, dict):
        ex_date = _format_date(calendar.get("Ex-Dividend Date"))
        div_date = calendar.get("Dividend Date")
        if div_date:
            pay_date = _format_date(div_date)

    # Use pre-fetched info for dividend data
    dividend_amount = _safe_float(info.get("lastDividendValue"))
    annual_dividend = _safe_float(info.get("dividendRate"))
    dividend_yield = _safe_float(info.get("dividendYield"))
    # Convert yield to decimal if it's in percentage form
    if dividend_yield is not None and dividend_yield > 1:
        dividend_yield = dividend_yield / 100

    dividends = {
        "ex_date": ex_date,
        "pay_date": pay_date,
        "amount": dividend_amount,
        "annual": annual_dividend,
        "yield": _safe_round(dividend_yield, 4),
    }

    # Splits
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

    # Analyst estimates (using pre-fetched info)
    analyst: dict[str, Any] = {
        "price_target": None,
        "recommendation": None,
        "num_analysts": None,
    }
    analyst_warnings: list[str] = []

    current_price = _safe_float(info.get("regularMarketPrice")) or _safe_float(info.get("currentPrice"))

    target_mean = _safe_float(info.get("targetMeanPrice"))
    target_low = _safe_float(info.get("targetLowPrice"))
    target_high = _safe_float(info.get("targetHighPrice"))
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
            "upside": _safe_round(upside, 4),
        }
        analyst["num_analysts"] = num_analysts

    rec = info.get("recommendationKey")
    if rec:
        analyst["recommendation"] = str(rec).lower()

    # Only warn if info was empty (fetch failed earlier)
    if not info:
        analyst_warnings.append("analyst_data_unavailable")

    if analyst_warnings:
        analyst["warnings"] = analyst_warnings

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("events_calendar", duration_ms),
        "data_provenance": {
            "events": build_provenance(
                source="yfinance",
                as_of=datetime.utcnow().isoformat() + "Z",
            ),
        },
        "symbol": normalized_symbol,
        "earnings": earnings,
        "dividends": dividends,
        "splits": splits_info,
        "analyst": analyst,
    }


def _format_date(value: Any) -> str | None:
    """Format a date value to YYYY-MM-DD string."""
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, str):
        # Try to parse and reformat
        try:
            dt = pd.to_datetime(value)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return value

    return None


def _safe_float(value: Any) -> float | None:
    """Convert to float or return None."""
    if value is None:
        return None
    try:
        result = float(value)
        if pd.isna(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def _safe_round(value: float | None, decimals: int) -> float | None:
    """Round to decimals or return None."""
    if value is None:
        return None
    return round(value, decimals)
