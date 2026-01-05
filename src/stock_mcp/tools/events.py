"""Events calendar tool."""

from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd
import yfinance as yf

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

    # Get next earnings date
    next_earnings_date = None
    days_until_earnings = None

    if isinstance(calendar, dict):
        # Calendar might have 'Earnings Date' as a list
        earnings_date_val = calendar.get("Earnings Date")
        if earnings_date_val:
            if isinstance(earnings_date_val, list) and len(earnings_date_val) > 0:
                next_earnings_date = _format_date(earnings_date_val[0])
            else:
                next_earnings_date = _format_date(earnings_date_val)
    elif isinstance(calendar, pd.DataFrame) and len(calendar) > 0:
        # Some versions return DataFrame
        if "Earnings Date" in calendar.index:
            val = calendar.loc["Earnings Date"].iloc[0]
            next_earnings_date = _format_date(val)

    if next_earnings_date:
        try:
            earnings_dt = datetime.strptime(next_earnings_date, "%Y-%m-%d")
            days_until_earnings = (earnings_dt - datetime.now()).days
        except (ValueError, TypeError):
            pass

    # Calculate beat rate
    beat_rate = beat_count / total_with_data if total_with_data > 0 else None

    earnings = {
        "next_date": next_earnings_date,
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

    try:
        info = ticker.info
        dividend_amount = _safe_float(info.get("lastDividendValue"))
        annual_dividend = _safe_float(info.get("dividendRate"))
        dividend_yield = _safe_float(info.get("dividendYield"))
        # Convert yield to decimal if it's in percentage form
        if dividend_yield is not None and dividend_yield > 1:
            dividend_yield = dividend_yield / 100
    except Exception:
        pass

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
