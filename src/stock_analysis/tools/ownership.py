"""Insider and institutional ownership analysis tool."""

from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

import pandas as pd

from stock_analysis.data.yfinance_client import fetch_ticker
from stock_analysis.utils.helpers import safe_float, safe_round
from stock_analysis.utils.provenance import (
    FetchError,
    build_meta,
    build_provenance,
    fetch_or_error,
    utcnow_isoformat_z,
)
from stock_analysis.utils.sanitize import sanitize_text


async def ownership_analysis(symbol: str) -> dict[str, Any]:
    """
    Analyze insider and institutional ownership for a symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with insider activity, institutional holders, and ownership percentages
    """
    start_time = perf_counter()
    normalized_symbol = symbol.upper().strip()
    warnings: list[str] = []

    try:
        ticker = await fetch_or_error(fetch_ticker(symbol), symbol)
    except FetchError as fe:
        return fe.response

    # --- Insider activity ---
    insider_activity = _build_insider_activity(ticker, warnings)

    # --- Institutional holders ---
    institutional = _build_institutional(ticker, warnings)

    # --- Major holders (ownership percentages) ---
    _apply_major_holders(ticker, insider_activity, institutional, warnings)

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("ownership_analysis", duration_ms),
        "data_provenance": {
            "ownership": build_provenance(
                source="yfinance",
                as_of=utcnow_isoformat_z(),
            ),
        },
        "symbol": normalized_symbol,
        "insider_activity": insider_activity,
        "institutional": institutional,
        "warnings": warnings or None,
    }


def _build_insider_activity(ticker: Any, warnings: list[str]) -> dict[str, Any]:
    """Extract and aggregate insider transaction data."""
    result: dict[str, Any] = {
        "net_shares_3m": None,
        "net_shares_6m": None,
        "net_shares_12m": None,
        "sentiment": "neutral",
        "recent_transactions": [],
    }

    try:
        txns = ticker.insider_transactions
        if txns is None or not isinstance(txns, pd.DataFrame) or txns.empty:
            warnings.append("insider_transactions_unavailable")
            return result

        now = datetime.now()
        cutoff_3m = now - timedelta(days=90)
        cutoff_6m = now - timedelta(days=180)
        cutoff_12m = now - timedelta(days=365)

        # Identify the date column
        date_col = _find_column(txns, ["startDate", "Start Date", "Date"])
        shares_col = _find_column(txns, ["shares", "Shares"])
        text_col = _find_column(txns, ["text", "Text", "Transaction"])
        insider_col = _find_column(txns, ["insider", "Insider", "Name", "Insider Trading"])
        value_col = _find_column(txns, ["value", "Value"])

        if date_col is None or shares_col is None:
            warnings.append("insider_transactions_columns_unrecognized")
            return result

        # Parse dates and compute net shares per period
        net_3m = 0
        net_6m = 0
        net_12m = 0
        scored_rows: list[tuple[float, int]] = []  # (abs_shares_or_value, row_index)

        for idx in range(len(txns)):
            row = txns.iloc[idx]
            txn_date = _parse_date(row.get(date_col))
            if txn_date is None:
                continue

            shares_val = safe_float(row.get(shares_col))
            if shares_val is None:
                continue

            # Determine sign: negative shares = sale, positive = purchase
            # Some yfinance versions use text like "Sale" / "Purchase"
            txn_text = str(row.get(text_col, "")).lower() if text_col else ""
            signed_shares = _sign_shares(shares_val, txn_text)

            if txn_date >= cutoff_3m:
                net_3m += signed_shares
            if txn_date >= cutoff_6m:
                net_6m += signed_shares
            if txn_date >= cutoff_12m:
                net_12m += signed_shares

            # Score for "largest" selection: prefer value, fallback to abs(shares)
            score = abs(safe_float(row.get(value_col)) or 0) if value_col else 0
            if score == 0:
                score = abs(shares_val)
            scored_rows.append((score, idx))

        result["net_shares_3m"] = net_3m
        result["net_shares_6m"] = net_6m
        result["net_shares_12m"] = net_12m

        # Derive sentiment from 3-month net
        if net_3m > 0:
            result["sentiment"] = "buying"
        elif net_3m < 0:
            result["sentiment"] = "selling"
        else:
            result["sentiment"] = "neutral"

        # Top 5 largest transactions
        scored_rows.sort(key=lambda x: x[0], reverse=True)
        for _, row_idx in scored_rows[:5]:
            row = txns.iloc[row_idx]
            txn_date = _parse_date(row.get(date_col))
            shares_val = safe_float(row.get(shares_col))
            txn_text = str(row.get(text_col, "")) if text_col else None
            insider_name = sanitize_text(str(row.get(insider_col))) if insider_col and row.get(insider_col) is not None else None
            txn_value = safe_float(row.get(value_col)) if value_col else None

            result["recent_transactions"].append({
                "date": txn_date.strftime("%Y-%m-%d") if txn_date else None,
                "insider": insider_name,
                "text": sanitize_text(txn_text, max_length=200) if txn_text else None,
                "shares": safe_round(shares_val, 0),
                "value": safe_round(txn_value, 2),
            })

    except Exception:
        warnings.append("insider_transactions_error")

    return result


def _build_institutional(ticker: Any, warnings: list[str]) -> dict[str, Any]:
    """Extract top institutional holders."""
    result: dict[str, Any] = {
        "top_holders": [],
        "total_institutional_pct": None,
    }

    try:
        holders = ticker.institutional_holders
        if holders is None or not isinstance(holders, pd.DataFrame) or holders.empty:
            warnings.append("institutional_holders_unavailable")
            return result

        name_col = _find_column(holders, ["Holder", "holder", "Name", "name"])
        shares_col = _find_column(holders, ["Shares", "shares"])
        pct_col = _find_column(holders, ["% Out", "pctHeld", "pct_held", "Percent"])
        value_col = _find_column(holders, ["Value", "value"])
        date_col = _find_column(holders, ["Date Reported", "dateReported", "Date"])

        for idx in range(min(len(holders), 10)):
            row = holders.iloc[idx]
            holder_name = sanitize_text(str(row.get(name_col))) if name_col and row.get(name_col) is not None else None
            holder_shares = safe_float(row.get(shares_col)) if shares_col else None
            holder_pct = safe_float(row.get(pct_col)) if pct_col else None
            holder_value = safe_float(row.get(value_col)) if value_col else None
            reported_date = _parse_date(row.get(date_col)) if date_col else None

            result["top_holders"].append({
                "name": holder_name,
                "shares": safe_round(holder_shares, 0),
                "pct_held": safe_round(holder_pct, 4),
                "value": safe_round(holder_value, 2),
                "date_reported": reported_date.strftime("%Y-%m-%d") if reported_date else None,
            })

    except Exception:
        warnings.append("institutional_holders_error")

    return result


def _apply_major_holders(
    ticker: Any,
    insider_activity: dict[str, Any],
    institutional: dict[str, Any],
    warnings: list[str],
) -> None:
    """Extract ownership percentages from major_holders and apply to results."""
    try:
        major = ticker.major_holders
        if major is None or not isinstance(major, pd.DataFrame) or major.empty:
            warnings.append("major_holders_unavailable")
            return

        # major_holders is typically a 2-column DataFrame:
        #   column 0: percentage value (e.g., "1.50%")
        #   column 1: description (e.g., "% of Shares Held by All Insider")
        for idx in range(len(major)):
            row = major.iloc[idx]
            description = str(row.iloc[1]).lower() if len(row) > 1 else ""
            raw_value = row.iloc[0] if len(row) > 0 else None

            pct = _parse_pct(raw_value)

            if "insider" in description:
                insider_activity["insider_pct"] = safe_round(pct, 4)
            elif "institution" in description:
                institutional["total_institutional_pct"] = safe_round(pct, 4)

    except Exception:
        warnings.append("major_holders_error")


# --- Private helpers ---


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_date(value: Any) -> datetime | None:
    """Parse a date value to datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().replace(tzinfo=None)
    try:
        return pd.to_datetime(value).to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


def _parse_pct(value: Any) -> float | None:
    """Parse a percentage value that may be a string like '1.50%' or a float."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        try:
            return float(cleaned) / 100
        except ValueError:
            return None
    f = safe_float(value)
    if f is None:
        return None
    # If value > 1, assume it's already in percent form
    if f > 1:
        return f / 100
    return f


def _sign_shares(shares: float, txn_text: str) -> float:
    """
    Determine signed share count from raw shares and transaction text.

    Positive = purchase/acquisition, negative = sale/disposition.
    """
    text_lower = txn_text.lower()
    if "sale" in text_lower or "sell" in text_lower or "disposition" in text_lower:
        return -abs(shares)
    if "purchase" in text_lower or "buy" in text_lower or "acquisition" in text_lower:
        return abs(shares)
    # If text doesn't indicate direction, use sign as-is
    return shares
