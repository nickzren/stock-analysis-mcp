"""Options-implied expected move through earnings (pure module).

Never gates: every failure path returns None — options data is never
required for a valid setup (step-1 locked ruling).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from stock_analysis.utils.helpers import safe_float, safe_round


def select_event_expiration(
    expirations: Sequence[str],
    earnings_date: date,
    today: date,
) -> str | None:
    """First expiration on or after the earnings date (and not expired)."""
    parsed: list[tuple[date, str]] = []
    for raw in expirations:
        try:
            parsed.append((date.fromisoformat(str(raw)[:10]), str(raw)))
        except ValueError:
            continue
    for exp, raw in sorted(parsed):
        if exp >= today and exp >= earnings_date:
            return raw
    return None


def compute_expected_move(
    calls: pd.DataFrame | None,
    puts: pd.DataFrame | None,
    spot: float | None,
) -> dict[str, Any] | None:
    """ATM-straddle expected move: (call + put) / spot at the nearest common strike."""
    if spot is None or spot <= 0:
        return None
    if calls is None or puts is None or calls.empty or puts.empty:
        return None
    if "strike" not in calls.columns or "strike" not in puts.columns:
        return None
    call_strikes = set(pd.to_numeric(calls["strike"], errors="coerce").dropna())
    put_strikes = set(pd.to_numeric(puts["strike"], errors="coerce").dropna())
    common = call_strikes & put_strikes
    if not common:
        return None
    strike = min(common, key=lambda s: (abs(s - spot), s))

    call_price, call_basis = _leg_price(calls, strike)
    put_price, put_basis = _leg_price(puts, strike)
    if call_price is None or put_price is None:
        return None
    total = call_price + put_price
    basis = (
        "atm_straddle_mid"
        if call_basis == "mid" and put_basis == "mid"
        else "atm_straddle_last"
    )
    return {
        "pct": safe_round(total / spot, 4),
        "dollars": safe_round(total, 2),
        "strike": float(strike),
        "basis": basis,
    }


def _leg_price(chain: pd.DataFrame, strike: float) -> tuple[float | None, str]:
    rows = chain[pd.to_numeric(chain["strike"], errors="coerce") == strike]
    if rows.empty:
        return None, ""
    row = rows.iloc[0]
    bid = _positive(row.get("bid"))
    ask = _positive(row.get("ask"))
    last = _positive(row.get("lastPrice"))
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0, "mid"
    if last is not None:
        return last, "last"
    return None, ""


def _positive(value: Any) -> float | None:
    number = safe_float(value)
    return number if number is not None and number > 0 else None
