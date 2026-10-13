"""Daily and 5m frames for the Step 2 freshness tests (WP2.0).

Ported from the freshness research probes (docs/plans/step2-v2-research/freshness.md:
probe_price_basis.py, probe_technicals_fields.py, probe_risk_nan_latest.py and
v3_corrected_gate.py), so the evidence lives in the repo. Row ids (R2, R16, ...) are the
annex truth-table ids.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytz

ET = pytz.timezone("America/New_York")
NAN = float("nan")


def at(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return ET.localize(datetime(y, m, d, hh, mm))


# The truth table's regular-session clock: Tue 11:00, last completed session Mon 2026-10-12.
REGULAR = at(2026, 10, 13, 11, 0)


def daily(end: str, last_close: float | None = 101.0, n: int = 260) -> pd.DataFrame:
    """``n`` business-day OHLCV bars ending on ``end``. ``last_close=None`` makes the last
    bar empty (every price and the volume NaN), as yfinance's ``keepna`` keeps it."""
    dates = pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d").tolist()
    closes = [100.0 + (i % 5) * 0.2 for i in range(n)]
    closes[-1] = NAN if last_close is None else last_close
    volume = [1e6] * n
    if last_close is None:
        volume[-1] = NAN
    df = pd.DataFrame({"date": dates, "open": closes, "high": [c + 0.5 for c in closes],
                       "low": [c - 0.5 for c in closes], "close": closes, "volume": volume})
    df.attrs.update({"fetched_at": None, "currency": "USD", "instrument_type": "EQUITY"})
    return df


def five_min(now: datetime, bars: Sequence[tuple[float, float]]) -> pd.DataFrame:
    """5m bars given as ``(minutes before now, close)``, oldest first (negative = future)."""
    times = [(now.astimezone(UTC) - timedelta(minutes=age)).isoformat() for age, _ in bars]
    closes = [close for _, close in bars]
    return pd.DataFrame({"date": times, "open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": [1.0] * len(bars)})
