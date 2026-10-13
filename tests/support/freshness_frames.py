"""Daily and 5m frames for the Step 2 freshness tests (WP2.0, extended in WP2.1).

Ported from the freshness research probes (docs/plans/step2-v2-research/freshness.md:
probe_price_basis.py, probe_technicals_fields.py, probe_risk_nan_latest.py,
v3_corrected_gate.py and v5_corrected_truth_table.py), so the evidence lives in the repo.
Row ids (R2, R16, ...) are the annex truth-table ids.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytz

from stock_analysis.utils.freshness import build_freshness
from stock_analysis.utils.market_calendar import classify_session
from stock_analysis.utils.price_basis import (
    build_daily_history,
    last_valid_bar_date,
    latest_intraday_bar,
    latest_session_close,
    select_price,
)
from tests.support import decision_matrix as dm

ET = pytz.timezone("America/New_York")
NAN = float("nan")


def at(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return ET.localize(datetime(y, m, d, hh, mm))


# The truth table's regular-session clock: Tue 11:00, last completed session Mon 2026-10-12.
REGULAR = at(2026, 10, 13, 11, 0)
OPEN_0931 = at(2026, 10, 13, 9, 31)
PRE_0800 = at(2026, 10, 13, 8)
AFTER_1700 = at(2026, 10, 13, 17)
OVERNIGHT_0300 = at(2026, 10, 13, 3)
LATE_2100 = at(2026, 10, 13, 21)
SATURDAY = at(2026, 10, 17, 12)
THANKSGIVING = at(2026, 11, 26, 11)
EARLY_CLOSE_1230 = at(2026, 11, 27, 12, 30)
EARLY_CLOSE_1330 = at(2026, 11, 27, 13, 30)
MONDAY_1100 = at(2026, 10, 12, 11)
CLOCK_ONLY_2031 = at(2031, 1, 1, 11)  # past the static calendar: the clock-only fallback


def daily(end: str, last_close: float | None = 101.0, n: int = 260, *, nan_dates: Sequence[str] = (),
          drop_dates: Sequence[str] = (), extra_rows: Sequence[tuple[str, float]] = ()) -> pd.DataFrame:
    """``n`` business-day OHLCV bars ending on ``end``. ``last_close=None`` makes the last
    bar empty (every price and the volume NaN), as yfinance's ``keepna`` keeps it; ``nan_dates``
    empties other rows the same way. ``drop_dates`` removes rows; ``extra_rows`` appends
    ``(date, close)`` rows (a weekend- or holiday-dated vendor row)."""
    dates = pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d").tolist()
    closes = [100.0 + (i % 5) * 0.2 for i in range(n)]
    closes[-1] = NAN if last_close is None else last_close
    volume = [1e6] * n
    if last_close is None:
        volume[-1] = NAN
    for i, day in enumerate(dates):
        if day in nan_dates:
            closes[i], volume[i] = NAN, NAN
    rows = [(d, c, v) for d, c, v in zip(dates, closes, volume, strict=True) if d not in drop_dates]
    rows += [(d, c, 1e6) for d, c in extra_rows]
    df = pd.DataFrame({"date": [r[0] for r in rows], "open": [r[1] for r in rows],
                       "high": [r[1] + 0.5 for r in rows], "low": [r[1] - 0.5 for r in rows],
                       "close": [r[1] for r in rows], "volume": [r[2] for r in rows]})
    df.attrs.update({"fetched_at": None, "currency": "USD", "instrument_type": "EQUITY"})
    return df


def five_min(now: datetime, bars: Sequence[tuple[float, float]]) -> pd.DataFrame:
    """5m bars given as ``(minutes before now, close)``, oldest first (negative = future)."""
    times = [(now.astimezone(UTC) - timedelta(minutes=age)).isoformat() for age, _ in bars]
    closes = [close for _, close in bars]
    return pd.DataFrame({"date": times, "open": closes, "high": closes, "low": closes,
                         "close": closes, "volume": [1.0] * len(bars)})


def fresh(now: datetime, minutes: float = 2) -> pd.DataFrame:
    return five_min(now, [(minutes, 101.0)])


DERIVE: Any = object()  # TruthRow.history default: technicals' own daily_history from the 1d frame


@dataclass(frozen=True)
class TruthRow:
    """One freshness truth-table row (annex v5_corrected_truth_table.py) and its expected result."""

    now: datetime
    frames: Callable[[], tuple[pd.DataFrame | None, pd.DataFrame | None]]  # (technicals' 1d, the 5m probe)
    codes: tuple[str, ...]
    expected_session: str
    sessions_behind: int | None
    price_source: str | None
    flags: tuple[str, ...] = ()
    card_reason: str | None = None  # the card's build_freshness reason_code on the same frames (after D-12)
    shared: bool = True  # False: analyze-only provenance faults outside the annex's 36 frame rows
    info: dict[str, Any] = field(default_factory=lambda: {"price": 99.0})
    history: Any = DERIVE  # a daily_history override; None = absent from the technicals payload


RG, AH, PM = REGULAR, AFTER_1700, PRE_0800
E_MON, E_TUE, E_FRI, E_WED = "2026-10-12", "2026-10-13", "2026-10-16", "2026-11-25"
INTRA, DAILY, INFO = "intraday_quote", "daily_close", "info_quote_fallback"
FORMING = ("includes_forming_bar",)
STALE_PAIR = ("daily_close_stale", "daily_history_stale")
UNAVAILABLE_PAIR = ("info_quote_fallback", "daily_history_unavailable")

TRUTH_TABLE: dict[str, TruthRow] = {
    # regular session (Tue 2026-10-13 11:00, E = Mon 10-12)
    "R1": TruthRow(RG, lambda: (daily(E_TUE), fresh(RG)), (), E_MON, 0, INTRA, FORMING),
    "R2": TruthRow(RG, lambda: (daily(E_MON), fresh(RG)), (), E_MON, 0, INTRA),
    "R9": TruthRow(RG, lambda: (daily(E_MON), five_min(RG, [(6, 101.0), (1, NAN)])), (), E_MON, 0, INTRA,
                   card_reason="nonfinite_close"),
    "R16": TruthRow(RG, lambda: (daily(E_MON), five_min(RG, [(6, 101.0), (1, 0.0)])), (), E_MON, 0, INTRA,
                    card_reason="nonpositive_close"),
    "R3": TruthRow(RG, lambda: (daily("2026-10-08"), fresh(RG)), ("daily_history_stale",), E_MON, 2, INTRA,
                   card_reason="stale_daily"),
    "R4": TruthRow(RG, lambda: (daily(E_TUE, None), fresh(RG)), ("daily_latest_bar_invalid",), E_MON, 0, INTRA,
                   card_reason="stale_daily"),
    "R10": TruthRow(RG, lambda: (daily(E_TUE, 0.0), fresh(RG)), ("daily_latest_bar_invalid",), E_MON, 0, INTRA,
                    card_reason="stale_daily"),
    "R11": TruthRow(RG, lambda: (daily(E_TUE, nan_dates=(E_MON,)), fresh(RG)), ("daily_history_stale",), E_MON, 1,
                    INTRA, FORMING),
    "R11b": TruthRow(RG, lambda: (daily(E_TUE, drop_dates=(E_MON,)), fresh(RG)), ("daily_history_stale",), E_MON,
                     1, INTRA, FORMING),
    "R12": TruthRow(RG, lambda: (daily(E_MON, None), fresh(RG)), ("daily_latest_bar_invalid", "daily_history_stale"),
                    E_MON, 1, INTRA, card_reason="stale_daily"),
    "R14": TruthRow(RG, lambda: (daily("2026-10-14"), fresh(RG)), ("daily_future_bar",), E_MON, 0, INTRA,
                    card_reason="stale_daily"),
    "W1": TruthRow(MONDAY_1100, lambda: (daily("2026-10-09", extra_rows=(("2026-10-10", 100.0),)),
                                         fresh(MONDAY_1100)),
                   ("daily_bar_after_last_session",), "2026-10-09", 0, INTRA),
    "R5": TruthRow(RG, lambda: (daily(E_TUE), fresh(RG, 40)), ("intraday_quote_stale",), E_MON, 0, DAILY, FORMING,
                   card_reason="stale_quote"),
    "R6": TruthRow(RG, lambda: (daily(E_MON), fresh(RG, 40)), ("intraday_quote_stale",), E_MON, 0, DAILY,
                   card_reason="stale_quote"),
    # the last 5m bar is Mon 15:55, 1056 minutes before 09:31
    "R15": TruthRow(OPEN_0931, lambda: (daily(E_MON), five_min(OPEN_0931, [(1056, 101.0)])),
                    ("intraday_quote_stale",), E_MON, 0, DAILY, card_reason="stale_quote"),
    "R7": TruthRow(RG, lambda: (daily(E_TUE), None), ("intraday_quote_unavailable",), E_MON, 0, DAILY, FORMING,
                   card_reason="no_bar_timestamp"),
    "R17": TruthRow(RG, lambda: (daily(E_MON), five_min(RG, [(1, 0.0)])), ("intraday_quote_unavailable",), E_MON, 0,
                    DAILY, card_reason="nonpositive_close"),
    "R8": TruthRow(RG, lambda: (daily(E_MON), fresh(RG, -10)), ("intraday_future_timestamp",), E_MON, 0, DAILY,
                   card_reason="future_timestamp"),
    "R13": TruthRow(RG, lambda: (None, None), UNAVAILABLE_PAIR, E_MON, None, INFO, card_reason="no_bar_timestamp",
                    history=None),
    # off-hours and early close
    "O1": TruthRow(AH, lambda: (daily(E_TUE), None), (), E_TUE, 0, DAILY),
    "O3": TruthRow(PM, lambda: (daily(E_MON), None), (), E_MON, 0, DAILY),
    "O6": TruthRow(OVERNIGHT_0300, lambda: (daily(E_MON), None), (), E_MON, 0, DAILY),
    "O7": TruthRow(LATE_2100, lambda: (daily(E_TUE), None), (), E_TUE, 0, DAILY),
    "O9": TruthRow(SATURDAY, lambda: (daily(E_FRI), None), (), E_FRI, 0, DAILY),
    "O12": TruthRow(THANKSGIVING, lambda: (daily(E_WED), None), (), E_WED, 0, DAILY),
    "E1": TruthRow(EARLY_CLOSE_1230, lambda: (daily(E_WED), fresh(EARLY_CLOSE_1230)), (), E_WED, 0, INTRA),
    "E2": TruthRow(EARLY_CLOSE_1330, lambda: (daily("2026-11-27", drop_dates=("2026-11-26",)), None), (),
                   "2026-11-27", 0, DAILY),
    "O2": TruthRow(AH, lambda: (daily(E_MON), None), STALE_PAIR, E_TUE, 1, DAILY, card_reason="stale_daily"),
    "P11b": TruthRow(PM, lambda: (daily("2026-10-09"), None), STALE_PAIR, E_MON, 1, DAILY, card_reason="stale_daily"),
    "O10": TruthRow(SATURDAY, lambda: (daily("2026-10-15"), None), STALE_PAIR, E_FRI, 1, DAILY,
                    card_reason="stale_daily"),
    "E3": TruthRow(EARLY_CLOSE_1330, lambda: (daily(E_WED), None), STALE_PAIR, "2026-11-27", 1, DAILY,
                   card_reason="stale_daily"),
    "O4": TruthRow(PM, lambda: (daily(E_TUE, None), None), ("daily_latest_bar_invalid",), E_MON, 0, DAILY,
                   card_reason="daily_unusable"),
    "O8": TruthRow(AH, lambda: (daily(E_TUE, None), None),
                   ("daily_close_stale", "daily_latest_bar_invalid", "daily_history_stale"), E_TUE, 1, DAILY,
                   card_reason="daily_unusable"),
    "O5": TruthRow(PM, lambda: (daily(E_TUE), None), ("daily_bar_after_last_session",), E_MON, 0, DAILY),
    "O13": TruthRow(THANKSGIVING, lambda: (daily("2026-11-26"), None), ("daily_bar_after_last_session",), E_WED, 0,
                    DAILY),
    "O11": TruthRow(SATURDAY, lambda: (daily(E_FRI, None, n=1), None), UNAVAILABLE_PAIR, E_FRI, None, INFO,
                    card_reason="daily_unusable", info={"price": 99.0, "quote_time": "2026-10-16T20:00:00+00:00"}),
    # missing or malformed history and quote provenance (regular session, fresh 5m); analyze only
    "M1": TruthRow(RG, lambda: (daily(E_MON), fresh(RG)), ("daily_history_unavailable",), E_MON, None, INTRA,
                   shared=False, history={"last_valid_date": "garbage", "last_row_valid": True}),
    "M2": TruthRow(RG, lambda: (daily(E_MON), fresh(RG)), ("daily_history_unavailable",), E_MON, None, INTRA,
                   shared=False, history={"last_valid_date": E_MON}),
    "M3": TruthRow(RG, lambda: (daily(E_MON), fresh(RG)), ("daily_history_unavailable",), E_MON, None, INTRA,
                   shared=False, history=None),
    "M4": TruthRow(RG, lambda: (daily(E_TUE, n=1), fresh(RG)), ("daily_history_unavailable",), E_MON, None, INTRA,
                   FORMING, shared=False),
    "M7": TruthRow(RG, lambda: (daily(E_MON), pd.DataFrame({"date": ["garbage"], "close": [101.0]})),
                   ("intraday_quote_unavailable",), E_MON, 0, DAILY, shared=False),
    "K1": TruthRow(CLOCK_ONLY_2031, lambda: (daily("2030-12-31"), fresh(CLOCK_ONLY_2031)), (), "2030-12-31", 0,
                   INTRA, ("calendar_unverified",), shared=False),
}
SHARED_ROWS = [name for name, row in TRUTH_TABLE.items() if row.shared]  # the annex's 36 frame rows


def _history(row: TruthRow, frame: pd.DataFrame | None) -> Any:
    return build_daily_history(frame, row.now) if row.history is DERIVE else row.history


def price_inputs(row: TruthRow) -> dict[str, Any]:
    """check_price_freshness's inputs as analyze builds them: the real select_price over the real
    latest_session_close / latest_intraday_bar (5m in session only), and technicals' own daily_history."""
    frame, probe = row.frames()
    intraday = latest_intraday_bar(probe) if classify_session(row.now) == "regular" else None
    _, provenance = select_price(now=row.now, intraday_quote=intraday,
                                 session_close=latest_session_close(frame, row.now), info_quote=row.info)
    return {"now": row.now, "price_provenance": provenance, "daily_history": _history(row, frame)}


def card_freshness(row: TruthRow) -> dict[str, Any]:
    """The trade card's build_freshness on the same frames."""
    frame, probe = row.frames()
    session = classify_session(row.now)
    return build_freshness(intraday_df=probe if session == "regular" else None, daily_df=frame,
                           session=session, now=row.now)


def scenario(row: TruthRow, name: str) -> dm.Scenario:
    """A decision-matrix S1 run (hold_or_add candidate) whose technicals carry this row's session_close,
    intraday_quote and daily_history from the real producers. A 1d frame of None makes technicals fail."""
    frame, probe = row.frames()
    technicals: dict[str, Any] | None = None
    if frame is not None:
        last_valid = last_valid_bar_date(frame)
        history = _history(row, frame)
        technicals = {
            **copy.deepcopy(dm.STRONG),
            "data_provenance": {"price": {"source": "yfinance", "as_of": "2026-10-13T15:00:00Z",
                                          "last_bar_date": None if last_valid is None else str(last_valid)[:10]}},
            "session_close": latest_session_close(frame, row.now),
            "intraday_quote": latest_intraday_bar(probe) if classify_session(row.now) == "regular" else None,
            "daily_history": dm.OMIT if history is None else history,
        }
    summary = {**dm.EQUITY, "current_price": row.info["price"], "quote_time": row.info.get("quote_time")}
    return dm.Scenario(f"FR {name}", technicals, dm.PROFITABLE, dm.CALM, news=dm.NEWS, summary=summary, now=row.now)
