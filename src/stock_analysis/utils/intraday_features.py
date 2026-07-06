"""Intraday (swing-mode) feature builders: VWAP, time-adjusted RVOL,
hourly trend, daily/hourly alignment.

Disclosure semantics: failures null the dependent fields and append
{id, reason} warnings — this module never raises for missing data and
never blocks a response (spec: freshness is disclosure here, not gating).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from stock_analysis.utils.freshness import build_freshness
from stock_analysis.utils.helpers import safe_round
from stock_analysis.utils.indicators import calculate_ema

SESSION_MINUTES = 390.0  # 9:30-16:00 ET
MIN_ELAPSED_FRACTION = 0.05
HOURLY_EMA_PERIOD = 20
HOURLY_SLOPE_BARS = 5


def build_intraday_block(
    *,
    df_5m: pd.DataFrame | None,
    df_1h: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    technicals_payload: dict[str, Any],
    session: str,
    now: datetime,
) -> dict[str, Any]:
    """Assemble the swing-mode intraday block. `now` must be tz-aware ET."""
    warnings: list[dict[str, str]] = []
    freshness = build_freshness(
        intraday_df=df_5m, daily_df=daily_df, session=session, now=now,
    )

    intraday_usable = df_5m is not None and len(df_5m) > 0
    if not intraday_usable:
        warnings.append({
            "id": "intraday_unavailable",
            "reason": "5-minute bars unavailable — VWAP and time-adjusted RVOL omitted",
        })
    elif freshness["stale"]:
        intraday_usable = False
        warnings.append({
            "id": "stale_intraday",
            "reason": "intraday data is stale — VWAP and time-adjusted RVOL omitted",
        })

    vwap = _session_vwap(df_5m) if intraday_usable else None

    rvol_ta: dict[str, Any] | None = None
    if session != "regular":
        warnings.append({
            "id": "off_session",
            "reason": "time-adjusted RVOL is only computed during regular hours",
        })
    elif intraday_usable:
        rvol_ta = _time_adjusted_rvol(df_5m, daily_df, now)

    hourly = _hourly_trend(df_1h)
    if hourly is None:
        warnings.append({
            "id": "hourly_unavailable",
            "reason": "hourly bars unavailable — trend and alignment omitted",
        })

    daily_state = _daily_state(technicals_payload)
    alignment = {
        "daily": daily_state,
        "aligned_pullback": (
            daily_state == "up"
            and hourly is not None
            and hourly["state"] == "pullback"
        ),
    }

    return {
        "freshness": freshness,
        "session_date": _session_date(df_5m),
        "vwap": vwap,
        "rvol_time_adjusted": rvol_ta,
        "hourly_trend": hourly,
        "alignment": alignment,
        "warnings": warnings,
    }


def _session_vwap(df_5m: pd.DataFrame | None) -> dict[str, Any] | None:
    if df_5m is None or len(df_5m) == 0:
        return None
    high = pd.to_numeric(df_5m["high"], errors="coerce")
    low = pd.to_numeric(df_5m["low"], errors="coerce")
    close = pd.to_numeric(df_5m["close"], errors="coerce")
    volume = pd.to_numeric(df_5m["volume"], errors="coerce")
    typical = (high + low + close) / 3.0
    total_volume = float(volume.sum())
    if pd.isna(total_volume) or total_volume <= 0:
        return None
    vwap = float((typical * volume).sum() / total_volume)
    last = close.iloc[-1]
    if pd.isna(last) or vwap <= 0:
        return None
    last_price = float(last)
    return {
        "value": safe_round(vwap, 2),
        "price_vs_vwap_pct": safe_round((last_price - vwap) / vwap, 4),
        "above": last_price > vwap,
    }


def _time_adjusted_rvol(
    df_5m: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    now: datetime,
) -> dict[str, Any] | None:
    if df_5m is None or len(df_5m) == 0 or daily_df is None or len(daily_df) < 2:
        return None
    # 20d average full-day volume from daily bars EXCLUDING the current bar.
    daily_volume = pd.to_numeric(daily_df["volume"], errors="coerce")
    prior = daily_volume.iloc[-21:-1] if len(daily_volume) >= 21 else daily_volume.iloc[:-1]
    if prior.isna().all():
        return None
    avg_full_day = float(prior.mean())
    if avg_full_day <= 0:
        return None
    cumulative = float(pd.to_numeric(df_5m["volume"], errors="coerce").sum())
    minutes = (now.hour - 9) * 60 + (now.minute - 30)
    elapsed = max(MIN_ELAPSED_FRACTION, min(1.0, minutes / SESSION_MINUTES))
    return {
        "value": safe_round(cumulative / (avg_full_day * elapsed), 2),
        "elapsed_session_pct": safe_round(elapsed * 100, 1),
    }


def _hourly_trend(df_1h: pd.DataFrame | None) -> dict[str, Any] | None:
    if df_1h is None or len(df_1h) < HOURLY_EMA_PERIOD + HOURLY_SLOPE_BARS:
        return None
    close = pd.to_numeric(df_1h["close"], errors="coerce")
    ema = calculate_ema(close, HOURLY_EMA_PERIOD).dropna()
    if len(ema) < HOURLY_SLOPE_BARS + 1:
        return None
    last_close = close.dropna().iloc[-1]
    ema_now = float(ema.iloc[-1])
    ema_then = float(ema.iloc[-(HOURLY_SLOPE_BARS + 1)])
    if pd.isna(last_close) or ema_now <= 0:
        return None
    above = float(last_close) > ema_now
    rising = ema_now > ema_then
    if above and rising:
        state = "advance"
    elif not above and rising:
        state = "pullback"
    elif above and not rising:
        state = "range"
    else:
        state = "breakdown"
    return {
        "state": state,
        "ema20_1h": safe_round(ema_now, 2),
        "price_vs_ema20_1h_pct": safe_round((float(last_close) - ema_now) / ema_now, 4),
    }


def _daily_state(technicals_payload: dict[str, Any]) -> str:
    ma = technicals_payload.get("moving_averages") or {}
    price_vs_sma50 = ma.get("price_vs_sma50")
    sma_20, sma_50 = ma.get("sma_20"), ma.get("sma_50")
    if price_vs_sma50 is None or sma_20 is None or sma_50 is None:
        return "sideways"
    if price_vs_sma50 > 0 and sma_20 > sma_50:
        return "up"
    if price_vs_sma50 < 0 and sma_20 < sma_50:
        return "down"
    return "sideways"


def _session_date(df_5m: pd.DataFrame | None) -> str | None:
    if df_5m is None or len(df_5m) == 0:
        return None
    return str(df_5m["date"].iloc[-1])[:10]
