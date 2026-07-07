"""Orchestrator for the analyze_trade_setup tool: fetch, gate, assemble."""

from __future__ import annotations

import asyncio
from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd
import pytz

from stock_analysis.data.cache_manager import classify_session
from stock_analysis.data.yfinance_client import fetch_history
from stock_analysis.tools.events import events_calendar
from stock_analysis.tools.risk_metrics import risk_metrics
from stock_analysis.tools.stock_summary import stock_summary
from stock_analysis.tools.technicals import technicals
from stock_analysis.tools.trade_setup.card import build_trade_setup_card
from stock_analysis.tools.trade_setup.features import compute_setup_features
from stock_analysis.tools.trade_setup.freshness import build_freshness
from stock_analysis.tools.trade_setup.setup_rules import PROBE_INTERVAL, PROBE_PERIOD
from stock_analysis.utils.provenance import build_error_response, build_meta
from stock_analysis.utils.validators import FetchParams

_ET = pytz.timezone("America/New_York")
_CORE_TOOL_NAMES = ["stock_summary", "technicals", "risk_metrics", "events_calendar"]


def validate_sizing_params(
    account_size: float | None,
    risk_per_trade_pct: float,
    max_position_pct: float,
) -> str | None:
    """Bounds for sizing inputs; None when valid, else the rejection message."""
    if account_size is not None and account_size <= 0:
        return f"account_size must be positive, got {account_size}"
    if not 0 < risk_per_trade_pct <= 100:
        return f"risk_per_trade_pct must be in (0, 100], got {risk_per_trade_pct}"
    if not 0 < max_position_pct <= 100:
        return f"max_position_pct must be in (0, 100], got {max_position_pct}"
    return None


async def analyze_trade_setup(
    symbol: str,
    account_size: float | None = None,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 10.0,
    _now: datetime | None = None,
) -> dict[str, Any]:
    start_time = perf_counter()
    normalized = symbol.upper().strip()

    param_error = validate_sizing_params(
        account_size, risk_per_trade_pct, max_position_pct
    )
    if param_error is not None:
        return build_error_response(
            error_type="invalid_parameters",
            message=param_error,
            symbol=normalized,
        )

    now = _now or datetime.now(_ET)
    session = classify_session(now)

    results = await asyncio.gather(
        stock_summary(normalized),
        technicals(normalized, _now=now),
        risk_metrics(normalized),
        events_calendar(normalized),
        _quiet_history(FetchParams(normalized, "1y", "1d", True)),
        _quiet_history(FetchParams(normalized, PROBE_PERIOD, PROBE_INTERVAL, True)),
        return_exceptions=True,
    )

    tool_failures: list[dict[str, Any]] = []
    cleaned: dict[str, dict[str, Any]] = {}
    for name, result in zip(_CORE_TOOL_NAMES, results[:4], strict=True):
        if isinstance(result, BaseException):
            tool_failures.append({"tool": name, "error": str(result)})
        elif isinstance(result, dict) and result.get("error"):
            tool_failures.append({
                "tool": name,
                "error": result.get("message") or result.get("error_type"),
                "error_type": result.get("error_type"),
            })
        else:
            cleaned[name] = result

    daily_df = results[4] if isinstance(results[4], pd.DataFrame) else None
    probe_df = results[5] if isinstance(results[5], pd.DataFrame) else None

    if "technicals" not in cleaned and "stock_summary" not in cleaned:
        # invalid_symbol only when every core failure says so; a transport
        # failure anywhere means an invalid-symbol verdict cannot be trusted.
        core_failures = [
            tf for tf in tool_failures if tf["tool"] in ("stock_summary", "technicals")
        ]
        unanimous_invalid = bool(core_failures) and all(
            tf.get("error_type") == "invalid_symbol" for tf in core_failures
        )
        response = build_error_response(
            error_type="invalid_symbol" if unanimous_invalid else "data_unavailable",
            message=(
                f"No usable data for '{normalized}'"
                if unanimous_invalid
                else f"Market data unavailable for '{normalized}' — upstream fetches failed"
            ),
            symbol=normalized,
        )
        response["data_quality"] = {"tool_failures": core_failures}
        return response

    freshness = build_freshness(
        intraday_df=probe_df, daily_df=daily_df, session=session, now=now,
    )
    features = compute_setup_features(daily_df)

    probe_close: float | None = None
    if session == "regular" and probe_df is not None and len(probe_df) > 0:
        last_close = pd.to_numeric(probe_df["close"], errors="coerce").iloc[-1]
        if not pd.isna(last_close):
            probe_close = float(last_close)

    actionable_price: float | None
    if probe_close is not None:
        actionable_price = probe_close
    elif features is not None:
        actionable_price = features["last_close"]
    else:
        actionable_price = (cleaned.get("technicals") or {}).get("current_price")

    card = build_trade_setup_card(
        symbol=normalized,
        summary_data=cleaned.get("stock_summary") or {},
        technicals_data=cleaned.get("technicals") or {},
        risk_data=cleaned.get("risk_metrics") or {},
        events_data=cleaned.get("events_calendar") or {},
        freshness=freshness,
        features=features,
        actionable_price=actionable_price,
        session=session,
        now=now,
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
        tool_failures=tool_failures,
    )
    duration_ms = (perf_counter() - start_time) * 1000
    card["meta"] = build_meta("analyze_trade_setup", duration_ms)
    return card


async def _quiet_history(params: FetchParams) -> pd.DataFrame | None:
    """Fetch history, returning None on any failure (freshness gate handles it)."""
    try:
        return await fetch_history(params)
    except Exception:
        return None
