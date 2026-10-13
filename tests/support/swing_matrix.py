"""Step 2 WP2.0: swing scenarios for the trade card and the watchlist scan (offline, clock pinned).

CARD_CASES are pure ``build_trade_setup_card`` inputs; SCAN_SCENARIOS are daily frames that
scan stage 1 (``screen_symbol``), the real ``analyze_trade_setup`` and ``scan_watchlist`` all
read. Ported from the gate_contract research probes (probe_card_matrix.py,
probe_verify_analyze.py, probe_scan_vs_card.py and probe_verify_scan.py) so the evidence lives
in the repo.
"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import pytz

import stock_analysis.tools.trade_setup.card as card_module
import stock_analysis.tools.watchlist as watchlist
from stock_analysis.utils.validators import FetchParams
from stock_analysis.utils.watch_store import save_scan_state
from tests.test_tools.test_setup_detection import make_features, make_technicals
from tests.test_tools.test_trade_card import EOD_FRESH, NOW_EVENING, STALE, UNVERIFIABLE, build_card

trade_setup = importlib.import_module("stock_analysis.tools.trade_setup.orchestrator")
technicals_tool = importlib.import_module("stock_analysis.tools.technicals")
risk_tool = importlib.import_module("stock_analysis.tools.risk_metrics")

# --- pure trade cards (build_card pins NOW_REGULAR, Tue 2026-03-10 10:30 ET) ------------------


def knife_technicals() -> dict[str, Any]:
    """A falling knife: death cross, below SMA200, falling SMA200, 3m return -35%."""
    technicals = make_technicals()
    technicals["moving_averages"]["rules"].update({
        "death_cross": {"triggered": True}, "golden_cross": {"triggered": False},
        "above_sma200": {"triggered": False},
    })
    technicals["moving_averages"]["sma_200_slope_pct_per_day"] = -0.002
    technicals["returns"]["return_3m"] = -0.35
    return technicals


ETF_SUMMARY = {"currency": "USD", "quote_type": "ETF", "name": "SPDR S&P 500"}
GBP_SUMMARY = {"currency": "GBp", "quote_type": "EQUITY", "name": "Vodafone"}
NO_SETUP = {"technicals_data": make_technicals(rsi={"value": 60.0, "bullish_divergence": False}),
            "features": make_features(high_20d_prior=120.0)}
KNIFE = {"technicals_data": knife_technicals()}
EARNINGS_IN_3D = {"events_data": {"earnings": {"days_until": 3, "next_date": "2026-03-13"}}}
UNVERIFIABLE_EVENTS = {"earnings": {"next_date": None, "days_until": None, "verification": "unverifiable",
                                    "sources_failed": ["calendar"]}}
EARNINGS_UNVERIFIABLE = {"events_data": UNVERIFIABLE_EVENTS}
EVENTS_TOOL_FAILED = {"events_data": {}, "tool_failures": [{"tool": "events_calendar", "error": "x"}]}
RISK_METRICS_FAILED = {"risk_data": {}, "tool_failures": [{"tool": "risk_metrics", "error": "x"}]}
WEAK_LIQUIDITY = {"risk_data": {"liquidity": {"avg_dollar_volume": 4e5}}}
OFF_SESSION = {"freshness": {**EOD_FRESH, "session": "after_hours"}, "session": "after_hours", "now": NOW_EVENING}
TRIGGER_CROSSED = {"actionable_price": 101.5}  # above the pullback's 101.0 trigger

CARD_CASES: dict[str, dict[str, Any]] = {
    "clean_armed": {},
    "clean_satisfied_regular": TRIGGER_CROSSED,
    "stale_only": {"freshness": STALE},
    "unverifiable_freshness": {"freshness": UNVERIFIABLE},
    "stale+knife": {"freshness": STALE, **KNIFE},
    "stale+weak_liq": {"freshness": STALE, **WEAK_LIQUIDITY},
    "stale+etf": {"freshness": STALE, "summary_data": ETF_SUMMARY},
    "knife": KNIFE,
    "knife+etf": {**KNIFE, "summary_data": ETF_SUMMARY},
    "knife+gbp": {**KNIFE, "summary_data": GBP_SUMMARY},
    "knife+earnings3": {**KNIFE, **EARNINGS_IN_3D},
    "knife+earn_unverifiable": {**KNIFE, **EARNINGS_UNVERIFIABLE},
    "weak_liq": WEAK_LIQUIDITY,
    "liq_missing": {"risk_data": {"liquidity": {}}},
    "etf_with_setup": {"summary_data": ETF_SUMMARY, **TRIGGER_CROSSED},
    "etf_no_setup": {"summary_data": ETF_SUMMARY, **NO_SETUP},
    "gbp_no_setup": {"summary_data": GBP_SUMMARY, **NO_SETUP},
    "no_setup+earnings3": {**NO_SETUP, **EARNINGS_IN_3D},
    "no_setup+earn_unverifiable": {**NO_SETUP, **EARNINGS_UNVERIFIABLE},
    "no_setup+events_tool_failed": {**NO_SETUP, **EVENTS_TOOL_FAILED},
    "setup+earnings3": EARNINGS_IN_3D,
    "setup+earn_unverifiable": EARNINGS_UNVERIFIABLE,
    "setup+etf+earnings3": {"summary_data": ETF_SUMMARY, **EARNINGS_IN_3D},
    "off_session_armed": OFF_SESSION,
    "off_session_trigger_crossed": {**OFF_SESSION, "actionable_price": 101.8},
    "off_session+earnings3": {**OFF_SESSION, **EARNINGS_IN_3D},
    "off_session+etf": {**OFF_SESSION, "summary_data": ETF_SUMMARY},
    "summary_tool_failed": {"summary_data": {}, "tool_failures": [{"tool": "stock_summary", "error": "x"}]},
    "events_tool_failed+stale": {"freshness": STALE, **EVENTS_TOOL_FAILED},
    # probe_verify_analyze.py G and F: a risk_metrics failure, then with no setup features too.
    "risk_metrics_failed": RISK_METRICS_FAILED,
    "risk_metrics_failed+no_features": {**RISK_METRICS_FAILED, "features": None},
}

# A detected setup whose stop equals its trigger: build_plan raises PlanInvariantError.
INVALID_GEOMETRY_SETUP = {
    "type": "pullback_in_uptrend", "quality": "A", "thesis": [], "invalidation": [],
    "trigger_price": 101.0, "trigger_satisfied": False, "trigger_condition": "x",
    "stop_price": 101.0, "stop_basis": "atr", "target_primary": None,
}
INVALID_PLAN_CASES: dict[str, dict[str, Any]] = {
    "plan_invalid": {},
    "plan_invalid+off_session": OFF_SESSION,
    "plan_invalid+etf": {"summary_data": ETF_SUMMARY},
}
ALL_CARD_CASES = (*CARD_CASES, *INVALID_PLAN_CASES)


def card_case(name: str) -> dict[str, Any]:
    """The pure trade card for one case; invalid-plan cases detect INVALID_GEOMETRY_SETUP."""
    if name not in INVALID_PLAN_CASES:
        return build_card(**copy.deepcopy(CARD_CASES[name]))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(card_module, "detect_setup", lambda *args, **kwargs: dict(INVALID_GEOMETRY_SETUP))
        return build_card(**copy.deepcopy(INVALID_PLAN_CASES[name]))


# --- scan stage 1 vs the real card on one daily frame ------------------------------------------

ET = pytz.timezone("America/New_York")
PRE_MARKET = ET.localize(datetime(2026, 3, 10, 8, 0))  # Tuesday pre-market; expected daily bar 2026-03-09
REGULAR = ET.localize(datetime(2026, 3, 10, 11, 0))  # the same Tuesday in session: the card reads a 5m quote
SYMBOL = "EXMP"

BREAKOUT = tuple(80.0 + i * 0.12 for i in range(260))
BORING = tuple(100.0 + (0.5 if i % 2 else -0.5) for i in range(260))
DECLINE = tuple(200.0 - i * 0.55 for i in range(260))  # below SMA200, death cross, 3m < -30%, near the 52w low
SPIKE_VOLUME = (1e6,) * 259 + (2.5e6,)
FLAT_VOLUME = (1e6,) * 260


@dataclass(frozen=True)
class ScanScenario:
    """One daily frame, plus the stock_summary and events_calendar the card reads alongside it."""

    source: str  # the probe scenario it reproduces
    closes: tuple[float, ...]
    volumes: tuple[float, ...]
    last_date: str = "2026-03-09"
    quote_type: str = "EQUITY"  # stock_summary's, and the history attrs' unless history_attrs is False
    history_attrs: bool = True  # False: the chart metadata carries no instrument_type
    earnings_days: int = 40
    earnings_unverifiable: bool = False

    def frame(self) -> pd.DataFrame:
        closes = list(self.closes)
        dates = pd.bdate_range(end=self.last_date, periods=len(closes)).strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame({"date": dates, "open": closes, "high": [c + 1 for c in closes],
                           "low": [c - 1 for c in closes], "close": closes, "volume": list(self.volumes)})
        df.attrs.update({"symbol": SYMBOL, "currency": "USD",
                         "instrument_type": self.quote_type if self.history_attrs else None})
        return df

    def events(self) -> dict[str, Any]:
        if self.earnings_unverifiable:
            return copy.deepcopy(UNVERIFIABLE_EVENTS)
        next_date = (PRE_MARKET.date() + timedelta(days=self.earnings_days)).isoformat()
        return {"earnings": {"days_until": self.earnings_days, "next_date": next_date,
                             "verification": "verified_date"}}


SCAN_SCENARIOS: dict[str, ScanScenario] = {
    # probe_scan_vs_card.py
    "fresh_breakout": ScanScenario("scan_vs_card", BREAKOUT, SPIKE_VOLUME),
    "stale_frame_boring": ScanScenario("scan_vs_card", BORING, FLAT_VOLUME, last_date="2026-02-20"),
    "stale_frame_breakout": ScanScenario("scan_vs_card", BREAKOUT, SPIKE_VOLUME, last_date="2026-02-20"),
    # 20-bar dollar volume collapsed (~$0.11M/day); the 1y mean volume x last close is ~$3.07M/day.
    "liq_recent_collapse_breakout": ScanScenario("scan_vs_card", BREAKOUT, (3e4,) * 239 + (1e3,) * 20 + (2.5e3,)),
    # The reverse: thin over the year, liquid over the last 20 bars.
    "liq_recent_surge_breakout": ScanScenario("scan_vs_card", BREAKOUT, (2e3,) * 239 + (5e4,) * 20 + (1.25e5,)),
    "etf_boring": ScanScenario("scan_vs_card", BORING, FLAT_VOLUME, quote_type="ETF"),
    "etf_breakout": ScanScenario("scan_vs_card", BREAKOUT, SPIKE_VOLUME, quote_type="ETF"),
    "boring_earnings_in_2d": ScanScenario("scan_vs_card", BORING, FLAT_VOLUME, earnings_days=2),
    "breakout_earnings_in_2d": ScanScenario("scan_vs_card", BREAKOUT, SPIKE_VOLUME, earnings_days=2),
    "breakout_events_unverifiable": ScanScenario("scan_vs_card", BREAKOUT, SPIKE_VOLUME,
                                                 earnings_unverifiable=True),
    # probe_verify_scan.py: A 21-49 bars, B no instrument_type in the history attrs, C stale vetoes, D ETF.
    "A_30_bars_breakout": ScanScenario("verify_scan A", BREAKOUT[-30:], (1e6,) * 29 + (2.5e6,)),
    "A_30_bars_boring": ScanScenario("verify_scan A", BORING[-30:], (1e6,) * 30),
    "A_45_bars_boring": ScanScenario("verify_scan A", BORING[-45:], (1e6,) * 45),
    "B_attrs_missing_breakout": ScanScenario("verify_scan B", BREAKOUT, SPIKE_VOLUME, history_attrs=False),
    "B_attrs_missing_boring": ScanScenario("verify_scan B", BORING, FLAT_VOLUME, history_attrs=False),
    "C_stale_knife": ScanScenario("verify_scan C", DECLINE, FLAT_VOLUME, last_date="2026-02-20"),
    "C_stale_thin": ScanScenario("verify_scan C", BORING, (3e3,) * 260, last_date="2026-02-20"),
    "D_etf_breakout": ScanScenario("verify_scan D", BREAKOUT, SPIKE_VOLUME, quote_type="ETF"),
}


def install(
    monkeypatch: pytest.MonkeyPatch, scenario: ScanScenario, probe_close: float | None = None
) -> pd.DataFrame:
    """Serve the scenario's daily frame to the scan, technicals, risk_metrics and the card.

    Intraday fetches fail (pre-market, so freshness reads the daily leg only), unless ``probe_close``
    gives the card one 5m bar at that close, stamped five minutes before REGULAR. There is no
    options chain offline, so the card's earnings expected move is None.
    """
    df = scenario.frame()
    probe = None if probe_close is None else pd.DataFrame({
        "date": [(REGULAR - timedelta(minutes=5)).astimezone(pytz.UTC).isoformat()], "open": [probe_close],
        "high": [probe_close], "low": [probe_close], "close": [probe_close], "volume": [1.0]})

    async def history(params: FetchParams) -> pd.DataFrame:
        if params.interval != "1d":
            raise RuntimeError("no intraday bars offline")
        return df.copy()

    async def quiet_history(params: FetchParams) -> pd.DataFrame | None:
        if params.interval == "1d":
            return df.copy()
        return None if probe is None else probe.copy()

    async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {"symbol": symbol, "name": "Example", "currency": "USD", "quote_type": scenario.quote_type,
                "current_price": float(df["close"].iloc[-1])}

    async def events(symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {"symbol": symbol, **scenario.events()}

    async def no_expected_move(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(watchlist, "fetch_history", history)
    monkeypatch.setattr(technicals_tool, "fetch_history", history)
    monkeypatch.setattr(risk_tool, "fetch_history", history)
    monkeypatch.setattr(trade_setup, "_quiet_history", quiet_history)
    monkeypatch.setattr(trade_setup, "stock_summary", summary)
    monkeypatch.setattr(trade_setup, "events_calendar", events)
    monkeypatch.setattr(trade_setup, "_fetch_expected_move", no_expected_move)
    return df


async def trade_card(now: datetime = PRE_MARKET) -> dict[str, Any]:
    """The real analyze_trade_setup card on the installed scenario."""
    card: dict[str, Any] = await trade_setup.analyze_trade_setup(SYMBOL, _now=now)
    return card


async def scan(data_dir: Path, prior_action: str | None = None, now: datetime = PRE_MARKET) -> dict[str, Any]:
    """One scan_watchlist run; a ``prior_action`` seeds a pre-WP2.1 scan state entry (no source, no
    rules_version; an actionable one forces the card)."""
    await watchlist.manage_watchlist("add", [SYMBOL], _today=now.date(), data_dir=data_dir)
    if prior_action is not None:
        save_scan_state(data_dir, {"scanned_at": "prior", "symbols": {SYMBOL: {
            "action": prior_action, "setup_type": "breakout", "trigger_satisfied": False,
            "blockers": [], "trigger_price": 100.0}}})
    result: dict[str, Any] = await watchlist.scan_watchlist(_now=now, data_dir=data_dir)
    return result
