"""Step 2 WP2.0: the analyze decision scenario matrix (offline, every tool stubbed, clock pinned).

The characterization tests in tests/test_step2/ and the committed fixtures under
tests/fixtures/decision_matrix/ are built from these scenarios. Their inputs come from the
Step 2 research probes, ported here so the evidence lives in the repo.

The baseline (before.json, README.md, snapshots_2_0_0/) records 0609fde. It is recorded once and
never overwritten, so later WPs keep the original 2.0.0 snapshots for the 3.0.0 migration tests:

    uv run --frozen --offline python -m tests.support.decision_matrix --baseline

A WP that changes decisions records the current table separately (after.json, after.md):

    uv run --frozen --offline python -m tests.support.decision_matrix --after
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from stock_analysis.tools.analyze.decision import decide
from stock_analysis.tools.analyze.gates import earnings_window
from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.utils.market_calendar import last_completed_session
from stock_analysis.utils.normalize import SNAPSHOT_VERSION
from stock_analysis.utils.provenance import build_meta, build_provenance

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

ET = pytz.timezone("America/New_York")
FRIDAY_EVENING = ET.localize(datetime(2026, 10, 9, 20, 0))  # off-session, last completed session Fri 10-09
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "decision_matrix"
SNAPSHOT_DIR = FIXTURE_DIR / "snapshots_2_0_0"
TOOLS = ("stock_summary", "technicals", "fundamentals_snapshot", "risk_metrics",
         "events_calendar", "stock_news", "ownership_analysis", "options_signals")
TIERS = ("decision", "standard", "full")
OMIT: Any = object()  # a technicals key set to OMIT is left out of the payload (e.g. daily_history)

# --- technicals --------------------------------------------------------------------------

STRONG = {
    "current_price": 100.0,
    "moving_averages": {"sma_20": 98.0, "sma_50": 95.0, "sma_200": 90.0,
                        "rules": {"above_sma20": {"triggered": True}, "above_sma50": {"triggered": True},
                                  "above_sma200": {"triggered": True}, "golden_cross": {"triggered": True},
                                  "death_cross": {"triggered": False}}},
    "rsi": {"value": 55.0, "rules": {"oversold": {"triggered": False}, "overbought": {"triggered": False}}},
    "macd": {"rules": {"bullish_cross": {"triggered": True}, "bearish_cross": {"triggered": False}}},
    "returns": {"return_1w": 0.02, "return_1w_zscore": 0.4, "return_1m": 0.12, "return_3m": 0.18,
                "return_1y": 0.4},
    "price_position": {"position_in_range": 0.7, "week_52_low": 70.0, "week_52_high": 120.0,
                       "from_52w_high": -0.17},
}
# Poor setup (below SMA200, MACD bearish, weak 3m); the falling-knife rubric scores only 1-2.
POOR_NOT_KNIFE = {
    "current_price": 88.0,
    "moving_averages": {"sma_20": 92.0, "sma_50": 94.0, "sma_200": 90.0, "sma_200_slope_pct_per_day": 0.001,
                        "rules": {"above_sma20": {"triggered": False}, "above_sma50": {"triggered": False},
                                  "above_sma200": {"triggered": False}, "golden_cross": {"triggered": False},
                                  "death_cross": {"triggered": False}}},
    "rsi": {"value": 45.0, "rules": {"oversold": {"triggered": False}, "overbought": {"triggered": False}}},
    "macd": {"rules": {"bullish_cross": {"triggered": False}, "bearish_cross": {"triggered": True}}},
    "returns": {"return_1w": -0.02, "return_1m": -0.06, "return_3m": -0.12, "return_1y": 0.05},
    "price_position": {"position_in_range": 0.4, "week_52_low": 70.0, "week_52_high": 115.0,
                       "from_52w_high": -0.23, "days_since_52w_high": 60},
    "price_action": {"higher_closes_2d": True},
}
# A falling knife (rubric score 8): death cross, deep 3m decline, near the 52w low.
KNIFE = {
    "current_price": 55.0,
    "moving_averages": {"sma_20": 58.0, "sma_50": 65.0, "sma_200": 80.0, "sma_200_slope_pct_per_day": -0.002,
                        "rules": {"above_sma20": {"triggered": False}, "above_sma50": {"triggered": False},
                                  "above_sma200": {"triggered": False}, "golden_cross": {"triggered": False},
                                  "death_cross": {"triggered": True}}},
    "rsi": {"value": 33.0, "rules": {"oversold": {"triggered": False}, "overbought": {"triggered": False}}},
    "macd": {"rules": {"bullish_cross": {"triggered": False}, "bearish_cross": {"triggered": True}}},
    "returns": {"return_1w": -0.03, "return_1m": -0.12, "return_3m": -0.35, "return_1y": -0.40},
    "price_position": {"position_in_range": 0.05, "week_52_low": 53.0, "week_52_high": 100.0,
                       "from_52w_high": -0.45, "days_since_52w_high": 200},
    "price_action": {"higher_closes_2d": True},
}
# Bearish but not a knife (score 1): below SMA200 only.
BEARISH = {
    "current_price": 90.0,
    "moving_averages": {"sma_20": 92.0, "sma_50": 95.0, "sma_200": 96.0, "sma_200_slope_pct_per_day": 0.001,
                        "rules": {"above_sma20": {"triggered": False}, "above_sma50": {"triggered": False},
                                  "above_sma200": {"triggered": False}, "golden_cross": {"triggered": False},
                                  "death_cross": {"triggered": False}}},
    "rsi": {"value": 40.0, "rules": {"oversold": {"triggered": False}, "overbought": {"triggered": False}}},
    "macd": {"rules": {"bullish_cross": {"triggered": False}, "bearish_cross": {"triggered": True}}},
    "returns": {"return_1w": -0.02, "return_1m": -0.06, "return_3m": -0.18, "return_1y": -0.05},
    "price_position": {"position_in_range": 0.3, "week_52_low": 80.0, "week_52_high": 110.0,
                       "from_52w_high": -0.18, "days_since_52w_high": 100},
}
WEAK = copy.deepcopy(STRONG)
WEAK["macd"]["rules"] = {"bullish_cross": {"triggered": False}, "bearish_cross": {"triggered": True}}
WEAK["moving_averages"]["rules"]["golden_cross"] = {"triggered": False}
WEAK["returns"]["return_3m"] = -0.12


def at_price(tech: dict[str, Any], price: float, **rules: bool) -> dict[str, Any]:
    """A copy of ``tech`` at another price, with some moving-average rules overridden."""
    out = copy.deepcopy(tech)
    out["current_price"] = price
    for name, triggered in rules.items():
        out["moving_averages"]["rules"][name] = {"triggered": triggered}
    return out


# --- fundamentals ------------------------------------------------------------------------

PROFITABLE = {"valuation": {"pe_trailing": 9.0, "trailing_eps": 11.0}, "profitability": {"net_margin": 0.2},
              "yield_metrics": {"fcf_yield": 0.09},
              "analyst_coverage": {"rating": "strong_buy", "rating_score": 1.6, "num_analysts": 20,
                                   "price_target_mean": 130.0}}
QUALITY = {"valuation": {"pe_trailing": 9.0, "trailing_eps": 6.0}, "growth": {"revenue_yoy": 0.12},
           "profitability": {"net_margin": 0.2}, "yield_metrics": {"fcf_yield": 0.09},
           "cash_flow": {"rules": {"positive_fcf": {"triggered": True}}},
           "financial_health": {"debt_to_equity": 0.2, "rules": {"net_cash_positive": {"triggered": True}}}}
HEADWIND = {**PROFITABLE, "valuation": {"pe_trailing": 60.0, "trailing_eps": 1.6}, "yield_metrics": {}}
QUALITY_NEUTRAL_VALUATION = {**QUALITY, "valuation": {"pe_trailing": 30.0, "trailing_eps": 3.0},
                             "yield_metrics": {"fcf_yield": 0.02}}
# Business quality "poor" (negative FCF, D/E 3.0): business broken, long-term fit avoid.
POOR_BUSINESS = {"valuation": {"pe_trailing": 40.0, "trailing_eps": 0.5},
                 "growth": {"revenue_yoy": -0.25, "rules": {"positive_revenue_growth": {"triggered": False}}},
                 "profitability": {"net_margin": 0.02},
                 "cash_flow": {"rules": {"positive_fcf": {"triggered": False}}},
                 "financial_health": {"debt_to_equity": 3.0, "rules": {"low_debt": {"triggered": False},
                                                                       "net_cash_positive": {"triggered": False}}}}
# Poor quality, but bullish fundamental signals (growth, profitable): bullish tilt with a strong chart.
POOR_BUT_BULLISH = {"valuation": {"pe_trailing": 40.0, "trailing_eps": 0.5},
                    "growth": {"revenue_yoy": 0.30, "rules": {"high_growth": {"triggered": True}}},
                    "profitability": {"net_margin": 0.02, "rules": {"profitable": {"triggered": True}}},
                    "cash_flow": {"rules": {"positive_fcf": {"triggered": False}}},
                    "financial_health": {"debt_to_equity": 3.0, "rules": {"low_debt": {"triggered": False}}}}
# Mixed quality from negative FCF, otherwise healthy.
MIXED_FCF = {"valuation": {"pe_trailing": 18.0, "trailing_eps": 5.0},
             "growth": {"revenue_yoy": 0.25, "rules": {"high_growth": {"triggered": True}}},
             "profitability": {"net_margin": 0.15, "rules": {"profitable": {"triggered": True}}},
             "cash_flow": {"free_cash_flow_ttm": -1e8, "rules": {"positive_fcf": {"triggered": False}}},
             "financial_health": {"debt_to_equity": 0.3, "rules": {"low_debt": {"triggered": True},
                                                                   "net_cash_positive": {"triggered": True}}}}

# --- risk, events, summaries, vendors ----------------------------------------------------

CALM = {"volatility": {"annualized": 0.22}, "drawdown": {"max_1y": -0.12}, "beta": {"value": 1.0},
        "atr": {"value": 2.0, "as_pct_of_price": 0.02}, "liquidity": {"avg_dollar_volume": 5e8}}
MODERATE = {"volatility": {"annualized": 0.35}, "drawdown": {"max_1y": -0.30}, "beta": {"value": 1.1},
            "atr": {"value": 2.0, "as_pct_of_price": 0.025}, "liquidity": {"avg_dollar_volume": 5e8}}
CRASH_RISK = {"volatility": {"annualized": 0.35}, "drawdown": {"max_1y": -0.45}, "beta": {"value": 1.1},
              "atr": {"value": 2.0, "as_pct_of_price": 0.036}, "liquidity": {"avg_dollar_volume": 5e8}}
EXTREME = {**CALM, "volatility": {"annualized": 0.75}, "drawdown": {"max_1y": -0.30},
           "atr": {"value": 5.0, "as_pct_of_price": 0.05}}
THIN = {**CALM, "liquidity": {"avg_dollar_volume": 4e5}}

FAR = {"earnings": {"days_until": 40, "next_date": "2026-11-18", "next_date_status": "available"}}
NEAR = {"earnings": {"days_until": 3, "next_date": "2026-10-12", "next_date_status": "available"}}

NEWS = {"articles": [{"title": "Why investors should buy Acme on this dip", "date": "2026-10-08"},
                     {"title": "Acme insiders keep adding shares", "date": "2026-10-07"}]}

# V1 only: S3 with every vendor subtree populated (output_contract annex §10). Each key is one the real
# producer emits (tools/news.py, options_signals.py, ownership.py, fundamentals.py); values are synthetic
# and consistent with ACME at 88. None of it moves the decision or the snapshot.
VENDOR_AS_OF = "2026-10-10T00:00:00Z"  # the matrix clock's fetch time


def vendor_envelope(tool: str, source: str) -> dict[str, Any]:
    return {"meta": build_meta(tool, 0.0),
            "data_provenance": {source: build_provenance(source="yfinance", as_of=VENDOR_AS_OF)}}


def _article(date: str, title: str, summary: str, slug: str, sentiment: str, positive: list[str] | None,
             negative: list[str] | None, tags: dict[str, list[str]]) -> dict[str, Any]:
    return {"date": date, "title": title, "summary": summary, "provider": "Example Wire",
            "url": f"https://news.example.com/acme/{slug}", "sentiment": sentiment, "matched_positive": positive,
            "matched_negative": negative, "catalyst_tags": {"bullish": [], "bearish": [], "neutral": [], **tags}}


# The two NEWS headlines first; sentiment and catalyst_intelligence are what news.py computes from all five.
NEWS_VENDOR_RICH = {
    **vendor_envelope("stock_news", "news"),
    "period_days": 7, "source_status": "ok", "source": "news", "rejected_count": 0, "article_count": 5,
    "articles": [
        _article("2026-10-08", "Why investors should buy Acme on this dip",
                 "Analysts keep a buy rating on the shares after the pullback.", "buy-the-dip", "positive",
                 ["buy", "buy rating"], None, {"bullish": ["analyst_upgrade"]}),
        _article("2026-10-07", "Acme insiders keep adding shares",
                 "Form 4 filings show a director buy of 8,000 shares in September.", "insiders-adding", "positive",
                 ["buy"], None, {"bullish": ["insider_buying"]}),
        _article("2026-10-06", "Acme price target raised to $130 at Example Securities",
                 "The broker cites steady demand and a 20% net margin.", "price-target-130", "positive",
                 ["price target raised", "raised"], None, {"bullish": ["analyst_upgrade"]}),
        _article("2026-10-05", "Acme to host investor day on October 20", "Management will review its five-year plan.",
                 "investor-day", "neutral", None, None, {"neutral": ["conference_or_presentation"]}),
        _article("2026-10-05", "Acme named in patent lawsuit by a rival", "The suit seeks unspecified damages.",
                 "patent-suit", "negative", None, ["lawsuit"], {"bearish": ["litigation_or_investigation"]}),
    ],
    "sentiment": {"overall": "positive", "confidence": "moderate", "counts": {"positive": 3, "negative": 1, "neutral": 1},
                  "method": "keyword_v2",
                  "headline_triggers": {"positive": ["buy", "buy rating", "price target raised", "raised"],
                                        "negative": ["lawsuit"]},
                  "sentiment_7d": "positive", "sample_size_7d": 5, "confidence_7d": "moderate",
                  "sentiment_30d": "positive", "sample_size_30d": 5, "confidence_30d": "moderate"},
    "catalyst_intelligence": {"bullish": [{"tag": "analyst_upgrade", "count": 2}, {"tag": "insider_buying", "count": 1}],
                              "bearish": [{"tag": "litigation_or_investigation", "count": 1}],
                              "neutral": [{"tag": "conference_or_presentation", "count": 1}],
                              "sample_size": 5, "method": "keyword_catalyst_v1"},
    "recent_earnings": None,  # FAR: no report inside the 7-day lookback
    "warnings": None,
}
# Put/call 0.92 and ATM IV 0.32 stay under analyze's signal thresholds (1.5, 0.5).
OPTIONS_ACTIVE = {
    **vendor_envelope("options_signals", "options"),
    "expiration_used": "2026-11-20",
    "implied_volatility": {"atm_avg_iv": 0.3184, "iv_rank_note": "normal"},
    "put_call_ratio": {"volume_based": 0.92, "oi_based": 0.81, "signal": "neutral"},
    "unusual_activity": {"count": 2, "top_strikes": [
        {"strike": 85.0, "type": "put", "volume": 4200, "open_interest": 1000, "ratio": 4.2},
        {"strike": 95.0, "type": "call", "volume": 1860, "open_interest": 600, "ratio": 3.1}]},
    "warnings": None,
}
# Net 3m = +8000 +5000 -1000, 6m adds -2500, 12m adds -5000; transactions top 5 by |value| as ownership.py
# orders them. insider_pct and total_institutional_pct come from major_holders, which yfinance 1.0 never
# shapes the way ownership.py parses, so live output omits them; the producer's schema has them.
INSIDER_BUYING = {
    **vendor_envelope("ownership_analysis", "ownership"),
    "insider_activity": {
        "net_shares_3m": 12000.0, "net_shares_6m": 9500.0, "net_shares_12m": 4500.0, "sentiment": "buying",
        "recent_transactions": [
            {"date": "2026-09-22", "insider": "DOE JANE", "text": "Purchase at price 87.50 per share.",
             "shares": 8000.0, "value": 700000.0},
            {"date": "2026-08-14", "insider": "ROE RICHARD", "text": "Purchase at price 91.00 per share.",
             "shares": 5000.0, "value": 455000.0},
            {"date": "2025-11-10", "insider": "ROE RICHARD", "text": "Sale at price 85.00 per share.",
             "shares": 5000.0, "value": 425000.0},
            {"date": "2026-05-20", "insider": "POE ALEX", "text": "Sale at price 98.00 per share.",
             "shares": 2500.0, "value": 245000.0},
            {"date": "2026-07-30", "insider": "POE ALEX", "text": "Sale at price 93.00 per share.",
             "shares": 1000.0, "value": 93000.0}],
        "insider_pct": 0.0125},
    "institutional": {
        "top_holders": [
            {"name": "Example Capital Management LLC", "shares": 9200000.0, "pct_held": 0.081,
             "value": 874000000.0, "date_reported": "2026-06-30"},
            {"name": "Sample Index Advisors Inc", "shares": 7100000.0, "pct_held": 0.0625,
             "value": 674500000.0, "date_reported": "2026-06-30"},
            {"name": "Placeholder Asset Partners LP", "shares": 3400000.0, "pct_held": 0.0299,
             "value": 323000000.0, "date_reported": "2026-06-30"}],
        "total_institutional_pct": 0.7},
    "warnings": None,
}
# PROFITABLE plus the blocks fundamentals_snapshot always emits. No data_provenance: it would feed data_quality.
PROFITABLE_VENDOR_RICH = {
    **PROFITABLE,
    "analyst_coverage": {"rating": "strong_buy", "rating_score": 1.6, "num_analysts": 20,
                         "price_target_low": 95.0, "price_target_mean": 130.0, "price_target_high": 160.0,
                         "price_target_median": 128.0, "upside_to_mean_target": 0.4773,
                         "upside_to_median_target": 0.4545},
    "short_interest": {"shares_short": 3_366_000, "short_pct_of_float": 0.03, "days_to_cover": 0.6,
                       "short_change_mom": 0.1, "as_of_date": "2026-09-15"},
    "ownership": {"insider_pct": 0.0125, "institutional_pct": 0.7, "float_shares": 112_200_000},
    "governance": {"audit_risk": 3, "board_risk": 4, "compensation_risk": 5, "shareholder_rights_risk": 2,
                   "overall_risk": 4},
    "analyst_estimates": {
        "eps_current_quarter": {"avg": 2.9, "low": 2.7, "high": 3.1, "num_analysts": 18.0},
        "eps_next_quarter": {"avg": 3.0, "low": 2.8, "high": 3.2, "num_analysts": 17.0},
        "eps_current_year": {"avg": 11.4, "low": 10.9, "high": 11.9, "num_analysts": 20.0},
        "eps_next_year": {"avg": 12.3, "low": 11.5, "high": 13.2, "num_analysts": 19.0},
        "rev_current_quarter": {"avg": 1.35e9, "low": 1.3e9, "high": 1.4e9, "num_analysts": 16.0},
        "rev_next_quarter": {"avg": 1.45e9, "low": 1.38e9, "high": 1.52e9, "num_analysts": 15.0},
        "rev_current_year": {"avg": 5.6e9, "low": 5.4e9, "high": 5.8e9, "num_analysts": 18.0},
        "rev_next_year": {"avg": 6.0e9, "low": 5.7e9, "high": 6.3e9, "num_analysts": 17.0},
        "forward_pe_from_estimates": 7.15},
    "quality": {"roic": 0.18, "gross_profit": 2.6e9, "ebitda": 1.5e9, "ebitda_margin": 0.27,
                "revenue_per_share": 49.3, "quick_ratio": 1.4},
    "valuation_context": {"pe_current": 9.0, "pe_5y_avg": None, "pe_percentile_5y": None, "ps_current": None,
                          "ps_5y_avg": None, "history_points": 0, "history_span_years": None, "status": "partial",
                          "status_reason": "historical_pe_ps_unavailable"},
}

EQUITY = {"name": "Acme", "currency": "USD", "quote_type": "EQUITY", "market_cap": 1e10}
ETF = {"name": "SPDR S&P 500", "currency": "USD", "quote_type": "ETF", "market_cap": None}
GBP_EQUITY = {**EQUITY, "name": "Vodafone", "currency": "GBp"}


@dataclass(frozen=True)
class Scenario:
    """Inputs for one analyze run. A tool payload of ``None`` makes that tool raise."""

    source: str  # the research case it reproduces
    technicals: dict[str, Any] | None
    fundamentals: dict[str, Any] | None
    risk: dict[str, Any] | None
    events: dict[str, Any] | None = field(default_factory=lambda: FAR)
    summary: dict[str, Any] = field(default_factory=lambda: EQUITY)
    news: dict[str, Any] = field(default_factory=dict)
    ownership: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    now: datetime = FRIDAY_EVENING
    account_size: float | None = 10_000.0
    profile: str = "balanced"

    def payloads(self) -> dict[str, dict[str, Any] | None]:
        """Tool payloads, with a current daily history unless the scenario sets its own: the
        technicals' latest completed-session close, its daily_history (derived from that close)
        and both daily tools' provenance at the session that just completed for ``now``.
        Technicals keys set to ``OMIT`` are left out."""
        session = last_completed_session(self.now.astimezone(ET)).isoformat()
        fetched = self.now.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        provenance = {"price": {"source": "yfinance", "as_of": fetched, "last_bar_date": session}}
        technicals, risk = self.technicals, self.risk
        price = (technicals or {}).get("current_price")
        if technicals is not None:
            technicals = {"data_provenance": provenance, **technicals}
            if price is not None:
                technicals.setdefault("session_close", {"close": price, "session_date": session,
                                                        "fetched_at": fetched, "cache_hit": False})
            if "daily_history" not in technicals:
                close = technicals.get("session_close")
                day = close.get("session_date") if isinstance(close, dict) else None
                technicals["daily_history"] = {"last_row_date": day, "last_valid_date": day,
                                               "last_row_valid": day is not None, "completed_valid_date": day}
            technicals = {key: value for key, value in technicals.items() if value is not OMIT}
        if risk is not None:
            risk = {"data_provenance": provenance, **risk}
        summary = {**({"current_price": price} if price is not None else {}), **self.summary}
        return {"stock_summary": summary, "technicals": technicals, "fundamentals_snapshot": self.fundamentals,
                "risk_metrics": risk, "events_calendar": self.events, "stock_news": self.news,
                "ownership_analysis": self.ownership, "options_signals": self.options}


SCENARIOS: dict[str, Scenario] = {
    # Output-contract matrix (output_contract annex §10).
    "S1_hold_or_add_clean": Scenario("OC S1", STRONG, PROFITABLE, CALM, news=NEWS),
    "S2_buy_clean": Scenario("OC S2", at_price(STRONG, 89.0), PROFITABLE, CALM),
    "S3_plain_avoid": Scenario("OC S3", POOR_NOT_KNIFE, PROFITABLE, CALM, news=NEWS),
    "S4_earnings_gate_on_hold_or_add": Scenario("OC S4", STRONG, PROFITABLE, CALM, events=NEAR),
    "S5_earnings_gate_on_avoid": Scenario("OC S5", POOR_NOT_KNIFE, PROFITABLE, CALM, events=NEAR),
    "S6_knife_dislocation": Scenario("OC S6", KNIFE, QUALITY, CRASH_RISK),
    "S7_etf_below_sma200": Scenario("OC S7", at_price(STRONG, 85.0, above_sma200=False, above_sma50=False,
                                                      above_sma20=False), {}, CALM, events={},
                                    summary=ETF),
    "S8_fundamentals_failed": Scenario("OC S8", STRONG, None, CALM),
    "S9_weak_liquidity": Scenario("OC S9", STRONG, PROFITABLE, THIN),
    "T1_extreme_risk_bullish": Scenario("OC T1", STRONG, QUALITY, EXTREME),
    "T2_starter_weak_setup_quality": Scenario("OC T2", WEAK, QUALITY, CALM),
    "T3_plain_wait_weak_setup": Scenario("OC T3", WEAK, PROFITABLE, CALM),
    "C1_headwind_hold_or_add": Scenario("OC C1", STRONG, HEADWIND, CALM),
    "C2_headwind_earnings_gate": Scenario("OC C2", STRONG, HEADWIND, CALM, events=NEAR),
    "C3_headwind_plain_avoid": Scenario("OC C3", POOR_NOT_KNIFE, HEADWIND, CALM),
    "V1_vendor_rich_plain_avoid": Scenario("OC V1", POOR_NOT_KNIFE, PROFITABLE_VENDOR_RICH, CALM,
                                           news=NEWS_VENDOR_RICH, ownership=INSIDER_BUYING,
                                           options=OPTIONS_ACTIVE),
    # Holder-guidance cases (policy_holder annex §7).
    "A_broken_bearish": Scenario("PH A", BEARISH, POOR_BUSINESS, MODERATE),
    "B_broken_bearish_knife": Scenario("PH B", KNIFE, POOR_BUSINESS, MODERATE),
    "C_quality_knife_dislocation": Scenario("PH C", KNIFE, QUALITY, MODERATE),
    "D_quality_attractive_poor_chart": Scenario("PH D", BEARISH, QUALITY, MODERATE),
    "D2_quality_poor_chart_earnings_3d": Scenario("PH D2", BEARISH, QUALITY, MODERATE, events=NEAR),
    "E_poor_business_bullish_chart": Scenario("PH E", STRONG, POOR_BUT_BULLISH, MODERATE),
    "E2_mixed_fcf_bullish_chart": Scenario("PH E2", STRONG, MIXED_FCF, MODERATE),
    "F3_quality_neutral_valuation_poor_chart": Scenario("PH F3", BEARISH, QUALITY_NEUTRAL_VALUATION, MODERATE),
    "G_unsupported_currency_knife": Scenario("PH G", KNIFE, QUALITY, MODERATE,
                                             summary={**EQUITY, "currency": "EUR"}),
    "ETF_strong": Scenario("PH ETF bullish", STRONG, {}, MODERATE, summary=ETF),
    "ETF_bearish": Scenario("PH ETF bearish", BEARISH, {}, MODERATE, summary=ETF),
    "GBp_strong": Scenario("WP4 GBp", STRONG, PROFITABLE, CALM, summary=GBP_EQUITY),
}


@contextlib.contextmanager
def stubbed(scenario: Scenario) -> Iterator[None]:
    """Replace every analyze tool call with the scenario's payload, restoring them afterwards."""
    saved = {name: getattr(orchestrator, name) for name in TOOLS}

    def tool(body: dict[str, Any] | None) -> Any:
        async def fake(symbol: str, **kwargs: Any) -> dict[str, Any]:
            if body is None:
                raise TimeoutError("stubbed tool failure")
            return {"symbol": symbol, **copy.deepcopy(body)}
        return fake

    try:
        for name, body in scenario.payloads().items():
            setattr(orchestrator, name, tool(body))
        yield
    finally:
        for name, original in saved.items():
            setattr(orchestrator, name, original)


async def run(name_or_scenario: str | Scenario, **overrides: Any) -> dict[str, Any]:
    """The full (untiered) analyze result for a scenario; ``overrides`` replace Scenario fields."""
    scenario = SCENARIOS[name_or_scenario] if isinstance(name_or_scenario, str) else name_or_scenario
    scenario = replace(scenario, **overrides) if overrides else scenario
    with stubbed(scenario):
        result: dict[str, Any] = await orchestrator.analyze_stock(
            "ACME", account_size=scenario.account_size, profile=scenario.profile, _now=scenario.now,
        )
    return result


def tiers(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {tier: project_analyze_result(copy.deepcopy(result), tier) for tier in TIERS}


def gate_ids(result: dict[str, Any]) -> list[str]:
    """The blocking gate ids on a full analyze result's decision card."""
    return [g["id"] for g in result["decision_card"]["hard_gates"]["blocking"]]


def decide_on(policy_action: Mapping[str, Any], blocking: Sequence[dict[str, Any]] = (), *,
              unevaluated: frozenset[str] = frozenset(), thesis: tuple[str, ...] = (), dislocation: bool = False,
              tilt: str | None = "neutral", support: dict[str, Any] | None = None) -> dict[str, Any]:
    """decide() on a policy's codes (long term insufficient_data unless given) and fixed blockers, with no
    earnings on the matrix clock's date."""
    today = FRIDAY_EVENING.date()
    return decide(candidate={"mid_term": policy_action["mid_term"],
                             "long_term": policy_action.get("long_term", "insufficient_data")},
                  blocking=list(blocking), unevaluated=unevaluated, thesis=thesis, dislocation=dislocation,
                  tilt=tilt, window=earnings_window({}, today), today=today, support=support)


# --- the before table --------------------------------------------------------------------

def _ids(gates: Any) -> list[str]:
    return [g.get("id") for g in gates or [] if isinstance(g, dict)]


def digest(result: dict[str, Any]) -> dict[str, Any]:
    """The decision-relevant fields of one full analyze result (no clock or timing fields)."""
    card = result.get("decision_card") or {}
    policy = result.get("policy_action") or {}
    modes = result.get("decision_modes") or {}
    dip = result.get("dip_assessment") or {}
    zones = result.get("action_zones") or {}
    dislocation = result.get("dislocation_framework") or {}
    verdict = result.get("verdict") or {}
    sizing = card.get("sizing") or {}
    snapshot = result.get("watchlist_snapshot") or {}
    shown = tiers(result)
    return {
        "action_now": card.get("action_now"),
        "hard_gates": _ids((card.get("hard_gates") or {}).get("blocking")),
        "horizon_fit": card.get("horizon_fit"),
        "policy": {"mid_term": policy.get("mid_term"), "long_term": policy.get("long_term")},
        "verdict": {"tilt": verdict.get("tilt"), "score": verdict.get("score"),
                    "fit": {k: (verdict.get("horizon_fit") or {}).get(k) for k in ("mid_term", "long_term")},
                    "business_quality": (verdict.get("decomposed") or {}).get("business_quality")},
        "modes": {m: {"action": (modes.get(m) or {}).get("action"),
                      "long_term_view": (modes.get(m) or {}).get("long_term_view")}
                  for m in ("core", "balanced", "speculative")},
        "dip": {"type": (dip.get("dip_classification") or {}).get("type"),
                "recommendation": (dip.get("assessment") or {}).get("recommendation"),
                "entry_actions": [s.get("action") for s in (dip.get("entry_timing") or {}).get("signals") or []]},
        "zone": zones.get("current_zone"),
        "valuation_gate": (zones.get("valuation_assessment") or {}).get("gate"),
        "dislocation": {"status": dislocation.get("status"),
                        "can_start_now": (dislocation.get("action") or {}).get("can_start_now"),
                        "buy_now": (dislocation.get("action") or {}).get("buy_now"),
                        "mismatch": (dislocation.get("mismatch_verdict") or {}).get("status"),
                        "thesis": (dislocation.get("thesis_integrity") or {}).get("status")},
        "card_sizing": {k: sizing.get(k) for k in ("starter_pct", "full_pct_cap", "starter_dollars",
                                                    "full_dollars", "whole_shares")},
        "sizing_present": {tier: {"card": bool((shown[tier].get("decision_card") or {}).get("sizing")),
                                  "range": bool((shown[tier].get("action_zones") or {}).get("position_sizing_range"))}
                           for tier in TIERS},
        "conditions": {k: (card.get("conditions") or {}).get(k) for k in ("buy_only_if", "add_only_if")},
        "snapshot": {k: snapshot.get(k) for k in ("action_mid_term", "action_long_term", "decision_action_now",
                                                  "hard_gate_ids", "zone", "snapshot_hash")},
    }


async def build() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """(before table, recorded snapshots) for every scenario."""
    table: dict[str, Any] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    for name, scenario in SCENARIOS.items():
        result = await run(scenario)
        table[name] = {"source": scenario.source, **digest(result)}
        snapshots[name] = result["watchlist_snapshot"]
    return table, snapshots


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _markdown(table: dict[str, Any], name: str, flag: str) -> str:
    lines = [f"# Decision matrix: {name} table", "",
             f"Generated by `python -m tests.support.decision_matrix {flag}` from the code it was recorded",
             f"on. Do not edit by hand. Full rows: `{name}.json`.", "",
             "| Scenario | Source | action_now | Hard gates | Policy mid / long | Modes core / balanced / speculative |",
             "|---|---|---|---|---|---|"]
    for name, row in table.items():
        modes = " / ".join(str(row["modes"][m]["action"]) for m in ("core", "balanced", "speculative"))
        lines.append(f"| {name} | {row['source']} | {row['action_now']} | {', '.join(row['hard_gates']) or '—'} "
                     f"| {row['policy']['mid_term']} / {row['policy']['long_term']} | {modes} |")
    return "\n".join(lines) + "\n"


def record_baseline(fixture_dir: Path = FIXTURE_DIR) -> None:
    """Write the baseline files that are missing. Refuses, writing nothing, if any existing one would change."""
    if SNAPSHOT_VERSION != "2.0.0":
        raise SystemExit(f"refusing to record 2.0.0 snapshots from SNAPSHOT_VERSION {SNAPSHOT_VERSION}")
    table, snapshots = asyncio.run(build())
    files = {fixture_dir / "before.json": _dump(table), fixture_dir / "README.md": _markdown(table, "before", "--baseline")}
    files |= {fixture_dir / SNAPSHOT_DIR.name / f"{name}.json": _dump(snapshot) for name, snapshot in snapshots.items()}
    changed = [path.relative_to(fixture_dir).as_posix() for path, text in files.items()
               if path.exists() and path.read_bytes() != text.encode()]
    if changed:
        raise SystemExit(f"refusing to overwrite the recorded baseline: {', '.join(changed)}")
    for path, text in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())


def write_after(fixture_dir: Path = FIXTURE_DIR) -> None:
    """Refresh the current decision table. Never touches the baseline."""
    table, _ = asyncio.run(build())
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "after.json").write_text(_dump(table))
    (fixture_dir / "after.md").write_text(_markdown(table, "after", "--after"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the decision-matrix fixtures.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true", help="record the missing 0609fde baseline files")
    mode.add_argument("--after", action="store_true", help="refresh after.json and after.md")
    args = parser.parse_args()
    if args.baseline:
        record_baseline()
    else:
        write_after()


if __name__ == "__main__":
    main()
