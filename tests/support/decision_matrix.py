"""Step 2 WP2.0: the analyze decision scenario matrix (offline, every tool stubbed, clock pinned).

The characterization tests in tests/test_step2/ and the committed before table under
tests/fixtures/decision_matrix/ are built from these scenarios. Their inputs come from the
Step 2 research probes, ported here so the evidence lives in the repo.

Regenerate the committed fixtures only on purpose (they record the code they were built from):

    uv run --frozen --offline python -m tests.support.decision_matrix --write
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import importlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz

from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.utils.market_calendar import last_completed_session
from stock_analysis.utils.normalize import SNAPSHOT_VERSION

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

ET = pytz.timezone("America/New_York")
FRIDAY_EVENING = ET.localize(datetime(2026, 10, 9, 20, 0))  # off-session, last completed session Fri 10-09
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "decision_matrix"
SNAPSHOT_DIR = FIXTURE_DIR / "snapshots_2_0_0"
TOOLS = ("stock_summary", "technicals", "fundamentals_snapshot", "risk_metrics",
         "events_calendar", "stock_news", "ownership_analysis", "options_signals")
TIERS = ("decision", "standard", "full")

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
INSIDER_BUYING = {"insider_activity": {"sentiment": "buying", "net_shares_3m": 1000},
                  "institutional": {"top_holders": [], "total_institutional_pct": 0.7}}

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
        technicals' latest completed-session close and both daily tools' provenance at the
        session that just completed for ``now``."""
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
    "V1_vendor_rich_plain_avoid": Scenario("OC V1", POOR_NOT_KNIFE, PROFITABLE, CALM, news=NEWS,
                                           ownership=INSIDER_BUYING),
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


def _markdown(table: dict[str, Any]) -> str:
    lines = ["# Decision matrix: before table", "",
             "Generated by `python -m tests.support.decision_matrix --write` from the code it was recorded",
             "on. Do not edit by hand. Full rows: `before.json`.", "",
             "| Scenario | Source | action_now | Hard gates | Policy mid / long | Modes core / balanced / speculative |",
             "|---|---|---|---|---|---|"]
    for name, row in table.items():
        modes = " / ".join(str(row["modes"][m]["action"]) for m in ("core", "balanced", "speculative"))
        lines.append(f"| {name} | {row['source']} | {row['action_now']} | {', '.join(row['hard_gates']) or '—'} "
                     f"| {row['policy']['mid_term']} / {row['policy']['long_term']} | {modes} |")
    return "\n".join(lines) + "\n"


def write_fixtures() -> None:
    if SNAPSHOT_VERSION != "2.0.0":
        raise SystemExit(f"refusing to record 2.0.0 snapshots from SNAPSHOT_VERSION {SNAPSHOT_VERSION}")
    table, snapshots = asyncio.run(build())
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "before.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    (FIXTURE_DIR / "README.md").write_text(_markdown(table))
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    for name, snapshot in snapshots.items():
        (SNAPSHOT_DIR / f"{name}.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the WP2.0 decision-matrix fixtures.")
    parser.add_argument("--write", action="store_true", help="overwrite tests/fixtures/decision_matrix/")
    args = parser.parse_args()
    if not args.write:
        parser.error("pass --write to regenerate the committed fixtures")
    write_fixtures()


if __name__ == "__main__":
    main()
