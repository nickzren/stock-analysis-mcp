"""Offline holder/position observations, not investment rules or future conformance.

The c9fba4a baseline is frozen. Use --after --output-dir <new directory> to
compare a later checkout. All provider tools are stubbed; HTTP is forbidden.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import curl_cffi.requests
import requests

from stock_analysis.tools import position
from tests.support import decision_matrix as dm
from tests.support.network_guard import blocked_request, violation_report

BASE_COMMIT = "c9fba4ada95941a6e298f7380f70b3ec76d6d233"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "holder_matrix"
CLOCK = datetime(2026, 10, 9, 20)  # Retained only for the frozen baseline metadata.
POSITION_CLOCK = dm.FRIDAY_EVENING


def fundamentals(base=dm.QUALITY, **blocks):
    return {**copy.deepcopy(base), **copy.deepcopy(blocks)}


SOUND = dm.Scenario("assessable positive FCF / low debt", dm.STRONG, dm.QUALITY, dm.MODERATE)
STOCK_CASES = {
    "sound_bullish": SOUND,
    "hold_with_avoid_fit": replace(SOUND, risk=dm.EXTREME, fundamentals=fundamentals(
        dm.MIXED_FCF, growth={"revenue_yoy": -0.4,
                            "rules": {"positive_revenue_growth": {"triggered": False}}})),
    "price_disagreement": replace(SOUND, technicals={**SOUND.payloads()["technicals"], "current_price": 105.0}),
    "sound_bearish": replace(SOUND, technicals=dm.BEARISH),
    "sound_knife": replace(SOUND, technicals=dm.KNIFE),
    "sound_core": replace(SOUND, profile="core"),
    "sound_speculative": replace(SOUND, profile="speculative"),
    "broken_bullish": replace(SOUND, fundamentals=dm.POOR_BUSINESS),
    "broken_bearish": dm.SCENARIOS["A_broken_bearish"],
    "critical_runway": dm.SCENARIOS["H_burner_runway_2q"],
    "unknown_runway": replace(SOUND, fundamentals=fundamentals(
        dm.BURNER, financial_health={"debt_to_equity": 0.1})),
    "watch_runway": replace(SOUND, fundamentals=fundamentals(
        dm.BURNER, financial_health={**dm.BURNER["financial_health"], "cash_and_st_investments": 3e8})),
    "confirmed_negative_fcf": dm.SCENARIOS["E2_mixed_fcf_bullish_chart"],
    "negative_fcf_no_debt": replace(SOUND, fundamentals=fundamentals(
        dm.MIXED_FCF, financial_health={"rules": dm.MIXED_FCF["financial_health"]["rules"]})),
    "nonburner_no_cash": replace(SOUND, fundamentals=fundamentals(
        dm.BURNER, cash_flow={"free_cash_flow_ttm": 1e8, "operating_cf_ttm": 1e8,
                             "rules": {"positive_fcf": {"triggered": True}}},
        financial_health={"debt_to_equity": 0.1})),
    "missing_fcf_and_debt": replace(SOUND, fundamentals=dm.PROFITABLE),
    "fundamentals_failed": replace(SOUND, fundamentals=None),
    "net_debt": replace(SOUND, fundamentals=fundamentals(financial_health={
        "debt_to_equity": 0.2, "rules": {"net_cash_positive": {"triggered": False}}})),
    "negative_equity": replace(SOUND, fundamentals=fundamentals(financial_health={
        "debt_to_equity": None, "leverage_status": "negative_equity",
        "rules": {"net_cash_positive": {"triggered": False}}})),
    "neutral_valuation_bearish": dm.SCENARIOS["F3_quality_neutral_valuation_poor_chart"],
    "earnings": replace(SOUND, events=dm.NEAR),
    "near_high": dm.SCENARIOS["T1_near_52w_high"],
    "stale_daily": replace(SOUND, technicals={**copy.deepcopy(dm.STRONG),
        "session_close": {"close": 100.0, "session_date": "2026-10-07",
                          "fetched_at": "2026-10-10T00:00:00Z", "cache_hit": False}}),
    "eur_stock": replace(SOUND, summary={**dm.EQUITY, "currency": "EUR"}),
    "pence_stock": replace(SOUND, summary=dm.GBP_EQUITY),
    "etf_bearish": dm.SCENARIOS["ETF_bearish"],
    "etf_spy": replace(SOUND, summary=dm.ETF, fundamentals={}),
    "etf_vti": replace(SOUND, summary={**dm.ETF, "name": "Vanguard Total Stock Market ETF"}, fundamentals={}),
    "preferred": replace(SOUND, summary={**dm.EQUITY, "name": "Acme Preferred Shares"}),
    "classification_unknown": replace(SOUND, summary={**dm.EQUITY, "quote_type": None}),
    "currency_unknown": replace(SOUND, summary={**dm.EQUITY, "currency": None}),
    "usd_adr": replace(SOUND, summary={**dm.EQUITY, "name": "Acme American Depositary Shares"}),
}

POSITION_CASES = {
    "gain": {},
    "loss": {"cost_basis": 120.0},
    "no_shares": {"shares": None},
    "fractional_shares": {"shares": 0.5},
    "zero_cost": {"cost_basis": 0.0},
    "negative_cost": {"cost_basis": -10.0},
    "zero_shares": {"shares": 0.0},
    "negative_shares": {"shares": -2.0},
    "exactly_365_days": {"purchase_date": "2025-10-09"},
    "exactly_366_days": {"purchase_date": "2025-10-08"},
    "leap_purchase": {"purchase_date": "2024-02-29"},
    "future_date": {"purchase_date": "2026-10-10"},
    "invalid_date": {"purchase_date": "2026-02-30"},
    "empty_latest": {"technicals": {"current_price": None}},
    "no_price": {"technicals": {"current_price": None, "session_close": None}},
    "zero_price": {"technicals": {"current_price": 0.0, "session_close": None}},
    "negative_price": {"technicals": {"current_price": -5.0, "session_close": None}},
    "price_disagreement": {"stock": "price_disagreement"},
    "stale_close": {"stock": "stale_daily"},
    "knife": {"stock": "sound_knife"},
    "broken_same_chart": {"stock": "broken_bullish"},
    "etf": {"stock": "etf_spy"},
    "preferred": {"stock": "preferred"},
    "eur": {"stock": "eur_stock"},
    "technical_error": {"technicals": {"error": True, "error_type": "data_unavailable",
                                        "message": "synthetic technicals failure"}},
}


def stock_digest(result: dict[str, Any]) -> dict[str, Any]:
    """Preserve absence, full snapshots and sizing, rather than inventing future fields."""
    return {key: copy.deepcopy(result[key]) for key in (
        "summary", "decision_card", "watchlist_snapshot", "executive_summary", "verdict",
        "fundamentals_summary", "dislocation_framework", "action_zones", "decision_modes",
    ) if key in result}


async def run_stock(name, *, reads=None):
    case = STOCK_CASES[name]
    with dm.stubbed(case), ExitStack() as stack:
        spies = {name: stack.enter_context(patch.object(
            dm.orchestrator, name, AsyncMock(wraps=getattr(dm.orchestrator, name)))) for name in dm.TOOLS}
        result = await dm.orchestrator.analyze_stock(
            "ACME", account_size=case.account_size, profile=case.profile, _now=case.now)
    if reads is not None:
        reads.update({name: spy.await_count for name, spy in spies.items()})
    return result


def position_inputs(name):
    spec = POSITION_CASES[name]
    case = STOCK_CASES[spec.get("stock", "sound_bullish")]
    tech = copy.deepcopy(case.payloads()["technicals"])
    tech["atr"] = {"value": 2.0}
    tech["price_position"]["low_1m"] = 85.0
    tech.update(copy.deepcopy(spec.get("technicals", {})))
    args = {"symbol": "ACME", "cost_basis": 80.0, "purchase_date": "2026-09-01", "shares": 20.0}
    args.update({key: value for key, value in spec.items() if key not in {"stock", "technicals"}})
    return args, tech


async def run_position(name, **overrides):
    args, tech = position_inputs(name)
    args.update(overrides)
    spec = POSITION_CASES[name]
    case = replace(STOCK_CASES[spec.get("stock", "sound_bullish")], technicals=tech,
                   now=POSITION_CLOCK)
    with dm.stubbed(case), ExitStack() as stack:
        spies = {name: stack.enter_context(patch.object(
            dm.orchestrator, name, AsyncMock(wraps=getattr(dm.orchestrator, name)))) for name in dm.TOOLS}
        result = await position.analyze_position(**args, _now=POSITION_CLOCK)
    result = copy.deepcopy(result)
    if "meta" in result:
        result["meta"].pop("duration_ms", None)
    # Source timing is measured separately; it is not deterministic business output.
    if "data_quality" in result:
        result["data_quality"].pop("tool_timings", None)
        for failure in result["data_quality"].get("tool_failures", []):
            failure.pop("duration_ms", None)
    reads = {name: spy.await_count for name, spy in spies.items()}
    return {"arguments": args, "technicals_input": tech,
            "technical_reads": reads["technicals"], "source_reads": reads, "result": result}


async def build():
    violations = []
    with patch.object(curl_cffi.requests.Session, "request", blocked_request("curl_cffi", violations)), \
            patch.object(requests.Session, "request", blocked_request("requests", violations)):
        stocks = {}
        for name, case in STOCK_CASES.items():
            reads = {}
            result = await run_stock(name, reads=reads)
            stocks[name] = {"inputs": case.payloads(), "profile": case.profile,
                            "account_size": case.account_size, "source_reads": reads,
                            "result": stock_digest(result)}
        positions = {name: await run_position(name) for name in POSITION_CASES}
    if violations:
        raise AssertionError(violation_report(violations))
    return {"baseline_code": BASE_COMMIT, "analyze_clock": dm.FRIDAY_EVENING.isoformat(),
            "position_clock": POSITION_CLOCK.isoformat(), "stocks": stocks, "positions": positions}


def markdown(recording):
    frozen = "recorded_code" in recording
    label = "v0.4.0 before" if frozen else "current output"
    holder_header = "" if frozen else " Long-term held / basis |"
    holder_separator = "" if frozen else "---|"
    lines = [f"# Holder/position: {label} table", "",
             "Generated by `tests.support.holder_matrix`; do not edit by hand.",
             "Synthetic observations, including defects; not desired behavior or investment/tax advice.",
             ("Long-term below means NEW MONEY. There is no long-term holder assessment in the baseline."
              if frozen else "Long-term new money and holder assessments are separate columns."),
             "Complete inputs, decision/sizing surfaces and snapshots are in the matching JSON.", "",
             "| Stock case | Mid-term new money | Mid-term held / basis | Long-term new money | Long-term fit |"
             + holder_header,
             "|---|---|---|---|---|" + holder_separator]
    for name, row in recording["stocks"].items():
        result = row["result"]
        card = result["decision_card"]
        held = card["if_held"]
        fit = result["verdict"]["horizon_fit"]["long_term"]
        holder_suffix = ""
        if not frozen:
            long_held = card["long_term"]["if_held"]
            holder_suffix = f" {long_held['action']} / {long_held['basis']} |"
        lines.append(f"| {name} | {card['new_money']['action']} | {held['action']} / {held['basis']} | "
                     f"{card['long_term']['new_money']['action']} | {fit} |{holder_suffix}")
    if frozen:
        lines += ["", "## Legacy position observations", "",
                  "Tax rates/classification and chart sell flags below describe the old implementation only.",
                  "The position tool reads technicals alone; it does not receive business or currency inputs.",
                  "The simulated host-local date is pinned independently of analyze's timezone-aware clock.", "",
                  "| Case | Reads | Price | P/L dollars | Days held | Tax type / rate | Sell flags | Error |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, row in recording["positions"].items():
            result = row["result"]
            pos, tax, holding = (result.get(key, {}) for key in ("position", "tax", "holding"))
            lines.append(f"| {name} | {row['technical_reads']} | {pos.get('current_price')} | "
                         f"{pos.get('gain_loss_dollars')} | {holding.get('days_held')} | "
                         f"{tax.get('type')} / {tax.get('rate_if_sold_now')} | "
                         f"{result.get('sell_signals', {}).get('active_count')} | {result.get('error_type', '—')} |")
    else:
        lines += ["", "## Current position observations", "",
                  "Each valid request runs one analysis: the eight stubbed source tools are each called once.",
                  "Counts below are tool calls, not network requests. Invalid inputs make no source calls.",
                  "Elapsed days use the injected America/New_York request date. Tax is not assessed.", "",
                  "| Case | Source calls | Holder / basis | Price | P/L dollars | Days held | Context | Allocation | Error |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, row in recording["positions"].items():
            result = row["result"]
            pos, holding, card = (result.get(key, {}) for key in ("position", "holding", "position_card"))
            holder = card.get("holder_assessment", {})
            lines.append(f"| {name} | {sum(row['source_reads'].values())} | "
                         f"{holder.get('action')} / {holder.get('basis')} | {pos.get('current_price')} | "
                         f"{pos.get('gain_loss_dollars')} | {holding.get('days_held')} | "
                         f"{card.get('context_status', {}).get('status')} | {card.get('review_status')} | "
                         f"{result.get('error_type', '—')} |")
    return "\n".join(lines) + "\n"


def write_recording(recording, directory):
    """Write current evidence only, preflighting both files before creating anything."""
    if "recorded_code" in recording:
        raise ValueError("frozen baseline cannot be relabelled as current output")
    files = {"after.json": json.dumps(recording, indent=2, sort_keys=True, allow_nan=False) + "\n",
             "after.md": markdown(recording)}
    for name, content in files.items():
        target = directory / name
        if target.exists() and target.read_bytes() != content.encode():
            raise ValueError(f"refusing to overwrite existing evidence: {target}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = directory / name
        if not target.exists():
            target.write_bytes(content.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--after", action="store_true")
    mode.add_argument("--baseline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.after:
        parser.error("the baseline is frozen; use --after for current output")
    write_recording(asyncio.run(build()), args.output_dir)


if __name__ == "__main__":
    main()
