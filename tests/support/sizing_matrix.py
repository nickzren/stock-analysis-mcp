"""Frozen v0.3.0 sizing evidence, including defects; not a future sizing implementation.

Run ``python -m tests.support.sizing_matrix --output-dir <directory>``.
Existing different recordings are never overwritten. All analyze tools are stubbed;
the transport guard also runs outside pytest. No live prices or user state are read.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import curl_cffi.requests
import requests

from stock_analysis.tools.analyze.investor_profile import resolve_investor_profile
from stock_analysis.tools.trade_setup.plan import build_plan
from tests.support import decision_matrix as dm
from tests.support.network_guard import blocked_request, violation_report

BASE_COMMIT = "7d581f11562cf68a6eafb6489ad7c67aa5f25bee"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sizing_matrix"

# Only deviations from the common synthetic inputs are repeated here.
CASES: dict[str, dict[str, Any]] = {
    **{f"{profile}_{distance}": {"profile": profile, "atr": atr}
       for profile in ("core", "balanced", "speculative")
       for distance, atr in (("narrow", 2.0), ("wide", 10.0))},
    "speculative_binding": {"profile": "speculative", "atr": 15.0},
    "risk_only": {"risk_per_trade_pct": 0.25},
    "cap_only_tight": {"max_position_pct": 2.0},
    "cap_only_wide": {"max_position_pct": 8.0},
    "both_overrides": {"risk_per_trade_pct": 0.25, "max_position_pct": 0.5},
    "tiny_risk": {"risk_per_trade_pct": 0.001},
    "tiny_cap": {"max_position_pct": 0.06},
    "small_account": {"account_size": 1000.0, "price": 50.0, "atr": 1.0},
    "no_account": {"account_size": None},
    "no_atr": {"atr": None},
    "negative_stop": {"price": 4.0, "atr": 1.8, "annualized": 0.75},
    "inverted_band": {"low_multiple": 1.05},
    "starter": {"starter": True},
    "earnings_blocked": {"earnings_blocked": True},
    "speculative_mode_override": {"template": "dislocation", "max_position_pct": 0.5},
}


def inputs(name: str) -> dict[str, Any]:
    base = {"profile": "balanced", "account_size": 10_000.0, "price": 100.0,
            "atr": 10.0, "annualized": 0.35, "low_multiple": 0.7}
    if CASES[name].get("template") == "dislocation":
        template = dm.SCENARIOS["T6_dislocation_buy"]
        base.update(price=template.technicals["current_price"], atr=template.risk["atr"]["value"],
                    annualized=template.risk["volatility"]["annualized"])
        base.pop("low_multiple")
    return {**base, **CASES[name]}


def scenario(name: str) -> dm.Scenario:
    values = inputs(name)
    if values.get("template") == "dislocation":
        return replace(dm.SCENARIOS["T6_dislocation_buy"], account_size=values["account_size"],
                       profile=values["profile"])
    price, atr = values["price"], values["atr"]
    tech = copy.deepcopy(dm.STRONG)
    tech["current_price"] = price
    tech["moving_averages"].update(sma_20=price * 1.08, sma_50=price * 1.06,
                                  sma_200=price * (0.9 if values.get("starter") else 1.01))
    tech["price_position"].update(week_52_low=price * values["low_multiple"], week_52_high=price * 1.4)
    risk = copy.deepcopy(dm.MODERATE)
    risk["volatility"]["annualized"] = values["annualized"]
    risk["atr"] = {"value": atr, "as_pct_of_price": None if atr is None else atr / price}
    return dm.Scenario("v0.3.0 sizing characterization", tech, copy.deepcopy(dm.QUALITY), risk,
                       events=dm.NEAR if values.get("earnings_blocked") else dm.FAR,
                       profile=values["profile"], account_size=values["account_size"])


async def run_case(name: str) -> dict[str, Any]:
    case = scenario(name)
    overrides = {key: CASES[name][key] for key in ("risk_per_trade_pct", "max_position_pct")
                 if key in CASES[name]}
    with dm.stubbed(case):
        return await dm.orchestrator.analyze_stock(
            "ACME", account_size=case.account_size, profile=case.profile, _now=case.now, **overrides)


def digest(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Sizing surfaces plus the actual decisions/snapshot, without timing-bearing metadata."""
    card, zones = result["decision_card"], result["action_zones"]
    values = inputs(name)
    return {
        "inputs": values,
        "limit_origins": {key: "caller" if key in CASES[name] else "profile"
                          for key in ("risk_per_trade_pct", "max_position_pct")},
        "resolved_profile": resolve_investor_profile(**{key: values[key] for key in (
            "profile", "account_size", "risk_per_trade_pct", "max_position_pct") if key in values}),
        "decision": dm.digest(result)["decision"],
        "snapshot": result["watchlist_snapshot"],
        "card": {key: card.get(key) for key in ("sizing", "entry", "exit")},
        "zones": {key: zones.get(key) for key in ("position_sizing_range", "stop_calculation",
                                                  "basis", "zone_warnings")},
        "modes": {name: {key: mode.get(key) for key in ("action", "starter_position", "max_position",
                                                       "capital_at_risk")}
                  for name, mode in result["decision_modes"].items()},
        "summary": result["executive_summary"],
        "tiers": {tier: {"sizing": payload["decision_card"]["sizing"],
                         "entry": payload["decision_card"]["entry"]}
                  for tier, payload in dm.tiers(result).items()},
    }


def swing_cases() -> dict[str, Any]:
    setup = {"type": "pullback_in_uptrend", "trigger_price": 100.0, "stop_price": 97.0,
             "trigger_satisfied": True, "trigger_condition": "synthetic trigger",
             "stop_basis": "synthetic_stop", "target_primary": None}
    cases = {"fractional_amounts": {"account_size": 10000.0, "risk_per_trade_pct": 1.0,
                                     "max_position_pct": 8.25},
             "zero_whole_shares": {"account_size": 1000.0, "risk_per_trade_pct": 1.0,
                                     "max_position_pct": 8.0},
             "no_account": {"account_size": None, "risk_per_trade_pct": 1.0,
                            "max_position_pct": 10.0}}
    return {name: {"inputs": {**limits, "entry": 100.0, "stop": 97.0},
                   "plan": build_plan(setup, action="trade_now", session="regular",
                                      now=dm.FRIDAY_EVENING, actionable_price=100.0, **limits)}
            for name, limits in cases.items()}


async def build() -> dict[str, Any]:
    violations: list[str] = []
    with patch.object(curl_cffi.requests.Session, "request", blocked_request("curl_cffi", violations)), \
            patch.object(requests.Session, "request", blocked_request("requests", violations)):
        rows = {name: digest(name, await run_case(name)) for name in CASES}
        swing = swing_cases()
    if violations:
        raise AssertionError(violation_report(violations))
    return {"recorded_code": BASE_COMMIT, "clock": dm.FRIDAY_EVENING.isoformat(),
            "notice": "Observed v0.3.0 behavior, including defects; not the approved future contract.",
            "analyze": rows, "swing": swing}


def _implied_loss(amount: Any, price: float, stop: float | None) -> str:
    if amount is None or stop is None or not 0 < stop < price:
        return "—"
    return f"{amount / price * (price - stop):g}"


def markdown(recording: dict[str, Any]) -> str:
    lines = ["# Sizing: v0.3.0 before table", "",
             "Generated from synthetic offline scenarios by `tests.support.sizing_matrix`. Do not edit by hand.",
             "This records defects, not recommendations. Details, full snapshots and limit origins: `before.json`.",
             "The loss column is computed for this report from the displayed dollars / price × stop distance;",
             "it is not a field emitted by v0.3.0 and can disagree with its whole-share count.", "",
             "Risk/maximum below are the resolved profile-helper values, not necessarily the raw input.",
             "For `tiny_risk`, the supplied 0.001% rounds to 0 here, then sizing falls back to 1%.", "",
             "| Case | Resolved risk / maximum (%), origin | New money / held | Stop | Starter / full (%) |"
             " Starter / full ($) | Starter whole shares | Implied loss starter / full ($) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, row in recording["analyze"].items():
        size = row["card"]["sizing"] or {}
        stop = row["card"]["exit"]["stop_or_invalidation"]
        price = row["inputs"]["price"]
        profile = row["resolved_profile"]
        limits = " / ".join(f"{profile[k]:g} ({row['limit_origins'][k]})"
                            for k in ("risk_per_trade_pct", "max_position_pct"))
        decision = row["decision"]
        lines.append(f"| {name} | {limits} | {decision['new_money']['action']} / {decision['if_held']['action']}"
                     f" | {stop} | {size.get('starter_pct')} / {size.get('full_pct_cap')}"
                     f" | {size.get('starter_dollars')} / {size.get('full_dollars')}"
                     f" | {size.get('whole_shares')} | {_implied_loss(size.get('starter_dollars'), price, stop)} /"
                     f" {_implied_loss(size.get('full_dollars'), price, stop)} |")
    lines += ["", "## Swing quantity mismatch", "",
              "| Case | Whole shares | Fractional shares | Position dollars | Reported max_loss_dollars |",
              "|---|---|---|---|---|"]
    for name, row in recording["swing"].items():
        plan = row["plan"]
        lines.append(f"| {name} | {plan['shares']} | {plan['fractional_shares']} |"
                     f" {plan['position_dollars']} | {plan['max_loss_dollars']} |")
    return "\n".join(lines) + "\n"


def write_recording(recording: dict[str, Any], directory: Path) -> None:
    files = {"before.json": json.dumps(recording, indent=2, sort_keys=True, allow_nan=False) + "\n",
             "before.md": markdown(recording)}
    for name, content in files.items():
        target = directory / name
        if target.exists() and target.read_bytes() != content.encode():
            raise ValueError(f"refusing to overwrite frozen baseline: {target}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = directory / name
        if not target.exists():
            target.write_bytes(content.encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_recording(asyncio.run(build()), args.output_dir)


if __name__ == "__main__":
    main()
