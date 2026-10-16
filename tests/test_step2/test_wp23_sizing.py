"""WP2.3: entry-sizing presence (plan §7 [FIX] keyed on entry_allowed, [D-9] a starter hides the full size).

Annex: docs/plans/step2-v2-research/output_contract.md §3, §10 "Corrected proposal", as amended by plan §0 (#12
entry_allowed, #21 mode max_position, #23 starter). Sizing numbers are unchanged (OC-G3); only presence moves.
"""

import json
import re
from typing import Any

import pytest

from stock_analysis.tools.analyze.decision import ENTRY_ACTIONS, FULL_ENTRY_CODES, NEW_MONEY_ACTIONS
from stock_analysis.tools.analyze.decision_card import build_entry_sizing
from stock_analysis.tools.analyze.executive_summary import size_text
from tests.support import decision_matrix as dm
from tests.support import freshness_frames as fr

BEFORE = json.loads((dm.FIXTURE_DIR / "before.json").read_text())
SIZE_TEXT = re.compile(r"size ~([\d.]+)(?:-([\d.]+))?%")
UNSUPPORTED = ["S7_etf_below_sma200", "ETF_strong", "ETF_bearish", "GBp_strong", "G_unsupported_currency_knife"]
STARTER_ROWS = ["S1_hold_or_add_clean", "T1_extreme_risk_bullish", "T2_starter_weak_setup_quality",
                "C1_headwind_hold_or_add", "E2_mixed_fcf_bullish_chart", "T4_starter_buy_the_dip"]
MODES = ("core", "balanced", "speculative")


def assert_sizing_presence(result: dict[str, Any]) -> None:
    """Plan §7's entry-sizing table, per path and tier, keyed on new_money.action only."""
    card = result["decision_card"]
    action = card["new_money"]["action"]
    entry, buy = action in ENTRY_ACTIONS, action == "buy"
    for tier, payload in dm.tiers(result).items():
        shown = payload["decision_card"]
        assert {"sizing", "entry"} <= shown.keys(), tier
        assert (shown["sizing"] is not None, shown["entry"] is not None) == (entry, entry), tier
        if entry:
            assert (shown["sizing"]["full_pct_cap"] is None, shown["sizing"]["full_dollars"] is None) == (
                not buy, not buy), tier
            assert set(shown["entry"]) == {"current_price", "buy_zone_low", "buy_zone_high"}, tier
        if tier == "decision":
            assert "action_zones" not in payload and "executive_summary" not in payload
            continue
        assert "position_sizing_range" in payload["action_zones"], tier
        assert (payload["action_zones"]["position_sizing_range"] is not None) is buy, tier
        sizes = SIZE_TEXT.findall(payload["executive_summary"])
        assert len(sizes) == int(entry), tier
        if entry:
            expected = f"size ~{shown['sizing']['starter_pct']:.1f}" + (
                f"-{shown['sizing']['full_pct_cap']:.1f}%" if buy else "%")
            assert expected in payload["executive_summary"], tier
    assert "policy_action" not in result
    for name in MODES:
        mode = result["decision_modes"][name]
        assert (mode["starter_position"] is not None) is (entry and mode["action"] in ENTRY_ACTIONS), name
        assert (mode["max_position"] is not None) is (entry and mode["action"] == "buy" and buy), name
    speculative = result["decision_modes"]["speculative"]
    assert (speculative["capital_at_risk"] is not None) is (speculative["starter_position"] is not None)
    dislocation = result["dislocation_framework"]["action"]
    assert dislocation["can_start_now"] is entry
    assert dislocation.get("buy_now") == (action if entry else None)


@pytest.mark.parametrize("name", list(dm.SCENARIOS))
async def test_entry_sizing_follows_new_money(name: str) -> None:
    """OC-RF2 (and RF5, the buy band): every sizing path per plan §7 at every tier: card sizing and the buy band
    iff new_money is buy or starter (keys always present); position_sizing_range iff buy; "size ~" iff entry;
    mode starter iff entry and the mode admits, mode max iff the mode and new_money are both buy (§0 #21);
    speculative capital_at_risk iff its starter; dislocation can_start_now == entry_allowed, buy_now the action;
    no policy_action (sizing_guidance deleted). PD D-9."""
    assert_sizing_presence(await dm.run(name))


@pytest.mark.parametrize("name", UNSUPPORTED)
async def test_unsupported_instruments_get_no_sizing(name: str) -> None:
    """OC-RF2: an unsupported instrument or currency always gates to a non-entry action, so no sizing path is
    shown at any tier (valuation_basis.py: "no sizing, dollar-liquidity claims or new-money action"). PD D-9."""
    result = await dm.run(name)
    support = result["summary"]["support"]

    assert "unsupported" in (support["instrument"]["status"], support["currency"]["status"])
    assert result["decision_card"]["new_money"]["action"] not in ENTRY_ACTIONS
    for tier, payload in dm.tiers(result).items():
        assert (payload["decision_card"]["sizing"], payload["decision_card"]["entry"]) == (None, None), tier
        assert (payload.get("action_zones") or {}).get("position_sizing_range") is None, tier
        assert not SIZE_TEXT.search(payload.get("executive_summary") or ""), tier


@pytest.mark.parametrize("name", STARTER_ROWS)
async def test_a_starter_hides_the_full_size_everywhere(name: str) -> None:
    """OC-RF3 (extended, plan §7:436): under new_money starter, at every tier, the card keeps the starter size
    (equal to the 2.0 before table on baseline rows) with full_pct_cap and full_dollars null; the zone band and
    every mode max_position are null; the summary shows "size ~<starter>%" only; dislocation buy_now is starter.
    PD D-9."""
    fund = dm.SCENARIOS[name].fundamentals
    if name in {"S1_hold_or_add_clean", "C1_headwind_hold_or_add"}:
        fund = {**fund, "financial_health": {"debt_to_equity": 0.5}}
    result = await dm.run(name, fundamentals=fund)
    card = result["decision_card"]
    assert card["new_money"]["action"] == "starter"

    for tier, payload in dm.tiers(result).items():
        sizing = payload["decision_card"]["sizing"]
        assert (sizing["full_pct_cap"], sizing["full_dollars"]) == (None, None), tier
        if name in BEFORE:
            before = BEFORE[name]["card_sizing"]
            assert {k: sizing[k] for k in ("starter_pct", "starter_dollars", "whole_shares")} == {
                k: before[k] for k in ("starter_pct", "starter_dollars", "whole_shares")}, tier
        if tier != "decision":
            assert payload["action_zones"]["position_sizing_range"] is None, tier
            assert SIZE_TEXT.findall(payload["executive_summary"]) == [(f"{sizing['starter_pct']:.1f}", "")], tier
    assert [result["decision_modes"][m]["max_position"] for m in MODES] == [None, None, None]
    assert result["dislocation_framework"]["action"]["buy_now"] == "starter"


@pytest.mark.parametrize("name", STARTER_ROWS)
async def test_a_starter_shows_no_full_entry_dip_code(name: str) -> None:
    """OC-RF3 (dip clause; lands with the dip area's §7 cap): under new_money starter neither the dip
    recommendation nor any entry signal is a full-entry code (strong_buy_the_dip, buy_the_dip,
    cautious_accumulation, accumulate, add_position). PD D-9."""
    fund = dm.SCENARIOS[name].fundamentals
    if name in {"S1_hold_or_add_clean", "C1_headwind_hold_or_add"}:
        fund = {**fund, "financial_health": {"debt_to_equity": 0.5}}
    result = await dm.run(name, fundamentals=fund)
    assessment = result["dip_assessment"]["assessment"]
    signals = result["dip_assessment"]["entry_timing"]["signals"]

    assert result["decision_card"]["new_money"]["action"] == "starter"
    assert assessment["recommendation"] not in FULL_ENTRY_CODES
    assert not FULL_ENTRY_CODES & {signal["action"] for signal in signals}


async def test_the_starter_scenario_caps_its_full_entry_dip_codes() -> None:
    """OC-RF3 (the plan §7 starter scenario, T4): dip data that would say buy_the_dip with accumulate and
    add_position signals is shown as starter_only and two starter_position signals, each capped_by
    new_money_starter, next to a starter-only card ($250, 2 shares, no full size), "size ~2.5%", balanced's
    starter alone and core's starter without its max (core is buy, new_money is not). PD D-9."""
    result = await dm.run("T4_starter_buy_the_dip")
    dip = result["dip_assessment"]
    modes = result["decision_modes"]

    assert (result["decision_card"]["new_money"]["action"], dm.gate_ids(result)) == ("starter", [])
    assert (dip["dip_classification"]["type"], dip["oversold_metrics"]["level"]) == ("healthy_pullback",
                                                                                     "mildly_oversold")
    assert (dip["assessment"]["recommendation"], dip["assessment"]["capped_by"]) == ("starter_only",
                                                                                     "new_money_starter")
    assert [(s["signal"], s["action"], s["capped_by"]) for s in dip["entry_timing"]["signals"]] == [
        ("pullback_in_uptrend", "starter_position", "new_money_starter"),
        ("break_5d_high", "starter_position", "new_money_starter")]
    for tier, payload in dm.tiers(result).items():
        sizing = payload["decision_card"]["sizing"]
        assert (sizing["starter_pct"], sizing["starter_dollars"], sizing["whole_shares"], sizing["full_pct_cap"],
                sizing["full_dollars"]) == (2.5, 250.0, 2, None, None), tier
    assert "size ~2.5%." in result["executive_summary"]
    assert (modes["balanced"]["action"], modes["balanced"]["starter_position"], modes["balanced"]["max_position"]) == (
        "starter", {"pct": 2.5, "dollars": 250.0, "shares": 2}, None)
    assert (modes["core"]["action"], modes["core"]["starter_position"]["pct"], modes["core"]["max_position"]) == (
        "buy", 2.0, None)


async def test_a_freshness_gate_hides_every_size() -> None:
    """FR-R-F2 (sizing clause): R3 (fresh 5m, technicals 2 sessions behind; starter without the gate) waits for
    data and shows no sizing on any path at any tier. RF (PD D-7, D-9)."""
    result = await dm.run(fr.scenario(fr.TRUTH_TABLE["R3"], "R3"))

    assert (result["decision_card"]["new_money"]["action"], result["decision_card"]["new_money"]["decided_by"]) == (
        "wait_for_data", "stale_data")
    assert_sizing_presence(result)
    assert result["decision_card"]["sizing"] is None
    assert result["diagnostics"]["pre_gate_policy"]["mid_term"] == "hold_or_add"  # admits without the gate


@pytest.mark.parametrize(("action", "full"), [(a, a == "buy") for a in NEW_MONEY_ACTIONS])
def test_entry_sizing_by_action(action: str, full: bool) -> None:
    """OC-RF2/RF3 (unit): build_entry_sizing gives the card sizing only for buy and starter, and a starter's
    full_pct_cap and full_dollars are null; size_text gives the band under buy, the starter pct alone under a
    starter, and nothing under a non-entry action whatever sizing it is handed. PD D-9."""
    zones = {"position_sizing_range": {"starter_pct": 3.0, "max_pct": 8.0, "dollars_for_account": {
        "min": 150.0, "max": 400.0}, "shares_range": {"min": 1, "max": 4, "at_price": 100.0}}}
    sizing = build_entry_sizing(zones, {"current_price": 100.0}, action)

    if action not in ENTRY_ACTIONS:
        assert (sizing, size_text(action, sizing)) == (None, None)
        assert size_text(action, build_entry_sizing(zones, {"current_price": 100.0}, "buy")) is None
        return
    assert (sizing["starter_pct"], sizing["starter_dollars"], sizing["whole_shares"]) == (3.0, 150.0, 1)
    assert (sizing["full_pct_cap"], sizing["full_dollars"]) == ((8.0, 400.0) if full else (None, None))
    assert size_text(action, sizing) == ("size ~3.0-8.0%" if full else "size ~3.0%")


@pytest.mark.parametrize("name", list(dm.SCENARIOS))
async def test_the_buy_band_needs_entry(name: str) -> None:
    """OC-RF5: decision_card.entry (the buy band) is {current_price, buy_zone_low, buy_zone_high} iff new_money is
    buy or starter and null otherwise (the key always present), at every tier; no zone field is left on the card
    (current_zone removed, price_zone is a diagnostic under action_zones). RF."""
    result = await dm.run(name)
    levels = result["action_zones"]["levels"]
    entry = result["decision_card"]["new_money"]["action"] in ENTRY_ACTIONS
    expected = {"current_price": result["summary"]["current_price"], "buy_zone_low": levels.get("strong_buy_below"),
                "buy_zone_high": levels.get("accumulate_near")} if entry else None
    for tier, payload in dm.tiers(result).items():
        card = payload["decision_card"]
        assert "entry" in card and card["entry"] == expected, tier
        assert not re.search(r"\b(current_zone|price_zone|zone)\b", json.dumps(sorted(_keys(card)))), tier


def _keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for value in node.values() for k in _keys(value)}
    if isinstance(node, list):
        return {k for value in node for k in _keys(value)}
    return set()
