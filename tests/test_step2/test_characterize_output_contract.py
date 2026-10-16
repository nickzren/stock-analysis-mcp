"""WP2.0: output-contract characterization (annex step2-v2-research/output_contract.md §10).

Each test records 0609fde behavior. CH tests name the WP that flips them; G tests hold
through 2a. Ported from the annex probes: probe_sizing.py (CH1-CH4), probe_starter.py (CH2),
probe_conditions.py (CH5, G1), probe_lint.py (CH6), probe_zone_snapshot.py (CH7, G6),
probe_profile.py (CH-2b, G2), probe_vendor_decision_tier.py (G1, G5) and
test_contract_prototype.py (CH1, G1). Pipeline cases run through
tests/support/decision_matrix.py, which pins the clock. Plan §0 #16 and §7(a) amend the annex
(G4, G6).
"""

import json
import re
from dataclasses import replace
from typing import Any

import pytest

from stock_analysis.tools.analyze.action_zones import (
    apply_dip_gates_to_action_zones,
    build_action_zones,
)
from stock_analysis.tools.analyze.verdict import build_horizon_fit
from tests.support import decision_matrix as dm

BEFORE = json.loads((dm.FIXTURE_DIR / "before.json").read_text())
HEADWIND_ROWS = ["C1_headwind_hold_or_add", "C2_headwind_earnings_gate", "C3_headwind_plain_avoid"]
ENTRY_ROWS = ["S2_buy_clean", "T1_extreme_risk_bullish", "T2_starter_weak_setup_quality"]
# Mode (starter_position, max_position) as (pct, dollars, shares) on 0609fde; before.json records the card only.
_PRESET_MODES = {"core": ((2.0, 200.0, 2), (8.0, 800.0, 8)), "balanced": ((2.5, 250.0, 2), (8.0, 800.0, 8)),
                 "speculative": ((1.0, 100.0, 1), (6.0, 600.0, 6))}
MODE_SIZING = {"S2_buy_clean": _PRESET_MODES, "T2_starter_weak_setup_quality": _PRESET_MODES,
               "T1_extreme_risk_bullish": dict.fromkeys(_PRESET_MODES, ((0.5, 50.0, 0), (3.0, 300.0, 3)))}
VENDOR_ROWS = ["S1_hold_or_add_clean", "S3_plain_avoid", "V1_vendor_rich_plain_avoid"]
KNIFE_ROWS = {name for name, row in BEFORE.items() if row["dip"]["type"] == "falling_knife"}
PROFILES_AND_CAPS: list[dict[str, Any]] = [
    {"profile": "balanced"},
    {"profile": "speculative"},
    {"profile": "core"},
    {"profile": "balanced", "max_position_pct": 2.0, "risk_per_trade_pct": 0.25},
]

SIZE_TEXT = re.compile(r"size ~([\d.]+)(?:-([\d.]+))?%")
CONDITIONAL_PATH = re.compile(r"(^|\.)(\w+_if|monitor|wait_for|required_before_full_size)(\[\*\])?(\.|$)")
ENTRY_TOKEN = re.compile(r"(?i)(?<![a-z])(buy|add|accumulate|starter)(?![a-z])")
IMPERATIVE = re.compile(
    r"(?i)^(buy|add|accumulate|start|enter|sell|exit|reduce|trim|hold|wait|avoid|consider)\b"
)
THIRD_PARTY_KEY = re.compile(
    r"analyst|news|headline|insider|institutional|ownership|options|short_interest|catalyst|governance"
)

# probe_zone_snapshot (a): a price at the 52-week low classifies as strong_buy before any cap.
AT_52W_LOW = {"current_price": 70.0, "moving_averages": {"sma_50": 80.0, "sma_200": 90.0},
              "price_position": {"week_52_low": 70.0, "week_52_high": 120.0}}
CHEAP = {"valuation": {"pe_trailing": 9.0}, "yield_metrics": {"fcf_yield": 0.09}}
EXTREME_VOL = {"volatility": {"annualized": 0.75}, "atr": {"value": 1.0, "as_pct_of_price": 0.014}}
CALM_VOL = {"volatility": {"annualized": 0.22}, "atr": {"value": 1.0, "as_pct_of_price": 0.014}}
UNPROFITABLE = {"valuation": {"trailing_eps": -1.0}, "profitability": {"net_margin": -0.15}}
# POOR_NOT_KNIFE moved to its 52-week low: knife score 2 (below SMA200, near the low), pullback 1.
EXTENDED_DECLINE = {
    **dm.POOR_NOT_KNIFE,
    "current_price": 70.0,
    "moving_averages": {**dm.POOR_NOT_KNIFE["moving_averages"], "sma_20": 72.0, "sma_50": 80.0, "sma_200": 90.0},
    "returns": {**dm.POOR_NOT_KNIFE["returns"], "return_3m": -0.10},
    "price_position": {"position_in_range": 0.02, "week_52_low": 69.0, "week_52_high": 120.0,
                       "from_52w_high": -0.42, "days_since_52w_high": 100},
}


def leaves(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Every (path, leaf) of a payload, with list indices collapsed to [*]."""
    if isinstance(node, dict):
        return [x for key, value in node.items() for x in leaves(value, f"{path}.{key}" if path else key)]
    if isinstance(node, list):
        return [x for value in node for x in leaves(value, f"{path}[*]")]
    return [(path, node)]


async def analyze(name: str, *, known_debt: bool = False, **kwargs: Any) -> dict[str, Any]:
    """Run profile/cap cases; known debt isolates sizing arithmetic from missing thesis evidence."""
    scenario = dm.SCENARIOS[name]
    if known_debt:
        scenario = replace(scenario, fundamentals={
            **scenario.fundamentals, "financial_health": {"debt_to_equity": 0.5}})
    with dm.stubbed(scenario):
        result: dict[str, Any] = await dm.orchestrator.analyze_stock(
            "ACME", account_size=scenario.account_size, _now=scenario.now, **kwargs
        )
    return result


def sizing_presence(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Which entry-sizing paths (annex §3 P1-P6, P9) are non-null, per tier."""
    presence: dict[str, dict[str, Any]] = {}
    for tier, payload in dm.tiers(result).items():
        card = payload["decision_card"]
        dislocation = (payload.get("dislocation_framework") or {}).get("action") or {}
        presence[tier] = {
            "card.sizing": card.get("sizing") is not None,
            "card.entry": card.get("entry") is not None,
            "position_sizing_range": (payload.get("action_zones") or {}).get("position_sizing_range") is not None,
            "summary_size": bool(SIZE_TEXT.search(payload.get("executive_summary") or "")),
            "modes": {name: tuple(mode.get(key) is not None
                                  for key in ("starter_position", "max_position", "capital_at_risk"))
                      for name, mode in (payload.get("decision_modes") or {}).items()},
            "can_start_now": dislocation.get("can_start_now"),
            "buy_now": dislocation.get("buy_now") is not None,
        }
    return presence


class TestEntrySizingUnderNonEntryActions:
    async def test_plain_avoid_carries_no_entry_sizing(self) -> None:
        """OC-CH1 (flipped by WP2.3): S3, a candidate avoid with unknown thesis, carries no entry sizing on any path: card
        sizing and buy band null at every tier, position_sizing_range null (key kept) and no "size ~" text at
        standard and full, no sizing_guidance (policy_action moved to diagnostics), no mode size and no
        capital_at_risk; no current_zone (price_zone below_sma200); the dip's buy_the_dip and entry codes are watch,
        capped by new_money; core follows the long-term wait_for_data cap. G from WP2.3 (PD D-9; plan §7)."""
        result = await dm.run("S3_plain_avoid")
        shown = dm.tiers(result)
        card = result["decision_card"]
        modes = result["decision_modes"]
        dip = result["dip_assessment"]

        assert (card["action_now"], dm.gate_ids(result)) == ("avoid", ["thesis_unverifiable"])
        for tier, payload in shown.items():
            assert (payload["decision_card"]["sizing"], payload["decision_card"]["entry"]) == (None, None), tier
        for tier in ("standard", "full"):
            assert "position_sizing_range" in shown[tier]["action_zones"], tier
            assert shown[tier]["action_zones"]["position_sizing_range"] is None, tier
            assert not SIZE_TEXT.search(shown[tier]["executive_summary"]), tier
        assert "policy_action" not in result
        assert "sizing_guidance" not in result["diagnostics"]["pre_gate_policy"]
        assert {name: (mode["starter_position"], mode["max_position"]) for name, mode in modes.items()} == dict.fromkeys(
            ("core", "balanced", "speculative"), (None, None))
        assert modes["speculative"]["capital_at_risk"] is None
        assert ("current_zone" in result["action_zones"], result["action_zones"]["price_zone"]) == (
            False, "below_sma200")
        assert (dip["assessment"]["recommendation"], dip["assessment"]["capped_by"]) == ("watch", "new_money_avoid")
        assert [(s["action"], s["capped_by"]) for s in dip["entry_timing"]["signals"]] == [
            ("watch", "new_money_avoid")] * 2
        assert modes["core"]["action"] == "wait_for_data"

    @pytest.mark.parametrize(("name", "action"), [("T3_plain_wait_weak_setup", "wait_for_data"),
                                                  ("C3_headwind_plain_avoid", "avoid")])
    async def test_plain_wait_and_headwind_avoid_carry_no_card_sizing(self, name: str, action: str) -> None:
        """OC-CH2 (flipped by WP2.3): T3 (wait for thesis data) and C3 (headwind avoid), both thesis gated, carry no card sizing
        and no buy band at any tier: sizing is keyed on entry_allowed. G from WP2.3 (PD D-9; plan §7)."""
        result = await dm.run(name)

        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == (action, ["thesis_unverifiable"])
        for tier, payload in dm.tiers(result).items():
            assert (payload["decision_card"]["sizing"], payload["decision_card"]["entry"]) == (None, None), tier

    async def test_starter_hides_the_full_size(self) -> None:
        """OC-CH2 (flipped by WP2.3): T2 (starter) shows only the starter size: card full_pct_cap and full_dollars
        null at every tier, no position_sizing_range, "size ~2.5%" with no full figure, and no mode max_position.
        G from WP2.3 (PD D-9: a starter hides the full size; plan §0 #21, #23)."""
        result = await dm.run("T2_starter_weak_setup_quality")
        shown = dm.tiers(result)

        assert result["decision_card"]["action_now"] == "starter"
        for tier, payload in shown.items():
            sizing = payload["decision_card"]["sizing"]
            assert (sizing["starter_pct"], sizing["starter_dollars"], sizing["full_pct_cap"],
                    sizing["full_dollars"]) == (2.5, 250.0, None, None), tier
        for tier in ("standard", "full"):
            assert shown[tier]["action_zones"]["position_sizing_range"] is None, tier
            assert SIZE_TEXT.findall(shown[tier]["executive_summary"]) == [("2.5", "")], tier
        assert [name for name, mode in result["decision_modes"].items() if mode["max_position"] is not None] == []

    async def test_knife_avoid_offers_no_starter(self) -> None:
        """OC-CH3 (flipped by WP2.3): S6 (avoid + falling_knife) has dislocation can_start_now false and buy_now
        null, speculative avoid (capped by new_money), no dislocation starter override on the dip and its
        starter_position entry signal capped to watch. G from WP2.3 (PD D-9, D-3; plan §7)."""
        result = await dm.run("S6_knife_dislocation")
        dislocation = result["dislocation_framework"]["action"]
        dip = result["dip_assessment"]

        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == ("avoid", ["falling_knife"])
        assert (dislocation["can_start_now"], dislocation["buy_now"]) == (False, None)
        assert result["decision_modes"]["speculative"]["action"] == "avoid"
        assert (dip["dip_classification"]["type"], dip["assessment"]["recommendation"]) == (
            "falling_knife", "do_not_catch_falling_knife")
        assert [(s["action"], s["capped_by"]) for s in dip["entry_timing"]["signals"]] == [("watch", "new_money_avoid")]

    async def test_unsupported_etf_gets_no_sizing_and_no_zone_on_the_card(self) -> None:
        """OC-CH4 (flipped by WP2.3): S7 (ETF, unsupported_instrument, new_money watch: the support gate tightens
        wait, WP2.2 D-5) carries no card sizing and no buy band at any tier, no position_sizing_range and no
        "size ~" text; action_zones has price_zone below_sma200 and no current_zone. G from WP2.3 (PD D-9; plan
        §7: unsupported instruments get no sizing at all)."""
        result = await dm.run("S7_etf_below_sma200")

        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == ("watch", ["unsupported_instrument"])
        for tier, payload in dm.tiers(result).items():
            card = payload["decision_card"]
            assert (card["sizing"], card["entry"]) == (None, None), tier
        assert result["action_zones"]["position_sizing_range"] is None
        assert not SIZE_TEXT.search(result["executive_summary"])
        assert ("current_zone" in result["action_zones"], result["action_zones"]["price_zone"]) == (
            False, "below_sma200")

    async def test_unsupported_etf_snapshot_carries_the_neutral_price_zone(self) -> None:
        """OC-CH4 (snapshot half, flipped by WP2.4): S7's watchlist_snapshot carries price_zone below_sma200 (the
        action_zones diagnostic) and no zone code. G from WP2.4 (PD D-9, D-10; plan §8, §0 #10)."""
        result = await dm.run("S7_etf_below_sma200")
        snapshot = result["watchlist_snapshot"]

        assert (snapshot["price_zone"], "zone" in snapshot) == (result["action_zones"]["price_zone"], False)
        assert snapshot["price_zone"] == "below_sma200"


class TestConditionsUnderAdmittingActions:
    async def test_headwind_starter_lists_the_valuation_condition_at_long_term_only(self) -> None:
        """OC-CH5 (flipped by WP2.3): C1 is a mid-term starter whose valuation headwind does not bind at mid term,
        so the card's buy_only_if and add_only_if are empty and monitor keeps the upgrade condition; the long
        term (wait from the headwind) lists it in long_term.new_money.conditions.buy_only_if and in core's
        required_before_full_size. G from WP2.3 (§7e, RF-E1's placement)."""
        result = await dm.run("C1_headwind_hold_or_add", fundamentals={
            **dm.HEADWIND, "financial_health": {"debt_to_equity": 0.5}})
        card = result["decision_card"]
        conditions = card["conditions"]
        upgrade = result["diagnostics"]["pre_gate_policy"]["conditions_to_upgrade"]

        assert (card["action_now"], card["long_term"]["new_money"]["action"]) == ("starter", "wait")
        assert (conditions["buy_only_if"], conditions["add_only_if"]) == ([], [])
        assert len(upgrade) == 1 and conditions["monitor"] == upgrade
        assert card["long_term"]["new_money"]["conditions"]["buy_only_if"] == upgrade
        assert result["decision_modes"]["core"]["required_before_full_size"] == upgrade


class TestSizingWordsInCodesAndText:
    @pytest.mark.parametrize("name", HEADWIND_ROWS)
    async def test_headwind_rows_put_no_sizing_code_on_the_card(self, name: str) -> None:
        """OC-CH6 (flipped by WP2.3): C1-C3 have no card.horizon_fit at any tier and their long-term action is
        long_term.new_money.action wait_for_data (missing thesis evidence). G from WP2.3 (PD D-9)."""
        for tier, payload in dm.tiers(await dm.run(name)).items():
            card = payload["decision_card"]
            assert ("horizon_fit" in card, card["long_term"]["new_money"]["action"]) == (False, "wait_for_data"), tier

    @pytest.mark.parametrize("name", HEADWIND_ROWS)
    async def test_headwind_rows_put_no_sizing_code_in_the_snapshot(self, name: str) -> None:
        """OC-CH6 (snapshot half, flipped by WP2.4): C1-C3 record long_term_action wait_for_data (long_term.new_money.action,
        plan §0 #8, #18), no action_long_term and no sizing word in any snapshot value. G from WP2.4 (PD D-9)."""
        snapshot = (await dm.run(name))["watchlist_snapshot"]

        assert (snapshot["long_term_action"], "action_long_term" in snapshot) == ("wait_for_data", False)
        assert not [v for v in snapshot.values() if isinstance(v, str) and re.search(r"(?i)siz", v)]

    def test_mixed_fundamentals_reason_carries_no_sizing_advice(self) -> None:
        """OC-CH6 (flipped by WP2.3): the verdict's long-term reason for mixed fundamentals names the state, not a
        position size (verdict.py:1020). G from WP2.3 (PD D-9; plan §7c D-text rewrites)."""
        fit = build_horizon_fit(setup_label="strong", business_quality_label="mixed",
                                business_quality_status="available", risk_label="medium",
                                signals={"bullish": [], "bearish": []})

        assert fit["long_term"] == "caution"
        assert [r for r in fit["reasons"] if r.startswith("long_term: mixed fundamentals")]
        assert not [r for r in fit["reasons"] if re.search(r"(?i)position|siz", r)]


class TestZoneCapsLeavePriceZoneUncapped:
    def test_extreme_risk_cap_leaves_the_uncapped_price_zone(self) -> None:
        """OC-CH7 (flipped by WP2.3): an extreme regime still caps the internal label strong_buy to accumulate
        (the cap warning stays), and price_zone keeps the uncapped classification, near_52w_low.
        G from WP2.3 (PD D-9; plan §7b; RF6 owns the five cap cases)."""
        zones = build_action_zones(current_price=70.0, tech_data=AT_52W_LOW, risk_data=EXTREME_VOL,
                                   fund_data=CHEAP, risk_regime={"classification": "extreme"})

        assert (zones["current_zone"], zones["price_zone"]) == ("accumulate", "near_52w_low")
        assert "zone_capped_due_to_extreme_risk" in zones["zone_warnings"]

    def test_extended_decline_cap_leaves_the_uncapped_price_zone(self) -> None:
        """OC-CH7 (flipped by WP2.3): the extended-decline dip gate caps the internal label strong_buy to
        accumulate, and price_zone stays near_52w_low. G from WP2.3 (PD D-9; plan §7b)."""
        regime = {"classification": "medium"}
        zones = build_action_zones(current_price=70.0, tech_data=AT_52W_LOW, risk_data=CALM_VOL,
                                   fund_data=CHEAP, risk_regime=regime)
        assert zones["current_zone"] == "strong_buy"

        capped = apply_dip_gates_to_action_zones(zones, {"dip_classification": {"type": "extended_decline"}}, regime)

        assert (capped["current_zone"], capped["zone_warnings"]) == ("accumulate", ["zone_capped_extended_decline"])
        assert capped["price_zone"] == "near_52w_low"

    async def test_extended_decline_label_stays_out_of_the_actionable_fields(self) -> None:
        """OC-CH7 (flipped by WP2.3): end to end, an S3 variant at its 52-week low (extended decline, calm risk)
        is a plain avoid with no zone on the card (entry null) or in action_zones.current_zone at any tier;
        action_zones.price_zone is near_52w_low and the capped accumulate label is only the full tier's
        diagnostics.pre_gate_policy.inputs.zone. G from WP2.3 (PD D-9; plan §7b, §0 #22)."""
        result = await dm.run("S3_plain_avoid", technicals=EXTENDED_DECLINE)
        zones = result["action_zones"]

        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == ("avoid", ["thesis_unverifiable"])
        assert result["dip_assessment"]["dip_classification"]["type"] == "extended_decline"
        assert (zones["price_zone"], zones["zone_warnings"]) == ("near_52w_low", ["zone_capped_extended_decline"])
        assert result["diagnostics"]["pre_gate_policy"]["inputs"]["zone"] == "accumulate"
        for tier, payload in dm.tiers(result).items():
            assert payload["decision_card"]["entry"] is None, tier
            if "action_zones" in payload:
                assert "current_zone" not in payload["action_zones"], tier
                assert payload["action_zones"]["price_zone"] == "near_52w_low", tier


class TestModesIgnoreUserCaps:
    async def test_balanced_max_position_ignores_max_position_pct(self) -> None:
        """OC-CH-2b: S2 with max_position_pct=2 caps the card at 2%, but modes.balanced.max_position stays 8%.
        Flips only in 2b (sizing arithmetic); 2a changes presence, never numbers."""
        result = await analyze("S2_buy_clean", known_debt=True, max_position_pct=2.0)

        assert result["decision_card"]["sizing"]["full_pct_cap"] == 2.0
        assert result["decision_modes"]["balanced"]["max_position"]["pct"] == 8.0


class TestConditionalValues:
    @pytest.mark.parametrize("name", list(dm.SCENARIOS))
    async def test_conditional_values_are_predicates(self, name: str) -> None:
        """OC-G1: every conditional value (*_if, monitor, wait_for, required_before_full_size) at every tier
        is a predicate: no buy/add/accumulate/starter token and no leading imperative (R-LEX-3).
        Holds throughout 2a, which changes which predicates are listed, not their form."""
        checked = 0
        for tier, payload in dm.tiers(await dm.run(name)).items():
            for path, value in leaves(payload):
                if CONDITIONAL_PATH.search(path) and isinstance(value, str):
                    checked += 1
                    assert not ENTRY_TOKEN.search(value), (tier, path, value)
                    assert not IMPERATIVE.match(value), (tier, path, value)
        assert checked

    async def test_third_party_paths_carry_buy_words(self) -> None:
        """OC-G1: on V1 the third-party paths carry buy words (analyst view, headlines, the insider pro and
        the insider_buying signal), so a payload-wide buy regex is wrong and the lint must be path-scoped.
        Holds throughout 2a, which never rewrites vendor content."""
        shown = dm.tiers(await dm.run("V1_vendor_rich_plain_avoid"))
        standard, decision = shown["standard"], shown["decision"]
        blanket = re.compile(r"(?i)buy")

        assert decision["decision_card"]["action_now"] == "avoid"
        assert blanket.search(standard["section_summaries"]["analyst_view"])
        assert any(blanket.search(h) for h in standard["news_summary"]["headlines"])
        assert any(blanket.search(p) for p in decision["verdict"]["pros"])
        assert "insider_buying" in decision["watchlist_snapshot"]["bullish_signals"]


class TestProfileInvariance:
    @pytest.mark.parametrize("name", list(dm.SCENARIOS))
    async def test_profile_and_caps_change_neither_the_action_nor_sizing_presence(self, name: str) -> None:
        """OC-G2: across profiles {balanced, speculative, core} and user caps, action_now (new_money.action
        from WP2.2) and which sizing paths are present stay the same. Holds throughout 2a:
        profile stays sizing-only and presence is keyed on entry_allowed (plan §3, §7)."""
        runs = [await analyze(name, **kwargs) for kwargs in PROFILES_AND_CAPS]

        assert len({run["decision_card"]["action_now"] for run in runs}) == 1
        presence = [sizing_presence(run) for run in runs]
        assert all(p == presence[0] for p in presence)

    async def test_profile_and_caps_reach_the_sizing_numbers(self) -> None:
        """OC-G2: the profile and caps do change S2's card sizing, so the invariance above is not vacuous.
        Known debt restores the S2 buy to isolate sizing arithmetic."""
        runs = [await analyze("S2_buy_clean", known_debt=True, **kwargs) for kwargs in PROFILES_AND_CAPS]

        assert [(run["decision_card"]["sizing"]["starter_pct"], run["decision_card"]["sizing"]["full_pct_cap"])
                for run in runs] == [(2.5, 8.0), (1.0, 6.0), (2.0, 8.0), (2.0, 2.0)]


class TestSizingNumbers:
    @pytest.mark.parametrize("name", ENTRY_ROWS)
    async def test_present_sizing_numbers_equal_the_before_table(self, name: str) -> None:
        """OC-G3: on the entry rows (S2 with known debt, T1 and T2 starter) every sizing number still shown, on the card,
        the zone band and the summary text at every tier, equals the 2.0 before table, and each mode size
        still shown equals its 0609fde value. Holds throughout 2a, which changes presence only (2b owns
        the arithmetic)."""
        fund = dm.SCENARIOS[name].fundamentals
        if name == "S2_buy_clean":
            fund = {**fund, "financial_health": {"debt_to_equity": 0.5}}
        result = await dm.run(name, fundamentals=fund)
        before = BEFORE[name]["card_sizing"]
        for tier, payload in dm.tiers(result).items():
            sizing = payload["decision_card"]["sizing"]
            shown = {key: sizing[key] for key in before if sizing.get(key) is not None}
            assert {"starter_pct", "starter_dollars"} <= shown.keys(), tier
            assert shown == {key: before[key] for key in shown}, tier

            band = (payload.get("action_zones") or {}).get("position_sizing_range")
            if band is not None:
                assert (band["starter_pct"], band["max_pct"]) == (before["starter_pct"], before["full_pct_cap"]), tier
                dollars = band["dollars_for_account"]
                assert (dollars["min"], dollars["max"]) == (before["starter_dollars"], before["full_dollars"]), tier
            for low, high in SIZE_TEXT.findall(payload.get("executive_summary") or ""):
                assert float(low) == before["starter_pct"], tier
                assert not high or float(high) == before["full_pct_cap"], tier

        for mode, recorded in MODE_SIZING[name].items():
            for key, expected in zip(("starter_position", "max_position"), recorded, strict=True):
                size = result["decision_modes"][mode].get(key)
                if size is not None:
                    assert (size["pct"], size["dollars"], size["shares"]) == expected, (mode, key)


class TestPreGateCandidate:
    @pytest.mark.parametrize("name", sorted(set(BEFORE) - KNIFE_ROWS))
    async def test_pre_gate_codes_equal_the_before_table(self, name: str) -> None:
        """OC-G4: the pre-gate candidate's mid- and long-term codes equal before.json's policy column.
        Holds throughout 2a: the knife rows are exempt because the knife override is the one
        candidate rule 2a removes (plan §0 #16, §1)."""
        policy = (await dm.run(name))["diagnostics"]["pre_gate_policy"]

        assert {"mid_term": policy["mid_term"], "long_term": policy["long_term"]} == BEFORE[name]["policy"]

    def test_the_exempt_knife_rows_are_the_falling_knife_rows(self) -> None:
        """OC-G4: the exemption covers exactly the four recorded falling_knife rows. Holds throughout 2a."""
        knife_rows = {"B_broken_bearish_knife", "C_quality_knife_dislocation",
                      "G_unsupported_currency_knife", "S6_knife_dislocation"}

        assert knife_rows == KNIFE_ROWS


class TestThirdPartyConfinement:
    @pytest.mark.parametrize("name", VENDOR_ROWS)
    async def test_third_party_content_stays_out_of_the_card_and_the_decision_tier(self, name: str) -> None:
        """OC-G5: no third-party path or text inside decision_card at any tier; analyst_coverage,
        news_summary and section_summaries only at standard and full; the analyst rating enters our
        prose only as "Analyst consensus: ..." and headlines never do. Holds throughout 2a (class T)."""
        result = await dm.run(name)
        shown = dm.tiers(result)
        headlines = result["news_summary"]["headlines"]
        rating = result["analyst_coverage"]["rating"].replace("_", " ").title()
        assert headlines and rating == "Strong Buy"

        for tier, payload in shown.items():
            for path, value in leaves(payload["decision_card"]):
                assert not THIRD_PARTY_KEY.search(path), (tier, path)
                assert not (isinstance(value, str) and (rating in value or value in headlines)), (tier, path)
        assert not {"analyst_coverage", "news_summary", "section_summaries"} & shown["decision"].keys()
        for tier in ("standard", "full"):
            assert shown[tier]["analyst_coverage"] and shown[tier]["news_summary"]["headlines"], tier

        prose = [result["executive_summary"], *(v for v in result["section_summaries"].values() if v)]
        assert any(rating in text for text in prose)
        for text in prose:
            assert text.count(rating) == text.count(f"Analyst consensus: {rating}"), text
            assert not any(headline in text for headline in headlines), text

    async def test_vendor_data_leaves_the_card_unchanged(self) -> None:
        """OC-G5: V1 is S3 plus every vendor subtree (targets, estimates, short interest, ownership, options,
        catalyst tags, governance), and its decision_card equals S3's at every tier, so no vendor value
        reaches the card. Holds throughout 2a (class T)."""
        vendor = dm.tiers(await dm.run("V1_vendor_rich_plain_avoid"))
        plain = dm.tiers(await dm.run("S3_plain_avoid"))

        for tier in dm.TIERS:
            assert vendor[tier]["decision_card"] == plain[tier]["decision_card"], tier


class TestSnapshotValuationFields:
    @pytest.mark.parametrize("name", list(BEFORE))
    async def test_snapshot_valuation_fields_follow_the_valuation_assessment(self, name: str) -> None:
        """OC-G6: snapshot valuation_gate == action_zones.valuation_assessment.gate and is_unprofitable ==
        (valuation_assessment.is_unprofitable or None), both equal to the recorded 2.0.0 snapshot.
        Holds throughout: WP2.3 re-sources both from valuation_assessment when policy_action moves (plan §7a)."""
        result = await dm.run(name)
        assessment = result["action_zones"]["valuation_assessment"]
        snapshot = result["watchlist_snapshot"]
        recorded = json.loads((dm.SNAPSHOT_DIR / f"{name}.json").read_text())

        expected = (assessment["gate"], assessment["is_unprofitable"] or None)
        assert (snapshot["valuation_gate"], snapshot["is_unprofitable"]) == expected
        assert (recorded["valuation_gate"], recorded["is_unprofitable"]) == expected

    async def test_an_unprofitable_stock_sets_is_unprofitable_on_both(self) -> None:
        """OC-G6: an unprofitable S3 variant sets is_unprofitable true in the assessment and the snapshot
        (no matrix row is unprofitable, so this keeps the guard above non-vacuous). Holds throughout."""
        result = await dm.run("S3_plain_avoid", fundamentals=UNPROFITABLE)
        assessment = result["action_zones"]["valuation_assessment"]
        snapshot = result["watchlist_snapshot"]

        assert assessment["is_unprofitable"] is True
        assert (snapshot["valuation_gate"], snapshot["is_unprofitable"]) == (assessment["gate"], True)
