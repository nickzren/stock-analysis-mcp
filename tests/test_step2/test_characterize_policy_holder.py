"""WP2.0: policy and holder characterization (annex step2-v2-research/policy_holder.md §10).

Each test records 0609fde behavior. CH tests name the WP that flips them; G tests hold
through 2a. Ported from the annex probes: test_p4_characterize.py (CH1-CH6),
p2_pipeline_cases.py (CH2, CH8), p5_runway_thesis_cap.py (CH7, plus a local analyze twin),
v1_pipeline.py (CH11), v2_core_and_dislocation.py (CH4 F3, CH9) and
v5_candidate_zone_knife.py (CH10). Pipeline cases run through
tests/support/decision_matrix.py, which pins the clock.
"""

import copy
from typing import Any

import pytest

from stock_analysis.tools.analyze.action_zones import apply_dip_gates_to_action_zones
from stock_analysis.tools.analyze.decision_context import _build_thesis_checkpoints
from stock_analysis.tools.analyze.dislocation_framework import classify_dislocation_opportunity
from stock_analysis.tools.analyze.executive_summary import build_policy_action
from stock_analysis.tools.analyze.investor_profile import build_decision_modes
from tests.support import decision_matrix as dm

KNIFE_DIP = {"dip_classification": {"type": "falling_knife"}}
NEAR_HIGH = [{"id": "near_52w_high", "reason": "r"}]


def policy(mid: str, long: str, tilt: str | None, *, zone: str = "hold_neutral", gate: str = "neutral",
           unprofitable: bool = False, dip: dict[str, Any] | None = None,
           quality: str = "poor") -> dict[str, Any]:
    verdict = {"tilt": tilt, "horizon_fit": {"mid_term": mid, "long_term": long},
               "decomposed": {"business_quality": quality}}
    zones = {"current_zone": zone, "valuation_assessment": {"gate": gate, "is_unprofitable": unprofitable}}
    return build_policy_action(verdict=verdict, action_zones=zones, decomposed=verdict["decomposed"],
                               dip_assessment=dip, fundamentals_summary={})


def thesis(result: dict[str, Any]) -> str:
    return result["dislocation_framework"]["thesis_integrity"]["status"]


class TestKnifeOverride:
    @pytest.mark.parametrize(("mid", "tilt", "before"),
                             [("avoid", "bearish", "avoid"), ("ok", None, "insufficient_data"),
                              ("ok", "bearish", "hold_or_reduce")])
    def test_the_knife_leaves_the_mid_code_alone(self, mid: str, tilt: str | None, before: str) -> None:
        """PH-CH1: a falling knife no longer rewrites avoid, insufficient_data or hold_or_reduce, and adds no
        rationale line. G from WP2.2 (plan §1: build_policy_action loses its knife override; the knife is the
        falling_knife gate)."""
        knifed = policy(mid, "avoid", tilt, dip=KNIFE_DIP)

        assert policy(mid, "avoid", tilt)["mid_term"] == knifed["mid_term"] == before
        assert not [line for line in knifed["rationale"] if "falling_knife" in line]

    @pytest.mark.parametrize(("dip_type", "zone", "knife_free", "policy_today"),
                             [(None, "accumulate", "buy", "buy"),
                              ("falling_knife", "hold_neutral", "hold_or_add", "hold_or_add"),
                              ("extended_decline", "accumulate", "buy", "buy")])
    def test_the_zone_cap_lets_the_knife_split_buy_from_hold_or_add(
        self, dip_type: str | None, zone: str, knife_free: str, policy_today: str
    ) -> None:
        """PH-CH10: ok + bullish + accumulate gives buy; the knife caps the zone to hold_neutral, so the
        knife-free candidate is hold_or_add; extended_decline leaves the zone (V5, annex §0.1). Without the knife
        override the policy shows that candidate. G from WP2.2 (knife row flipped; annex §0.1: the knife reaches
        the candidate only through the zone cap)."""
        verdict = {"tilt": "bullish", "horizon_fit": {"mid_term": "ok", "long_term": "ok"}, "decomposed": {}}
        dip = {"dip_classification": {"type": dip_type}}
        zones = apply_dip_gates_to_action_zones(
            {"current_zone": "accumulate", "zone_warnings": [],
             "valuation_assessment": {"gate": "neutral", "is_unprofitable": False}}, dip)

        def mid_code(dip_assessment: dict[str, Any]) -> str:
            action = build_policy_action(verdict=verdict, action_zones=copy.deepcopy(zones), decomposed={},
                                         dip_assessment=dip_assessment, fundamentals_summary={})
            return str(action["mid_term"])

        assert zones["current_zone"] == zone
        assert mid_code({"dip_classification": {"type": None}}) == knife_free
        assert mid_code(dip) == policy_today


class TestZoneAndDislocationCodes:
    @pytest.mark.parametrize(("mid", "code"), [("ok", "hold_or_add"), ("caution", "speculative_small_position")])
    def test_the_reduce_zone_with_a_bullish_tilt_admits_new_money_a7(self, mid: str, code: str) -> None:
        """PH-CH3: A7: reduce zone + bullish gives hold_or_add (ok) or speculative_small_position (caution),
        both mapping to starter; the near_52w_high gate caps that at wait. PD D-4, D-2 (flipped by WP2.2;
        the policy codes stay, §1 "A7 is a new-money cap, not a candidate change")."""
        action = policy(mid, "ok", "bullish", zone="reduce", quality="strong")
        capped = dm.decide_on(action, NEAR_HIGH)["new_money"]

        assert action["mid_term"] == code
        assert dm.decide_on(action)["new_money"]["action"] == "starter"
        assert (capped["action"], capped["decided_by"]) == ("wait", "near_52w_high")

    @pytest.mark.parametrize(("mid", "code", "without_dislocation", "now", "basis"),
                             [("ok", "hold_or_reduce", "hold_or_reduce", "wait", "price_broken_more_than_business"),
                              ("caution", "speculative_small_position", "wait_for_entry", "starter",
                               "candidate_admits_new_money"),
                              ("avoid", "avoid", "avoid", "avoid", "price_broken_more_than_business")])
    def test_dislocation_with_a_bearish_tilt_is_non_monotone_in_mid_fit(
        self, mid: str, code: str, without_dislocation: str, now: str, basis: str
    ) -> None:
        """PH-CH9: dislocation + bearish: ok gives hold_or_reduce, caution speculative_small_position,
        avoid avoid; only the caution branch reads the dislocation flag (V2). new_money maps them (wait,
        starter, avoid) and if_held holds for all three. PD D-2, D-1b (flipped by WP2.2; the policy codes stay,
        §0 #16; PH-RF13 owns the if_held rows)."""
        verdict = {"tilt": "bearish", "decomposed": {"business_quality": "strong"},
                   "horizon_fit": {"mid_term": mid, "long_term": "ok"}}
        zones = {"current_zone": "accumulate",
                 "valuation_assessment": {"gate": "attractive", "is_unprofitable": False}}
        fundamentals = {"growth": {"revenue_yoy": 0.12},
                        "profitability": {"net_margin": 0.2, "fcf_positive": True},
                        "health": {"net_cash_positive": True, "debt_to_equity": 0.5},
                        "burn_metrics": {"status": "not_applicable"}}
        deep_dip = {"dip_depth": {"from_52w_high": -0.45, "severity": "deep"}}

        def policy_with(dip: dict[str, Any] | None) -> dict[str, Any]:
            return build_policy_action(verdict=verdict, action_zones=zones, decomposed=verdict["decomposed"],
                                       dip_assessment=dip, fundamentals_summary=fundamentals)

        signals = classify_dislocation_opportunity(verdict, zones, fundamentals, deep_dip)
        dislocated = policy_with(deep_dip)

        assert signals["is_dislocation_candidate"] is True
        assert dislocated["mid_term"] == code
        assert policy_with(None)["mid_term"] == without_dislocation
        decided = dm.decide_on(dislocated, dislocation=True, tilt="bearish")
        assert (decided["new_money"]["action"], decided["if_held"]["action"], decided["if_held"]["basis"]) == (
            now, "hold", basis)


def modes(long_fit: str, policy_action: dict[str, Any], tilt: str | None, gate: str) -> dict[str, Any]:
    verdict = {"tilt": tilt, "horizon_fit": {"mid_term": "ok", "long_term": long_fit, "long_term_gates": []},
               "decomposed": {"business_quality": "strong", "risk": "medium"}}
    return build_decision_modes(summary={"current_price": 100.0}, verdict=verdict, policy_action=policy_action,
                                action_zones={"valuation_assessment": {"gate": gate}}, dip_assessment=None,
                                decision_context={}, news_summary={}, fundamentals_summary={}, events_summary={})


class TestCoreMode:
    def test_core_reads_long_term_fit_not_the_long_term_policy_c11_a2(self) -> None:
        """PH-CH4: core is build_position for long ok with a headwind (C11) or a null tilt and neutral valuation (A2).
        Flipped by WP2.2 (PD D-3, PH-RF18): core = most_restrictive(CORE[long fit], long_term.new_money)."""
        headwind = modes("ok", {"mid_term": "hold", "long_term": "hold_size_conservatively"}, "neutral", "headwind")
        missing_tilt = modes("ok", {"mid_term": "insufficient_data", "long_term": "insufficient_data"}, None,
                             "neutral")

        assert headwind["core"]["action"] == "build_position"
        assert missing_tilt["core"]["action"] == "build_position"

    @pytest.mark.parametrize(("name", "long_fit", "long_policy", "gates", "core"), [
        ("C1_headwind_hold_or_add", "ok", "hold_size_conservatively", [], "build_position"),
        ("F3_quality_neutral_valuation_poor_chart", "ok", "hold", [], "build_position"),
        ("D_quality_attractive_poor_chart", "ok", "accumulate", [], "build_position"),
        ("D2_quality_poor_chart_earnings_3d", "ok", "accumulate", ["earnings_blackout"], "build_position"),
        ("C_quality_knife_dislocation", "ok", "accumulate", ["falling_knife"], "build_position"),
        ("E2_mixed_fcf_bullish_chart", "caution", "small_position_with_stops", [], "not_core_yet"),
    ])
    async def test_core_follows_long_term_fit_alone_through_the_pipeline(
        self, name: str, long_fit: str, long_policy: str, gates: list[str], core: str
    ) -> None:
        """PH-CH4: core reads long_term_fit only: not the long-term policy (C1, F3), the mid decision
        (D), fired gates (D2, C) or the long candidate's code (E2) (annex §7 'core (today)').
        Flipped by WP2.2 (PD D-3, D-5b; PH-RF15-RF18): core follows long_term.new_money under its own cap."""
        result = await dm.run(name)

        assert result["verdict"]["horizon_fit"]["long_term"] == long_fit
        assert result["policy_action"]["long_term"] == long_policy
        assert dm.gate_ids(result) == gates
        assert result["decision_modes"]["core"]["action"] == core


class TestLongTermPolicy:
    def test_long_ok_with_a_bearish_tilt_is_hold_not_reduce(self) -> None:
        """PH-CH5: long ok + bearish gives hold.
        Guard (G) through 2a: the long-term policy rules are unchanged (plan §1); the long-term holder read is Step 3."""
        assert policy("ok", "ok", "bearish", quality="strong")["long_term"] == "hold"

    @pytest.mark.parametrize("tilt", ["bullish", "neutral", "bearish", None])
    def test_an_unprofitable_caution_ignores_tilt(self, tilt: str | None) -> None:
        """PH-CH5: long caution + unprofitable gives speculative_only whatever the tilt.
        Guard (G) through 2a: the long-term policy rules are unchanged (plan §1); the long-term holder read is Step 3."""
        assert policy("ok", "caution", tilt, unprofitable=True)["long_term"] == "speculative_only"


# fundamentals_snapshot payloads for the analyze twin of p5: cash 1e8 against a quarterly FCF burn
# of 5e7 is a 2.0-quarter runway; without the cash figure the runway is unavailable.
BURNER_SNAPSHOT = dm.BURNER
UNKNOWN_RUNWAY_SNAPSHOT = {**dm.BURNER, "financial_health": {
    k: v for k, v in dm.BURNER["financial_health"].items() if k != "cash_and_st_investments"}}


class TestThesisSurfaces:
    def test_thesis_stop_triggers_are_unevaluated_templates(self) -> None:
        """PH-CH6: thesis_stop_triggers are static templates: runway_critical is true in the data
        (2 quarters) yet no trigger carries a fired key.
        Guard (G) through 2a: the thesis_broken cap reads the dislocation statuses, never these templates."""
        out = _build_thesis_checkpoints(is_unprofitable=True, net_margin=-0.3, fcf=-1e8, cash_runway_quarters=2.0,
                                        revenue_yoy=-0.4, valuation_gate="neutral", risk_label="high",
                                        business_quality="unprofitable")
        triggers = {t["trigger"]: t for t in out["thesis_stop_triggers"]}

        assert triggers["runway_critical"]["condition"] == "cash_runway < 4 quarters"
        assert all(set(t) == {"trigger", "condition", "action", "rationale"} for t in triggers.values())

    @pytest.mark.parametrize(("snapshot", "runway", "balance", "thesis_status", "gates", "now"), [
        (BURNER_SNAPSHOT, 2.0, "unsafe", "broken", ["thesis_broken"], "avoid"),
        (UNKNOWN_RUNWAY_SNAPSHOT, None, "safe", "watch", ["missing_runway"], "wait_for_data"),
    ], ids=["runway_2q", "runway_unknown"])
    async def test_a_known_short_runway_is_blocked_like_an_unknown_one(
        self, snapshot: dict[str, Any], runway: float | None, balance: str, thesis_status: str,
        gates: list[str], now: str
    ) -> None:
        """PH-CH7: through analyze (weak bullish chart), the 2-quarter burner (thesis broken) is blocked by
        thesis_broken (new_money avoid) and its unknown-runway twin by missing_runway: the p5 inversion is closed.
        PD D-1c (G from WP2.2; flipped from "no gate, starter"; PH-RF12 owns the burner)."""
        result = await dm.run(dm.Scenario("PH CH7 burner (p5)", dm.WEAK, snapshot, dm.MODERATE))
        dislocation = result["dislocation_framework"]

        assert result["fundamentals_summary"]["burn_metrics"]["cash_runway_quarters"] == runway
        assert (dislocation["business_integrity"]["status"], dislocation["balance_sheet_safety"]["status"]) == (
            "mixed", balance)
        assert thesis(result) == thesis_status
        assert (result["policy_action"]["mid_term"], result["policy_action"]["long_term"]) == (
            "speculative_small_position", "speculative_only")
        assert dm.gate_ids(result) == gates
        assert result["decision_card"]["action_now"] == now

    async def test_case_e_shows_a_broken_thesis_next_to_hold_or_add(self) -> None:
        """PH-CH8: case E (poor business, bullish chart): thesis broken, core avoid and long-term
        avoid_core_position; the pre-gate mid-term policy stays hold_or_add next to new_money avoid [thesis_broken].
        Action clause flipped by WP2.2 (PD D-1c, PH-RF12); flipped again by WP2.3 (policy_action moves to
        diagnostics.pre_gate_policy, D-9)."""
        result = await dm.run("E_poor_business_bullish_chart")
        dislocation = result["dislocation_framework"]

        assert result["verdict"]["decomposed"]["business_quality"] == "poor"
        assert (thesis(result), dislocation["mismatch_verdict"]["status"]) == (
            "broken", "business_broken_more_than_price")
        assert result["policy_action"]["long_term"] == "avoid_core_position"
        assert result["decision_modes"]["core"]["action"] == "avoid"
        assert (result["policy_action"]["mid_term"], result["decision_card"]["action_now"]) == (
            "hold_or_add", "avoid")
        assert dm.gate_ids(result) == ["thesis_broken"]


class TestGateCombination:
    async def test_case_g_unsupported_currency_with_a_knife(self) -> None:
        """PH-CH11: case G (EUR quality crash + knife): action_now avoid, long-term policy watch
        (the support cap on accumulate), core watch, gates [falling_knife, unsupported_currency].
        Flipped by WP2.2 (PD D-5, D-5b; PH-RF21): gates reach the long term, so long_term.new_money and core avoid."""
        result = await dm.run("G_unsupported_currency_knife")
        policy_action = result["policy_action"]

        assert result["decision_card"]["action_now"] == "avoid"
        assert dm.gate_ids(result) == ["falling_knife", "unsupported_currency"]
        assert (policy_action["mid_term"], policy_action["long_term"]) == ("avoid", "watch")
        assert policy_action["support_gates"] == ["unsupported_currency"]
        assert "long_term=ok, valuation=attractive, zone=hold_neutral" in policy_action["rationale"]
        assert result["decision_modes"]["core"]["action"] == "watch"
