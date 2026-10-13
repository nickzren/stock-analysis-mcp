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
from stock_analysis.tools.analyze.decision_card import _compute_hard_gates, _resolve_action_now
from stock_analysis.tools.analyze.decision_context import _build_thesis_checkpoints
from stock_analysis.tools.analyze.dislocation_framework import classify_dislocation_opportunity
from stock_analysis.tools.analyze.executive_summary import build_policy_action
from stock_analysis.tools.analyze.investor_profile import build_decision_modes
from tests.support import decision_matrix as dm

KNIFE_DIP = {"dip_classification": {"type": "falling_knife"}}
NO_GATES: dict[str, Any] = {"any_blocking": False, "blocking": []}
KNIFE_GATE = {"any_blocking": True, "blocking": [{"id": "falling_knife", "reason": "r"}]}


def policy(mid: str, long: str, tilt: str | None, *, zone: str = "hold_neutral", gate: str = "neutral",
           unprofitable: bool = False, dip: dict[str, Any] | None = None,
           quality: str = "poor") -> dict[str, Any]:
    verdict = {"tilt": tilt, "horizon_fit": {"mid_term": mid, "long_term": long},
               "decomposed": {"business_quality": quality}}
    zones = {"current_zone": zone, "valuation_assessment": {"gate": gate, "is_unprofitable": unprofitable}}
    return build_policy_action(verdict=verdict, action_zones=zones, decomposed=verdict["decomposed"],
                               dip_assessment=dip, fundamentals_summary={})


def action_now(policy_action: dict[str, Any], hard_gates: dict[str, Any] = NO_GATES) -> str:
    return _resolve_action_now(policy_action, hard_gates)[0]


def thesis(result: dict[str, Any]) -> str:
    return result["dislocation_framework"]["thesis_integrity"]["status"]


class TestKnifeOverride:
    @pytest.mark.parametrize(("mid", "tilt", "before"),
                             [("avoid", "bearish", "avoid"), ("ok", None, "insufficient_data"),
                              ("ok", "bearish", "hold_or_reduce")])
    def test_the_knife_rewrites_non_dislocation_mid_codes_to_wait_for_entry(
        self, mid: str, tilt: str | None, before: str
    ) -> None:
        """PH-CH1: a falling knife rewrites avoid, insufficient_data and hold_or_reduce to wait_for_entry.
        Flipped by WP2.2: build_policy_action loses its knife override (plan §1; TestR95FallingKnife migrates)."""
        knifed = policy(mid, "avoid", tilt, dip=KNIFE_DIP)

        assert policy(mid, "avoid", tilt)["mid_term"] == before
        assert knifed["mid_term"] == "wait_for_entry"
        assert "dip_type=falling_knife" in knifed["rationale"]

    def test_the_knife_loosens_action_now_from_avoid_to_wait(self) -> None:
        """PH-CH2: the knife turns action_now avoid into wait; its gate never overrides wait.
        Flipped by WP2.2 (PH-RF2): adding the knife never loosens new money."""
        plain = policy("avoid", "avoid", "bearish")
        knifed = policy("avoid", "avoid", "bearish", dip=KNIFE_DIP)

        assert action_now(plain) == "avoid"
        assert action_now(knifed, KNIFE_GATE) == "wait"

    async def test_codex_probe_a_to_b_loosens_action_now_while_the_thesis_stays_broken(self) -> None:
        """PH-CH2: pipeline A (broken + bearish) gives action_now avoid; B (A + knife) gives wait.
        Flipped by WP2.2 (PD D-1c, PH-RF3): A and B both give new_money avoid and if_held review_exit."""
        a = await dm.run("A_broken_bearish")
        b = await dm.run("B_broken_bearish_knife")

        assert (a["policy_action"]["mid_term"], a["decision_card"]["action_now"], dm.gate_ids(a)) == (
            "avoid", "avoid", [])
        assert (b["policy_action"]["mid_term"], b["decision_card"]["action_now"], dm.gate_ids(b)) == (
            "wait_for_entry", "wait", ["falling_knife"])
        assert "dip_type=falling_knife" in b["policy_action"]["rationale"]
        assert thesis(a) == thesis(b) == "broken"

    @pytest.mark.parametrize(("dip_type", "zone", "knife_free", "policy_today"),
                             [(None, "accumulate", "buy", "buy"),
                              ("falling_knife", "hold_neutral", "hold_or_add", "wait_for_entry"),
                              ("extended_decline", "accumulate", "buy", "buy")])
    def test_the_zone_cap_lets_the_knife_split_buy_from_hold_or_add(
        self, dip_type: str | None, zone: str, knife_free: str, policy_today: str
    ) -> None:
        """PH-CH10: ok + bullish + accumulate gives buy; the knife caps the zone to hold_neutral, so the
        knife-free candidate is hold_or_add; extended_decline leaves the zone (V5, annex §0.1).
        Flipped by WP2.2 (knife row only): without the knife override the policy shows that candidate."""
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
    @pytest.mark.parametrize(("mid", "code", "now"), [("ok", "hold_or_add", "hold_or_add"),
                                                      ("caution", "speculative_small_position", "starter")])
    def test_the_reduce_zone_with_a_bullish_tilt_admits_new_money_a7(self, mid: str, code: str,
                                                                     now: str) -> None:
        """PH-CH3: A7: reduce zone + bullish gives hold_or_add (ok) or speculative_small_position
        (caution), both new-money action_now codes with no gate.
        Flipped by WP2.2 (PD D-4, PH-RF11): the near_52w_high cap makes new_money wait; the policy codes stay."""
        action = policy(mid, "ok", "bullish", zone="reduce", quality="strong")

        assert action["mid_term"] == code
        assert action_now(action) == now

    @pytest.mark.parametrize(("mid", "code", "without_dislocation", "now"),
                             [("ok", "hold_or_reduce", "hold_or_reduce", "hold_or_reduce"),
                              ("caution", "speculative_small_position", "wait_for_entry", "starter"),
                              ("avoid", "avoid", "avoid", "avoid")])
    def test_dislocation_with_a_bearish_tilt_is_non_monotone_in_mid_fit(
        self, mid: str, code: str, without_dislocation: str, now: str
    ) -> None:
        """PH-CH9: dislocation + bearish: ok gives hold_or_reduce, caution speculative_small_position,
        avoid avoid; only the caution branch reads the dislocation flag (V2).
        Flipped by WP2.2 (ok row's action_now only; PD D-2): as the new_money alias it maps hold_or_reduce to
        wait. The policy codes stay (§0 #16); PH-RF13 (PD D-1b) adds if_held hold for all three."""
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
        assert action_now(dislocated) == now


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


BURNER_VERDICT = {"tilt": "bullish",
                  "horizon_fit": {"mid_term": "caution", "long_term": "caution",
                                  "long_term_gates": ["unprofitable", "low_runway_confidence"]},
                  "decomposed": {"business_quality": "unprofitable"}}
BURNER_ZONES = {"current_zone": "hold_neutral", "valuation_assessment": {"gate": "neutral", "is_unprofitable": True}}


def burner_fundamentals(burn: dict[str, Any]) -> dict[str, Any]:
    return {"growth": {"revenue_yoy": 0.3}, "profitability": {"net_margin": -0.2, "fcf_positive": False},
            "health": {"net_cash_positive": True, "debt_to_equity": 0.1},
            "valuation": {"valuation_note": "pe_not_meaningful"}, "burn_metrics": burn}


# fundamentals_snapshot payloads for the analyze twin of p5: cash 1e8 against a quarterly FCF burn
# of 5e7 is a 2.0-quarter runway; without the cash figure the runway is unavailable.
BURNER_HEALTH = {"debt_to_equity": 0.1, "rules": {"net_cash_positive": {"triggered": True},
                                                  "low_debt": {"triggered": True}}}
BURNER_SNAPSHOT = {"valuation": {"pe_trailing": None, "trailing_eps": -1.0, "ps_trailing": 4.0},
                   "growth": {"revenue_yoy": 0.3, "rules": {"high_growth": {"triggered": True}}},
                   "profitability": {"net_margin": -0.2},
                   "cash_flow": {"free_cash_flow_ttm": -2e8, "operating_cf_ttm": -1.6e8,
                                 "rules": {"positive_fcf": {"triggered": False}}},
                   "financial_health": {**BURNER_HEALTH, "cash_and_st_investments": 1e8}}
UNKNOWN_RUNWAY_SNAPSHOT = {**BURNER_SNAPSHOT, "financial_health": BURNER_HEALTH}


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

    @pytest.mark.parametrize(("burn", "balance", "gates", "now"), [
        ({"status": "available", "cash_runway_quarters": 2.0}, "unsafe", [], "starter"),
        ({"status": "unavailable"}, "safe", ["missing_runway"], "wait_for_data"),
    ], ids=["runway_2q", "runway_unknown"])
    def test_a_known_short_runway_has_no_gate_but_an_unknown_one_does(
        self, burn: dict[str, Any], balance: str, gates: list[str], now: str
    ) -> None:
        """PH-CH7: a 2-quarter burner (balance unsafe) gets no gate and starter codes (mid
        speculative_small_position, long speculative_only); its unknown-runway twin is blocked by
        missing_runway (p5).
        Flipped by WP2.2 (PD D-1c, PH-RF12) if the thesis_broken cap lands in these helpers; else retire it
        (the analyze twin below flips either way)."""
        fundamentals = burner_fundamentals(burn)
        signals = classify_dislocation_opportunity(verdict=BURNER_VERDICT, action_zones=BURNER_ZONES,
                                                   fundamentals_summary=fundamentals, dip_assessment=None)
        action = build_policy_action(verdict=BURNER_VERDICT, action_zones=BURNER_ZONES,
                                     decomposed=BURNER_VERDICT["decomposed"], dip_assessment=None,
                                     fundamentals_summary=fundamentals)
        hard_gates = _compute_hard_gates(events_data={}, data_quality={}, dip_assessment=None,
                                         fundamentals_summary=fundamentals,
                                         risk_data={"liquidity": {"avg_dollar_volume": 5e8}})

        assert (signals["business_status"], signals["balance_status"]) == ("mixed", balance)
        assert (action["mid_term"], action["long_term"]) == ("speculative_small_position", "speculative_only")
        assert [g["id"] for g in hard_gates["blocking"]] == gates
        assert action_now(action, hard_gates) == now

    @pytest.mark.parametrize(("snapshot", "runway", "balance", "thesis_status", "gates", "now"), [
        (BURNER_SNAPSHOT, 2.0, "unsafe", "broken", [], "starter"),
        (UNKNOWN_RUNWAY_SNAPSHOT, None, "safe", "watch", ["missing_runway"], "wait_for_data"),
    ], ids=["runway_2q", "runway_unknown"])
    async def test_the_runway_inversion_holds_through_analyze(
        self, snapshot: dict[str, Any], runway: float | None, balance: str, thesis_status: str,
        gates: list[str], now: str
    ) -> None:
        """PH-CH7: through analyze (weak bullish chart), the 2-quarter burner shows thesis broken and
        action_now starter with no gate; its unknown-runway twin is blocked by missing_runway.
        Flipped by WP2.2 (PD D-1c, PH-RF12): the thesis_broken cap makes the 2-quarter burner's action_now avoid."""
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
        avoid_core_position, yet action_now hold_or_add with no gate.
        Flipped by WP2.2 (PD D-1c, PH-RF12): new_money avoid [thesis_broken] and if_held review_exit."""
        result = await dm.run("E_poor_business_bullish_chart")
        dislocation = result["dislocation_framework"]

        assert result["verdict"]["decomposed"]["business_quality"] == "poor"
        assert (thesis(result), dislocation["mismatch_verdict"]["status"]) == (
            "broken", "business_broken_more_than_price")
        assert result["policy_action"]["long_term"] == "avoid_core_position"
        assert result["decision_modes"]["core"]["action"] == "avoid"
        assert (result["policy_action"]["mid_term"], result["decision_card"]["action_now"]) == (
            "hold_or_add", "hold_or_add")
        assert dm.gate_ids(result) == []


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
