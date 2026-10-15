"""WP2.3 derived surfaces (plan §3 [D-3], §7 [D-9]): the executive summary's horizon segments, the decision
modes, the dislocation action and check, the dip caps and price_zone, all derived from decide()'s decision and
never feeding it. Annex: output_contract.md §4, §6, §7 and policy_holder.md §8, §10, "Corrected proposal", as
amended by plan §0 (#8, #9, #10, #11, #21, #23)."""

from itertools import product
from typing import Any

import pytest

from stock_analysis.tools.analyze import investor_profile
from stock_analysis.tools.analyze.action_zones import (
    PRICE_ZONES,
    _classify_current_zone,
    apply_dip_gates_to_action_zones,
    build_action_zones,
)
from stock_analysis.tools.analyze.decision import (
    CORE,
    ENTRY_CODES,
    FULL_ENTRY_CODES,
    NEW_MONEY_ACTIONS,
    entry_allowed,
    most_restrictive,
)
from stock_analysis.tools.analyze.dip_assessment import (
    align_dip_assessment_with_action,
    cap_dip_assessment,
)
from stock_analysis.tools.analyze.dislocation_framework import build_dislocation_framework
from tests.support import decision_matrix as dm

ROWS = list(dm.SCENARIOS)
PRICE_ZONE_NAMES = {"near_52w_low", "near_52w_high", "below_sma200", "above_sma50", "at_or_below_sma50",
                    "undetermined"}


def ids(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blockers]


def tight(action: str) -> int:
    return NEW_MONEY_ACTIONS.index(action)


# --- (b) decision modes (PH-RF17 balanced, PH-RF19, SC-CH8, SC-CH12) ------------------------------------------


@pytest.mark.parametrize("name", ROWS)
async def test_modes_are_capped_at_their_horizon(name: str) -> None:
    """PH-RF19: every mode speaks the new-money vocabulary; balanced == new_money and its long_term_view ==
    long_term.new_money; speculative is never looser than new_money; core == most_restrictive(CORE[fit],
    long_term.new_money). A tightened mode names its cap in capped_by; an untouched one has none. PD D-3."""
    result = await dm.run(name)
    card = result["decision_card"]
    modes = result["decision_modes"]
    new_money, long = card["new_money"]["action"], card["long_term"]["new_money"]["action"]
    fit = result["verdict"]["horizon_fit"]["long_term"]

    assert {m["action"] for m in modes.values()} <= set(NEW_MONEY_ACTIONS)
    assert (modes["balanced"]["action"], modes["balanced"]["long_term_view"]) == (new_money, long)
    assert "capped_by" not in modes["balanced"]
    speculative = modes["speculative"]
    assert tight(speculative["action"]) >= tight(new_money)
    if "capped_by" in speculative:
        assert (speculative["action"], speculative["capped_by"]) == (new_money, f"new_money_{new_money}")
    else:
        assert speculative["action"] in {"starter", "wait"}
    core = modes["core"]
    assert core["action"] == most_restrictive((CORE[fit], long))
    assert ("capped_by" in core) is (core["action"] != CORE[fit])
    if "capped_by" in core:
        assert core["capped_by"] == f"long_term_{long}"


def test_speculative_maps_its_own_code_then_caps_it() -> None:
    """PH-RF19 (unit): speculative's own codes map starter_then_add/starter_position_only -> starter and
    wait/watch -> wait (a strict lookup); a mode names its cap in capped_by only when the cap tightened its own
    code. PD D-3."""
    assert investor_profile._SPECULATIVE_ACTIONS == {
        "starter_then_add": "starter", "starter_position_only": "starter", "wait": "wait", "watch": "wait"}
    assert investor_profile._capped("starter", "avoid", "new_money") == {"action": "avoid",
                                                                         "capped_by": "new_money_avoid"}
    assert investor_profile._capped("wait", "wait", "long_term") == {"action": "wait"}


@pytest.mark.parametrize(("name", "long_term"), [("E2_mixed_fcf_bullish_chart", "starter"),
                                                 ("C1_headwind_hold_or_add", "wait")])
async def test_balanced_follows_new_money_in_direction_2(name: str, long_term: str) -> None:
    """PH-RF17 (balanced clause): E2 and C1 give balanced starter == new_money with long_term_view == the
    long-term decision (E2 starter, C1 wait) and core wait. PD D-3."""
    result = await dm.run(name)
    modes = result["decision_modes"]

    assert (result["decision_card"]["new_money"]["action"], modes["balanced"]["action"],
            modes["balanced"]["long_term_view"], modes["core"]["action"]) == ("starter", "starter", long_term, "wait")


# --- (c) dislocation (S-C10, OC-CH3, IS-CH2) -----------------------------------------------------------------


@pytest.mark.parametrize("name", ROWS)
async def test_the_dislocation_action_follows_new_money(name: str) -> None:
    """OC-RF4/P6: dislocation can_start_now == entry_allowed(new_money), buy_now is new_money's action under buy
    or starter and null otherwise, the mode copies equal the capped modes with their capped_by, and its buy and
    add conditions mirror the card's. PD D-9."""
    result = await dm.run(name)
    card = result["decision_card"]
    action = result["dislocation_framework"]["action"]
    new_money = card["new_money"]["action"]

    assert action["can_start_now"] is entry_allowed(new_money)
    if result["dislocation_framework"].get("status") == "not_applicable":
        return
    modes = result["decision_modes"]
    assert action["buy_now"] == (new_money if entry_allowed(new_money) else None)
    assert {m: action[m] for m in modes} == {m: mode["action"] for m, mode in modes.items()}
    assert action["capped_by"] == {m: mode["capped_by"] for m, mode in modes.items() if "capped_by" in mode}
    assert (action["buy_only_if"], action["add_only_if"]) == (card["conditions"]["buy_only_if"],
                                                              card["conditions"]["add_only_if"])


def test_can_start_now_reads_new_money_not_the_modes() -> None:
    """OC-RF4/P6 (unit): a buy whose speculative mode stays at wait still can start (buy_now buy); a starter
    offers starter; a wait offers nothing, whatever the modes say. PD D-9."""
    def framework(action: str, modes: dict[str, str]) -> dict[str, Any]:
        built: dict[str, Any] = build_dislocation_framework(
            verdict={}, action_zones={}, fundamentals_summary={}, dip_assessment=None, decision_context={},
            decision_modes={m: {"action": a} for m, a in modes.items()},
            decision={"new_money": {"action": action, "blocked_by": []}},
            conditions={"buy_only_if": [], "add_only_if": []})["action"]
        return built

    assert [framework("buy", {"balanced": "buy", "speculative": "wait"})[k] for k in ("can_start_now", "buy_now")] \
        == [True, "buy"]
    assert framework("starter", {"balanced": "starter"})["buy_now"] == "starter"
    assert [framework("wait", {"balanced": "wait", "speculative": "starter"})[k]
            for k in ("can_start_now", "buy_now")] == [False, None]


@pytest.mark.parametrize("name", ROWS)
async def test_the_standard_dislocation_check_carries_the_decision(name: str) -> None:
    """SC-CH10 (flipped, S-C10): the standard tier's dislocation_check carries decision = {new_money, blocked_by
    ids} from the card, for applicable and not-applicable reads alike. PD D-9."""
    result = await dm.run(name)
    card = result["decision_card"]
    check = dm.tiers(result)["standard"]["dislocation_check"]

    assert check["decision"] == {"new_money": card["new_money"]["action"],
                                 "blocked_by": ids(card["new_money"]["blocked_by"])}


# --- (d) dip caps -------------------------------------------------------------------------------------------

DIP_RECS = ["buy_the_dip", "cautious_accumulation", "do_not_catch_falling_knife", "small_speculative_position",
            "starter_only_wait_for_stabilization", "strong_buy_the_dip", "wait_for_better_setup"]
DIP_SIGNALS = ["accumulate", "add_position", "consider_small_position", "starter_position", "watch_for_reversal"]


def _dip(recommendation: str) -> dict[str, Any]:
    return {"assessment": {"dip_quality": "good", "recommendation": recommendation, "rationale": "r"},
            "entry_timing": {"signals": [{"signal": f"s{i}", "action": a, "rationale": "r"}
                                         for i, a in enumerate(DIP_SIGNALS)], "wait_for": []}}


@pytest.mark.parametrize(("action", "recommendation"), list(product(NEW_MONEY_ACTIONS, DIP_RECS)))
def test_the_dip_caps(action: str, recommendation: str) -> None:
    """Dip caps (plan §7 dip row): buy leaves the dip alone; starter caps the full-entry recommendations to
    starter_only and accumulate/add_position to starter_position; a non-entry action caps every entry code to
    watch; each capped field carries capped_by new_money_<action>, a capped recommendation states the cap, and a
    support gate in blocked_by is named in support_gates. PD D-9 (starter) / RF (non-entry)."""
    support = [{"id": "unsupported_currency", "class": "support"}] if action == "watch" else []
    capped = _dip(recommendation)
    cap_dip_assessment(capped, {"action": action, "blocked_by": support})
    assessment, signals = capped["assessment"], capped["entry_timing"]["signals"]
    by = f"new_money_{action}"

    if action == "buy":
        assert capped == _dip(recommendation)
        return
    if action == "starter":
        expected = dict.fromkeys(("accumulate", "add_position"), "starter_position")
        rec = "starter_only" if recommendation in ("strong_buy_the_dip", "buy_the_dip", "cautious_accumulation") \
            else recommendation
        rationale = "No full-size entry under the new-money decision."
    else:
        expected = dict.fromkeys(("accumulate", "add_position", "consider_small_position", "starter_position"), "watch")
        rec = "watch" if recommendation not in ("do_not_catch_falling_knife", "wait_for_better_setup") \
            else recommendation
        rationale = f"No new-money entry: new_money is {action}."
    for original, signal in zip(DIP_SIGNALS, signals, strict=True):
        assert signal["action"] == expected.get(original, original)
        assert signal.get("capped_by") == (by if original in expected else None)
    assert assessment["recommendation"] == rec
    assert assessment.get("capped_by") == (by if rec != recommendation else None)
    assert assessment["rationale"] == (rationale if rec != recommendation else "r")
    assert assessment.get("support_gates") == (["unsupported_currency"] if support and rec != recommendation
                                               else None)


@pytest.mark.parametrize("name", ROWS)
async def test_no_dip_entry_code_outside_its_predicate(name: str) -> None:
    """OC-RF4 (dip clause): a dip recommendation or signal in ENTRY_CODES needs entry_allowed, a full-entry one
    needs new_money buy. PD D-9."""
    result = await dm.run(name)
    new_money = result["decision_card"]["new_money"]["action"]
    dip = result["dip_assessment"] or {}
    codes = [(dip.get("assessment") or {}).get("recommendation"),
             *(s["action"] for s in (dip.get("entry_timing") or {}).get("signals") or [])]
    for code in codes:
        assert code not in ENTRY_CODES or entry_allowed(new_money), code
        assert code not in FULL_ENTRY_CODES or new_money == "buy", code


@pytest.mark.parametrize("name", ROWS)
async def test_the_dip_override_follows_a_dislocation_starter(name: str) -> None:
    """OC-M9 (call site, annex §7): analyze reframes the dip as entry-timing only exactly when the mismatch is
    price_broken_more_than_business and new_money is starter; S6 and C (knife, avoid) keep
    do_not_catch_falling_knife. PD D-9."""
    result = await dm.run(name)
    assessment = (result["dip_assessment"] or {}).get("assessment") or {}
    mismatch = (result["dislocation_framework"].get("mismatch_verdict") or {}).get("status")
    starter = result["decision_card"]["new_money"]["action"] == "starter"

    assert ("timing_only" in assessment) is (mismatch == "price_broken_more_than_business" and starter)


def test_the_starter_override_needs_a_starter() -> None:
    """OC-M9: align_dip_assessment_with_action takes starter_only (new_money == starter), not can_start_now: a
    dislocation starter reframes the falling-knife dip; anything else leaves it alone. RF."""
    def dip() -> dict[str, Any]:
        return {"assessment": {"dip_quality": "avoid", "recommendation": "do_not_catch_falling_knife",
                               "rationale": "Trend is broken - wait for stabilization"}}

    started = align_dip_assessment_with_action(dip(), "price_broken_more_than_business", starter_only=True,
                                               add_only_if=["x"])
    assert started is not None and started["assessment"]["recommendation"] == "starter_only_wait_for_stabilization"
    assert started["assessment"]["add_only_if"] == ["x"]
    assert align_dip_assessment_with_action(dip(), "price_broken_more_than_business", starter_only=False) == dip()


# --- (e) price_zone --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROWS)
async def test_price_zone_is_the_uncapped_classification(name: str) -> None:
    """OC-RF6 (non-snapshot): action_zones.price_zone is in the closed set and equals the uncapped zone
    classification of the run's own price, levels and SMAs; no zone field remains in action_zones or the card at
    any tier. PD D-9."""
    result = await dm.run(name)
    zones = result["action_zones"]
    technicals = dm.SCENARIOS[name].technicals or {}
    averages = technicals.get("moving_averages") or {}
    price = result["summary"]["current_price"]
    expected = None if price is None else PRICE_ZONES[_classify_current_zone(
        price, zones["levels"], averages.get("sma_50"), averages.get("sma_200"))]

    assert zones["price_zone"] == expected
    assert zones["price_zone"] in {*PRICE_ZONE_NAMES, None}
    for tier, payload in dm.tiers(result).items():
        assert "current_zone" not in (payload.get("action_zones") or {}), tier
        entry = payload["decision_card"]["entry"]
        assert entry is None or set(entry) == {"current_price", "buy_zone_low", "buy_zone_high"}, tier
    assert dm.tiers(result)["standard"]["action_zones"]["price_zone"] == expected


AT_52W_LOW = {"current_price": 70.0, "moving_averages": {"sma_50": 80.0, "sma_200": 90.0},
              "price_position": {"week_52_low": 70.0, "week_52_high": 120.0}}
CALM_VOL = {"volatility": {"annualized": 0.22}, "atr": {"value": 1.0, "as_pct_of_price": 0.014}}


@pytest.mark.parametrize(("risk", "fund", "dip_type", "regime", "warning"), [
    ({"classification": "extreme"}, {"valuation": {"pe_trailing": 9.0}}, None, None, "zone_capped_due_to_extreme_risk"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 60.0}}, None, None, "zone_downgraded_valuation_headwind"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 9.0}}, "falling_knife", None, "zone_capped_falling_knife"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 9.0}}, "extended_decline", None,
     "zone_capped_extended_decline"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 9.0}}, None, {"classification": "extreme"},
     "zone_capped_extreme_risk"),
])
def test_the_five_zone_caps_leave_price_zone_uncapped(
        risk: dict[str, Any], fund: dict[str, Any], dip_type: str | None, regime: dict[str, Any] | None,
        warning: str) -> None:
    """OC-RF6 (unit) and OC-CH7 (flipped): each of the five caps still emits its warning and still caps the
    candidate's legacy zone, while price_zone stays near_52w_low, the uncapped classification. PD D-9."""
    zones = build_action_zones(current_price=70.0, tech_data=AT_52W_LOW, risk_data=CALM_VOL, fund_data=fund,
                               risk_regime=risk)
    zones = apply_dip_gates_to_action_zones(zones, {"dip_classification": {"type": dip_type}}, regime)

    assert warning in zones["zone_warnings"]
    assert zones["current_zone"] != "strong_buy"
    assert zones["price_zone"] == "near_52w_low"


RANGE_70_120 = {"week_52_low": 70.0, "week_52_high": 120.0}
SMAS = {"sma_50": 100.0, "sma_200": 90.0}


@pytest.mark.parametrize(("price", "averages", "uncapped", "price_zone"), [
    (70.0, SMAS, "strong_buy", "near_52w_low"),
    (119.0, SMAS, "reduce", "near_52w_high"),
    (85.0, SMAS, "accumulate", "below_sma200"),
    (105.0, SMAS, "hold_bullish", "above_sma50"),
    (95.0, SMAS, "hold_neutral", "at_or_below_sma50"),
    (95.0, {}, "undetermined", "undetermined"),
])
def test_each_uncapped_label_has_its_price_zone_name(
        price: float, averages: dict[str, float], uncapped: str, price_zone: str) -> None:
    """OC-RF6 (unit): each uncapped classification gets its plan §1.2 name, including hold_neutral (price between
    SMA200 and SMA50) as at_or_below_sma50, which no matrix row reaches. PD D-9."""
    zones = build_action_zones(current_price=price, tech_data={"moving_averages": averages,
                                                               "price_position": RANGE_70_120},
                               risk_data=CALM_VOL, fund_data={"valuation": {"pe_trailing": 9.0}})

    assert _classify_current_zone(price, zones["levels"], averages.get("sma_50"), averages.get("sma_200")) == uncapped
    assert zones["price_zone"] == price_zone


@pytest.mark.parametrize("name", ROWS)
async def test_near_52w_high_reads_the_same_on_price_zone(name: str) -> None:
    """G-ZONE (2.3): on every row the gate's input moves from the legacy zone to price_zone without changing its
    status: reduce iff near_52w_high, and null/undetermined iff null/undetermined, so decide() is unchanged. G."""
    result = await dm.run(name)
    legacy = result["diagnostics"]["pre_gate_policy"]["inputs"]["zone"]
    price_zone = result["action_zones"]["price_zone"]

    assert (legacy == "reduce") is (price_zone == "near_52w_high")
    assert (legacy in (None, "undetermined")) is (price_zone in (None, "undetermined"))
