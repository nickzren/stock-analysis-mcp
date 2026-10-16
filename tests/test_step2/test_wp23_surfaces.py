"""WP2.3 derived surfaces (plan §3 [D-3], §7 [D-9]): the executive summary's horizon segments, the decision
modes, the dislocation action and check, the dip caps and price_zone, all derived from decide()'s decision and
never feeding it. Annex: output_contract.md §4, §6, §7 and policy_holder.md §8, §10, "Corrected proposal", as
amended by plan §0 (#8, #9, #10, #11, #21, #23)."""

import re
from datetime import date
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
    ENTRY_ACTIONS,
    ENTRY_CODES,
    EVIDENCE_BASES,
    FULL_ENTRY_CODES,
    IF_HELD_ACTIONS,
    NEW_MONEY_ACTIONS,
    NON_EVIDENCE_BASES,
    entry_allowed,
    most_restrictive,
)
from stock_analysis.tools.analyze.dip_assessment import (
    align_dip_assessment_with_action,
    cap_dip_assessment,
)
from stock_analysis.tools.analyze.dislocation_framework import build_dislocation_framework
from stock_analysis.tools.analyze.executive_summary import horizon_segments
from stock_analysis.tools.analyze.gates import GATES, THESIS_CLEARS_WHEN
from stock_analysis.tools.analyze.reassessment import PERMANENT_CLEARS_WHEN
from tests.support import decision_matrix as dm
from tests.support import output_contract as oc

ROWS = list(dm.SCENARIOS)
PREFIXES = ("Decision (mid-term, new money): ", "If held: ", "Long-term (1-5y): ")
BUY_WORD = re.compile(r"(?i)\b(buy|add|accumulate|starter)\b")
PRICE_ZONE_NAMES = {"near_52w_low", "near_52w_high", "below_sma200", "above_sma50", "at_or_below_sma50",
                    "undetermined"}


def words(code: str) -> str:
    return code.replace("_", " ")


def ids(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blockers]


def tight(action: str) -> int:
    return NEW_MONEY_ACTIONS.index(action)


# --- (a) executive summary: three horizon segments (OC-RF8, PH-RF20, RA.15) -----------------------------------


@pytest.mark.parametrize("name", ROWS)
async def test_the_summary_opens_with_three_linted_horizon_segments(name: str) -> None:
    """OC-RF8: the summary's first three sentences are the new-money, if-held and long-term segments, in order.
    Segment 1 names new_money's action and gates and carries "size ~" iff entry_allowed (a starter shows its
    starter pct alone); each segment passes R-LEX-1 against its own horizon; every later sentence passes R-LEX-2.
    PD D-9."""
    result = await dm.run(name)
    card = result["decision_card"]
    new_money, held, long = card["new_money"], card["if_held"], card["long_term"]["new_money"]
    for tier in ("standard", "full"):
        assert dm.tiers(result)[tier]["executive_summary"] == result["executive_summary"], tier
    sentences = oc.segments(result["executive_summary"])
    first, second, third, body = *sentences[:3], sentences[3:]

    assert [s[:len(p)] for s, p in zip(sentences, PREFIXES, strict=False)] == list(PREFIXES)
    assert first.startswith(PREFIXES[0] + words(new_money["action"]))
    assert (("blocked by " + ", ".join(ids(new_money["blocked_by"]))) in first) is bool(new_money["blocked_by"])
    sizing = card["sizing"] or {}
    size = (f"size ~{sizing['starter_pct']:.1f}-{sizing['full_pct_cap']:.1f}%" if new_money["action"] == "buy"
            else f"size ~{sizing.get('starter_pct') or 0:.1f}%")
    assert ("size ~" in first) is entry_allowed(new_money["action"])
    if entry_allowed(new_money["action"]):
        assert f"; {size}" in first and first.count("size ~") == 1
    assert second == f"If held: {words(held['action'])}."
    assert third.startswith(PREFIXES[2] + words(long["action"]))
    assert result["executive_summary"].count("size ~") == first.count("size ~")
    assert not oc.advice(first) or entry_allowed(new_money["action"]), first
    assert not oc.advice(third) or long["action"] in ENTRY_ACTIONS, third
    assert "size ~" not in third
    assert not re.search(r"(?i)\badd\b|hold or add", second) or held["add_allowed"]
    for sentence in body:
        assert not oc.advice(sentence), sentence


THESIS_WAIT = ("wait for data; blocked by thesis_unverifiable; no reassessment date: re-run analyze after "
               "applicability verified and required business and cash-runway checks evaluable.")


@pytest.mark.parametrize(("name", "segments"), [
    ("S2_buy_clean", ("Decision (mid-term, new money): " + THESIS_WAIT, "If held: insufficient data.",
                      "Long-term (1-5y): " + THESIS_WAIT)),
    ("S1_hold_or_add_clean", ("Decision (mid-term, new money): " + THESIS_WAIT, "If held: insufficient data.",
                              "Long-term (1-5y): " + THESIS_WAIT)),
    ("S3_plain_avoid", ("Decision (mid-term, new money): avoid; blocked by thesis_unverifiable.",
                       "If held: insufficient data.", "Long-term (1-5y): " + THESIS_WAIT)),
    ("S4_earnings_gate_on_hold_or_add", (
        "Decision (mid-term, new money): wait for data; blocked by thesis_unverifiable, earnings_blackout; "
        "reassess on or after Tue Oct 13 (not a buy date).", "If held: insufficient data.",
        "Long-term (1-5y): wait for data; blocked by thesis_unverifiable, earnings_blackout; "
        "reassess on or after Tue Oct 13 (not a buy date).")),
    ("A7_near_52w_high_bullish", (
        "Decision (mid-term, new money): wait for data; blocked by thesis_unverifiable, near_52w_high; "
        "no reassessment date: re-run analyze after applicability verified and required business and "
        "cash-runway checks evaluable and price leaves the near-52-week-high band.", "If held: insufficient data.",
        "Long-term (1-5y): " + THESIS_WAIT)),
    ("GBp_strong", (
        "Decision (mid-term, new money): watch; blocked by thesis_unverifiable, unsupported_currency; "
        "outside supported scope: no reassessment date.", "If held: insufficient data.",
        "Long-term (1-5y): watch; blocked by thesis_unverifiable, unsupported_currency; "
        "outside supported scope: no reassessment date.")),
    ("A_broken_bearish", ("Decision (mid-term, new money): avoid; blocked by thesis_broken.",
                          "If held: review exit.", "Long-term (1-5y): avoid; blocked by thesis_broken.")),
])
async def test_named_segments(name: str, segments: tuple[str, str, str]) -> None:
    """OC-RF8 / RA.15 (summary clauses): the three segments word for word on the named rows (thesis-blocked
    buy/starter candidates, candidate avoid, dated, conditions-only and unsupported reassessment).
    PD D-9. Sparse business inputs now give holder uncertainty (WP2.5 C2), and now block new money with a separate gate."""
    assert oc.segments((await dm.run(name))["executive_summary"])[:3] == list(segments)


@pytest.mark.parametrize(("name", "mid", "held", "long"), [
    ("D_quality_attractive_poor_chart", "avoid", "hold", "buy"),
    ("E2_mixed_fcf_bullish_chart", "starter; size ~2.5%", "hold", "starter"),
    ("C1_headwind_hold_or_add", "starter; size ~2.5%", "hold", "wait")])
async def test_the_summary_names_both_horizons(name: str, mid: str, held: str, long: str) -> None:
    """PH-RF20: the summary names the mid-term and the long-term decision, each labelled, so a disagreement reads
    as two horizons (D: avoid vs buy; C1: starter vs wait). PD D-3."""
    fund = dm.SCENARIOS[name].fundamentals
    if name == "C1_headwind_hold_or_add":
        fund = {**fund, "financial_health": {"debt_to_equity": 0.5}}
    summary = (await dm.run(name, fundamentals=fund))["executive_summary"]

    assert summary.startswith(f"Decision (mid-term, new money): {mid}. If held: {held}. Long-term (1-5y): {long}.")


def _card(earliest: dict[str, Any] | None, action: str = "wait") -> dict[str, Any]:
    new_money = {"action": action, "blocked_by": [{"id": "earnings_blackout"}], "earliest_reassessment": earliest}
    return {"new_money": new_money, "if_held": {"action": "hold"},
            "long_term": {"new_money": {"action": "buy", "blocked_by": [], "earliest_reassessment": None}}}


def test_the_reassessment_wordings() -> None:
    """RA.15 (summary clauses): plan §6's worked example renders "reassess on or after Fri Oct 16 (not a buy
    date)"; a null date renders "no reassessment date: re-run analyze after <clears_when ...>" with the
    outstanding conditions in order; a permanent support gate says so; never "now". RF."""
    dated = {"date": "2026-10-16", "basis": "earnings_clear", "earnings_risk_through": "2026-10-15",
             "session_method": "calendar_static", "outstanding": [{"id": "earnings_reverify", "clears_when": "x"}]}
    undated = {"date": None, "basis": "conditions_only", "earnings_risk_through": None, "session_method": None,
               "outstanding": [{"id": "falling_knife", "clears_when": "a"}, {"id": "earnings_reverify",
                                                                           "clears_when": "b"}]}
    unsupported = {**undated, "basis": "unsupported"}

    assert horizon_segments(_card(dated))[0] == (
        "Decision (mid-term, new money): wait; blocked by earnings_blackout; reassess on or after Fri Oct 16 "
        "(not a buy date).")
    assert horizon_segments(_card(undated))[0].endswith("; no reassessment date: re-run analyze after a and b.")
    assert horizon_segments(_card(unsupported))[0].endswith("; outside supported scope: no reassessment date.")
    assert horizon_segments(_card(None))[0] == "Decision (mid-term, new money): wait; blocked by earnings_blackout."


@pytest.mark.parametrize("name", ROWS)
async def test_a_reassessment_date_never_sits_next_to_a_buy_word(name: str) -> None:
    """RA.15 (summary clauses) and RA.11: in every horizon segment carrying earliest_reassessment, the only buy
    word is "not a buy date" and nothing says "now"; segments without it say nothing about reassessment; no
    re-entry wording anywhere in the summary. RF."""
    result = await dm.run(name)
    card = result["decision_card"]
    summary = result["executive_summary"]
    first, _, third = oc.segments(summary)[:3]

    assert not re.search(r"(?i)re-?entr|re-?enter|resum", summary)
    for segment, new_money in ((first, card["new_money"]), (third, card["long_term"]["new_money"])):
        earliest = new_money["earliest_reassessment"]
        assert ("reassess" in segment) is (earliest is not None), segment
        if earliest is not None:
            assert not BUY_WORD.search(segment.replace("not a buy date", "")), segment
            assert not re.search(r"(?i)\bnow\b", segment), segment
        if earliest is not None and earliest["date"]:
            day = date.fromisoformat(earliest["date"])
            assert f"reassess on or after {day:%a} {day:%b} {day.day} (not a buy date)" in segment


def test_no_clears_when_text_ends_a_sentence() -> None:
    """Segmentation: no clears_when the summary can quote contains ". " or ends with ".", so the segments split
    on sentence ends. G."""
    texts = [gate.clears_when for gate in GATES.values()] + list(THESIS_CLEARS_WHEN.values()) + [
        PERMANENT_CLEARS_WHEN]
    assert not [t for t in texts if ". " in t or t.endswith(".")]


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


_SPECULATIVE_OWN_INPUTS = {  # own code -> (horizon_fit.mid_term, tilt, dip recommendation)
    "starter_then_add": ("ok", "bullish", None),
    "starter_position_only": ("caution", "bearish", "small_speculative_position"),
    "wait": ("caution", "bearish", "wait_for_better_setup"),
    "watch": ("caution", "bearish", None),
}


@pytest.mark.parametrize(("own", "new_money"), list(product(_SPECULATIVE_OWN_INPUTS, NEW_MONEY_ACTIONS)))
def test_speculative_maps_its_own_code_then_caps_it(own: str, new_money: str) -> None:
    """PH-RF19 (unit): speculative's own codes map starter_then_add/starter_position_only -> starter and
    wait/watch -> wait (a strict lookup); over 4 own codes x 7 new-money actions build_decision_modes gives
    most_restrictive(map(own), new_money), with capped_by iff that differs from map(own). PD D-3."""
    assert investor_profile._SPECULATIVE_ACTIONS == {
        "starter_then_add": "starter", "starter_position_only": "starter", "wait": "wait", "watch": "wait"}
    mid_term_fit, tilt, recommendation = _SPECULATIVE_OWN_INPUTS[own]
    speculative = investor_profile.build_decision_modes(
        summary={"current_price": 10.0},
        verdict={"tilt": tilt, "horizon_fit": {"mid_term": mid_term_fit, "long_term": "ok"}},
        action_zones={}, dip_assessment={"assessment": {"recommendation": recommendation}}, decision_context={},
        news_summary={}, fundamentals_summary={}, events_summary={},
        decision={"new_money": {"action": new_money}, "long_term": {"new_money": {"action": "buy"}}},
        conditions={"mid_term": {"add_only_if": []}, "long_term": {"buy_only_if": []}}, why=[],
    )["speculative"]
    mapped = investor_profile._SPECULATIVE_ACTIONS[own]
    action = most_restrictive((mapped, new_money))

    assert speculative["action"] == action
    assert speculative.get("capped_by") == (None if action == mapped else f"new_money_{action}")


@pytest.mark.parametrize(("name", "long_term"), [("E2_mixed_fcf_bullish_chart", "starter"),
                                                 ("C1_headwind_hold_or_add", "wait")])
async def test_balanced_follows_new_money_in_direction_2(name: str, long_term: str) -> None:
    """PH-RF17 (balanced clause): E2 and C1 give balanced starter == new_money with long_term_view == the
    long-term decision (E2 starter, C1 wait) and core wait. PD D-3."""
    fund = dm.SCENARIOS[name].fundamentals
    if name == "C1_headwind_hold_or_add":
        fund = {**fund, "financial_health": {"debt_to_equity": 0.5}}
    result = await dm.run(name, fundamentals=fund)
    modes = result["decision_modes"]

    assert (result["decision_card"]["new_money"]["action"], modes["balanced"]["action"],
            modes["balanced"]["long_term_view"], modes["core"]["action"]) == ("starter", "starter", long_term, "wait")


# --- (c) dislocation (S-C10, OC-CH3, IS-CH2) -----------------------------------------------------------------


@pytest.mark.parametrize("name", ROWS)
async def test_the_dislocation_action_follows_new_money(name: str) -> None:
    """OC-RF4/P6: dislocation can_start_now == entry_allowed(new_money), buy_now is new_money's action under buy
    or starter and null otherwise, the mode copies equal the capped modes with their capped_by, and its buy and
    add conditions mirror the card's. RF."""
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
    offers starter; a wait offers nothing, whatever the modes say. RF."""
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
    watch; each capped field carries capped_by new_money_<action>, a capped recommendation states the cap, and
    support_gates names the support gates in blocked_by, never another gate. PD D-9 (starter) / RF (non-entry)."""
    support = [{"id": "unsupported_currency", "class": "support"}] if action == "watch" else []
    event = [{"id": "earnings_blackout", "class": "event"}] if action not in ENTRY_ACTIONS else []
    capped = _dip(recommendation)
    cap_dip_assessment(capped, {"action": action, "blocked_by": [*support, *event]})
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
    needs new_money buy. RF."""
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
    price_broken_more_than_business and new_money is starter (T5, T6 extreme); T6 (the same dislocation under buy)
    is left alone, and S6 and C (knife, avoid) keep do_not_catch_falling_knife. PD D-9."""
    result = await dm.run(name)
    assessment = (result["dip_assessment"] or {}).get("assessment") or {}
    mismatch = (result["dislocation_framework"].get("mismatch_verdict") or {}).get("status")
    starter = result["decision_card"]["new_money"]["action"] == "starter"

    assert ("timing_only" in assessment) is (mismatch == "price_broken_more_than_business" and starter)


VOL60 = "risk_regime <= high (volatility < 60%)"
DISLOCATION_ROWS = {  # name -> the reframed dip's add_only_if (None: under buy, no override)
    "T5_dislocation_starter": [], "T6_dislocation_starter_extreme": [VOL60], "T6_dislocation_buy": None}


@pytest.mark.parametrize("name", DISLOCATION_ROWS)
async def test_a_dislocation_starter_reframes_the_dip(name: str) -> None:
    """OC-M9 (call site, override side) and balanced suitability (spec-surfaces §3.4): on a price-vs-business
    dislocation under new_money starter the wait_for_better_setup dip becomes starter_only_wait_for_stabilization
    with dip_quality starter_only, entry-timing scope, and add_only_if the card's mid-term add_only_if (T5: [], never
    its long-term [VOL60]; T6 extreme: [VOL60]); balanced suitability is starter_only. The same dislocation under
    buy (T6) keeps its own dip and is favorable. RF."""
    result = await dm.run(name)
    card = result["decision_card"]
    assessment = result["dip_assessment"]["assessment"]
    add_only_if = DISLOCATION_ROWS[name]

    assert result["dislocation_framework"]["mismatch_verdict"]["status"] == "price_broken_more_than_business"
    assert card["new_money"]["action"] == ("buy" if add_only_if is None else "starter")
    if add_only_if is None:
        assert (assessment["recommendation"], assessment["dip_quality"]) == ("buy_the_dip", "good")
        assert not {"scope", "timing_only", "add_only_if"} & assessment.keys()
    else:
        assert {k: assessment[k] for k in ("recommendation", "dip_quality", "scope", "timing_only", "add_only_if")} \
            == {"recommendation": "starter_only_wait_for_stabilization", "dip_quality": "starter_only",
                "scope": "entry_timing_only", "timing_only": True, "add_only_if": add_only_if}
        assert add_only_if == card["conditions"]["add_only_if"]
    assert result["decision_modes"]["balanced"]["suitability"] == ("favorable" if add_only_if is None
                                                                   else "starter_only")
    if name == "T5_dislocation_starter":
        assert card["long_term"]["new_money"]["conditions"]["buy_only_if"] == [VOL60]


def test_the_starter_override_needs_a_starter() -> None:
    """OC-M9: align_dip_assessment_with_action takes starter_only (new_money == starter), not can_start_now: a
    dislocation starter reframes the falling-knife dip and mirrors the whole mid-term add_only_if (no truncation);
    anything else leaves it alone. RF."""
    def dip() -> dict[str, Any]:
        return {"assessment": {"dip_quality": "avoid", "recommendation": "do_not_catch_falling_knife",
                               "rationale": "Trend is broken - wait for stabilization"}}

    conditions = ["a", "b", "c", "d", "e"]
    started = align_dip_assessment_with_action(dip(), "price_broken_more_than_business", starter_only=True,
                                               add_only_if=conditions)
    assert started is not None and started["assessment"]["recommendation"] == "starter_only_wait_for_stabilization"
    assert started["assessment"]["add_only_if"] == conditions
    assert align_dip_assessment_with_action(dip(), "price_broken_more_than_business", starter_only=False) == dip()


# --- (e) price_zone --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROWS)
async def test_price_zone_is_the_uncapped_classification(name: str) -> None:
    """OC-RF6 (snapshot clause: the §0 #22 pin, WP2.4): action_zones.price_zone is in the closed set and equals the uncapped zone
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


# --- (f) OC-RF9 code hygiene ----------------------------------------------------------------------------------

CODE = re.compile(r"[a-z][a-z0-9_]*")


@pytest.mark.parametrize("name", ROWS)
async def test_no_actionable_field_carries_a_sizing_word(name: str) -> None:
    """OC-RF9 (amended by plan §0 #8): no A code (a snake_case value of a registry A field) at any tier contains
    size or sizing (gate reasons are blocker text, not codes); C1-C3 show the long-term wait_for_data, never
    hold_size_conservatively, which survives only under diagnostics. PD D-9."""
    result = await dm.run(name)
    for tier, payload in dm.tiers(result).items():
        for path, value, field in oc.classified(payload):
            if field.cls == oc.A and isinstance(value, str) and CODE.fullmatch(value):
                assert not re.search(r"(?i)siz(e|ing)", value), (tier, path, value)
        assert "horizon_fit" not in payload["decision_card"], tier
    if name[:2] in {"C1", "C2", "C3"}:
        assert (result["decision_card"]["long_term"]["new_money"]["action"],
                result["decision_modes"]["balanced"]["long_term_view"]) == ("wait_for_data", "wait_for_data")


def test_no_canonical_code_carries_a_sizing_word() -> None:
    """OC-RF9 (unit): no new-money, holder, basis, speculative-mode, price-zone or dip-cap code says size or
    sizing. PD D-9."""
    codes = {*NEW_MONEY_ACTIONS, *IF_HELD_ACTIONS, *EVIDENCE_BASES, *NON_EVIDENCE_BASES, *PRICE_ZONES.values(),
             *investor_profile._SPECULATIVE_ACTIONS.values(), "starter_only", "starter_position", "watch"}
    assert not [c for c in codes if re.search(r"(?i)siz(e|ing)", c)]
