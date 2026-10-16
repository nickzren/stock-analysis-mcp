"""Unknown applicable thesis evidence blocks entry without inventing a broken business."""

import copy
import importlib
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from stock_analysis.tools.analyze.decision_card import evaluate_gates
from stock_analysis.tools.analyze.gates import GATES, NOT_APPLICABLE
from tests.support import decision_matrix as dm
from tests.support import output_contract as oc

diff_analysis = importlib.import_module("stock_analysis.tools.diff_analysis")
GATE = "thesis_unverifiable"
CLEAN = "S2_buy_clean"


def ids(items: list[dict[str, Any]]) -> set[str]:
    return {item["id"] for item in items}


def horizons(card: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return card["new_money"], card["long_term"]["new_money"]


async def compare(monkeypatch: pytest.MonkeyPatch, previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setattr(diff_analysis, "analyze_stock", AsyncMock(return_value=current))
    return await diff_analysis.what_changed("ACME", previous_snapshot=previous)


@pytest.mark.parametrize("name", [CLEAN, "S1_hold_or_add_clean", "C1_headwind_hold_or_add"])
async def test_partial_thesis_blocks_entry_and_every_derived_surface(name: str) -> None:
    result = await dm.run(name)
    for tier, shown in dm.tiers(result).items():
        card = shown["decision_card"]
        for money in horizons(card):
            assert money["action"] == "wait_for_data", (name, tier)
            blocker = next(gate for gate in money["blocked_by"] if gate["id"] == GATE)
            assert (blocker["class"], blocker["effect"]) == ("data", "veto")
            assert money["decided_by"] == GATE
            assert "thesis_broken" in ids(money["unevaluated"])
            assert GATE not in ids(money["unevaluated"])
        assert card["action_now"] == "wait_for_data"
        assert card["if_held"] == {"action": "insufficient_data", "basis": "thesis_unknown", "add_allowed": False}
        assert card["sizing"] is None and card["entry"] is None
        assert GATE in shown["watchlist_snapshot"]["hard_gate_ids"]
        assert "thesis_broken" in shown["watchlist_snapshot"]["unevaluated_gate_ids"]
        assert not oc.registry_violations(shown, tier)
        assert not oc.actionable_violations(shown, card)
        assert not oc.freshness_violations(shown)


async def test_conditions_and_reassessment_name_evidence_not_a_failed_thesis() -> None:
    result = await dm.run(CLEAN)
    card = result["decision_card"]
    assert GATE in dm.gate_ids(result)
    clearing = "the stock type is confirmed and the required business and cash-runway data are available"
    assert GATES[GATE].clears_when == clearing
    assert card["conditions"]["buy_only_if"] == [clearing]
    assert card["long_term"]["new_money"]["conditions"]["buy_only_if"] == [clearing]
    for money in horizons(card):
        review = money["earliest_reassessment"]
        assert review["date"] is None
        assert review["basis"] == "conditions_only"
        assert review["outstanding"] == [{"id": GATE, "clears_when": clearing}]
    assert result["dislocation_framework"]["action"]["buy_only_if"] == [clearing]
    assert result["decision_modes"]["core"]["required_before_full_size"] == [clearing]
    for tier, shown in dm.tiers(result).items():
        shown_card = shown["decision_card"]
        if tier == "decision":
            assert "executive_summary" not in shown
        else:
            assert shown["executive_summary"].count(f"re-run analyze after {clearing}.") == 2
        assert shown_card["conditions"]["buy_only_if"] == [clearing]
        assert shown_card["long_term"]["new_money"]["conditions"]["buy_only_if"] == [clearing]
        for money in horizons(shown_card):
            assert money["earliest_reassessment"]["outstanding"] == [{"id": GATE, "clears_when": clearing}]


def test_new_data_gate_applies_only_to_analyze() -> None:
    assert GATE in GATES
    gate = GATES[GATE]
    assert (gate.gate_class, gate.effect) == ("data", "veto")
    assert dict(gate.outcome) == {"mid_term": "wait_for_data", "long_term": "wait_for_data",
                                  "swing": NOT_APPLICABLE, "screen": NOT_APPLICABLE}


@pytest.mark.parametrize("signals", [
    {"business_status": "broken", "business_evaluable": True, "runway_evaluable": False},
    {"balance_status": "unsafe", "business_evaluable": False, "runway_evaluable": True},
])
def test_known_failure_dominates_a_missing_thesis_component(signals: dict[str, Any]) -> None:
    gates, unevaluated = evaluate_gates(
        events_data=dm.FAR, data_quality={"fundamentals_status": "available"}, technicals_data=dm.STRONG,
        fundamentals_summary={}, risk_data=dm.CALM, support=None, freshness_blockers=[],
        thesis_signals=signals, price_zone="below_sma50",
    )
    assert "thesis_broken" in ids(gates["blocking"])
    assert GATE not in ids(gates["blocking"]) and GATE not in unevaluated
    assert "thesis_broken" not in unevaluated
    decision = dm.decide_on({"mid_term": "buy", "long_term": "accumulate"}, gates["blocking"])
    assert all(money["action"] == "avoid" for money in horizons(decision))


@pytest.mark.parametrize("summary,decisive", [
    ({**dm.EQUITY, "quote_type": None}, "unsupported_instrument"),
    ({**dm.EQUITY, "currency": None}, "unsupported_currency"),
])
async def test_support_watch_is_not_loosened_by_the_data_gate(summary: dict[str, Any], decisive: str) -> None:
    result = await dm.run(CLEAN, summary=summary)
    for money in horizons(result["decision_card"]):
        assert GATE in ids(money["blocked_by"])
        assert money["action"] == "watch"
        assert money["decided_by"] == decisive


@pytest.mark.parametrize("mid,long,expected", [
    ("buy", "accumulate", "wait_for_data"),
    ("hold_or_add", "hold_or_add", "wait_for_data"),
    ("insufficient_data", "insufficient_data", "insufficient_data"),
    ("avoid", "avoid_core_position", "avoid"),
])
def test_decided_by_names_the_data_gate_only_when_it_changes_the_action(mid: str, long: str, expected: str) -> None:
    decision = dm.decide_on({"mid_term": mid, "long_term": long}, [{"id": GATE, "reason": "thesis unavailable"}])
    for money in horizons(decision):
        assert money["action"] == expected
        assert money["decided_by"] == (GATE if expected == "wait_for_data" else None)
        assert GATE in ids(money["blocked_by"])


async def test_unsupported_instrument_does_not_acquire_a_stock_thesis_gate() -> None:
    result = await dm.run(CLEAN, summary=dm.ETF)
    for money in horizons(result["decision_card"]):
        assert money["action"] == "watch"
        assert "unsupported_instrument" in ids(money["blocked_by"])
        assert GATE not in ids(money["blocked_by"])
        assert "thesis_broken" not in ids(money["unevaluated"])


async def test_complete_business_evidence_keeps_buy_and_a_holder_answer() -> None:
    result = await dm.run(CLEAN, fundamentals=dm.QUALITY)
    card = result["decision_card"]
    assert card["new_money"]["action"] == "buy"
    assert card["long_term"]["new_money"]["action"] == "buy"
    assert card["if_held"]["action"] == "hold" and card["if_held"]["add_allowed"] is True
    assert card["sizing"] is not None
    assert GATE not in dm.gate_ids(result)
    assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]


async def test_confirmed_negative_fcf_can_still_have_an_evaluable_thesis() -> None:
    payload = copy.deepcopy(dm.MIXED_FCF)
    payload["financial_health"].pop("debt_to_equity")
    result = await dm.run(CLEAN, fundamentals=payload)
    assert result["verdict"]["decomposed"]["business_quality_status"] == "available"
    assert GATE not in dm.gate_ids(result)
    assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]
    assert result["decision_card"]["if_held"]["basis"] != "thesis_unknown"


async def test_nonburner_needs_no_cash_balance_or_numeric_runway() -> None:
    payload = copy.deepcopy(dm.BURNER)
    payload["cash_flow"].update(free_cash_flow_ttm=1e8, operating_cf_ttm=1e8)
    payload["cash_flow"]["rules"]["positive_fcf"]["triggered"] = True
    payload["financial_health"].pop("cash_and_st_investments")
    result = await dm.run(CLEAN, fundamentals=payload)
    burn = result["fundamentals_summary"]["burn_metrics"]
    assert burn["status"] == "not_applicable" and burn["cash_runway_quarters"] is None
    assert GATE not in dm.gate_ids(result)
    assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]
    assert result["decision_card"]["if_held"]["basis"] != "thesis_unknown"


async def test_wholly_missing_fundamentals_disclose_the_cap_without_a_holder_guess() -> None:
    result = await dm.run(CLEAN, fundamentals=None)
    for money in horizons(result["decision_card"]):
        assert GATE in ids(money["blocked_by"])
        assert money["action"] not in {"buy", "starter"}
        assert "thesis_broken" in ids(money["unevaluated"])
    assert result["decision_card"]["if_held"] == {
        "action": "insufficient_data", "basis": "thesis_unknown", "add_allowed": False,
    }


async def test_partial_loss_and_recovery_report_the_new_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    complete = await dm.run(CLEAN, fundamentals=dm.QUALITY)
    unknown = await dm.run(CLEAN)
    lost = await compare(monkeypatch, complete["watchlist_snapshot"], unknown)
    assert GATE in lost["changes"]["hard_gates"]["new"]
    assert f"+hard_gate: {GATE}" in lost["material_changes"]
    assert "thesis_broken" not in lost["changes"]["hard_gates"]["cleared"]
    assert unknown["watchlist_snapshot"]["new_money_action"] == "wait_for_data"
    recovered = await compare(monkeypatch, unknown["watchlist_snapshot"], complete)
    assert GATE in recovered["changes"]["hard_gates"]["cleared"]
    assert f"-hard_gate: {GATE}" in recovered["material_changes"]
    assert "-unevaluated_gate: thesis_broken" in recovered["material_changes"]


async def test_broken_thesis_loss_is_unknown_not_clear_and_entry_stays_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    known = await dm.run(CLEAN, fundamentals=dm.POOR_BUSINESS)
    partial = copy.deepcopy(dm.POOR_BUSINESS)
    partial["financial_health"].pop("debt_to_equity")
    unknown = await dm.run(CLEAN, fundamentals=partial)
    assert GATE in dm.gate_ids(unknown)
    assert all(money["action"] not in {"buy", "starter"} for money in horizons(unknown["decision_card"]))
    change = await compare(monkeypatch, known["watchlist_snapshot"], unknown)
    assert "thesis_broken" in change["changes"]["hard_gates"]["not_evaluated"]
    assert "thesis_broken" not in change["changes"]["hard_gates"]["cleared"]
    assert GATE in change["changes"]["hard_gates"]["new"]


async def test_legacy_snapshot_warning_names_the_new_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    old = json.loads((dm.SNAPSHOT_DIR / f"{CLEAN}.json").read_text())
    current = await dm.run(CLEAN)
    change = await compare(monkeypatch, old, current)
    assert GATE in change["changes"]["hard_gates"]["new"]
    assert len(change["warnings"]) == 1
    assert "2.0.0 -> 3.0.0" in change["warnings"][0]
    assert GATE in change["warnings"][0]
