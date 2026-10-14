"""WP2.1: the GATES registry (plan §4; annex step2-v2-research/gate_contract.md, GC-R1, GC-R2, GC-R23).

The registry is data: each gate's class, effect, outcome per surface, clears_when and unevaluated
rule. These tests pin its contents and check every gate id analyze, analyze_trade_setup and
scan_watchlist emit against it. GC-R2's swing column is here, its screen column with scan stage 1,
and its analyze columns in test_wp22_decide.py (WP2.2). Analyze runs go through
tests/support/decision_matrix.py and swing runs through tests/support/swing_matrix.py; every run
pins its clock.
"""

from pathlib import Path
from typing import Any

import pytest

from stock_analysis.tools.analyze.gates import (
    ANALYZE_GATE_IDS,
    CLASS_RANK,
    GATES,
    NOT_APPLICABLE,
    SCREEN_UNEVALUATED,
    SURFACES,
    UNEVALUATED,
    disclosure_order,
    gate_class_for,
    gate_rank,
)
from tests.support import decision_matrix as dm
from tests.support import swing_matrix as sm
from tests.test_step2 import test_characterize_gate_contract as gc
from tests.test_tools.test_trade_card import build_card

NA, UE, WFD = NOT_APPLICABLE, UNEVALUATED, "wait_for_data"
ACTIONABLE = frozenset({"trade_now", "enter_on_trigger"})
PULLBACK = "pullback_in_uptrend"

# id: (class, effect, mid_term, long_term, swing, screen), in registry (within-rank disclosure) order
REGISTRY: dict[str, tuple[str, ...]] = {
    "stale_data": ("data", "veto", WFD, WFD, WFD, WFD),
    "freshness_unverifiable": ("data", "veto", WFD, WFD, WFD, WFD),
    "data_quality_critical": ("data", "veto", WFD, WFD, WFD, WFD),
    "missing_runway": ("data", "veto", WFD, WFD, NA, NA),
    "falling_knife": ("price_risk", "veto", "avoid", "avoid", "avoid", "avoid"),
    "weak_liquidity": ("liquidity", "veto", "avoid", "avoid", "avoid", "avoid"),
    "liquidity_missing": ("liquidity", "veto", WFD, WFD, "avoid", "avoid"),
    "thesis_broken": ("thesis", "cap", "avoid", "avoid", NA, NA),
    "unsupported_instrument": ("support", "cap", "watch", "watch", "watch", "watch"),
    "unsupported_currency": ("support", "cap", "watch", "watch", "watch", "watch"),
    "earnings_blackout": ("event", "cap", "wait", "wait", "watch", UE),
    "earnings_unverifiable": ("event", "cap", "wait", "wait", "watch", UE),
    "near_52w_high": ("zone", "cap", "wait", NA, NA, NA),
    "market_closed": ("session", "conditional", NA, NA, "enter_on_trigger", UE),
    "plan_invariant_violation": ("plan", "cap", NA, NA, "watch", UE),
}
EFFECT_BY_CLASS = {"data": "veto", "price_risk": "veto", "liquidity": "veto", "thesis": "cap", "support": "cap",
                   "event": "cap", "zone": "cap", "session": "conditional", "plan": "cap"}
OUTCOMES = {"mid_term": {WFD, "avoid", "watch", "wait", NA}, "long_term": {WFD, "avoid", "watch", "wait", NA},
            "swing": {WFD, "avoid", "watch", "enter_on_trigger", NA}, "screen": {WFD, "avoid", "watch", NA, UE}}
# the surfaces where an applicable gate can go uncomputed
UNEVALUATED_WHEN = {
    "stale_data": {"screen"}, "freshness_unverifiable": {"screen"}, "missing_runway": {"mid_term", "long_term"},
    "falling_knife": set(SURFACES), "thesis_broken": {"mid_term", "long_term"},
    "unsupported_instrument": {"screen"}, "unsupported_currency": {"screen"},
    "earnings_blackout": {"mid_term", "long_term", "swing"}, "near_52w_high": {"mid_term"},
}
SWING_ORDER = ("stale_data", "freshness_unverifiable", "data_quality_critical", "falling_knife", "weak_liquidity",
               "liquidity_missing", "unsupported_instrument", "unsupported_currency", "earnings_blackout",
               "earnings_unverifiable", "market_closed", "plan_invariant_violation")


def ids(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blockers]


# --- GC-R1: the registry and the ids every tool emits ----------------------------------------


def test_registry_complete() -> None:
    """GC-R1: GATES holds the 15 plan §4 gates in disclosure order, each with its class, effect, outcome per
    surface, clears_when and unevaluated rule; ranks, the listed exceptions and the derived orders follow.
    RF: plan §4 registry, §0 #19 ranks, R2.6(a)(c)."""
    assert list(GATES) == list(REGISTRY)
    assert {g: (gate.gate_class, gate.effect, *(gate.outcome[s] for s in SURFACES))
            for g, gate in GATES.items()} == REGISTRY
    assert CLASS_RANK == {"data": 1, "price_risk": 2, "liquidity": 2, "thesis": 3, "support": 4, "event": 5,
                          "zone": 6, "session": 7, "plan": 8}
    assert set(EFFECT_BY_CLASS) == set(CLASS_RANK)
    for g, gate in GATES.items():
        assert gate.effect == EFFECT_BY_CLASS[gate.gate_class], g
        assert gate.predicate and gate.clears_when, g
        assert set(gate.outcome) == set(SURFACES), g
        assert all(gate.outcome[s] in OUTCOMES[s] for s in SURFACES), g
        assert set(gate.unevaluated_when) == UNEVALUATED_WHEN.get(g, set()), g
        assert all(reason and gate.outcome[s] not in (NA, UE) for s, reason in gate.unevaluated_when.items()), g
        assert gate.outcome["long_term"] == NA or gate.outcome["mid_term"] != NA, g

    assert {g: gate.analyze_class for g, gate in GATES.items() if gate.analyze_class} == {"liquidity_missing": "data"}
    assert [gate_class_for("liquidity_missing", s) for s in SURFACES] == ["data", "data", "liquidity", "liquidity"]
    assert [gate_rank("liquidity_missing", s) for s in SURFACES] == [1, 1, 2, 2]
    per_tool = ("stale_data", "freshness_unverifiable", "data_quality_critical")
    assert {g: set(gate.tool_predicates) for g, gate in GATES.items() if gate.tool_predicates} == {
        g: {s for s in SURFACES if GATES[g].outcome[s] != NA} for g in per_tool}
    assert all(text for g in per_tool for text in GATES[g].tool_predicates.values())
    assert {g for g, gate in GATES.items() if gate.requires_supported} == {"weak_liquidity", "liquidity_missing"}

    assert ANALYZE_GATE_IDS == disclosure_order("mid_term") == (
        "stale_data", "freshness_unverifiable", "data_quality_critical", "missing_runway", "liquidity_missing",
        "falling_knife", "weak_liquidity", "thesis_broken", "unsupported_instrument", "unsupported_currency",
        "earnings_blackout", "earnings_unverifiable", "near_52w_high")
    assert disclosure_order("swing") == disclosure_order("screen") == SWING_ORDER
    assert {"earnings_blackout", "earnings_unverifiable", "market_closed",
            "plan_invariant_violation"} == SCREEN_UNEVALUATED


# Every analyze run in this file: the decision matrix, GC-C3's pipeline rows and the GC-R11 sinks
# (fundamentals, risk_metrics and events_calendar fail).
US_SINK = dm.Scenario("GC-R11", dm.KNIFE, None, None, events=None)
ETF_SINK = dm.Scenario("GC-R11", dm.KNIFE, None, None, events=None, summary=dm.ETF)
ANALYZE_RUNS: dict[str, dm.Scenario] = {
    **dm.SCENARIOS, **{name: row[0] for name, row in gc.ANALYZE_PIPELINE_ROWS.items()},
    "us_sink": US_SINK, "etf_sink": ETF_SINK,
}


async def analyze_gate_ids() -> dict[str, list[str]]:
    """Each analyze run's gate ids: hard_gates.blocking, diagnostics.pre_gate_policy's and the dip's support_gates."""
    emitted: dict[str, list[str]] = {}
    for name, scenario in ANALYZE_RUNS.items():
        result = await dm.run(scenario)
        assessment = result["dip_assessment"].get("assessment") or {}
        emitted[name] = [*dm.gate_ids(result), *(result["diagnostics"]["pre_gate_policy"].get("support_gates") or []),
                         *(assessment.get("support_gates") or [])]
    return emitted


async def swing_cards(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Every pure card case, then the real analyze_trade_setup card on every GC-C4 frame."""
    cards = {name: sm.card_case(name) for name in sm.ALL_CARD_CASES}
    for name, scenario in sm.SCAN_SCENARIOS.items():
        sm.install(monkeypatch, scenario)
        cards[f"pipeline:{name}"] = await sm.trade_card()
    return cards


async def test_every_emitted_gate_id_is_registered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R1: every gate id analyze (mid_term), the card (swing) and scan rows emit is in GATES and applicable
    on the emitting surface; a scan row is the screen's when no full card ran, and its ids are leg-stripped.
    RF: R2.6(a)."""
    emitted: list[tuple[str, str, str]] = []  # (surface, source, gate id)
    for name, gate_ids in (await analyze_gate_ids()).items():
        emitted += [("mid_term", name, g) for g in gate_ids]
    for name, card in (await swing_cards(monkeypatch)).items():
        emitted += [("swing", name, g) for g in ids(card["blockers"])]
    for name, scenario in sm.SCAN_SCENARIOS.items():
        sm.install(monkeypatch, scenario)
        result = await sm.scan(tmp_path / name)
        surface = "screen" if result["full_cards"] == 0 else "swing"
        emitted += [(surface, f"scan:{name}", b.split("@")[0]) for row in result["rows"] for b in row["blockers"]]

    assert [(surface, source, g) for surface, source, g in emitted
            if g not in GATES or GATES[g].outcome[surface] == NA] == []


# --- GC-R2: the swing column -----------------------------------------------------------------

# gate: (a card case name or build_card overrides that fire it, the card's setup type, its plan's entry validity)
SWING_CASES: dict[str, tuple[str | dict[str, Any], str | None, str | None]] = {
    "stale_data": ("stale_only", None, None),
    "freshness_unverifiable": ("unverifiable_freshness", None, None),
    "data_quality_critical": ({"tool_failures": [{"tool": "technicals", "error": "x"}]}, None, None),
    "falling_knife": ("knife", None, None),
    "weak_liquidity": ("weak_liq", None, None),
    "liquidity_missing": ("liq_missing", None, None),
    # support and event caps keep the setup and null the plan; no_setup passes (NO_SETUP_PASSES)
    "unsupported_instrument": ("etf_with_setup", PULLBACK, None),
    "unsupported_currency": ({"summary_data": sm.GBP_SUMMARY, **sm.TRIGGER_CROSSED}, PULLBACK, None),
    "earnings_blackout": ("setup+earnings3", PULLBACK, None),
    "earnings_unverifiable": ("setup+earn_unverifiable", PULLBACK, None),
    # conditional: the plan is kept, valid from the next session
    "market_closed": ("off_session_armed", PULLBACK, "next_session"),
    "plan_invariant_violation": ("plan_invalid", PULLBACK, None),
}
NO_SETUP_PASSES = {"unsupported_instrument": "etf_no_setup", "unsupported_currency": "gbp_no_setup",
                   "earnings_blackout": "no_setup+earnings3"}


@pytest.mark.parametrize("gate_id", GATES)
async def test_gate_outcome_per_horizon_swing(gate_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """GC-R2: each gate's card case gives the registry's swing outcome with the gate among its blockers; a
    not_applicable gate is on no pure or pipeline card. RF: R2.6(d), swing column."""
    outcome = GATES[gate_id].outcome["swing"]
    if outcome == NA:
        cards = await swing_cards(monkeypatch)
        assert [name for name, card in cards.items() if gate_id in ids(card["blockers"])] == []
        return
    case, setup_type, plan_valid = SWING_CASES[gate_id]
    card = sm.card_case(case) if isinstance(case, str) else build_card(**case)
    plan = card["plan"]

    assert card["action"] == outcome
    assert gate_id in ids(card["blockers"])
    assert ((card["setup"] or {}).get("type"), None if plan is None else plan["entry"]["valid"]) == (
        setup_type, plan_valid)
    if gate_id in NO_SETUP_PASSES:
        companion = sm.card_case(NO_SETUP_PASSES[gate_id])
        assert (companion["action"], gate_id in ids(companion["blockers"])) == ("no_setup", True)


# --- GC-R23: conditional gates keep the plan ---------------------------------------------------


async def test_registry_conditional_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """GC-R23: market_closed is the only conditional gate; an actionable card carries only conditional blockers
    and keeps its plan, and analyze emits no session or plan gate. RF: R2.6(g)."""
    assert [g for g, gate in GATES.items() if gate.effect == "conditional"] == ["market_closed"]

    cards = await swing_cards(monkeypatch)
    for name, card in cards.items():
        if card["action"] in ACTIONABLE:
            assert all(GATES[g].effect == "conditional" for g in ids(card["blockers"])), name
            assert card["plan"] is not None, name
    for name in ("off_session_armed", "off_session_trigger_crossed"):
        assert (cards[name]["action"], cards[name]["plan"]["entry"]["valid"]) == ("enter_on_trigger", "next_session")

    session_or_plan = {"market_closed", "plan_invariant_violation"}
    assert [name for name, gate_ids in (await analyze_gate_ids()).items() if session_or_plan & set(gate_ids)] == []
