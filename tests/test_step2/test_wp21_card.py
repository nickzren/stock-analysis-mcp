"""WP2.1: the trade card's disclosure (plan §1 changes (1)-(3); annex step2-v2-research/gate_contract.md
GC-R9-R11, reassessment.md RA.18; S-C8).

The card discloses earnings_unverifiable on every outcome, lists the applicable gates it could not compute
as unevaluated, merges duplicate ids, and gives each blocker its registry class and effect, the gate that
decided its action and its horizon. Its resolution order, actions, setups and plans are frozen (2c). RA.18's
scan clause and S-C8's scan rows land with scan stage 1. Card runs go through
tests/test_tools/test_trade_card.py and tests/support/swing_matrix.py; every run pins its clock.
"""

import copy
from typing import Any

import pytest

import stock_analysis.tools.trade_setup.card as card_module
from stock_analysis.tools.analyze.gates import gate_rank
from stock_analysis.utils.validators import FetchParams
from tests.support import swing_matrix as sm
from tests.test_step2.test_characterize_reassessment import UNVERIFIABLE
from tests.test_step2.test_characterize_surfaces import pullback_card
from tests.test_step2.test_gate_registry import swing_cards
from tests.test_tools.test_trade_card import build_card

TECHNICALS_FAILED = {"tool": "technicals", "error": "x"}
EVENTS_FAILED = {"tool": "events_calendar", "error": "x"}
KNIFE_UNEVALUATED = {"id": "falling_knife", "reason": "technicals unavailable — falling-knife rubric not computed"}
BLACKOUT_UNEVALUATED = {"id": "earnings_blackout", "reason": "events calendar unavailable — earnings date not checked"}
DISCLOSURE_ONLY = frozenset({"wait_for_data", "avoid", "no_setup"})  # actions an event cap never changes


def ids(items: list[dict[str, Any]]) -> list[str]:
    return [item["id"] for item in items]


# --- GC-R9: earnings_unverifiable on every outcome ----------------------------------------------

# variant: (events_data, tool failures appended to the case's own)
UNVERIFIABLE_VARIANTS: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
    "unverifiable": (sm.UNVERIFIABLE_EVENTS, []),
    "tool_failed": ({}, [EVENTS_FAILED]),
}


@pytest.mark.parametrize("variant", UNVERIFIABLE_VARIANTS)
@pytest.mark.parametrize("name", sm.ALL_CARD_CASES)
def test_earnings_unverifiable_on_every_card_outcome(
    name: str, variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GC-R9: an unverifiable calendar or an events tool failure lists earnings_unverifiable exactly once on
    every card case; wait_for_data, avoid and no_setup keep their action, every other outcome becomes watch.
    RF: plan §1 change (2), R2.6(f)."""
    events_data, failures = UNVERIFIABLE_VARIANTS[variant]
    own_action = sm.card_case(name)["action"]
    case = copy.deepcopy({**sm.CARD_CASES, **sm.INVALID_PLAN_CASES}[name])
    case.update(events_data=copy.deepcopy(events_data), tool_failures=[*case.get("tool_failures", []), *failures])
    if name in sm.INVALID_PLAN_CASES:
        monkeypatch.setattr(card_module, "detect_setup", lambda *args, **kwargs: dict(sm.INVALID_GEOMETRY_SETUP))

    card = build_card(**case)

    assert ids(card["blockers"]).count("earnings_unverifiable") == 1
    assert card["action"] == (own_action if own_action in DISCLOSURE_ONLY else "watch")


# --- GC-R10, card half: unevaluated ---------------------------------------------------------------


def test_unevaluated_lists() -> None:
    """GC-R10: the card lists falling_knife unevaluated without technicals, and earnings_blackout when the
    events tool failed and no blackout fired, in rank order; every other card case lists nothing, and no
    unevaluated id is also a blocker. RF: R2.6(f), card half."""
    no_technicals = build_card(technicals_data={}, tool_failures=[TECHNICALS_FAILED])
    no_events = build_card(events_data={}, tool_failures=[EVENTS_FAILED])
    neither = build_card(technicals_data={}, events_data={}, tool_failures=[TECHNICALS_FAILED, EVENTS_FAILED])
    cases = {name: sm.card_case(name) for name in sm.ALL_CARD_CASES}

    assert (no_technicals["action"], ids(no_technicals["blockers"]), no_technicals["unevaluated"]) == (
        "wait_for_data", ["data_quality_critical"], [KNIFE_UNEVALUATED])
    assert (no_events["action"], ids(no_events["blockers"]), no_events["unevaluated"]) == (
        "watch", ["earnings_unverifiable"], [BLACKOUT_UNEVALUATED])
    assert neither["unevaluated"] == [KNIFE_UNEVALUATED, BLACKOUT_UNEVALUATED]
    assert {name: ids(card["unevaluated"]) for name, card in cases.items() if card["unevaluated"]} == {
        "no_setup+events_tool_failed": ["earnings_blackout"], "events_tool_failed+stale": ["earnings_blackout"]}
    for card in (no_technicals, no_events, neither, *cases.values()):
        assert not set(ids(card["unevaluated"])) & set(ids(card["blockers"])), card["unevaluated"]


async def test_pipeline_technicals_failure_lists_the_knife(monkeypatch: pytest.MonkeyPatch) -> None:
    """GC-R10: when the technicals tool fails, the real analyze_trade_setup lists falling_knife unevaluated
    and data_quality_critical decides wait_for_data. RF: R2.6(f), card half."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["fresh_breakout"])

    async def no_history(params: FetchParams) -> None:
        raise RuntimeError("history fetch failed")

    monkeypatch.setattr(sm.technicals_tool, "fetch_history", no_history)

    card = await sm.trade_card()

    assert (card["action"], card["decided_by"], card["unevaluated"]) == (
        "wait_for_data", "data_quality_critical", [KNIFE_UNEVALUATED])


# --- GC-R11, card: unique ids ---------------------------------------------------------------------


async def test_card_blocker_ids_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    """GC-R11: the two data_quality_critical branches merge into one entry with both reasons; every pure and
    pipeline card lists each blocker and unevaluated id once. RF: plan §4 "unique ids per output", R2.6(a)."""
    merged = sm.card_case("risk_metrics_failed+no_features")["blockers"]

    assert [(b["id"], b["reason"]) for b in merged] == [
        ("data_quality_critical", "critical data unavailable: critical_tool_failed=risk_metrics; "
                                  "daily price history unavailable or incomplete — cannot compute setup features"),
        ("liquidity_missing", "avg_dollar_volume unavailable — cannot confirm tradability for small account")]
    for name, card in (await swing_cards(monkeypatch)).items():
        for items in (card["blockers"], card["unevaluated"]):
            assert len(ids(items)) == len(set(ids(items))), name


# --- Card disclosure: class, effect, decided_by ------------------------------------------------------

# id: (class, effect) on the swing surface; liquidity_missing is a liquidity veto here (data in analyze)
CLASS_EFFECT: dict[str, tuple[str, str]] = {
    "stale_data": ("data", "veto"), "freshness_unverifiable": ("data", "veto"),
    "data_quality_critical": ("data", "veto"), "falling_knife": ("price_risk", "veto"),
    "weak_liquidity": ("liquidity", "veto"), "liquidity_missing": ("liquidity", "veto"),
    "unsupported_instrument": ("support", "cap"), "unsupported_currency": ("support", "cap"),
    "earnings_blackout": ("event", "cap"), "earnings_unverifiable": ("event", "cap"),
    "market_closed": ("session", "conditional"), "plan_invariant_violation": ("plan", "cap"),
}


async def test_blocker_class_and_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Card disclosure: every blocker on every pure and pipeline card carries its swing class and effect, in
    non-decreasing rank, and no card carries earliest_reassessment; the cards cover every swing gate.
    RF: plan §1 change (1), R2.6(a)(c), §0 #13."""
    seen: set[str] = set()
    for name, card in (await swing_cards(monkeypatch)).items():
        blockers = card["blockers"]
        ranks = [gate_rank(b["id"], "swing") for b in blockers]

        assert [(b["class"], b["effect"]) for b in blockers] == [CLASS_EFFECT[b["id"]] for b in blockers], name
        assert ranks == sorted(ranks), name
        assert "earliest_reassessment" not in card, name
        seen.update(ids(blockers))

    assert seen == set(CLASS_EFFECT)


# case: the gate that moved the action off the swing base (None: the action is the base, or no setup)
DECIDED_BY: dict[str, str | None] = {
    **dict.fromkeys(["stale_only", "stale+knife", "stale+weak_liq", "stale+etf", "events_tool_failed+stale"],
                    "stale_data"),
    "unverifiable_freshness": "freshness_unverifiable",
    **dict.fromkeys(["risk_metrics_failed", "risk_metrics_failed+no_features"], "data_quality_critical"),
    **dict.fromkeys(["knife", "knife+etf", "knife+gbp", "knife+earnings3", "knife+earn_unverifiable"],
                    "falling_knife"),
    "weak_liq": "weak_liquidity",
    "liq_missing": "liquidity_missing",
    **dict.fromkeys(["etf_with_setup", "setup+etf+earnings3", "off_session+etf", "summary_tool_failed",
                     "plan_invalid+etf"], "unsupported_instrument"),
    **dict.fromkeys(["setup+earnings3", "off_session+earnings3"], "earnings_blackout"),
    "setup+earn_unverifiable": "earnings_unverifiable",
    **dict.fromkeys(["plan_invalid", "plan_invalid+off_session"], "plan_invariant_violation"),
    **dict.fromkeys(["clean_armed", "clean_satisfied_regular", "etf_no_setup", "gbp_no_setup", "no_setup+earnings3",
                     "no_setup+earn_unverifiable", "no_setup+events_tool_failed", "off_session_armed",
                     "off_session_trigger_crossed"], None),
}
PIPELINE_DECIDED_BY: dict[str, str | None] = {
    **dict.fromkeys(["stale_frame_boring", "stale_frame_breakout", "C_stale_knife", "C_stale_thin"], "stale_data"),
    **dict.fromkeys(["A_30_bars_breakout", "A_30_bars_boring", "A_45_bars_boring"], "data_quality_critical"),
    "liq_recent_surge_breakout": "weak_liquidity",
    **dict.fromkeys(["etf_breakout", "D_etf_breakout"], "unsupported_instrument"),
    "breakout_earnings_in_2d": "earnings_blackout",
    "breakout_events_unverifiable": "earnings_unverifiable",
    **dict.fromkeys(["fresh_breakout", "liq_recent_collapse_breakout", "B_attrs_missing_breakout",
                     "B_attrs_missing_boring", "etf_boring", "boring_earnings_in_2d"], None),
}


@pytest.mark.parametrize("name", sm.ALL_CARD_CASES)
def test_decided_by(name: str) -> None:
    """Card disclosure: decided_by is the first data veto on wait_for_data, the first price-risk or liquidity
    veto on avoid, and the first cap on a watch whose base was actionable; otherwise null, so a no_setup card
    and the conditional market_closed never decide. RF: plan §1 change (1), §4 decided_by."""
    assert sm.card_case(name)["decided_by"] == DECIDED_BY[name]


@pytest.mark.parametrize("name", sm.SCAN_SCENARIOS)
async def test_pipeline_decided_by(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Card disclosure: the real analyze_trade_setup's decided_by on every GC-C4 frame.
    RF: plan §1 change (1), §4 decided_by."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])

    assert (await sm.trade_card())["decided_by"] == PIPELINE_DECIDED_BY[name]


# --- RA.18 and S-C8: card clauses -----------------------------------------------------------------

RA18 = sm.ScanScenario("RA.18", sm.BORING, sm.FLAT_VOLUME, earnings_unverifiable=True)


async def test_ra18_no_setup_card_lists_earnings_unverifiable(monkeypatch: pytest.MonkeyPatch) -> None:
    """RA.18: a card with no setup lists earnings_unverifiable, on RA-CH4's pure inputs (an unverifiable
    calendar, an events tool failure) and on the real analyze_trade_setup over a boring chart.
    RF: RA.18 card clause (disclosed on every outcome)."""
    sm.install(monkeypatch, RA18)
    pipeline = await sm.trade_card()
    monkeypatch.setattr(card_module, "detect_setup", lambda *args, **kwargs: None)

    pure = [build_card(events_data=UNVERIFIABLE), build_card(events_data={}, tool_failures=[EVENTS_FAILED])]

    assert [(card["action"], ids(card["blockers"])) for card in [*pure, pipeline]] == [
        ("no_setup", ["earnings_unverifiable"])] * 3


async def test_the_card_states_its_swing_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    """S-C8: the trade card states horizon swing, on SC-CH8's pullback and on every pure and pipeline card.
    RF: plan §4 horizon labels, card clause."""
    cards = await swing_cards(monkeypatch)

    assert pullback_card()["horizon"] == "swing"
    assert [name for name, card in cards.items() if card.get("horizon") != "swing"] == []
