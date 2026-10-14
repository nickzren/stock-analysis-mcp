"""WP2.1: analyze's gate inputs (plan §4, B1, B3; annex step2-v2-research/gate_contract.md, GC-R7, GC-R11, GC-R12).

Analyze reads the falling knife from its technicals for every instrument, fires earnings_unverifiable
through the card's B1 helper, discloses each fired gate once in registry order, and no longer gates
on a missing beta. No test asserts an action a new gate input produces: until decide() (WP2.2),
action_now is the interim hard-gate resolution. GC-R7's stage-1 clause lands with scan stage 1.
Analyze runs go through tests/support/decision_matrix.py and card runs through
tests/test_tools/test_trade_card.py; every run pins its clock.
"""

from typing import Any

import pytest

from stock_analysis.tools.analyze.gates import ANALYZE_GATE_IDS, is_falling_knife_technicals
from tests.support import decision_matrix as dm
from tests.support import swing_matrix as sm
from tests.test_step2 import test_characterize_gate_contract as gc
from tests.test_step2.test_gate_registry import ETF_SINK, US_SINK
from tests.test_tools.test_setup_detection import make_technicals
from tests.test_tools.test_trade_card import build_card

SUPPORTS: dict[str, dict[str, Any]] = {
    "us": dm.EQUITY, "etf": dm.ETF, "gbp": dm.GBP_EQUITY, "eur": {**dm.EQUITY, "currency": "EUR"},
    "unverified_instrument": {**dm.EQUITY, "quote_type": None},
    "unverified_currency": {**dm.EQUITY, "currency": None},
}
TECHS: dict[str, dict[str, Any]] = {"knife": dm.KNIFE, "crash": gc.CRASH, "strong": dm.STRONG,
                                    "poor": dm.POOR_NOT_KNIFE, "bearish": dm.BEARISH}
# a hold_or_add candidate, and a wait candidate where the gate is disclosed without binding
CHARTS: dict[str, dict[str, Any]] = {"strong": dm.STRONG, "crash": gc.CRASH}
KNIFE_RUNS = {f"{tech}-{support}": dm.Scenario("GC-R7", TECHS[tech], dm.PROFITABLE, dm.MODERATE,
                                               summary=SUPPORTS[support])
              for tech in TECHS for support in SUPPORTS}

UNVERIFIABLE_REASON = "earnings calendar unverifiable ({}) — event risk unverified"
# name: (events payload (None: the tool raises), the event gate ids analyze emits)
EVENT_CASES: dict[str, tuple[dict[str, Any] | None, list[str]]] = {
    "unverifiable": (sm.UNVERIFIABLE_EVENTS, ["earnings_unverifiable"]),
    "tool_failed": (None, ["earnings_unverifiable"]),
    "verified_date": ({"earnings": {**dm.FAR["earnings"], "verification": "verified_date"}}, []),
    "verified_none": ({"earnings": {"next_date": None, "days_until": None, "verification": "verified_none"}}, []),
    "stub": (dm.NEAR, ["earnings_blackout"]),  # no verification: B1 does not fire
}
EVENT_RUNS = {f"{case}-{chart}": dm.Scenario("GC-R12", CHARTS[chart], dm.PROFITABLE, dm.CALM, events=events)
              for case, (events, _) in EVENT_CASES.items() for chart in CHARTS}

BETA_RISKS: dict[str, dict[str, Any]] = {"beta_none": {**dm.CALM, "beta": {"value": None}},
                                          "beta_absent": {k: v for k, v in dm.CALM.items() if k != "beta"}}
BETA_RUNS = {name: dm.Scenario("B3", dm.STRONG, dm.PROFITABLE, risk) for name, risk in BETA_RISKS.items()}
RISK_FAILED = dm.Scenario("B3", dm.STRONG, dm.PROFITABLE, None)


def ids(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blockers]


def analyze_gates(result: dict[str, Any]) -> dict[str, Any]:
    gates: dict[str, Any] = result["decision_card"]["hard_gates"]
    return gates


# --- GC-R7: one knife predicate for every instrument -------------------------------------------


@pytest.mark.parametrize("name", KNIFE_RUNS)
async def test_knife_shared_predicate(name: str) -> None:
    """GC-R7: analyze fires falling_knife iff is_falling_knife_technicals does, unsupported and unverified
    instruments included, and checks agrees; the action is not asserted (interim until decide()).
    PD D-5a: the knife is evaluated for every instrument."""
    scenario = KNIFE_RUNS[name]
    knife, _ = is_falling_knife_technicals(scenario.technicals or {})
    result = await dm.run(scenario)

    assert ("falling_knife" in dm.gate_ids(result)) is knife
    assert analyze_gates(result)["checks"]["falling_knife"] is knife


@pytest.mark.parametrize("support", SUPPORTS)
def test_knife_shared_predicate_card(support: str) -> None:
    """GC-R7: the card lists falling_knife for a knife chart under every support class, and not for a sound one.
    PD D-5a: the card and analyze share is_falling_knife_technicals."""
    knife = build_card(technicals_data=sm.knife_technicals(), summary_data=SUPPORTS[support])
    sound = build_card(technicals_data=make_technicals(), summary_data=SUPPORTS[support])

    assert "falling_knife" in ids(knife["blockers"])
    assert "falling_knife" not in ids(sound["blockers"])


# --- GC-R12, event half: B1 in analyze -----------------------------------------------------------


@pytest.mark.parametrize("case", EVENT_CASES)
async def test_analyze_event_ids(case: str) -> None:
    """GC-R12: analyze fires earnings_unverifiable on verification unverifiable or an events tool failure,
    never on verified_date, verified_none or a stub without verification, on a hold_or_add and a wait
    candidate alike; checks agrees. PD D-6: B1 fires in analyze."""
    _, event_ids = EVENT_CASES[case]
    for chart in CHARTS:
        result = await dm.run(EVENT_RUNS[f"{case}-{chart}"])
        gates = analyze_gates(result)

        assert [g for g in dm.gate_ids(result) if g.startswith("earnings_")] == event_ids, chart
        assert gates["checks"]["earnings_unverifiable"] is ("earnings_unverifiable" in event_ids), chart
        if case == "tool_failed":
            assert [g["reason"] for g in gates["blocking"] if g["id"] == "earnings_unverifiable"] == [
                UNVERIFIABLE_REASON.format("tool fetch")], chart


def test_unverifiable_reason_names_the_cause() -> None:
    """GC-R12: an unverifiable date on an assumed estimate window says so in analyze's reason.
    PD D-6: analyze and the card share the B1 reason."""
    events = {"earnings": {"verification": "unverifiable", "windows": [{"window_basis": "assumed"}]}}
    gates = gc.hard_gates({"events_data": events})

    assert [g["reason"] for g in gates["blocking"] if g["id"] == "earnings_unverifiable"] == [
        UNVERIFIABLE_REASON.format("estimate window assumed")]


@pytest.mark.parametrize(("events", "tool_failures"), [
    (sm.UNVERIFIABLE_EVENTS, []), ({}, [{"tool": "events_calendar", "error": "x"}])], ids=["unverifiable", "tool_failed"])
def test_event_blocker_parity_with_card(events: dict[str, Any], tool_failures: list[dict[str, Any]]) -> None:
    """GC-R12: analyze's earnings_unverifiable blocker is the default-setup card's, id and reason.
    PD D-6: one B1 predicate (gates.check_earnings_unverifiable)."""
    gates = gc.hard_gates({"events_data": events, "data_quality": {"tool_failures": tool_failures}})
    card = build_card(events_data=events, tool_failures=tool_failures)

    def unverifiable(blockers: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [(b["id"], b["reason"]) for b in blockers if b["id"] == "earnings_unverifiable"]

    assert len(unverifiable(card["blockers"])) == 1
    assert unverifiable(gates["blocking"]) == unverifiable(card["blockers"])


# name: (an earnings payload the events tool never emits, the event gate ids analyze and the card list)
MALFORMED_EARNINGS: dict[str, tuple[dict[str, Any], list[str]]] = {
    "window_not_a_dict": ({"verification": "unverifiable", "windows": [None]}, ["earnings_unverifiable"]),
    # no events date that far back: 1e12 days is outside the blackout's reach, so only B1 fires
    "days_until_1e12": ({"days_until": 1e12, "next_date": "2026-10-12", "verification": "unverifiable"},
                        ["earnings_unverifiable"]),
    # the events date falls before year 1: the blackout still fires on the count
    "events_date_before_year_1": ({"days_until": 3, "next_date": "0001-01-02", "verification": "verified_date"},
                                  ["earnings_blackout"]),
}


@pytest.mark.parametrize("case", MALFORMED_EARNINGS)
async def test_malformed_earnings_fail_closed(case: str) -> None:
    """GC-R12: a non-dict window or a days_until no calendar date can carry never raises in analyze or the
    card; earnings_unverifiable still fires on an unverifiable calendar, and the blackout fires on its day
    count alone. RF: fail closed (plan §4, B1)."""
    earnings, event_ids = MALFORMED_EARNINGS[case]
    result = await dm.run("S1_hold_or_add_clean", events={"earnings": earnings})
    card = build_card(events_data={"earnings": earnings})

    assert [g for g in dm.gate_ids(result) if g.startswith("earnings_")] == event_ids
    assert [g for g in ids(card["blockers"]) if g.startswith("earnings_")] == event_ids


# --- GC-R11, analyze: one entry per id, in registry order ----------------------------------------

ANALYZE_RUNS: dict[str, dm.Scenario] = {
    **dm.SCENARIOS, **{name: row[0] for name, row in gc.ANALYZE_PIPELINE_ROWS.items()},
    **KNIFE_RUNS, **EVENT_RUNS, **BETA_RUNS, "risk_failed": RISK_FAILED, "us_sink": US_SINK, "etf_sink": ETF_SINK,
}


async def test_blocker_ids_unique() -> None:
    """GC-R11: duplicate blockers merge into one entry (reasons joined, codes unioned); every analyze run
    discloses each gate id once, in gates.ANALYZE_GATE_IDS order; the sinks list every gate that fired.
    RF: plan §4 "unique ids per output, in rank order", R2.6(a)."""
    stale = [{"id": "stale_data", "reason": "a", "codes": ["daily_close_stale"]},
             {"id": "stale_data", "reason": "b", "codes": ["daily_history_stale"]}]
    assert gc.hard_gates({"freshness_blockers": stale})["blocking"] == [
        {"id": "stale_data", "reason": "a; b", "codes": ["daily_close_stale", "daily_history_stale"]}]

    emitted: dict[str, list[str]] = {}
    for name, scenario in ANALYZE_RUNS.items():
        result = await dm.run(scenario)
        blocking = dm.gate_ids(result)
        assessment = result["dip_assessment"].get("assessment") or {}
        for gate_ids in (blocking, result["policy_action"].get("support_gates") or [],
                         assessment.get("support_gates") or []):
            assert len(gate_ids) == len(set(gate_ids)), name
        assert blocking == [g for g in ANALYZE_GATE_IDS if g in blocking], name
        emitted[name] = blocking

    assert emitted["us_sink"] == ["data_quality_critical", "liquidity_missing", "falling_knife",
                                  "earnings_unverifiable"]
    assert emitted["etf_sink"] == ["data_quality_critical", "falling_knife", "unsupported_instrument",
                                   "earnings_unverifiable"]


# --- B3: a missing beta is disclosed, not gated ----------------------------------------------------


@pytest.mark.parametrize("risk", BETA_RISKS)
async def test_a_missing_beta_alone_is_disclosed_not_gated(risk: str) -> None:
    """B3: with beta None or absent, data_quality_critical does not fire, missing_critical is empty and
    data_gaps lists beta_unavailable at every tier; completeness still counts beta, and the action is the
    beta-present run's. RF: plan B3 (beta is not a required input)."""
    result = await dm.run(BETA_RUNS[risk])
    with_beta = await dm.run(BETA_RUNS[risk], risk=dm.CALM)
    data_quality = result["data_quality"]

    assert "data_quality_critical" not in dm.gate_ids(result)
    assert analyze_gates(result)["checks"]["data_quality_critical"] is False
    assert data_quality["missing_critical"] == []
    assert {tier: "beta_unavailable" in shown["data_quality"]["data_gaps"]
            for tier, shown in dm.tiers(result).items()} == dict.fromkeys(dm.TIERS, True)
    assert (data_quality["completeness"], result["risk_summary"]["beta"]) == (0.8, None)
    assert result["decision_card"]["action_now"] == with_beta["decision_card"]["action_now"]


async def test_a_risk_metrics_failure_still_fires_data_quality() -> None:
    """B3: a failed risk_metrics still fires data_quality_critical, as a critical tool failure only, with
    liquidity_missing, and discloses beta_unavailable. RF: plan B3 (the tool stays critical)."""
    result = await dm.run(RISK_FAILED)
    gates = analyze_gates(result)

    assert [(g["id"], g["reason"]) for g in gates["blocking"] if g["id"] == "data_quality_critical"] == [
        ("data_quality_critical", "critical data unavailable: critical_tool_failed=risk_metrics")]
    assert "liquidity_missing" in dm.gate_ids(result)
    assert "beta_unavailable" in result["data_quality"]["data_gaps"]


# --- G-GI1: gate inputs never move the candidate or the scores ------------------------------------


@pytest.mark.parametrize("chart", CHARTS)
async def test_gate_inputs_never_move_the_candidate_or_scores(chart: str) -> None:
    """G-GI1: the policy codes, verdict score and tilt and the dip type are the same with B1's inputs
    (events failing or unverifiable) and B3's (beta None) as without them.
    G: R2.6(h) in 2.1; new gate inputs only gate, they never feed the candidate."""
    def candidate(result: dict[str, Any]) -> tuple[Any, ...]:
        policy, verdict = result["policy_action"], result["verdict"]
        return (policy["mid_term"], policy["long_term"], verdict["score"], verdict["tilt"],
                (result["dip_assessment"].get("dip_classification") or {}).get("type"))

    base = dm.Scenario("G-GI1", CHARTS[chart], dm.PROFITABLE, dm.CALM, events=dm.FAR)
    expected = candidate(await dm.run(base))
    for overrides in ({"events": None}, {"events": sm.UNVERIFIABLE_EVENTS}, {"risk": BETA_RISKS["beta_none"]}):
        assert candidate(await dm.run(base, **overrides)) == expected, overrides
