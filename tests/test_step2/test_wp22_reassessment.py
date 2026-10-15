"""WP2.2: earliest_reassessment and analyze's next_review rule (plan §6; BL-D9 rule half, RA.17).

Annex: docs/plans/step2-v2-research/reassessment.md, "Corrected proposal" (rules R1-R9, worked examples E1-E29,
tests RA.2-RA.13, RA.17), as amended by plan §0 #11 (analyze only, per horizon: long_term.new_money carries its
own object; not on the card), #13 (outstanding in registry rank order) and §6 (presence: blocked_by non-empty and
map(candidate) in {buy, starter} at that horizon). The date is a day to re-run analyze, never permission to buy.
Events runs use the real tool over fixed provider reads (test_characterize_reassessment); analyze runs go through
tests/support/decision_matrix.py with the clock pinned to the run's evening; pure runs call earliest_reassessment
directly or decide() through dm.decide_on.
"""

import copy
import inspect
import re
from datetime import date
from itertools import permutations
from typing import Any

import pytest

from stock_analysis.tools.analyze import decision, gates, reassessment
from stock_analysis.tools.analyze.gates import (
    GATES,
    THESIS_CLEARS_WHEN,
    EarningsWindow,
    earnings_window,
)
from stock_analysis.utils.market_calendar import add_trading_days
from tests.support import decision_matrix as dm
from tests.test_step2 import test_characterize_reassessment as ra
from tests.test_step2.test_characterize_reassessment import (
    E1,
    E2,
    E3,
    E28,
    E29,
    EventsAt,
    Evidence,
    evening,
)
from tests.test_step2.test_wp21_earnings_clock import analyze_with_real_events
from tests.test_tools.test_earnings_verification import (
    FAILED_CALENDAR,
    FAILED_PAGE,
    calendar,
    page,
    window_info,
)
from tests.test_tools.test_trade_card import build_card

events_at = ra.events_at  # the real events tool over fixed provider reads, run at noon ET on a given day
D = date
REVERIFY = {"id": "earnings_reverify", "clears_when": GATES["earnings_unverifiable"].clears_when}
NEVER = "never: outside this tool's supported scope"
FALLBACK = "clock_only_no_holidays_fallback"
BANNED = re.compile(r"(?i)re-?entr|re-?enter|resum|\bbuy\b")


def dated(day: str, through: str, method: str = "calendar_static",
          outstanding: tuple[dict[str, str], ...] = (REVERIFY,)) -> dict[str, Any]:
    return {"date": day, "basis": "earnings_clear", "earnings_risk_through": through, "session_method": method,
            "outstanding": list(outstanding)}


def undated(*outstanding: dict[str, str], basis: str = "conditions_only") -> dict[str, Any]:
    return {"date": None, "basis": basis, "earnings_risk_through": None, "session_method": None,
            "outstanding": list(outstanding)}


D13 = dated("2026-10-13", "2026-10-12")  # the matrix's NEAR events (Mon Oct 12) run on Fri Oct 9


def er(result: dict[str, Any], horizon: str = "mid_term") -> Any:
    card = result["decision_card"]
    new_money = card["new_money"] if horizon == "mid_term" else card["long_term"]["new_money"]
    return new_money["earliest_reassessment"]


async def analyze(events_at: EventsAt, evidence: Evidence, today: date, scenario: str = "S2_buy_clean",
                  **overrides: Any) -> dict[str, Any]:
    return await dm.run(scenario, events=await events_at(evidence, today), now=evening(today), **overrides)


# Dated rows (annex worked examples): evidence, run date, date, earnings_risk_through, session_method.
DATED = {
    "E1": (E1, D(2026, 10, 12), "2026-10-16", "2026-10-15", "calendar_static"),
    "E2": (E2, D(2026, 10, 8), "2026-10-16", "2026-10-15", "calendar_static"),
    "E3": (E3, D(2026, 10, 8), "2026-10-16", "2026-10-15", "calendar_static"),
    "E4": (E2, D(2026, 10, 10), "2026-10-16", "2026-10-15", "calendar_static"),
    "E5": (Evidence(calendar("2026-10-07", "2026-10-15", is_estimate=True),
                    info=window_info("2026-10-09", "2026-10-20")), D(2026, 10, 8), "2026-10-21", "2026-10-20",
           "calendar_static"),
    "E6": (Evidence(calendar("2026-10-07", "2026-10-15"), page("2026-10-14")), D(2026, 10, 8),
           "2026-10-16", "2026-10-15", "calendar_static"),
    "E7": (Evidence(calendar("2026-10-22"), page("2026-10-20")), D(2026, 10, 16), "2026-10-23", "2026-10-22",
           "calendar_static"),
    "E8": (Evidence(calendar("2026-10-14"), page("2026-10-19")), D(2026, 10, 12), "2026-10-20", "2026-10-19",
           "calendar_static"),
    "E9": (Evidence(calendar("2026-10-16"), page("2026-10-23")), D(2026, 10, 12), "2026-10-26", "2026-10-23",
           "calendar_static"),
    "E10": (Evidence(calendar("2026-10-16")), D(2026, 10, 12), "2026-10-19", "2026-10-16", "calendar_static"),
    "E11": (Evidence(calendar("2026-11-25")), D(2026, 11, 23), "2026-11-27", "2026-11-25", "calendar_static"),
    "E12": (Evidence(calendar("2026-12-23")), D(2026, 12, 21), "2026-12-24", "2026-12-23", "calendar_static"),
    "E12b": (Evidence(calendar("2026-12-24")), D(2026, 12, 21), "2026-12-28", "2026-12-24", "calendar_static"),
    "E13": (Evidence(calendar("2027-07-02")), D(2027, 6, 30), "2027-07-06", "2027-07-02", "calendar_static"),
    "E14": (Evidence(calendar("2031-12-24")), D(2031, 12, 22), "2031-12-25", "2031-12-24", FALLBACK),
    "E14b": (Evidence(calendar("2030-12-31")), D(2030, 12, 27), "2031-01-01", "2030-12-31", FALLBACK),
    "E15": (E1, D(2026, 10, 15), "2026-10-16", "2026-10-15", "calendar_static"),
    "E24": (Evidence(calendar("2026-11-30")), D(2026, 11, 26), "2026-12-01", "2026-11-30", "calendar_static"),
    "E25": (Evidence(calendar("2026-10-12", "2026-10-17")), D(2026, 10, 13), "2026-10-19", "2026-10-17",
            "calendar_static"),
}
RA4_ROWS = ["E3", "E5", "E6", "E7", "E8", "E9", "E25"]  # unions, chains and assumed windows
RA5_ROWS = ["E4", "E10", "E11", "E12", "E12b", "E13", "E14", "E14b", "E15", "E24", "E25"]  # the NYSE calendar


async def test_codex_example_is_friday_oct_16(events_at: EventsAt) -> None:
    """RA.2: window Oct 7-15 run Thu Oct 8 gives Fri Oct 16 at both horizons (risk through Oct 15), never
    next_date + 1 (Oct 9); the gate still decides new money (wait). RF."""
    result = await analyze(events_at, E2, D(2026, 10, 8))

    for horizon in ("mid_term", "long_term"):
        assert er(result, horizon) == dated("2026-10-16", "2026-10-15"), horizon
    assert er(result)["date"] != "2026-10-09"
    assert (result["decision_card"]["new_money"]["action"], dm.gate_ids(result)) == ("wait", ["earnings_blackout"])


@pytest.mark.parametrize("name", DATED)
async def test_rerun_blocks_until_the_date(events_at: EventsAt, name: str) -> None:
    """RA.3: with identical evidence, the real events tool still fires the blackout on every session between
    today and the date, and not on the date (E28 and E29 are the documented limits). RF."""
    evidence, today, day, _, _ = DATED[name]
    clear = date.fromisoformat(er(await analyze(events_at, evidence, today))["date"])
    session = add_trading_days(today, 1)

    assert clear.isoformat() == day
    while session < clear:
        assert earnings_window((await events_at(evidence, session))["earnings"], session).fires, session
        session = add_trading_days(session, 1)
    assert not earnings_window((await events_at(evidence, clear))["earnings"], clear).fires


@pytest.mark.parametrize("name", sorted({*RA4_ROWS, *RA5_ROWS, "E1", "E2"}))
async def test_dated_example(events_at: EventsAt, name: str) -> None:
    """RA.4 (unions and chains: E3, E5-E9, E25) and RA.5 (calendar: E4, E10-E15, E24, E25; E14 and E14b fall
    back to weekdays): the full object at both horizons; outstanding is just earnings_reverify (E3, E5, E25 also
    fire earnings_unverifiable, shown as earnings_reverify). RF."""
    evidence, today, day, through, method = DATED[name]
    result = await analyze(events_at, evidence, today)

    for horizon in ("mid_term", "long_term"):
        assert er(result, horizon) == dated(day, through, method), horizon


async def test_after_the_event_only_reverify(events_at: EventsAt) -> None:
    """RA.6: at E1's date the calendar is stale: earnings_unverifiable alone blocks, with no date and only
    earnings_reverify outstanding. RF."""
    result = await analyze(events_at, E1, D(2026, 10, 16))

    assert dm.gate_ids(result) == ["earnings_unverifiable"]
    for horizon in ("mid_term", "long_term"):
        assert er(result, horizon) == undated(REVERIFY), horizon


@pytest.mark.parametrize("order", list(permutations(["earnings_blackout", "earnings_unverifiable",
                                                     "data_quality_critical", "falling_knife", "stale_data"])))
def test_outstanding_rank_order(order: tuple[str, ...]) -> None:
    """RA.7: whatever order the gates fire in, outstanding is in registry rank order (plan §0 #13), the event
    gates collapse into one earnings_reverify listed last, and the blackout dates the object. RF."""
    today = D(2026, 10, 12)
    window = earnings_window({"days_until": 3, "next_date": "2026-10-15"}, today)
    result = dm.decide_on({"mid_term": "buy", "long_term": "accumulate"},
                          [{"id": g, "reason": "r"} for g in order], support=None)
    blocked = result["new_money"]["blocked_by"]
    obj = reassessment.earliest_reassessment(blocked_by=blocked, admits=True, window=window, today=today,
                                             support=None)

    assert [o["id"] for o in obj["outstanding"]] == ["stale_data", "data_quality_critical", "falling_knife",
                                                     "earnings_reverify"]
    assert obj["date"] == "2026-10-16"


async def test_knife_only_is_undated(events_at: EventsAt) -> None:
    """RA.7 (E19): a knife alone on a long-term buy is undated, outstanding the knife's clears_when. RF."""
    result = await analyze(events_at, Evidence(calendar("2026-11-20")), D(2026, 10, 12),
                           scenario="C_quality_knife_dislocation")

    assert er(result) is None  # the mid-term candidate is avoid
    assert er(result, "long_term") == undated({"id": "falling_knife", "clears_when": GATES["falling_knife"].clears_when})


async def test_lower_bound_with_an_undated_veto() -> None:
    """RA.7 (E18 type): weak liquidity with the NEAR blackout keeps the earnings date as a lower bound and lists
    the veto before earnings_reverify. RF."""
    result = await dm.run("S9_weak_liquidity", events=dm.NEAR)
    expected = dated("2026-10-13", "2026-10-12", outstanding=(
        {"id": "weak_liquidity", "clears_when": GATES["weak_liquidity"].clears_when}, REVERIFY))

    assert er(result) == er(result, "long_term") == expected


@pytest.mark.parametrize("events", ["FAR", "NEAR"])
async def test_a_freshness_gate_is_conditions_only(events: str) -> None:
    """RA.7 (E20 type; FR "earliest null" superseded by plan §6): stale_data is an undated condition; with the
    NEAR blackout the date stays, stale_data first. RF."""
    stale = {**dm.at_price(dm.STRONG, 89.0), "daily_history": {
        "last_row_date": "2026-10-07", "last_valid_date": "2026-10-07", "last_row_valid": True,
        "completed_valid_date": "2026-10-07"}}
    result = await dm.run("S2_buy_clean", technicals=stale, events=getattr(dm, events))
    condition = {"id": "stale_data", "clears_when": GATES["stale_data"].clears_when}

    assert er(result) == (undated(condition) if events == "FAR" else
                          dated("2026-10-13", "2026-10-12", outstanding=(condition, REVERIFY)))


@pytest.mark.parametrize("case", ["E16", "E17", "tool_failure"])
async def test_unverifiable_only(events_at: EventsAt, case: str) -> None:
    """RA.8: an assumed future window (E16), every source failing (E17) or an events tool failure gives
    earnings_unverifiable alone: undated, earnings_reverify outstanding; an unevaluated blackout is never listed.
    RF."""
    if case == "tool_failure":
        result = await dm.run("S2_buy_clean", events=None)
    else:
        evidence = (Evidence(calendar("2026-10-20", "2026-10-24")) if case == "E16"
                    else Evidence(FAILED_CALENDAR, FAILED_PAGE, info_fails=True))
        result = await analyze(events_at, evidence, D(2026, 10, 12))

    assert dm.gate_ids(result) == ["earnings_unverifiable"]
    for horizon in ("mid_term", "long_term"):
        assert er(result, horizon) == undated(REVERIFY), horizon


async def test_permanent_support_lists_only_itself(events_at: EventsAt) -> None:
    """RA.9: an unsupported instrument or currency gives basis unsupported, no date, and only the permanent
    gate(s) outstanding (E21; GBp; a GBp-quoted ETF lists both; case G's long term omits the knife and
    blackout). RF."""
    etf = await analyze(events_at, E1, D(2026, 10, 12), scenario="ETF_strong")
    gbp = await dm.run("GBp_strong", events=dm.NEAR)
    gbp_etf = await dm.run("ETF_strong", summary={**dm.ETF, "currency": "GBp"})
    case_g = await dm.run("G_unsupported_currency_knife", events=dm.NEAR)

    assert (er(etf), er(etf, "long_term")) == (
        undated({"id": "unsupported_instrument", "clears_when": NEVER}, basis="unsupported"), None)
    assert dm.gate_ids(gbp_etf) == ["unsupported_instrument", "unsupported_currency"]
    assert (er(gbp_etf), er(gbp_etf, "long_term")) == (
        undated({"id": "unsupported_instrument", "clears_when": NEVER},
                {"id": "unsupported_currency", "clears_when": NEVER}, basis="unsupported"), None)
    for horizon in ("mid_term", "long_term"):
        assert er(gbp, horizon) == undated({"id": "unsupported_currency", "clears_when": NEVER}, basis="unsupported")
    assert (er(case_g), er(case_g, "long_term")) == (
        None, undated({"id": "unsupported_currency", "clears_when": NEVER}, basis="unsupported"))


async def test_unverified_support_is_dated(events_at: EventsAt) -> None:
    """RA.9 (E22): an unknown quote type is an ordinary undated condition beside the dated blackout. RF."""
    result = await analyze(events_at, E1, D(2026, 10, 12), summary={**dm.EQUITY, "quote_type": None})
    condition = {"id": "unsupported_instrument", "clears_when": "quote type verified as supported"}

    assert er(result) == dated("2026-10-16", "2026-10-15", outstanding=(condition, REVERIFY))
    assert er(result, "long_term") is None  # the unverified instrument's long-term candidate is insufficient_data


async def test_named_cases() -> None:
    """RA.10 (amended, with D2; plan §6): present per horizon iff a fired gate binds: S2 none; S5 and D2 none at
    mid term (avoid candidate) but dated at long term (buy -> wait); C2 dated at mid term only (long term
    hold_size_conservatively maps to wait, so its blackout is disclosed, not binding); S4 both. RF."""
    expected = {"S2_buy_clean": (None, None), "S5_earnings_gate_on_avoid": (None, D13),
                "D2_quality_poor_chart_earnings_3d": (None, D13), "C2_headwind_earnings_gate": (D13, None),
                "S4_earnings_gate_on_hold_or_add": (D13, D13)}
    for name, (mid, long) in expected.items():
        result = await dm.run(name)
        assert (er(result), er(result, "long_term")) == (mid, long), name
    d2 = (await dm.run("D2_quality_poor_chart_earnings_3d"))["decision_card"]
    assert (d2["new_money"]["action"], [b["id"] for b in d2["new_money"]["blocked_by"]]) == (
        "avoid", ["earnings_blackout"])
    assert d2["long_term"]["new_money"]["action"] == "wait"


PRESENT = {  # (scenario, horizon) with an object; every other matrix row is null at both horizons
    "FAR": {("S4_earnings_gate_on_hold_or_add", "mid_term"), ("S4_earnings_gate_on_hold_or_add", "long_term"),
            ("S5_earnings_gate_on_avoid", "long_term"), ("D2_quality_poor_chart_earnings_3d", "long_term"),
            ("C2_headwind_earnings_gate", "mid_term"), ("S9_weak_liquidity", "mid_term"),
            ("S9_weak_liquidity", "long_term"), ("S6_knife_dislocation", "long_term"),
            ("C_quality_knife_dislocation", "long_term"), ("E_poor_business_bullish_chart", "mid_term"),
            ("G_unsupported_currency_knife", "long_term"), ("ETF_strong", "mid_term"), ("GBp_strong", "mid_term"),
            ("GBp_strong", "long_term"), ("A7_near_52w_high_bullish", "mid_term"),
            ("H_burner_runway_2q", "mid_term"), ("H_burner_runway_2q", "long_term")},
}


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_presence_grid(name: str) -> None:
    """RA.10: over the matrix, the object is present exactly where a gate binds (the after table's rows). RF."""
    result = await dm.run(name)

    for horizon in ("mid_term", "long_term"):
        assert (er(result, horizon) is not None) is ((name, horizon) in PRESENT["FAR"]), horizon


@pytest.mark.parametrize("name", ["S4_earnings_gate_on_hold_or_add", "S2_buy_clean", "ETF_strong"])
async def test_placement_and_tiers(name: str) -> None:
    """RA.10: the key is in both new_money blocks (possibly null) and nowhere else on the card; every tier
    carries it as the full result does (plan §0 #11). RF."""
    result = await dm.run(name)
    card = result["decision_card"]

    assert "earliest_reassessment" in card["new_money"] and "earliest_reassessment" in card["long_term"]["new_money"]
    for block in (card, card["if_held"], card["long_term"], card["hard_gates"]):
        assert "earliest_reassessment" not in block
    for tier, shown in dm.tiers(result).items():
        assert er(shown) == er(result) and er(shown, "long_term") == er(result, "long_term"), tier


async def test_earliest_reads_analyze_clock() -> None:
    """RA.12 (earliest clause): with the wall clock frozen in 2027, analyze run at evening Oct 8 dates both the
    object and next_review from its own clock. RF."""
    result = await analyze_with_real_events(E2, evening(D(2026, 10, 8)))
    next_review = result["decision_card"]["next_review"]

    for horizon in ("mid_term", "long_term"):
        assert (er(result, horizon)["date"], er(result, horizon)["earnings_risk_through"]) == (
            "2026-10-16", "2026-10-15"), horizon
    assert next_review == {"date": "2026-10-16", "reason": "after earnings (Oct 15)"}
    assert "2027" not in repr(er(result)) + repr(er(result, "long_term")) + repr(next_review)


def test_no_clock_call() -> None:
    """RA.12: the reassessment and the clear-session helper never read the clock. RF."""
    for source in (inspect.getsource(reassessment), inspect.getsource(EarningsWindow.clear_session)):
        assert "now(" not in source and "today(" not in source


@pytest.mark.parametrize(("end", "expected"), [
    ("2026-10-15", (D(2026, 10, 16), D(2026, 10, 15))), ("2027-10-13", (D(2027, 10, 14), D(2027, 10, 13))),
    ("2027-10-14", None), ("9999-12-31", None)], ids=["this_week", "366_days", "367_days", "date_max"])
def test_a_far_window_is_undatable(end: str, expected: tuple[date, date] | None,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """RA risk 7: an interval ending more than 366 days after today is undatable: clear_session returns None
    within a bounded number of session steps (a date.max end no longer steps ~2M sessions); an interval ending
    366 days out, or this week, keeps its date. RF."""
    today, real, steps = D(2026, 10, 12), gates.add_trading_days, []

    def counted(day: date, n: int) -> date:
        steps.append(day)
        assert len(steps) <= 260, "unbounded session stepping"
        return real(day, n)

    monkeypatch.setattr(gates, "add_trading_days", counted)

    assert earnings_window({"windows": [{"start": "2026-10-12", "end": end}]}, today).clear_session(today) == expected


@pytest.mark.parametrize("earnings", [{"days_until": 3}, {"days_until": 3, "next_date": "2026-10-15"},
                                      {"days_until": 3, "next_date": "2026-10-13"}],
                         ids=["days_only", "events_run_oct_12", "events_run_oct_10"])
async def test_undatable(earnings: dict[str, Any]) -> None:
    """RA.13: a stub with days_until alone, or events whose run date (next_date - days_until) is not analyze's
    today, fires the blackout but is never dated: conditions_only with earnings_reverify. RF."""
    result = await dm.run("S2_buy_clean", events={"earnings": earnings})

    assert dm.gate_ids(result) == ["earnings_blackout"]
    for horizon in ("mid_term", "long_term"):
        assert er(result, horizon) == undated(REVERIFY), horizon


@pytest.mark.parametrize(("evidence", "today", "day", "through"), [
    (E2, D(2026, 10, 8), "2026-10-16", "Oct 15"), (E2, D(2026, 10, 10), "2026-10-16", "Oct 15"),
    (E1, D(2026, 10, 15), "2026-10-16", "Oct 15"),
    (Evidence(calendar("2026-10-12", "2026-10-17")), D(2026, 10, 13), "2026-10-19", "Oct 17")],
    ids=["E2", "E4", "E15", "E25"])
async def test_after_the_active_window(events_at: EventsAt, evidence: Evidence, today: date, day: str,
                                       through: str) -> None:
    """RA.17: inside an active earnings interval analyze's next_review is the first clear session, reason
    "after earnings (<last interval day>)", equal to the reassessment date; the trade card keeps the run date
    (CH clause: flipped by 2c). RF."""
    result = await analyze(events_at, evidence, today)
    card = build_card(events_data=await events_at(evidence, today))

    assert result["decision_card"]["next_review"] == {"date": day, "reason": f"after earnings ({through})"}
    assert er(result)["date"] == day
    assert card["next_review"] == {"date": today.isoformat(), "reason": "next earnings (~0 days)"}


async def test_an_undatable_active_window_is_never_reviewed_today(events_at: EventsAt) -> None:
    """RA.17: inside an active earnings interval that ends more than 366 days out (undatable, RA risk 7), the
    reassessment date stays null and next_review is the default re-evaluation with its cause, never today. RF."""
    result = await analyze(events_at, Evidence(calendar("2026-10-12", "2027-10-14")), D(2026, 10, 12))

    assert er(result)["date"] is None and "earnings_blackout" in dm.gate_ids(result)
    assert result["decision_card"]["next_review"] == {
        "date": "2026-11-11", "reason": "default 30-day re-evaluation (earnings window undatable)"}


@pytest.mark.parametrize("name", DATED)
async def test_after_today_on_every_dated_row(events_at: EventsAt, name: str) -> None:
    """RA.17: analyze's next_review is after today on every dated row. RF."""
    evidence, today, _, _, _ = DATED[name]

    assert (await analyze(events_at, evidence, today))["decision_card"]["next_review"]["date"] > today.isoformat()


async def test_unchanged_outside_a_window(events_at: EventsAt) -> None:
    """RA.17: before the event (E1 at Oct 12) and with earnings far off, next_review is the next earnings date. G."""
    before = await analyze(events_at, E1, D(2026, 10, 12))

    assert before["decision_card"]["next_review"] == {"date": "2026-10-15", "reason": "next earnings (~3 days)"}
    assert (await dm.run("S2_buy_clean"))["decision_card"]["next_review"] == {
        "date": "2026-11-18", "reason": "next earnings (~40 days)"}


async def test_known_limits(events_at: EventsAt) -> None:
    """E28, E29: the documented limits: E28 (hidden second page row: the date can be too early; the re-run on it
    blocks again) and E29 (info union: conservative; the one-session gap blocks new money on
    earnings_unverifiable). G: documented limits, the leading-edge anchor is Step 4."""
    assert er(await analyze(events_at, E28, D(2026, 10, 12))) == dated("2026-10-15", "2026-10-14")
    assert er(await analyze(events_at, E28, D(2026, 10, 15))) == dated("2026-10-20", "2026-10-19")
    assert er(await analyze(events_at, E29, D(2026, 10, 12))) == dated("2026-10-26", "2026-10-23")
    gap = await analyze(events_at, E29, D(2026, 10, 15))
    assert (dm.gate_ids(gap), er(gap)) == (["earnings_unverifiable"], undated(REVERIFY))


def strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in ("earliest_reassessment", "meta")}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [*value, *(s for v in value.values() for s in strings(v))]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return [value] if isinstance(value, str) else []


@pytest.mark.parametrize("events", ["FAR", "NEAR"])
@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_the_reassessment_never_changes_an_output(name: str, events: str,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """G-RA1: replacing earliest_reassessment with a sentinel changes no other field of the result (digest,
    card, snapshot hash): a date to check again never changes an action (plan §6). Where present, the object
    sits next to a gated, non-entry action with add_allowed false, lists at least one condition with unique ids
    (earnings_reverify last), its nullable fields agree, and no key or string reads as re-entry, resume or buy
    (RA.11, earliest clause). G."""
    real = await dm.run(name, events=getattr(dm, events))
    sentinel = {"date": "1999-01-01", "basis": "earnings_clear", "earnings_risk_through": None,
                "session_method": None, "outstanding": []}
    monkeypatch.setattr(decision, "earliest_reassessment", lambda **_: copy.deepcopy(sentinel))
    fake = await dm.run(name, events=getattr(dm, events))

    assert er(fake) == er(fake, "long_term") == sentinel
    assert dm.digest(strip(real)) == dm.digest(strip(fake))
    assert strip(real["decision_card"]) == strip(fake["decision_card"])
    assert real["watchlist_snapshot"]["snapshot_hash"] == fake["watchlist_snapshot"]["snapshot_hash"]
    card = real["decision_card"]
    for horizon, new_money in (("mid_term", card["new_money"]), ("long_term", card["long_term"]["new_money"])):
        obj = new_money["earliest_reassessment"]
        if obj is None:
            continue
        outstanding = [o["id"] for o in obj["outstanding"]]
        assert new_money["action"] not in ("buy", "starter") and new_money["blocked_by"], horizon
        assert outstanding and len(outstanding) == len(set(outstanding)), horizon
        assert "earnings_reverify" not in outstanding[:-1], horizon
        assert (obj["date"] is None) is (obj["earnings_risk_through"] is None) is (obj["session_method"] is None)
        assert (obj["date"] is None) is (obj["basis"] != "earnings_clear"), horizon
        assert obj["date"] is None or obj["date"] > "2026-10-09"
        assert [s for s in strings(obj) if BANNED.search(s)] == [], horizon
    if card["new_money"]["earliest_reassessment"] is not None:
        assert card["if_held"]["add_allowed"] is False


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_no_reentry_wording_at_any_tier(name: str) -> None:
    """RA.11 (the analyze surface, since WP2.3 including the summary's reassessment clauses, why and diagnostics;
    the guide, prompts and tool description are test_wp23_registry's): no key or string at any analyze tier
    matches re-entry or resume wording. G."""
    for tier, shown in dm.tiers(await dm.run(name, events=dm.NEAR)).items():
        assert not re.search(r"(?i)re-?entr|re-?enter|resum", repr(shown)), tier


def test_the_thesis_condition_names_its_component() -> None:
    """Thesis condition: the thesis gate's outstanding condition is clears_when_for the blocker, only the failing
    component (plan §4, PH-RF12b), so case E never asks for a runway figure. G."""
    window = earnings_window({}, D(2026, 10, 9))
    blocked = [{"id": "thesis_broken", "reason": "r", "codes": ["business_broken"], "class": "thesis",
                "effect": "cap"}]
    obj = reassessment.earliest_reassessment(blocked_by=blocked, admits=True, window=window, today=D(2026, 10, 9),
                                             support=None)

    assert obj == undated({"id": "thesis_broken", "clears_when": THESIS_CLEARS_WHEN["business_broken"]})
