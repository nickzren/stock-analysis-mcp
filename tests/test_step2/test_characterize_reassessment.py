"""Step 2 WP2.0: characterize the earnings evidence behind `earliest_reassessment`.

Annex `docs/plans/step2-v2-research/reassessment.md` ("Tests", as amended by plan §0 #11).
Records how the events tool, the blackout gate, the NYSE calendar, analyze and the trade card
behave around an earnings blackout on 0609fde, before WP2.1/2.2 add the shared earnings helper
and the reassessment date. Ported from the research probes probe_events_semantics.py,
probe_analyze_window.py and proto_reassessment.py (reassessment/), and probe_limits.py,
probe_presence.py and prop_clear_session.py (reassessment_verify/).

Category G holds before and after 2a; category CH records today's behavior and names the WP
that flips it. `shown_intervals`, `covering` and `clear_session` are test-local stand-ins for
the annex rule (union of the shown intervals, then step NYSE sessions), not product code.
"""

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pytest

import stock_analysis.tools.events as events
import stock_analysis.tools.trade_setup.card as card_module
from stock_analysis.tools.analyze.gates import EARNINGS_BLACKOUT_DAYS, check_earnings_blackout
from stock_analysis.utils.market_calendar import add_trading_days, calendar_covers
from tests.support import decision_matrix as dm
from tests.support.fake_source import FakeSource
from tests.support.fake_ticker import FakeTicker
from tests.test_tools.test_earnings_verification import (
    FAILED_CALENDAR,
    FAILED_PAGE,
    NO_CALENDAR_DATES,
    NO_PAGE_DATES,
    STALE_CALENDAR,
    UNVERIFIED_PAGE,
    calendar,
    epoch,
    page,
    raw_calendar,
    window_info,
)
from tests.test_tools.test_trade_card import blocker_ids, build_card

Interval = tuple[date, date]


@dataclass(frozen=True)
class Evidence:
    """The provider reads behind one events run; unchanged between re-runs."""

    cal: FakeSource
    pg: FakeSource = NO_PAGE_DATES
    info: dict[str, Any] = field(default_factory=dict)
    info_fails: bool = False


# Worked examples (annex table), with the run date each is evaluated on.
E1 = Evidence(calendar("2026-10-15"))  # verified Thu Oct 15; E15 is the same evidence run on Oct 15
E2 = Evidence(calendar("2026-10-07", "2026-10-15", is_estimate=True),
              info=window_info("2026-10-07", "2026-10-15"))  # confirmed window (Codex's example)
E3 = Evidence(calendar("2026-10-07", "2026-10-15"))  # the same window, assumed
S5 = Evidence(calendar("2026-10-10", "2026-10-14"), page("2026-10-30"))  # a later page row stays visible
E28 = Evidence(NO_CALENDAR_DATES, page("2026-10-19", "2026-10-14"))  # limit L1: hidden second page row
E29 = Evidence(NO_CALENDAR_DATES, info={**window_info("2026-10-21", "2026-10-23"),
                                        "earningsTimestamp": epoch("2026-10-14")})  # limit L2: info union

EventsAt = Callable[..., Awaitable[dict[str, Any]]]


@pytest.fixture
def events_at(monkeypatch: pytest.MonkeyPatch) -> EventsAt:
    """The real events tool over fixed provider reads, run at noon ET on ``today``."""

    async def run(evidence: Evidence, today: date) -> dict[str, Any]:
        async def fake_fetch_ticker(symbol: str) -> FakeTicker:
            return FakeTicker(symbol, splits=None)

        async def fake_fetch_info(symbol: str) -> dict[str, Any]:
            if evidence.info_fails:
                raise ConnectionError("info down")
            return evidence.info

        monkeypatch.setattr(events, "fetch_ticker", fake_fetch_ticker)
        monkeypatch.setattr(events, "fetch_info", fake_fetch_info)
        monkeypatch.setattr(events, "read_calendar_events", evidence.cal)
        monkeypatch.setattr(events, "read_earnings_page", evidence.pg)
        now = dm.ET.localize(datetime(today.year, today.month, today.day, 12, 0))
        return await events.events_calendar("ACME", _now=now)

    return run


def evening(day: date) -> datetime:
    """An off-session analyze clock on ``day``, like the decision matrix's own."""
    return dm.ET.localize(datetime(day.year, day.month, day.day, 20, 0))


def fires(earnings: dict[str, Any]) -> bool:
    return check_earnings_blackout({"earnings": earnings})[0]


def shown_intervals(earnings: dict[str, Any]) -> set[Interval]:
    """Every earnings interval the output shows: windows, verified dates, and next_date."""
    out = {(date.fromisoformat(w["start"]), date.fromisoformat(w["end"])) for w in earnings["windows"]}
    out |= {(date.fromisoformat(d),) * 2 for d in earnings["candidate_dates"].values()}
    if earnings["next_date"]:
        next_day = date.fromisoformat(earnings["next_date"])
        if not any(start <= next_day <= end for start, end in out):
            out.add((next_day, next_day))
    return out


def covering(intervals: set[Interval], day: date) -> list[Interval]:
    """The intervals that black out ``day``: not over, and starting within the blackout reach."""
    return [(start, end) for start, end in intervals
            if end >= day and (max(start, day) - day).days <= EARNINGS_BLACKOUT_DAYS]


def clear_session(intervals: set[Interval], today: date) -> tuple[date, date]:
    """The first session after ``today`` that no interval covers, and the latest risk date passed."""
    through = max(end for _, end in covering(intervals, today))
    day = add_trading_days(today, 1)
    while hits := covering(intervals, day):
        through = max(through, *(end for _, end in hits))
        day = add_trading_days(day, 1)
    return day, through


class TestEventsSemantics:
    async def test_c1_active_window_reads_day_zero_until_it_ends(self, events_at: EventsAt) -> None:
        """RA-C1: E2 re-run on each day Oct 8-15 gives next_date = run date, days_until 0 and the
        blackout; Oct 16 is clear and unverifiable. G: events is unchanged in 2a."""
        for offset in range(8):
            day = date(2026, 10, 8) + timedelta(days=offset)
            earnings = (await events_at(E2, day))["earnings"]

            assert (earnings["next_date"], earnings["days_until"], fires(earnings)) == (day.isoformat(), 0, True)

        after = (await events_at(E2, date(2026, 10, 16)))["earnings"]
        assert (fires(after), after["verification"]) == (False, "unverifiable")

    async def test_c2_next_date_plus_one_is_still_blacked_out(self, events_at: EventsAt) -> None:
        """RA-C2: the v1 counterexample. E2 on Oct 8 gives next_date Oct 8, and a re-run on
        next_date + 1 (Oct 9) still fires. G: events and the gate predicate are unchanged in 2a."""
        earnings = (await events_at(E2, date(2026, 10, 8)))["earnings"]
        day_after = date.fromisoformat(earnings["next_date"]) + timedelta(days=1)

        assert fires((await events_at(E2, day_after))["earnings"])

    @pytest.mark.parametrize(
        ("evidence", "today", "clear", "verification", "next_date"),
        [(E1, date(2026, 10, 12), date(2026, 10, 16), "unverifiable", None),
         (E2, date(2026, 10, 8), date(2026, 10, 16), "unverifiable", None),
         (E3, date(2026, 10, 8), date(2026, 10, 16), "unverifiable", None),
         (S5, date(2026, 10, 8), date(2026, 10, 15), "verified_date", "2026-10-30")],
        ids=["E1", "E2", "E3", "S5_later_page_row"],
    )
    async def test_c4_verification_at_the_first_clear_session(
        self, events_at: EventsAt, evidence: Evidence, today: date, clear: date, verification: str,
        next_date: str | None,
    ) -> None:
        """RA-C4: the R5 premise. With unchanged sources, the re-run at the first clear session
        reads unverifiable for E1, E2 and E3, and verified_date when a later page row is
        visible (S5). G: events is unchanged in 2a."""
        earnings = (await events_at(evidence, today))["earnings"]
        assert clear_session(shown_intervals(earnings), today)[0] == clear

        rerun = (await events_at(evidence, clear))["earnings"]
        assert (fires(rerun), rerun["verification"], rerun["next_date"]) == (False, verification, next_date)


# Every events input of TestTruthTable, TestWindows and TestInfo
# (tests/test_tools/test_earnings_verification.py), all run on its date, Thu Oct 8 2026.
EVIDENCE_ROWS = [
    # TestTruthTable.test_row (rows 1-10; row 9 also feeds test_row_9_discloses_the_disagreement)
    pytest.param(Evidence(calendar("2026-10-20")), id="truth_1_point"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info=window_info("2026-10-20", "2026-10-24")), id="truth_1_confirmed_window"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24")), id="truth_1b_assumed_future_window"),
    pytest.param(Evidence(calendar("2026-10-07", "2026-10-15", is_estimate=True),
                          info=window_info("2026-10-07", "2026-10-15")), id="truth_2a_confirmed_active_window"),
    pytest.param(Evidence(calendar("2026-10-07", "2026-10-15")), id="truth_2b_assumed_active_window"),
    pytest.param(Evidence(FAILED_CALENDAR, page("2026-10-20")), id="truth_3"),
    pytest.param(Evidence(FAILED_CALENDAR), id="truth_4"),
    pytest.param(Evidence(STALE_CALENDAR, page("2026-10-09")), id="truth_5"),
    pytest.param(Evidence(STALE_CALENDAR), id="truth_6"),
    pytest.param(Evidence(NO_CALENDAR_DATES), id="truth_7"),
    pytest.param(Evidence(NO_CALENDAR_DATES, UNVERIFIED_PAGE), id="truth_8"),
    pytest.param(Evidence(calendar("2026-10-22"), page("2026-10-20")), id="truth_9"),
    pytest.param(Evidence(FAILED_CALENDAR, FAILED_PAGE, info_fails=True), id="truth_10"),
    # TestWindows (r2_4 reuses truth rows 2a/2b; r2_5's future window is truth row 1b)
    pytest.param(Evidence(calendar("2026-10-24", "2026-10-20")), id="windows_malformed_reversed"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-22", "2026-10-24")), id="windows_malformed_three"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info=window_info("2026-10-20", "2026-10-24")), id="windows_basis_consistent"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True)), id="windows_basis_range_absent"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=False)),
                 id="windows_basis_contradicts_estimate_flag"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info=window_info("2026-10-21", "2026-10-24")), id="windows_basis_contradicts_range"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info=window_info("2026-10-20", "2026-10-24", is_estimate=False)),
                 id="windows_basis_flags_disagree"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info={"earningsTimestampStart": epoch("2026-10-20"), "earningsTimestampEnd": "soon"}),
                 id="windows_basis_malformed_range"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info={**window_info("2026-10-20", "2026-10-24", is_estimate=None),
                                "isEarningsDateEstimate": "false"}), id="windows_basis_info_flag_string"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info={**window_info("2026-10-20", "2026-10-24", is_estimate=None),
                                "isEarningsDateEstimate": 1}), id="windows_basis_info_flag_int"),
    pytest.param(Evidence(FakeSource.ok("calendar", {"earnings": {
        "earningsDate": [epoch("2026-10-20"), epoch("2026-10-24")], "isEarningsDateEstimate": "true"}}),
        info=window_info("2026-10-20", "2026-10-24")), id="windows_basis_calendar_flag_string"),
    pytest.param(Evidence(raw_calendar(epoch("2026-10-20", 16), epoch("2026-10-20", 8))),
                 id="windows_reversed_epochs_same_day"),
    pytest.param(Evidence(raw_calendar(epoch("2026-10-24"), epoch("2026-10-20"))),
                 id="windows_reversed_epochs_different_days"),
    pytest.param(Evidence(raw_calendar(epoch("2026-10-20", 8), epoch("2026-10-20", 16))),
                 id="windows_two_entries_one_day"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24"), info={"earningsTimestamp": epoch("2026-10-20")}),
                 id="windows_info_echo_assumed"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=False),
                          info={**window_info("2026-10-20", "2026-10-24", is_estimate=False),
                                "earningsTimestamp": epoch("2026-10-20")}), id="windows_info_echo_conflicting"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24"), info={"earningsTimestamp": epoch("2026-10-18")}),
                 id="windows_info_date_outside"),
    pytest.param(Evidence(calendar("2026-10-07", "2026-10-15"), page("2026-10-14")),
                 id="windows_independent_date_resolves_assumed"),
    pytest.param(Evidence(calendar("2026-10-10", "2026-10-14"), page("2026-10-30")),
                 id="windows_card_keeps_uncertainty"),
    # TestInfo
    pytest.param(Evidence(NO_CALENDAR_DATES, info={"earningsTimestamp": epoch("2026-10-20"),
                                                   "isEarningsDateEstimate": True}), id="info_timestamp_source"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info={"earningsTimestamp": "soon"}), id="info_invalid_text"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info={"earningsTimestamp": 10**400}), id="info_invalid_overflow"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info={**window_info("2026-10-07", "2026-10-15"),
                                                   "earningsTimestamp": epoch("2026-10-07")}),
                 id="info_window_today_calendar_none"),
    pytest.param(Evidence(FAILED_CALENDAR, info={**window_info("2026-10-07", "2026-10-15"),
                                                 "earningsTimestamp": epoch("2026-10-07")}),
                 id="info_window_today_calendar_failed"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info=window_info("2026-10-07", "2026-10-20")),
                 id="info_range_without_timestamp"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True),
                          info={**window_info("2026-10-07", "2026-10-15"), "earningsTimestamp": epoch("2026-10-07")}),
                 id="info_window_drives_blackout"),
    pytest.param(Evidence(calendar("2026-10-20", "2026-10-24", is_estimate=True), page("2026-10-12"),
                          info=window_info("2026-10-20", "2026-10-24")), id="info_point_date_driver"),
    pytest.param(Evidence(NO_CALENDAR_DATES, page("2026-10-30"),
                          info={**window_info("2026-10-07", "2026-10-20"), "earningsTimestamp": epoch("2026-10-20")}),
                 id="info_range_start_wins"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info={**window_info("2026-10-27", "2026-11-02"),
                                                   "earningsTimestamp": epoch("2026-07-29")}),
                 id="info_last_report_timestamp"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info={"earningsTimestampStart": epoch("2026-10-20"),
                                                   "earningsTimestamp": epoch("2026-10-20")}),
                 id="info_malformed_end_missing"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info={"earningsTimestampStart": "soon",
                                                   "earningsTimestampEnd": epoch("2026-10-24"),
                                                   "earningsTimestamp": epoch("2026-10-20")}),
                 id="info_malformed_start_not_epoch"),
    pytest.param(Evidence(NO_CALENDAR_DATES, info={"earningsTimestampStart": epoch("2026-10-24"),
                                                   "earningsTimestampEnd": epoch("2026-10-20"),
                                                   "earningsTimestamp": epoch("2026-10-20")}),
                 id="info_malformed_reversed"),
    pytest.param(Evidence(FAILED_CALENDAR, info={**window_info("2026-10-20", "2026-10-24"),
                                                 "earningsTimestamp": epoch("2026-10-20")}),
                 id="info_window_alone"),
]


def random_evidence(rng: random.Random, base: date) -> Evidence:
    """One random mix of calendar, page and info evidence around ``base`` (prop_clear_session.py)."""

    def day(offset: int) -> str:
        return (base + timedelta(days=offset)).isoformat()

    kind = rng.choice(["none", "single", "window", "window_flag"])
    if kind == "none":
        cal = NO_CALENDAR_DATES
    elif kind == "single":
        cal = calendar(day(rng.randint(-3, 25)))
    else:
        start = rng.randint(-6, 20)
        end = start + rng.randint(1, 10)
        cal = calendar(day(start), day(end), is_estimate=True if kind == "window_flag" else None)
    future_rows = sorted({day(rng.randint(-4, 28)) for _ in range(rng.choice([0, 1, 1, 2]))}, reverse=True)
    pg = page(*future_rows, "2026-07-29", reported={"2026-07-29": 1.1}) if future_rows else NO_PAGE_DATES
    info: dict[str, Any] = {}
    info_kind = rng.choice(["none", "timestamp", "range", "both"])
    if info_kind in ("timestamp", "both"):
        info["earningsTimestamp"] = epoch(day(rng.randint(-5, 25)))
    if info_kind in ("range", "both"):
        start = rng.randint(-6, 20)
        end = start + rng.randint(0, 10)
        info.update(window_info(day(start), day(end), is_estimate=rng.choice([True, False, None])))
    return Evidence(cal, pg, info)


class TestShownIntervalsReproduceTheGate:
    @pytest.mark.parametrize("evidence", EVIDENCE_ROWS)
    async def test_c3_on_every_earnings_verification_row(self, events_at: EventsAt, evidence: Evidence) -> None:
        """RA-C3a: covers(today) over windows, verified dates and next_date equals
        check_earnings_blackout on every TestTruthTable, TestWindows and TestInfo input.
        G: neither events nor the gate predicate changes in 2a."""
        today = date(2026, 10, 8)
        earnings = (await events_at(evidence, today))["earnings"]

        assert bool(covering(shown_intervals(earnings), today)) is fires(earnings)

    async def test_c3_on_a_seeded_random_corpus(self, events_at: EventsAt) -> None:
        """RA-C3b: the same equality on 300 seeded random evidence mixes and run dates around
        Thanksgiving 2026. G: neither events nor the gate predicate changes in 2a."""
        rng = random.Random(7)
        base = date(2026, 11, 16)
        mismatches: list[tuple[date, dict[str, Any]]] = []
        fired = 0
        for _ in range(300):
            evidence = random_evidence(rng, base)
            today = base + timedelta(days=rng.randint(-2, 14))
            earnings = (await events_at(evidence, today))["earnings"]
            fired += fires(earnings)
            if bool(covering(shown_intervals(earnings), today)) is not fires(earnings):
                mismatches.append((today, earnings))

        assert mismatches == []
        assert 100 <= fired <= 200  # 156 of the 300 fire: both outcomes are well covered


class TestGateAndCalendar:
    @pytest.mark.parametrize(("earnings", "fired"), [({"days_until": 3}, True), ({"days_until": 5}, True),
                                                     ({"days_until": 6}, False), ({}, False)],
                             ids=["days_until_3", "days_until_5", "days_until_6", "empty"])
    async def test_c5_degraded_stubs_fire_on_days_until_alone(self, earnings: dict[str, Any], fired: bool) -> None:
        """RA-C5: a stub with only days_until fires on 0 <= days_until <= 5, in the gate and in
        analyze's hard gates; days_until 6 and an empty stub do not. G: the predicate is kept."""
        hard_gates = (await dm.run("S2_buy_clean", events={"earnings": earnings}))["decision_card"]["hard_gates"]

        assert fires(earnings) is fired
        assert ("earnings_blackout" in [g["id"] for g in hard_gates["blocking"]]) is fired

    @pytest.mark.parametrize(
        ("start", "after"),
        [(date(2026, 11, 25), date(2026, 11, 27)),  # Thanksgiving; Nov 27 is an early close
         (date(2026, 12, 23), date(2026, 12, 24)),  # Dec 24 is an early close
         (date(2027, 7, 2), date(2027, 7, 6)),  # weekend, then the observed Jul 5 holiday
         (date(2030, 12, 31), date(2031, 1, 1))],  # past calendar coverage: weekday-only fallback
        ids=["thanksgiving", "christmas_eve", "independence_day_observed", "coverage_boundary"],
    )
    def test_c6_next_session_steps_over_closures(self, start: date, after: date) -> None:
        """RA-C6: add_trading_days(+1) keeps early closes, skips full holidays and weekends, and
        past 2030 falls back to weekdays (Jan 1 2031, a real holiday). G: the calendar is
        unchanged in 2a."""
        assert add_trading_days(start, 1) == after
        assert calendar_covers(after) is (after.year <= 2030)

    @pytest.mark.parametrize(
        ("evidence", "shown_date", "risk_through", "blocked"),
        [(E28, date(2026, 10, 15), date(2026, 10, 14), ["10-13", "10-14", "10-15", "10-16", "10-19"]),
         (E29, date(2026, 10, 26), date(2026, 10, 23),
          ["10-13", "10-14", "10-16", "10-19", "10-20", "10-21", "10-22", "10-23"])],
        ids=["E28_hidden_page_row", "E29_info_union"],
    )
    async def test_c7_known_limits_of_the_shown_evidence(
        self, events_at: EventsAt, evidence: Evidence, shown_date: date, risk_through: date, blocked: list[str]
    ) -> None:
        """RA-C7: run Mon Oct 12. E28: the shown evidence clears on Oct 15, but the real re-run
        blocks through Oct 19 (a hidden later page row). E29: the shown window clears on Oct 26,
        but the real tool clears on Oct 15 for one session, then blocks Oct 16-23.
        G: documented limits; events is unchanged in 2a."""
        today = date(2026, 10, 12)
        earnings = (await events_at(evidence, today))["earnings"]
        assert clear_session(shown_intervals(earnings), today) == (shown_date, risk_through)

        sessions = [add_trading_days(today, n) for n in range(1, 11)]  # Oct 13 to Oct 26
        reruns = [(await events_at(evidence, day))["earnings"] for day in sessions]
        assert [f"{day:%m-%d}" for day, rerun in zip(sessions, reruns, strict=True) if fires(rerun)] == blocked


BLACKOUT = {"earnings": {"days_until": 3, "next_date": "2026-10-15", "verification": "verified_date"}}
UNVERIFIABLE = {"earnings": {"days_until": 40, "next_date": "2026-04-19", "verification": "unverifiable",
                             "sources_failed": ["calendar"]}}
HOLD = dm.Scenario("RA C8 hold", dm.at_price(dm.STRONG, 85.0, above_sma200=False, above_sma50=False,
                                             above_sma20=False), dm.PROFITABLE, dm.CALM, events=dm.NEAR)


async def with_and_without_blackout(scenario: dm.Scenario) -> tuple[dict[str, Any], dict[str, Any]]:
    """The scenario's decision card with a 3-day blackout, and with earnings 40 days out."""
    gated = (await dm.run(scenario, events=dm.NEAR))["decision_card"]
    clear = (await dm.run(scenario, events=dm.FAR))["decision_card"]
    return gated, clear


def gate_ids(card: dict[str, Any]) -> list[str]:
    return [g["id"] for g in card["hard_gates"]["blocking"]]


class TestGatesThatDoNotBind:
    @pytest.mark.parametrize(("scenario", "action_now"),
                             [(dm.SCENARIOS["S5_earnings_gate_on_avoid"], "avoid"),
                              (dm.SCENARIOS["T3_plain_wait_weak_setup"], "wait")],
                             ids=["candidate_avoid", "candidate_wait_for_entry"])
    async def test_c8_analyze_lists_a_blackout_that_does_not_bind(
        self, scenario: dm.Scenario, action_now: str
    ) -> None:
        """RA-C8a: an avoid or wait_for_entry candidate plus a blackout keeps the action it has
        without the blackout, and still lists the gate. G: the event outcome (wait) never
        tightens avoid or wait, and hard_gates.blocking keeps every fired gate (plan §2)."""
        gated, clear = await with_and_without_blackout(scenario)

        assert (gated["action_now"], clear["action_now"]) == (action_now, action_now)
        assert ("earnings_blackout" in gate_ids(gated), "earnings_blackout" in gate_ids(clear)) == (True, False)

    async def test_c8_analyze_hold_passes_a_blackout_through(self) -> None:
        """RA-C8b: a hold candidate (new_money wait) plus a blackout keeps wait, as without the blackout,
        lists the gate and carries no earliest_reassessment (the gate does not bind).
        G from WP2.2 (PD D-2; flipped from action_now hold)."""
        gated, clear = await with_and_without_blackout(HOLD)

        assert (gated["action_now"], clear["action_now"]) == ("wait", "wait")
        assert (gate_ids(gated), gate_ids(clear)) == (["earnings_blackout"], [])
        assert gated["new_money"]["earliest_reassessment"] is None

    def test_c8_card_no_setup_lists_the_blackout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RA-C8c: the trade card with no setup plus a blackout stays no_setup and lists the
        blackout. G: the card's resolution order is frozen in 2a and it keeps disclosing every
        fired gate."""
        monkeypatch.setattr(card_module, "detect_setup", lambda *args, **kwargs: None)

        card = build_card(events_data=BLACKOUT)

        assert card["action"] == "no_setup"
        assert "earnings_blackout" in blocker_ids(card)
