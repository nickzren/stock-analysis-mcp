"""WP2.1: the shared earnings window (B4) and analyze's clock (BL-D9 clock half).

Annex docs/plans/step2-v2-research/reassessment.md, "Corrected proposal" (RA.1, RA.12, RA.15) and its
"Related changes (B4)". earnings_window keeps the blackout predicate, names the interval behind it and
owns analyze's 7-day warning; analyze passes its own clock to events and to the default next_review.
The reassessment date and RA.17's next_review rule are WP2.2 (test_wp22_reassessment.py). Events runs use
the real tool over fixed provider reads (test_characterize_reassessment);
analyze runs go through tests/support/decision_matrix.py and card runs through
tests/test_tools/test_trade_card.py.
"""

import random
from datetime import date, datetime, timedelta, tzinfo
from typing import Any

import pytest

import stock_analysis.tools.analyze.decision_card as decision_card_module
import stock_analysis.tools.events as events
from stock_analysis.tools.analyze.gates import (
    EARNINGS_BLACKOUT_DAYS,
    EarningsWindow,
    earnings_window,
)
from tests.support import decision_matrix as dm
from tests.support.fake_ticker import FakeTicker
from tests.test_step2 import test_characterize_reassessment as ra
from tests.test_step2.test_characterize_reassessment import (
    E1,
    E2,
    EVIDENCE_ROWS,
    EventsAt,
    Evidence,
    evening,
    fires,
    random_evidence,
)
from tests.test_tools.test_earnings_verification import calendar
from tests.test_tools.test_trade_card import build_card

events_at = ra.events_at  # the real events tool over fixed provider reads, run at noon ET on a given day

WALL = dm.ET.localize(datetime(2027, 3, 1, 12, 0))  # a wall clock far from every run date
OCT_8 = date(2026, 10, 8)


class FrozenWall(datetime):
    """datetime whose now() reads WALL, so any wall-clock read shows up in the output."""

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> datetime:
        return WALL.astimezone(tz) if tz is not None else WALL.replace(tzinfo=None)


async def analyze_with_real_events(
    evidence: Evidence, now: datetime, calls: list[tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """S2_buy_clean through analyze with the real events tool over ``evidence`` and the wall clock
    frozen at WALL; ``calls`` records the arguments analyze passes to events_calendar."""
    scenario = dm.SCENARIOS["S2_buy_clean"]
    with dm.stubbed(scenario), pytest.MonkeyPatch.context() as mp:
        async def fake_fetch_ticker(symbol: str) -> FakeTicker:
            return FakeTicker(symbol, splits=None)

        async def fake_fetch_info(symbol: str) -> dict[str, Any]:
            return evidence.info

        mp.setattr(events, "fetch_ticker", fake_fetch_ticker)
        mp.setattr(events, "fetch_info", fake_fetch_info)
        mp.setattr(events, "read_calendar_events", evidence.cal)
        mp.setattr(events, "read_earnings_page", evidence.pg)
        mp.setattr(events, "datetime", FrozenWall)
        mp.setattr(decision_card_module, "datetime", FrozenWall)
        real = events.events_calendar

        async def spy(symbol: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if calls is not None:
                calls.append((args, kwargs))
            return await real(symbol, *args, **kwargs)

        mp.setattr(dm.orchestrator, "events_calendar", spy)
        result: dict[str, Any] = await dm.orchestrator.analyze_stock(
            "ACME", account_size=scenario.account_size, profile=scenario.profile, _now=now)
    return result


def todays_predicate(earnings: dict[str, Any]) -> bool:
    """The blackout predicate check_earnings_blackout had before the helper, restated."""
    days = earnings.get("days_until")
    return isinstance(days, (int, float)) and 0 <= days <= EARNINGS_BLACKOUT_DAYS


def assert_parity(window: EarningsWindow, earnings: dict[str, Any], today: date) -> None:
    assert window.fires is todays_predicate(earnings)
    assert window.fires is fires(earnings)  # check_earnings_blackout delegates to the helper
    assert window.datable is window.fires
    assert bool(window.covers(today)) is window.fires
    assert (window.reason is not None) is window.fires


def blackout_reasons(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["reason"] for b in blockers if b["id"] == "earnings_blackout"]


class TestRA1:
    @pytest.mark.parametrize("evidence", EVIDENCE_ROWS)
    async def test_parity_corpus(self, events_at: EventsAt, evidence: Evidence) -> None:
        """RA.1: on every earnings-verification input run on Oct 8, earnings_window fires exactly
        when today's blackout predicate does, and is then datable: its shown intervals cover the
        run date. RF: one helper behind the blackout (B4)."""
        earnings = (await events_at(evidence, OCT_8))["earnings"]

        assert_parity(earnings_window(earnings, OCT_8), earnings, OCT_8)

    async def test_seeded_corpus(self, events_at: EventsAt) -> None:
        """RA.1: the same parity on RA-C3b's 300 seeded random evidence mixes and run dates around
        Thanksgiving 2026; 156 fire. RF: one helper behind the blackout (B4)."""
        rng = random.Random(7)
        base = date(2026, 11, 16)
        fired = 0
        for _ in range(300):
            evidence = random_evidence(rng, base)
            today = base + timedelta(days=rng.randint(-2, 14))
            earnings = (await events_at(evidence, today))["earnings"]
            window = earnings_window(earnings, today)
            assert_parity(window, earnings, today)
            fired += window.fires

        assert fired == 156

    @pytest.mark.parametrize(("earnings", "fired"), [({"days_until": 3}, True), ({"days_until": 0}, True),
                                                     ({"days_until": 5}, True), ({"days_until": 6}, False),
                                                     ({}, False)],
                             ids=["days_until_3", "days_until_0", "days_until_5", "days_until_6", "empty"])
    def test_stubs(self, earnings: dict[str, Any], fired: bool) -> None:
        """RA.1: a stub with only days_until fires on 0 <= days_until <= 5 and is never datable (no
        interval shown). RF: the predicate is kept (RA-C5)."""
        window = earnings_window(earnings, OCT_8)

        assert (window.fires, window.datable) == (fired, False)

    async def test_other_day_is_undatable(self, events_at: EventsAt) -> None:
        """RA.1: E2's Oct 8 output is datable on Oct 8 only; read on Oct 9 it describes another day.
        RF: datability needs the events run date to be the caller's today."""
        earnings = (await events_at(E2, OCT_8))["earnings"]

        assert earnings_window(earnings, OCT_8).datable is True
        assert earnings_window(earnings, date(2026, 10, 9)).datable is False


class TestB4DayZeroWarning:
    @pytest.mark.parametrize("days", [0, 3, 7, 8])
    async def test_warning_window(self, days: int) -> None:
        """B4: analyze warns earnings_within_7_days for 0 <= days_until <= 7, day 0 included.
        RF: day 0 is inside the window (orchestrator tested days_until for truthiness)."""
        next_date = (date(2026, 10, 9) + timedelta(days=days)).isoformat()

        result = await dm.run("S2_buy_clean", events={"earnings": {"days_until": days, "next_date": next_date}})

        assert ("earnings_within_7_days" in result["data_quality"]["warnings"]) is (days <= 7)

    async def test_e2_day_zero_real_events(self, events_at: EventsAt) -> None:
        """B4: E2 run on Oct 8 (day 0 inside the active window) warns, and decision_context's
        next_catalyst keeps days_until 0. RF: day 0 is a real days_until."""
        events_data = await events_at(E2, OCT_8)

        result = await dm.run("S2_buy_clean", events=events_data, now=evening(OCT_8))

        assert "earnings_within_7_days" in result["data_quality"]["warnings"]
        assert result["decision_context"]["next_catalyst"] == {"event": "earnings", "date": "2026-10-08",
                                                               "days_until": 0}

    @pytest.mark.parametrize(("earnings", "warns"), [({"days_until": 0}, True), ({"days_until": 7}, True),
                                                     ({"days_until": 8}, False), ({"days_until": -1}, False),
                                                     ({"days_until": None}, False), ({}, False)],
                             ids=["0", "7", "8", "-1", "none", "empty"])
    def test_helper_warns(self, earnings: dict[str, Any], warns: bool) -> None:
        """B4: the helper's warning reach is 0 <= days_until <= 7. RF: one helper for the warning."""
        assert earnings_window(earnings, OCT_8).warns is warns


class TestRA12:
    async def test_analyze_passes_now_to_events(self) -> None:
        """RA.12: analyze calls events_calendar with its own clock. RF: BL-D9 clock half."""
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        now = evening(OCT_8)

        await analyze_with_real_events(E1, now, calls)

        assert calls == [((), {"_now": now})]

    async def test_no_wall_clock_date_in_catalyst_or_review(self) -> None:
        """RA.12: with the wall clock months away, E1 (verified Oct 15) run on Oct 8 gives Oct 15,
        7 days, in events_summary, decision_context and next_review. RF: BL-D9 clock half."""
        result = await analyze_with_real_events(E1, evening(OCT_8))

        catalyst = result["events_summary"]["next_catalyst"]
        assert (catalyst["date"], catalyst["days_until"]) == ("2026-10-15", 7)
        assert result["decision_context"]["next_catalyst"] == {"event": "earnings", "date": "2026-10-15",
                                                               "days_until": 7}
        assert result["decision_card"]["next_review"] == {"date": "2026-10-15", "reason": "next earnings (~7 days)"}

    async def test_default_review_counts_from_analyze_clock(self) -> None:
        """RA.12: earnings more than 90 days out give the default next_review, 30 days from analyze's
        clock (Oct 8 -> Nov 7), not from the wall clock. RF: BL-D9 clock half."""
        result = await analyze_with_real_events(Evidence(calendar("2027-02-01")), evening(OCT_8))

        assert result["decision_card"]["next_review"] == {"date": "2026-11-07",
                                                          "reason": "default 30-day re-evaluation"}


def window_earnings(next_date: str, days_until: int, start: str, end: str) -> dict[str, Any]:
    """An events earnings block whose next date comes from one window, as the events tool shapes it."""
    window = {"start": start, "end": end}
    return {"next_date": next_date, "days_until": days_until, "next_window": window, "windows": [window]}


class TestRA15Reason:
    @pytest.mark.parametrize(("evidence", "today", "reason"),
                             [(E2, OCT_8, "earnings window Oct 7–15 active"),
                              (E1, date(2026, 10, 15), "earnings today"),
                              (E1, date(2026, 10, 12), "earnings in 3 days (<= 5)")],
                             ids=["E2", "E15", "E1"])
    async def test_reason(self, events_at: EventsAt, evidence: Evidence, today: date, reason: str) -> None:
        """RA.15: the blackout reason names the active window, says 'earnings today' on a single
        date's day 0, and counts days otherwise, in analyze and on the trade card (whose 2026-03-10
        clock pins the events run date as the anchor). RF: reason text (B4)."""
        events_data = await events_at(evidence, today)

        analyze = await dm.run("S2_buy_clean", events=events_data, now=evening(today))
        card = build_card(events_data=events_data)

        assert blackout_reasons(analyze["decision_card"]["hard_gates"]["blocking"]) == [reason]
        assert blackout_reasons(card["blockers"]) == [reason]

    @pytest.mark.parametrize(("earnings", "reason"), [
        (window_earnings("2026-09-30", 0, "2026-09-28", "2026-10-02"), "earnings window Sep 28–Oct 2 active"),
        (window_earnings("2026-12-31", 0, "2026-12-30", "2027-01-04"),
         "earnings window Dec 30, 2026–Jan 4, 2027 active"),
        (window_earnings("2026-10-08", 0, "2026-10-08", "2026-10-08"), "earnings today"),
        (window_earnings("2026-10-12", 4, "2026-10-12", "2026-10-16"), "earnings in 4 days (<= 5)"),
        ({"days_until": 0}, "earnings today"),
    ], ids=["across_months", "across_years", "same_day_window", "window_ahead", "stub_day_0"])
    def test_window_format(self, earnings: dict[str, Any], reason: str) -> None:
        """RA.15: windows read 'Oct 7–15', 'Sep 28–Oct 2' or 'Dec 30, 2026–Jan 4, 2027'; a same-day
        window or a bare day 0 reads 'earnings today'; a window not yet open counts days.
        RF: reason text (B4)."""
        assert earnings_window(earnings, OCT_8).reason == reason
