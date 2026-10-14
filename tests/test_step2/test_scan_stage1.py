"""WP2.1: scan stage 1 and the scan state (plan §4 "Scan stage 1", "Diffing"; annex
step2-v2-research/gate_contract.md GC-R14-R21, GC-R24, GC-R2 screen column, GC-R7 stage-1 clause, GC-R11
scan clause; reassessment.md RA.18 scan clause; S-C8 rows).

Stage 1 evaluates every gate the daily frame allows (the card's bar floor and actionable daily price, the
card's daily freshness leg, the knife, liquidity on the card's basis, support from the history attrs), then
resolves rank-first as the card does; events, session and plan stay unevaluated. Scan rows and state entries
disclose their source, their unevaluated (id, leg) pairs and the swing horizon; _diff compares only the pairs
neither side left unevaluated, and a rules_version change is warned, not suppressed. Runs go through
tests/support/swing_matrix.py and tests/support/freshness_frames.py; every run pins its clock.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import stock_analysis.tools.watchlist as watchlist
import stock_analysis.utils.freshness as freshness_module
from stock_analysis.tools.analyze.gates import (
    GATES,
    NOT_APPLICABLE,
    SCREEN_UNEVALUATED,
    UNEVALUATED,
)
from stock_analysis.utils import scan_screen
from stock_analysis.utils.freshness import FRESHNESS_GATE_IDS, build_freshness
from stock_analysis.utils.market_calendar import classify_session
from stock_analysis.utils.watch_store import load_scan_state, save_scan_state
from tests.support import freshness_frames as ff
from tests.support import swing_matrix as sm

ALWAYS = sorted(SCREEN_UNEVALUATED)  # events, session and plan: stage 1 never evaluates them
QUOTE = ["freshness_unverifiable@quote", "stale_data@quote"]  # in session stage 1 reads no intraday quote
SHORT = ["data_quality_critical", "liquidity_missing"]  # under 50 bars risk_metrics has no liquidity basis
CLOCKS: dict[str, datetime] = {"pre_market": sm.PRE_MARKET, "regular": sm.REGULAR}
ROW_KEYS = ["symbol", "action", "setup_type", "trigger_price", "distance_to_trigger_pct", "blockers", "unevaluated",
            "source", "horizon", "earnings_in_days", "expected_move_pct", "changed"]
STATE_KEYS = {"action", "setup_type", "trigger_satisfied", "blockers", "trigger_price", "source", "unevaluated"}
LEGS = {"quote", "daily"}
Row = tuple[str, str, list[str]]  # a scan row's source, action and blockers


def ids(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blockers]


def screen(df: pd.DataFrame | None, now: datetime = sm.PRE_MARKET, session: str = "pre_market") -> dict[str, Any]:
    result: dict[str, Any] = scan_screen.screen_symbol(df, now=now, session=session)
    return result


def row_of(result: dict[str, Any]) -> dict[str, Any]:
    (row,) = result["rows"]
    return row


def scan_row(result: dict[str, Any]) -> tuple[str | None, str, list[str]]:
    """(the scan's only row's source, its action, its blockers)."""
    row = row_of(result)
    return row.get("source"), row["action"], row["blockers"]


def gate_ids(entries: list[str]) -> set[str]:
    """Row or screen entries with their leg stripped."""
    return {entry.split("@")[0] for entry in entries}


def with_attrs(scenario: sm.ScanScenario, **attrs: Any) -> pd.DataFrame:
    df = scenario.frame()
    df.attrs.update(attrs)
    return df


async def failed_fetch(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("fetch failed")


# --- GC-R14: liquidity on the card's basis ---------------------------------------------------------


@pytest.mark.parametrize("name", sm.SCAN_SCENARIOS)
async def test_screen_liquidity_equals_card(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """GC-R14: stage 1's liquidity basis is risk_metrics' liquidity.avg_dollar_volume on the same frame (the
    frame's mean volume x the last close), and None where risk_metrics needs more bars. PD D-11: stage 1
    computes the card's liquidity predicate."""
    df = sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])
    risk = await sm.risk_tool.risk_metrics(sm.SYMBOL)

    assert scan_screen._card_liquidity(df) == (None if risk.get("error") else risk["liquidity"]["avg_dollar_volume"])


# name: (the card basis in dollars, the first scan's row)
LIQUIDITY_ROWS: dict[str, tuple[float, Row]] = {
    # thin over the last 20 bars, liquid over the year: promotes, and the card arms it
    "liq_recent_collapse_breakout": (3072857.31, ("card", "enter_on_trigger", ["market_closed"])),
    # the reverse stops at stage 1, as the card would
    "liq_recent_surge_breakout": (684850.92, ("screen", "avoid", ["weak_liquidity"])),
}


@pytest.mark.parametrize("name", LIQUIDITY_ROWS)
async def test_liquidity_on_the_card_basis_decides_the_row(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R14: the collapse breakout promotes and the card arms it; the surge breakout is avoid [weak_liquidity]
    at stage 1 instead of a promoted row the card avoids. PD D-11: stage 1 computes the card's liquidity
    predicate."""
    basis, row = LIQUIDITY_ROWS[name]
    df = sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])

    assert scan_screen._card_liquidity(df) == basis
    assert scan_row(await sm.scan(tmp_path)) == row


# --- GC-R15: the card's bar floor -------------------------------------------------------------------

# frame: (closes, volumes, (action_hint, blocker_ids))
FLOOR_FRAMES: dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[str, list[str]]]] = {
    "boring_30": (sm.BORING[-30:], (1e6,) * 30, ("wait_for_data", SHORT)),
    "boring_45": (sm.BORING[-45:], (1e6,) * 45, ("wait_for_data", SHORT)),
    "breakout_30": (sm.BREAKOUT[-30:], (1e6,) * 29 + (2.5e6,), ("wait_for_data", SHORT)),
    "breakout_49": (sm.BREAKOUT[-49:], (1e6,) * 48 + (2.5e6,), ("wait_for_data", SHORT)),
    "breakout_50": (sm.BREAKOUT[-50:], (1e6,) * 49 + (2.5e6,), ("candidate", [])),
}


@pytest.mark.parametrize("frame", FLOOR_FRAMES)
def test_screen_bar_floor_matches_card(frame: str) -> None:
    """GC-R15: under 50 bars stage 1 is wait_for_data with the card's own blockers (data_quality_critical and
    liquidity_missing); at 50 bars a breakout promotes. PD D-11: the floor is the card's (50 bars, features, an
    actionable daily price)."""
    closes, volumes, expected = FLOOR_FRAMES[frame]
    result = screen(sm.ScanScenario("GC-R15", closes, volumes).frame())

    assert (result["action_hint"], result["blocker_ids"]) == expected


async def test_a_short_frame_never_reaches_the_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R15: a 30-bar breakout is a screen row, wait_for_data with the card's blockers, and no card runs.
    PD D-11: the floor is the card's."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["A_30_bars_breakout"])
    result = await sm.scan(tmp_path)

    assert (scan_row(result), result["full_cards"]) == (("screen", "wait_for_data", SHORT), 0)


# --- GC-R16: the card's daily freshness leg ---------------------------------------------------------

STALE_FRAMES: dict[str, list[str]] = {
    "stale_frame_boring": ["stale_data@daily"],
    "stale_frame_breakout": ["stale_data@daily"],
    # every fired gate is disclosed, though the data veto decides
    "C_stale_knife": ["falling_knife", "stale_data@daily"],
    "C_stale_thin": ["stale_data@daily", "weak_liquidity"],
}


@pytest.mark.parametrize("session", CLOCKS)
@pytest.mark.parametrize("name", STALE_FRAMES)
def test_screen_stale_frame_is_wait_for_data(name: str, session: str) -> None:
    """GC-R16: a daily frame 12 sessions behind is wait_for_data [stale_data@daily] at stage 1, pre-market and
    in session, with every other fired gate disclosed. RF: stage 1 runs the card's daily freshness leg."""
    result = screen(sm.SCAN_SCENARIOS[name].frame(), CLOCKS[session], session)

    assert (result["promote"], result["action_hint"], result["setup_type"], result["blocker_ids"]) == (
        False, "wait_for_data", None, STALE_FRAMES[name])


DAILY_FRAMES: dict[str, Callable[[], pd.DataFrame]] = {
    "current": lambda: ff.daily(ff.E_MON),
    "two_sessions_behind": lambda: ff.daily("2026-10-08"),
    "empty_last_bar": lambda: ff.daily(ff.E_MON, last_close=None),
    "future_dated": lambda: ff.daily("2026-10-14"),
}
OFF_SESSION_CLOCKS = {"pre_market": ff.PRE_0800, "after_hours": ff.AFTER_1700, "closed": ff.SATURDAY}


@pytest.mark.parametrize("quote", [False, True], ids=["no_quote", "fresh_quote"])
@pytest.mark.parametrize("session", OFF_SESSION_CLOCKS)
@pytest.mark.parametrize("frame", DAILY_FRAMES)
def test_off_session_daily_leg_is_build_freshness(frame: str, session: str, quote: bool) -> None:
    """GC-R16: off-session, daily_leg_freshness is the card's whole build_freshness block, which reads only the
    daily frame. RF: one daily-leg predicate."""
    now = OFF_SESSION_CLOCKS[session]
    daily_df = DAILY_FRAMES[frame]()

    assert freshness_module.daily_leg_freshness(daily_df, session=session, now=now) == build_freshness(
        intraday_df=ff.fresh(now) if quote else None, daily_df=daily_df, session=session, now=now)


@pytest.mark.parametrize("frame", DAILY_FRAMES)
def test_in_session_daily_leg_is_the_cards_daily_ok(frame: str) -> None:
    """GC-R16: in session, daily_leg_freshness is stale exactly when the card's build_freshness, given a fresh
    5m bar, blames the daily frame (stale_daily). RF: one daily-leg predicate."""
    daily_df = DAILY_FRAMES[frame]()
    leg = freshness_module.daily_leg_freshness(daily_df, session="regular", now=ff.REGULAR)
    card = build_freshness(intraday_df=ff.fresh(ff.REGULAR), daily_df=daily_df, session="regular", now=ff.REGULAR)

    assert leg["stale"] is (card["reason_code"] == "stale_daily")
    assert leg["stale"] is (frame != "current")


# --- GC-R17 and GC-R18: support from the history attrs ------------------------------------------------


async def test_screen_support_is_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R17: unsupported support caps at stage 1: no setup stays no_setup, a setup is watch and keeps its type,
    trigger and trigger_satisfied; the ETF breakout's screen row carries them. PD D-11: support is a cap, not
    terminal."""
    boring = screen(sm.SCAN_SCENARIOS["etf_boring"].frame())
    breakout = screen(sm.SCAN_SCENARIOS["etf_breakout"].frame())
    gbp = screen(with_attrs(sm.SCAN_SCENARIOS["fresh_breakout"], currency="GBp"))
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["etf_breakout"])
    result = await sm.scan(tmp_path)
    row = row_of(result)

    assert (boring["action_hint"], boring["setup_type"], boring["blocker_ids"]) == (
        "no_setup", None, ["unsupported_instrument"])
    assert (breakout["promote"], breakout["action_hint"], breakout["setup_type"], breakout["trigger_price"],
            breakout["trigger_satisfied"], breakout["blocker_ids"]) == (
        False, "watch", "breakout", 111.96, False, ["unsupported_instrument"])
    assert (gbp["action_hint"], gbp["setup_type"], gbp["blocker_ids"]) == (
        "watch", "breakout", ["unsupported_currency"])
    assert (row.get("source"), row["action"], row["setup_type"], row["trigger_price"],
            row["distance_to_trigger_pct"], result["full_cards"]) == ("screen", "watch", "breakout", 111.96, 0.0079, 0)


# an ETF that swings +-5, then +-0.1 for 25 bars, then closes at 101.5 on spike volume: above its 101.1 trigger
CROSSED_ETF = sm.ScanScenario("GC-R17", tuple(100 + (5 if i % 2 else -5) for i in range(234))
                              + tuple(100 + (0.1 if i % 2 else -0.1) for i in range(25)) + (101.5,),
                              sm.SPIKE_VOLUME, quote_type="ETF")


async def test_a_capped_setup_keeps_trigger_satisfied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R17: a capped breakout whose last close is above its trigger keeps trigger_satisfied True on the screen
    result and on the screen row's state entry. PD D-11: support is a cap, not terminal."""
    result = screen(CROSSED_ETF.frame())
    sm.install(monkeypatch, CROSSED_ETF)
    scan = await sm.scan(tmp_path)
    entry = load_scan_state(tmp_path)[0]["symbols"][sm.SYMBOL]

    assert (result["action_hint"], result["setup_type"], result["trigger_price"], result["trigger_satisfied"]) == (
        "watch", "breakout", 101.1, True)
    assert (scan_row(scan), entry["trigger_satisfied"]) == (("screen", "watch", ["unsupported_instrument"]), True)


async def test_screen_unverified_support_promotes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R18: support the history attrs leave unverified is unevaluated at stage 1, per dimension: a breakout
    promotes and the card arms it, a boring chart is no_setup. PD D-11: unverified is not unsupported."""
    breakout = screen(sm.SCAN_SCENARIOS["B_attrs_missing_breakout"].frame())
    boring = screen(sm.SCAN_SCENARIOS["B_attrs_missing_boring"].frame())
    no_currency = screen(with_attrs(sm.SCAN_SCENARIOS["fresh_breakout"], currency=None))
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["B_attrs_missing_breakout"])
    result = await sm.scan(tmp_path)

    assert (breakout["promote"], breakout["blocker_ids"]) == (True, [])
    assert ("unsupported_instrument" in breakout["unevaluated"], "unsupported_currency" in breakout["unevaluated"]) == (
        True, False)
    assert (scan_row(result), result["full_cards"]) == (("card", "enter_on_trigger", ["market_closed"]), 1)
    assert (boring["action_hint"], boring["blocker_ids"], "unsupported_instrument" in boring["unevaluated"]) == (
        "no_setup", [], True)
    assert (no_currency["action_hint"], no_currency["blocker_ids"]) == ("candidate", [])
    assert "unsupported_currency" in no_currency["unevaluated"]


# --- GC-R19: rows disclose their source, unevaluated pairs and horizon -----------------------------------


def malformed(entries: list[str]) -> list[str]:
    """Entries that name no registered gate, or carry a leg other than exactly a freshness id's quote or daily."""
    bad = []
    for entry in entries:
        gate_id, _, leg = entry.partition("@")
        if gate_id not in GATES or leg not in (LEGS if gate_id in FRESHNESS_GATE_IDS else {""}):
            bad.append(entry)
    return bad


async def test_scan_rows_disclose_source_and_unevaluated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R19: rows and state entries carry source, unevaluated and horizon; a screen row lists the gates stage 1
    never evaluates (plus the quote leg in session, unverified support, and the knife without a payload); a card
    row lists what the card skipped (the daily leg when the quote leg returned first); freshness entries carry
    their leg. RF: annex status rule 3, plan §4 disclosure."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["B_attrs_missing_boring"])
    unverified = row_of(await sm.scan(tmp_path / "unverified"))
    unverified_regular = row_of(await sm.scan(tmp_path / "unverified_regular", now=sm.REGULAR))
    sm.install(monkeypatch, sm.ScanScenario("GC-R19", sm.BORING[-10:], (1e6,) * 10))
    short = row_of(await sm.scan(tmp_path / "short"))
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["fresh_breakout"])
    card = row_of(await sm.scan(tmp_path / "card"))
    card_no_quote = row_of(await sm.scan(tmp_path / "card_no_quote", now=sm.REGULAR))
    entries = [load_scan_state(tmp_path / name)[0]["symbols"][sm.SYMBOL] for name in ("unverified", "card")]

    assert [list(row) for row in (unverified, card)] == [ROW_KEYS, ROW_KEYS]
    assert [set(entry) for entry in entries] == [STATE_KEYS, STATE_KEYS]
    assert (unverified["source"], unverified["horizon"], unverified["unevaluated"]) == (
        "screen", "swing", sorted([*ALWAYS, "unsupported_instrument"]))
    assert unverified_regular["unevaluated"] == sorted([*ALWAYS, *QUOTE, "unsupported_instrument"])
    assert (short["source"], short["action"], short["blockers"]) == ("screen", "wait_for_data", SHORT)
    assert "falling_knife" in short["unevaluated"]
    assert (card["source"], card["horizon"], card["unevaluated"]) == ("card", "swing", [])
    assert [card_no_quote[key] for key in ("source", "action", "blockers", "unevaluated")] == [
        "card", "wait_for_data", ["freshness_unverifiable@quote"], ["freshness_unverifiable@daily", "stale_data@daily"]]
    for row in (unverified, unverified_regular, short, card, card_no_quote):
        assert malformed(row["blockers"] + row["unevaluated"]) == [], row


async def test_card_rows_list_what_the_card_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R19: a card row's unevaluated list is the card's own (earnings_blackout when the events tool failed),
    plus the daily leg only when the quote leg returned first: an in-session card with a fresh 5m bar leaves
    nothing unevaluated, so its daily-leg firing on a stale frame next scan is a reported blockers change, and
    no pair is both a blocker and unevaluated. RF: annex status rule 3, plan §4 disclosure."""
    close = scan_screen.build_screen_payload(sm.SCAN_SCENARIOS["fresh_breakout"].frame())["current_price"]
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["fresh_breakout"], close)
    fresh = await sm.scan(tmp_path / "quote", now=sm.REGULAR)
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["stale_frame_breakout"], close)
    stale = await sm.scan(tmp_path / "quote", now=sm.REGULAR)
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["fresh_breakout"])
    monkeypatch.setattr(sm.trade_setup, "events_calendar", failed_fetch)
    no_events = await sm.scan(tmp_path / "events")
    rows = [row_of(result) for result in (fresh, stale, no_events)]

    assert [[row[key] for key in ("source", "action", "blockers", "unevaluated")] for row in rows] == [
        ["card", "enter_on_trigger", [], []], ["card", "wait_for_data", ["stale_data@daily"], []],
        ["card", "watch", ["earnings_unverifiable"], ["earnings_blackout"]]]
    assert "blockers: [] -> ['stale_data@daily']" in stale["changes"][0]["notes"]
    for row in rows:
        assert not set(row["blockers"]) & set(row["unevaluated"]), row


# case: (clock, the 5m probe, the daily frame, build_freshness's reason_code, the leg it failed on)
LEG_CASES: dict[str, tuple[datetime, Callable[[], pd.DataFrame | None], Callable[[], pd.DataFrame], str | None,
                           str | None]] = {
    "no_bar_timestamp": (ff.REGULAR, lambda: None, lambda: ff.daily(ff.E_MON), "no_bar_timestamp", "quote"),
    "nonfinite_close": (ff.REGULAR, lambda: ff.five_min(ff.REGULAR, [(2, ff.NAN)]), lambda: ff.daily(ff.E_MON),
                        "nonfinite_close", "quote"),
    "nonpositive_close": (ff.REGULAR, lambda: ff.five_min(ff.REGULAR, [(2, 0.0)]), lambda: ff.daily(ff.E_MON),
                          "nonpositive_close", "quote"),
    "future_timestamp": (ff.REGULAR, lambda: ff.five_min(ff.REGULAR, [(-10, 101.0)]), lambda: ff.daily(ff.E_MON),
                         "future_timestamp", "quote"),
    "stale_quote": (ff.REGULAR, lambda: ff.fresh(ff.REGULAR, 20), lambda: ff.daily(ff.E_MON), "stale_quote", "quote"),
    "regular_stale_daily": (ff.REGULAR, lambda: ff.fresh(ff.REGULAR), lambda: ff.daily("2026-10-08"),
                            "stale_daily", "daily"),
    "regular_fresh": (ff.REGULAR, lambda: ff.fresh(ff.REGULAR), lambda: ff.daily(ff.E_MON), None, None),
    "after_hours_stale_daily": (ff.AFTER_1700, lambda: None, lambda: ff.daily(ff.E_MON), "stale_daily", "daily"),
    "after_hours_daily_unusable": (ff.AFTER_1700, lambda: None, lambda: ff.daily(ff.E_TUE, last_close=None),
                                   "daily_unusable", "daily"),
    "after_hours_fresh": (ff.AFTER_1700, lambda: None, lambda: ff.daily(ff.E_TUE), None, None),
}


@pytest.mark.parametrize("case", LEG_CASES)
def test_freshness_leg(case: str) -> None:
    """GC-R19: freshness_leg names the leg a card's freshness block failed on: the quote leg for every early
    return and a stale quote in session, the daily leg for stale_daily and daily_unusable, none when fresh.
    RF: freshness entries carry their leg (plan §0 #20)."""
    now, quote, daily_df, reason_code, leg = LEG_CASES[case]
    freshness = build_freshness(intraday_df=quote(), daily_df=daily_df(), session=classify_session(now), now=now)

    assert (freshness["reason_code"], freshness_module.freshness_leg(freshness)) == (reason_code, leg)


# --- GC-R20: soundness on identical inputs ----------------------------------------------------------

EXTRA_FRAMES: dict[str, sm.ScanScenario] = {
    "floor_49_breakout": sm.ScanScenario("GC-R20", sm.BREAKOUT[-49:], (1e6,) * 48 + (2.5e6,)),
    "floor_50_breakout": sm.ScanScenario("GC-R20", sm.BREAKOUT[-50:], (1e6,) * 49 + (2.5e6,)),
    "zero_last_close": sm.ScanScenario("GC-R20", sm.BORING[:-1] + (0.0,), sm.FLAT_VOLUME),
}
SOUNDNESS_FRAMES = {**sm.SCAN_SCENARIOS, **EXTRA_FRAMES}
# the veto and cap pairs stage 1 evaluates
EVALUATED = frozenset({"data_quality_critical", "stale_data@daily", "freshness_unverifiable@daily", "falling_knife",
                       "weak_liquidity", "liquidity_missing", "unsupported_instrument", "unsupported_currency"})
# pre_market: the card prices off the daily close; regular_probe_at_close: a fresh 5m bar at the screen's last
# close; regular_no_quote: no 5m bar. In session with no valid quote and a zero daily close the card adds
# data_quality_critical for its fallback price, which stage 1 never reads (scan spec A11), so that frame runs
# pre-market only.
SOUNDNESS_CASES = [(name, mode) for mode in ("pre_market", "regular_probe_at_close", "regular_no_quote")
                   for name in SOUNDNESS_FRAMES if mode == "pre_market" or name != "zero_last_close"]


@pytest.mark.parametrize(("name", "mode"), SOUNDNESS_CASES)
async def test_stage1_soundness(name: str, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """GC-R20: on the same frame, support, tool results and detector price, a non-promoted screen row equals
    the card over every pair neither left unevaluated (S1; without a quote the card's quote-leg veto explains
    any action difference), and a promoted row meets no card veto or cap stage 1 evaluates (S2). RF: plan §4
    soundness S1/S2."""
    scenario = SOUNDNESS_FRAMES[name]
    df = scenario.frame()
    now, session = (sm.PRE_MARKET, "pre_market") if mode == "pre_market" else (sm.REGULAR, "regular")
    probe_close = scan_screen.build_screen_payload(df)["current_price"] if mode == "regular_probe_at_close" else None
    sm.install(monkeypatch, scenario, probe_close)
    result = screen(df.copy(), now, session)
    card = await sm.trade_card(now)
    card_summary = watchlist._summary_from_card(card)
    fired = set(card_summary["blockers"])
    skip = set(result["unevaluated"]) | set(card_summary["unevaluated"])

    if mode == "regular_no_quote":
        assert fired & set(QUOTE) == {"freshness_unverifiable@quote"}
    if result["promote"]:
        assert not (fired - skip) & EVALUATED
        return
    assert fired - skip == set(result["blocker_ids"]) - skip
    if mode == "regular_no_quote":
        assert card["action"] in {result["action_hint"], GATES["freshness_unverifiable"].outcome["swing"]}
        return
    assert card["action"] == result["action_hint"]
    if result["action_hint"] == "watch":
        setup = card["setup"] or {}
        assert (setup.get("type"), setup.get("trigger_price")) == (result["setup_type"], result["trigger_price"])


def test_in_session_a_zero_daily_close_is_the_quote_legs() -> None:
    """GC-R20: zero_last_close runs pre-market only (scan spec A11), so its in-session screen row is pinned here:
    in session stage 1 leaves the actionable daily price to the quote leg, so the zero close fails only the daily
    freshness leg and adds no data_quality_critical. PD D-11 (plan §6 #9: the price clause is off-session only)."""
    result = screen(EXTRA_FRAMES["zero_last_close"].frame(), sm.REGULAR, "regular")

    assert (result["action_hint"], result["blocker_ids"]) == (
        "wait_for_data", ["falling_knife", "stale_data@daily", "weak_liquidity"])


# --- GC-R21: _diff compares the pairs neither side left unevaluated ------------------------------------

# name: the card-forced row, then the screen row; both rows' setup type and trigger
NO_PHANTOM: dict[str, tuple[Row, Row, tuple[str | None, float | None]]] = {
    # stage 1 never evaluates earnings: the card's blackout is not a change
    "boring_earnings_in_2d": (("card", "no_setup", ["earnings_blackout"]), ("screen", "no_setup", []), (None, None)),
    # nor its residual market_closed; the screen keeps the capped setup
    "D_etf_breakout": (("card", "watch", ["market_closed", "unsupported_instrument"]),
                       ("screen", "watch", ["unsupported_instrument"]), ("breakout", 111.96)),
}


@pytest.mark.parametrize("name", NO_PHANTOM)
async def test_scan_diff_ignores_unevaluated_blockers(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R21: a frame scanned by the card (prior state actionable), then by the screen, reports no change: the
    pairs the screen left unevaluated are skipped and the capped setup is kept. RF: plan §4 Diffing, §0 #20."""
    card_row, screen_row, setup = NO_PHANTOM[name]
    sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])

    by_card = await sm.scan(tmp_path, prior_action="enter_on_trigger")
    by_screen = await sm.scan(tmp_path)

    assert (scan_row(by_card), scan_row(by_screen)) == (card_row, screen_row)
    assert [(row_of(r)["setup_type"], row_of(r)["trigger_price"]) for r in (by_card, by_screen)] == [setup, setup]
    assert (by_card["full_cards"], by_screen["full_cards"]) == (1, 0)
    assert (by_screen["changes"], row_of(by_screen)["changed"]) == ([], False)


async def test_a_daily_leg_firing_still_diffs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R21: stale_data@daily, evaluated on both sides, is a change after a card's watch [earnings_blackout];
    the blackout the screen left unevaluated is not. RF: plan §4 Diffing, §0 #20."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["breakout_earnings_in_2d"])
    by_card = await sm.scan(tmp_path)
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["stale_frame_breakout"])
    by_screen = await sm.scan(tmp_path)

    assert (scan_row(by_card), by_card["full_cards"], by_screen["full_cards"]) == (
        ("card", "watch", ["earnings_blackout"]), 1, 0)
    assert by_screen["changes"] == [{"symbol": sm.SYMBOL, "from": "watch", "to": "wait_for_data", "notes": [
        "action: watch -> wait_for_data", "setup_type: breakout -> None", "blockers: [] -> ['stale_data@daily']",
        "trigger_price: 111.96 -> None"]}]


async def test_a_quote_leg_firing_is_never_a_phantom_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R21: in session without a 5m bar the card's freshness_unverifiable@quote (and the blackout) disappear
    when the screen takes the row, and only the action is reported. RF: plan §4 Diffing, §0 #20."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["breakout_earnings_in_2d"])
    by_card = await sm.scan(tmp_path, now=sm.REGULAR)
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["boring_earnings_in_2d"])
    by_screen = await sm.scan(tmp_path, now=sm.REGULAR)

    assert (scan_row(by_card), scan_row(by_screen)) == (
        ("card", "wait_for_data", ["earnings_blackout", "freshness_unverifiable@quote"]), ("screen", "no_setup", []))
    assert [change["notes"] for change in by_screen["changes"]] == [["action: wait_for_data -> no_setup"]]


async def test_a_pair_only_the_prior_left_unevaluated_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R21: in session without a 5m bar the card's quote leg returns before its daily leg, so the card row
    leaves stale_data@daily unevaluated; the screen's stale_data@daily on the same frame next scan is no change.
    RF: plan §4 Diffing, §0 #20 (the pairs neither side left unevaluated)."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["stale_frame_breakout"])
    by_card = await sm.scan(tmp_path, prior_action="enter_on_trigger", now=sm.REGULAR)
    by_screen = await sm.scan(tmp_path, now=sm.REGULAR)

    assert (scan_row(by_card), row_of(by_card)["unevaluated"], scan_row(by_screen)) == (
        ("card", "wait_for_data", ["freshness_unverifiable@quote"]), ["freshness_unverifiable@daily", "stale_data@daily"],
        ("screen", "wait_for_data", ["stale_data@daily"]))
    assert (by_screen["changes"], row_of(by_screen)["changed"]) == ([], False)


# pre-WP2.1 entries (no source, no unevaluated, no legs) on the frames that would show a phantom change
LEGACY_ENTRIES: dict[str, dict[str, Any]] = {
    "boring_earnings_in_2d": {"action": "no_setup", "setup_type": None, "trigger_satisfied": False,
                              "blockers": ["earnings_blackout"], "trigger_price": None},
    "stale_frame_boring": {"action": "wait_for_data", "setup_type": None, "trigger_satisfied": False,
                           "blockers": ["stale_data"], "trigger_price": None},
}


@pytest.mark.parametrize("name", LEGACY_ENTRIES)
async def test_a_legacy_entry_compares_at_id_level(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R21: against a pre-WP2.1 entry the blockers compare at id level (legs stripped), skipping ids the new
    row left unevaluated and fired on no leg: an unchanged frame reports no change, with rules_changed.
    RF: plan §4 Diffing (rule C5)."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])
    await watchlist.manage_watchlist("add", [sm.SYMBOL], _today=sm.PRE_MARKET.date(), data_dir=tmp_path)
    save_scan_state(tmp_path, {"scanned_at": "prior", "symbols": {sm.SYMBOL: LEGACY_ENTRIES[name]}})

    result = await watchlist.scan_watchlist(_now=sm.PRE_MARKET, data_dir=tmp_path)

    assert (result["changes"], [w["id"] for w in result["warnings"]]) == ([], ["rules_changed"])


async def test_a_legacy_entry_keeps_an_id_fired_on_another_leg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R21: in session the screen leaves stale_data unevaluated on the quote leg but fires it on the daily
    leg, so against a pre-WP2.1 no_setup entry the id still compares and the stale frame is reported.
    RF: plan §4 Diffing (rule C5: skip only the ids the new row fired on no leg)."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["stale_frame_boring"])
    await watchlist.manage_watchlist("add", [sm.SYMBOL], _today=sm.REGULAR.date(), data_dir=tmp_path)
    save_scan_state(tmp_path, {"scanned_at": "prior", "symbols": {sm.SYMBOL: {
        "action": "no_setup", "setup_type": None, "trigger_satisfied": False, "blockers": [], "trigger_price": None}}})

    result = await watchlist.scan_watchlist(_now=sm.REGULAR, data_dir=tmp_path)
    row = row_of(result)

    assert (row["blockers"], "stale_data@quote" in row["unevaluated"]) == (["stale_data@daily"], True)
    assert result["changes"] == [{"symbol": sm.SYMBOL, "from": "no_setup", "to": "wait_for_data", "notes": [
        "action: no_setup -> wait_for_data", "blockers: [] -> ['stale_data']"]}]


# --- GC-R24: rules_version --------------------------------------------------------------------------

STORED_VERSIONS: dict[str, dict[str, Any]] = {"absent": {}, "one": {"rules_version": 1},
                                              "string": {"rules_version": "2.0"}}


@pytest.mark.parametrize("stored", STORED_VERSIONS)
async def test_scan_rules_changed_warning(stored: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R24: the scan state saves rules_version 2 and the response echoes it; a first scan warns first_scan
    only, a same-version scan nothing, and a state with another or no version warns rules_changed while still
    reporting its changes. RF: rules_version (critique-v2, snapshot annex)."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["liq_recent_surge_breakout"])
    first = await sm.scan(tmp_path)
    saved, _ = load_scan_state(tmp_path)
    second = await sm.scan(tmp_path)
    save_scan_state(tmp_path, {**STORED_VERSIONS[stored], "scanned_at": "prior", "symbols": {sm.SYMBOL: {
        "action": "enter_on_trigger", "setup_type": "breakout", "trigger_satisfied": False, "blockers": [],
        "trigger_price": 100.0}}})
    third = await sm.scan(tmp_path)

    assert [w["id"] for w in first["warnings"]] == ["first_scan"]
    assert saved["rules_version"] == first["rules_version"] == watchlist.SCAN_RULES_VERSION == 2
    assert second["warnings"] == []
    assert [w["id"] for w in third["warnings"]] == ["rules_changed"]
    assert [change["to"] for change in third["changes"]] == ["avoid"]
    assert load_scan_state(tmp_path)[0]["rules_version"] == 2


# --- GC-R2: the screen column -----------------------------------------------------------------------

FRESH_DECLINE = sm.ScanScenario("GC-R2", sm.DECLINE, sm.FLAT_VOLUME)
NAN_VOLUME = sm.ScanScenario("GC-R2", sm.BORING, (float("nan"),) * 260)
FUTURE_ROW = sm.ScanScenario("GC-R2", sm.BORING, sm.FLAT_VOLUME, last_date="2026-03-11")
FRESH_BORING = sm.ScanScenario("GC-R2", sm.BORING, sm.FLAT_VOLUME)

# gate: (a frame that fires it at stage 1, pre-market; the blocker ids; the setup type kept)
SCREEN_CASES: dict[str, tuple[Callable[[], pd.DataFrame], list[str], str | None]] = {
    "stale_data": (sm.SCAN_SCENARIOS["stale_frame_boring"].frame, ["stale_data@daily"], None),
    "freshness_unverifiable": (FUTURE_ROW.frame, ["freshness_unverifiable@daily"], None),
    "data_quality_critical": (sm.SCAN_SCENARIOS["A_30_bars_boring"].frame, SHORT, None),
    "falling_knife": (FRESH_DECLINE.frame, ["falling_knife"], None),
    "weak_liquidity": (sm.SCAN_SCENARIOS["liq_recent_surge_breakout"].frame, ["weak_liquidity"], None),
    "liquidity_missing": (NAN_VOLUME.frame, ["liquidity_missing"], None),
    # support caps keep the setup; no_setup passes (SCREEN_NO_SETUP_PASSES)
    "unsupported_instrument": (sm.SCAN_SCENARIOS["etf_breakout"].frame, ["unsupported_instrument"], "breakout"),
    "unsupported_currency": (lambda: with_attrs(sm.SCAN_SCENARIOS["fresh_breakout"], currency="GBp"),
                             ["unsupported_currency"], "breakout"),
}
SCREEN_NO_SETUP_PASSES: dict[str, Callable[[], pd.DataFrame]] = {
    "unsupported_instrument": sm.SCAN_SCENARIOS["etf_boring"].frame,
    "unsupported_currency": lambda: with_attrs(FRESH_BORING, currency="GBp"),
}


async def screen_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """The rows stage 1 produced on a first scan of every GC-C4 frame, pre-market and in session."""
    rows: list[dict[str, Any]] = []
    for name, scenario in sm.SCAN_SCENARIOS.items():
        sm.install(monkeypatch, scenario)
        for session, now in CLOCKS.items():
            row = row_of(await sm.scan(tmp_path / f"{name}-{session}", now=now))
            if row.get("source") == "screen":
                rows.append(row)
    return rows


@pytest.mark.parametrize("gate_id", GATES)
async def test_gate_outcome_per_horizon_screen(
    gate_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R2: each gate's stage-1 frame gives the registry's screen outcome with the gate among its blockers
    (legs stripped); an unevaluated gate is listed on every screen row; a not_applicable gate is never a
    stage-1 blocker or unevaluated entry. RF: R2.6(d), screen column."""
    outcome = GATES[gate_id].outcome["screen"]
    if outcome == UNEVALUATED:
        rows = await screen_rows(monkeypatch, tmp_path)
        assert rows
        assert [row for row in rows if gate_id not in row["unevaluated"]] == []
        return
    if outcome == NOT_APPLICABLE:
        frames = [*(s.frame for s in sm.SCAN_SCENARIOS.values()), *(case[0] for case in SCREEN_CASES.values())]
        results = [screen(frame(), now, session) for frame in frames for session, now in CLOCKS.items()]
        assert [r for r in results if gate_id in gate_ids(r["blocker_ids"] + r["unevaluated"])] == []
        return
    frame, blocker_ids, setup_type = SCREEN_CASES[gate_id]
    result = screen(frame())

    assert (result["action_hint"], result["blocker_ids"], result["setup_type"]) == (outcome, blocker_ids, setup_type)
    assert gate_id in gate_ids(result["blocker_ids"])
    if gate_id in SCREEN_NO_SETUP_PASSES:
        companion = screen(SCREEN_NO_SETUP_PASSES[gate_id]())
        assert (companion["action_hint"], companion["blocker_ids"]) == ("no_setup", [gate_id])


# --- GC-R7, stage-1 clause: one knife predicate -------------------------------------------------------

# variant: (the frame's scenario, its history attrs override)
KNIFE_VARIANTS: dict[str, tuple[sm.ScanScenario, dict[str, Any]]] = {
    "equity": (FRESH_DECLINE, {}),
    "etf": (sm.ScanScenario("GC-R7", sm.DECLINE, sm.FLAT_VOLUME, quote_type="ETF"), {}),
    "gbp_attrs": (FRESH_DECLINE, {"currency": "GBp"}),
    "attrs_missing": (sm.ScanScenario("GC-R7", sm.DECLINE, sm.FLAT_VOLUME, history_attrs=False), {}),
}


@pytest.mark.parametrize("variant", KNIFE_VARIANTS)
async def test_knife_shared_predicate_stage1(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """GC-R7: stage 1 and the real card both list falling_knife on a fresh knife frame under every support
    variant (EQUITY, ETF, GBp attrs, attrs missing); no action assertion. PD D-5a: the knife is evaluated for
    every instrument, with is_falling_knife_technicals."""
    scenario, attrs = KNIFE_VARIANTS[variant]
    df = sm.install(monkeypatch, scenario)
    df.attrs.update(attrs)

    assert "falling_knife" in screen(df.copy())["blocker_ids"]
    assert "falling_knife" in ids((await sm.trade_card())["blockers"])


# --- GC-R11, scan clause: unique ids ------------------------------------------------------------------


@pytest.mark.parametrize("name", sm.SCAN_SCENARIOS)
async def test_scan_row_ids_unique(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GC-R11: a row's blockers and unevaluated list each (id, leg) pair once, and only the freshness ids carry
    a leg, on first and card-forced scans, pre-market and in session. RF: plan §4 "unique ids per output"."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])
    for session, now in CLOCKS.items():
        for prior in (None, "enter_on_trigger"):
            row = row_of(await sm.scan(tmp_path / f"{session}-{prior}", prior_action=prior, now=now))
            for entries in (row["blockers"], row["unevaluated"]):
                assert len(entries) == len(set(entries)), (session, prior, entries)
                assert malformed(entries) == [], (session, prior)


# --- RA.18 and S-C8: scan clauses ---------------------------------------------------------------------

# what 0609fde saves for breakout_events_unverifiable when risk_metrics fails
RA18_LEGACY_ENTRY = {"action": "wait_for_data", "setup_type": None, "trigger_satisfied": False,
                     "blockers": ["data_quality_critical", "liquidity_missing"], "trigger_price": None}
RA18_CHANGE = {"symbol": sm.SYMBOL, "from": "wait_for_data", "to": "wait_for_data", "notes": [
    "blockers: ['data_quality_critical', 'liquidity_missing'] -> "
    "['data_quality_critical', 'earnings_unverifiable', 'liquidity_missing']"]}


async def test_ra18_first_scan_after_upgrade_reports_the_one_time_diff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RA.18: the first scan after the upgrade reports the card's newly disclosed earnings_unverifiable on a
    wait_for_data card as a one-time blockers change against the pre-WP2.1 entry, with rules_changed; the next
    scan reports nothing. RF: RA.18 scan clause (reported, not suppressed)."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["breakout_events_unverifiable"])
    monkeypatch.setattr(sm.risk_tool, "fetch_history", failed_fetch)
    await watchlist.manage_watchlist("add", [sm.SYMBOL], _today=sm.PRE_MARKET.date(), data_dir=tmp_path)
    save_scan_state(tmp_path, {"scanned_at": "prior", "symbols": {sm.SYMBOL: RA18_LEGACY_ENTRY}})

    first = await watchlist.scan_watchlist(_now=sm.PRE_MARKET, data_dir=tmp_path)
    second = await watchlist.scan_watchlist(_now=sm.PRE_MARKET, data_dir=tmp_path)

    assert (first["full_cards"], row_of(first)["blockers"]) == (
        1, ["data_quality_critical", "earnings_unverifiable", "liquidity_missing"])
    assert first["changes"] == [RA18_CHANGE]
    assert [w["id"] for w in first["warnings"]] == ["rules_changed"]
    assert second["changes"] == []


async def test_a_legacy_entry_kept_past_a_failed_scan_still_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GC-R24: a pre-WP2.1 entry kept unchanged by a first post-upgrade scan whose daily fetch failed is still
    compared at id level by the next scan, so that scan's one-time RA.18 diff comes with rules_changed although
    the state already says rules_version 2; the scan after it is quiet. RF: rules_version (plan §4), RA.18 scan
    clause (reported, not suppressed)."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["breakout_events_unverifiable"])
    monkeypatch.setattr(sm.risk_tool, "fetch_history", failed_fetch)
    daily_fetch = watchlist.fetch_history
    await watchlist.manage_watchlist("add", [sm.SYMBOL], _today=sm.PRE_MARKET.date(), data_dir=tmp_path)
    save_scan_state(tmp_path, {"scanned_at": "prior", "symbols": {sm.SYMBOL: RA18_LEGACY_ENTRY}})

    monkeypatch.setattr(watchlist, "fetch_history", failed_fetch)
    failed = await watchlist.scan_watchlist(_now=sm.PRE_MARKET, data_dir=tmp_path)
    kept, _ = load_scan_state(tmp_path)
    monkeypatch.setattr(watchlist, "fetch_history", daily_fetch)
    second = await watchlist.scan_watchlist(_now=sm.PRE_MARKET, data_dir=tmp_path)
    third = await watchlist.scan_watchlist(_now=sm.PRE_MARKET, data_dir=tmp_path)

    assert ([w["id"] for w in failed["warnings"]], failed["errors"], failed["changes"]) == (
        ["rules_changed"], [{"symbol": sm.SYMBOL, "error_type": "data_unavailable"}], [])
    assert (kept["rules_version"], kept["symbols"][sm.SYMBOL]) == (2, RA18_LEGACY_ENTRY)
    assert (second["changes"], [w["id"] for w in second["warnings"]]) == ([RA18_CHANGE], ["rules_changed"])
    assert (third["changes"], third["warnings"]) == ([], [])


async def test_scan_rows_state_swing_horizon(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """S-C8: a card row and a screen row both state horizon swing. RF: plan §4 horizon labels, scan rows."""
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["fresh_breakout"])
    card = row_of(await sm.scan(tmp_path / "card"))
    sm.install(monkeypatch, sm.SCAN_SCENARIOS["stale_frame_boring"])
    stale = row_of(await sm.scan(tmp_path / "screen"))

    assert [(row.get("source"), row.get("horizon")) for row in (card, stale)] == [
        ("card", "swing"), ("screen", "swing")]
