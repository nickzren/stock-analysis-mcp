"""WP2.1: analyze's freshness gate (plan §5, PD D-7) and the one valid-price predicate (PD D-12).

Annex: docs/plans/step2-v2-research/freshness.md, "Corrected proposal" (truth table, tests R-F1 to
R-F17), as amended by plan §0 #4 (ids stale_data and freshness_unverifiable, each with its codes) and
§5 (the pinned cross-tool differences). Until decide() (WP2.2), action_now is the interim hard-gate
resolution: only the allow-row parity and the CH interim test assert an action. Frames and clocks
come from tests/support/freshness_frames.py; analyze runs go through tests/support/decision_matrix.py.
"""

import importlib
import math
from datetime import datetime
from typing import Any

import pytest

from stock_analysis.tools.analyze.gates import FRESHNESS_CODE_IDS, check_price_freshness
from stock_analysis.utils.freshness import build_freshness, freshness_blockers
from stock_analysis.utils.helpers import valid_price
from stock_analysis.utils.market_calendar import classify_session
from stock_analysis.utils.price_basis import build_daily_history, select_price
from tests.support import decision_matrix as dm
from tests.support import freshness_frames as fr

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")

STALE_DATA, FRESHNESS_UNVERIFIABLE = "stale_data", "freshness_unverifiable"
# Plan §5's id table, written out so a mis-mapped code in gates.py fails the row tests.
CODE_IDS = {
    "info_quote_fallback": FRESHNESS_UNVERIFIABLE, "intraday_quote_unavailable": FRESHNESS_UNVERIFIABLE,
    "intraday_quote_stale": STALE_DATA, "intraday_future_timestamp": FRESHNESS_UNVERIFIABLE,
    "daily_close_stale": STALE_DATA, "daily_history_unavailable": FRESHNESS_UNVERIFIABLE,
    "daily_latest_bar_invalid": FRESHNESS_UNVERIFIABLE, "daily_history_stale": STALE_DATA,
    "daily_future_bar": FRESHNESS_UNVERIFIABLE, "daily_bar_after_last_session": FRESHNESS_UNVERIFIABLE,
}
LEG_H_CODES = ("daily_history_unavailable", "daily_latest_bar_invalid", "daily_history_stale", "daily_future_bar",
               "daily_bar_after_last_session")
FRESHNESS_KEYS = ["session", "expected_session", "price_source", "daily_last_valid_date", "daily_completed_session",
                  "daily_sessions_behind", "codes", "flags"]  # data_quality.freshness, exactly and in order
# rows whose history has no usable last_valid_date or last_row_valid (rule 1): every daily_* value is null
RULE_1_ROWS = {"R13", "O11", "M1", "M2", "M3"}


def split(codes: tuple[str, ...] | list[str]) -> list[tuple[str, list[str]]]:
    """The expected blockers for an ordered code list: (id, its codes), stale_data first."""
    expected = []
    for gate_id in (STALE_DATA, FRESHNESS_UNVERIFIABLE):
        mine = [code for code in codes if CODE_IDS[code] == gate_id]
        if mine:
            expected.append((gate_id, mine))
    return expected


def pairs(blockers: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    return [(b["id"], b["codes"]) for b in blockers]


def freshness_ids(result: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """An analyze run's freshness blockers as (id, codes); the other gates are filtered out."""
    return pairs([g for g in result["decision_card"]["hard_gates"]["blocking"]
                  if g["id"] in (STALE_DATA, FRESHNESS_UNVERIFIABLE)])


def pure(name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return check_price_freshness(**fr.price_inputs(fr.TRUTH_TABLE[name]))


async def run_row(name: str) -> dict[str, Any]:
    return await dm.run(fr.scenario(fr.TRUTH_TABLE[name], name))


async def codes_of(name: str) -> list[str]:
    codes: list[str] = (await run_row(name))["data_quality"]["freshness"]["codes"]
    return codes


# --- FR-R-F1, F16, F17: the pure gate -------------------------------------------------------


class TestPureGate:
    @pytest.mark.parametrize("name", fr.TRUTH_TABLE)
    def test_row(self, name: str) -> None:
        """FR-R-F1: check_price_freshness over every truth-table row gives the disclosure's exact keys, the
        session, the codes in order, the flags, E, the history's newest valid row and completed bar (null under
        rule 1) and the daily sessions behind; each id carries exactly its own codes, stale_data first.
        PD D-7: legs Q and H, every session."""
        row = fr.TRUTH_TABLE[name]
        history = fr.price_inputs(row)["daily_history"]
        dates = (None, None) if name in RULE_1_ROWS else (history["last_valid_date"], history["completed_valid_date"])

        blockers, disclosure = pure(name)

        assert list(disclosure) == FRESHNESS_KEYS
        assert (disclosure["session"], disclosure["daily_last_valid_date"], disclosure["daily_completed_session"]) == (
            classify_session(row.now), *dates)
        assert disclosure["price_source"] == row.price_source
        assert (disclosure["codes"], disclosure["flags"]) == (list(row.codes), list(row.flags))
        assert (disclosure["expected_session"], disclosure["daily_sessions_behind"]) == (
            row.expected_session, row.sessions_behind)
        assert pairs(blockers) == split(row.codes)

    @pytest.mark.parametrize(("now", "provenance", "codes", "source"), [
        pytest.param(fr.REGULAR, None, [], None, id="provenance_absent"),
        pytest.param(fr.REGULAR, select_price(now=fr.REGULAR, info_quote={"price": None})[1], [], None,
                     id="no_price"),
        pytest.param(fr.PRE_0800, select_price(now=fr.PRE_0800, session_close={"close": 100.0, "session_date": None},
                                               info_quote={"price": 99.0})[1],
                     ["info_quote_fallback"], "info_quote_fallback", id="session_close_without_date"),
    ])
    def test_missing_provenance(self, now: datetime, provenance: Any, codes: list[str], source: str | None) -> None:
        """FR-R-F1: with a current daily history, no selected price adds no code (that is
        data_quality_critical's case, current_price missing); a session_close without a date falls to
        the info quote. PD D-7: leg Q reads the selected price only."""
        history = build_daily_history(fr.daily("2026-10-12"), now)

        blockers, disclosure = check_price_freshness(now=now, price_provenance=provenance, daily_history=history)

        assert (disclosure["codes"], disclosure["price_source"]) == (codes, source)
        assert pairs(blockers) == split(codes)

    def test_reason_names_the_dates(self) -> None:
        """FR-R-F1: the stale_data reason names the sessions behind, the last completed bar and E (R3).
        PD D-7."""
        blockers, _ = pure("R3")
        assert blockers[0]["reason"] == (
            "daily history 2 sessions behind (last completed bar 2026-10-08, expected 2026-10-12)")

    def test_the_forming_row_never_changes_legh(self) -> None:
        """FR-R-F16: the R11/R11b history is one session behind in session (today's forming row aside),
        and the same history pre-market, before the vendor adds today's row, is too (P11b).
        PD D-7: leg H measures the newest completed valid bar."""
        seen = {}
        for name in ("R11", "R11b", "P11b"):
            _, disclosure = pure(name)
            seen[name] = ([c for c in disclosure["codes"] if c in LEG_H_CODES], disclosure["daily_sessions_behind"])

        assert seen == {name: (["daily_history_stale"], 1) for name in ("R11", "R11b", "P11b")}


MALFORMED: dict[str, Any] = {
    "not_a_dict": "2026-10-12",
    "list": [],
    "date_garbage": {"last_valid_date": "garbage", "last_row_valid": True, "completed_valid_date": "2026-10-12"},
    "datetime_string": {"last_valid_date": "2026-10-12T00:00:00", "last_row_valid": True,
                        "completed_valid_date": "2026-10-12"},
    "date_not_a_string": {"last_valid_date": 20261012, "last_row_valid": True, "completed_valid_date": "2026-10-12"},
    "row_valid_missing": {"last_valid_date": "2026-10-12", "completed_valid_date": "2026-10-12"},
    "row_valid_int": {"last_valid_date": "2026-10-12", "last_row_valid": 1, "completed_valid_date": "2026-10-12"},
    "row_valid_string": {"last_valid_date": "2026-10-12", "last_row_valid": "true",
                         "completed_valid_date": "2026-10-12"},
    "row_valid_none": {"last_valid_date": "2026-10-12", "last_row_valid": None, "completed_valid_date": "2026-10-12"},
    "completed_garbage": {"last_valid_date": "2026-10-12", "last_row_valid": True, "completed_valid_date": "x"},
    "completed_missing": {"last_valid_date": "2026-10-12", "last_row_valid": True},
}
# rule 1 accepts these: only the completed date is unusable, so the newest valid row stays disclosed
COMPLETED_UNUSABLE = {"completed_garbage", "completed_missing"}


class TestMalformedHistory:
    @pytest.mark.parametrize("name", MALFORMED)
    def test_fails_closed_without_raising(self, name: str) -> None:
        """FR-R-F17: a malformed daily_history gives daily_history_unavailable, one freshness_unverifiable
        blocker, a null completed session and sessions behind, and a null newest valid row unless only the
        completed date is unusable; it never raises. RF: fail closed (plan §5)."""
        inputs = {**fr.price_inputs(fr.TRUTH_TABLE["R2"]), "daily_history": MALFORMED[name]}

        blockers, disclosure = check_price_freshness(**inputs)

        assert disclosure["codes"] == ["daily_history_unavailable"]
        assert pairs(blockers) == [(FRESHNESS_UNVERIFIABLE, ["daily_history_unavailable"])]
        assert (disclosure["daily_last_valid_date"], disclosure["daily_completed_session"],
                disclosure["daily_sessions_behind"]) == (
            "2026-10-12" if name in COMPLETED_UNUSABLE else None, None, None)

    async def test_analyze_survives_it(self) -> None:
        """FR-R-F17: analyze returns with the blocker when technicals carries a malformed daily_history.
        RF: fail closed (plan §5)."""
        result = await dm.run("S1_hold_or_add_clean", technicals={**dm.STRONG, "daily_history": "garbage"})

        assert freshness_ids(result) == [(FRESHNESS_UNVERIFIABLE, ["daily_history_unavailable"])]


# --- FR-R-F3 to F9: analyze end to end ------------------------------------------------------


class TestAnalyzeRows:
    async def test_r5_r6_differ_only_by_the_forming_flag(self) -> None:
        """FR-R-F3: a stale in-session quote blocks whether or not the vendor added today's row; only the
        includes_forming_bar flag differs. PD D-7."""
        r5, r6 = await run_row("R5"), await run_row("R6")

        assert freshness_ids(r5) == freshness_ids(r6) == [(STALE_DATA, ["intraday_quote_stale"])]
        assert r5["decision_card"]["action_now"] == r6["decision_card"]["action_now"]
        assert (r5["data_quality"]["freshness"]["flags"], r6["data_quality"]["freshness"]["flags"]) == (
            ["includes_forming_bar"], [])

    @pytest.mark.parametrize(("name", "code"), [
        ("R7", "intraday_quote_unavailable"), ("R17", "intraday_quote_unavailable"),
        ("R8", "intraday_future_timestamp"), ("R15", "intraday_quote_stale"),
    ])
    async def test_rejected_quotes(self, name: str, code: str) -> None:
        """FR-R-F4: every rejected in-session quote blocks with its own code. PD D-7."""
        assert await codes_of(name) == [code]

    @pytest.mark.parametrize("name", ["R4", "R10", "O4"])
    async def test_invalid_latest_row(self, name: str) -> None:
        """FR-R-F5: a latest daily row without a valid (finite, > 0) close blocks. PD D-7, D-12."""
        assert await codes_of(name) == ["daily_latest_bar_invalid"]

    @pytest.mark.parametrize("name", ["O2", "O10", "E3"])
    async def test_off_hours_one_session_behind(self, name: str) -> None:
        """FR-R-F6: off-hours, a close one session behind blocks on both legs. PD D-7."""
        assert await codes_of(name) == ["daily_close_stale", "daily_history_stale"]

    @pytest.mark.parametrize("name", ["O11", "R13"])
    async def test_info_fallback(self, name: str) -> None:
        """FR-R-F7: the info quote fallback blocks even with a current quote_time; R13 (technicals failed)
        also keeps data_quality_critical. PD D-7."""
        result = await run_row(name)

        assert result["data_quality"]["freshness"]["codes"] == ["info_quote_fallback", "daily_history_unavailable"]
        assert ("data_quality_critical" in dm.gate_ids(result)) is (name == "R13")

    async def test_payload_without_daily_history(self) -> None:
        """FR-R-F8: a technicals payload without daily_history fails closed. PD D-7."""
        assert freshness_ids(await run_row("M3")) == [(FRESHNESS_UNVERIFIABLE, ["daily_history_unavailable"])]

    @pytest.mark.parametrize(("name", "code"), [
        ("O5", "daily_bar_after_last_session"), ("O13", "daily_bar_after_last_session"),
        ("W1", "daily_bar_after_last_session"), ("R14", "daily_future_bar"),
    ])
    async def test_rows_after_the_last_session(self, name: str, code: str) -> None:
        """FR-R-F9: a valid row after the last completed session (not today's forming row) blocks, and a
        row after today is a future bar. PD D-7."""
        assert await codes_of(name) == [code]


ALLOW_ROWS = ["R1", "R2", "R9", "R16", "O1", "O3", "O6", "O7", "O9", "O12", "E1", "E2"]


@pytest.mark.parametrize("name", ALLOW_ROWS)
async def test_allow_rows_match_the_ungated_run(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-R-F10 (amended): allow rows raise no freshness id, and action_now equals the same run with the
    gate disabled; R1 carries the forming-bar flag. PD D-7."""
    gated = await run_row(name)
    real = orchestrator.check_price_freshness
    monkeypatch.setattr(orchestrator, "check_price_freshness", lambda **kw: ([], real(**kw)[1]))
    ungated = await run_row(name)

    assert (freshness_ids(gated), gated["data_quality"]["freshness"]["codes"]) == ([], [])
    assert gated["decision_card"]["action_now"] == ungated["decision_card"]["action_now"]
    assert gated["data_quality"]["freshness"]["flags"] == (["includes_forming_bar"] if name == "R1" else [])


@pytest.mark.parametrize(("name", "behind"), [("R3", 2), ("R11", 1)])
async def test_price_is_technicals_own_completed_bar(name: str, behind: int) -> None:
    """FR-R-F11: component_freshness.price counts from technicals' own completed bar (the gate's number),
    risk from risk_metrics' own provenance, so the gate and component_freshness never disagree.
    RF: one daily history per tool (plan §5)."""
    data_quality = (await run_row(name))["data_quality"]
    components = data_quality["component_freshness"]

    assert components["price"]["sessions_behind"] == data_quality["freshness"]["daily_sessions_behind"] == behind
    assert components["risk"]["sessions_behind"] == 0
    assert data_quality["staleness_warnings"] == [f"price_stale_{behind}_sessions"]


LAGGING_RISK = {**dm.CALM, "data_provenance": {"price": {"source": "yfinance", "as_of": dm.VENDOR_AS_OF,
                                                         "last_bar_date": "2026-10-07"}}}  # E is Fri 10-09


async def test_risk_is_risk_metrics_own_provenance() -> None:
    """FR-R-F11: component_freshness.risk reads risk_metrics' own provenance, never technicals': unavailable
    when risk_metrics fails, and two sessions behind when its own last bar lags technicals' current one.
    RF: one daily history per tool (plan §5)."""
    failed = (await dm.run("S1_hold_or_add_clean", risk=None))["data_quality"]
    lagging = (await dm.run("S1_hold_or_add_clean", risk=LAGGING_RISK))["data_quality"]
    components = lagging["component_freshness"]

    assert failed["component_freshness"]["risk"] == {"as_of": None, "last_bar_date": None, "status": "unavailable"}
    assert (components["price"]["sessions_behind"], components["risk"]["sessions_behind"]) == (0, 2)
    assert lagging["staleness_warnings"] == ["risk_stale_2_sessions"]


# --- FR-R-F13: analyze vs the card on shared frames ------------------------------------------

OUTCOME_DIFF = {"R9": "card", "R16": "card", "R11": "analyze", "R11b": "analyze", "O5": "analyze",
                "O13": "analyze", "W1": "analyze"}  # row: the only tool that blocks
ID_DIFF = {"R4", "R10", "R12", "R14", "O8"}  # both block, data class, different ids


def test_the_pinned_differences() -> None:
    """FR-R-F13: on shared frames, every freshness id either tool raises is data class, and the outcome
    and id differences equal plan §5's pins after D-12 (R16 joins R9; R10 joins the id list; R17 matches).
    PD D-7, D-12."""
    outcome, id_diff = {}, set()
    for name in fr.SHARED_ROWS:
        analyze = {b["id"] for b in pure(name)[0]}
        card = {b["id"] for b in freshness_blockers(fr.card_freshness(fr.TRUTH_TABLE[name]))}
        assert analyze | card <= {STALE_DATA, FRESHNESS_UNVERIFIABLE}, name
        if bool(analyze) != bool(card):
            outcome[name] = "analyze" if analyze else "card"
        elif analyze != card:
            id_diff.add(name)

    assert (outcome, id_diff) == (OUTCOME_DIFF, ID_DIFF)


@pytest.mark.parametrize("name", fr.SHARED_ROWS)
def test_card_reason_codes(name: str) -> None:
    """FR-R-F13: the card keeps its own reason codes on the same frames (no code-to-code map); a 0.0 close
    has its own, nonpositive_close. PD D-7, D-12."""
    assert fr.card_freshness(fr.TRUTH_TABLE[name])["reason_code"] == fr.TRUTH_TABLE[name].card_reason


# --- GC-R12, freshness half: ids and codes in analyze ----------------------------------------


def test_the_code_map() -> None:
    """GC-R12 (freshness half): ten codes, each mapped to one data-class id. PD D-7 (plan §0 #4, §5)."""
    assert FRESHNESS_CODE_IDS == CODE_IDS


@pytest.mark.parametrize("name", ["R6", "O2", "R3", "R7", "R8", "R13", "R4", "R14", "O5", "R12", "O8"])
async def test_every_code_reaches_analyze_at_every_tier(name: str) -> None:
    """GC-R12 (freshness half): the blockers carry their codes on decision_card.hard_gates, and
    data_quality.freshness is identical at the decision, standard and full tiers. PD D-7."""
    result = await run_row(name)
    codes = list(fr.TRUTH_TABLE[name].codes)

    assert freshness_ids(result) == split(codes)
    assert list(result["data_quality"]["freshness"]) == FRESHNESS_KEYS
    for tier, shown in dm.tiers(result).items():
        assert shown["data_quality"]["freshness"] == result["data_quality"]["freshness"], tier
        assert shown["data_quality"]["freshness"]["codes"] == codes, tier


# --- D-12: one valid-price predicate -----------------------------------------------------------


class TestOneValidPricePredicate:
    @pytest.mark.parametrize("close", [math.nan, math.inf, -math.inf, 0.0, -1.0, 1e-9, 101.0])
    def test_build_freshness_uses_valid_price_on_both_legs(self, close: float) -> None:
        """D-12: the card's 5m leg and daily leg accept a close iff valid_price does (finite and > 0); a
        finite non-positive 5m close is nonpositive_close. PD D-12."""
        ok = valid_price(close) is not None
        quote = build_freshness(intraday_df=fr.five_min(fr.REGULAR, [(1, close)]), daily_df=fr.daily("2026-10-12"),
                                session="regular", now=fr.REGULAR)
        daily_in = build_freshness(intraday_df=fr.fresh(fr.REGULAR), daily_df=fr.daily("2026-10-12", close),
                                   session="regular", now=fr.REGULAR)
        daily_off = build_freshness(intraday_df=None, daily_df=fr.daily("2026-10-13", close),
                                    session="after_hours", now=fr.AFTER_1700)

        assert quote["reason_code"] == (None if ok else "nonpositive_close" if math.isfinite(close)
                                        else "nonfinite_close")
        assert daily_in["reason_code"] == (None if ok else "stale_daily")
        assert daily_off["reason_code"] == (None if ok else "daily_unusable")


# --- The 2.1 interim action (CH) ---------------------------------------------------------------

# One session behind the matrix clock's last completed session (Fri 2026-10-09).
STALE_HISTORY = {"last_row_date": "2026-10-08", "last_valid_date": "2026-10-08", "last_row_valid": True,
                 "completed_valid_date": "2026-10-08"}


@pytest.mark.parametrize(("scenario", "action"), [
    ("S1_hold_or_add_clean", "wait_for_data"), ("S2_buy_clean", "wait_for_data"),
    ("T2_starter_weak_setup_quality", "wait_for_data"), ("T3_plain_wait_weak_setup", "wait"),
    ("S3_plain_avoid", "avoid"), ("S9_weak_liquidity", "wait_for_data"), ("ETF_strong", "watch"),
    ("S8_fundamentals_failed", "insufficient_data"),
])
async def test_freshness_reaches_action_now_through_the_hard_gates(scenario: str, action: str) -> None:
    """Interim: a freshness blocker is one more hard gate, so _resolve_action_now sends buy, starter and
    hold_or_add to wait_for_data (watch beside a support gate) and passes the other candidates through.
    CH: flipped by WP2.2 (decide() owns the action: FR-R-F2 action, GC-R3 to R5 rewritten; T3 and S9 move)."""
    stale = {**(dm.SCENARIOS[scenario].technicals or {}), "daily_history": STALE_HISTORY}
    result = await dm.run(scenario, technicals=stale)

    assert freshness_ids(result) == [(STALE_DATA, ["daily_history_stale"])]
    assert result["decision_card"]["action_now"] == action
