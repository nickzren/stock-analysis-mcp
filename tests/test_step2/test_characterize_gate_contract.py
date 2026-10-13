"""WP2.0: gate-contract characterization (annex step2-v2-research/gate_contract.md, "Tests" C1-C4).

Each test records 0609fde behavior. CH tests name the WP that flips them; G tests hold
through 2a. Ported from the annex probes: probe_card_matrix.py and probe_verify_analyze.py F/G
(GC-C1, GC-C2), probe_analyze_matrix.py, probe_verify_policy_cap.py, probe_verify_analyze.py E
and probe_verify_b2_legs.py (GC-C3), probe_scan_vs_card.py and probe_verify_scan.py (GC-C4).
Analyze runs go through tests/support/decision_matrix.py and swing runs through
tests/support/swing_matrix.py; every run pins its clock.
"""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest

from stock_analysis.tools.analyze.decision_card import _compute_hard_gates, _resolve_action_now
from stock_analysis.utils.freshness import build_freshness, freshness_blockers
from stock_analysis.utils.scan_screen import _avg_dollar_volume, screen_symbol
from stock_analysis.utils.valuation_basis import classify_support
from tests.support import decision_matrix as dm
from tests.support import swing_matrix as sm
from tests.test_tools.test_trade_card import build_card

ACTIONABLE = frozenset({"trade_now", "enter_on_trigger"})
PULLBACK = "pullback_in_uptrend"


def ids(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blockers]


# --- GC-C1 / GC-C2: the trade card ---------------------------------------------------------

CardRow = tuple[str, list[str], str | None, tuple[str, float, str] | None]


def card_row(card: dict[str, Any]) -> CardRow:
    """(action, blocker ids in emission order, setup type, plan entry (type, trigger, valid))."""
    plan = card["plan"]
    entry = None if plan is None else (plan["entry"]["type"], plan["entry"]["trigger_price"], plan["entry"]["valid"])
    return card["action"], ids(card["blockers"]), (card["setup"] or {}).get("type"), entry


FROZEN_CARD_ROWS: dict[str, CardRow] = {
    "clean_armed": ("enter_on_trigger", [], PULLBACK, ("buy_stop", 101.0, "good_till_time_stop")),
    "clean_satisfied_regular": ("trade_now", [], PULLBACK, ("market", 101.0, "good_till_time_stop")),
    # data vetoes first, and every other evaluable gate is still disclosed
    "stale_only": ("wait_for_data", ["stale_data"], None, None),
    "unverifiable_freshness": ("wait_for_data", ["freshness_unverifiable"], None, None),
    "stale+knife": ("wait_for_data", ["stale_data", "falling_knife"], None, None),
    "stale+weak_liq": ("wait_for_data", ["stale_data", "weak_liquidity"], None, None),
    "stale+etf": ("wait_for_data", ["stale_data", "unsupported_instrument"], None, None),
    "risk_metrics_failed": ("wait_for_data", ["data_quality_critical", "liquidity_missing"], None, None),
    # then price_risk / liquidity veto to avoid, outranking support and events
    "knife": ("avoid", ["falling_knife"], None, None),
    "knife+etf": ("avoid", ["falling_knife", "unsupported_instrument"], None, None),
    "knife+gbp": ("avoid", ["falling_knife", "unsupported_currency"], None, None),
    "knife+earnings3": ("avoid", ["falling_knife", "earnings_blackout"], None, None),
    "weak_liq": ("avoid", ["weak_liquidity"], None, None),
    "liq_missing": ("avoid", ["liquidity_missing"], None, None),
    # support and events are caps: they never turn no_setup into watch
    "etf_with_setup": ("watch", ["unsupported_instrument"], PULLBACK, None),
    "etf_no_setup": ("no_setup", ["unsupported_instrument"], None, None),
    "gbp_no_setup": ("no_setup", ["unsupported_currency"], None, None),
    "summary_tool_failed": ("watch", ["unsupported_instrument", "unsupported_currency"], PULLBACK, None),
    "no_setup+earnings3": ("no_setup", ["earnings_blackout"], None, None),
    "setup+earnings3": ("watch", ["earnings_blackout"], PULLBACK, None),
    "setup+earn_unverifiable": ("watch", ["earnings_unverifiable"], PULLBACK, None),
    "setup+etf+earnings3": ("watch", ["unsupported_instrument", "earnings_blackout"], PULLBACK, None),
    # market_closed is conditional: the plan survives, even with the trigger already crossed (D2/D6, 2c)
    "off_session_armed": ("enter_on_trigger", ["market_closed"], PULLBACK, ("buy_stop", 101.0, "next_session")),
    "off_session_trigger_crossed": ("enter_on_trigger", ["market_closed"], PULLBACK,
                                    ("buy_stop", 101.0, "next_session")),
    # market_closed and plan_invariant_violation are emitted by code path (Q-GC7: 2c normalizes)
    "off_session+earnings3": ("watch", ["earnings_blackout"], PULLBACK, None),
    "off_session+etf": ("watch", ["unsupported_instrument", "market_closed"], PULLBACK, None),
    "plan_invalid": ("watch", ["plan_invariant_violation"], PULLBACK, None),
    "plan_invalid+off_session": ("watch", ["market_closed", "plan_invariant_violation"], PULLBACK, None),
    "plan_invalid+etf": ("watch", ["unsupported_instrument"], PULLBACK, None),
}

# The rows WP2.1's card disclosure fix changes (plan §4 "Gate inputs"; annex C1).
WP21_CARD_ROWS: dict[str, CardRow] = {
    # earnings_unverifiable is appended only inside the setup branch (card.py:122-127)
    "no_setup+earn_unverifiable": ("no_setup", [], None, None),
    "no_setup+events_tool_failed": ("no_setup", [], None, None),
    "knife+earn_unverifiable": ("avoid", ["falling_knife"], None, None),
    "events_tool_failed+stale": ("wait_for_data", ["stale_data"], None, None),
    # both data_quality_critical branches fire (card.py:68-78)
    "risk_metrics_failed+no_features": ("wait_for_data", ["data_quality_critical", "data_quality_critical",
                                                          "liquidity_missing"], None, None),
}


class TestCardGateMatrix:
    def test_the_matrix_covers_every_ported_card_case(self) -> None:
        """GC-C1: the frozen and WP2.1 rows partition the ported card cases (probe_card_matrix.py plus F/G).
        Holds throughout 2a: the ported case set does not change."""
        assert not set(FROZEN_CARD_ROWS) & set(WP21_CARD_ROWS)
        assert sorted({*FROZEN_CARD_ROWS, *WP21_CARD_ROWS}) == sorted(sm.ALL_CARD_CASES)

    @pytest.mark.parametrize("name", FROZEN_CARD_ROWS)
    def test_frozen_rows(self, name: str) -> None:
        """GC-C1: the card's action, blocker ids, setup and plan entry for each gate combination.
        Holds throughout 2a: the card's rank-first order and emission are frozen (plan §4; 2c owns them)."""
        assert card_row(sm.card_case(name)) == FROZEN_CARD_ROWS[name]

    @pytest.mark.parametrize("name", WP21_CARD_ROWS)
    def test_rows_the_disclosure_fix_changes(self, name: str) -> None:
        """GC-C1: no_setup, avoid and wait_for_data cards omit earnings_unverifiable; dq can repeat.
        Flipped by WP2.1: the card discloses earnings_unverifiable on every outcome and merges ids (GC-R9, R11)."""
        assert card_row(sm.card_case(name)) == WP21_CARD_ROWS[name]


class TestCardActionableBlockers:
    def test_pure_cards(self) -> None:
        """GC-C2: an actionable card's blocker ids are a subset of {market_closed}, and it has a plan.
        Holds throughout 2a: market_closed is the only conditional gate and keeps the plan (R2.6(g))."""
        actionable: dict[str, tuple[str, list[str]]] = {}
        for name in sm.ALL_CARD_CASES:
            card = sm.card_case(name)
            if card["action"] in ACTIONABLE:
                assert set(ids(card["blockers"])) <= {"market_closed"}, name
                assert card["plan"] is not None, name
                actionable[name] = (card["action"], ids(card["blockers"]))

        assert actionable == {
            "clean_armed": ("enter_on_trigger", []), "clean_satisfied_regular": ("trade_now", []),
            "off_session_armed": ("enter_on_trigger", ["market_closed"]),
            "off_session_trigger_crossed": ("enter_on_trigger", ["market_closed"]),
        }

    async def test_pipeline_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GC-C2: the same holds for the real analyze_trade_setup on every GC-C4 frame.
        Holds throughout 2a: market_closed is the only conditional gate and keeps the plan (R2.6(g))."""
        actionable: list[str] = []
        for name, scenario in sm.SCAN_SCENARIOS.items():
            sm.install(monkeypatch, scenario)
            card = await sm.trade_card()
            if card["action"] in ACTIONABLE:
                assert set(ids(card["blockers"])) <= {"market_closed"}, name
                assert card["plan"] is not None, name
                actionable.append(name)

        assert actionable == ["fresh_breakout", "liq_recent_collapse_breakout", "B_attrs_missing_breakout"]


# --- GC-C3: analyze ------------------------------------------------------------------------

US = classify_support(quote_type="EQUITY", currency="USD", symbol="X")
ETF = classify_support(quote_type="ETF", currency="USD", symbol="SPY")
GBP = classify_support(quote_type="EQUITY", currency="GBp", symbol="VOD.L")
LIQ_OK = {"liquidity": {"avg_dollar_volume": 5e8}}
KNIFE_DIP = {"dip_classification": {"type": "falling_knife"}}
POLICY_CODES = ("buy", "speculative_small_position", "hold_or_add", "hold", "wait_for_entry", "hold_or_reduce",
                "avoid")
PASS_THROUGH = ("hold", "wait", "hold_or_reduce", "avoid")  # gates touch only buy, starter and hold_or_add
UNGATED = ("buy", "starter", "hold_or_add", *PASS_THROUGH)
NEW_MONEY_WAIT_FOR_DATA = ("wait_for_data",) * 3 + PASS_THROUGH
NEW_MONEY_WATCH = ("watch",) * 3 + PASS_THROUGH

# name: (_compute_hard_gates inputs, blocking ids, action_now for each of POLICY_CODES)
ANALYZE_GATE_ROWS: dict[str, tuple[dict[str, Any], list[str], tuple[str, ...]]] = {
    "earnings3": ({"events_data": {"earnings": {"days_until": 3}}}, ["earnings_blackout"], NEW_MONEY_WAIT_FOR_DATA),
    "earnings0": ({"events_data": {"earnings": {"days_until": 0}}}, ["earnings_blackout"], NEW_MONEY_WAIT_FOR_DATA),
    "earn_unverifiable": ({"events_data": {"earnings": {"days_until": None, "verification": "unverifiable"}}},
                          [], UNGATED),
    "dq_tool_failed": ({"data_quality": {"tool_failures": [{"tool": "risk_metrics"}]}},
                       ["data_quality_critical"], NEW_MONEY_WAIT_FOR_DATA),
    # the input B3 (WP2.1) retires: a missing beta alone no longer fires data_quality_critical
    "dq_beta_missing": ({"data_quality": {"missing_critical": ["beta"]}}, ["data_quality_critical"],
                        NEW_MONEY_WAIT_FOR_DATA),
    "knife": ({"dip_assessment": KNIFE_DIP}, ["falling_knife"], NEW_MONEY_WAIT_FOR_DATA),
    "weak_liq": ({"risk_data": {"liquidity": {"avg_dollar_volume": 4e5}}}, ["weak_liquidity"],
                 NEW_MONEY_WAIT_FOR_DATA),
    "liq_missing": ({"risk_data": {"liquidity": {}}}, ["liquidity_missing"], NEW_MONEY_WAIT_FOR_DATA),
    "missing_runway": ({"fundamentals_summary": {"burn_metrics": {"status": "unavailable"},
                                                 "valuation": {"valuation_note": "pe_not_meaningful"}}},
                       ["missing_runway"], NEW_MONEY_WAIT_FOR_DATA),
    "etf": ({"support": ETF}, ["unsupported_instrument"], NEW_MONEY_WATCH),
    "gbp": ({"support": GBP}, ["unsupported_currency"], NEW_MONEY_WATCH),
    # support wins over every other gate; the card gives data and knife precedence instead
    "etf+dq_tool_failed": ({"support": ETF, "data_quality": {"tool_failures": [{"tool": "technicals"}]}},
                           ["data_quality_critical", "unsupported_instrument"], NEW_MONEY_WATCH),
    "gbp+knife": ({"support": GBP, "dip_assessment": KNIFE_DIP}, ["falling_knife", "unsupported_currency"],
                  NEW_MONEY_WATCH),
    "knife+dq": ({"dip_assessment": KNIFE_DIP, "data_quality": {"tool_failures": [{"tool": "technicals"}]}},
                 ["data_quality_critical", "falling_knife"], NEW_MONEY_WAIT_FOR_DATA),
    "knife+earnings3": ({"dip_assessment": KNIFE_DIP, "events_data": {"earnings": {"days_until": 3}}},
                        ["earnings_blackout", "falling_knife"], NEW_MONEY_WAIT_FOR_DATA),
}


def hard_gates(inputs: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {"events_data": {}, "data_quality": {}, "dip_assessment": None,
                            "fundamentals_summary": {}, "risk_data": LIQ_OK, "support": US}
    gates: dict[str, Any] = _compute_hard_gates(**{**base, **copy.deepcopy(inputs)})
    return gates


# Probe probe_analyze_matrix.py's crashed chart: the strong chart at 55 with every trend broken.
CRASH = dm.at_price(dm.STRONG, 55.0, above_sma200=False, above_sma50=False, above_sma20=False,
                    golden_cross=False, death_cross=True)
CRASH["moving_averages"]["sma_200_slope_pct_per_day"] = -0.002
CRASH["returns"]["return_3m"] = -0.35
CRASH["price_position"]["position_in_range"] = 0.05

# name: (scenario, action_now, gate ids, policy_action (mid, long, support_gates), dip type or status)
ANALYZE_PIPELINE_ROWS: dict[str, tuple[dm.Scenario, str, list[str], tuple[str, str, Any], str]] = {
    # flipped by WP2.2: the knife is a price_risk veto -> avoid (D-5a), over a wait candidate too (D-5)
    "us_crash": (dm.Scenario("GC analyze_matrix", CRASH, dm.PROFITABLE, dm.CRASH_RISK),
                 "wait", ["falling_knife"], ("wait_for_entry", "accumulate", None), "falling_knife"),
    "gbp_crash": (dm.Scenario("GC analyze_matrix", CRASH, dm.PROFITABLE, dm.CRASH_RISK, summary=dm.GBP_EQUITY),
                  "wait", ["falling_knife", "unsupported_currency"],
                  ("wait_for_entry", "watch", ["unsupported_currency"]), "falling_knife"),
    # flipped by WP2.1: the knife is evaluated for every instrument (today the dip is not_applicable)
    "etf_crash": (dm.Scenario("GC analyze_matrix", CRASH, dm.PROFITABLE, dm.CRASH_RISK, summary=dm.ETF),
                  "avoid", ["unsupported_instrument"], ("avoid", "insufficient_data", ["unsupported_instrument"]),
                  "not_applicable"),
    # flipped by WP2.1: B1 fires earnings_unverifiable in analyze (D-6)
    "crash_earnings_unverifiable": (dm.Scenario("GC analyze_matrix", CRASH, dm.PROFITABLE, dm.CRASH_RISK,
                                                events=sm.UNVERIFIABLE_EVENTS),
                                    "wait", ["falling_knife"], ("wait_for_entry", "accumulate", None), "falling_knife"),
    "strong_earnings_unverifiable": (dm.Scenario("GC analyze_matrix", dm.STRONG, dm.PROFITABLE, dm.CALM,
                                                 events=sm.UNVERIFIABLE_EVENTS),
                                     "hold_or_add", [], ("hold_or_add", "accumulate", None), "healthy_pullback"),
    # flipped by WP2.3: policy_action moves to diagnostics.pre_gate_policy (D-9); the equity's
    # hold_or_add also becomes starter in WP2.2 (D-2)
    "etf_strong": (dm.Scenario("GC verify_policy_cap", dm.STRONG, dm.PROFITABLE, dm.CALM, summary=dm.ETF),
                   "watch", ["unsupported_instrument"], ("watch", "insufficient_data", ["unsupported_instrument"]),
                   "not_applicable"),
    "equity_strong": (dm.Scenario("GC verify_policy_cap", dm.STRONG, dm.PROFITABLE, dm.CALM),
                      "hold_or_add", [], ("hold_or_add", "accumulate", None), "healthy_pullback"),
}


def dip_label(result: dict[str, Any]) -> str:
    dip = result["dip_assessment"]
    return str((dip.get("dip_classification") or {}).get("type") or dip.get("status"))


REGULAR_SESSION = dm.ET.localize(datetime(2026, 3, 10, 11, 0))  # Tuesday; last completed session 2026-03-09
FIVE_MINUTE_BAR = "2026-03-10T14:55:00+00:00"  # 10:55 ET
DAILY_FRAME_FEB_20 = {"price": {"source": "yfinance", "as_of": "2026-03-10T15:00:00Z", "last_bar_date": "2026-02-20"}}
FRESH_QUOTE_STALE_DAILY = dm.Scenario(
    "GC verify_b2_legs",
    {**dm.STRONG, "data_provenance": DAILY_FRAME_FEB_20,
     "intraday_quote": {"close": 100.0, "bar_time": FIVE_MINUTE_BAR, "fetched_at": None, "cache_hit": False},
     "session_close": {"close": 99.0, "session_date": "2026-02-20", "fetched_at": None, "cache_hit": False}},
    dm.PROFITABLE, {**dm.CALM, "data_provenance": DAILY_FRAME_FEB_20}, now=REGULAR_SESSION,
)


class TestAnalyzeGateBefore:
    @pytest.mark.parametrize("name", ANALYZE_GATE_ROWS)
    def test_gate_resolution_matrix(self, name: str) -> None:
        """GC-C3: every analyze gate sends buy, starter and hold_or_add to wait_for_data (support: watch).
        Flipped by WP2.2 (decide(): D-2 mapping, D-5 most-restrictive, D-5a outcomes); earn_unverifiable by WP2.1 (B1),
        and dq_beta_missing's input is retired by WP2.1 (B3)."""
        inputs, expected_ids, expected_actions = ANALYZE_GATE_ROWS[name]
        gates = hard_gates(inputs)

        assert ids(gates["blocking"]) == expected_ids
        assert tuple(_resolve_action_now({"mid_term": code}, gates)[0] for code in POLICY_CODES) == expected_actions

    @pytest.mark.parametrize("name", ANALYZE_PIPELINE_ROWS)
    async def test_pipeline_rows(self, name: str) -> None:
        """GC-C3: full analyze on crashed, strong and unsupported charts (the knife is skipped for an ETF).
        Flipped row by row by WP2.1 (knife scope, B1), WP2.2 (D-2, D-5a) and WP2.3 (D-9), as each row notes."""
        scenario, action_now, expected_ids, (mid, long, support_gates), dip = ANALYZE_PIPELINE_ROWS[name]
        result = await dm.run(scenario)
        policy = result["policy_action"]

        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == (action_now, expected_ids)
        assert (policy["mid_term"], policy["long_term"], policy.get("support_gates")) == (mid, long, support_gates)
        assert dip_label(result) == dip

    async def test_the_support_cap_runs_before_the_decision_card(self) -> None:
        """GC-C3: an ETF's policy candidate is already watch, so the card's support branch never decides.
        Flipped by WP2.3: the capped candidate moves to diagnostics.pre_gate_policy, the rationale to decision_trace."""
        result = await dm.run(ANALYZE_PIPELINE_ROWS["etf_strong"][0])

        assert result["decision_card"]["rationale"] == ["policy_action.mid_term=watch -> watch"]

    async def test_a_stock_summary_failure_splits_the_gates_not_the_action(self) -> None:
        """GC-C3: with stock_summary down, analyze shows watch [dq, unsupported_*] and the card watch [unsupported_*].
        Holds throughout 2a: most-restrictive keeps data+support at watch (D-5) and the card keeps watch (Q-GC2 (i))."""
        failed = {"error": True, "error_type": "data_unavailable", "message": "upstream"}
        result = await dm.run(dm.Scenario("GC verify_analyze E", dm.STRONG, dm.PROFITABLE, dm.CALM, summary=failed))
        card = build_card(summary_data={}, tool_failures=[{"tool": "stock_summary", "error": "x"}])

        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == (
            "watch", ["data_quality_critical", "unsupported_instrument", "unsupported_currency"])
        assert (card["action"], ids(card["blockers"])) == ("watch", ["unsupported_instrument", "unsupported_currency"])

    async def test_a_fresh_quote_hides_a_stale_daily_frame_from_analyze(self) -> None:
        """GC-C3: in session, analyze prices off a fresh 5m quote and only warns that the daily frame is 11
        sessions behind; the card's build_freshness blocks the same legs as stale_daily.
        Flipped by WP2.1: analyze's freshness gate checks the daily leg in every session (D-7, GC-R12)."""
        result = await dm.run(FRESH_QUOTE_STALE_DAILY)
        provenance = result["summary"]["price_provenance"]
        data_quality = result["data_quality"]
        daily = pd.DataFrame({"date": ["2026-02-19", "2026-02-20"], "close": [99.0, 99.0]})
        probe = pd.DataFrame({"date": [pd.Timestamp(FIVE_MINUTE_BAR)], "close": [100.0]})
        freshness = build_freshness(intraday_df=probe, daily_df=daily, session="regular", now=REGULAR_SESSION)

        assert (provenance["price_source"], provenance["staleness"]["stale"]) == ("intraday_quote", False)
        assert {k: data_quality["component_freshness"]["price"][k] for k in ("sessions_behind", "stale")} == {
            "sessions_behind": 11, "stale": True}
        assert data_quality["staleness_warnings"] == ["price_stale_11_sessions", "risk_stale_11_sessions"]
        assert (result["decision_card"]["action_now"], dm.gate_ids(result)) == ("hold_or_add", [])
        assert (freshness["reason_code"], ids(freshness_blockers(freshness))) == ("stale_daily", ["stale_data"])


# --- GC-C4: scan stage 1 vs the card -------------------------------------------------------


Outcome = tuple[str, str | None, list[str]]  # action, setup type, blocker ids


class ScanBefore(NamedTuple):
    screen: Outcome  # stage 1: action_hint, setup_type, blocker_ids
    card: Outcome  # analyze_trade_setup
    row: tuple[str, str, list[str]]  # first scan: the stage that produced the row, its action, its blockers
    forced: tuple[str, list[str]]  # prior state enter_on_trigger, so the card decides: action, blockers


PROMOTED: Outcome = ("candidate", "breakout", [])
NO_SETUP: Outcome = ("no_setup", None, [])
CAPPED_SCREEN: Outcome = ("watch", None, ["unsupported_instrument"])  # support is terminal at stage 1
STALE = ["stale_data"]
SHORT_FRAME = ["data_quality_critical", "liquidity_missing"]  # risk_metrics needs 50 bars
ARMED_OFF_SESSION: Outcome = ("enter_on_trigger", "breakout", ["market_closed"])
ETF_BREAKOUT = ScanBefore(CAPPED_SCREEN, ("watch", "breakout", ["unsupported_instrument", "market_closed"]),
                          ("screen", "watch", ["unsupported_instrument"]),
                          ("watch", ["market_closed", "unsupported_instrument"]))

SCAN_BEFORE: dict[str, ScanBefore] = {
    "fresh_breakout": ScanBefore(PROMOTED, ARMED_OFF_SESSION, ("card", "enter_on_trigger", ["market_closed"]),
                                 ("enter_on_trigger", ["market_closed"])),
    "stale_frame_boring": ScanBefore(NO_SETUP, ("wait_for_data", None, STALE), ("screen", "no_setup", []),
                                     ("wait_for_data", STALE)),
    "stale_frame_breakout": ScanBefore(PROMOTED, ("wait_for_data", None, STALE), ("card", "wait_for_data", STALE),
                                       ("wait_for_data", STALE)),
    "liq_recent_collapse_breakout": ScanBefore(("avoid", None, ["weak_liquidity"]), ARMED_OFF_SESSION,
                                               ("screen", "avoid", ["weak_liquidity"]),
                                               ("enter_on_trigger", ["market_closed"])),
    "liq_recent_surge_breakout": ScanBefore(PROMOTED, ("avoid", None, ["weak_liquidity"]),
                                            ("card", "avoid", ["weak_liquidity"]), ("avoid", ["weak_liquidity"])),
    "etf_boring": ScanBefore(CAPPED_SCREEN, ("no_setup", None, ["unsupported_instrument"]),
                             ("screen", "watch", ["unsupported_instrument"]), ("no_setup", ["unsupported_instrument"])),
    "etf_breakout": ETF_BREAKOUT,
    "boring_earnings_in_2d": ScanBefore(NO_SETUP, ("no_setup", None, ["earnings_blackout"]),
                                        ("screen", "no_setup", []), ("no_setup", ["earnings_blackout"])),
    "breakout_earnings_in_2d": ScanBefore(PROMOTED, ("watch", "breakout", ["earnings_blackout"]),
                                          ("card", "watch", ["earnings_blackout"]), ("watch", ["earnings_blackout"])),
    "breakout_events_unverifiable": ScanBefore(PROMOTED, ("watch", "breakout", ["earnings_unverifiable"]),
                                               ("card", "watch", ["earnings_unverifiable"]),
                                               ("watch", ["earnings_unverifiable"])),
    "A_30_bars_breakout": ScanBefore(PROMOTED, ("wait_for_data", None, SHORT_FRAME),
                                     ("card", "wait_for_data", SHORT_FRAME), ("wait_for_data", SHORT_FRAME)),
    "A_30_bars_boring": ScanBefore(NO_SETUP, ("wait_for_data", None, SHORT_FRAME), ("screen", "no_setup", []),
                                   ("wait_for_data", SHORT_FRAME)),
    "A_45_bars_boring": ScanBefore(NO_SETUP, ("wait_for_data", None, SHORT_FRAME), ("screen", "no_setup", []),
                                   ("wait_for_data", SHORT_FRAME)),
    # The history attrs carry no instrument_type; stock_summary says EQUITY/USD.
    "B_attrs_missing_breakout": ScanBefore(CAPPED_SCREEN, ARMED_OFF_SESSION,
                                           ("screen", "watch", ["unsupported_instrument"]),
                                           ("enter_on_trigger", ["market_closed"])),
    "B_attrs_missing_boring": ScanBefore(CAPPED_SCREEN, NO_SETUP, ("screen", "watch", ["unsupported_instrument"]),
                                         ("no_setup", [])),
    "C_stale_knife": ScanBefore(("avoid", None, ["falling_knife"]),
                                ("wait_for_data", None, ["stale_data", "falling_knife"]),
                                ("screen", "avoid", ["falling_knife"]),
                                ("wait_for_data", ["falling_knife", "stale_data"])),
    "C_stale_thin": ScanBefore(("avoid", None, ["weak_liquidity"]),
                               ("wait_for_data", None, ["stale_data", "weak_liquidity"]),
                               ("screen", "avoid", ["weak_liquidity"]),
                               ("wait_for_data", ["stale_data", "weak_liquidity"])),
    "D_etf_breakout": ETF_BREAKOUT,
}


# Fresh, liquid, supported 260-bar frames: no gate stage 1 gains in WP2.1 fires on them, and
# earnings stay unevaluated at stage 1, so WP2.1 leaves every column of these rows as it is.
ROWS_WP21_KEEPS = ("fresh_breakout", "boring_earnings_in_2d", "breakout_earnings_in_2d",
                   "breakout_events_unverifiable")


def scan_row(result: dict[str, Any]) -> tuple[str, str, list[str]]:
    """(the stage that produced the scan's only row, its action, its blockers)."""
    (row,) = result["rows"]
    return "card" if result["full_cards"] else "screen", row["action"], row["blockers"]


async def check_before_row(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stage 1, the card, a first scan and a card-forced scan on one frame, against SCAN_BEFORE."""
    expected = SCAN_BEFORE[name]
    df = sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])

    screen = screen_symbol(df.copy())
    card = await sm.trade_card()
    first = await sm.scan(tmp_path / "first")
    forced = await sm.scan(tmp_path / "forced", prior_action="enter_on_trigger")

    assert (screen["action_hint"], screen["setup_type"], screen["blocker_ids"]) == expected.screen
    assert screen["promote"] is (screen["action_hint"] == "candidate")
    assert (card["action"], (card["setup"] or {}).get("type"), ids(card["blockers"])) == expected.card
    assert scan_row(first) == expected.row
    assert first["rows"][0]["earnings_in_days"] == (
        card["event_risk"]["earnings_in_days"] if expected.row[0] == "card" else None)
    assert scan_row(forced) == ("card", *expected.forced)


# name: (the card-forced row, then the screen row: action and blockers; the notes the second scan reports)
PHANTOM_CHANGES: dict[str, tuple[tuple[str, list[str]], tuple[str, list[str]], list[str]]] = {
    # stage 1 never evaluates earnings, so the screen row drops the card's blackout
    "boring_earnings_in_2d": (("no_setup", ["earnings_blackout"]), ("no_setup", []),
                              ["blockers: ['earnings_blackout'] -> []"]),
    # the card's residual market_closed, and the setup the screen drops (probe_verify_scan.py D)
    "D_etf_breakout": (("watch", ["market_closed", "unsupported_instrument"]), ("watch", ["unsupported_instrument"]),
                       ["setup_type: breakout -> None",
                        "blockers: ['market_closed', 'unsupported_instrument'] -> ['unsupported_instrument']",
                        "trigger_price: 111.96 -> None"]),
}


class TestScanVsCardBefore:
    def test_the_table_covers_every_ported_scenario(self) -> None:
        """GC-C4: the before table has a row for each probe_scan_vs_card.py and probe_verify_scan.py frame.
        Holds throughout 2a: WP2.1 changes rows, not the scenario set."""
        assert list(SCAN_BEFORE) == list(sm.SCAN_SCENARIOS)
        assert set(ROWS_WP21_KEEPS) < set(SCAN_BEFORE)

    @pytest.mark.parametrize("name", [name for name in SCAN_BEFORE if name not in ROWS_WP21_KEEPS])
    async def test_before_table(self, name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """GC-C4: stage 1, the card and the scan rows on one frame (stale, 21-49 bars, liquidity basis, ETF,
        missing attrs), pre-market. Flipped by WP2.1: stage 1 evaluates the card's gates, then
        resolves rank-first (D-11; GC-R14-R18)."""
        await check_before_row(name, monkeypatch, tmp_path)

    @pytest.mark.parametrize("name", ROWS_WP21_KEEPS)
    async def test_before_table_rows_wp21_keeps(
        self, name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """GC-C4: the same columns on fresh, liquid, supported frames (clean breakout, earnings in 2 days,
        unverifiable calendar). Holds throughout 2a: no gate stage 1 gains in WP2.1 fires on these frames."""
        await check_before_row(name, monkeypatch, tmp_path)

    @pytest.mark.parametrize(("name", "screen_millions", "card_millions"),
                             [("liq_recent_collapse_breakout", 0.11, 3.07), ("liq_recent_surge_breakout", 5.49, 0.68)])
    async def test_stage1_and_the_card_measure_liquidity_differently(
        self, name: str, screen_millions: float, card_millions: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GC-C4: stage 1 averages close x volume over the prior 20 bars; the card multiplies the 1y mean
        volume by the last close. Flipped by WP2.1: stage 1 uses the card's basis (D-11, GC-R14)."""
        df = sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])
        risk = await sm.risk_tool.risk_metrics(sm.SYMBOL)

        assert round(_avg_dollar_volume(df) / 1e6, 2) == screen_millions
        assert round(risk["liquidity"]["avg_dollar_volume"] / 1e6, 2) == card_millions

    @pytest.mark.parametrize("name", PHANTOM_CHANGES)
    async def test_a_card_then_screen_sequence_reports_a_phantom_change(
        self, name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """GC-C4: a frame scanned by the card (prior state actionable), then by the screen, is reported as
        changed although nothing moved. Flipped by WP2.1: screen rows keep the setup and _diff skips
        (id, leg) pairs either side left unevaluated (GC-R17, GC-R21; plan §0 #20)."""
        (card_action, card_blockers), screen_row, notes = PHANTOM_CHANGES[name]
        sm.install(monkeypatch, sm.SCAN_SCENARIOS[name])

        by_card = await sm.scan(tmp_path, prior_action="enter_on_trigger")
        by_screen = await sm.scan(tmp_path)

        assert scan_row(by_card) == ("card", card_action, card_blockers)
        assert scan_row(by_screen) == ("screen", *screen_row)
        assert by_screen["changes"] == [{"symbol": sm.SYMBOL, "from": card_action, "to": card_action,
                                         "notes": notes}]
