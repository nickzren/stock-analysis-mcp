"""Unknown input is not recovery; disclosure fixes retain existing gate actions."""

import copy
import importlib
from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from stock_analysis.tools.analyze.decision_card import evaluate_gates
from stock_analysis.tools.analyze.gates import check_thesis_broken
from stock_analysis.tools.watchlist import _diff, _summary_from_card, _summary_from_screen
from stock_analysis.utils.scan_screen import build_screen_payload, screen_symbol
from stock_analysis.utils.valuation_basis import classify_support
from tests.support import decision_matrix as dm
from tests.support import freshness_frames as fr
from tests.support import swing_matrix as sm
from tests.test_tools.test_trade_card import build_card


def evaluate(**changes: Any) -> tuple[dict[str, Any], frozenset[str]]:
    args = {"events_data": dm.FAR, "data_quality": {}, "technicals_data": dm.STRONG,
            "fundamentals_summary": {}, "risk_data": {}, "support": None, "freshness_blockers": [],
            "thesis_signals": {}, "price_zone": "above_sma50"}
    args.update(changes)
    return evaluate_gates(**args)


def ids(items: list[dict[str, Any]]) -> set[str]:
    return {item["id"] for item in items}


@pytest.mark.parametrize("fund_status,signals,expected", [
    ("unverified", {}, "unevaluated"),
    ("not_applicable", {}, "not_applicable"),
    ("not_meaningful", {"business_evaluable": False, "runway_evaluable": True}, "unevaluated"),
    ("not_meaningful", {"business_evaluable": True, "runway_evaluable": False}, "unevaluated"),
    ("not_meaningful", {"business_evaluable": False, "balance_status": "unsafe"}, "fired"),
    ("not_meaningful", {"business_status": "broken", "runway_evaluable": False}, "fired"),
    ("meaningful", {"business_evaluable": True, "runway_evaluable": True}, "clear"),
])
def test_thesis_checks_known_failures_before_partial_unknowns(fund_status, signals, expected):
    assert check_thesis_broken(fund_status, signals)[0] == expected


@pytest.mark.parametrize("quote_type,currency,unknown", [
    (None, "USD", True), ("EQUITY", None, True), (None, None, True),
    ("ETF", None, False), (None, "GBP", False), ("EQUITY", "GBP", False),
])
def test_liquidity_applicability_unknown_is_not_clear(quote_type, currency, unknown):
    support = classify_support(quote_type=quote_type, currency=currency)
    hard, unevaluated = evaluate(support=support, risk_data=dm.THIN)
    assert ("weak_liquidity" in unevaluated) is unknown
    assert not hard["checks"]["weak_liquidity"]
    card = build_card(summary_data={"quote_type": quote_type, "currency": currency}, risk_data=dm.THIN)
    assert ("weak_liquidity" in ids(card["unevaluated"])) is unknown
    assert "weak_liquidity" not in ids(card["blockers"])


@pytest.mark.parametrize("row,expected", [("R3", False), ("R13", True), ("R1", False)])
async def test_freshness_evaluability_tracks_required_history(row, expected):
    result = await dm.run(fr.scenario(fr.TRUTH_TABLE[row], row))
    assert ("stale_data" in result["watchlist_snapshot"]["unevaluated_gate_ids"]) is expected


@pytest.mark.parametrize("source,codes,completed,behind,stale,unknown", [
    ("intraday_quote", [], "2026-10-12", 0, False, False),
    ("intraday_quote", ["daily_latest_bar_invalid"], "2026-10-12", 0, False, False),
    ("intraday_quote", ["daily_history_unavailable"], None, None, False, True),
    (None, [], "2026-10-12", 0, False, True),
    (None, ["daily_history_unavailable"], None, None, False, True),
    ("daily_close", ["intraday_quote_unavailable"], "2026-10-12", 0, False, True),
    ("daily_close", ["intraday_quote_stale", "daily_history_unavailable"], None, None, True, False),
    (None, ["daily_history_stale"], "2026-10-09", 1, True, False),
])
def test_stale_gate_preserves_known_stale_leg(source, codes, completed, behind, stale, unknown):
    freshness = {"price_source": source, "codes": codes, "daily_completed_session": completed,
                 "daily_sessions_behind": behind}
    blockers = [{"id": "stale_data", "reason": "known stale", "codes": codes}] if stale else []
    hard, unevaluated = evaluate(data_quality={"freshness": freshness}, freshness_blockers=blockers)
    assert hard["checks"]["stale_data"] is stale
    assert ("stale_data" in unevaluated) is unknown


async def compare(monkeypatch, previous, current):
    module = importlib.import_module("stock_analysis.tools.diff_analysis")

    async def analyze(*args, **kwargs):
        return current

    monkeypatch.setattr(module, "analyze_stock", analyze)
    return await module.what_changed("ACME", previous_snapshot=previous["watchlist_snapshot"])


async def test_history_loss_then_recovery_is_not_false_clear(monkeypatch):
    stale = await dm.run(fr.scenario(fr.TRUTH_TABLE["R3"], "stale"))
    missing = await dm.run(fr.scenario(fr.TRUTH_TABLE["R3"], "stale"), technicals=None)
    lost = await compare(monkeypatch, stale, missing)
    assert "stale_data" in lost["changes"]["hard_gates"]["not_evaluated"]
    assert "stale_data" not in lost["changes"]["hard_gates"]["cleared"]
    recovered_stale = await compare(monkeypatch, missing, stale)
    assert "stale_data" in recovered_stale["changes"]["hard_gates"]["new"]
    current = await dm.run(fr.scenario(fr.TRUTH_TABLE["R1"], "current"))
    recovered_current = await compare(monkeypatch, stale, current)
    assert "stale_data" in recovered_current["changes"]["hard_gates"]["cleared"]


@pytest.mark.parametrize("bars,unknown,fired", [(60, True, False), (150, True, False),
                                                (190, False, True), (260, False, True)])
def test_short_knife_history_disclosed_without_suppressing_positive(bars, unknown, fired):
    prices = pd.Series([50 - 30 * i / (bars - 1) for i in range(bars)],
                       index=pd.bdate_range(end="2026-10-12", periods=bars))
    frame = pd.DataFrame({"close": prices, "high": prices + 1, "low": prices - 1,
                          "open": prices, "volume": 2e6})
    frame.attrs.update(instrument_type="EQUITY", currency="USD")
    technicals = build_screen_payload(frame)
    hard, unevaluated = evaluate(technicals_data=technicals)
    assert hard["checks"]["falling_knife"] is fired
    assert ("falling_knife" in unevaluated) is unknown
    card = build_card(technicals_data=technicals)
    assert ("falling_knife" in ids(card["blockers"])) is fired
    assert ("falling_knife" in ids(card["unevaluated"])) is unknown
    screen = screen_symbol(frame, now=fr.PRE_0800, session="pre_market")
    assert ("falling_knife" in screen["blocker_ids"]) is fired
    assert ("falling_knife" in screen["unevaluated"]) is unknown


def test_gapped_knife_inputs_unknown_but_known_non_decline_is_clear():
    partial = copy.deepcopy(dm.POOR_NOT_KNIFE)
    partial["returns"]["return_3m"] = None
    partial["moving_averages"]["sma_200_slope_pct_per_day"] = None
    assert "falling_knife" in evaluate(technicals_data=partial)[1]
    assert "falling_knife" not in evaluate(technicals_data=dm.STRONG)[1]


@pytest.mark.parametrize("summary,unknown,missing", [
    ({"currency": "USD", "quote_type": "EQUITY"}, True, True),
    ({"currency": "USD", "quote_type": "ETF"}, False, False),
    ({"currency": "GBP", "quote_type": "EQUITY"}, False, False),
])
def test_card_missing_liquidity_stays_avoid_with_unknown_gate(summary, unknown, missing):
    card = build_card(summary_data=summary, risk_data={})
    assert ("weak_liquidity" in ids(card["unevaluated"])) is unknown
    assert ("liquidity_missing" in ids(card["blockers"])) is missing
    if missing:
        assert card["action"] == "avoid"


async def test_real_volume_loss_is_unknown_on_card_and_screen(monkeypatch):
    base = sm.ScanScenario("liquidity", sm.BREAKOUT, (3000.0,) * 259 + (7500.0,))
    previous: list[dict[str, Any]] | None = None
    for scenario, weak, unknown in ((base, True, False),
                                    (replace(base, volumes=(float("nan"),) * 260), False, True),
                                    (base, True, False)):
        frame = sm.install(monkeypatch, scenario)
        screen = screen_symbol(frame, now=sm.PRE_MARKET, session="pre_market")
        card = await sm.trade_card()
        for blockers, unevaluated, action in ((set(screen["blocker_ids"]), set(screen["unevaluated"]),
                                               screen["action_hint"]),
                                              (ids(card["blockers"]), ids(card["unevaluated"]), card["action"])):
            assert ("weak_liquidity" in blockers) is weak
            assert ("weak_liquidity" in unevaluated) is unknown
            assert ("liquidity_missing" in blockers) is unknown
            assert action == "avoid"
        current = [_summary_from_card(card), _summary_from_screen(screen)]
        if previous is not None:
            for before, after in zip(previous, current, strict=True):
                changed, reasons = _diff(before, after)
                assert changed and any("liquidity_missing" in reason for reason in reasons)
                assert all("weak_liquidity" not in reason for reason in reasons)
        previous = current


async def test_unknown_currency_does_not_clear_weak_liquidity(monkeypatch):
    prior = await dm.run("S9_weak_liquidity")
    unknown = await dm.run("S9_weak_liquidity", summary={"quote_type": "EQUITY", "currency": None})
    for decision in (unknown["decision_card"]["new_money"],
                     unknown["decision_card"]["long_term"]["new_money"]):
        assert "weak_liquidity" in ids(decision["unevaluated"])
        assert "unsupported_currency" in ids(decision["blocked_by"])
    lost = await compare(monkeypatch, prior, unknown)
    assert "weak_liquidity" in lost["changes"]["hard_gates"]["not_evaluated"]
    assert "weak_liquidity" not in lost["changes"]["hard_gates"]["cleared"]
    recovered = await compare(monkeypatch, unknown, prior)
    assert "weak_liquidity" in recovered["changes"]["hard_gates"]["new"]


async def test_real_technicals_short_history_unknown_reaches_snapshot(monkeypatch):
    scenario = sm.ScanScenario("short", tuple(50 - 30 * i / 59 for i in range(60)), (2e6,) * 60)
    sm.install(monkeypatch, scenario)
    technicals = await sm.technicals_tool.technicals(sm.SYMBOL)
    assert technicals["moving_averages"]["sma_200"] is None
    assert technicals["returns"]["return_3m"] is None
    result = await dm.run("S1_hold_or_add_clean", technicals=technicals)
    assert "falling_knife" not in result["watchlist_snapshot"]["hard_gate_ids"]
    assert "falling_knife" in result["watchlist_snapshot"]["unevaluated_gate_ids"]
    assert "falling_knife" not in ids(result["decision_card"]["new_money"]["blocked_by"])
