"""Observed v0.4.0 behavior. Later packages replace defect pins, never the before files."""

import hashlib
import json
from unittest.mock import patch

import pytest

from stock_analysis.tools import position
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm

BEFORE = json.loads((hm.FIXTURE_DIR / "before.json").read_text())


@pytest.mark.parametrize("name", hm.STOCK_CASES)
async def test_stock_observation_includes_inputs_decisions_sizing_and_complete_snapshot(name):
    reads = {}
    result = await hm.run_stock(name, reads=reads)
    row = BEFORE["stocks"][name]
    assert reads == row["source_reads"] == dict.fromkeys(dm.TOOLS, 1)
    case = hm.STOCK_CASES[name]
    assert row["inputs"] == case.payloads()
    assert row["profile"] == case.profile
    assert row["account_size"] == case.account_size
    assert hm.stock_digest(result) == row["result"]
    for tier, payload in dm.tiers(result).items():
        assert payload["watchlist_snapshot"] == result["watchlist_snapshot"], tier
        card = payload["decision_card"]
        for key in ("new_money", "if_held", "long_term", "sizing", "sizing_status"):
            assert card[key] == result["decision_card"][key], (name, tier, key)
        assert "if_held" not in card["long_term"]  # absent, not known-null
    assert result["watchlist_snapshot"]["snapshot_version"] == "3.0.0"
    assert "long_term_if_held_action" not in result["watchlist_snapshot"]


@pytest.mark.parametrize("name", hm.POSITION_CASES)
async def test_position_observation_preserves_all_output_except_duration(name):
    row = await hm.run_position(name)
    assert row == BEFORE["positions"][name]
    result = row["result"]
    assert not {"position_card", "holder_assessment", "review_status", "concentration_check"}.intersection(result)
    assert row["technical_reads"] == (0 if name == "invalid_date" else 1)


async def test_same_chart_hides_broken_business_from_legacy_position_tool():
    sound = await hm.run_stock("sound_bullish")
    broken = await hm.run_stock("broken_bullish")
    assert sound["decision_card"]["if_held"]["action"] == "hold"
    assert broken["decision_card"]["if_held"]["action"] == "review_exit"
    assert await hm.run_position("gain") == await hm.run_position("broken_same_chart")


async def test_current_midterm_etf_answer_is_not_a_longterm_holder_assessment():
    strong = await hm.run_stock("etf_spy")
    weak = await hm.run_stock("etf_bearish")
    assert strong["decision_card"]["if_held"]["action"] == "hold"
    assert weak["decision_card"]["if_held"]["action"] == "reduce"
    for result in (strong, weak):
        assert result["summary"]["support"]["instrument"]["status"] == "unsupported"
        assert "if_held" not in result["decision_card"]["long_term"]
    assert (await hm.run_position("etf"))["result"] == (await hm.run_position("gain"))["result"]


async def test_unknown_thesis_is_distinct_from_confirmed_negative_fcf_and_nonburner():
    for name in ("missing_fcf_and_debt", "fundamentals_failed", "unknown_runway", "classification_unknown"):
        result = await hm.run_stock(name)
        assert result["decision_card"]["if_held"]["action"] == "insufficient_data"
        assert "thesis_broken" in result["watchlist_snapshot"]["unevaluated_gate_ids"]
    for name in ("confirmed_negative_fcf", "negative_fcf_no_debt", "nonburner_no_cash"):
        result = await hm.run_stock(name)
        assert result["decision_card"]["if_held"]["action"] == "hold"
        assert "thesis_broken" not in result["watchlist_snapshot"]["unevaluated_gate_ids"]
    nonburner = (await hm.run_stock("nonburner_no_cash"))["fundamentals_summary"]["burn_metrics"]
    assert nonburner["status"] == "not_applicable"
    assert nonburner["cash_runway_quarters"] is None


async def test_watch_balance_factors_are_not_critical_runway():
    for name in ("net_debt", "negative_equity", "watch_runway"):
        result = await hm.run_stock(name)
        assert result["dislocation_framework"]["balance_sheet_safety"]["status"] == "watch"
        assert result["decision_card"]["if_held"]["action"] == "hold"
    critical = await hm.run_stock("critical_runway")
    assert critical["decision_card"]["if_held"]["basis"] == "runway_critical"
    assert critical["fundamentals_summary"]["burn_metrics"]["cash_runway_quarters"] == 2.0


async def test_legacy_position_does_not_enforce_currency_or_use_analyze_selected_close():
    gain = (await hm.run_position("gain"))["result"]
    assert (await hm.run_position("eur"))["result"] == gain
    assert (await hm.run_stock("eur_stock"))["summary"]["currency"] == "EUR"
    position_result = (await hm.run_position("price_disagreement"))["result"]
    analyzed = await hm.run_stock("price_disagreement")
    assert analyzed["summary"]["current_price"] == 100
    assert analyzed["summary"]["price_provenance"]["price_source"] == "daily_close"
    assert position_result["position"]["current_price"] == 105
    assert (await hm.run_position("stale_close"))["result"]["position"]["current_value"] == 2000
    assert (await hm.run_stock("stale_daily"))["summary"]["price_provenance"]["staleness"]["stale"] is True
    assert gain["support_levels"]["atr_1x_below"] == 98
    assert position_result["support_levels"]["atr_1x_below"] == 103  # recomputed, not anchored


async def test_personal_inputs_only_change_legacy_position_amounts_and_tax_not_signals():
    rows = [(await hm.run_position(name))["result"] for name in (
        "gain", "loss", "fractional_shares", "no_shares", "exactly_366_days")]
    for row in rows:
        assert row["current_signals"] == rows[0]["current_signals"]
        assert row["sell_signals"] == rows[0]["sell_signals"]
        assert "watchlist_snapshot" not in row
    gain = rows[0]
    assert gain["position"]["current_value"] == 2000
    assert gain["position"]["gain_loss_dollars"] == 400
    assert gain["tax"]["rate_if_sold_now"] == .37
    assert gain["tax"]["potential_savings"] == 68
    assert rows[-1]["tax"]["rate_if_sold_now"] == .20
    assert (await hm.run_position("exactly_365_days"))["result"]["holding"]["is_long_term"] is False
    assert (await hm.run_position("exactly_366_days"))["result"]["holding"]["is_long_term"] is True


async def test_source_inventory_is_one_technicals_read_and_no_analyze(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("legacy position started full analysis")
    monkeypatch.setattr(dm.orchestrator, "analyze_stock", forbidden)
    row = await hm.run_position("gain")
    assert row["technical_reads"] == 1
    assert row["result"]["meta"]["schema_version"] == "7"


def test_clock_is_explicit_host_local_and_not_wall_clock_dependent():
    with patch.object(position, "datetime", hm.PositionClock):
        result = position._build_holding(purchase_date="2024-02-29", purchase_dt=hm.PositionClock(2024, 2, 29))
    assert result["days_held"] == 953


@pytest.mark.parametrize("filename,expected", [
    ("before.json", "8394ba9b3e87f0f2d506470988d219c7a27ffe590dc2599c863c5eeb6e5db140"),
    ("before.md", "48e1c6446989d30ba38bf5abd164a93c408e5445682496b89034c4642362a296"),
])
def test_frozen_before_hash_survives_later_holder_changes(filename, expected):
    assert hashlib.sha256((hm.FIXTURE_DIR / filename).read_bytes()).hexdigest() == expected


async def test_hold_with_avoid_fit_keeps_the_actual_risk_reasons():
    result = await hm.run_stock("hold_with_avoid_fit")
    assert result["decision_card"]["if_held"]["action"] == "hold"
    assert result["verdict"]["horizon_fit"]["long_term"] == "avoid"
    assert "long_term: extreme_risk + revenue_decline_40pct + negative_fcf" in (
        result["verdict"]["horizon_fit"]["reasons"])
