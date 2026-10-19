"""Observed v0.4.0 behavior. Later packages replace defect pins, never the before files."""

import copy
import hashlib
import json

import pytest

from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm
from tests.support import output_contract as oc

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
    previous_fields = hm.stock_digest(result)
    assert isinstance(previous_fields["decision_card"]["long_term"].pop("if_held"), dict)
    before = copy.deepcopy(row["result"])
    old_snapshot = before.pop("watchlist_snapshot")
    snapshot = previous_fields.pop("watchlist_snapshot")
    assert {key: snapshot[key] for key in old_snapshot if key not in {"snapshot_version", "snapshot_hash"}} == {
        key: value for key, value in old_snapshot.items() if key not in {"snapshot_version", "snapshot_hash"}}
    old_segments = oc.segments(before.pop("executive_summary"))
    new_segments = oc.segments(previous_fields.pop("executive_summary"))
    assert new_segments[:2] == old_segments[:2]
    assert new_segments[3:] == old_segments[3:]
    assert previous_fields == before
    for tier, payload in dm.tiers(result).items():
        assert payload["watchlist_snapshot"] == result["watchlist_snapshot"], tier
        card = payload["decision_card"]
        for key in ("new_money", "if_held", "long_term", "sizing", "sizing_status"):
            assert card[key] == result["decision_card"][key], (name, tier, key)
        assert isinstance(card["long_term"]["if_held"], dict)
    assert result["watchlist_snapshot"]["snapshot_version"] == "4.0.0"
    assert result["watchlist_snapshot"]["long_term_if_held_action"] == result["decision_card"]["long_term"]["if_held"]["action"]


INVALID_POSITIONS = {"negative_cost", "zero_shares", "negative_shares", "future_date", "invalid_date"}


@pytest.mark.parametrize("name", hm.POSITION_CASES)
async def test_position_migration_preserves_inputs_but_replaces_legacy_decision_and_tax(name):
    row = await hm.run_position(name)
    assert row["arguments"] == BEFORE["positions"][name]["arguments"]
    assert row["technicals_input"] == BEFORE["positions"][name]["technicals_input"]
    result = row["result"]
    expected_reads = 0 if name in INVALID_POSITIONS else 1
    assert row["source_reads"] == dict.fromkeys(dm.TOOLS, expected_reads)
    if name in INVALID_POSITIONS:
        assert result["error_type"] == "invalid_parameters"
        return
    assert result["position_card"]["horizon"] == "mid_term"
    assert result["position_card"]["review_status"] == "not_requested"
    assert result["tax"]["status"] == "not_assessed"
    assert set(result["tax"]) == {"status", "reason"}
    assert set(result["holding"]) == {"purchase_date", "days_held"}
    assert not {"current_signals", "sell_signals", "support_levels"}.intersection(result)
    assert "active_count" not in result["diagnostics"]["technicals"]["flags"]


async def test_same_chart_no_longer_hides_broken_business_from_position_tool():
    sound = (await hm.run_position("gain"))["result"]
    broken = (await hm.run_position("broken_same_chart"))["result"]
    assert sound["position_card"]["holder_assessment"]["action"] == "hold"
    assert broken["position_card"]["holder_assessment"]["action"] == "review_exit"
    assert sound["diagnostics"]["technicals"] == broken["diagnostics"]["technicals"]


async def test_etf_has_separate_longterm_coverage_and_personal_money_context():
    analyzed = await hm.run_stock("etf_spy")
    mid = (await hm.run_position("etf"))["result"]
    long = (await hm.run_position("etf", horizon="long_term", portfolio_value=10_000,
                                  max_holding_pct=10))["result"]
    assert mid["position_card"]["holder_assessment"] == analyzed["decision_card"]["if_held"]
    assert long["position_card"]["holder_assessment"] == analyzed["decision_card"]["long_term"]["if_held"]
    assert long["position_card"]["holder_assessment"]["action"] == "not_assessed"
    assert long["position_card"]["review_status"] == "review_allocation"
    assert long["position"]["current_value"] == mid["position"]["current_value"] == 2000


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


async def test_position_uses_analyze_selected_price_and_applies_currency_boundary():
    gain = (await hm.run_position("gain"))["result"]
    eur = (await hm.run_position("eur"))["result"]
    assert eur["position"]["current_value"] is None
    assert eur["position"]["gain_loss_dollars"] is None
    assert eur["position_card"]["holder_assessment"]["action"] == "hold"
    position_result = (await hm.run_position("price_disagreement"))["result"]
    analyzed = await hm.run_stock("price_disagreement")
    assert position_result["position"]["current_price"] == analyzed["summary"]["current_price"] == 100
    assert gain["diagnostics"]["technicals"]["reference_levels"]["atr_1x_below"] == 98
    assert position_result["diagnostics"]["technicals"]["reference_levels"]["atr_1x_below"] == 98
    stale = (await hm.run_position("stale_close", portfolio_value=10_000, max_holding_pct=10))["result"]
    assert stale["position"]["current_value"] == 2000
    assert stale["position_card"]["review_status"] == "not_evaluated"


async def test_personal_inputs_change_only_context_and_never_create_tax_or_exit_guidance():
    rows = [(await hm.run_position(name))["result"] for name in (
        "gain", "loss", "fractional_shares", "no_shares", "exactly_366_days")]
    for row in rows:
        assert row["diagnostics"]["technicals"] == rows[0]["diagnostics"]["technicals"]
        assert row["position_card"]["holder_assessment"] == rows[0]["position_card"]["holder_assessment"]
        assert row["position_card"]["new_money"] == rows[0]["position_card"]["new_money"]
        assert "watchlist_snapshot" not in row
        assert row["tax"]["status"] == "not_assessed"
        assert set(row["holding"]) == {"purchase_date", "days_held"}
    assert rows[0]["position"]["current_value"] == 2000
    assert rows[0]["position"]["gain_loss_dollars"] == 400
    assert (await hm.run_position("exactly_365_days"))["result"]["holding"]["days_held"] == 365
    assert rows[-1]["holding"]["days_held"] == 366


async def test_source_inventory_changes_from_technicals_only_to_one_full_analysis():
    row = await hm.run_position("gain")
    assert BEFORE["positions"]["gain"]["technical_reads"] == 1
    assert row["source_reads"] == dict.fromkeys(dm.TOOLS, 1)
    assert row["result"]["meta"]["schema_version"] == "8"


async def test_position_clock_is_explicit_et_not_host_wall_time():
    result = (await hm.run_position("leap_purchase"))["result"]
    assert hm.POSITION_CLOCK.isoformat() == "2026-10-09T20:00:00-04:00"
    assert result["holding"]["days_held"] == 953


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
