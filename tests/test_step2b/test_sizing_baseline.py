"""Preserve the v0.3.0 recording and identify each approved sizing change.

The shared calculator now governs public sizing; decisions and snapshots remain
identical to the frozen before evidence.
"""

import copy
import hashlib
import json
from contextlib import suppress

import pytest
import requests

from tests.support import decision_matrix as dm
from tests.support import sizing_matrix as sm

BEFORE = json.loads((sm.FIXTURE_DIR / "before.json").read_text())


@pytest.mark.parametrize("name", sm.CASES)
async def test_current_sizing_matches_after_and_preserves_decisions_and_snapshots(name):
    result = await sm.run_case(name)
    row = sm.digest(name, result)
    after = json.loads((sm.FIXTURE_DIR / "after.json").read_text())["analyze"][name]
    assert row == after
    assert row["decision"] == BEFORE["analyze"][name]["decision"]
    assert row["snapshot"] == BEFORE["analyze"][name]["snapshot"]
    for tier, payload in dm.tiers(result).items():
        assert payload["watchlist_snapshot"] == result["watchlist_snapshot"], tier
        assert payload["decision_card"]["sizing"] == result["decision_card"]["sizing"], tier
        assert payload["decision_card"]["sizing_status"] == result["decision_card"]["sizing_status"], tier


@pytest.mark.parametrize(("name", "starter", "full", "whole"), [
    ("balanced_wide", 200.0, 500.0, 2),
    ("risk_only", 100.0, 100.0, 1),
    ("cap_only_tight", 200.0, 200.0, 2),
    ("cap_only_wide", 200.0, 500.0, 2),
    ("both_overrides", 0.0, 0.0, 0),
    ("tiny_cap", 0.0, 0.0, 0),
    ("small_account", 0.0, 50.0, 0),
])
async def test_whole_share_amounts_enforce_default_and_override_limits(name, starter, full, whole):
    result = await sm.run_case(name)
    size = result["decision_card"]["sizing"]
    assert (size["starter_dollars"], size["full_dollars"], size["whole_shares"]) == (starter, full, whole)
    assert result["decision_card"]["new_money"]["action"] == "buy"
    for scenario in (size['starter'], size['full']):
        assert scenario['planned_loss_at_stop'] <= size['limits']['risk_budget_dollars']
        assert scenario['position_dollars'] <= scenario['allocation_cap_dollars']


async def test_stop_budget_and_tiny_limits_are_enforced_without_rounding():
    card = (await sm.run_case("risk_only"))["decision_card"]
    assert card['sizing']['full']['allocation_cap_pct'] == 1.25
    assert card['sizing']['full']['planned_loss_at_stop'] == 20
    assert card['sizing']['limits']['risk_budget_dollars'] == 25
    tiny = (await sm.run_case("tiny_risk"))['decision_card']['sizing']
    assert tiny['limits']['risk_per_trade_pct'] == .001
    assert tiny['full']['allocation_cap_pct'] == .005
    cap = (await sm.run_case("tiny_cap"))['decision_card']['sizing']
    assert cap['full']['allocation_cap_pct'] == .06


async def test_sizing_inputs_do_not_change_decisions_or_any_snapshot_field():
    reference = await sm.run_case("balanced_wide")
    names = ["core_wide", "speculative_wide", "risk_only", "cap_only_tight", "cap_only_wide",
             "both_overrides", "tiny_risk", "tiny_cap", "no_account"]
    for name in names:
        result = await sm.run_case(name)
        assert result["watchlist_snapshot"] == reference["watchlist_snapshot"], name
        assert dm.digest(result)["decision"] == dm.digest(reference)["decision"], name


async def test_missing_invalid_stop_and_inverted_band_suppress_only_unavailable_estimates():
    missing = await sm.run_case("no_atr")
    assert missing["decision_card"]["new_money"]["action"] == "buy"
    assert missing["decision_card"]["exit"]["stop_or_invalidation"] is None
    assert missing["decision_card"]["sizing"] is None
    assert missing["decision_card"]["sizing_status"]["reason"] == "stop_price_missing"
    assert "atr_unavailable_using_default" in missing["action_zones"]["zone_warnings"]
    assert missing["action_zones"]["basis"]["strong_buy_below"] == "52w_low_plus_2pct_fallback"
    negative = await sm.run_case("negative_stop")
    assert negative["decision_card"]["new_money"]["action"] == "starter"
    assert negative["decision_card"]["exit"]["stop_or_invalidation"] == -0.5
    assert negative["decision_card"]["sizing"] is None
    assert negative["decision_card"]["sizing_status"]["reason"] == "stop_price_invalid"
    inverted = await sm.run_case("inverted_band")
    entry = inverted["decision_card"]["entry"]
    assert (entry["buy_zone_low"], entry["buy_zone_high"]) == (None, None)
    assert entry["entry_band_unavailable"]["reason"] == "inverted"
    assert inverted["decision_card"]["new_money"]["action"] == "buy"


async def test_modes_obey_explicit_limits_and_account_is_not_risk():
    result = await sm.run_case("speculative_mode_override")
    assert result["decision_card"]["sizing"]["starter_pct"] == 0.5
    mode = result["decision_modes"]["speculative"]
    assert mode["starter_position"]["pct"] == 0.5
    assert mode["capital_at_risk"] is None
    assert result["decision_card"]["sizing"]["assumptions"]["account_size"] == 10_000
    ordinary = await sm.run_case("both_overrides")
    assert ordinary["decision_modes"]["balanced"]["max_position"]["pct"] == 0.5
    assert ordinary["decision_card"]["sizing"]["full_pct_cap"] == 0.5


async def test_missing_account_starter_and_blocked_visibility_controls():
    no_account = (await sm.run_case("no_account"))["decision_card"]["sizing"]
    assert no_account["starter_pct"] == 2.5
    assert all(no_account[k] is None for k in ("whole_shares", "starter_dollars", "full_dollars", "fractional_shares"))
    starter = (await sm.run_case("starter"))["decision_card"]
    assert starter["new_money"]["action"] == "starter"
    assert starter["sizing"]["starter_dollars"] == 200
    assert starter["sizing"]["full_dollars"] is None
    blocked = (await sm.run_case("earnings_blocked"))["decision_card"]
    assert blocked["new_money"]["action"] == "wait"
    assert blocked["sizing"] is None


def test_swing_amounts_follow_one_whole_quantity():
    rows = sm.swing_cases()
    size = rows["fractional_amounts"]["plan"]
    assert (size["shares"], size["fractional_shares"], size["position_dollars"], size["max_loss_dollars"]) == (
        8, None, 800.0, 24.0)
    zero = rows["zero_whole_shares"]["plan"]
    assert (zero["shares"], zero["position_dollars"], zero["max_loss_dollars"]) == (0, 0.0, 0.0)
    assert zero["entry"]["condition_text"] == "The setup qualifies, but no whole-share position fits this budget."


def test_frozen_recording_can_be_copied_but_not_replaced(tmp_path):
    recording = copy.deepcopy(BEFORE)
    sm.write_recording(recording, tmp_path)
    for filename in ("before.json", "before.md"):
        assert (tmp_path / filename).read_bytes() == (sm.FIXTURE_DIR / filename).read_bytes()
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    sm.write_recording(recording, tmp_path)  # idempotent
    changed = copy.deepcopy(recording)
    changed["analyze"]["balanced_wide"]["card"]["sizing"]["full_pct_cap"] = 5.0
    with pytest.raises(ValueError, match="refusing to overwrite frozen baseline"):
        sm.write_recording(changed, tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original
    other = tmp_path / "partial"
    other.mkdir()
    (other / "before.md").write_text("conflict")
    with pytest.raises(ValueError, match="refusing to overwrite frozen baseline"):
        sm.write_recording(recording, other)
    assert not (other / "before.json").exists()  # preflight all files before writing


async def test_generator_detects_swallowed_network_attempt_outside_pytest_guard(monkeypatch):
    real_run = sm.run_case

    async def accidental_read(name):
        with suppress(RuntimeError):
            requests.get("https://example.invalid/forbidden")
        return await real_run(name)

    monkeypatch.setattr(sm, "run_case", accidental_read)
    with pytest.raises(AssertionError, match="example.invalid/forbidden"):
        await sm.build()


@pytest.mark.parametrize(("name", "regime", "starter", "full", "stop"), [
    ("balanced_low_wide", "low", 2.5, 5.0, 80.0),
    ("balanced_wide", "medium", 2.5, 5.0, 80.0),
    ("balanced_high_wide", "high", 2.0, 5.0, 80.0),
])
async def test_default_budget_is_enforced_in_all_full_size_regimes(name, regime, starter, full, stop):
    row = await sm.run_case(name)
    size = row["decision_card"]["sizing"]
    zones = row["action_zones"]
    assert zones["position_sizing_range"]["rationale"] == f"{regime}_risk_regime"
    assert (size["starter_pct"], size["full_pct_cap"]) == (starter, full)
    assert row["decision_card"]["exit"]["stop_or_invalidation"] == stop
    assert size["full_dollars"] / 100 * (100 - stop) == 100


async def test_rounded_distance_and_missing_atr_percentage_warning_branches():
    rounded = (await sm.run_case("rounded_stop_warning"))["action_zones"]
    assert "stop_too_tight" not in rounded["zone_warnings"]
    assert rounded["stop_calculation"]["stop_distance_pct"] == 0.0404
    # The tool stub supplies an unrounded percentage. The stop meets the raw
    # 2x ATR boundary, but its display-rounded distance falls below that input.
    assert (99 - 95) / 99 == 2 * (2 / 99) > 0.0404
    assert "stop_too_tight" not in (await sm.run_case("balanced_high_wide"))["action_zones"]["zone_warnings"]
    missing = (await sm.run_case("missing_atr_pct"))["action_zones"]
    assert missing["stop_calculation"]["atr_pct"] is None
    assert missing["stop_calculation"]["stop_price"] == 96.0
    assert "stop_too_tight" in missing["zone_warnings"]


@pytest.mark.parametrize(("filename", "expected"), [
    ("before.json", "0e62ae4b886689e8d9cdc37b5deccc9abf90f3efefc91dad365ed1585d601763"),
    ("before.md", "4f22dce449e3bcff3c7658138922b1062a17f1d6e107d172d5b41f1f7793441a"),
])
def test_frozen_baseline_bytes_survive_future_sizing_changes(filename, expected):
    # Keep this independent pin when live-output defect tests change in 2b.1/2b.2.
    assert hashlib.sha256((sm.FIXTURE_DIR / filename).read_bytes()).hexdigest() == expected


async def test_extreme_regime_preserves_starter_only_visibility():
    row = await sm.run_case("balanced_extreme_wide")
    card = row["decision_card"]
    assert card["new_money"]["action"] == "starter"
    assert card["sizing"]["starter_pct"] == 0.5
    assert card["sizing"]["starter_dollars"] == 0.0
    assert card["sizing"]["full_pct_cap"] is None
    assert card["sizing"]["full_dollars"] is None
    assert card["exit"]["stop_or_invalidation"] == 62.5
    assert row["action_zones"]["position_sizing_range"] is None
    assert "extreme_risk_regime" in row["action_zones"]["zone_warnings"]
