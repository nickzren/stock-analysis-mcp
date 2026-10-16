"""Characterize v0.3.0; these defect pins must change deliberately in later 2b work.

The immutable recording remains available for before/after comparisons. No new
arithmetic or input validation is implemented by these tests.
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
async def test_current_sizing_surfaces_match_frozen_recording(name):
    result = await sm.run_case(name)
    assert sm.digest(name, result) == BEFORE["analyze"][name]
    for tier, payload in dm.tiers(result).items():
        assert payload["watchlist_snapshot"] == result["watchlist_snapshot"], tier


@pytest.mark.parametrize(("name", "starter", "full", "whole"), [
    ("balanced_wide", 250.0, 800.0, 2),
    ("risk_only", 250.0, 800.0, 2),  # .25% risk is shown but not enforced
    ("cap_only_tight", 200.0, 200.0, 2),
    ("cap_only_wide", 250.0, 800.0, 2),
    ("both_overrides", 50.0, 50.0, 0),
    ("tiny_cap", 10.0, 10.0, 0),  # .06% silently rises to .1%
    ("small_account", 25.0, 80.0, 0),  # dollars still imply fractional shares
])
async def test_current_amounts_independently_pin_default_and_override_defects(name, starter, full, whole):
    result = await sm.run_case(name)
    size = result["decision_card"]["sizing"]
    assert (size["starter_dollars"], size["full_dollars"], size["whole_shares"]) == (starter, full, whole)
    assert result["decision_card"]["new_money"]["action"] == "buy"


async def test_stop_budget_is_only_a_diagnostic_and_tiny_risk_falls_back():
    result = await sm.run_case("risk_only")
    size = result["action_zones"]["position_sizing_range"]
    assert size["stop_implied_max"]["pct"] == 1.2  # rounded 1.25; not applied
    assert size["max_pct"] == 8.0
    assert result["decision_card"]["exit"]["stop_or_invalidation"] == 80.0
    card = result["decision_card"]
    implied_loss = card["sizing"]["full_dollars"] / card["sizing"]["at_price"] * (
        card["sizing"]["at_price"] - card["exit"]["stop_or_invalidation"])
    assert implied_loss == 160 > 25
    tiny = await sm.run_case("tiny_risk")
    assert tiny["action_zones"]["position_sizing_range"]["risk_per_trade_pct"] == 1.0


async def test_sizing_inputs_do_not_change_decisions_or_any_snapshot_field():
    reference = await sm.run_case("balanced_wide")
    names = ["core_wide", "speculative_wide", "risk_only", "cap_only_tight", "cap_only_wide",
             "both_overrides", "tiny_risk", "tiny_cap", "no_account"]
    for name in names:
        result = await sm.run_case(name)
        assert result["watchlist_snapshot"] == reference["watchlist_snapshot"], name
        assert dm.digest(result)["decision"] == dm.digest(reference)["decision"], name


async def test_missing_stop_negative_stop_and_inverted_band_currently_survive():
    missing = await sm.run_case("no_atr")
    assert missing["decision_card"]["new_money"]["action"] == "buy"
    assert missing["decision_card"]["exit"]["stop_or_invalidation"] is None
    assert missing["decision_card"]["sizing"]["starter_dollars"] == 250
    assert "atr_unavailable_using_default" in missing["action_zones"]["zone_warnings"]
    assert missing["action_zones"]["basis"]["strong_buy_below"] == "52w_low_plus_1atr"
    negative = await sm.run_case("negative_stop")
    assert negative["decision_card"]["new_money"]["action"] == "starter"
    assert negative["decision_card"]["exit"]["stop_or_invalidation"] == -0.5
    assert negative["decision_card"]["sizing"]["starter_dollars"] == 50
    inverted = await sm.run_case("inverted_band")
    entry = inverted["decision_card"]["entry"]
    assert (entry["buy_zone_low"], entry["buy_zone_high"]) == (115.5, 101.0)
    assert inverted["decision_card"]["new_money"]["action"] == "buy"


async def test_modes_ignore_explicit_limits_and_capital_at_risk_is_the_account():
    result = await sm.run_case("speculative_mode_override")
    assert result["decision_card"]["sizing"]["starter_pct"] == 0.5
    mode = result["decision_modes"]["speculative"]
    assert mode["starter_position"]["pct"] == 1.0
    assert mode["capital_at_risk"] == "10.0K"
    ordinary = await sm.run_case("both_overrides")
    assert ordinary["decision_modes"]["balanced"]["max_position"]["pct"] == 8.0
    assert ordinary["decision_card"]["sizing"]["full_pct_cap"] == 0.5


async def test_missing_account_starter_and_blocked_visibility_controls():
    no_account = (await sm.run_case("no_account"))["decision_card"]["sizing"]
    assert no_account["starter_pct"] == 2.5
    assert all(no_account[k] is None for k in ("whole_shares", "starter_dollars", "full_dollars", "fractional_shares"))
    starter = (await sm.run_case("starter"))["decision_card"]
    assert starter["new_money"]["action"] == "starter"
    assert starter["sizing"]["starter_dollars"] == 250
    assert starter["sizing"]["full_dollars"] is None
    blocked = (await sm.run_case("earnings_blocked"))["decision_card"]
    assert blocked["new_money"]["action"] == "wait"
    assert blocked["sizing"] is None


def test_swing_amounts_follow_fractional_not_whole_quantity():
    rows = sm.swing_cases()
    assert rows == BEFORE["swing"]
    size = rows["fractional_amounts"]["plan"]
    assert (size["shares"], size["fractional_shares"], size["position_dollars"], size["max_loss_dollars"]) == (
        8, 8.25, 825.0, 24.75)
    zero = rows["zero_whole_shares"]["plan"]
    assert (zero["shares"], zero["position_dollars"], zero["max_loss_dollars"]) == (0, 80.0, 2.4)
    assert zero["entry"]["condition_text"] == "Trigger already satisfied — enter at market"


async def test_generator_is_reproducible_and_cannot_replace_a_frozen_file(tmp_path):
    recording = await sm.build()
    assert recording == BEFORE
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
    ("balanced_low_wide", "low", 2.5, 10.0, 80.0),
    ("balanced_wide", "medium", 2.5, 8.0, 80.0),
    ("balanced_high_wide", "high", 2.0, 6.0, 80.0),
])
async def test_default_budget_can_be_exceeded_in_full_size_regimes(name, regime, starter, full, stop):
    row = await sm.run_case(name)
    size = row["decision_card"]["sizing"]
    zones = row["action_zones"]
    assert zones["position_sizing_range"]["rationale"] == f"{regime}_risk_regime"
    assert (size["starter_pct"], size["full_pct_cap"]) == (starter, full)
    assert row["decision_card"]["exit"]["stop_or_invalidation"] == stop
    assert size["full_dollars"] / 100 * (100 - stop) > 100


async def test_rounded_distance_and_missing_atr_percentage_warning_branches():
    rounded = (await sm.run_case("rounded_stop_warning"))["action_zones"]
    assert "stop_too_tight" in rounded["zone_warnings"]
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
    assert card["sizing"]["starter_dollars"] == 50.0
    assert card["sizing"]["full_pct_cap"] is None
    assert card["sizing"]["full_dollars"] is None
    assert card["exit"]["stop_or_invalidation"] == 62.5
    assert row["action_zones"]["position_sizing_range"] is None
    assert "extreme_risk_regime" in row["action_zones"]["zone_warnings"]
