"""Step 3a.3: the long-term holder action joins versioned stock snapshots."""

import copy
import hashlib
import importlib
import json
from typing import Any

import pytest

from stock_analysis.utils.normalize import (
    COMPATIBLE_SNAPSHOT_VERSIONS,
    SNAPSHOT_FIELDS,
    SNAPSHOT_VERSION,
    build_watchlist_snapshot,
)
from tests.test_tools.test_snapshot_consistency import raw_result

DIFF = importlib.import_module("stock_analysis.tools.diff_analysis")
HOLDER = "long_term_if_held_action"
ACTIONS = ("hold", "review_exit", "insufficient_data", "not_assessed", None)
# Frozen released field names, independent of the new producer.
RELEASED_FIELD_HASHES = {
    "1.0.0": "2238c9eca081abfffc7f52892474db2ed4f51ec3bbae783b431d068430773877",
    "1.1.0": "e68b892deafd4f8f65573c9cedfb3070c8ae6af1e11b1c50ce12955cc11c5bb6",
    "2.0.0": "951c13f629b3fd7d1850cb307d53145f6586e970ad12ad0d78d585d98c550f5f",
    "3.0.0": "ac0a5d910444bbfb95f46f5857211d73ba54ac46c3cd98cfa8dafc871cf15155",
}


def current(action: str | None = "hold", **changes: Any) -> dict[str, Any]:
    result = raw_result(**changes)
    result["decision_card"]["long_term"]["if_held"] = {"action": action}
    result["watchlist_snapshot"] = build_watchlist_snapshot(result)
    return result


def released(result: dict[str, Any], version: str) -> dict[str, Any]:
    snapshot = result["watchlist_snapshot"]
    return {**{key: snapshot.get(key) for key in SNAPSHOT_FIELDS[version]},
            "snapshot_version": version, "snapshot_hash": "released-version-hash"}


@pytest.fixture
def detect(monkeypatch: pytest.MonkeyPatch) -> Any:
    async def run(previous: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        calls = []

        async def analyze(*args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(args)
            return result

        monkeypatch.setattr(DIFF, "analyze_stock", analyze)
        response = await DIFF.what_changed("ACME", previous)
        response["_analysis_calls"] = len(calls)
        return response

    return run


def test_new_field_and_version_ship_together_with_released_sets_frozen() -> None:
    assert SNAPSHOT_VERSION == "4.0.0"
    assert {"1.0.0", "1.1.0", "2.0.0", "3.0.0", "4.0.0"} == COMPATIBLE_SNAPSHOT_VERSIONS
    for version, expected in RELEASED_FIELD_HASHES.items():
        actual = hashlib.sha256(json.dumps(sorted(SNAPSHOT_FIELDS[version])).encode()).hexdigest()
        assert actual == expected
    snapshot = current()["watchlist_snapshot"]
    assert set(snapshot) == SNAPSHOT_FIELDS["4.0.0"] == SNAPSHOT_FIELDS["3.0.0"] | {HOLDER}
    assert snapshot[HOLDER] == "hold"
    assert DIFF.validate_previous_snapshot(snapshot, "ACME") == []


@pytest.mark.parametrize(("before", "after"), [(a, b) for a in ACTIONS for b in ACTIONS if a != b])
async def test_every_holder_transition_is_material_and_hashed(detect: Any, before: Any, after: Any) -> None:
    response = await detect(current(before)["watchlist_snapshot"], current(after))
    assert response["material_changes"] == [f"{HOLDER}: {before} -> {after}"]
    assert response["hash_changed"] is True
    assert f"Long-term holder: {before} -> {after}." in response["summary"]
    if "not_assessed" in (before, after):
        assert "assessment coverage changed" in response["summary"]
        assert "recovered" not in response["summary"] and "deteriorated" not in response["summary"]


async def test_3_to_4_preserves_common_data_without_inventing_a_holder_change(detect: Any) -> None:
    result = current()
    previous = released(result, "3.0.0")
    response = await detect(previous, result)
    assert response["material_changes"] == []
    assert response["hash_changed"] is None
    assert response["not_comparable"] == [HOLDER]
    assert HOLDER not in response["changes"]
    assert len(response["warnings"]) == 1
    assert "4.0.0 adds long_term_if_held_action" in response["warnings"][0]
    assert "3.0.0 rule changes" not in response["warnings"][0]


@pytest.mark.parametrize(("field", "replacement", "expected"), [
    ("new_money_action", "wait", "new_money_action: buy -> wait"),
    ("if_held_action", "reduce", "if_held_action: hold -> reduce"),
    ("long_term_action", "avoid", "long_term_action: buy -> avoid"),
    ("price_zone", "near_52w_high", "price_zone: below_sma200 -> near_52w_high"),
    ("unevaluated_gate_ids", ["thesis_broken"], "+unevaluated_gate: thesis_broken"),
    ("bullish_signals", [], "-bullish: above_sma200"),
])
async def test_3_to_4_compares_retained_fields(detect: Any, field: str, replacement: Any, expected: str) -> None:
    result = current()
    previous = released(result, "3.0.0")
    result["watchlist_snapshot"][field] = replacement
    response = await detect(previous, result)
    assert expected in response["material_changes"]
    assert field not in response["not_comparable"]
    assert response["not_comparable"] == [HOLDER]


@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "2.0.0"])
async def test_older_versions_keep_action_signal_and_unevaluated_restrictions(detect: Any, version: str) -> None:
    result = current()
    previous = released(result, version)
    previous.update({"new_money_action": "avoid", "if_held_action": "reduce", "long_term_action": "avoid",
                     "price_zone": "reduce", "unevaluated_gate_ids": ["thesis_broken"], HOLDER: "review_exit"})
    response = await detect(previous, result)
    assert response["material_changes"] == []
    assert {HOLDER, "new_money_action", "if_held_action", "long_term_action", "price_zone",
            "unevaluated_gate_ids"} <= set(response["not_comparable"])
    assert ("bullish_signals" in response["not_comparable"]) is (version != "2.0.0")
    assert len(response["warnings"]) == 1
    assert "3.0.0 rule changes" in response["warnings"][0]
    assert "4.0.0 adds long_term_if_held_action" in response["warnings"][0]


@pytest.mark.parametrize("action", ACTIONS)
def test_new_holder_enum_and_null_validate(action: Any) -> None:
    assert DIFF.validate_previous_snapshot(current(action)["watchlist_snapshot"], "ACME") == []


@pytest.mark.parametrize("bad", ["reduce", "buy", "", "unknown", True, 1, [], {}])
async def test_malformed_new_holder_action_is_rejected_before_analysis(detect: Any, bad: Any) -> None:
    result = current()
    previous = {**result["watchlist_snapshot"], HOLDER: bad}
    response = await detect(previous, result)
    assert response["error_type"] == "invalid_snapshot"
    assert any(HOLDER in problem for problem in response["problems"])
    assert response["_analysis_calls"] == 0


async def test_missing_new_holder_key_is_truncation_before_analysis(detect: Any) -> None:
    result = current()
    previous = {key: value for key, value in result["watchlist_snapshot"].items() if key != HOLDER}
    response = await detect(previous, result)
    assert response["problems"] == [f"missing required key: {HOLDER}"]
    assert response["_analysis_calls"] == 0


def test_3_0_0_ignores_stray_holder_values_and_keeps_old_action_validation() -> None:
    previous = released(current(), "3.0.0")
    previous.update({HOLDER: ["wrong-shape"], "new_money_action": "future_unknown_code"})
    assert DIFF.validate_previous_snapshot(previous, "ACME") == []
    del previous["if_held_action"]
    assert DIFF.validate_previous_snapshot(previous, "ACME") == ["missing required key: if_held_action"]


def test_personal_context_and_holder_reasons_never_enter_the_hash() -> None:
    result = current()
    before = copy.deepcopy(result["watchlist_snapshot"])
    result["position"] = {"cost_basis": 10, "shares": 500, "portfolio_value": 100000}
    result["review_status"] = "review_allocation"
    result["decision_card"]["long_term"]["if_held"].update({"basis": "changed", "watch_factors": ["concern"]})
    assert build_watchlist_snapshot(result) == before
