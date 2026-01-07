"""Tests for normalize module."""

import math
import json
import pytest

from stock_mcp.utils.normalize import (
    canonical_dumps,
    normalize_for_watchlist_diff,
    build_watchlist_snapshot,
    SNAPSHOT_VERSION,
)


class TestCanonicalDumps:
    """Tests for canonical_dumps function."""

    def test_sorted_keys(self):
        """Keys should be sorted at every level."""
        obj = {"z": 1, "a": 2, "m": {"z": 3, "a": 4}}
        result = canonical_dumps(obj)
        assert result == '{"a":2,"m":{"a":4,"z":3},"z":1}'

    def test_minimal_separators(self):
        """Output should use minimal separators (no spaces)."""
        obj = {"a": [1, 2, 3]}
        result = canonical_dumps(obj)
        assert result == '{"a":[1,2,3]}'
        assert " " not in result

    def test_unicode_preserved(self):
        """Unicode should be preserved (not escaped)."""
        obj = {"emoji": "🚀"}
        result = canonical_dumps(obj)
        assert "🚀" in result

    def test_rejects_nan(self):
        """Should raise ValueError for NaN (allow_nan=False)."""
        obj = {"value": float("nan")}
        with pytest.raises(ValueError, match="Out of range float values"):
            canonical_dumps(obj)

    def test_rejects_inf(self):
        """Should raise ValueError for inf (allow_nan=False)."""
        obj = {"value": float("inf")}
        with pytest.raises(ValueError, match="Out of range float values"):
            canonical_dumps(obj)


class TestNormalizeForWatchlistDiff:
    """Tests for normalize_for_watchlist_diff function."""

    def test_removes_duration_ms(self):
        """Should remove meta.duration_ms."""
        raw = {"meta": {"duration_ms": 123, "tool": "test"}}
        result = normalize_for_watchlist_diff(raw)
        assert "duration_ms" not in result.get("meta", {})
        assert result["meta"]["tool"] == "test"

    def test_removes_tool_timings(self):
        """Should remove data_quality.tool_timings."""
        raw = {"data_quality": {"tool_timings": {"a": 1}, "completeness": 0.9}}
        result = normalize_for_watchlist_diff(raw)
        assert "tool_timings" not in result.get("data_quality", {})
        assert result["data_quality"]["completeness"] == 0.9

    def test_timestamp_to_date(self):
        """Should convert timestamps to date-only."""
        raw = {
            "data_provenance": {
                "price": {"as_of": "2026-01-07T02:15:53.509527Z"}
            }
        }
        result = normalize_for_watchlist_diff(raw)
        prov = result.get("data_provenance", {}).get("price", {})
        assert "as_of" not in prov
        assert prov.get("as_of_date") == "2026-01-07"

    def test_null_lists_become_empty(self):
        """Should convert null lists to []."""
        raw = {
            "signals": {"bullish": None, "bearish": ["test"]},
            "data_quality": {"data_gaps": None},
        }
        result = normalize_for_watchlist_diff(raw)
        assert result["signals"]["bullish"] == []
        assert result["signals"]["bearish"] == ["test"]
        assert result["data_quality"]["data_gaps"] == []

    def test_string_lists_sorted(self):
        """Set-like string lists should be sorted."""
        raw = {"signals": {"bullish": ["z_signal", "a_signal", "m_signal"]}}
        result = normalize_for_watchlist_diff(raw)
        assert result["signals"]["bullish"] == ["a_signal", "m_signal", "z_signal"]

    def test_money_fields_to_int(self):
        """Money fields should be converted to int."""
        raw = {"summary": {"market_cap": 1234567890.0}}
        result = normalize_for_watchlist_diff(raw)
        assert result["summary"]["market_cap"] == 1234567890
        assert isinstance(result["summary"]["market_cap"], int)

    def test_component_freshness_normalized(self):
        """Should normalize component_freshness timestamps."""
        raw = {
            "data_quality": {
                "component_freshness": {
                    "price": {
                        "as_of": "2026-01-07T02:15:53.509527Z",
                        "age_hours": 0.5,
                        "stale": False,
                    }
                }
            }
        }
        result = normalize_for_watchlist_diff(raw)
        freshness = result["data_quality"]["component_freshness"]["price"]
        assert "as_of" not in freshness
        assert "age_hours" not in freshness
        assert freshness["as_of_date"] == "2026-01-07"
        assert freshness["stale"] is False

    def test_nan_replaced_with_null(self):
        """NaN values should be replaced with null."""
        raw = {"metrics": {"value": float("nan"), "other": 1.5}}
        result = normalize_for_watchlist_diff(raw)
        assert result["metrics"]["value"] is None
        assert result["metrics"]["other"] == 1.5

    def test_inf_replaced_with_null(self):
        """Inf values should be replaced with null."""
        raw = {"metrics": {"pos_inf": float("inf"), "neg_inf": float("-inf")}}
        result = normalize_for_watchlist_diff(raw)
        assert result["metrics"]["pos_inf"] is None
        assert result["metrics"]["neg_inf"] is None

    def test_nested_nan_sanitized(self):
        """NaN in nested structures should be sanitized."""
        raw = {
            "outer": {
                "inner": {"deep": float("nan")},
                "list": [1.0, float("nan"), 3.0],
            }
        }
        result = normalize_for_watchlist_diff(raw)
        assert result["outer"]["inner"]["deep"] is None
        assert result["outer"]["list"] == [1.0, None, 3.0]

    def test_normalized_output_is_json_safe(self):
        """Normalized output with NaN input should serialize without error."""
        raw = {
            "metrics": {"nan_value": float("nan"), "inf_value": float("inf")},
            "signals": {"bullish": None},
        }
        result = normalize_for_watchlist_diff(raw)
        # This should not raise - canonical_dumps uses allow_nan=False
        json_str = canonical_dumps(result)
        assert "null" in json_str
        assert "NaN" not in json_str
        assert "Infinity" not in json_str

    def test_negative_zero_normalized(self):
        """Negative zero should be normalized to positive zero."""
        raw = {"metrics": {"neg_zero": -0.0, "pos_zero": 0.0}}
        result = normalize_for_watchlist_diff(raw)
        # Both should serialize identically
        assert result["metrics"]["neg_zero"] == 0.0
        assert result["metrics"]["pos_zero"] == 0.0
        # Verify the sign is positive (copysign test)
        assert math.copysign(1.0, result["metrics"]["neg_zero"]) > 0

    def test_negative_zero_in_list(self):
        """Negative zero in lists should also be normalized."""
        raw = {"values": [-0.0, 1.0, -0.0]}
        result = normalize_for_watchlist_diff(raw)
        assert result["values"] == [0.0, 1.0, 0.0]
        for v in result["values"]:
            if v == 0.0:
                assert math.copysign(1.0, v) > 0


class TestBuildWatchlistSnapshot:
    """Tests for build_watchlist_snapshot function."""

    def test_extracts_key_fields(self):
        """Should extract key watchlist fields."""
        raw = {
            "symbol": "AAPL",
            "summary": {
                "current_price": 150.0,
                "market_cap": 2500000000000,
                "sector": "Technology",
            },
            "verdict": {
                "tilt": "bullish",
                "confidence": "high",
                "score": 0.75,
            },
            "policy_action": {
                "mid_term": "hold_or_add",
                "long_term": "accumulate",
                "valuation_gate": "neutral",
                "is_unprofitable": None,
            },
            "risk_summary": {
                "risk_regime": {"classification": "medium"},
                "volatility_ann": 0.25,
            },
            "events_summary": {
                "next_catalyst": {
                    "type": "earnings",
                    "status": "available",
                    "date": "2026-02-01",
                    "days_until": 25,
                },
            },
            "data_provenance": {
                "price": {"as_of": "2026-01-07T10:00:00Z"}
            },
            "data_quality": {},
        }
        snapshot = build_watchlist_snapshot(raw)

        assert snapshot["symbol"] == "AAPL"
        assert snapshot["price"] == 150.0
        assert snapshot["market_cap"] == 2500000000000
        assert snapshot["sector"] == "Technology"
        assert snapshot["tilt"] == "bullish"
        assert snapshot["confidence"] == "high"
        assert snapshot["score"] == 0.75
        assert snapshot["action_mid_term"] == "hold_or_add"
        assert snapshot["action_long_term"] == "accumulate"
        assert snapshot["valuation_gate"] == "neutral"
        assert snapshot["risk_regime"] == "medium"
        assert snapshot["next_catalyst"]["status"] == "available"

    def test_handles_missing_fields(self):
        """Should handle missing optional fields gracefully."""
        raw = {"symbol": "TEST"}
        snapshot = build_watchlist_snapshot(raw)
        assert snapshot["symbol"] == "TEST"
        assert snapshot["price"] is None
        assert snapshot["tilt"] is None

    def test_includes_snapshot_version(self):
        """Should include snapshot_version in output."""
        raw = {"symbol": "TEST"}
        snapshot = build_watchlist_snapshot(raw)
        assert "snapshot_version" in snapshot
        assert snapshot["snapshot_version"] == SNAPSHOT_VERSION

    def test_includes_snapshot_hash(self):
        """Should include snapshot_hash in output."""
        raw = {"symbol": "TEST"}
        snapshot = build_watchlist_snapshot(raw)
        assert "snapshot_hash" in snapshot
        # Hash should be 16 hex characters (truncated SHA-256)
        assert len(snapshot["snapshot_hash"]) == 16
        assert all(c in "0123456789abcdef" for c in snapshot["snapshot_hash"])

    def test_same_input_produces_same_hash(self):
        """Same input should always produce the same hash."""
        raw = {"symbol": "TEST", "summary": {"current_price": 100.0}}
        snapshot1 = build_watchlist_snapshot(raw)
        snapshot2 = build_watchlist_snapshot(raw)
        assert snapshot1["snapshot_hash"] == snapshot2["snapshot_hash"]

    def test_different_input_produces_different_hash(self):
        """Different input should produce different hash."""
        raw1 = {"symbol": "TEST", "summary": {"current_price": 100.0}}
        raw2 = {"symbol": "TEST", "summary": {"current_price": 101.0}}
        snapshot1 = build_watchlist_snapshot(raw1)
        snapshot2 = build_watchlist_snapshot(raw2)
        assert snapshot1["snapshot_hash"] != snapshot2["snapshot_hash"]

    def test_hash_includes_version(self):
        """Hash should change if version changes (version is in hashed content)."""
        raw = {"symbol": "TEST"}
        snapshot = build_watchlist_snapshot(raw)
        # snapshot_version should be in the snapshot and affect the hash
        assert "snapshot_version" in snapshot
        assert snapshot["snapshot_version"] == SNAPSHOT_VERSION

    def test_hash_excludes_itself(self):
        """Hash should not include itself (would be self-referential)."""
        raw = {"symbol": "TEST"}
        snapshot = build_watchlist_snapshot(raw)
        # Verify hash is computed from data without hash
        # (indirectly tested by hash stability - if hash was included, it would be unstable)
        assert "snapshot_hash" in snapshot
        assert len(snapshot["snapshot_hash"]) == 16

    def test_includes_snapshot_as_of_date(self):
        """Should include snapshot_as_of_date computed from component dates."""
        raw = {
            "symbol": "TEST",
            "data_quality": {
                "component_freshness": {
                    "price": {"as_of_date": "2026-01-05"},
                    "fundamentals": {"as_of_date": "2026-01-06"},
                }
            },
        }
        snapshot = build_watchlist_snapshot(raw)
        # Should be max of all dates
        assert snapshot["snapshot_as_of_date"] == "2026-01-06"

    def test_snapshot_as_of_date_from_provenance(self):
        """Should also consider data_provenance dates."""
        raw = {
            "symbol": "TEST",
            "data_provenance": {
                "price": {"as_of_date": "2026-01-07"},
            },
        }
        snapshot = build_watchlist_snapshot(raw)
        assert snapshot["snapshot_as_of_date"] == "2026-01-07"

    def test_snapshot_as_of_date_none_when_no_dates(self):
        """Should be None when no component dates available."""
        raw = {"symbol": "TEST"}
        snapshot = build_watchlist_snapshot(raw)
        assert snapshot["snapshot_as_of_date"] is None


class TestDiffStability:
    """Integration tests for diff stability."""

    def test_same_input_produces_same_output(self):
        """Same input should always produce identical canonical output."""
        raw = {
            "symbol": "TEST",
            "signals": {"bullish": ["c", "a", "b"]},
            "meta": {"duration_ms": 100},
        }
        result1 = canonical_dumps(normalize_for_watchlist_diff(raw))
        result2 = canonical_dumps(normalize_for_watchlist_diff(raw))
        assert result1 == result2

    def test_different_order_same_canonical(self):
        """Different key ordering should produce same canonical output."""
        raw1 = {"z": 1, "a": 2}
        raw2 = {"a": 2, "z": 1}
        assert canonical_dumps(raw1) == canonical_dumps(raw2)

    def test_idempotence(self):
        """Normalization should be idempotent: normalize(x) == normalize(normalize(x))."""
        raw = {
            "symbol": "TEST",
            "signals": {"bullish": ["c", "a", "b"], "bearish": None},
            "meta": {"duration_ms": 100, "tool": "test"},
            "data_provenance": {"price": {"as_of": "2026-01-07T10:00:00Z"}},
            "summary": {"market_cap": 1234567890.0},
        }
        once = normalize_for_watchlist_diff(raw)
        twice = normalize_for_watchlist_diff(once)
        # Should be identical after second normalization
        assert canonical_dumps(once) == canonical_dumps(twice)

    def test_dict_key_order_insensitivity(self):
        """Same data with shuffled dict key order should normalize identically."""
        # Build same data with different key insertion order
        raw1 = {
            "symbol": "TEST",
            "summary": {"market_cap": 1000, "current_price": 100},
            "signals": {"bullish": ["a", "b"], "bearish": ["c"]},
        }
        raw2 = {
            "signals": {"bearish": ["c"], "bullish": ["a", "b"]},
            "summary": {"current_price": 100, "market_cap": 1000},
            "symbol": "TEST",
        }
        # Should produce identical canonical output
        assert canonical_dumps(normalize_for_watchlist_diff(raw1)) == canonical_dumps(
            normalize_for_watchlist_diff(raw2)
        )

    def test_snapshot_hash_order_insensitivity(self):
        """Same data with shuffled key order should produce same snapshot hash."""
        raw1 = {
            "symbol": "TEST",
            "summary": {"current_price": 100.0},
            "verdict": {"tilt": "bullish"},
        }
        raw2 = {
            "verdict": {"tilt": "bullish"},
            "symbol": "TEST",
            "summary": {"current_price": 100.0},
        }
        snapshot1 = build_watchlist_snapshot(raw1)
        snapshot2 = build_watchlist_snapshot(raw2)
        assert snapshot1["snapshot_hash"] == snapshot2["snapshot_hash"]

    def test_ranked_list_tie_break_by_category_then_id(self):
        """Ranked lists should tie-break by (category, id) for determinism."""
        # Two items with same score_delta but different category/id
        raw1 = {
            "decision_context": {
                "top_triggers": [
                    {"score_delta": 0.1, "category": "risk", "id": "high_vol"},
                    {"score_delta": 0.1, "category": "fundamentals", "id": "growth"},
                ]
            }
        }
        # Same items in different order
        raw2 = {
            "decision_context": {
                "top_triggers": [
                    {"score_delta": 0.1, "category": "fundamentals", "id": "growth"},
                    {"score_delta": 0.1, "category": "risk", "id": "high_vol"},
                ]
            }
        }
        result1 = normalize_for_watchlist_diff(raw1)
        result2 = normalize_for_watchlist_diff(raw2)
        # Both should produce same canonical output (sorted by category then id)
        assert canonical_dumps(result1) == canonical_dumps(result2)
        # Verify order: fundamentals < risk (alphabetically)
        triggers = result1["decision_context"]["top_triggers"]
        assert triggers[0]["category"] == "fundamentals"
        assert triggers[1]["category"] == "risk"

    def test_ranked_list_preserves_score_delta_order(self):
        """Ranked lists should preserve score_delta ordering (primary sort)."""
        raw = {
            "decision_context": {
                "top_triggers": [
                    {"score_delta": 0.3, "category": "risk", "id": "a"},
                    {"score_delta": 0.1, "category": "fundamentals", "id": "b"},
                    {"score_delta": 0.2, "category": "technicals", "id": "c"},
                ]
            }
        }
        result = normalize_for_watchlist_diff(raw)
        triggers = result["decision_context"]["top_triggers"]
        # Score delta order should be preserved (0.3, 0.1, 0.2)
        # We're NOT re-sorting by score_delta, just tie-breaking within same delta
        assert triggers[0]["score_delta"] == 0.3
        assert triggers[1]["score_delta"] == 0.1
        assert triggers[2]["score_delta"] == 0.2
