"""Regression tests for server JSON response sanitization.

Strict JSON parsers (most non-Python clients) reject `NaN` and `Infinity`
literals. The MCP response helper must coerce those to `null` rather than
emitting non-spec JSON.
"""

import json

from stock_analysis.server import _json_response, _sanitize_for_json


class TestSanitizeForJson:
    """_sanitize_for_json replaces NaN/Infinity with None recursively."""

    def test_top_level_nan(self) -> None:
        assert _sanitize_for_json(float("nan")) is None

    def test_top_level_positive_infinity(self) -> None:
        assert _sanitize_for_json(float("inf")) is None

    def test_top_level_negative_infinity(self) -> None:
        assert _sanitize_for_json(float("-inf")) is None

    def test_normal_float_preserved(self) -> None:
        assert _sanitize_for_json(3.14) == 3.14

    def test_nested_dict_nan(self) -> None:
        result = _sanitize_for_json({"a": 1.0, "b": float("nan"), "c": {"d": float("inf")}})
        assert result == {"a": 1.0, "b": None, "c": {"d": None}}

    def test_nested_list_nan(self) -> None:
        result = _sanitize_for_json([1.0, float("nan"), {"x": float("-inf")}])
        assert result == [1.0, None, {"x": None}]


class TestJsonResponseStrictness:
    """_json_response must produce strict-JSON parseable output even if NaN slips in."""

    def test_nan_value_becomes_null_in_output(self) -> None:
        output = _json_response({"metric": float("nan"), "ok": 1.0})
        # Strict JSON: must NOT contain NaN/Infinity literals
        assert "NaN" not in output
        assert "Infinity" not in output
        # Round-trip parse with strict mode (json.loads is strict by default)
        parsed = json.loads(output)
        assert parsed == {"metric": None, "ok": 1.0}

    def test_nested_infinity_becomes_null(self) -> None:
        output = _json_response({
            "summary": {"value": float("inf"), "name": "ACME"},
            "items": [1.0, float("-inf"), 3.0],
        })
        parsed = json.loads(output)
        assert parsed == {
            "summary": {"value": None, "name": "ACME"},
            "items": [1.0, None, 3.0],
        }

    def test_clean_dict_passes_through(self) -> None:
        payload = {"a": 1, "b": "text", "c": [1.0, 2.0], "d": {"e": True}}
        parsed = json.loads(_json_response(payload))
        assert parsed == payload
