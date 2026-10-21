"""Major-holder fields preserve the units of each supported provider layout."""

from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stock_analysis.data.provider import SourceRead, SourceStatus
from stock_analysis.tools.ownership import _apply_major_holders


def modern_frame(insider: Any = 0.0172, institutional: Any = 1.05) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict({
        "insidersPercentHeld": insider,
        "institutionsPercentHeld": institutional,
        "institutionsFloatPercentHeld": 1.2,
        "institutionsCount": 321,
    }, orient="index")
    frame.columns.name = "Breakdown"
    return frame.rename(columns={frame.columns[0]: "Value"})


def apply_major(payload: Any, *, status: SourceStatus = "ok") -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    insider = {"sentiment": "buying", "net_shares_3m": 123.0}
    institutional = {"top_holders": [{"name": "ACME"}], "total_institutional_pct": None}
    warnings: list[str] = []
    read = SourceRead("major_holders", status, payload)
    _apply_major_holders(read, insider, institutional, warnings)
    return insider, institutional, warnings


def test_modern_provider_layout_maps_only_the_two_fraction_fields() -> None:
    insider, institutional, warnings = apply_major(modern_frame())

    assert insider == {"sentiment": "buying", "net_shares_3m": 123.0, "insider_pct": 0.0172}
    assert institutional == {"top_holders": [{"name": "ACME"}], "total_institutional_pct": 1.05}
    assert warnings == []


@pytest.mark.parametrize("field", ["insidersPercentHeld", "institutionsPercentHeld"])
@pytest.mark.parametrize("value,expected", [
    (0, 0.0),
    (1, 1.0),
    (1.05, 1.05),
    (0.01723456, 0.0172),
    (np.int64(2), 2.0),
    (np.float32(0.25), 0.25),
    (np.float64(1.05), 1.05),
    (Fraction(1, 4), 0.25),
])
def test_modern_finite_nonnegative_reals_are_unscaled(field: str, value: Any, expected: float) -> None:
    frame = modern_frame().astype(object)
    frame.at[field, "Value"] = value

    insider, institutional, warnings = apply_major(frame)

    target = insider["insider_pct"] if field == "insidersPercentHeld" else institutional["total_institutional_pct"]
    assert target == expected
    assert warnings == []


@pytest.mark.parametrize("field", ["insidersPercentHeld", "institutionsPercentHeld"])
@pytest.mark.parametrize("value", [
    None,
    float("nan"),
    float("inf"),
    float("-inf"),
    -0.01,
    True,
    False,
    np.bool_(True),
    np.bool_(False),
    "1.50%",
    "0.0172",
    "not available",
    0.2 + 0j,
    [],
])
def test_modern_invalid_values_are_null_with_an_error(field: str, value: Any) -> None:
    frame = modern_frame().astype(object)
    frame.at[field, "Value"] = value

    insider, institutional, warnings = apply_major(frame)

    assert insider["insider_pct"] == (None if field == "insidersPercentHeld" else 0.0172)
    assert institutional["total_institutional_pct"] == (None if field == "institutionsPercentHeld" else 1.05)
    assert warnings == ["major_holders_error"]


@pytest.mark.parametrize("payload", [None, {}, pd.DataFrame()])
def test_missing_major_holders_are_null_with_unavailable(payload: Any) -> None:
    insider, institutional, warnings = apply_major(payload)

    assert insider["insider_pct"] is None
    assert institutional["total_institutional_pct"] is None
    assert warnings == ["major_holders_unavailable"]


@pytest.mark.parametrize("missing", [
    ["insidersPercentHeld"],
    ["institutionsPercentHeld"],
    ["insidersPercentHeld", "institutionsPercentHeld"],
])
def test_missing_modern_keys_preserve_available_values(missing: list[str]) -> None:
    frame = modern_frame().drop(index=missing)

    insider, institutional, warnings = apply_major(frame)

    assert insider["insider_pct"] == (None if "insidersPercentHeld" in missing else 0.0172)
    assert institutional["total_institutional_pct"] == (None if "institutionsPercentHeld" in missing else 1.05)
    assert warnings == ["major_holders_unavailable"]


def test_invalid_modern_fields_warn_once() -> None:
    insider, institutional, warnings = apply_major(modern_frame("invalid", "invalid"))

    assert insider["insider_pct"] is None
    assert institutional["total_institutional_pct"] is None
    assert warnings == ["major_holders_error"]


def test_missing_and_invalid_modern_fields_disclose_both_conditions() -> None:
    frame = modern_frame("invalid").drop(index="institutionsPercentHeld")

    insider, institutional, warnings = apply_major(frame)

    assert insider["insider_pct"] is None
    assert institutional["total_institutional_pct"] is None
    assert sorted(warnings) == ["major_holders_error", "major_holders_unavailable"]


def test_failed_source_does_not_consume_payload() -> None:
    insider, institutional, warnings = apply_major(modern_frame(), status="failed")

    assert insider["insider_pct"] is None
    assert institutional["total_institutional_pct"] is None
    assert warnings == ["major_holders_error"]


@pytest.mark.parametrize("value,expected", [("1.50%", 0.015), (1.5, 0.015), (0.015, 0.015)])
@pytest.mark.parametrize("columns", [[0, 1], ["Value", "Description"]])
def test_legacy_two_column_frames_keep_percent_conversion(value: Any, expected: float, columns: list[Any]) -> None:
    frame = pd.DataFrame([
        [value, "% of Shares Held by All Insider"],
        ["75.00%", "% of Shares Held by Institutions"],
    ], columns=columns)

    insider, institutional, warnings = apply_major(frame)

    assert insider["insider_pct"] == expected
    assert institutional["total_institutional_pct"] == 0.75
    assert warnings == []
