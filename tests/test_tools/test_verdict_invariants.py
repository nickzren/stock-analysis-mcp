"""Tests for verdict invariant validation."""

import logging
from typing import Any

import pytest

from stock_analysis.tools.analyze.verdict import _validate_verdict_invariants


@pytest.fixture
def base_verdict() -> dict[str, Any]:
    """Return a valid verdict invariant payload."""
    return {
        "coverage": {
            "price": {"fetched": True, "used_in_score": True, "reason_excluded": None},
            "fundamentals": {"fetched": True, "used_in_score": True, "reason_excluded": None},
            "risk": {"fetched": True, "used_in_score": True, "reason_excluded": None},
        },
        "components": {
            "technicals": 0.5,
            "fundamentals": 0.3,
            "risk": -0.2,
        },
        "weights_full": {"technicals": 0.3, "fundamentals": 0.45, "risk": 0.25},
        "weights_used": {
            "technicals": 0.3,
            "fundamentals": 0.45,
            "risk": 0.25,
        },
        "coverage_factor": 1.0,
    }


class TestVerdictInvariants:
    """Tests for _validate_verdict_invariants function."""

    def test_valid_verdict_no_warnings(self, caplog, base_verdict):
        """A valid verdict should not produce any warnings."""
        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(base_verdict)

        assert "invariant violation" not in caplog.text.lower()

    def test_used_in_score_true_but_component_none(self, caplog, base_verdict):
        """Should warn if used_in_score=True but component score is None."""
        verdict = base_verdict
        verdict["coverage"]["fundamentals"]["used_in_score"] = False
        verdict["coverage"]["fundamentals"]["reason_excluded"] = "test"
        verdict["coverage"]["risk"]["used_in_score"] = False
        verdict["coverage"]["risk"]["reason_excluded"] = "test"
        verdict["components"] = {"technicals": None, "fundamentals": None, "risk": None}
        verdict["weights_used"] = {}
        verdict["coverage_factor"] = None

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "coverage.price.used_in_score=True but components.technicals=None" in caplog.text

    def test_component_score_but_used_in_score_false(self, caplog, base_verdict):
        """Should warn if component has score but used_in_score=False."""
        verdict = base_verdict
        verdict["coverage"]["price"]["used_in_score"] = False
        verdict["coverage"]["price"]["reason_excluded"] = "test"

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "components.technicals=0.5 but coverage.price.used_in_score=False" in caplog.text

    def test_component_score_not_in_weights_used(self, caplog, base_verdict):
        """Should warn if component has score but not in weights_used."""
        verdict = base_verdict
        verdict["weights_used"] = {"technicals": 0.3, "risk": 0.25}
        verdict["coverage_factor"] = 0.55

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "components.fundamentals=0.3 but fundamentals not in weights_used" in caplog.text

    def test_null_component_in_weights_used(self, caplog, base_verdict):
        """Should warn if null component is in weights_used."""
        verdict = base_verdict
        verdict["coverage"]["fundamentals"]["used_in_score"] = False
        verdict["coverage"]["fundamentals"]["reason_excluded"] = "test"
        verdict["components"]["fundamentals"] = None

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "components.fundamentals=None but fundamentals in weights_used" in caplog.text

    def test_coverage_factor_mismatch(self, caplog, base_verdict):
        """Should warn if coverage_factor doesn't match expected coverage factor."""
        verdict = base_verdict
        verdict["coverage"]["risk"] = {
            "fetched": False,
            "used_in_score": False,
            "reason_excluded": "test",
        }
        verdict["components"]["risk"] = None
        verdict["weights_used"] = {"technicals": 0.4, "fundamentals": 0.6}

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "coverage_factor=1.0 but expected_coverage_factor=0.75" in caplog.text

    def test_empty_verdict_no_crash(self, caplog):
        """Should handle empty verdict without crashing."""
        verdict = {}

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        # Should not crash, may or may not have warnings depending on implementation

    def test_null_weights_used_no_crash(self, caplog):
        """Should handle null weights_used without crashing."""
        verdict = {
            "coverage": {
                "price": {"fetched": True, "used_in_score": False, "reason_excluded": "test"},
            },
            "components": {
                "technicals": None,
            },
            "weights_used": None,  # Null
            "coverage_factor": None,
        }

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        # Should not crash
