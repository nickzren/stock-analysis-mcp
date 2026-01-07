"""Tests for verdict invariant validation."""

import logging

import pytest

from stock_mcp.tools.analyze import _validate_verdict_invariants


class TestVerdictInvariants:
    """Tests for _validate_verdict_invariants function."""

    def test_valid_verdict_no_warnings(self, caplog):
        """A valid verdict should not produce any warnings."""
        verdict = {
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
            "weights_used": {
                "technicals": 0.3,
                "fundamentals": 0.45,
                "risk": 0.25,
            },
            "coverage_factor": 1.0,
        }

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "invariant violation" not in caplog.text.lower()

    def test_used_in_score_true_but_component_none(self, caplog):
        """Should warn if used_in_score=True but component score is None."""
        verdict = {
            "coverage": {
                "price": {"fetched": True, "used_in_score": True, "reason_excluded": None},  # Violation!
                "fundamentals": {"fetched": True, "used_in_score": False, "reason_excluded": "test"},
                "risk": {"fetched": True, "used_in_score": False, "reason_excluded": "test"},
            },
            "components": {
                "technicals": None,  # But coverage says used_in_score=True!
                "fundamentals": None,
                "risk": None,
            },
            "weights_used": {},
            "coverage_factor": None,
        }

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "coverage.price.used_in_score=True but components.technicals=None" in caplog.text

    def test_component_score_but_used_in_score_false(self, caplog):
        """Should warn if component has score but used_in_score=False."""
        verdict = {
            "coverage": {
                "price": {"fetched": True, "used_in_score": False, "reason_excluded": "test"},  # Violation!
                "fundamentals": {"fetched": True, "used_in_score": True, "reason_excluded": None},
                "risk": {"fetched": True, "used_in_score": True, "reason_excluded": None},
            },
            "components": {
                "technicals": 0.5,  # Has score but coverage says not used!
                "fundamentals": 0.3,
                "risk": -0.2,
            },
            "weights_used": {
                "technicals": 0.3,
                "fundamentals": 0.45,
                "risk": 0.25,
            },
            "coverage_factor": 1.0,
        }

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "components.technicals=0.5 but coverage.price.used_in_score=False" in caplog.text

    def test_component_score_not_in_weights_used(self, caplog):
        """Should warn if component has score but not in weights_used."""
        verdict = {
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
            "weights_used": {
                "technicals": 0.3,
                # fundamentals missing!
                "risk": 0.25,
            },
            "coverage_factor": 0.55,
        }

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "components.fundamentals=0.3 but fundamentals not in weights_used" in caplog.text

    def test_null_component_in_weights_used(self, caplog):
        """Should warn if null component is in weights_used."""
        verdict = {
            "coverage": {
                "price": {"fetched": True, "used_in_score": True, "reason_excluded": None},
                "fundamentals": {"fetched": True, "used_in_score": False, "reason_excluded": "test"},
                "risk": {"fetched": True, "used_in_score": True, "reason_excluded": None},
            },
            "components": {
                "technicals": 0.5,
                "fundamentals": None,  # None but in weights_used!
                "risk": -0.2,
            },
            "weights_used": {
                "technicals": 0.3,
                "fundamentals": 0.45,  # Shouldn't be here!
                "risk": 0.25,
            },
            "coverage_factor": 1.0,
        }

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "components.fundamentals=None but fundamentals in weights_used" in caplog.text

    def test_coverage_factor_mismatch(self, caplog):
        """Should warn if coverage_factor doesn't match sum of weights_used."""
        verdict = {
            "coverage": {
                "price": {"fetched": True, "used_in_score": True, "reason_excluded": None},
                "fundamentals": {"fetched": True, "used_in_score": True, "reason_excluded": None},
                "risk": {"fetched": False, "used_in_score": False, "reason_excluded": "test"},
            },
            "components": {
                "technicals": 0.5,
                "fundamentals": 0.3,
                "risk": None,
            },
            "weights_used": {
                "technicals": 0.3,
                "fundamentals": 0.45,
            },
            "coverage_factor": 1.0,  # Wrong! Should be 0.75
        }

        with caplog.at_level(logging.WARNING):
            _validate_verdict_invariants(verdict)

        assert "coverage_factor=1.0 but sum(weights_used)=0.75" in caplog.text

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
