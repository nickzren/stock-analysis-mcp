"""Pure first-match holder policy; component combinations unavailable in live classifiers."""

from copy import deepcopy
from itertools import product

import pytest

from stock_analysis.tools.analyze.holder import (
    ComponentEvidence,
    HolderEvidence,
    long_term_if_held,
)

BUSINESS = {
    "intact": ComponentEvidence("evaluated", "intact", "Business intact.", {"net_margin": 0.2}),
    "mixed": ComponentEvidence("evaluated", "mixed", "Business mixed.", {"net_margin": -0.01}),
    "broken": ComponentEvidence("evaluated", "broken", "Business broken.", {"net_margin": -0.2}),
    "unknown": ComponentEvidence("not_evaluated", "unknown", "Business inputs missing.", {}),
}
WATCH_FACTOR = {"code": "net_debt", "reason": "Debt exceeds cash.", "evidence": {"net_cash_positive": False}}
RUNWAY = {
    "pass": ComponentEvidence("evaluated", "adequate", "Runway sufficient.", {"cash_runway_quarters": 10}),
    "watch": ComponentEvidence("evaluated", "watch", "Runway needs review.", {"cash_runway_quarters": 6}),
    "critical": ComponentEvidence("evaluated", "critical", "Runway under four quarters.",
                                  {"cash_runway_quarters": 2}),
    "exempt": ComponentEvidence("not_applicable", "not_applicable", "Existing runway rule does not apply.", {}),
    "unknown": ComponentEvidence("not_evaluated", "unknown", "Cash-flow inputs missing.", {}),
}
# Table is the approved ordering, with expected output written independently of the policy.
# Columns: pass, watch, critical, exempt, unknown runway.
EXPECTED = {
    "intact": ("hold", "hold", "runway_critical", "hold", "thesis_unknown"),
    "mixed": ("hold", "hold", "runway_critical", "hold", "thesis_unknown"),
    "broken": ("business_broken", "business_broken", "runway_critical", "business_broken", "business_broken"),
    "unknown": ("thesis_unknown", "thesis_unknown", "runway_critical", "thesis_unknown", "thesis_unknown"),
}
EXPECTED_ACTION = {"hold": "hold", "runway_critical": "review_exit",
                   "business_broken": "review_exit", "thesis_unknown": "insufficient_data"}


def evidence(business="intact", runway="pass", applicability="applicable"):
    return HolderEvidence(applicability, BUSINESS[business], RUNWAY[runway], (deepcopy(WATCH_FACTOR),))


@pytest.mark.parametrize("business,runway", product(BUSINESS, RUNWAY))
def test_applicable_component_matrix(business, runway):
    expected = EXPECTED[business][tuple(RUNWAY).index(runway)]
    result = long_term_if_held(evidence(business, runway))
    assert result["action"] == EXPECTED_ACTION[expected]
    assert result["basis"] == ("no_demonstrated_thesis_failure" if expected == "hold" else expected)
    assert result["assessment_status"] == ("not_evaluated" if expected == "thesis_unknown" else "evaluated")
    reasons = []
    if runway == "critical":
        reasons.append("runway_critical")
    if business == "broken":
        reasons.append("business_broken")
    if business == "unknown":
        reasons.append("business_not_evaluated")
    if runway == "unknown":
        reasons.append("cash_runway_not_evaluated")
    if reasons:
        assert result["reasons"] == reasons
    assert result["checks"]["business"]["state"] == BUSINESS[business].state
    assert result["checks"]["cash_runway"]["state"] == RUNWAY[runway].state
    assert "add_allowed" not in result


@pytest.mark.parametrize("business,runway", product(BUSINESS, RUNWAY))
@pytest.mark.parametrize("applicability,action,basis,state", [
    ("not_applicable", "not_assessed", "unsupported_instrument", "not_applicable"),
    ("unverified", "insufficient_data", "applicability_unverified", "not_evaluated"),
])
def test_applicability_precedes_every_failure_combination(business, runway, applicability, action, basis, state):
    result = long_term_if_held(evidence(business, runway, applicability))
    assert result["action"] == action
    assert result["basis"] == basis
    assert result["assessment_status"] == state
    assert not result["watch_factors"]
    assert result["checks"]["business"]["state"] == state
    assert result["checks"]["cash_runway"]["state"] == state
    assert "runway_critical" not in result["reasons"]
    assert "business_broken" not in result["reasons"]
    assert "add_allowed" not in result


@pytest.mark.parametrize("business,runway,basis,reasons", [
    ("unknown", "critical", "runway_critical", ["runway_critical", "business_not_evaluated"]),
    ("broken", "unknown", "business_broken", ["business_broken", "cash_runway_not_evaluated"]),
    ("broken", "critical", "runway_critical", ["runway_critical", "business_broken"]),
])
def test_known_failure_preserves_unknown_or_other_failed_check(business, runway, basis, reasons):
    result = long_term_if_held(evidence(business, runway))
    assert result["action"] == "review_exit"
    assert result["basis"] == basis
    assert result["reasons"] == reasons


@pytest.mark.parametrize("component,classification", [("business", "broken"), ("cash_runway", "critical")])
def test_failure_classification_without_evaluated_state_is_not_proof(component, classification):
    components = {"business": BUSINESS["intact"], "cash_runway": RUNWAY["pass"]}
    components[component] = ComponentEvidence("not_evaluated", classification, "Not evaluated.", {})
    result = long_term_if_held(HolderEvidence("applicable", **components))
    assert result["action"] == "insufficient_data"
    assert result["basis"] == "thesis_unknown"
    assert result["reasons"] == [f"{component}_not_evaluated"]


def test_absent_evidence_is_not_a_pass():
    result = long_term_if_held(None)
    assert result["action"] == "insufficient_data"
    assert result["basis"] == "applicability_unverified"
    assert result["assessment_status"] == "not_evaluated"
    assert not result["watch_factors"]


def test_component_periods_are_kept_separate_without_changing_input():
    periods = {
        "free_cash_flow": {"period": "TTM", "period_end": "2026-06-30"},
        "operating_cash_flow": {"period": "FY 2025", "period_end": "2025-12-31"},
        "liquidity": {"period": None, "reason": "No input-specific period supplied."},
    }
    source = HolderEvidence("applicable", BUSINESS["mixed"], ComponentEvidence(
        "evaluated", "watch", "Runway needs review.",
        {"cash_runway_quarters": 6, "inputs": {"free_cash_flow": -100}}, periods,
        "Input periods differ; no composite period is assigned."), (deepcopy(WATCH_FACTOR),))
    original = deepcopy(source)
    result = long_term_if_held(source)
    assert source == original
    assert result["checks"]["cash_runway"]["periods"] == periods
    assert "period" not in result["checks"]["cash_runway"]
    assert result["checks"]["business"]["periods"] is None
    assert result["checks"]["business"]["periods_reason"]
    assert result["watch_factors"] == [WATCH_FACTOR]
    assert long_term_if_held(source) == result
    result["checks"]["cash_runway"]["evidence"]["inputs"]["free_cash_flow"] = 999
    result["checks"]["cash_runway"]["periods"]["free_cash_flow"]["period_end"] = "2099-01-01"
    result["watch_factors"][0]["evidence"]["net_cash_positive"] = True
    assert source == original


def test_applicable_company_cannot_skip_the_business_component():
    absent = ComponentEvidence('not_applicable', 'not_applicable', 'No business assessment.', {})
    result = long_term_if_held(HolderEvidence('applicable', absent, RUNWAY['exempt']))
    assert result['action'] == 'insufficient_data'
    assert result['reasons'] == ['business_not_evaluated']


@pytest.mark.parametrize("component,label", [
    (component, label)
    for component in ("business", "cash_runway")
    for label in ("unknown", "", "future_label")
] + [("business", "adequate"), ("cash_runway", "intact")])
@pytest.mark.parametrize("other_failed", [False, True])
def test_evaluated_requires_a_recognized_component_label(component, label, other_failed):
    components = {"business": BUSINESS["broken" if other_failed else "intact"],
                  "cash_runway": RUNWAY["critical" if other_failed else "pass"]}
    invalid = ComponentEvidence("evaluated", label, "Previously marked evaluated.",
                                {"reported_value": 12}, {"reported_value": None}, "Period unavailable.")
    components[component] = invalid
    source = HolderEvidence("applicable", **components)
    original = deepcopy(source)
    result = long_term_if_held(source)
    missing = f"{component}_not_evaluated"
    if other_failed:
        failure = "runway_critical" if component == "business" else "business_broken"
        assert result["action"] == "review_exit"
        assert result["basis"] == failure
        assert result["reasons"] == [failure, missing]
        assert result["assessment_status"] == "evaluated"
    else:
        assert result["action"] == "insufficient_data"
        assert result["basis"] == "thesis_unknown"
        assert result["reasons"] == [missing]
        assert result["assessment_status"] == "not_evaluated"
    check = result["checks"][component]
    assert check["state"] == "not_evaluated"
    assert check["classification"] == label  # Keep the unrecognized label as diagnostic evidence.
    assert "Unrecognized" in check["reason"]
    assert check["evidence"] == invalid.evidence
    assert check["periods"] == invalid.periods
    assert check["periods_reason"] == invalid.periods_reason
    assert any(component.replace("_", " ") in text for text in result["review_when"])
    assert source == original


def test_both_evaluated_unknown_labels_remain_two_missing_checks():
    unknown = ComponentEvidence("evaluated", "unknown", "Unrecognized.", {})
    result = long_term_if_held(HolderEvidence("applicable", unknown, unknown))
    assert result["action"] == "insufficient_data"
    assert result["reasons"] == ["business_not_evaluated", "cash_runway_not_evaluated"]
    assert all(check["state"] == "not_evaluated" for check in result["checks"].values())


@pytest.mark.parametrize("component", ["business", "cash_runway"])
@pytest.mark.parametrize("state", [None, "", "future_state", True, 42])
@pytest.mark.parametrize("other_failed", [False, True])
def test_invalid_component_states_are_missing_with_preserved_evidence(component, state, other_failed):
    components = {"business": BUSINESS["broken" if other_failed else "intact"],
                  "cash_runway": RUNWAY["critical" if other_failed else "pass"]}
    classification = "broken" if component == "business" else "critical"
    invalid = ComponentEvidence(state, classification, "Previously reported check.",
        {"reported_value": {"amount": 12}}, {"reported_value": {"period": "TTM"}}, "Input period supplied.")
    components[component] = invalid
    source = HolderEvidence("applicable", **components, watch_factors=(deepcopy(WATCH_FACTOR),))
    original = deepcopy(source)
    result = long_term_if_held(source)
    missing = f"{component}_not_evaluated"
    if other_failed:
        failure = "runway_critical" if component == "business" else "business_broken"
        assert result["action"] == "review_exit"
        assert result["basis"] == failure
        assert result["reasons"] == [failure, missing]
        assert result["assessment_status"] == "evaluated"
    else:
        assert result["action"] == "insufficient_data"
        assert result["basis"] == "thesis_unknown"
        assert result["reasons"] == [missing]
        assert result["assessment_status"] == "not_evaluated"
    check = result["checks"][component]
    assert check["state"] == "not_evaluated"
    assert check["classification"] == classification
    assert check["reason"] == f"Unrecognized {component.replace('_', ' ')} state; this check was not evaluated."
    assert check["evidence"] == invalid.evidence
    assert check["periods"] == invalid.periods
    assert check["periods_reason"] == invalid.periods_reason
    assert result["watch_factors"] == [WATCH_FACTOR]
    assert any(component.replace("_", " ") in text for text in result["review_when"])
    assert source == original
    check["evidence"]["reported_value"]["amount"] = 999
    check["periods"]["reported_value"]["period"] = "changed"
    assert source == original


@pytest.mark.parametrize("runway", ["exempt", "critical"])
def test_business_not_applicable_is_normalized_without_losing_known_runway_failure(runway):
    invalid = ComponentEvidence("not_applicable", "not_applicable", "Business skipped.",
        {"reported_value": 12}, {"reported_value": None}, "Input period unavailable.")
    source = HolderEvidence("applicable", invalid, RUNWAY[runway])
    original = deepcopy(source)
    result = long_term_if_held(source)
    check = result["checks"]["business"]
    assert check["state"] == "not_evaluated"
    assert check["reason"] == "Business evidence is required for an applicable instrument; this check was not evaluated."
    assert check["classification"] == invalid.classification
    assert check["evidence"] == invalid.evidence
    assert check["periods"] == invalid.periods
    assert check["periods_reason"] == invalid.periods_reason
    if runway == "critical":
        assert result["action"] == "review_exit"
        assert result["basis"] == "runway_critical"
        assert result["reasons"] == ["runway_critical", "business_not_evaluated"]
    else:
        assert result["action"] == "insufficient_data"
        assert result["basis"] == "thesis_unknown"
        assert result["reasons"] == ["business_not_evaluated"]
        assert result["checks"]["cash_runway"]["state"] == "not_applicable"
    assert source == original
