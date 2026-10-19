"""Step3a.1 pure component rules and unchanged released decisions/snapshots."""

import copy
from datetime import date

import pytest

from stock_analysis.tools.analyze import decision
from stock_analysis.tools.analyze.gates import earnings_window
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm


def decide_with(evidence=None, **overrides):
    values = {"candidate": {"mid_term": "buy", "long_term": "accumulate"},
              "blocking": [], "unevaluated": [], "thesis": (), "dislocation": False,
              "tilt": "bullish", "window": earnings_window({}, date(2026, 10, 9)),
              "today": date(2026, 10, 9), "support": None, "holder_evidence": evidence}
    return decision.decide(**{**values, **overrides})


async def test_longterm_holder_is_computed_at_every_tier_without_snapshot_change():
    result = await hm.run_stock("sound_bullish")
    held = result["decision_card"]["long_term"]["if_held"]
    assert held["action"] == "hold"
    assert held["basis"] == "no_demonstrated_thesis_failure"
    assert held["assessment_status"] == "evaluated"
    assert held["add_allowed"] is True
    for payload in dm.tiers(result).values():
        assert payload["decision_card"]["long_term"]["if_held"] == held
        assert payload["watchlist_snapshot"] == result["watchlist_snapshot"]
        assert payload["watchlist_snapshot"]["snapshot_version"] == "3.0.0"
        assert "long_term_if_held_action" not in payload["watchlist_snapshot"]


@pytest.mark.parametrize("name,action,basis", [
    ("sound_bullish", "hold", "no_demonstrated_thesis_failure"),
    ("sound_bearish", "hold", "no_demonstrated_thesis_failure"),
    ("sound_knife", "hold", "no_demonstrated_thesis_failure"),
    ("hold_with_avoid_fit", "hold", "no_demonstrated_thesis_failure"),
    ("broken_bullish", "review_exit", "business_broken"),
    ("critical_runway", "review_exit", "runway_critical"),
    ("fundamentals_failed", "insufficient_data", "thesis_unknown"),
    ("classification_unknown", "insufficient_data", "applicability_unverified"),
    ("etf_spy", "not_assessed", "unsupported_instrument"),
    ("etf_vti", "not_assessed", "unsupported_instrument"),
    ("preferred", "not_assessed", "unsupported_instrument"),
])
async def test_observed_components_drive_the_approved_longterm_rules(name, action, basis):
    result = await hm.run_stock(name)
    holder = result["decision_card"]["long_term"]["if_held"]
    assert (holder["action"], holder["basis"]) == (action, basis)
    assert holder["add_allowed"] == (result["decision_card"]["long_term"]["new_money"]["action"] in {"buy", "starter"})


def test_legacy_internal_decide_without_evidence_does_not_assume_checks_pass():
    held = decide_with()["long_term"]["if_held"]
    assert held["action"] == "insufficient_data"
    assert held["basis"] == "applicability_unverified"


async def test_longterm_addition_uses_its_own_horizon():
    result = await hm.run_stock("near_high")
    card = result["decision_card"]
    assert card["if_held"]["add_allowed"] is False
    assert card["long_term"]["if_held"]["add_allowed"] is True


async def test_chart_profile_and_entry_gates_never_become_longterm_exit_rules():
    base = (await hm.run_stock("sound_bullish"))["decision_card"]["long_term"]["if_held"]
    for name in ("sound_bearish", "sound_knife", "sound_core", "sound_speculative", "earnings", "stale_daily", "eur_stock"):
        held = copy.deepcopy((await hm.run_stock(name))["decision_card"]["long_term"]["if_held"])
        reference = copy.deepcopy(base)
        held.pop("add_allowed")
        reference.pop("add_allowed")
        assert held == reference, name
