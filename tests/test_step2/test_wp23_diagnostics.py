"""WP2.3: the pre-gate candidate moves to diagnostics (plan §2, §7a [D-9], §0 #6, #22).

Top-level policy_action becomes the full-tier diagnostics.pre_gate_policy (support-capped, with role and
superseded_by; no alias), decision_card.rationale becomes diagnostics.decision_trace and card.horizon_fit goes.
The 2.0.0 snapshot keeps its fields, read from the new places, until WP2.4 (§0 #22). Annex: output_contract.md
§8, §10 RF11 and policy_holder.md §10 RF3, RF9, "Corrected proposal", as amended by plan §0.
"""

from typing import Any

import pytest

from stock_analysis.tools.analyze.decision import ENTRY_ACTIONS, ENTRY_CANDIDATE_CODES
from tests.support import decision_matrix as dm

UNSUPPORTED = ["S7_etf_below_sma200", "ETF_strong", "ETF_bearish", "GBp_strong", "G_unsupported_currency_knife"]
PRE_GATE_KEYS = ["mid_term", "long_term", "rationale", "conditions_to_upgrade", "conditions_to_downgrade", "inputs"]


def ids(blockers: list[dict[str, Any]]) -> list[str]:
    return [b["id"] for b in blockers]


@pytest.mark.parametrize("name", list(dm.SCENARIOS))
async def test_the_pre_gate_policy_is_a_full_tier_diagnostic(name: str) -> None:
    """OC-RF11 (non-snapshot paths): the top-level policy_action is gone at every tier with no alias; only the
    full tier carries diagnostics = {pre_gate_policy, decision_trace}; pre_gate_policy keeps the candidate
    codes, trace and conditions, moves zone and the valuation fields under inputs, drops sizing_guidance, and
    carries role and superseded_by equal to the card; the card keeps no copy (no rationale, no horizon_fit).
    PD D-9."""
    result = await dm.run(name)
    card = result["decision_card"]
    shown = dm.tiers(result)
    pre_gate = result["diagnostics"]["pre_gate_policy"]

    for tier, payload in shown.items():
        assert "policy_action" not in payload, tier
        assert ("diagnostics" in payload) is (tier == "full"), tier
        assert not {"rationale", "horizon_fit"} & payload["decision_card"].keys(), tier
    assert list(result["diagnostics"]) == ["pre_gate_policy", "decision_trace"]
    support_capped = "support_gates" in pre_gate
    assert list(pre_gate) == [*PRE_GATE_KEYS, *(["support_gates"] if support_capped else []), "role",
                              "superseded_by"]
    assert list(pre_gate["inputs"]) == ["zone", "valuation_gate", "valuation_basis", "is_unprofitable"]
    assert pre_gate["role"] == "pre_gate_input_not_advice"
    assert pre_gate["superseded_by"] == {
        "new_money": card["new_money"]["action"], "if_held": card["if_held"]["action"],
        "long_term": card["long_term"]["new_money"]["action"], "blocked_by": ids(card["new_money"]["blocked_by"])}
    assert pre_gate["inputs"]["valuation_gate"] == result["action_zones"]["valuation_assessment"]["gate"]
    trace = [f"pre_gate_policy.mid_term={pre_gate['mid_term']} -> {card['new_money']['action']}"]
    assert result["diagnostics"]["decision_trace"] == trace + (
        [f"decided_by={card['new_money']['decided_by']}"] if card["new_money"]["decided_by"] else [])


@pytest.mark.parametrize("name", list(dm.SCENARIOS))
async def test_the_interim_snapshot_reads_the_diagnostics(name: str) -> None:
    """§0 #22 pin (plan §7a, OC-G6): until WP2.4 the 2.0.0 snapshot keeps its fields: action_mid_term,
    action_long_term and zone read diagnostics.pre_gate_policy (codes, inputs.zone); valuation_gate and
    is_unprofitable read action_zones.valuation_assessment. PD D-9. CH: flipped by WP2.4 (snapshot 3.0.0)."""
    result = await dm.run(name)
    pre_gate = result["diagnostics"]["pre_gate_policy"]
    assessment = result["action_zones"]["valuation_assessment"]
    snapshot = result["watchlist_snapshot"]

    assert (snapshot["action_mid_term"], snapshot["action_long_term"], snapshot["zone"]) == (
        pre_gate["mid_term"], pre_gate["long_term"], pre_gate["inputs"]["zone"])
    assert (snapshot["valuation_gate"], snapshot["is_unprofitable"]) == (
        assessment["gate"], assessment["is_unprofitable"] or None)


@pytest.mark.parametrize("name", ["A_broken_bearish", "B_broken_bearish_knife"])
async def test_the_emitted_candidate_has_no_knife_override(name: str) -> None:
    """PH-RF3 (candidate clause): A and B (A plus a falling knife) both emit pre_gate_policy.mid_term avoid with no
    knife rationale (v0.2.0's knife override turned B into wait_for_entry); superseded_by shows the final avoid /
    review_exit next to it. RF (PD D-1c)."""
    result = await dm.run(name)
    pre_gate = result["diagnostics"]["pre_gate_policy"]
    blocked = {"A_broken_bearish": ["thesis_broken"], "B_broken_bearish_knife": ["falling_knife", "thesis_broken"]}

    assert (pre_gate["mid_term"], pre_gate["long_term"]) == ("avoid", "avoid_core_position")
    assert not [line for line in pre_gate["rationale"] if "falling_knife" in line]
    assert pre_gate["superseded_by"] == {"new_money": "avoid", "if_held": "review_exit", "long_term": "avoid",
                                         "blocked_by": blocked[name]}


@pytest.mark.parametrize("name", UNSUPPORTED)
async def test_an_unsupported_instrument_emits_no_new_money_candidate(name: str) -> None:
    """PH-RF9 (WP4 support cap on the emitted candidate, plan §0 #6): for an ETF, a GBp or EUR stock the emitted
    pre_gate_policy codes are not new-money codes (support-capped at watch) and support_gates names the
    support gates new_money discloses; new_money admits nothing. RF."""
    result = await dm.run(name)
    pre_gate = result["diagnostics"]["pre_gate_policy"]
    new_money = result["decision_card"]["new_money"]

    assert not {pre_gate["mid_term"], pre_gate["long_term"]} & ENTRY_CANDIDATE_CODES
    assert pre_gate["support_gates"] == [g for g in ids(new_money["blocked_by"]) if g.startswith("unsupported_")]
    assert new_money["action"] not in ENTRY_ACTIONS
