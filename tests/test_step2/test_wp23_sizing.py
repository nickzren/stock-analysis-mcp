"""WP2.3: entry-sizing presence (plan §7 [FIX] keyed on entry_allowed, [D-9] a starter hides the full size).

Annex: docs/plans/step2-v2-research/output_contract.md §3, §10 "Corrected proposal", as amended by plan §0 (#12
entry_allowed, #21 mode max_position, #23 starter). Sizing numbers are unchanged (OC-G3); only presence moves.
"""

import pytest

from stock_analysis.tools.analyze.decision import FULL_ENTRY_CODES
from tests.support import decision_matrix as dm

STARTER_ROWS = ["S1_hold_or_add_clean", "T1_extreme_risk_bullish", "T2_starter_weak_setup_quality",
                "C1_headwind_hold_or_add", "E2_mixed_fcf_bullish_chart", "T4_starter_buy_the_dip"]


@pytest.mark.parametrize("name", STARTER_ROWS)
async def test_a_starter_shows_no_full_entry_dip_code(name: str) -> None:
    """OC-RF3 (dip clause; lands with the dip area's §7 cap): under new_money starter neither the dip
    recommendation nor any entry signal is a full-entry code (strong_buy_the_dip, buy_the_dip,
    cautious_accumulation, accumulate, add_position). PD D-9."""
    result = await dm.run(name)
    assessment = result["dip_assessment"]["assessment"]
    signals = result["dip_assessment"]["entry_timing"]["signals"]

    assert result["decision_card"]["new_money"]["action"] == "starter"
    assert assessment["recommendation"] not in FULL_ENTRY_CODES
    assert not FULL_ENTRY_CODES & {signal["action"] for signal in signals}
