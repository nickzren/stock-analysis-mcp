"""Gated starters list the condition for entry, not an unrelated full-position upgrade."""

import pytest

from tests.support import decision_matrix as dm


@pytest.mark.parametrize(("name", "mid_conditions"), [
    ("T1_extreme_risk_bullish", ["no earnings date or window within 5 days"]),
    ("T1_near_52w_high", ["no earnings date or window within 5 days",
                           "price leaves the near-52-week-high band"]),
])
async def test_gated_starter_conditions_follow_each_final_horizon(name: str, mid_conditions: list[str]) -> None:
    result = await dm.run(name, events=dm.NEAR)
    long_conditions = ["no earnings date or window within 5 days"]
    for tier, shown in dm.tiers(result).items():
        card = shown["decision_card"]
        assert card["new_money"]["action"] == "wait", tier
        assert card["long_term"]["new_money"]["action"] == "wait", tier
        assert card["conditions"]["buy_only_if"] == mid_conditions, tier
        assert card["long_term"]["new_money"]["conditions"]["buy_only_if"] == long_conditions, tier
    assert result["decision_modes"]["core"]["required_before_full_size"] == long_conditions
    assert result["dislocation_framework"]["action"]["buy_only_if"] == mid_conditions
