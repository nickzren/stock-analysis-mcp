"""WP2.3 derived surfaces (plan §3 [D-3], §7 [D-9]): the executive summary's horizon segments, the decision
modes, the dislocation action and check, the dip caps and price_zone, all derived from decide()'s decision and
never feeding it. Annex: output_contract.md §4, §6, §7 and policy_holder.md §8, §10, "Corrected proposal", as
amended by plan §0 (#8, #9, #10, #11, #21, #23)."""

from typing import Any

import pytest

from stock_analysis.tools.analyze.action_zones import (
    PRICE_ZONES,
    _classify_current_zone,
    apply_dip_gates_to_action_zones,
    build_action_zones,
)
from tests.support import decision_matrix as dm

ROWS = list(dm.SCENARIOS)


# --- (e) price_zone --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROWS)
async def test_price_zone_is_the_uncapped_classification(name: str) -> None:
    """OC-RF6 (non-snapshot): action_zones.price_zone is in the closed set and equals the uncapped zone
    classification of the run's own price, levels and SMAs; no zone field remains in action_zones or the card at
    any tier. PD D-9."""
    result = await dm.run(name)
    zones = result["action_zones"]
    technicals = dm.SCENARIOS[name].technicals or {}
    averages = technicals.get("moving_averages") or {}
    price = result["summary"]["current_price"]
    expected = None if price is None else PRICE_ZONES[_classify_current_zone(
        price, zones["levels"], averages.get("sma_50"), averages.get("sma_200"))]

    assert zones["price_zone"] == expected
    assert zones["price_zone"] in {*PRICE_ZONES.values(), None}
    for tier, payload in dm.tiers(result).items():
        assert "current_zone" not in (payload.get("action_zones") or {}), tier
        assert set(payload["decision_card"]["entry"] or {}) <= {"current_price", "buy_zone_low", "buy_zone_high"}, tier
    assert dm.tiers(result)["standard"]["action_zones"]["price_zone"] == expected


AT_52W_LOW = {"current_price": 70.0, "moving_averages": {"sma_50": 80.0, "sma_200": 90.0},
              "price_position": {"week_52_low": 70.0, "week_52_high": 120.0}}
CALM_VOL = {"volatility": {"annualized": 0.22}, "atr": {"value": 1.0, "as_pct_of_price": 0.014}}


@pytest.mark.parametrize(("risk", "fund", "dip_type", "regime", "warning"), [
    ({"classification": "extreme"}, {"valuation": {"pe_trailing": 9.0}}, None, None, "zone_capped_due_to_extreme_risk"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 60.0}}, None, None, "zone_downgraded_valuation_headwind"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 9.0}}, "falling_knife", None, "zone_capped_falling_knife"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 9.0}}, "extended_decline", None,
     "zone_capped_extended_decline"),
    ({"classification": "medium"}, {"valuation": {"pe_trailing": 9.0}}, None, {"classification": "extreme"},
     "zone_capped_extreme_risk"),
])
def test_the_five_zone_caps_leave_price_zone_uncapped(
        risk: dict[str, Any], fund: dict[str, Any], dip_type: str | None, regime: dict[str, Any] | None,
        warning: str) -> None:
    """OC-RF6 (unit) and OC-CH7 (flipped): each of the five caps still emits its warning and still caps the
    candidate's legacy zone, while price_zone stays near_52w_low, the uncapped classification. PD D-9."""
    zones = build_action_zones(current_price=70.0, tech_data=AT_52W_LOW, risk_data=CALM_VOL, fund_data=fund,
                               risk_regime=risk)
    zones = apply_dip_gates_to_action_zones(zones, {"dip_classification": {"type": dip_type}}, regime)

    assert warning in zones["zone_warnings"]
    assert zones["current_zone"] != "strong_buy"
    assert zones["price_zone"] == "near_52w_low"


@pytest.mark.parametrize("name", ROWS)
async def test_near_52w_high_reads_the_same_on_price_zone(name: str) -> None:
    """G-ZONE (2.3): on every row the gate's input moves from the legacy zone to price_zone without changing its
    status: reduce iff near_52w_high, and null/undetermined iff null/undetermined, so decide() is unchanged. G."""
    result = await dm.run(name)
    legacy = result["diagnostics"]["pre_gate_policy"]["inputs"]["zone"]
    price_zone = result["action_zones"]["price_zone"]

    assert (legacy == "reduce") is (price_zone == "near_52w_high")
    assert (legacy in (None, "undetermined")) is (price_zone in (None, "undetermined"))
