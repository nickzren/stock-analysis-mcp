"""WP2.2: earliest_reassessment (plan §6; BL-D9 rule half).

Annex: docs/plans/step2-v2-research/reassessment.md, "Corrected proposal" (rules R1-R9, worked examples E1-E29,
tests RA.2-RA.13, RA.17), as amended by plan §0 #11 (analyze only, per horizon: long_term.new_money carries its
own object; not on the card), #13 (outstanding in registry rank order) and §6 (presence: blocked_by non-empty and
map(candidate) in {buy, starter} at that horizon). The date is a day to re-run analyze, never permission to buy.
Pure runs call earliest_reassessment directly or decide() through dm.decide_on.
"""

import inspect
from datetime import date
from itertools import permutations
from typing import Any

import pytest

from stock_analysis.tools.analyze import reassessment
from stock_analysis.tools.analyze.gates import (
    THESIS_CLEARS_WHEN,
    EarningsWindow,
    earnings_window,
)
from tests.support import decision_matrix as dm

D = date


def undated(*outstanding: dict[str, str], basis: str = "conditions_only") -> dict[str, Any]:
    return {"date": None, "basis": basis, "earnings_risk_through": None, "session_method": None,
            "outstanding": list(outstanding)}


@pytest.mark.parametrize("order", list(permutations(["earnings_blackout", "earnings_unverifiable",
                                                     "data_quality_critical", "falling_knife", "stale_data"])))
def test_outstanding_rank_order(order: tuple[str, ...]) -> None:
    """RA.7: whatever order the gates fire in, outstanding is in registry rank order (plan §0 #13), the event
    gates collapse into one earnings_reverify listed last, and the blackout dates the object. RF."""
    today = D(2026, 10, 12)
    window = earnings_window({"days_until": 3, "next_date": "2026-10-15"}, today)
    result = dm.decide_on({"mid_term": "buy", "long_term": "accumulate"},
                          [{"id": g, "reason": "r"} for g in order], support=None)
    blocked = result["new_money"]["blocked_by"]
    obj = reassessment.earliest_reassessment(blocked_by=blocked, admits=True, window=window, today=today,
                                             support=None)

    assert [o["id"] for o in obj["outstanding"]] == ["stale_data", "data_quality_critical", "falling_knife",
                                                     "earnings_reverify"]
    assert obj["date"] == "2026-10-16"


def test_no_clock_call() -> None:
    """RA.12: the reassessment and the clear-session helper never read the clock. G."""
    for source in (inspect.getsource(reassessment), inspect.getsource(EarningsWindow.clear_session)):
        assert "now(" not in source and "today(" not in source


def test_the_thesis_condition_names_its_component() -> None:
    """Thesis condition: the thesis gate's outstanding condition is clears_when_for the blocker, only the failing
    component (plan §4, PH-RF12b), so case E never asks for a runway figure. G."""
    window = earnings_window({}, D(2026, 10, 9))
    blocked = [{"id": "thesis_broken", "reason": "r", "codes": ["business_broken"], "class": "thesis",
                "effect": "cap"}]
    obj = reassessment.earliest_reassessment(blocked_by=blocked, admits=True, window=window, today=D(2026, 10, 9),
                                             support=None)

    assert obj == undated({"id": "thesis_broken", "clears_when": THESIS_CLEARS_WHEN["business_broken"]})
