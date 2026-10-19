"""Elapsed holding days are descriptive calendar arithmetic, never a tax category."""

from datetime import UTC, datetime, timedelta

import pytest

from stock_analysis.tools import position
from tests.support import decision_matrix as dm
from tests.support import holder_matrix as hm


@pytest.mark.parametrize("days", [364, 365, 366, 500])
async def test_holding_days_do_not_generate_a_tax_classification(days):
    now = hm.POSITION_CLOCK
    purchased = (now.date() - timedelta(days=days)).isoformat()
    with dm.stubbed(hm.SOUND):
        result = await position.analyze_position("ACME", purchase_date=purchased, _now=now)
    assert result["holding"] == {"purchase_date": purchased, "days_held": days}
    assert result["tax"]["status"] == "not_assessed"
    assert set(result["tax"]) == {"status", "reason"}


@pytest.mark.parametrize("now,expected", [
    (datetime(2026, 10, 10, 3, 59, tzinfo=UTC), 0),
    (datetime(2026, 10, 10, 4, 1, tzinfo=UTC), 1),
])
async def test_elapsed_days_switch_at_et_midnight_not_utc(now, expected):
    with dm.stubbed(hm.SOUND):
        result = await position.analyze_position("ACME", purchase_date="2026-10-09", _now=now)
    assert result["holding"] == {"purchase_date": "2026-10-09", "days_held": expected}


async def test_leap_date_counts_elapsed_days_without_one_year_tax_shortcut():
    now = datetime(2025, 2, 28, 17, tzinfo=UTC)
    with dm.stubbed(hm.SOUND):
        result = await position.analyze_position("ACME", purchase_date="2024-02-29", _now=now)
    assert result["holding"] == {"purchase_date": "2024-02-29", "days_held": 365}
    assert result["tax"]["status"] == "not_assessed"
