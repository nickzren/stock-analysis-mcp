"""WP2.0: the decision scenario matrix and its committed before table.

`tests/fixtures/decision_matrix/` records today's decisions and 2.0.0 snapshots for every
scenario in `tests/support/decision_matrix.py`. Later Step 2 WPs change rows on purpose and
record each intended change in their before/after table.
"""

import json
from datetime import UTC
from typing import Any

import pytest

from stock_analysis.tools.diff_analysis import validate_previous_snapshot
from tests.support import decision_matrix as dm

BEFORE = json.loads((dm.FIXTURE_DIR / "before.json").read_text())


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


class TestBeforeTable:
    async def test_the_committed_before_table_matches_the_code(self) -> None:
        """DM-CH1: every scenario's decision digest equals tests/fixtures/decision_matrix/before.json.
        Flipped row by row by WP2.1-2.4, each of which lists its intended changes."""
        table, _ = await dm.build()

        assert set(table) == set(BEFORE)
        for name, row in table.items():
            assert _normalized(row) == BEFORE[name], name

    async def test_the_recorded_snapshots_match_the_code(self) -> None:
        """DM-CH2: each scenario's watchlist_snapshot equals its recorded 2.0.0 fixture.
        Flipped by WP2.4 (snapshot 3.0.0); the 2.0.0 fixtures stay as recorded."""
        _, snapshots = await dm.build()

        assert {p.stem for p in dm.SNAPSHOT_DIR.glob("*.json")} == set(snapshots)
        for name, snapshot in snapshots.items():
            assert _normalized(snapshot) == json.loads((dm.SNAPSHOT_DIR / f"{name}.json").read_text()), name


class TestMatrixGuards:
    def test_every_scenario_pins_a_tz_aware_clock(self) -> None:
        """DM-G1: no scenario depends on the wall clock. Holds throughout Step 2."""
        for name, scenario in dm.SCENARIOS.items():
            assert scenario.now.tzinfo is not None, name

    @pytest.mark.parametrize("name", [n for n, s in dm.SCENARIOS.items() if s.technicals is not None])
    async def test_scenarios_price_off_a_current_completed_close(self, name: str) -> None:
        """DM-G2: the matrix baseline is fresh (plan §5): a completed-session close, 0 sessions
        behind, so a freshness gate fires only where a scenario says so. Holds throughout Step 2."""
        provenance = (await dm.run(name))["summary"]["price_provenance"]

        assert (provenance["price_source"], provenance["staleness"]["sessions_behind"]) == ("daily_close", 0)

    @pytest.mark.parametrize("path", sorted(dm.SNAPSHOT_DIR.glob("*.json")), ids=lambda p: p.stem)
    def test_recorded_snapshots_are_valid_2_0_0(self, path: Any) -> None:
        """DM-G3: the recorded fixtures are genuine 2.0.0 snapshots that detect_changes accepts.
        Holds throughout Step 2 (3.0.0 keeps 2.0.0 compatible, plan §8)."""
        snapshot = json.loads(path.read_text())

        assert snapshot["snapshot_version"] == "2.0.0"
        assert validate_previous_snapshot(snapshot, "ACME") == []

    def test_the_generator_refuses_to_record_from_another_snapshot_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DM-G4: snapshots_2_0_0/ can only be written by 2.0.0 code. Holds throughout Step 2."""
        monkeypatch.setattr(dm, "SNAPSHOT_VERSION", "3.0.0")

        with pytest.raises(SystemExit, match="refusing"):
            dm.write_fixtures()

    def test_the_default_clock_is_friday_evening_after_the_close(self) -> None:
        """DM-G5: the shared clock is off-session, after Friday 2026-10-09's close. Holds throughout Step 2."""
        assert dm.FRIDAY_EVENING.astimezone(UTC).isoformat() == "2026-10-10T00:00:00+00:00"
