"""WP2.0: the decision scenario matrix and its committed baseline.

`tests/fixtures/decision_matrix/` records 0609fde's decisions (before.json) and 2.0.0 snapshots for
every baseline scenario in `tests/support/decision_matrix.py`. The baseline is never re-recorded: later
Step 2 WPs change rows on purpose and record the current table separately (after.json, `--after`); each
WP lists the cells it changes in ROW_CHANGES (DM-CH3). Scenarios added after WP2.0 are in after.json only.
"""

import hashlib
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

from stock_analysis.tools.diff_analysis import validate_previous_snapshot
from tests.support import decision_matrix as dm

BEFORE = json.loads((dm.FIXTURE_DIR / "before.json").read_text())
AFTER = json.loads((dm.FIXTURE_DIR / "after.json").read_text())
AFTER_ONLY = {"A7_near_52w_high_bullish", "H_burner_runway_2q"}  # WP2.2: D-4 and D-1c's runway clause

# The cells each WP changes (before -> after), over before.json's paths; snapshot_hash follows the
# snapshot fields (DM-CH2). WP2.2: D-2 mapping, D-5/D-5a combination and outcomes, D-1c thesis cap, D-3 core
# rule, and the knife override's removal (plan §1, B only).
CORE_CHANGES = {
    **dict.fromkeys(["S1_hold_or_add_clean", "S2_buy_clean", "S3_plain_avoid", "T2_starter_weak_setup_quality",
                     "T3_plain_wait_weak_setup", "V1_vendor_rich_plain_avoid", "D_quality_attractive_poor_chart"],
                    ("build_position", "buy")),
    **dict.fromkeys(["S4_earnings_gate_on_hold_or_add", "S5_earnings_gate_on_avoid", "C1_headwind_hold_or_add",
                     "C2_headwind_earnings_gate", "C3_headwind_plain_avoid", "D2_quality_poor_chart_earnings_3d",
                     "F3_quality_neutral_valuation_poor_chart"], ("build_position", "wait")),
    **dict.fromkeys(["S6_knife_dislocation", "S9_weak_liquidity", "C_quality_knife_dislocation"],
                    ("build_position", "avoid")),
    **dict.fromkeys(["T1_extreme_risk_bullish", "E2_mixed_fcf_bullish_chart"], ("not_core_yet", "wait")),
    **dict.fromkeys(["S7_etf_below_sma200", "ETF_strong", "ETF_bearish"], ("needs_more_data", "watch")),
    "S8_fundamentals_failed": ("needs_more_data", "insufficient_data"),
    "G_unsupported_currency_knife": ("watch", "avoid"),
}
ACTION_CHANGES = {
    **dict.fromkeys(["S1_hold_or_add_clean", "C1_headwind_hold_or_add", "E2_mixed_fcf_bullish_chart"],
                    ("hold_or_add", "starter")),
    **dict.fromkeys(["S4_earnings_gate_on_hold_or_add", "C2_headwind_earnings_gate"], ("wait_for_data", "wait")),
    "S7_etf_below_sma200": ("hold", "watch"),
    "S9_weak_liquidity": ("wait_for_data", "avoid"),
    "E_poor_business_bullish_chart": ("hold_or_add", "avoid"),
    "B_broken_bearish_knife": ("wait", "avoid"),
}
GATE_CHANGES = {"A_broken_bearish": ([], ["thesis_broken"]), "E_poor_business_bullish_chart": ([], ["thesis_broken"]),
                "B_broken_bearish_knife": (["falling_knife"], ["falling_knife", "thesis_broken"])}
KNIFE_OVERRIDE_CHANGE = ("wait_for_entry", "avoid")  # B's mid-term policy without the knife override
SNAPSHOT_CHANGES: dict[str, dict[str, Any]] = {
    name: {**({"decision_action_now": ACTION_CHANGES[name]} if name in ACTION_CHANGES else {}),
           **({"hard_gate_ids": GATE_CHANGES[name]} if name in GATE_CHANGES else {}),
           **({"action_mid_term": KNIFE_OVERRIDE_CHANGE} if name == "B_broken_bearish_knife" else {})}
    for name in {*ACTION_CHANGES, *GATE_CHANGES}
}
ROW_CHANGES: dict[str, dict[str, Any]] = {
    name: {**({"modes.core.action": CORE_CHANGES[name]} if name in CORE_CHANGES else {}),
           **({"action_now": ACTION_CHANGES[name]} if name in ACTION_CHANGES else {}),
           **({"hard_gates": GATE_CHANGES[name]} if name in GATE_CHANGES else {}),
           **{f"snapshot.{field}": change for field, change in SNAPSHOT_CHANGES.get(name, {}).items()},
           **(dict.fromkeys(("policy.mid_term", "horizon_fit.mid_term", "modes.balanced.action"),
                            KNIFE_OVERRIDE_CHANGE) if name == "B_broken_bearish_knife" else {})}
    for name in {*CORE_CHANGES, *SNAPSHOT_CHANGES}
}
BASELINE_SHA256 = "167642b05e592bf77317f45025fc563c3ed203e5a86dedc8b5e8df4a569f7e4d"


# V1's vendor subtrees and the tiers that project them (projection.py: governance, ownership_flow and
# options_signals are full-tier only).
V1_STANDARD_PATHS = [
    "news_summary.catalyst_intelligence.bullish", "news_summary.catalyst_intelligence.bearish",
    "news_summary.catalyst_intelligence.neutral", "news_summary.sentiment.overall",
    "short_interest.shares_short", "short_interest.short_pct_of_float", "short_interest.days_to_cover",
    "short_interest.short_change_mom", "short_interest.as_of_date",
    "ownership.insider_pct", "ownership.institutional_pct", "ownership.float_shares",
    "analyst_coverage.price_target_low", "analyst_coverage.price_target_median", "analyst_coverage.price_target_high",
    "fundamentals_summary.analyst_estimates.eps_next_year", "fundamentals_summary.analyst_estimates.rev_next_quarter",
    "section_summaries.ownership", "section_summaries.short_interest", "section_summaries.governance",
    "section_summaries.valuation_context",
]
V1_FULL_PATHS = V1_STANDARD_PATHS + [
    "options_signals.expiration_used", "options_signals.implied_volatility.atm_avg_iv",
    "options_signals.put_call_ratio.volume_based", "options_signals.unusual_activity.top_strikes",
    "ownership_flow.insider_activity.recent_transactions", "ownership_flow.institutional.top_holders",
    "governance.overall_risk", "quality.roic", "valuation_context.status", "decision_modes.speculative.next_90d_catalysts",
    "data_provenance.news", "data_provenance.options", "data_provenance.ownership",
]


def at(payload: dict[str, Any], path: str) -> Any:
    for key in path.split("."):
        payload = (payload or {}).get(key)
    return payload


def baseline_sha256(fixture_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = [fixture_dir / "before.json", fixture_dir / "README.md", *(fixture_dir / dm.SNAPSHOT_DIR.name).glob("*.json")]
    for path in sorted(paths):
        digest.update(path.relative_to(fixture_dir).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


@pytest.fixture
def one_row_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace dm.build with S1's recorded row and snapshot, so the generator tests stay fast."""
    snapshot = json.loads((dm.SNAPSHOT_DIR / "S1_hold_or_add_clean.json").read_text())
    built: dict[str, Any] = {"table": {"S1_hold_or_add_clean": BEFORE["S1_hold_or_add_clean"]},
                             "snapshots": {"S1_hold_or_add_clean": snapshot}}

    async def build() -> tuple[dict[str, Any], dict[str, Any]]:
        return built["table"], built["snapshots"]

    monkeypatch.setattr(dm, "build", build)
    return built


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _flat(row: Any, path: str = "") -> dict[str, Any]:
    if not isinstance(row, dict):
        return {path: row}
    return {k: v for key, value in row.items() for k, v in _flat(value, f"{path}.{key}" if path else key).items()}


class TestBeforeTable:
    async def test_the_committed_after_table_matches_the_code(self) -> None:
        """DM-CH1: every scenario's decision digest equals tests/fixtures/decision_matrix/after.json, recorded
        with `--after` (WP2.2 first changed rows; before.json stays as recorded, DM-G6). CH: re-recorded by every
        WP that changes a row, which extends ROW_CHANGES (DM-CH3)."""
        table, _ = await dm.build()

        assert set(table) == set(AFTER) == set(BEFORE) | AFTER_ONLY
        for name, row in table.items():
            assert _normalized(row) == AFTER[name], name

    async def test_the_recorded_snapshots_match_the_code(self) -> None:
        """DM-CH2: each baseline scenario's watchlist_snapshot equals its recorded 2.0.0 fixture except for the
        fields SNAPSHOT_CHANGES lists (decision_action_now follows new_money, hard_gate_ids gains thesis_broken,
        B's action_mid_term loses the knife override, WP2.2), and the hash moves exactly on those rows.
        CH: retired by WP2.4 (snapshot 3.0.0). The 2.0.0 fixtures stay as recorded (DM-G6)."""
        _, snapshots = await dm.build()
        recorded = {p.stem: json.loads(p.read_text()) for p in dm.SNAPSHOT_DIR.glob("*.json")}

        assert set(recorded) == set(BEFORE) <= set(snapshots)
        for name, old in recorded.items():
            new = _normalized(snapshots[name])
            changed = {k: (old[k], new[k]) for k in old if k != "snapshot_hash" and new[k] != old[k]}
            assert set(new) == set(old), name
            assert changed == {k: tuple(v) for k, v in SNAPSHOT_CHANGES.get(name, {}).items()}, name
            assert (new["snapshot_hash"] != old["snapshot_hash"]) is bool(changed), name

    def test_the_after_table_changes_only_the_listed_cells(self) -> None:
        """DM-CH3: after.json differs from before.json, over before.json's paths, exactly in ROW_CHANGES;
        the after-only rows are the listed ones. CH: each WP that re-records after.json extends ROW_CHANGES."""
        delta: dict[str, dict[str, Any]] = {}
        for name, row in BEFORE.items():
            old, new = _flat(row), _flat(AFTER[name])
            cells = {k: (old[k], new[k]) for k in old if k != "snapshot.snapshot_hash" and new[k] != old[k]}
            if cells:
                delta[name] = cells

        assert set(AFTER) - set(BEFORE) == AFTER_ONLY
        assert delta == {name: {k: tuple(v) for k, v in cells.items()} for name, cells in ROW_CHANGES.items()}


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
        self, monkeypatch: pytest.MonkeyPatch, one_row_build: dict[str, Any], tmp_path: Path
    ) -> None:
        """DM-G4: snapshots_2_0_0/ can only be written by 2.0.0 code. Holds throughout Step 2."""
        monkeypatch.setattr(dm, "SNAPSHOT_VERSION", "3.0.0")

        with pytest.raises(SystemExit, match="^refusing to record 2.0.0 snapshots from SNAPSHOT_VERSION 3.0.0$"):
            dm.record_baseline(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_the_default_clock_is_friday_evening_after_the_close(self) -> None:
        """DM-G5: the shared clock is off-session, after Friday 2026-10-09's close. Holds throughout Step 2."""
        assert dm.FRIDAY_EVENING.astimezone(UTC).isoformat() == "2026-10-10T00:00:00+00:00"

    def test_the_committed_baseline_is_byte_for_byte_as_recorded(self) -> None:
        """DM-G6: before.json, README.md and snapshots_2_0_0/ are exactly the files recorded from 0609fde
        at WP2.0. Holds throughout Step 2: the baseline is never re-recorded."""
        assert baseline_sha256(dm.FIXTURE_DIR) == BASELINE_SHA256

    def test_recording_the_baseline_never_overwrites_it(self, one_row_build: dict[str, Any], tmp_path: Path) -> None:
        """DM-G7: --baseline writes missing files and accepts identical ones, but refuses, writing nothing,
        when an existing file would change. Holds throughout Step 2."""
        dm.record_baseline(tmp_path)
        recorded = baseline_sha256(tmp_path)
        dm.record_baseline(tmp_path)
        assert baseline_sha256(tmp_path) == recorded

        snapshot_file = tmp_path / dm.SNAPSHOT_DIR.name / "S1_hold_or_add_clean.json"
        snapshot_text = snapshot_file.read_text()
        (tmp_path / "README.md").unlink()
        snapshots = one_row_build["snapshots"]
        snapshots["S1_hold_or_add_clean"] = {**snapshots["S1_hold_or_add_clean"], "decision_action_now": "wait"}

        with pytest.raises(SystemExit, match="refusing to overwrite the recorded baseline: snapshots_2_0_0/S1"):
            dm.record_baseline(tmp_path)
        assert snapshot_file.read_text() == snapshot_text
        assert not (tmp_path / "README.md").exists()

    def test_a_table_only_change_is_refused_too(self, one_row_build: dict[str, Any], tmp_path: Path) -> None:
        """DM-G7: a changed row with an unchanged snapshot (WP2.2-2.3 move modes, dip and sizing that way)
        is refused for before.json and README.md. Holds throughout Step 2."""
        dm.record_baseline(tmp_path)
        recorded = baseline_sha256(tmp_path)
        one_row_build["table"]["S1_hold_or_add_clean"] = {**BEFORE["S1_hold_or_add_clean"], "action_now": "wait"}

        with pytest.raises(SystemExit, match="refusing to overwrite the recorded baseline: before.json, README.md$"):
            dm.record_baseline(tmp_path)
        assert baseline_sha256(tmp_path) == recorded

    def test_the_after_table_leaves_the_baseline_alone(self, one_row_build: dict[str, Any], tmp_path: Path) -> None:
        """DM-G8: --after writes only after.json and after.md. Holds throughout Step 2."""
        dm.record_baseline(tmp_path)
        recorded = baseline_sha256(tmp_path)
        one_row_build["table"]["S1_hold_or_add_clean"] = {**BEFORE["S1_hold_or_add_clean"], "action_now": "wait"}

        dm.write_after(tmp_path)

        assert baseline_sha256(tmp_path) == recorded
        assert json.loads((tmp_path / "after.json").read_text())["S1_hold_or_add_clean"]["action_now"] == "wait"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["README.md", "after.json", "after.md", "before.json",
                                                              dm.SNAPSHOT_DIR.name]

    async def test_v1_carries_every_vendor_subtree(self) -> None:
        """DM-G9: V1 populates catalyst, options, ownership, short-interest, analyst, governance, quality and
        valuation-context data at every tier that projects them, so the output-contract checks see real
        vendor leaves; the decision tier still shows the insider pro and signal. Holds throughout Step 2."""
        shown = dm.tiers(await dm.run("V1_vendor_rich_plain_avoid"))

        for tier, paths in (("standard", V1_STANDARD_PATHS), ("full", V1_FULL_PATHS)):
            empty = [path for path in paths if at(shown[tier], path) in (None, "", [], {})]
            assert empty == [], tier
        assert shown["decision"]["watchlist_snapshot"]["bullish_signals"] == ["insider_buying"]
        assert shown["decision"]["verdict"]["pros"] == ["Insider buying activity (net positive 3M)"]
