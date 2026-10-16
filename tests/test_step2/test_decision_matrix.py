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

from stock_analysis.tools.analyze.gates import GATES, THESIS_CLEARS_WHEN
from stock_analysis.tools.diff_analysis import validate_previous_snapshot
from stock_analysis.utils.normalize import SNAPSHOT_FIELDS
from tests.support import decision_matrix as dm

BEFORE = json.loads((dm.FIXTURE_DIR / "before.json").read_text())
AFTER = json.loads((dm.FIXTURE_DIR / "after.json").read_text())
AFTER_ONLY = {"A7_near_52w_high_bullish", "H_burner_runway_2q",  # WP2.2: D-4 and D-1c's runway clause
              "T4_starter_buy_the_dip",  # WP2.3: the plan §7 starter scenario (OC-RF3, extended)
              # WP2.3 review: the dislocation starter and its buy control (OC-M9), per-horizon targets (OC-RF7)
              "T5_dislocation_starter", "T6_dislocation_buy", "T6_dislocation_starter_extreme",
              "T1_near_52w_high", "T1_crash_risk"}

# The cells each WP changes (before -> after), over before.json's paths; snapshot_hash follows the
# snapshot fields (DM-CH2 until WP2.4). WP2.2: D-2 mapping, D-5/D-5a combination and outcomes, D-1c thesis cap, D-3 core
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
WP22_CHANGES: dict[str, dict[str, Any]] = {
    name: {**({"modes.core.action": CORE_CHANGES[name]} if name in CORE_CHANGES else {}),
           **({"action_now": ACTION_CHANGES[name]} if name in ACTION_CHANGES else {}),
           **({"hard_gates": GATE_CHANGES[name]} if name in GATE_CHANGES else {}),
           **{f"snapshot.{field}": change for field, change in SNAPSHOT_CHANGES.get(name, {}).items()},
           **(dict.fromkeys(("policy.mid_term", "horizon_fit.mid_term", "modes.balanced.action"),
                            KNIFE_OVERRIDE_CHANGE) if name == "B_broken_bearish_knife" else {})}
    for name in {*CORE_CHANGES, *SNAPSHOT_CHANGES}
}

# WP2.3: every surface derives from decide()'s decision, which does not move (no decision.*, action_now,
# hard_gates, policy.*, zone or snapshot.* cell); the card's horizon_fit is removed (plan §7a), so B's WP2.2
# horizon_fit.mid_term cell now ends at None. Entry rows (new_money) keep their sizing; the rest lose it.
ENTRY_ROWS = {"S2_buy_clean": "buy", **dict.fromkeys(["S1_hold_or_add_clean", "T1_extreme_risk_bullish",
              "T2_starter_weak_setup_quality", "C1_headwind_hold_or_add", "E2_mixed_fcf_bullish_chart"], "starter")}
SIZING_KEYS = ("starter_pct", "full_pct_cap", "starter_dollars", "full_dollars", "whole_shares")
STARTER_HIDES = ("full_pct_cap", "full_dollars")  # plan §7 [D-9], §0 #23
BALANCED_CHANGES = {  # D-3: balanced = new_money (B keeps its WP2.2 knife-override cell, now wait_for_entry -> avoid)
    **dict.fromkeys(["S1_hold_or_add_clean", "C1_headwind_hold_or_add", "E2_mixed_fcf_bullish_chart"],
                    ("hold_or_add", "starter")),
    **dict.fromkeys(["T1_extreme_risk_bullish", "T2_starter_weak_setup_quality"], ("speculative_small_position", "starter")),
    **dict.fromkeys(["S4_earnings_gate_on_hold_or_add", "C2_headwind_earnings_gate"], ("hold_or_add", "wait")),
    **dict.fromkeys(["S9_weak_liquidity", "E_poor_business_bullish_chart"], ("hold_or_add", "avoid")),
    "S7_etf_below_sma200": ("hold", "watch"), "T3_plain_wait_weak_setup": ("wait_for_entry", "wait"),
}
LONG_VIEW_CHANGES = {  # D-3, PH-RF19: balanced.long_term_view = long_term.new_money.action (plan §0 #8)
    **dict.fromkeys(["S1_hold_or_add_clean", "S2_buy_clean", "S3_plain_avoid", "T2_starter_weak_setup_quality",
                     "T3_plain_wait_weak_setup", "V1_vendor_rich_plain_avoid", "D_quality_attractive_poor_chart"],
                    ("accumulate", "buy")),
    **dict.fromkeys(["S4_earnings_gate_on_hold_or_add", "S5_earnings_gate_on_avoid",
                     "D2_quality_poor_chart_earnings_3d"], ("accumulate", "wait")),
    **dict.fromkeys(["S6_knife_dislocation", "S9_weak_liquidity", "C_quality_knife_dislocation"], ("accumulate", "avoid")),
    **dict.fromkeys(["C1_headwind_hold_or_add", "C2_headwind_earnings_gate", "C3_headwind_plain_avoid"],
                    ("hold_size_conservatively", "wait")),
    **dict.fromkeys(["A_broken_bearish", "B_broken_bearish_knife", "E_poor_business_bullish_chart"],
                    ("avoid_core_position", "avoid")),
    **dict.fromkeys(["T1_extreme_risk_bullish", "E2_mixed_fcf_bullish_chart"], ("small_position_with_stops", "starter")),
    **dict.fromkeys(["S7_etf_below_sma200", "ETF_strong", "ETF_bearish"], ("insufficient_data", "watch")),
    "F3_quality_neutral_valuation_poor_chart": ("hold", "wait"), "G_unsupported_currency_knife": ("watch", "avoid"),
}
SPECULATIVE_CHANGES = {  # D-3: its own code mapped, then capped by new_money (S-C12)
    **dict.fromkeys(["S9_weak_liquidity", "A_broken_bearish", "B_broken_bearish_knife", "D_quality_attractive_poor_chart",
                     "D2_quality_poor_chart_earnings_3d", "E_poor_business_bullish_chart",
                     "F3_quality_neutral_valuation_poor_chart"], ("wait", "avoid")),
    **dict.fromkeys(["S3_plain_avoid", "S5_earnings_gate_on_avoid", "C3_headwind_plain_avoid",
                     "V1_vendor_rich_plain_avoid", "G_unsupported_currency_knife", "ETF_bearish"], ("watch", "avoid")),
    **dict.fromkeys(["S6_knife_dislocation", "C_quality_knife_dislocation"], ("starter_position_only", "avoid")),
    "S2_buy_clean": ("starter_then_add", "starter"), "S8_fundamentals_failed": ("wait", "insufficient_data"),
    "GBp_strong": ("wait", "watch"),
}
DISLOCATION_CHANGES = {  # D-9 (plan §7): can_start_now == entry_allowed; buy_now = new_money.action iff entry
    "S2_buy_clean": {"can_start_now": (False, True), "buy_now": (None, "buy")},
    **dict.fromkeys(["S1_hold_or_add_clean", "C1_headwind_hold_or_add", "E2_mixed_fcf_bullish_chart"],
                    {"can_start_now": (False, True), "buy_now": (None, "starter")}),
    **dict.fromkeys(["T1_extreme_risk_bullish", "T2_starter_weak_setup_quality"], {"buy_now": ("small_starter", "starter")}),
    **dict.fromkeys(["S6_knife_dislocation", "C_quality_knife_dislocation"],
                    {"can_start_now": (True, False), "buy_now": ("small_starter", None)}),
}
DIP_CHANGES = {  # D-9 (plan §7): non-entry caps entry codes to watch; the dislocation override needs a starter
    **dict.fromkeys(["S3_plain_avoid", "S5_earnings_gate_on_avoid", "C3_headwind_plain_avoid",
                     "V1_vendor_rich_plain_avoid"],
                    {"recommendation": ("buy_the_dip", "watch"),
                     "entry_actions": (["accumulate", "starter_position"], ["watch", "watch"])}),
    **dict.fromkeys(["S6_knife_dislocation", "C_quality_knife_dislocation"],
                    {"recommendation": ("starter_only_wait_for_stabilization", "do_not_catch_falling_knife"),
                     "entry_actions": (["starter_position"], ["watch"])}),
    "B_broken_bearish_knife": {"entry_actions": (["starter_position"], ["watch"])},
}
EARNINGS, LIQUIDITY = GATES["earnings_blackout"].clears_when, GATES["weak_liquidity"].clears_when
PE_HEADWIND_OLD = ["P/E < 25x (currently headwind)"]
CONDITION_CHANGES = {  # plan §7e [FIX BL-E1, BL-E2]: binding conditions only; add iff if_held hold
    **dict.fromkeys(["S4_earnings_gate_on_hold_or_add"], {"buy_only_if": ([], [EARNINGS]),
                                                          "add_only_if": ([], [EARNINGS])}),
    "S9_weak_liquidity": {"buy_only_if": ([], [LIQUIDITY]), "add_only_if": ([], [LIQUIDITY])},
    "C2_headwind_earnings_gate": {"buy_only_if": (PE_HEADWIND_OLD, [EARNINGS]), "add_only_if": ([], [EARNINGS])},
    **dict.fromkeys(["C1_headwind_hold_or_add", "C3_headwind_plain_avoid"], {"buy_only_if": (PE_HEADWIND_OLD, [])}),
    **dict.fromkeys(["A_broken_bearish", "B_broken_bearish_knife"],
                    {"buy_only_if": (["net_margin > 0%", "debt_to_equity < 1.5x", "revenue_yoy > -20%"], [])}),
    "E_poor_business_bullish_chart": {"buy_only_if": (["net_margin > 0%", "debt_to_equity < 1.5x"],
                                                      [THESIS_CLEARS_WHEN["business_broken"]])},
    "S8_fundamentals_failed": {"buy_only_if": (["revenue data available for valuation"], [])},
    "S6_knife_dislocation": {"add_only_if": (["risk_regime <= medium (volatility < 40%)"], [])},
    "T1_extreme_risk_bullish": {"add_only_if": (["risk_regime <= high (volatility < 60%)"], [])},
}


def _wp23_cells(name: str) -> dict[str, Any]:
    row, entry = BEFORE[name], ENTRY_ROWS.get(name)
    hidden = () if entry == "buy" else STARTER_HIDES if entry == "starter" else SIZING_KEYS
    return {
        **{f"horizon_fit.{h}": (row["horizon_fit"][h], None) for h in ("mid_term", "long_term")},  # §7a: removed
        **{f"card_sizing.{k}": (row["card_sizing"][k], None) for k in hidden if row["card_sizing"][k] is not None},
        **({f"sizing_present.{t}.card": (True, False) for t in dm.TIERS} if entry is None else {}),
        **({f"sizing_present.{t}.range": (True, False) for t in ("standard", "full")} if entry != "buy" else {}),
        **({"modes.balanced.action": BALANCED_CHANGES[name]} if name in BALANCED_CHANGES else {}),
        **({"modes.balanced.long_term_view": LONG_VIEW_CHANGES[name]} if name in LONG_VIEW_CHANGES else {}),
        **({"modes.speculative.action": SPECULATIVE_CHANGES[name]} if name in SPECULATIVE_CHANGES else {}),
        **{f"dislocation.{k}": v for k, v in DISLOCATION_CHANGES.get(name, {}).items()},
        **{f"dip.{k}": v for k, v in DIP_CHANGES.get(name, {}).items()},
        **{f"conditions.{k}": v for k, v in CONDITION_CHANGES.get(name, {}).items()},
    }


# WP2.4: snapshot 3.0.0 retires the 2.0.0 decision fields (plan §8); after.json's snapshot block carries the 3.0.0
# fields instead (DM-RF1), so every retired cell ends at None.
RETIRED_SNAPSHOT_FIELDS = ("action_mid_term", "action_long_term", "decision_action_now", "zone")


def _wp24_cells(name: str) -> dict[str, Any]:
    snapshot = BEFORE[name]["snapshot"]
    return {f"snapshot.{k}": (snapshot[k], None) for k in RETIRED_SNAPSHOT_FIELDS if snapshot[k] is not None}


ROW_CHANGES: dict[str, dict[str, Any]] = {
    name: {**WP22_CHANGES.get(name, {}), **_wp23_cells(name), **_wp24_cells(name)} for name in BEFORE}
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
    monkeypatch.setattr(dm, "SNAPSHOT_VERSION", "2.0.0")  # the recorded rows are 2.0.0; the code is 3.0.0 (DM-G4)
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

    async def test_the_recorded_snapshots_keep_their_retained_fields(self) -> None:
        """DM-G10 (replaces DM-CH2, retired by WP2.4): each baseline scenario's 3.0.0 snapshot has the 3.0.0 field
        set, and every field it keeps from 2.0.0 equals the recorded fixture except hard_gate_ids on the WP2.2
        thesis rows (GATE_CHANGES). G from WP2.4; the 2.0.0 fixtures stay as recorded (DM-G6)."""
        _, snapshots = await dm.build()
        recorded = {p.stem: json.loads(p.read_text()) for p in dm.SNAPSHOT_DIR.glob("*.json")}
        retained = (SNAPSHOT_FIELDS["2.0.0"] & SNAPSHOT_FIELDS["3.0.0"]) - {"snapshot_version", "snapshot_hash"}

        assert set(recorded) == set(BEFORE) <= set(snapshots)
        for name, old in recorded.items():
            new = _normalized(snapshots[name])
            changed = {k: (old[k], new[k]) for k in retained if new[k] != old[k]}
            assert set(new) == SNAPSHOT_FIELDS["3.0.0"], name
            assert changed == ({"hard_gate_ids": GATE_CHANGES[name]} if name in GATE_CHANGES else {}), name

    def test_the_after_table_snapshot_follows_the_decision(self) -> None:
        """DM-RF1 (SN-R8 / OC-RF12, matrix clause): in after.json every row's snapshot block is its decision:
        new_money, if_held and long-term actions, the sorted fired gates and price_zone (plan §8). RF (WP2.4)."""
        for name, row in AFTER.items():
            decision = row["decision"]
            assert row["snapshot"] | {"snapshot_hash": None} == {
                "new_money_action": decision["new_money"]["action"], "if_held_action": decision["if_held"]["action"],
                "long_term_action": decision["long_term"]["action"], "hard_gate_ids": sorted(row["hard_gates"]),
                "price_zone": row["price_zone"], "snapshot_hash": None}, name

    def test_the_after_table_changes_only_the_listed_cells(self) -> None:
        """DM-CH3: after.json differs from before.json, over before.json's paths, exactly in ROW_CHANGES;
        the after-only rows are the listed ones. CH: each WP that re-records after.json extends ROW_CHANGES."""
        delta: dict[str, dict[str, Any]] = {}
        for name, row in BEFORE.items():
            old, new = _flat(row), _flat(AFTER[name])
            cells = {k: (old[k], new.get(k)) for k in old if k != "snapshot.snapshot_hash" and new.get(k) != old[k]}
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
        assert set(snapshot) == SNAPSHOT_FIELDS["2.0.0"]  # the frozen F20 literal is the genuine 0609fde field set
        assert validate_previous_snapshot(snapshot, "ACME") == []

    def test_the_generator_refuses_to_record_from_another_snapshot_version(self, tmp_path: Path) -> None:
        """DM-G4: snapshots_2_0_0/ can only be written by 2.0.0 code; from WP2.4 the code is 3.0.0, so --baseline
        always refuses before building anything. Holds throughout Step 2."""
        assert dm.SNAPSHOT_VERSION == "3.0.0"
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
