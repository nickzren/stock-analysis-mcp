"""WP2.0 snapshot characterization (annex step2-v2-research/snapshot.md §8, tests SN-C1 to SN-C9).

Today's watchlist_snapshot and detect_changes rules on the WP2.0 decision matrix
(tests/support/decision_matrix.py, clock pinned). They are written version-agnostic so they
stay green through 2a: each previous snapshot is the live producer's own (or a frozen 1.x
producer's), never a version literal.

Ported annex probes: probe_rules.py §1 (SN-C4), §5 and §6
(SN-C6); probe_profile.py and probe_profile_verify.py (SN-C5); probe_orig.py §5 (SN-C2).
"""

import copy
import importlib
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import pytest

from stock_analysis.tools.analyze.gates import ANALYZE_GATE_IDS as GATE_IDS
from stock_analysis.utils import normalize
from stock_analysis.utils.normalize import SNAPSHOT_IDENTITY_KEYS, build_watchlist_snapshot
from tests.support import decision_matrix as dm
from tests.support.legacy_snapshot import legacy_compact_snapshot, legacy_enriched_snapshot

diff_module = importlib.import_module("stock_analysis.tools.diff_analysis")

Detect = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]

CLEAN = "S2_buy_clean"  # no gate, action buy, every diffed field known


def edited(result: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """A copy of an analyze result with ``a__b`` paths set (a callable maps the old value),
    and its snapshot rebuilt by the live producer."""
    raw = copy.deepcopy(result)
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = raw
        for key in parents:
            node = node[key]
        node[leaf] = value(node.get(leaf)) if callable(value) else value
    raw["watchlist_snapshot"] = build_watchlist_snapshot(raw)
    return raw


def at(result: dict[str, Any], path: str) -> Any:
    for key in path.split("__"):
        result = result[key]
    return result


def firing(gate_id: str) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Adds a gate to the blocking list, keeping any gate that already fired."""
    return lambda gates: [*gates, {"id": gate_id, "reason": "fired"}]


@pytest.fixture
def detect(monkeypatch: pytest.MonkeyPatch) -> Detect:
    """detect_changes against ``previous``, with analyze returning ``current`` (a result the
    caller analyzed with the clock pinned; what_changed itself would read the wall clock)."""

    async def run(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        async def analyze(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return current

        monkeypatch.setattr(diff_module, "analyze_stock", analyze)
        response: dict[str, Any] = await diff_module.what_changed(
            "ACME", previous_snapshot=copy.deepcopy(previous))
        return response

    return run


# --- SN-C1, SN-C2: gate ids ------------------------------------------------------------


async def test_every_gate_analyze_evaluates_has_a_case() -> None:
    """SN-C1: guard, set(GATE_IDS) == the gates analyze evaluates (decision_card.hard_gates.checks).
    G from WP2.1: set(GATE_IDS) == checks; WP2.2 adds thesis_broken and near_52w_high by deleting
    REGISTERED_NOT_EVALUATED."""
    checks = (await dm.run(CLEAN))["decision_card"]["hard_gates"]["checks"]
    assert set(GATE_IDS) == set(checks)


@pytest.mark.parametrize("gate_id", GATE_IDS)
async def test_every_gate_firing_or_clearing_is_material_and_moves_the_hash(
    detect: Detect, gate_id: str,
) -> None:
    """SN-C1: same-version, a gate firing gives +hard_gate: id and clearing gives -hard_gate: id.
    G: the snapshot keeps reading decision_card.hard_gates.blocking through 2a (plan §2)."""
    clean = await dm.run(CLEAN)
    gated = edited(clean, decision_card__hard_gates__blocking=firing(gate_id))

    fired = await detect(clean["watchlist_snapshot"], gated)
    cleared = await detect(gated["watchlist_snapshot"], clean)

    assert (fired["material_changes"], fired["hash_changed"]) == ([f"+hard_gate: {gate_id}"], True)
    assert (cleared["material_changes"], cleared["hash_changed"]) == ([f"-hard_gate: {gate_id}"], True)


async def test_a_gate_on_a_non_new_money_candidate_is_one_gate_line(detect: Detect) -> None:
    """SN-C2: a knife firing on an avoid candidate, action unchanged, gives only the gate line.
    G: hard_gate_ids lists every fired gate, not only those that changed the action (annex §2)."""
    avoid = await dm.run("S3_plain_avoid")
    knife = edited(avoid, decision_card__hard_gates__blocking=firing("falling_knife"))
    assert avoid["decision_card"]["action_now"] == "avoid"  # not a new-money candidate
    assert "falling_knife" not in avoid["watchlist_snapshot"]["hard_gate_ids"]

    response = await detect(avoid["watchlist_snapshot"], knife)

    assert (response["material_changes"], response["hash_changed"]) == (["+hard_gate: falling_knife"], True)


@pytest.mark.parametrize("name", ["S6_knife_dislocation", "C_quality_knife_dislocation"])
async def test_the_snapshot_lists_a_knife_on_an_avoid_candidate(name: str) -> None:
    """SN-C2: the producer's own avoid-candidate knife rows (before table) carry falling_knife.
    G: hard_gate_ids is every fired gate whatever the candidate, through 2a (annex §2)."""
    result = await dm.run(name)
    assert "falling_knife" in result["watchlist_snapshot"]["hard_gate_ids"]


# --- SN-C3, SN-C4: diffed and hashed fields ----------------------------------------------

# One raw change per diffed field of CLEAN, with the line it must produce.
MUTATIONS: dict[str, tuple[dict[str, Any], str]] = {
    "price": ({"summary__current_price": 95.23}, "price +7.0%"),
    "score": ({"verdict__score": 0.5}, "score +0.20"),
    "hard_gate_ids": ({"decision_card__hard_gates__blocking": firing("falling_knife")}, "+hard_gate: falling_knife"),
    "mid_term_fit": ({"verdict__horizon_fit__mid_term": "poor"}, "mid_term_fit: ok -> poor"),
    "long_term_fit": ({"verdict__horizon_fit__long_term": "avoid"}, "long_term_fit: ok -> avoid"),
    "bullish_signals": ({"signals__bullish": lambda old: [*old, "insider_buying"]}, "+bullish: insider_buying"),
    "bearish_signals": ({"signals__bearish": ["high_put_call_ratio"]}, "+bearish: high_put_call_ratio"),
    "tilt": ({"verdict__tilt": "neutral"}, "tilt: bullish -> neutral"),
    "confidence": ({"verdict__confidence": "medium"}, "confidence: low -> medium"),
    "risk_regime": ({"risk_summary__risk_regime__classification": "high"}, "risk_regime: medium -> high"),
    "valuation_gate": ({"policy_action__valuation_gate": "stretched"}, "valuation_gate: attractive -> stretched"),
    "support_instrument": ({"summary__support__instrument__status": "unverified"},
                           "support_instrument: supported -> unverified"),
    "support_currency": ({"summary__support__currency__status": "unverified"},
                         "support_currency: supported -> unverified"),
    "price_source": ({"summary__price_provenance__price_source": "info_quote_fallback"},
                     "price_source: daily_close -> info_quote_fallback"),
    # Action and zone rows: WP2.4 swaps them for new_money/if_held/long_term_action and price_zone.
    "zone": ({"action_zones__current_zone": "hold_neutral"}, "zone: accumulate -> hold_neutral"),
    "decision_action_now": ({"decision_card__action_now": "wait"}, "action_now: buy -> wait"),
    "action_mid_term": ({"policy_action__mid_term": "wait_for_entry"}, "action_mid_term: buy -> wait_for_entry"),
    "action_long_term": ({"policy_action__long_term": "hold"}, "action_long_term: accumulate -> hold"),
}


def test_every_diffed_field_has_a_mutation_case() -> None:
    """SN-C3: guard, set(MUTATIONS) == set(_DIFFED_FIELDS).
    CH, flipped by WP2.4: the snapshot 3.0.0 field set changes _DIFFED_FIELDS (plan §8)."""
    assert set(MUTATIONS) == set(diff_module._DIFFED_FIELDS)


@pytest.mark.parametrize("field", MUTATIONS)
async def test_each_diffed_field_change_through_the_raw_is_material_and_moves_the_hash(
    detect: Detect, field: str,
) -> None:
    """SN-C3: each mutation, made in the analyze result, is one material line and moves the hash.
    CH, flipped by WP2.3 (policy_action and zone raw paths move, plan §0 #22) and WP2.4 (rows)."""
    changes, line = MUTATIONS[field]
    clean = await dm.run(CLEAN)

    response = await detect(clean["watchlist_snapshot"], edited(clean, **changes))

    assert (response["material_changes"], response["hash_changed"]) == ([line], True)


async def test_every_compared_field_is_hashed_and_every_field_is_type_checked() -> None:
    """SN-C4: on the producer's keys, diffed ⊆ hashed and every non-identity key is in _FIELD_TYPES.
    G: version-free invariants over the live constants; 3.0.0 keeps both (annex §2, §3)."""
    keys = set((await dm.run(CLEAN))["watchlist_snapshot"])

    assert set(diff_module._DIFFED_FIELDS) <= keys - normalize._UNHASHED_SNAPSHOT_KEYS
    assert keys - set(SNAPSHOT_IDENTITY_KEYS) <= set(diff_module._FIELD_TYPES)


# --- SN-C5, SN-C6: what the snapshot ignores -------------------------------------------

PROFILE_INPUTS: list[dict[str, Any]] = [
    {},  # what detect_changes runs (diff_analysis.py:369)
    {"profile": "core"},
    {"profile": "speculative"},
    {"profile": "balanced", "account_size": 5000, "risk_per_trade_pct": 2.0, "max_position_pct": 20.0},
    {"profile": "speculative", "account_size": 1_000_000, "max_position_pct": 1.0},
]


@pytest.mark.parametrize("name", ["S4_earnings_gate_on_hold_or_add", CLEAN, "ETF_strong", "S9_weak_liquidity",
                                  "C_quality_knife_dislocation"])
async def test_the_snapshot_is_the_same_for_every_profile_and_account(name: str) -> None:
    """SN-C5: analyze with core/balanced/speculative and account inputs gives one snapshot.
    G: detect_changes always runs the default profile, so the snapshot never depends on one."""
    scenario = dm.SCENARIOS[name]
    snapshots = []
    for inputs in PROFILE_INPUTS:
        with dm.stubbed(scenario):
            result = await dm.orchestrator.analyze_stock("ACME", _now=scenario.now, **inputs)
        snapshots.append(result["watchlist_snapshot"])

    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:])


@pytest.mark.parametrize("name", list(dm.SCENARIOS))
async def test_the_snapshot_and_its_hash_are_deterministic(name: str) -> None:
    """SN-C6: two analyze runs, and two builds from one result, give the same snapshot and hash.
    G: the hash is a pure function of the data in every version (annex R2.11e)."""
    first, second = await dm.run(name), await dm.run(name)
    rebuilt = build_watchlist_snapshot(copy.deepcopy(first))

    assert first["watchlist_snapshot"] == second["watchlist_snapshot"]
    assert rebuilt["snapshot_hash"] == first["watchlist_snapshot"]["snapshot_hash"]


async def test_gate_reason_text_is_not_hashed(detect: Detect) -> None:
    """SN-C6: only gate ids are hashed; the earnings reason carries the day count (gates.py:30).
    G: reasons stay out of the hash in 3.0.0 (plan §8)."""
    gated = await dm.run("S4_earnings_gate_on_hold_or_add")
    reason = "earnings in 2 days (<= 5)"
    reworded = edited(gated, decision_card__hard_gates__blocking=lambda gates: [
        {**gate, "reason": reason} if gate["id"] == "earnings_blackout" else gate for gate in gates])
    reasons = {gate["id"]: gate["reason"] for gate in gated["decision_card"]["hard_gates"]["blocking"]}
    assert reasons["earnings_blackout"] != reason

    response = await detect(gated["watchlist_snapshot"], reworded)

    assert (response["material_changes"], response["hash_changed"]) == ([], False)


SUNDAY_NOON = dm.ET.localize(datetime(2026, 10, 11, 12, 0))  # same last completed session (Fri 10-09)
WINDOW = {"source": "calendar", "start": "2026-11-16", "end": "2026-11-20", "window_basis": "assumed"}


def in_window(date: str) -> dict[str, Any]:
    """An earnings date inside a window: the date is the clock, the window bounds are the data."""
    return {"earnings": {"next_date": date, "next_date_status": "available", "next_date_source": "calendar",
                         "days_until": 40, "next_window": WINDOW}}


CACHED = {**dm.at_price(dm.STRONG, 89.0),
          "session_close": {"close": 89.0, "session_date": "2026-10-09", "fetched_at": "2026-10-10T00:00:00Z",
                            "cache_hit": True}}
# (scenario, previous overrides, current overrides, the result paths that move): only the clock
# or the fetch moves. The whole results always differ (meta.duration_ms, tool_timings).
CLOCK_ONLY: dict[str, tuple[str, dict[str, Any], dict[str, Any], tuple[str, ...]]] = {
    "fetched_later": (CLEAN, {}, {"now": SUNDAY_NOON},
                      ("summary__price_provenance__fetched_at", "watchlist_snapshot__snapshot_as_of_date")),
    "cache_hit": (CLEAN, {}, {"technicals": CACHED}, ("summary__price_provenance__cache_hit",)),
    "days_until": (CLEAN, {}, {"events": {"earnings": {**dm.FAR["earnings"], "days_until": 39}}},
                   ("events_summary__next_catalyst__days_until",)),
    "days_until_in_the_blackout": ("S4_earnings_gate_on_hold_or_add", {},
                                   {"events": {"earnings": {**dm.NEAR["earnings"], "days_until": 2}}},
                                   ("events_summary__next_catalyst__days_until", "decision_card__hard_gates__blocking")),
    "window_date": (CLEAN, {"events": in_window("2026-11-18")}, {"events": in_window("2026-11-19")},
                    ("events_summary__next_catalyst__date",)),
}


@pytest.mark.parametrize("case", CLOCK_ONLY)
async def test_a_clock_only_move_gives_no_alert_and_keeps_the_hash(detect: Detect, case: str) -> None:
    """SN-C6: fetched_at, fetch date and age, cache_hit, days_until and the window date (price_stale: SN-C9).
    G: the clock stays out of the snapshot in 3.0.0 (plan §8)."""
    name, before, after, moved = CLOCK_ONLY[case]
    previous, current = await dm.run(name, **before), await dm.run(name, **after)
    for path in moved:
        assert at(previous, path) != at(current, path), path  # the analyze result did move

    response = await detect(previous["watchlist_snapshot"], current)

    assert (response["material_changes"], response["hash_changed"]) == ([], False)


async def test_a_quote_staleness_age_gives_no_alert_and_keeps_the_hash(detect: Detect) -> None:
    """SN-C6: an hours-unit staleness age (quote_staleness, price_basis.py:117) is clock only.
    G: ages stay out of the snapshot in 3.0.0 (plan §8)."""
    clean = await dm.run(CLEAN)

    def quote_aged(hours: float) -> dict[str, Any]:
        return edited(clean, summary__price_provenance__staleness={"unit": "hours", "age_hours": hours, "stale": False})

    response = await detect(quote_aged(0.25)["watchlist_snapshot"], quote_aged(1.75))

    assert (response["material_changes"], response["hash_changed"]) == ([], False)


# --- SN-C7: 1.x previous snapshots -------------------------------------------------------


def as_1_0_0(snapshot: dict[str, Any]) -> dict[str, Any]:
    """1.0.0 = 1.1.0 without decision_action_now and hard_gate_ids (commit 146c063)."""
    older = {k: v for k, v in snapshot.items() if k not in ("decision_action_now", "hard_gate_ids")}
    return {**older, "snapshot_version": "1.0.0"}


# The frozen producers read the 0609fde raw shape (top-level policy_action). WP2.3 moves
# policy_action (plan §0 #22), so from then on they need a 0609fde-shaped view of the result,
# or valuation_gate and the 1.x action fields read null: harness drift, not a behavior change.
LEGACY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "1.0.0_compact": lambda raw: as_1_0_0(legacy_compact_snapshot(raw)),
    "1.0.0_enriched": lambda raw: as_1_0_0(legacy_enriched_snapshot(raw)),
    "1.1.0_compact": legacy_compact_snapshot,
    "1.1.0_enriched": legacy_enriched_snapshot,
}


@pytest.mark.parametrize("name", list(dm.SCENARIOS))
@pytest.mark.parametrize("kind", LEGACY)
async def test_an_old_snapshot_of_the_same_data_has_no_phantom_change(
    detect: Detect, kind: str, name: str,
) -> None:
    """SN-C7: a 1.x snapshot of the same result gives no material change and no hash comparison.
    G: 1.x stays accepted with no phantom changes in 3.0.0 (annex R2.11c)."""
    result = await dm.run(name)

    response = await detect(LEGACY[kind](result), result)

    assert (response["material_changes"], response["hash_changed"]) == ([], None)
    if kind.startswith("1.0.0"):
        assert "hard_gate_ids" in response["not_comparable"]


@pytest.mark.parametrize("kind", LEGACY)
async def test_old_signal_lists_are_never_compared(detect: Detect, kind: str) -> None:
    """SN-C7: 1.x signal lists were ambiguous (a failed tool read []), so they are not compared.
    G: 1.x signals are never compared in 3.0.0 either (annex §4, UNRELIABLE)."""
    clean = await dm.run(CLEAN)
    moved = edited(clean, signals__bullish=lambda old: [*old, "insider_buying"],
                   signals__bearish=["high_put_call_ratio"])

    response = await detect(LEGACY[kind](clean), moved)

    assert response["material_changes"] == []
    assert {"bullish_signals", "bearish_signals"} <= set(response["not_comparable"])


OLD_SNAPSHOT_ALERTS: dict[str, tuple[dict[str, Any], str]] = {
    "gate": MUTATIONS["hard_gate_ids"],
    "tilt": MUTATIONS["tilt"],
    "price": MUTATIONS["price"],
    "score": MUTATIONS["score"],
}


@pytest.mark.parametrize(
    ("kind", "change"),
    [(kind, change) for kind in LEGACY for change in OLD_SNAPSHOT_ALERTS
     if not (change == "gate" and kind.startswith("1.0.0"))],  # 1.0.0 has no gate ids
)
async def test_a_real_change_still_alerts_against_an_old_snapshot(detect: Detect, kind: str, change: str) -> None:
    """SN-C7: a gate, tilt, price or score change alerts against a 1.x snapshot (the gate half of
    test_real_changes_still_show_against_an_old_snapshot). G: these fields stay compared (annex §4)."""
    changes, line = OLD_SNAPSHOT_ALERTS[change]
    clean = await dm.run(CLEAN)

    response = await detect(LEGACY[kind](clean), edited(clean, **changes))

    assert response["material_changes"] == [line]


# --- SN-C8, SN-C9 ------------------------------------------------------------------------


async def test_a_known_signal_list_that_becomes_unknown_is_one_change(detect: Detect) -> None:
    """SN-C8: same-version, options_signals failing makes both lists null: one line each, no ± items.
    G: the null-list rule is unchanged in 3.0.0 (annex §2, all other F20 fields)."""
    known = await dm.run("E2_mixed_fcf_bullish_chart")
    unknown = await dm.run("E2_mixed_fcf_bullish_chart", options=None)
    previous = known["watchlist_snapshot"]
    assert previous["bullish_signals"] and previous["bearish_signals"]  # items that could read as removed

    response = await detect(previous, unknown)

    signal_lines = [line for line in response["material_changes"] if line.lstrip("+-").startswith(("bullish", "bearish"))]
    assert signal_lines == [f"bullish_signals: {previous['bullish_signals']} -> None",
                            f"bearish_signals: {previous['bearish_signals']} -> None"]


@pytest.mark.parametrize("stale", [True, False])
async def test_price_stale_flipping_alone_is_not_an_alert(detect: Detect, stale: bool) -> None:
    """SN-C9: the price_stale field (an age past a threshold) is kept but neither hashed nor compared.
    G: 3.0.0 keeps price_stale out of the hash and the diff (annex §2); a freshness gate fired from
    WP2.1 alerts through hard_gate_ids instead (plan §0 #5, D-8; SN-R21)."""
    clean = await dm.run(CLEAN)
    previous = edited(clean, summary__price_provenance__staleness__stale=not stale)["watchlist_snapshot"]
    current = edited(clean, summary__price_provenance__staleness__stale=stale)
    assert (previous["price_stale"], current["watchlist_snapshot"]["price_stale"]) == (not stale, stale)

    response = await detect(previous, current)

    assert (response["material_changes"], response["hash_changed"]) == ([], False)
