"""WP2.4: watchlist_snapshot 3.0.0 and detect_changes (plan §8 [FIX], D-8, D-10; §0 #4, #5, #10, #18, #22).

The snapshot records the canonical decision (new_money_action, if_held_action, long_term_action), every fired gate
(hard_gate_ids) and the neutral price_zone; it drops the pre-gate codes, decision_action_now and zone.
detect_changes keeps accepting 1.0.0, 1.1.0 and 2.0.0 snapshots, validates and compares each on its own version's
fields, requires a 3.0.0 previous to be complete, and discloses the fields it could not compare.

Annex: docs/plans/step2-v2-research/snapshot.md "Corrected proposal" §1-§8 (tests R1-R22), as amended by plan §0;
reassessment.md RA.16; freshness.md R-F15 (inverted by D-8). The previous 2.0.0 snapshots are the WP2.0 recordings
in tests/fixtures/decision_matrix/snapshots_2_0_0/ (frozen, DM-G6).
"""

import copy
import importlib
import json
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import pytest

from stock_analysis.server import detect_changes
from stock_analysis.tools.analyze import decision
from stock_analysis.tools.analyze.decision import IF_HELD_ACTIONS, NEW_MONEY_ACTIONS
from stock_analysis.tools.analyze.gates import ANALYZE_GATE_IDS
from stock_analysis.utils.normalize import (
    _UNHASHED_SNAPSHOT_KEYS,
    COMPATIBLE_SNAPSHOT_VERSIONS,
    SNAPSHOT_FIELDS,
    SNAPSHOT_VERSION,
)
from tests.support import decision_matrix as dm
from tests.support import freshness_frames as fr
from tests.support import output_contract as oc
from tests.support.legacy_snapshot import (
    as_0609fde,
    legacy_compact_snapshot,
    legacy_enriched_snapshot,
)
from tests.test_step2.test_characterize_snapshot import CLEAN, MUTATIONS, as_1_0_0, edited
from tests.test_step2.test_decision_matrix import ACTION_CHANGES, GATE_CHANGES
from tests.test_tools.test_snapshot_consistency import raw_result

diff_module = importlib.import_module("stock_analysis.tools.diff_analysis")
decision_card_module = importlib.import_module("stock_analysis.tools.analyze.decision_card")

Detect = Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]

ACTIONS = ("new_money_action", "if_held_action", "long_term_action")
V3_ONLY = ["if_held_action", "long_term_action", "new_money_action", "price_zone"]
RETIRED_2_0_0 = ["action_long_term", "action_mid_term", "decision_action_now", "zone"]
# Frozen 3.0.0 vocabularies. Do not update: a vocabulary change bumps SNAPSHOT_VERSION (annex SN §3, R5).
NEW_MONEY_3_0_0 = frozenset({"buy", "starter", "wait", "wait_for_data", "insufficient_data", "watch", "avoid"})
IF_HELD_3_0_0 = frozenset({"hold", "reduce", "review_exit", "insufficient_data"})
# The gate ids 0609fde's decision card evaluated (decision_card.py:61-77 there); every other analyze gate is new.
GATES_2_0_0 = frozenset({"data_quality_critical", "missing_runway", "liquidity_missing", "falling_knife",
                         "weak_liquidity", "unsupported_instrument", "unsupported_currency", "earnings_blackout"})
RULE_NOTE = ("3.0.0 rule changes (a gate change may come from these, not from the data): new analyze gates "
             "earnings_unverifiable, stale_data, freshness_unverifiable, thesis_broken and near_52w_high; "
             "falling_knife and weak_liquidity give avoid, earnings gates give wait, liquidity_missing is a data "
             "gate; falling_knife is evaluated for every instrument, unsupported ones included; a missing beta "
             "alone no longer fires data_quality_critical; zone is replaced by price_zone")
GOLDEN = {"3.0.0": "e61598dde22138de"}  # raw_result() in test_snapshot_consistency


def version_warning(previous: str, retired: list[str]) -> str:
    """The one version warning (plan §8, annex SN §4): the retired fields, then the 3.0.0 rule note."""
    return (f"snapshot_version changed ({previous} -> 3.0.0); field semantics may differ; retired, not compared: "
            f"{', '.join(retired)}; {RULE_NOTE}")


def recorded(name: str) -> dict[str, Any]:
    """The WP2.0 recording of a baseline scenario: a genuine 0609fde 2.0.0 snapshot."""
    snapshot: dict[str, Any] = json.loads((dm.SNAPSHOT_DIR / f"{name}.json").read_text())
    return snapshot


@pytest.fixture
def detect(monkeypatch: pytest.MonkeyPatch) -> Detect:
    """detect_changes against ``previous`` with analyze returning ``current``; the response records whether
    analyze ran (``_analyzed``)."""

    async def run(previous: Any, current: dict[str, Any]) -> dict[str, Any]:
        calls: list[str] = []

        async def analyze(symbol: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(symbol)
            return current

        monkeypatch.setattr(diff_module, "analyze_stock", analyze)
        response: dict[str, Any] = await diff_module.what_changed("ACME", previous_snapshot=copy.deepcopy(previous))
        response["_analyzed"] = calls
        return response

    return run


_CLEAN: dict[str, Any] = {}


async def clean() -> dict[str, Any]:
    """S2 (buy, no gate), analyzed once per session."""
    if not _CLEAN:
        _CLEAN["result"] = await dm.run(CLEAN)
    return copy.deepcopy(_CLEAN["result"])


# --- Versions (SN-R14, R18) -------------------------------------------------------------------------------------


def test_the_accepted_versions_are_a_literal() -> None:
    """SN-R14: the accepted set is the literal {1.0.0, 1.1.0, 2.0.0, 3.0.0}, one frozen field set per version, and
    the producer's version is 3.0.0 (a plain bump would have rejected 2.0.0, normalize.py:34). RF."""
    assert SNAPSHOT_VERSION == "3.0.0"
    assert COMPATIBLE_SNAPSHOT_VERSIONS == {"1.0.0", "1.1.0", "2.0.0", "3.0.0"} == set(SNAPSHOT_FIELDS)
    assert [len(SNAPSHOT_FIELDS[v]) for v in ("1.0.0", "1.1.0", "2.0.0", "3.0.0")] == [22, 24, 30, 30]
    assert SNAPSHOT_FIELDS["3.0.0"] == (SNAPSHOT_FIELDS["2.0.0"] - set(RETIRED_2_0_0)) | set(V3_ONLY)
    assert set().union(*SNAPSHOT_FIELDS.values()) - {"snapshot_version", "symbol"} == set(diff_module._FIELD_TYPES)


@pytest.mark.parametrize("kind", ["1.0.0", "1.1.0", "2.0.0", "3.0.0"])
async def test_hashes_compare_only_within_3_0_0(detect: Detect, kind: str) -> None:
    """SN-R18: hash_changed is None against a 1.0.0, 1.1.0 or 2.0.0 previous and a bool against a 3.0.0 one.
    RF (the 3.0.0 case; the older-version cases are G, mutation-checked by m0)."""
    result = await clean()
    previous = {"1.0.0": as_1_0_0(legacy_compact_snapshot(as_0609fde(result))),
                "1.1.0": legacy_compact_snapshot(as_0609fde(result)), "2.0.0": recorded(CLEAN),
                "3.0.0": result["watchlist_snapshot"]}[kind]

    response = await detect(previous, result)

    assert response["hash_changed"] is (False if kind == "3.0.0" else None)


# --- The producer (SN-R8, R20, OC-RF12) -------------------------------------------------------------------------


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_the_snapshot_records_the_decision(name: str) -> None:
    """SN-R8 and OC-RF12 (amended, plan §0 #10): every scenario's snapshot has exactly the 3.0.0 fields; its
    actions are new_money, if_held and long_term.new_money (plan §0 #18), hard_gate_ids is every fired gate
    (decision_card.hard_gates.blocking, never null when the card is built), price_zone is action_zones.price_zone;
    no action_mid_term, action_long_term, decision_action_now, zone or blocked_by key. RF."""
    result = await dm.run(name)
    card, snapshot = result["decision_card"], result["watchlist_snapshot"]

    assert set(snapshot) == SNAPSHOT_FIELDS["3.0.0"]
    assert [snapshot[f] for f in ACTIONS] == [card["new_money"]["action"], card["if_held"]["action"],
                                               card["long_term"]["new_money"]["action"]]
    assert snapshot["hard_gate_ids"] == sorted(g["id"] for g in card["hard_gates"]["blocking"])
    assert snapshot["price_zone"] == result["action_zones"]["price_zone"]
    assert not {*RETIRED_2_0_0, "blocked_by"} & set(snapshot)


async def test_a_gated_hold_or_add_records_no_pre_gate_code() -> None:
    """SN-R8: S4 (earnings in 3 days on a hold_or_add candidate) records new money wait, never hold_or_add.
    RF."""
    result = await dm.run("S4_earnings_gate_on_hold_or_add")
    snapshot = result["watchlist_snapshot"]

    assert result["diagnostics"]["pre_gate_policy"]["mid_term"] == "hold_or_add"
    assert [snapshot[f] for f in ACTIONS] == ["wait", "hold", "wait"]
    assert "hold_or_add" not in snapshot.values()


async def test_the_zone_sentence_reads_price_zone(detect: Detect) -> None:
    """OC-RF12 (amended): 3 -> 3, a price_zone move is one line and the summary's zone sentence names price_zone;
    the capped pre-gate zone moving alone is not a change (SN-R19). RF."""
    result = await clean()

    response = await detect(result["watchlist_snapshot"], edited(result, action_zones__price_zone="near_52w_high"))

    assert response["material_changes"] == ["price_zone: below_sma200 -> near_52w_high"]
    assert response["summary"] == "ACME: Price zone moved from below_sma200 to near_52w_high."


def test_the_hash_is_pinned_per_version() -> None:
    """SN-R20: a fixed raw gives the literal 3.0.0 field set and GOLDEN[SNAPSHOT_VERSION]; a scope change
    without a version bump fails here. RF."""
    snapshot = raw_result()["watchlist_snapshot"]

    assert set(snapshot) == SNAPSHOT_FIELDS["3.0.0"]
    assert snapshot["snapshot_hash"] == GOLDEN[SNAPSHOT_VERSION]


# --- 3 -> 3 (SN-R1, R2, R4, R6, R7, R19) ------------------------------------------------------------------------

ACTION_PATHS = {
    "new_money_action": ("decision_card__new_money__action", "New money", NEW_MONEY_ACTIONS),
    "if_held_action": ("decision_card__if_held__action", "If held", IF_HELD_ACTIONS),
    "long_term_action": ("decision_card__long_term__new_money__action", "Long term", NEW_MONEY_ACTIONS),
}
TRANSITIONS = [(field, a, b) for field, (_, _, vocabulary) in ACTION_PATHS.items()
               for a in (*vocabulary, None) for b in (*vocabulary, None) if a != b]


@pytest.mark.parametrize(("field", "before", "after"), TRANSITIONS)
async def test_every_action_transition_is_material(detect: Detect, field: str, before: str | None,
                                                   after: str | None) -> None:
    """SN-R1, R2: 3 -> 3, every ordered transition of each action field over its vocabulary and None (56 + 20 +
    56) gives exactly its line, its summary sentence and hash_changed True. RF."""
    path, label, _ = ACTION_PATHS[field]
    result = await clean()

    response = await detect(edited(result, **{path: before})["watchlist_snapshot"], edited(result, **{path: after}))

    assert (response["material_changes"], response["hash_changed"]) == ([f"{field}: {before} -> {after}"], True)
    assert f"{label}: {before} -> {after}." in response["summary"]


@pytest.mark.parametrize("field", ACTIONS)
async def test_a_null_action_is_a_valid_known_null(field: str) -> None:
    """SN-R2 (validation clause): a 3.0.0 snapshot with a null action validates; null is a known value, not a
    missing key. G."""
    snapshot = (await clean())["watchlist_snapshot"]
    assert diff_module.validate_previous_snapshot({**snapshot, field: None}, "ACME") == []


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_the_producer_s_own_snapshot_round_trips(detect: Detect, name: str) -> None:
    """SN-R4, R6 (and R7's never-null clause): the producer's snapshot validates, compares to itself with no
    change, hash_changed False and nothing not_comparable, and every action is in the frozen 3.0.0 vocabulary.
    RF."""
    result = await dm.run(name)
    snapshot = result["watchlist_snapshot"]

    response = await detect(snapshot, result)

    assert diff_module.validate_previous_snapshot(snapshot, "ACME") == []
    assert (response["material_changes"], response["hash_changed"], response["not_comparable"]) == ([], False, [])
    assert response["warnings"] is None and "Not compared" not in response["summary"]
    assert {snapshot["new_money_action"], snapshot["long_term_action"]} <= NEW_MONEY_3_0_0
    assert snapshot["if_held_action"] in IF_HELD_3_0_0
    assert isinstance(snapshot["hard_gate_ids"], list)


def test_the_vocabularies_are_frozen_for_3_0_0() -> None:
    """SN-R5 (guard): decision.py's vocabularies equal the frozen 3.0.0 literals. Do not update them: change
    SNAPSHOT_VERSION instead. G."""
    assert (set(NEW_MONEY_ACTIONS), set(IF_HELD_ACTIONS)) == (NEW_MONEY_3_0_0, IF_HELD_3_0_0)


async def test_an_unknown_action_code_is_accepted_and_compared(detect: Detect) -> None:
    """SN-R5: no runtime vocabulary check: a 3.0.0 previous with an unknown code validates, and its change is an
    ordinary scalar line. RF."""
    result = await clean()
    previous = {**result["watchlist_snapshot"], "new_money_action": "hold_or_add"}

    response = await detect(previous, result)

    assert response["material_changes"] == ["new_money_action: hold_or_add -> buy"]


def knife_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decision_card_module, "is_falling_knife_technicals", lambda technicals: (True, ["stub"]))


GATE_CLASSES: dict[str, tuple[dict[str, Any], list[str]]] = {
    "earnings_blackout": ({"events": dm.NEAR}, ["new_money_action: buy -> wait", "long_term_action: buy -> wait"]),
    "weak_liquidity": ({"risk": dm.THIN}, ["new_money_action: buy -> avoid", "long_term_action: buy -> avoid"]),
    "falling_knife": ({}, ["new_money_action: buy -> avoid", "long_term_action: buy -> avoid"]),
    "unsupported_currency": ({"summary": {**dm.EQUITY, "currency": "EUR"}},
                             ["new_money_action: buy -> watch", "long_term_action: buy -> watch",
                              "support_currency: supported -> unsupported"]),
}


@pytest.mark.parametrize("gate", GATE_CLASSES)
async def test_a_gate_firing_through_analyze_moves_the_recorded_decision(
    detect: Detect, monkeypatch: pytest.MonkeyPatch, gate: str,
) -> None:
    """SN-R7: end to end through analyze_stock (stubbed tools), each gate class firing on the clean buy gives its
    action lines and +hard_gate (the producer reads decision_card, not the candidate); the reverse clears them;
    if_held never moves (R2.6(h)). The freshness class is FR-R-F15. RF."""
    overrides, lines = GATE_CLASSES[gate]
    result = await clean()
    if gate == "falling_knife":
        knife_fires(monkeypatch)
    gated = await dm.run(CLEAN, **overrides)

    fired = await detect(result["watchlist_snapshot"], gated)
    cleared = await detect(gated["watchlist_snapshot"], result)

    assert (fired["material_changes"], fired["hash_changed"]) == ([*lines, f"+hard_gate: {gate}"], True)
    assert cleared["material_changes"] == [*(_reverse(line) for line in lines), f"-hard_gate: {gate}"]


def _reverse(line: str) -> str:
    field, change = line.split(": ")
    before, after = change.split(" -> ")
    return f"{field}: {after} -> {before}"


@pytest.mark.parametrize("name", ["S6_knife_dislocation", "C_quality_knife_dislocation"])
async def test_a_knife_on_an_avoid_candidate_is_recorded(name: str) -> None:
    """SN-R7 (avoid + knife): the knife is in hard_gate_ids although the candidate was already avoid (every fired
    gate, annex §2). RF."""
    snapshot = (await dm.run(name))["watchlist_snapshot"]
    assert (snapshot["new_money_action"], snapshot["hard_gate_ids"]) == ("avoid", ["falling_knife"])


async def test_a_candidate_only_change_keeps_the_hash(detect: Detect) -> None:
    """SN-R19: the pre-gate candidate (diagnostics.pre_gate_policy codes and zone), the decision trace and the
    action_now alias move while new_money, if_held and long_term stay: no alert, same hash. RF."""
    result = await clean()
    moved = edited(result, diagnostics__pre_gate_policy__mid_term="avoid",
                   diagnostics__pre_gate_policy__long_term="avoid_core_position",
                   diagnostics__pre_gate_policy__inputs__zone="reduce",
                   diagnostics__decision_trace=["pre_gate_policy.mid_term=avoid -> buy"],
                   decision_card__action_now="wait")

    response = await detect(result["watchlist_snapshot"], moved)

    assert (response["material_changes"], response["hash_changed"]) == ([], False)


# --- Validation (SN-R3, R13) ------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(SNAPSHOT_FIELDS["3.0.0"]))
async def test_a_3_0_0_snapshot_missing_any_field_is_invalid(detect: Detect, key: str) -> None:
    """SN-R3: a 3.0.0 previous must carry every 3.0.0 field (null allowed): one missing key gives invalid_snapshot
    with exactly that problem, before any analysis. RF."""
    result = await clean()
    previous = {k: v for k, v in result["watchlist_snapshot"].items() if k != key}

    response = await detect(previous, result)

    assert (response["error_type"], response["problems"], response["_analyzed"]) == (
        "invalid_snapshot", [f"missing required key: {key}"], [])


STRAY = {"new_money_action": 5, "if_held_action": ["hold"], "long_term_action": {"a": 1}, "price_zone": True}


async def test_stray_3_0_0_keys_on_a_2_0_0_snapshot_are_ignored(detect: Detect) -> None:
    """SN-R13: 3.0.0 keys of any type on a 2.0.0 previous are neither validated nor compared, and not_comparable
    is unchanged. RF."""
    previous = {**recorded(CLEAN), **STRAY}
    response = await detect(previous, await clean())

    assert diff_module.validate_previous_snapshot(previous, "ACME") == []
    assert (response["material_changes"], response["not_comparable"]) == ([], V3_ONLY)
    assert not set(STRAY) & set(response["changes"])


# --- 2 -> 3 (SN-R9-R12, R15, R22) [D-10] ------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(set(dm.SCENARIOS) & {p.stem for p in dm.SNAPSHOT_DIR.glob("*.json")}))
async def test_the_recorded_2_0_0_snapshots_migrate(detect: Detect, name: str) -> None:
    """SN-R9 (PD D-10) and SN-R22 (amended): against its WP2.0 recording, each baseline scenario reports exactly
    the rule-driven gate lines of the before/after table (+thesis_broken on A, B and E, WP2.2), never an action or
    zone line; the three actions and price_zone are not_comparable and named in the summary; hash_changed is
    None; one version warning names the retired fields and the 3.0.0 rule changes. PD D-10."""
    response = await detect(recorded(name), await dm.run(name))
    before, after = GATE_CHANGES.get(name, ([], []))

    assert response["material_changes"] == [f"+hard_gate: {g}" for g in after if g not in before]
    assert (response["not_comparable"], response["hash_changed"]) == (V3_ONLY, None)
    assert response["summary"].endswith(
        "Not compared with the previous 2.0.0 snapshot: if_held_action, long_term_action, new_money_action, "
        "price_zone.")
    assert response["warnings"] == [version_warning("2.0.0", RETIRED_2_0_0)]


@pytest.mark.parametrize("name", sorted(set(ACTION_CHANGES) - set(GATE_CHANGES)))
async def test_a_relabelled_outcome_is_not_a_change(detect: Detect, name: str) -> None:
    """SN-R10: the rows whose 2.0.0 action_now differs from 3.0.0 new_money with the same gates (hold_or_add ->
    starter; earnings wait_for_data -> wait; weak_liquidity wait_for_data -> avoid; unsupported hold -> watch)
    give no material change: actions are not compared across versions. RF."""
    response = await detect(recorded(name), await dm.run(name))
    assert response["material_changes"] == []


async def test_a_relabelled_knife_outcome_is_not_a_change(detect: Detect) -> None:
    """SN-R10 (knife): 0609fde gave wait_for_data for a knife on an admitting candidate; 3.0.0 gives avoid with the
    same falling_knife id: no material change (S6's recording with that 2.0.0 action). RF."""
    previous = {**recorded("S6_knife_dislocation"), "decision_action_now": "wait_for_data"}
    response = await detect(previous, await dm.run("S6_knife_dislocation"))

    assert response["material_changes"] == []


# A retained diffed field changed in the current run, against the 2.0.0 recording of the row, with the line the
# same change gives 3 -> 3 (SN-C3) and so 2 -> 2.
RETAINED: dict[str, tuple[str, dict[str, Any], str]] = {
    **{field: (CLEAN, *MUTATIONS[field]) for field in (
        "price", "score", "hard_gate_ids", "mid_term_fit", "long_term_fit", "bullish_signals", "bearish_signals",
        "tilt", "confidence", "risk_regime", "valuation_gate", "support_instrument", "support_currency",
        "price_source")},
    "-hard_gate": ("S4_earnings_gate_on_hold_or_add", {"decision_card__hard_gates__blocking": []},
                   "-hard_gate: earnings_blackout"),
    "-bearish": ("E2_mixed_fcf_bullish_chart", {"signals__bearish": []}, "-bearish: negative_free_cash_flow"),
}


def test_every_retained_diffed_field_has_a_case() -> None:
    """SN-R11 (guard): RETAINED covers every diffed field 2.0.0 and 3.0.0 share. G."""
    shared = set(diff_module._DIFFED_FIELDS) & SNAPSHOT_FIELDS["2.0.0"]
    assert {field for field in RETAINED if not field.startswith("-")} == shared


@pytest.mark.parametrize("case", RETAINED)
async def test_retained_fields_alert_across_2_to_3(detect: Detect, case: str) -> None:
    """SN-R11 (amended for price_zone, plan §8): each retained field change, signals included, gives the same single
    line as a same-version compare. RF (the 14 single-field cases are a 2 -> 2 control on 71f5bde; m0 fails them,
    m1 the signal cases)."""
    name, changes, line = RETAINED[case]
    response = await detect(recorded(name), edited(await dm.run(name), **changes))

    assert response["material_changes"] == [line]


async def test_a_known_signal_list_that_becomes_unknown_across_2_to_3(detect: Detect) -> None:
    """SN-R12: E2's recorded lists vs a run with options_signals failing: one change per list, no +/- items.
    RF."""
    previous = recorded("E2_mixed_fcf_bullish_chart")
    response = await detect(previous, await dm.run("E2_mixed_fcf_bullish_chart", options=None))

    assert response["material_changes"] == [f"bullish_signals: {previous['bullish_signals']} -> None",
                                             f"bearish_signals: {previous['bearish_signals']} -> None"]


# 0609fde's 2.0.0 snapshot of S2 with beta missing (ported probe: fa36d96, dm.run(S2, risk=CALM without beta)).
S2_NO_BETA_2_0_0 = {"decision_action_now": "wait_for_data", "hard_gate_ids": ["data_quality_critical"],
                    "risk_regime": "low", "snapshot_hash": "70bed780b8b3f846"}


async def test_a_rule_driven_gate_change_is_reported_with_the_rule_note(detect: Detect) -> None:
    """SN-R15: a 2.0.0 data_quality_critical that fired on a missing beta alone clears under 3.0.0 (B3): reported
    as -hard_gate, with the rule note; no day-0 claim. PD D-10."""
    risk = {k: v for k, v in dm.CALM.items() if k != "beta"}
    response = await detect({**recorded(CLEAN), **S2_NO_BETA_2_0_0}, await dm.run(CLEAN, risk=risk))

    assert response["material_changes"] == ["-hard_gate: data_quality_critical"]
    assert response["warnings"] == [version_warning("2.0.0", RETIRED_2_0_0)] and "day 0" not in RULE_NOTE
    assert response["summary"].startswith("ACME: Cleared hard gate(s): data_quality_critical.")


def test_the_rule_note_names_every_new_analyze_gate() -> None:
    """SN-R15 (guard): the rule note names exactly the analyze gates 0609fde did not have. G."""
    new = set(ANALYZE_GATE_IDS) - GATES_2_0_0
    assert set(ANALYZE_GATE_IDS) >= GATES_2_0_0
    assert new == {"earnings_unverifiable", "stale_data", "freshness_unverifiable", "thesis_broken", "near_52w_high"}
    assert all(g in diff_module._RULE_CHANGES_3_0_0[0] for g in new)


# --- 1.x -> 3 (SN-R16, R17) [D-10] ------------------------------------------------------------------------------

LEGACY = {
    "1.0.0_compact": (lambda raw: as_1_0_0(legacy_compact_snapshot(as_0609fde(raw))), ["action_long_term",
                                                                                        "action_mid_term"]),
    "1.0.0_enriched": (lambda raw: as_1_0_0(legacy_enriched_snapshot(as_0609fde(raw))),
                       ["action_long_term", "action_mid_term", "zone"]),
    "1.1.0_compact": (lambda raw: legacy_compact_snapshot(as_0609fde(raw)),
                      ["action_long_term", "action_mid_term", "decision_action_now"]),
    "1.1.0_enriched": (lambda raw: legacy_enriched_snapshot(as_0609fde(raw)), RETIRED_2_0_0),
}
FIELDS_2_0_0 = ["as_of_kind", "price_as_of", "price_source", "price_stale", "support_currency", "support_instrument"]


@pytest.mark.parametrize("kind", LEGACY)
async def test_a_1_x_snapshot_migrates(detect: Detect, kind: str) -> None:
    """SN-R16 (plan §0 #10 counts): a 1.x snapshot of the same result: no phantom change; not_comparable = the
    legacy signals, the 2.0.0 fields, the three actions and price_zone (+ the fits for compact; + hard_gate_ids
    for 1.0.0); the warning retires the pre-gate codes, decision_action_now from 1.1.0 and zone from the enriched
    copy. PD D-10."""
    make, retired = LEGACY[kind]
    result = await clean()

    response = await detect(make(result), result)

    extra = (["long_term_fit", "mid_term_fit"] if kind.endswith("compact") else []) + (
        ["hard_gate_ids"] if kind.startswith("1.0.0") else [])
    assert response["material_changes"] == []
    assert response["not_comparable"] == sorted(["bearish_signals", "bullish_signals", *FIELDS_2_0_0, *V3_ONLY,
                                                  *extra])
    assert response["warnings"] == [version_warning(kind[:5], retired)]


async def test_an_action_only_change_against_1_1_0_is_disclosed(detect: Detect) -> None:
    """SN-R17: 1.1.0 -> 3.0.0 with only new money moving (buy -> avoid): no material change (v0.2.0 alerted
    "action_now: buy -> avoid"), and the summary names new_money_action as not compared. PD D-10."""
    result = await clean()
    previous = legacy_compact_snapshot(as_0609fde(result))

    response = await detect(previous, edited(result, decision_card__new_money__action="avoid"))

    assert response["material_changes"] == []
    assert response["summary"].startswith("ACME: No material changes detected. Not compared with the previous "
                                          "1.1.0 snapshot: ")
    assert "new_money_action" in response["summary"].split("Not compared")[1]


# --- Freshness (FR-R-F15 inverted, SN-R21) [D-8] ----------------------------------------------------------------

FRESHNESS_ROWS = {"R3": "stale_data", "R6": "stale_data", "R7": "freshness_unverifiable",
                  "R14": "freshness_unverifiable"}


@pytest.mark.parametrize("row", FRESHNESS_ROWS)
async def test_a_freshness_only_change_is_material(detect: Detect, monkeypatch: pytest.MonkeyPatch,
                                                   row: str) -> None:
    """FR-R-F15 (inverted by D-8, plan §0 #5): the same run with and without the freshness gate differs only by the
    gate: +hard_gate and the new-money and long-term lines are material and the hash moves; clearing reports the
    reverse. PD D-8."""
    scenario = fr.scenario(fr.TRUTH_TABLE[row], row)
    gated = await dm.run(scenario)
    real = dm.orchestrator.check_price_freshness
    monkeypatch.setattr(dm.orchestrator, "check_price_freshness", lambda **kw: ([], real(**kw)[1]))
    ungated = await dm.run(scenario)
    gate = FRESHNESS_ROWS[row]

    fired = await detect(ungated["watchlist_snapshot"], gated)
    cleared = await detect(gated["watchlist_snapshot"], ungated)

    assert (fired["material_changes"], fired["hash_changed"]) == (
        ["new_money_action: starter -> wait_for_data", "long_term_action: buy -> wait_for_data",
         f"+hard_gate: {gate}"], True)
    assert cleared["material_changes"] == ["new_money_action: wait_for_data -> starter",
                                           "long_term_action: wait_for_data -> buy", f"-hard_gate: {gate}"]


MONDAY_EVENING = fr.TruthRow(fr.at(2026, 10, 12, 17), lambda: (fr.daily(fr.E_MON), None), (), fr.E_MON, 0, fr.DAILY)


async def test_a_post_close_lag_alerts_through_the_gate_not_the_age(detect: Detect) -> None:
    """SN-R21 (D-8): saved Monday after the close, re-run Tuesday after the close before Tuesday's bar exists (O2):
    stale_data fires, so +hard_gate and the new-money change alert; price_stale flips too but is never compared
    (D-8: a growing age alone is never an alert). PD D-8."""
    monday = await dm.run(fr.scenario(MONDAY_EVENING, "Mon 17:00"))
    tuesday = await dm.run(fr.scenario(fr.TRUTH_TABLE["O2"], "O2"))

    response = await detect(monday["watchlist_snapshot"], tuesday)

    assert (monday["watchlist_snapshot"]["price_stale"], tuesday["watchlist_snapshot"]["price_stale"]) == (False, True)
    assert (response["material_changes"], response["hash_changed"]) == (
        ["new_money_action: starter -> wait_for_data", "long_term_action: buy -> wait_for_data",
         "+hard_gate: stale_data"], True)


def test_the_tool_description_says_a_freshness_gate_is_material() -> None:
    """SN-R21 (wording, D-8): detect_changes' description no longer says staleness never counts; a freshness gate
    firing or clearing is material, price_stale and ages are not. PD D-8."""
    doc = " ".join((detect_changes.__doc__ or "").split())
    assert "staleness never count" not in doc
    assert "stale_data" in doc and "freshness_unverifiable" in doc and "price_stale" in doc


# --- earliest_reassessment (RA.16) ------------------------------------------------------------------------------

SUNDAY_NOON = dm.ET.localize(datetime(2026, 10, 11, 12, 0))


async def test_a_moved_reassessment_date_alone_is_not_a_change(detect: Detect, monkeypatch: pytest.MonkeyPatch) -> None:
    """RA.16: earliest_reassessment is not in the snapshot or its hash. S4 run Friday evening (dated Tue Oct 13)
    and Sunday noon (same stubs: undatable, conditions_only) record the same snapshot; a sentinel date gives no
    change either. G (green on 71f5bde, where the 2.0.0 snapshot never read the field)."""
    friday = await dm.run("S4_earnings_gate_on_hold_or_add")
    sunday = await dm.run("S4_earnings_gate_on_hold_or_add", now=SUNDAY_NOON)
    sentinel = {"date": "1999-01-01", "basis": "earnings_clear", "earnings_risk_through": None,
                "session_method": None, "outstanding": []}
    monkeypatch.setattr(decision, "earliest_reassessment", lambda **_: copy.deepcopy(sentinel))
    faked = await dm.run("S4_earnings_gate_on_hold_or_add")
    dates = [r["decision_card"]["new_money"]["earliest_reassessment"]["date"] for r in (friday, sunday, faked)]

    assert dates == ["2026-10-13", None, "1999-01-01"]
    for current in (sunday, faked):
        response = await detect(friday["watchlist_snapshot"], current)
        assert (response["material_changes"], response["hash_changed"]) == ([], False)
    assert not [k for k in friday["watchlist_snapshot"] if "reassess" in k]
    assert {"snapshot_hash", "snapshot_as_of_date", "price_stale"} == _UNHASHED_SNAPSHOT_KEYS


# --- The freshness invariant, snapshot clause (plan §10 WP2.4, §0 #22) -----------------------------------------

BLOCKING_ROWS = [name for name, row in fr.TRUTH_TABLE.items() if row.codes]


@pytest.mark.parametrize("name", BLOCKING_ROWS)
async def test_no_entry_wording_in_the_snapshot_under_a_freshness_gate(name: str) -> None:
    """FI (snapshot clause): when stale_data or freshness_unverifiable fires, no watchlist_snapshot value carries an
    entry code or advice phrase (the 2.0.0 snapshot kept the pre-gate hold_or_add / accumulate), and the freshness
    id is in hard_gate_ids. RF."""
    result = await dm.run(fr.scenario(fr.TRUTH_TABLE[name], name))
    snapshot = result["watchlist_snapshot"]

    assert {"stale_data", "freshness_unverifiable"} & set(snapshot["hard_gate_ids"])
    assert [(path, value) for path, value in oc.leaves(snapshot) if oc.advice(value)] == []
