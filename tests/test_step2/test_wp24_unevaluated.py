"""WP2.4: a gate whose input failed is not evaluated, never cleared (snapshot 3.0.0 unevaluated_gate_ids).

When a gate's input fails, analyze moves the gate from decision_card.hard_gates.blocking to new_money.unevaluated
and long_term.new_money.unevaluated, so it leaves hard_gate_ids. The snapshot records those ids
(unevaluated_gate_ids, hashed). detect_changes reports a fired gate that is now unevaluated under
hard_gates.not_evaluated, as +unevaluated_gate and "Gate(s) not evaluated", never as -hard_gate or "Cleared hard
gate(s)"; a gate evaluated again is -unevaluated_gate and "Gate(s) evaluated again". "Cleared" is left for a gate
that was evaluated and did not fire. An older previous snapshot has no unevaluated_gate_ids: the current side alone
decides.
"""

import copy
import importlib
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from stock_analysis.utils.normalize import _UNHASHED_SNAPSHOT_KEYS, build_watchlist_snapshot
from tests.support import decision_matrix as dm
from tests.support.legacy_snapshot import as_0609fde, legacy_compact_snapshot
from tests.test_step2.test_characterize_snapshot import CLEAN, edited
from tests.test_step2.test_wp24_snapshot import recorded

diff_module = importlib.import_module("stock_analysis.tools.diff_analysis")

Detect = Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]]

FIELD = "unevaluated_gate_ids"
S4 = "S4_earnings_gate_on_hold_or_add"
# Earnings in 3 days from an unverified source: earnings_blackout and earnings_unverifiable both fire.
UNVERIFIED_NEAR = {"earnings": {**dm.NEAR["earnings"], "verification": "unverifiable"}}
# A7 without moving averages or a 52-week range: the price zone is undetermined, so near_52w_high is unevaluated.
_A7 = dm.SCENARIOS["A7_near_52w_high_bullish"].technicals or {}
# S9 without its dollar volume: liquidity_missing fires and weak_liquidity cannot be computed.
NO_DOLLAR_VOLUME = {**dm.THIN, "liquidity": {"avg_dollar_volume": None}}
A7_NO_ZONE = {**copy.deepcopy(_A7), "moving_averages": {"rules": copy.deepcopy(_A7["moving_averages"]["rules"])},
              "price_position": {}}


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


def gate_lines(response: dict[str, Any]) -> list[str]:
    """The ±hard_gate and ±unevaluated_gate material lines, in order."""
    prefixes = tuple(f"{sign}{kind}_gate: " for kind in ("hard", "unevaluated") for sign in "+-")
    return [line for line in response["material_changes"] if line.startswith(prefixes)]


# --- The producer ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", dm.SCENARIOS)
async def test_the_snapshot_records_every_unevaluated_gate(name: str) -> None:
    """SN-U1: every scenario's snapshot carries unevaluated_gate_ids, the sorted unique ids of
    decision_card.new_money.unevaluated and long_term.new_money.unevaluated ([] when none; S8's missing_runway and
    thesis_broken). RF."""
    result = await dm.run(name)
    card = result["decision_card"]
    listed = [*card["new_money"]["unevaluated"], *card["long_term"]["new_money"]["unevaluated"]]

    assert result["watchlist_snapshot"][FIELD] == sorted({gate["id"] for gate in listed})


@pytest.mark.parametrize(("card", "expected"), [
    (None, None),
    ({"new_money": {"action": "buy"}, "long_term": {"new_money": {"action": "buy"}}}, None),
    ({"new_money": {"unevaluated": []}, "long_term": {"new_money": {"unevaluated": []}}}, []),
    ({"new_money": {"unevaluated": [{"id": "thesis_broken", "reason": "r"}]}}, ["thesis_broken"]),
    ({"long_term": {"new_money": {"unevaluated": [{"id": "falling_knife", "reason": "r"}]}}}, ["falling_knife"]),
    ({"new_money": {"unevaluated": [{"id": "near_52w_high", "reason": "r"}, {"id": "falling_knife", "reason": "r"}]},
      "long_term": {"new_money": {"unevaluated": [{"id": "falling_knife", "reason": "r"}]}}},
     ["falling_knife", "near_52w_high"]),
])
def test_the_unevaluated_ids_follow_the_null_convention(card: Any, expected: list[str] | None) -> None:
    """SN-U1: null when neither unevaluated list is a list (hard_gate_ids' convention: unknown, not known-empty);
    otherwise the sorted union of both horizons' ids. RF."""
    assert build_watchlist_snapshot({"symbol": "ACME", "decision_card": card})[FIELD] == expected


# --- Transitions (3 -> 3) ----------------------------------------------------------------------------------------

GATE = "falling_knife"
STATES: dict[str, dict[str, Any]] = {
    "clear": {},
    "fired": {"decision_card__hard_gates__blocking": [{"id": GATE, "reason": "fired"}]},
    "unevaluated": {"decision_card__new_money__unevaluated": [{"id": GATE, "reason": "not computed"}],
                    "decision_card__long_term__new_money__unevaluated": [{"id": GATE, "reason": "not computed"}]},
}
TRANSITIONS: dict[tuple[str, str], tuple[list[str], str]] = {
    ("clear", "fired"): ([f"+hard_gate: {GATE}"], f"New hard gate(s): {GATE}."),
    ("fired", "clear"): ([f"-hard_gate: {GATE}"], f"Cleared hard gate(s): {GATE}."),
    ("clear", "unevaluated"): ([f"+unevaluated_gate: {GATE}"], f"Gate(s) not evaluated: {GATE}."),
    ("fired", "unevaluated"): ([f"+unevaluated_gate: {GATE}"], f"Gate(s) not evaluated: {GATE}."),
    ("unevaluated", "clear"): ([f"-unevaluated_gate: {GATE}"], f"Gate(s) evaluated again: {GATE}."),
    ("unevaluated", "fired"): ([f"+hard_gate: {GATE}", f"-unevaluated_gate: {GATE}"],
                               f"New hard gate(s): {GATE}. Gate(s) evaluated again: {GATE}."),
}


async def test_the_field_is_hashed() -> None:
    """SN-U9: a gate listed unevaluated, nothing else moving, changes only unevaluated_gate_ids and snapshot_hash;
    the field is not in the unhashed set. RF."""
    result = await dm.run(CLEAN)
    moved = edited(result, **STATES["unevaluated"])["watchlist_snapshot"]
    before = result["watchlist_snapshot"]

    assert FIELD not in _UNHASHED_SNAPSHOT_KEYS
    assert {key for key in {*before, *moved} if before.get(key) != moved.get(key)} == {FIELD, "snapshot_hash"}


@pytest.mark.parametrize(("before", "after"), TRANSITIONS)
async def test_each_gate_transition_has_its_line(detect: Detect, before: str, after: str) -> None:
    """SN-U5: on S2, one gate moving between fired, evaluated clear and unevaluated (the card edited, the snapshot
    rebuilt) gives exactly its lines and summary sentence, and moves the hash; only fired -> clear is "Cleared",
    fired -> unevaluated is not_evaluated and never -hard_gate. RF (clear <-> fired are G controls)."""
    lines, sentence = TRANSITIONS[(before, after)]
    result = await dm.run(CLEAN)

    response = await detect(edited(result, **STATES[before])["watchlist_snapshot"], edited(result, **STATES[after]))

    assert (response["material_changes"], response["summary"], response["hash_changed"]) == (
        lines, f"ACME: {sentence}", True)
    gates = response["changes"]["hard_gates"]
    assert (gates["cleared"], gates["not_evaluated"]) == (
        [GATE] if (before, after) == ("fired", "clear") else [], [GATE] if (before, after) == ("fired", "unevaluated")
        else [])


async def test_an_unknown_unevaluated_list_is_one_change(detect: Detect) -> None:
    """SN-U10: the field going from known to unknown (null: the card's lists absent) and back is one line, never
    ±unevaluated_gate items (the null rule of hard_gate_ids and the signal lists). RF."""
    result = await dm.run(CLEAN)
    unknown = edited(result, decision_card__new_money__unevaluated=None,
                     decision_card__long_term__new_money__unevaluated=None)
    listed = edited(result, **STATES["unevaluated"])

    lost = await detect(result["watchlist_snapshot"], unknown)
    found = await detect(unknown["watchlist_snapshot"], listed)

    assert lost["material_changes"] == [f"{FIELD}: [] -> None"]
    assert found["material_changes"] == [f"{FIELD}: None -> ['{GATE}']"]


# --- Through analyze: a gate input fails, then recovers -----------------------------------------------------------

# (scenario, its overrides, the overrides that fail a gate's input, the failed run's gate lines, not_evaluated)
FAILURES: dict[str, tuple[str, dict[str, Any], dict[str, Any], list[str], list[str]]] = {
    "earnings_already_unverifiable": (S4, {"events": UNVERIFIED_NEAR}, {"events": None},
                                      ["+unevaluated_gate: earnings_blackout"], ["earnings_blackout"]),
    "S4_events_failed": (S4, {}, {"events": None},
                         ["+hard_gate: earnings_unverifiable", "+unevaluated_gate: earnings_blackout"],
                         ["earnings_blackout"]),
    "A_fundamentals_failed": ("A_broken_bearish", {}, {"fundamentals": None},
                              ["+hard_gate: data_quality_critical", "+unevaluated_gate: missing_runway",
                               "+unevaluated_gate: thesis_broken"], ["thesis_broken"]),
    "C_technicals_failed": ("C_quality_knife_dislocation", {}, {"technicals": None},
                            ["+hard_gate: data_quality_critical", "+hard_gate: freshness_unverifiable",
                             "+unevaluated_gate: falling_knife", "+unevaluated_gate: near_52w_high"],
                            ["falling_knife"]),
    "A7_zone_undetermined": ("A7_near_52w_high_bullish", {}, {"technicals": A7_NO_ZONE},
                             ["+unevaluated_gate: near_52w_high"], ["near_52w_high"]),
    "S9_risk_failed": ("S9_weak_liquidity", {}, {"risk": None},
                       ["+hard_gate: data_quality_critical", "+hard_gate: liquidity_missing",
                        "+unevaluated_gate: weak_liquidity"], ["weak_liquidity"]),
    "S9_dollar_volume_missing": ("S9_weak_liquidity", {}, {"risk": NO_DOLLAR_VOLUME},
                                 ["+hard_gate: liquidity_missing", "+unevaluated_gate: weak_liquidity"],
                                 ["weak_liquidity"]),
}


async def runs(case: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the evaluated run, the run whose gate input failed) of a FAILURES case."""
    name, overrides, failing, _, _ = FAILURES[case]
    return await dm.run(name, **overrides), await dm.run(name, **{**overrides, **failing})


def ids(lines: list[str], prefix: str) -> list[str]:
    return [line.split(": ", 1)[1] for line in lines if line.startswith(prefix)]


@pytest.mark.parametrize("case", FAILURES)
async def test_a_fired_gate_whose_input_failed_is_not_cleared(detect: Detect, case: str) -> None:
    """SN-U2: a gate input fails (events fetch, fundamentals, technicals, an undetermined price zone): the fired
    gate leaves hard_gate_ids for unevaluated_gate_ids, and detect_changes lists it under hard_gates.not_evaluated
    with +unevaluated_gate and "Gate(s) not evaluated", never -hard_gate or "Cleared". RF."""
    *_, lines, not_evaluated = FAILURES[case]
    evaluated, failed = await runs(case)

    response = await detect(evaluated["watchlist_snapshot"], failed)

    assert gate_lines(response) == lines
    gates = response["changes"]["hard_gates"]
    assert (gates["cleared"], gates["not_evaluated"]) == ([], not_evaluated)
    assert f"Gate(s) not evaluated: {', '.join(ids(lines, '+unevaluated_gate'))}." in response["summary"]
    assert "Cleared" not in response["summary"] and response["hash_changed"] is True


async def test_the_earnings_case_reports_only_the_unevaluated_gate(detect: Detect) -> None:
    """SN-U3 (the reported case): with earnings already unverifiable (earnings_blackout and earnings_unverifiable
    fired), the events fetch failing leaves one line, +unevaluated_gate: earnings_blackout, and a summary saying it
    was not evaluated (it was "Cleared hard gate(s): earnings_blackout." with no warning). RF."""
    evaluated, failed = await runs("earnings_already_unverifiable")

    response = await detect(evaluated["watchlist_snapshot"], failed)

    assert (response["material_changes"], response["summary"], response["warnings"]) == (
        ["+unevaluated_gate: earnings_blackout"], "ACME: Gate(s) not evaluated: earnings_blackout.", None)
    assert [evaluated["watchlist_snapshot"]["hard_gate_ids"], failed["watchlist_snapshot"]["hard_gate_ids"],
            failed["watchlist_snapshot"][FIELD]] == [["earnings_blackout", "earnings_unverifiable"],
                                                     ["earnings_unverifiable"], ["earnings_blackout"]]


@pytest.mark.parametrize("case", FAILURES)
async def test_a_recovered_input_evaluates_the_gate_again(detect: Detect, case: str) -> None:
    """SN-U4 (recovery, the reverse runs): a gate unevaluated before that fires again is +hard_gate; every
    unevaluated gate is -unevaluated_gate and "Gate(s) evaluated again", never "Cleared"; only the gates the failure
    fired (data_quality_critical, freshness_unverifiable, earnings_unverifiable), now evaluated clear, are cleared.
    RF."""
    *_, lines, not_evaluated = FAILURES[case]
    evaluated, failed = await runs(case)
    unevaluated, fired_by_failure = ids(lines, "+unevaluated_gate"), ids(lines, "+hard_gate")

    response = await detect(failed["watchlist_snapshot"], evaluated)

    assert gate_lines(response) == [*(f"+hard_gate: {g}" for g in not_evaluated),
                                    *(f"-hard_gate: {g}" for g in fired_by_failure),
                                    *(f"-unevaluated_gate: {g}" for g in unevaluated)]
    gates = response["changes"]["hard_gates"]
    assert (gates["new"], gates["cleared"], gates["not_evaluated"]) == (not_evaluated, fired_by_failure, [])
    assert f"Gate(s) evaluated again: {', '.join(unevaluated)}." in response["summary"]


async def test_an_unevaluated_gate_evaluated_clear_is_not_cleared(detect: Detect) -> None:
    """SN-U4 (unevaluated -> clear): A7 with an undetermined zone (near_52w_high unevaluated) re-run at a price in
    the above-SMA50 band (evaluated, not fired): -unevaluated_gate and "Gate(s) evaluated again", never "Cleared".
    RF."""
    undetermined = await dm.run("A7_near_52w_high_bullish", technicals=A7_NO_ZONE)
    above_sma50 = await dm.run("A7_near_52w_high_bullish", technicals=dm.STRONG)

    response = await detect(undetermined["watchlist_snapshot"], above_sma50)

    assert above_sma50["action_zones"]["price_zone"] == "above_sma50"
    assert gate_lines(response) == ["-unevaluated_gate: near_52w_high"]
    assert "Gate(s) evaluated again: near_52w_high." in response["summary"] and "Cleared" not in response["summary"]


async def test_a_gate_evaluated_and_not_fired_is_still_cleared(detect: Detect) -> None:
    """SN-U6 (genuine clearance): S4's earnings_blackout, then earnings 40 days out (the events fetch works): it is
    cleared, not not_evaluated: -hard_gate and "Cleared hard gate(s)". RF (the Cleared clause is a G control)."""
    response = await detect((await dm.run(S4))["watchlist_snapshot"], await dm.run(S4, events=dm.FAR))
    gates = response["changes"]["hard_gates"]

    assert (gates["cleared"], gates["not_evaluated"]) == (["earnings_blackout"], [])
    assert gate_lines(response) == ["-hard_gate: earnings_blackout"]
    assert "Cleared hard gate(s): earnings_blackout." in response["summary"]


# --- Older previous snapshots and validation ----------------------------------------------------------------------

OLDER: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "2.0.0": lambda result: recorded(S4),  # the WP2.0 recording of S4
    "1.1.0": lambda result: legacy_compact_snapshot(as_0609fde(result)),
}


@pytest.mark.parametrize("kind", OLDER)
async def test_an_older_snapshot_s_gate_now_unevaluated_is_not_cleared(detect: Detect, kind: str) -> None:
    """SN-U7: a 2.0.0 (recorded) or 1.1.0 previous where earnings_blackout fired, against S4 with the events fetch
    failing: unevaluated_gate_ids is not_comparable and named as not compared, and the current side alone still
    reports the gate as not evaluated, never as cleared. RF."""
    previous = OLDER[kind](await dm.run(S4))

    response = await detect(previous, await dm.run(S4, events=None))

    assert previous["hard_gate_ids"] == ["earnings_blackout"] and FIELD not in previous
    assert gate_lines(response) == ["+hard_gate: earnings_unverifiable", "+unevaluated_gate: earnings_blackout"]
    assert FIELD in response["not_comparable"] and "unevaluated_gates" not in response["changes"]
    assert response["changes"]["hard_gates"]["not_evaluated"] == ["earnings_blackout"]
    assert "Gate(s) not evaluated: earnings_blackout." in response["summary"] and "Cleared" not in response["summary"]
    assert response["summary"].endswith(f", {FIELD}.")


async def test_a_3_0_0_previous_without_the_field_is_invalid(detect: Detect) -> None:
    """SN-U8: a 3.0.0 previous must carry unevaluated_gate_ids (null allowed): without it, invalid_snapshot before
    any analysis; a non-string item is a type problem. RF."""
    result = await dm.run(CLEAN)
    snapshot = result["watchlist_snapshot"]

    response = await detect({k: v for k, v in snapshot.items() if k != FIELD}, result)

    assert (response.get("error_type"), response.get("problems"), response["_analyzed"]) == (
        "invalid_snapshot", [f"missing required key: {FIELD}"], [])
    assert diff_module.validate_previous_snapshot({**snapshot, FIELD: None}, "ACME") == []
    assert diff_module.validate_previous_snapshot({**snapshot, FIELD: [GATE, 1]}, "ACME") == [
        f"{FIELD}: expected string list, got list"]


@pytest.mark.parametrize("risk", [None, NO_DOLLAR_VOLUME], ids=["risk_failed", "dollar_volume_missing"])
async def test_missing_liquidity_leaves_weak_liquidity_unevaluated(risk: dict[str, Any] | None) -> None:
    """SN-U11: without a dollar volume for a supported stock, weak_liquidity is unevaluated at both horizons (with
    its reason) while liquidity_missing still fires and decides wait_for_data, as before; for an unsupported ETF
    neither liquidity gate applies, so nothing is unevaluated. RF."""
    result = await dm.run("S9_weak_liquidity", risk=risk)
    etf = await dm.run("ETF_strong", risk=risk)
    card = result["decision_card"]

    for decision in (card["new_money"], card["long_term"]["new_money"]):
        assert decision["action"] == "wait_for_data"
        assert "liquidity_missing" in [blocker["id"] for blocker in decision["blocked_by"]]
        assert {"id": "weak_liquidity", "reason": "average daily dollar volume unavailable"} in decision["unevaluated"]
    assert result["watchlist_snapshot"][FIELD] == ["weak_liquidity"]
    assert "liquidity_missing" in result["watchlist_snapshot"]["hard_gate_ids"]
    assert "weak_liquidity" not in etf["watchlist_snapshot"][FIELD]
