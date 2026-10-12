"""What-changed diff analysis between stock snapshots."""

import math
from collections.abc import Callable
from time import perf_counter
from typing import Any

from stock_analysis.tools.analyze import analyze_stock
from stock_analysis.utils.helpers import safe_float, safe_round
from stock_analysis.utils.normalize import (
    COMPATIBLE_SNAPSHOT_VERSIONS,
    SNAPSHOT_IDENTITY_KEYS,
    SNAPSHOT_VERSION,
)
from stock_analysis.utils.provenance import build_meta

# Type of every optional snapshot field a previous snapshot may carry (null is
# always allowed: it is a known null).
_FIELD_TYPES: dict[str, str] = {
    "price": "number", "market_cap": "number", "score": "number", "volatility_ann": "number",
    "is_unprofitable": "bool", "price_stale": "bool",
    "hard_gate_ids": "string_list", "bullish_signals": "string_list", "bearish_signals": "string_list",
    "next_catalyst": "object",
    **dict.fromkeys(
        ("sector", "tilt", "confidence", "mid_term_fit", "long_term_fit", "action_mid_term", "action_long_term",
         "valuation_gate", "decision_action_now", "zone", "risk_regime", "support_instrument", "support_currency",
         "price_source", "price_as_of", "as_of_kind", "snapshot_as_of_date", "snapshot_hash"),
        "string",
    ),
}


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "number": _finite_number,
    "bool": lambda v: isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "string_list": lambda v: isinstance(v, list) and all(isinstance(item, str) for item in v),
    "object": lambda v: isinstance(v, dict),
}
# Known keys of the nested next_catalyst object (other keys are ignored).
_CATALYST_FIELD_TYPES = {
    "type": "string", "status": "string", "date": "string", "source": "string", "reason": "string",
    "days_until": "number", "window": "object",
}
# Fields a pre-2.0.0 snapshot filled ambiguously (see what_changed).
_LEGACY_AMBIGUOUS_FIELDS = frozenset({"bullish_signals", "bearish_signals"})
# Thresholds are crossed only by more than float error (1.40 -> 1.47 is exactly 5%).
_THRESHOLD_TOLERANCE = 1e-9
# Price sources a snapshot moves between with the time of day alone (a fresh bar
# in the session, the completed close outside it); a switch between them is
# recorded but not material.
_ROUTINE_PRICE_SOURCES = frozenset({"intraday_quote", "daily_close"})
# Decision fields compared as scalars: (snapshot field, changes key, label).
_SCALAR_FIELDS = (
    ("tilt", "tilt", "tilt"),
    ("confidence", "confidence", "confidence"),
    ("risk_regime", "risk_regime", "risk_regime"),
    ("zone", "action_zone", "zone"),
    ("decision_action_now", "decision_action_now", "action_now"),
    ("action_mid_term", "action_mid_term", "action_mid_term"),
    ("action_long_term", "action_long_term", "action_long_term"),
    ("valuation_gate", "valuation_gate", "valuation_gate"),
    ("support_instrument", "support_instrument", "support_instrument"),
    ("support_currency", "support_currency", "support_currency"),
    ("price_source", "price_source", "price_source"),
)
# Every snapshot field the diff compares (is_unprofitable and price_stale are kept
# but not compared: analyze stores a known False as null, and staleness moves with
# the clock alone).
_DIFFED_FIELDS = (
    "price", "score", "hard_gate_ids", "mid_term_fit", "long_term_fit", "bullish_signals", "bearish_signals",
    *(field for field, _, _ in _SCALAR_FIELDS),
)


def validate_previous_snapshot(snapshot: Any, symbol: str | None) -> list[str]:
    """Problems that make ``snapshot`` unusable as the previous snapshot for ``symbol``
    (empty when it is usable; ``symbol`` None skips the symbol match). Unknown extra
    fields are ignored."""
    if not isinstance(snapshot, dict):
        return ["previous_snapshot must be an object"]
    problems = [f"missing required key: {key}" for key in SNAPSHOT_IDENTITY_KEYS if key not in snapshot]
    version = snapshot.get("snapshot_version")
    if "snapshot_version" in snapshot and not (isinstance(version, str) and version in COMPATIBLE_SNAPSHOT_VERSIONS):
        problems.append(f"unsupported snapshot_version: {version!r}")
    snapshot_symbol = snapshot.get("symbol")
    if isinstance(snapshot_symbol, str):
        if symbol is not None and snapshot_symbol.upper().strip() != symbol:
            problems.append(f"symbol {snapshot_symbol!r} does not match {symbol}")
    elif "symbol" in snapshot:
        problems.append(f"symbol: expected string, got {type(snapshot_symbol).__name__}")
    for key, kind in _FIELD_TYPES.items():
        value = snapshot.get(key)
        if value is not None and not _TYPE_CHECKS[kind](value):
            problems.append(f"{key}: expected {kind.replace('_', ' ')}, got {type(value).__name__}")
    catalyst = snapshot.get("next_catalyst")
    if isinstance(catalyst, dict):
        for key, kind in _CATALYST_FIELD_TYPES.items():
            value = catalyst.get(key)
            if value is not None and not _TYPE_CHECKS[kind](value):
                problems.append(f"next_catalyst.{key}: expected {kind}, got {type(value).__name__}")
        window = catalyst.get("window")
        if isinstance(window, dict):
            for bound in ("start", "end"):
                value = window.get(bound)
                if value is not None and not isinstance(value, str):
                    problems.append(f"next_catalyst.window.{bound}: expected string, got {type(value).__name__}")
    return problems


def _diff_scalar(
    prev: dict[str, Any],
    curr: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Diff a simple scalar field, returning previous/current/changed. A null is a
    known value, so null -> value is a change."""
    previous = prev.get(key)
    current = curr.get(key)
    return {
        "previous": previous,
        "current": current,
        "changed": previous != current,
    }


def _diff_price(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Diff price with percentage change and materiality check."""
    previous = safe_float(prev.get("price"))
    current = safe_float(curr.get("price"))

    change_pct: float | None = None
    # A price appearing or disappearing is material in itself.
    material = (previous is None) != (current is None)
    if previous and current and previous != 0:
        raw_pct = (current - previous) / previous
        change_pct = safe_round(raw_pct, 4)
        material = abs(raw_pct) > 0.05 + _THRESHOLD_TOLERANCE  # judged before rounding

    return {
        "previous": previous,
        "current": current,
        "change_pct": change_pct,
        "material": material,
    }


def _diff_score(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Diff score with absolute change and materiality check."""
    previous = safe_float(prev.get("score"))
    current = safe_float(curr.get("score"))

    change: float | None = None
    material = (previous is None) != (current is None)
    if previous is not None and current is not None:
        change = safe_round(current - previous, 6)
        material = abs(current - previous) > 0.10 + _THRESHOLD_TOLERANCE  # judged before rounding

    return {
        "previous": previous,
        "current": current,
        "change": change,
        "material": material,
    }


def _diff_signals(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Diff bullish and bearish signal lists. A list absent from either snapshot is
    not compared (its keys are left out); when either side is null (unknown) no
    signal is reported as new or removed, only that the list changed."""
    diff: dict[str, Any] = {}
    for side in ("bullish", "bearish"):
        key = f"{side}_signals"
        if key not in prev or key not in curr:
            continue
        before, after = prev[key], curr[key]
        known = before is not None and after is not None
        diff[f"new_{side}"] = sorted(set(after) - set(before)) if known else []
        diff[f"removed_{side}"] = sorted(set(before) - set(after)) if known else []
        diff[f"{side}_changed"] = set(before) != set(after) if known else (before is None) != (after is None)
    return diff


def _diff_hard_gates(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Diff the set of active hard-gate ids — what fired or cleared between snapshots."""
    previous = prev.get("hard_gate_ids")
    current = curr.get("hard_gate_ids")
    if previous is None or current is None:
        # Unknown on either side: no gate is reported as new or cleared, only that
        # the list went from unknown to known (or back).
        return {"previous": previous, "current": current, "new": [], "cleared": [],
                "changed": (previous is None) != (current is None)}
    prev_gates = set(previous)
    curr_gates = set(current)
    return {
        "previous": sorted(prev_gates),
        "current": sorted(curr_gates),
        "new": sorted(curr_gates - prev_gates),
        "cleared": sorted(prev_gates - curr_gates),
        "changed": prev_gates != curr_gates,
    }


def _build_material_changes(changes: dict[str, Any]) -> list[str]:
    """Build a human-readable list of material changes."""
    items: list[str] = []

    price = changes.get("price", {})
    if price.get("material") and price.get("change_pct") is not None:
        direction = "+" if price["change_pct"] >= 0 else ""
        items.append(f"price {direction}{price['change_pct'] * 100:.1f}%")
    elif price.get("material"):
        items.append(f"price: {price['previous']} -> {price['current']}")

    score = changes.get("score", {})
    if score.get("material") and score.get("change") is not None:
        direction = "+" if score["change"] >= 0 else ""
        items.append(f"score {direction}{score['change']:.2f}")
    elif score.get("material"):
        items.append(f"score: {score['previous']} -> {score['current']}")

    for _, changes_key, label in _SCALAR_FIELDS:
        change = changes.get(changes_key, {})
        if change.get("changed") and change.get("material", True):
            items.append(f"{label}: {change['previous']} -> {change['current']}")

    hard_gates = changes.get("hard_gates", {})
    for gate_id in hard_gates.get("new", []):
        items.append(f"+hard_gate: {gate_id}")
    for gate_id in hard_gates.get("cleared", []):
        items.append(f"-hard_gate: {gate_id}")

    horizon = changes.get("horizon_fit", {})
    mid = horizon.get("mid_term", {})
    if mid.get("changed"):
        items.append(f"mid_term_fit: {mid['previous']} -> {mid['current']}")
    long = horizon.get("long_term", {})
    if long.get("changed"):
        items.append(f"long_term_fit: {long['previous']} -> {long['current']}")

    signals = changes.get("signals", {})
    for sig in signals.get("new_bullish", []):
        items.append(f"+bullish: {sig}")
    for sig in signals.get("removed_bullish", []):
        items.append(f"-bullish: {sig}")
    for sig in signals.get("new_bearish", []):
        items.append(f"+bearish: {sig}")
    for sig in signals.get("removed_bearish", []):
        items.append(f"-bearish: {sig}")

    return items


def _build_summary(
    symbol: str,
    changes: dict[str, Any],
    material_changes: list[str],
) -> str:
    """Build a 1-2 sentence narrative summary of changes."""
    if not material_changes:
        return f"{symbol}: No material changes detected."

    parts: list[str] = [f"{symbol}:"]

    price = changes.get("price", {})
    if price.get("material") and price.get("change_pct") is not None:
        direction = "up" if price["change_pct"] >= 0 else "down"
        parts.append(f"Price {direction} {abs(price['change_pct']) * 100:.1f}%.")
    elif price.get("material"):
        parts.append(f"Price {price['previous']} -> {price['current']}.")

    tilt = changes.get("tilt", {})
    if tilt.get("changed"):
        parts.append(f"Tilt shifted from {tilt['previous']} to {tilt['current']}.")

    risk = changes.get("risk_regime", {})
    if risk.get("changed"):
        parts.append(f"Risk regime changed to {risk['current']}.")

    zone = changes.get("action_zone", {})
    if zone.get("changed"):
        parts.append(f"Zone moved from {zone['previous']} to {zone['current']}.")

    action_now = changes.get("decision_action_now", {})
    if action_now.get("changed"):
        parts.append(
            f"Action now: {action_now['previous']} -> {action_now['current']}."
        )

    hard_gates = changes.get("hard_gates", {})
    if hard_gates.get("new"):
        parts.append(f"New hard gate(s): {', '.join(hard_gates['new'])}.")
    if hard_gates.get("cleared"):
        parts.append(f"Cleared hard gate(s): {', '.join(hard_gates['cleared'])}.")

    score = changes.get("score", {})
    if score.get("material") and score.get("change") is not None:
        direction = "improved" if score["change"] > 0 else "declined"
        parts.append(f"Score {direction} by {abs(score['change']):.2f}.")
    elif score.get("material"):
        parts.append(f"Score {score['previous']} -> {score['current']}.")

    # Fallback when only the changes without a sentence above are material
    if len(parts) == 1:
        parts.append(f"{len(material_changes)} change(s): {'; '.join(material_changes[:3])}.")

    return " ".join(parts)


async def what_changed(
    symbol: str,
    previous_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze what changed for a stock since a previous snapshot.

    Runs a fresh analysis, extracts an enriched snapshot, and diffs it
    against the previous_snapshot if provided.

    Args:
        symbol: Stock ticker symbol.
        previous_snapshot: A prior enriched snapshot to diff against.

    Returns:
        Dict with current snapshot, changes (if previous provided),
        material changes list, and narrative summary.
    """
    start = perf_counter()
    normalized_symbol = symbol.upper().strip()

    def invalid_snapshot(problems: list[str]) -> dict[str, Any]:
        return {
            "meta": build_meta("what_changed", (perf_counter() - start) * 1000),
            "symbol": normalized_symbol,
            "error": True,
            "error_type": "invalid_snapshot",
            "message": f"previous_snapshot is not a usable {normalized_symbol} snapshot",
            "problems": problems,
        }

    # Shape problems are caught before analysis; the symbol is matched after it,
    # against the ticker analyze resolved (a company name resolves to its ticker).
    if previous_snapshot is not None:
        problems = validate_previous_snapshot(previous_snapshot, None)
        if problems:
            return invalid_snapshot(problems)

    try:
        result = await analyze_stock(normalized_symbol)
    except Exception as exc:
        duration_ms = (perf_counter() - start) * 1000
        return {
            "meta": build_meta("what_changed", duration_ms),
            "symbol": normalized_symbol,
            "error": True,
            "error_type": "analysis_failed",
            "message": f"Failed to analyze {normalized_symbol}: {exc}",
        }

    if result.get("error"):
        duration_ms = (perf_counter() - start) * 1000
        return {
            "meta": build_meta("what_changed", duration_ms),
            "symbol": normalized_symbol,
            "error": True,
            "error_type": result.get("error_type", "analysis_failed"),
            "message": result.get("message", "Analysis returned an error"),
        }

    # The snapshot analyze returns at every detail level, used as is (WP7).
    current_snapshot = result.get("watchlist_snapshot") or {}
    if previous_snapshot is not None:
        resolved = str(current_snapshot.get("symbol") or result.get("symbol") or normalized_symbol).upper()
        problems = validate_previous_snapshot(previous_snapshot, resolved)
        if problems:
            return invalid_snapshot(problems)
    duration_ms = (perf_counter() - start) * 1000

    response: dict[str, Any] = {
        "meta": build_meta("what_changed", duration_ms),
        "symbol": normalized_symbol,
        "current_snapshot": current_snapshot,
        "has_previous": previous_snapshot is not None,
    }

    if previous_snapshot is None:
        response["material_changes"] = []
        response["summary"] = f"{normalized_symbol}: Initial snapshot captured. No previous data to compare."
        response["warnings"] = None
        return response

    warnings: list[str] = []

    prev_version = previous_snapshot.get("snapshot_version")
    curr_version = current_snapshot.get("snapshot_version")
    if prev_version and curr_version and prev_version != curr_version:
        warnings.append(
            f"snapshot_version changed ({prev_version} -> {curr_version}); "
            "field semantics may differ"
        )

    # Before 2.0.0 a signal list read [] (or partial) when a tool feeding it had
    # failed, so a legacy list can't be told apart from a known one: treat it as
    # absent. Gate ids stay: analyze always built the gate list.
    if prev_version != SNAPSHOT_VERSION:
        previous_snapshot = {k: v for k, v in previous_snapshot.items() if k not in _LEGACY_AMBIGUOUS_FIELDS}

    # Only fields both snapshots carry are compared (an absent field is unknown);
    # the rest are listed as not_comparable.
    def comparable(key: str) -> bool:
        return key in previous_snapshot and key in current_snapshot

    changes: dict[str, Any] = {}
    if comparable("price"):
        changes["price"] = _diff_price(previous_snapshot, current_snapshot)
    if comparable("score"):
        changes["score"] = _diff_score(previous_snapshot, current_snapshot)
    for field, changes_key, _ in _SCALAR_FIELDS:
        if comparable(field):
            changes[changes_key] = _diff_scalar(previous_snapshot, current_snapshot, field)
    source = changes.get("price_source")
    if source and {source["previous"], source["current"]} <= _ROUTINE_PRICE_SOURCES:
        source["material"] = False
    if comparable("hard_gate_ids"):
        changes["hard_gates"] = _diff_hard_gates(previous_snapshot, current_snapshot)
    horizon_fit = {
        name: _diff_scalar(previous_snapshot, current_snapshot, f"{name}_fit")
        for name in ("mid_term", "long_term")
        if comparable(f"{name}_fit")
    }
    if horizon_fit:
        changes["horizon_fit"] = horizon_fit
    changes["signals"] = _diff_signals(previous_snapshot, current_snapshot)

    material_changes = _build_material_changes(changes)
    # A list that was unknown (null) and is now known, or the reverse, is a change
    # even when no item moved.
    for field in ("hard_gate_ids", "bullish_signals", "bearish_signals"):
        if comparable(field) and (previous_snapshot[field] is None) != (current_snapshot[field] is None):
            material_changes.append(f"{field}: {previous_snapshot[field]} -> {current_snapshot[field]}")
    if any(comparable(field) for field in _DIFFED_FIELDS):
        summary = _build_summary(normalized_symbol, changes, material_changes)
    else:
        summary = f"{normalized_symbol}: Nothing to compare: the previous snapshot has none of the diffed fields."

    response["changes"] = changes
    response["material_changes"] = material_changes
    response["not_comparable"] = sorted(set(current_snapshot) - set(previous_snapshot))
    # The hash also moves on non-material data (a small price move, a newer bar);
    # fetch times, cache hits and ages are kept out of it.
    # Hashes only compare within one version: each version hashes a different field set.
    previous_hash = previous_snapshot.get("snapshot_hash")
    response["hash_changed"] = None
    if previous_hash is not None and prev_version == curr_version:
        response["hash_changed"] = previous_hash != current_snapshot.get("snapshot_hash")
    response["summary"] = summary
    response["warnings"] = warnings or None

    return response
