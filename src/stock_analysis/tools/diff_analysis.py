"""What-changed diff analysis between stock snapshots."""

import math
from collections.abc import Callable
from time import perf_counter
from typing import Any

from stock_analysis.tools.analyze import analyze_stock
from stock_analysis.utils.helpers import safe_float, safe_round
from stock_analysis.utils.normalize import COMPATIBLE_SNAPSHOT_VERSIONS, SNAPSHOT_IDENTITY_KEYS
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
_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
    "bool": lambda v: isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "string_list": lambda v: isinstance(v, list) and all(isinstance(item, str) for item in v),
    "object": lambda v: isinstance(v, dict),
}
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
    ("is_unprofitable", "is_unprofitable", "is_unprofitable"),
    ("support_instrument", "support_instrument", "support_instrument"),
    ("support_currency", "support_currency", "support_currency"),
    ("price_source", "price_source", "price_source"),
    ("price_stale", "price_stale", "price_stale"),
)


def validate_previous_snapshot(snapshot: Any, symbol: str) -> list[str]:
    """Problems that make ``snapshot`` unusable as the previous snapshot for ``symbol``
    (empty when it is usable). Unknown extra fields are ignored."""
    if not isinstance(snapshot, dict):
        return ["previous_snapshot must be an object"]
    problems = [f"missing required key: {key}" for key in SNAPSHOT_IDENTITY_KEYS if key not in snapshot]
    version = snapshot.get("snapshot_version", "")
    if "snapshot_version" in snapshot and version not in COMPATIBLE_SNAPSHOT_VERSIONS:
        problems.append(f"unsupported snapshot_version: {version!r}")
    snapshot_symbol = snapshot.get("symbol")
    if "symbol" in snapshot and (
        not isinstance(snapshot_symbol, str) or snapshot_symbol.upper().strip() != symbol
    ):
        problems.append(f"symbol {snapshot_symbol!r} does not match {symbol}")
    for key, kind in _FIELD_TYPES.items():
        value = snapshot.get(key)
        if value is not None and not _TYPE_CHECKS[kind](value):
            problems.append(f"{key}: expected {kind.replace('_', ' ')}, got {type(value).__name__}")
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
    if previous and current and previous != 0:
        change_pct = safe_round((current - previous) / previous, 4)

    # A price appearing or disappearing is material in itself.
    material = abs(change_pct) > 0.05 if change_pct is not None else (previous is None) != (current is None)

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
    if previous is not None and current is not None:
        change = safe_round(current - previous, 6)

    material = abs(change) > 0.10 if change is not None else (previous is None) != (current is None)

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
    """Diff bullish and bearish signal lists; a list absent from either snapshot is
    not compared (its keys are left out), and a null list counts as empty."""
    diff: dict[str, Any] = {}
    for side in ("bullish", "bearish"):
        key = f"{side}_signals"
        if key not in prev or key not in curr:
            continue
        before = set(prev[key] or [])
        after = set(curr[key] or [])
        diff[f"new_{side}"] = sorted(after - before)
        diff[f"removed_{side}"] = sorted(before - after)
    return diff


def _diff_hard_gates(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Diff the set of active hard-gate ids — what fired or cleared between snapshots."""
    previous = prev.get("hard_gate_ids")
    current = curr.get("hard_gate_ids")
    prev_gates = set(previous or [])
    curr_gates = set(current or [])
    return {
        "previous": previous if previous is None else sorted(prev_gates),
        "current": current if current is None else sorted(curr_gates),
        "new": sorted(curr_gates - prev_gates),
        "cleared": sorted(prev_gates - curr_gates),
        # Known-unknown (null) -> a known list is a change even when it is empty.
        "changed": prev_gates != curr_gates or (previous is None) != (current is None),
    }


def _build_material_changes(changes: dict[str, Any]) -> list[str]:
    """Build a human-readable list of material changes."""
    items: list[str] = []

    price = changes.get("price", {})
    if price.get("material") and price.get("change_pct") is not None:
        direction = "+" if price["change_pct"] >= 0 else ""
        items.append(f"price {direction}{price['change_pct'] * 100:.1f}%")

    score = changes.get("score", {})
    if score.get("material") and score.get("change") is not None:
        direction = "+" if score["change"] >= 0 else ""
        items.append(f"score {direction}{score['change']:.2f}")

    for _, changes_key, label in _SCALAR_FIELDS:
        change = changes.get(changes_key, {})
        if change.get("changed"):
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

    if previous_snapshot is not None:
        problems = validate_previous_snapshot(previous_snapshot, normalized_symbol)
        if problems:
            return {
                "meta": build_meta("what_changed", (perf_counter() - start) * 1000),
                "symbol": normalized_symbol,
                "error": True,
                "error_type": "invalid_snapshot",
                "message": f"previous_snapshot is not a usable {normalized_symbol} snapshot",
                "problems": problems,
            }

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
    summary = _build_summary(normalized_symbol, changes, material_changes)

    response["changes"] = changes
    response["material_changes"] = material_changes
    response["not_comparable"] = sorted(set(current_snapshot) - set(previous_snapshot))
    # The hash also moves on non-material data (a small price move, a newer bar);
    # fetch times, cache hits and ages are kept out of it.
    previous_hash = previous_snapshot.get("snapshot_hash")
    response["hash_changed"] = None if previous_hash is None else previous_hash != current_snapshot.get("snapshot_hash")
    response["summary"] = summary
    response["warnings"] = warnings or None

    return response
