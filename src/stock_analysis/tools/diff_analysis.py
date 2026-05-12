"""What-changed diff analysis between stock snapshots."""

from time import perf_counter
from typing import Any

from stock_analysis.tools.analyze import analyze_stock
from stock_analysis.utils.helpers import safe_float, safe_round
from stock_analysis.utils.provenance import build_meta


def _extract_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Extract an enriched snapshot from a full analysis result.

    Combines the watchlist_snapshot with additional fields needed for
    diffing: zone, horizon fit, and signal lists.
    """
    snapshot = dict(result.get("watchlist_snapshot", {}))

    action_zones = result.get("action_zones", {})
    snapshot["zone"] = action_zones.get("current_zone")

    verdict = result.get("verdict", {})
    horizon_fit = verdict.get("horizon_fit", {})
    snapshot["mid_term_fit"] = horizon_fit.get("mid_term")
    snapshot["long_term_fit"] = horizon_fit.get("long_term")

    signals = result.get("signals", {})
    snapshot["bullish_signals"] = sorted(signals.get("bullish", []))
    snapshot["bearish_signals"] = sorted(signals.get("bearish", []))

    return snapshot


def _diff_scalar(
    prev: dict[str, Any],
    curr: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Diff a simple scalar field, returning previous/current/changed."""
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

    material = abs(change_pct) > 0.05 if change_pct is not None else False

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

    material = abs(change) > 0.10 if change is not None else False

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
    """Diff bullish and bearish signal lists."""
    prev_bullish = set(prev.get("bullish_signals", []))
    curr_bullish = set(curr.get("bullish_signals", []))
    prev_bearish = set(prev.get("bearish_signals", []))
    curr_bearish = set(curr.get("bearish_signals", []))

    return {
        "new_bullish": sorted(curr_bullish - prev_bullish),
        "removed_bullish": sorted(prev_bullish - curr_bullish),
        "new_bearish": sorted(curr_bearish - prev_bearish),
        "removed_bearish": sorted(prev_bearish - curr_bearish),
    }


def _diff_hard_gates(
    prev: dict[str, Any],
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Diff the set of active hard-gate ids — what fired or cleared between snapshots."""
    prev_gates = set(prev.get("hard_gate_ids") or [])
    curr_gates = set(curr.get("hard_gate_ids") or [])
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

    score = changes.get("score", {})
    if score.get("material") and score.get("change") is not None:
        direction = "+" if score["change"] >= 0 else ""
        items.append(f"score {direction}{score['change']:.2f}")

    tilt = changes.get("tilt", {})
    if tilt.get("changed"):
        items.append(f"tilt: {tilt['previous']} -> {tilt['current']}")

    risk = changes.get("risk_regime", {})
    if risk.get("changed"):
        items.append(f"risk_regime: {risk['previous']} -> {risk['current']}")

    zone = changes.get("action_zone", {})
    if zone.get("changed"):
        items.append(f"zone: {zone['previous']} -> {zone['current']}")

    action_now = changes.get("decision_action_now", {})
    if action_now.get("changed"):
        items.append(
            f"action_now: {action_now['previous']} -> {action_now['current']}"
        )

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

    # Fallback if only signal changes are material
    if len(parts) == 1:
        parts.append(f"{len(material_changes)} signal change(s) detected.")

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

    current_snapshot = _extract_snapshot(result)
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

    changes: dict[str, Any] = {
        "price": _diff_price(previous_snapshot, current_snapshot),
        "score": _diff_score(previous_snapshot, current_snapshot),
        "tilt": _diff_scalar(previous_snapshot, current_snapshot, "tilt"),
        "risk_regime": _diff_scalar(previous_snapshot, current_snapshot, "risk_regime"),
        "action_zone": _diff_scalar(previous_snapshot, current_snapshot, "zone"),
        "decision_action_now": _diff_scalar(
            previous_snapshot, current_snapshot, "decision_action_now"
        ),
        "hard_gates": _diff_hard_gates(previous_snapshot, current_snapshot),
        "horizon_fit": {
            "mid_term": _diff_scalar(previous_snapshot, current_snapshot, "mid_term_fit"),
            "long_term": _diff_scalar(previous_snapshot, current_snapshot, "long_term_fit"),
        },
        "signals": _diff_signals(previous_snapshot, current_snapshot),
    }

    material_changes = _build_material_changes(changes)
    summary = _build_summary(normalized_symbol, changes, material_changes)

    response["changes"] = changes
    response["material_changes"] = material_changes
    response["summary"] = summary
    response["warnings"] = warnings or None

    return response
