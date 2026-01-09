"""Normalization utilities for diff-stable watchlist snapshots.

This module provides functions to produce deterministic, diff-stable JSON
output for watchlist comparisons. The key insight is that raw output contains
high-churn fields (timestamps, durations, float noise) that create spurious
diffs when comparing analyses over time.

The normalization contract:
1. Key ordering: sorted at every level
2. Null vs empty: lists always [], dicts always {}, scalars null
3. Arrays: set-like lists sorted, ranked lists preserve order with tie-breaks
4. Timestamps: normalized to date-only where time doesn't matter
5. Float precision: use existing display strings where available
6. Money fields: coerced to int to avoid .0 noise
7. NaN/inf sanitization: replaced with null for JSON safety
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime
from typing import Any, Callable

# Snapshot format version - bump when normalization logic changes
SNAPSHOT_VERSION = "1.0.0"


def canonical_dumps(obj: Any) -> str:
    """Produce canonical JSON string with sorted keys and minimal separators.

    Uses allow_nan=False to fail fast if NaN/inf values slip through
    sanitization. This ensures JSON validity across all parsers.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


# List normalization rules: path -> behavior
# "set": sort lexicographically (order not semantic)
# "ranked": preserve order, stable tie-break by id field
# "ordered": preserve order exactly (semantic meaning)
LIST_NORMALIZATION_RULES: dict[tuple[str, ...], str] = {
    # Set-like lists (sort)
    ("signals", "bullish"): "set",
    ("signals", "bearish"): "set",
    ("signals", "neutral"): "set",
    ("action_zones", "zone_warnings"): "set",
    ("data_quality", "missing_critical"): "set",
    ("data_quality", "data_gaps"): "set",
    ("data_quality", "staleness_warnings"): "set",
    ("data_quality", "warnings"): "set",
    ("relative_performance", "warnings"): "set",
    ("decision_context", "news", "headline_triggers", "bullish"): "set",
    ("decision_context", "news", "headline_triggers", "bearish"): "set",
    ("decision_context", "thesis_checkpoints", "review_triggers"): "set",
    ("policy_action", "rationale"): "set",
    ("policy_action", "conditions_to_upgrade"): "set",
    ("policy_action", "conditions_to_downgrade"): "set",
    # Ranked lists (preserve order, tie-break)
    ("decision_context", "top_triggers"): "ranked",
    ("verdict", "pros"): "ranked",
    ("verdict", "cons"): "ranked",
    # Ordered by (horizon, gate)
    ("decision_context", "horizon_drivers"): "ordered_by_horizon_gate",
    # Object lists sorted by id
    ("decision_context", "fundamentals", "bullish_if"): "sorted_by_id",
    ("decision_context", "fundamentals", "bearish_if"): "sorted_by_id",
    ("decision_context", "valuation", "bullish_if"): "sorted_by_id",
    ("decision_context", "valuation", "bearish_if"): "sorted_by_id",
    ("decision_context", "risk", "bullish_if"): "sorted_by_id",
    ("decision_context", "risk", "bearish_if"): "sorted_by_id",
    ("decision_context", "technicals", "bullish_if"): "sorted_by_id",
    ("decision_context", "technicals", "bearish_if"): "sorted_by_id",
    ("dip_assessment", "dip_confidence", "missing"): "set",
}

# Paths where null should become [] for stability
NULL_TO_EMPTY_LIST_PATHS: set[tuple[str, ...]] = {
    ("signals", "bullish"),
    ("signals", "bearish"),
    ("signals", "neutral"),
    ("action_zones", "zone_warnings"),
    ("data_quality", "missing_critical"),
    ("data_quality", "data_gaps"),
    ("data_quality", "staleness_warnings"),
    ("data_quality", "tool_failures"),
    ("data_quality", "warnings"),
    ("relative_performance", "warnings"),
    ("market_context", "sanity_warnings"),
    ("policy_action", "rationale"),
    ("policy_action", "conditions_to_upgrade"),
    ("policy_action", "conditions_to_downgrade"),
    ("verdict", "pros"),
    ("verdict", "cons"),
    ("decision_context", "horizon_drivers"),
    ("decision_context", "top_triggers"),
    ("dip_assessment", "dip_confidence", "missing"),
}

# Money fields to coerce to int (avoid .0 noise)
MONEY_FIELD_PATHS: list[tuple[str, ...]] = [
    ("summary", "market_cap"),
    ("fundamentals_summary", "burn_metrics", "liquidity"),
    ("fundamentals_summary", "burn_metrics", "quarterly_fcf_burn"),
    ("fundamentals_summary", "burn_metrics", "quarterly_ocf_burn"),
    ("fundamentals_summary", "cash_flow", "free_cash_flow_ttm"),
]

# Timestamp paths to normalize to date-only
TIMESTAMP_PATHS: list[tuple[str, ...]] = [
    ("data_provenance", "price", "as_of"),
    ("data_provenance", "fundamentals", "as_of"),
    ("data_provenance", "news", "as_of"),
    ("data_provenance", "events", "as_of"),
    ("data_provenance", "risk", "as_of"),
]


def normalize_for_watchlist_diff(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize raw analysis output for diff-stable watchlist comparisons.

    This produces a deterministic view that minimizes spurious diffs caused by:
    - Runtime fields (duration_ms, tool_timings)
    - Timestamp precision (convert to date-only)
    - Null vs [] inconsistency
    - Unsorted set-like lists
    - Float precision noise
    - NaN/inf values from yfinance

    Args:
        raw: Raw analyze_stock output

    Returns:
        Normalized dict suitable for canonical JSON serialization
    """
    data = copy.deepcopy(raw)

    # 0) Sanitize NaN/inf values FIRST (critical for JSON safety)
    data = _sanitize_nan_inf(data)

    # 1) Remove high-churn runtime fields
    _delete_path(data, ("meta", "duration_ms"))
    _delete_path(data, ("data_quality", "tool_timings"))

    # 2) Normalize timestamps to dates (YYYY-MM-DD)
    for path in TIMESTAMP_PATHS:
        _coerce_iso_to_date(data, path, out_key="as_of_date")
        _delete_path(data, path)  # Remove original timestamp

    # Normalize component_freshness timestamps
    _normalize_component_freshness(data)

    # 3) Enforce "lists are never null"
    _coerce_null_lists_to_empty(data, NULL_TO_EMPTY_LIST_PATHS)

    # 4) Apply list normalization rules
    for path, rule in LIST_NORMALIZATION_RULES.items():
        if rule == "set":
            _sort_string_list(data, path)
        elif rule == "sorted_by_id":
            _sort_object_list(data, path, key=lambda x: str(x.get("id", "")))
        elif rule == "ordered_by_horizon_gate":
            _sort_object_list(
                data, path,
                key=lambda x: (x.get("horizon", ""), x.get("gate", ""))
            )
        elif rule == "ranked":
            _stable_tie_break_ranked_list(data, path)
        # "ordered" means keep as-is

    # 5) Coerce money fields to int
    _coerce_money_fields_to_int(data)

    return data


# ---------------- Path helpers ----------------

def _delete_path(root: dict[str, Any], path: tuple[str, ...]) -> None:
    """Delete a nested key if it exists."""
    parent = root
    for k in path[:-1]:
        if not isinstance(parent, dict) or k not in parent:
            return
        parent = parent[k]
    if isinstance(parent, dict):
        parent.pop(path[-1], None)


def _get_path(root: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Get value at nested path, or None if not found."""
    cur: Any = root
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set value at nested path, creating intermediate dicts as needed."""
    cur: Any = root
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


# ---------------- Normalization helpers ----------------

def _is_nan_or_inf(x: Any) -> bool:
    """Check if value is NaN or inf, handling numpy types safely."""
    try:
        # Works for float, numpy.float64, etc.
        return math.isnan(x) or math.isinf(x)
    except (TypeError, ValueError):
        # Not a numeric type that supports isnan/isinf
        return False


def _is_negative_zero(x: Any) -> bool:
    """Check if value is -0.0 (which creates diff noise)."""
    try:
        return x == 0.0 and math.copysign(1.0, x) < 0
    except (TypeError, ValueError):
        return False


def _sanitize_nan_inf(obj: Any) -> Any:
    """Recursively replace NaN, inf, -inf with None and -0.0 with 0.0.

    This is critical because:
    1. JSON spec doesn't support NaN/inf (Python's json uses allow_nan=True by default)
    2. Some JSON parsers (JavaScript) will choke on these values
    3. yfinance can return NaN for missing data
    4. -0.0 creates spurious diffs (mathematically equal but different JSON repr)
    """
    if isinstance(obj, dict):
        return {k: _sanitize_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_nan_inf(item) for item in obj]
    elif _is_nan_or_inf(obj):
        return None
    elif _is_negative_zero(obj):
        return 0.0
    return obj


def _coerce_iso_to_date(
    root: dict[str, Any],
    path: tuple[str, ...],
    out_key: str,
) -> None:
    """Convert ISO timestamp to date-only string at sibling key."""
    value = _get_path(root, path)
    if not isinstance(value, str):
        return
    try:
        # Accept "2026-01-07T02:03:49.220783Z"
        iso = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        out_date = dt.date().isoformat()
    except Exception:
        return
    parent_path = path[:-1]
    parent = _get_path(root, parent_path)
    if isinstance(parent, dict):
        parent[out_key] = out_date


def _normalize_component_freshness(root: dict[str, Any]) -> None:
    """Normalize component_freshness: as_of -> as_of_date, drop age_hours."""
    freshness = _get_path(root, ("data_quality", "component_freshness"))
    if not isinstance(freshness, dict):
        return
    for _component, block in list(freshness.items()):
        if not isinstance(block, dict):
            continue
        # Convert as_of to as_of_date and drop age_hours
        as_of = block.get("as_of")
        if isinstance(as_of, str):
            try:
                dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                block["as_of_date"] = dt.date().isoformat()
            except Exception:
                pass
        block.pop("as_of", None)
        block.pop("age_hours", None)
        # Ensure stale exists
        if "stale" not in block:
            block["stale"] = False


def _coerce_null_lists_to_empty(
    root: dict[str, Any],
    paths: set[tuple[str, ...]],
) -> None:
    """Convert null lists to [] for diff stability."""
    for path in paths:
        val = _get_path(root, path)
        if val is None:
            _set_path(root, path, [])


def _sort_string_list(root: dict[str, Any], path: tuple[str, ...]) -> None:
    """Sort a list of strings lexicographically."""
    val = _get_path(root, path)
    if not isinstance(val, list):
        return
    # Handle both string lists and object lists with string keys
    if all(isinstance(x, str) for x in val):
        _set_path(root, path, sorted(val))


def _sort_object_list(
    root: dict[str, Any],
    path: tuple[str, ...],
    key: Callable[[dict[str, Any]], Any],
) -> None:
    """Sort a list of objects by a key function."""
    val = _get_path(root, path)
    if not isinstance(val, list) or not val:
        return
    if not all(isinstance(x, dict) for x in val):
        return
    _set_path(root, path, sorted(val, key=key))


def _stable_tie_break_ranked_list(
    root: dict[str, Any],
    path: tuple[str, ...],
) -> None:
    """Preserve ranking but tie-break equal items deterministically.

    For ranked lists like top_triggers, items are grouped by score_delta,
    then ties within each group are broken by (category, id) for stability.
    This ensures deterministic ordering even when multiple triggers have
    the same score contribution.
    """
    items = _get_path(root, path)
    if not isinstance(items, list) or not items:
        return

    # For ranked lists (like pros/cons), we preserve order but stabilize ties
    # Tie-break chain: score_delta -> category -> id
    out: list[Any] = []
    i = 0
    while i < len(items):
        cur = items[i]
        if not isinstance(cur, dict):
            out.append(cur)
            i += 1
            continue

        # Find tie group (items with same score_delta or all items if no score_delta)
        delta = cur.get("score_delta")
        tie_group = [cur]
        j = i + 1
        while j < len(items):
            next_item = items[j]
            if not isinstance(next_item, dict):
                break
            if delta is not None and next_item.get("score_delta") != delta:
                break
            if delta is None:
                # For lists without score_delta (pros/cons), just add all
                pass
            tie_group.append(next_item)
            j += 1

        # Sort tie group with full tie-break chain: category -> id
        # This handles cases where multiple triggers have same score_delta
        if len(tie_group) > 1 and delta is not None:
            tie_group.sort(
                key=lambda x: (
                    str(x.get("category", "")),
                    str(x.get("id", "")),
                )
            )

        out.extend(tie_group)
        i = j if delta is not None else i + 1

    _set_path(root, path, out)


def _coerce_money_fields_to_int(root: dict[str, Any]) -> None:
    """Convert dollar amounts from float to int to avoid .0 noise."""
    for path in MONEY_FIELD_PATHS:
        val = _get_path(root, path)
        if isinstance(val, float) and math.isfinite(val):
            _set_path(root, path, int(round(val)))


def _compute_snapshot_as_of_date(normalized: dict[str, Any]) -> str | None:
    """Compute the latest data date across all components.

    This provides a single "time anchor" for the snapshot, which is the max
    of all known component freshness dates.
    """
    dates: list[str] = []

    # Collect all as_of_date values from component_freshness
    freshness = _get_path(normalized, ("data_quality", "component_freshness"))
    if isinstance(freshness, dict):
        for _component, block in freshness.items():
            if isinstance(block, dict):
                as_of_date = block.get("as_of_date")
                if isinstance(as_of_date, str):
                    dates.append(as_of_date)

    # Collect from data_provenance (already normalized to as_of_date)
    provenance = normalized.get("data_provenance", {})
    if isinstance(provenance, dict):
        for _component, block in provenance.items():
            if isinstance(block, dict):
                as_of_date = block.get("as_of_date")
                if isinstance(as_of_date, str):
                    dates.append(as_of_date)

    # Return max date (lexicographic sort works for ISO dates)
    return max(dates) if dates else None


def build_watchlist_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Build a watchlist snapshot with key summary fields.

    This is a reduced view optimized for watchlist displays and diffs,
    containing only the most relevant decision fields.

    The snapshot includes:
    - snapshot_version: Format version for compatibility checking
    - snapshot_hash: SHA-256 hash of canonical JSON for change detection
    - snapshot_as_of_date: Latest data date across all components (time anchor)

    Args:
        raw: Raw analyze_stock output

    Returns:
        Compact snapshot dict with version and hash
    """
    normalized = normalize_for_watchlist_diff(raw)

    # Extract key fields for watchlist
    summary = normalized.get("summary", {})
    verdict = normalized.get("verdict", {})
    policy = normalized.get("policy_action", {})
    events = normalized.get("events_summary", {})
    risk = normalized.get("risk_summary", {})

    # Compute snapshot_as_of_date (max of all component dates)
    snapshot_as_of_date = _compute_snapshot_as_of_date(normalized)

    # Build the core snapshot data (without hash - we'll add it after)
    # Include snapshot_version in hashed content so version changes invalidate hash
    snapshot_data = {
        "snapshot_version": SNAPSHOT_VERSION,
        "symbol": normalized.get("symbol"),
        "price": summary.get("current_price"),
        "market_cap": summary.get("market_cap"),
        "sector": summary.get("sector"),
        # Verdict
        "tilt": verdict.get("tilt"),
        "confidence": verdict.get("confidence"),
        "score": verdict.get("score"),
        # Policy action (primary decision output)
        "action_mid_term": policy.get("mid_term"),
        "action_long_term": policy.get("long_term"),
        "valuation_gate": policy.get("valuation_gate"),
        "is_unprofitable": policy.get("is_unprofitable"),
        # Key risk
        "risk_regime": risk.get("risk_regime", {}).get("classification"),
        "volatility_ann": risk.get("volatility_ann"),
        # Next catalyst
        "next_catalyst": events.get("next_catalyst"),
        # Data freshness - single time anchor for diffs
        "snapshot_as_of_date": snapshot_as_of_date,
    }

    # Compute hash of canonical JSON for change detection
    # Hash includes snapshot_version so version bumps change the hash
    canonical_json = canonical_dumps(snapshot_data)
    snapshot_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]

    return {
        **snapshot_data,
        "snapshot_hash": snapshot_hash,
    }
