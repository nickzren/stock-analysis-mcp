"""Frozen copies of the 1.1.0 snapshot producers (commit 1006647), so WP7's legacy
tests compare against what analyze and detect_changes really returned before 2.0.0:
``legacy_compact_snapshot`` is analyze's ``watchlist_snapshot`` and
``legacy_enriched_snapshot`` detect_changes' ``current_snapshot``. Do not update."""

import hashlib
from typing import Any

from stock_analysis.utils.normalize import (
    _compute_snapshot_as_of_date,
    canonical_dumps,
    normalize_for_watchlist_diff,
)


def legacy_compact_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
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
    decision_card = normalized.get("decision_card", {})
    hard_gates = decision_card.get("hard_gates", {}) if isinstance(decision_card, dict) else {}
    hard_gate_blocking = hard_gates.get("blocking", []) if isinstance(hard_gates, dict) else []
    hard_gate_ids: list[str] = sorted(
        str(g["id"]) for g in hard_gate_blocking
        if isinstance(g, dict) and g.get("id")
    )

    # Compute snapshot_as_of_date (max of all component dates)
    snapshot_as_of_date = _compute_snapshot_as_of_date(normalized)

    # Build the core snapshot data (without hash - we'll add it after)
    # Include snapshot_version in hashed content so version changes invalidate hash
    snapshot_data = {
        "snapshot_version": "1.1.0",
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
        # Decision card (compact action surface — diffs surface action / gate changes)
        "decision_action_now": decision_card.get("action_now") if isinstance(decision_card, dict) else None,
        "hard_gate_ids": hard_gate_ids,
        # Key risk
        "risk_regime": risk.get("risk_regime", {}).get("classification"),
        "volatility_ann": risk.get("annualized_volatility", risk.get("volatility_ann")),
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


def legacy_enriched_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    """Extract an enriched snapshot from a full analysis result.

    Combines the watchlist_snapshot with additional fields needed for
    diffing: zone, horizon fit, and signal lists.
    """
    snapshot = legacy_compact_snapshot(result)

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
