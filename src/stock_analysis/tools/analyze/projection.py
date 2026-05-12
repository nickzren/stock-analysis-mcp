"""Projection helpers for token-efficient analyze responses."""

from typing import Any

ANALYZE_DETAIL_LEVELS = frozenset({"decision", "standard", "full"})

SUMMARY_KEYS = ("name", "current_price", "sector", "industry", "market_cap", "currency")
VERDICT_DECISION_KEYS = ("score", "tilt", "confidence", "components", "pros", "cons")
DATA_QUALITY_DECISION_KEYS = ("completeness", "missing_critical", "warnings")
ACTION_ZONES_STANDARD_KEYS = (
    "current_zone",
    "levels",
    "stop_calculation",
    "valuation_assessment",
    "position_sizing_range",
)
EVENTS_SUMMARY_STANDARD_KEYS = ("next_catalyst", "days_to_earnings")
NEWS_SUMMARY_STANDARD_KEYS = (
    "article_count",
    "sentiment",
    "headlines",
    "catalyst_intelligence",
    "summary",
)
RISK_SUMMARY_STANDARD_KEYS = (
    "risk_regime",
    "annualized_volatility",
    "beta",
    "max_drawdown_1y",
    "atr_pct",
    "summary",
)
FUNDAMENTALS_SUMMARY_STANDARD_KEYS = (
    "valuation",
    "growth",
    "profitability",
    "health",
    "cash_flow",
    "burn_metrics",
    "analyst_estimates",
    "summary",
)


def normalize_analyze_detail(detail: str) -> str:
    """Normalize and validate an analyze detail level."""
    normalized = (detail or "").strip().lower()
    if normalized not in ANALYZE_DETAIL_LEVELS:
        allowed = ", ".join(sorted(ANALYZE_DETAIL_LEVELS))
        raise ValueError(f"Invalid detail '{detail}'. Expected one of: {allowed}.")
    return normalized


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Return a shallow allowlist projection, preserving configured key order."""
    return {key: source.get(key) for key in keys if key in source}


def _project_data_quality(result: dict[str, Any]) -> dict[str, Any]:
    """Return compact data-quality fields without dropping error diagnostics."""
    data_quality = result.get("data_quality") or {}
    projected = _pick(data_quality, DATA_QUALITY_DECISION_KEYS)
    tool_failures = data_quality.get("tool_failures")
    if tool_failures or result.get("error"):
        projected["tool_failures"] = tool_failures or []
    return projected


def _base_decision_projection(result: dict[str, Any]) -> dict[str, Any]:
    projected: dict[str, Any] = {
        "meta": result.get("meta"),
        "symbol": result.get("symbol"),
        "summary": _pick(result.get("summary") or {}, SUMMARY_KEYS),
        "decision_card": result.get("decision_card"),
        "verdict": _pick(result.get("verdict") or {}, VERDICT_DECISION_KEYS),
        "data_quality": _project_data_quality(result),
        "watchlist_snapshot": result.get("watchlist_snapshot"),
    }
    for key in ("error", "error_type", "message"):
        if key in result:
            projected[key] = result[key]
    return projected


def _standard_projection(result: dict[str, Any]) -> dict[str, Any]:
    projected = _base_decision_projection(result)
    projected.update({
        "executive_summary": result.get("executive_summary"),
        "section_summaries": result.get("section_summaries"),
        "action_zones": _pick(result.get("action_zones") or {}, ACTION_ZONES_STANDARD_KEYS),
        "events_summary": _pick(result.get("events_summary") or {}, EVENTS_SUMMARY_STANDARD_KEYS),
        "news_summary": _pick(result.get("news_summary") or {}, NEWS_SUMMARY_STANDARD_KEYS),
        "risk_summary": _pick(result.get("risk_summary") or {}, RISK_SUMMARY_STANDARD_KEYS),
        "fundamentals_summary": _pick(
            result.get("fundamentals_summary") or {},
            FUNDAMENTALS_SUMMARY_STANDARD_KEYS,
        ),
        "analyst_coverage": result.get("analyst_coverage"),
        "short_interest": result.get("short_interest"),
        "ownership": result.get("ownership"),
        "sector_comparison": result.get("sector_comparison"),
        "relative_performance": result.get("relative_performance"),
    })
    return projected


def project_analyze_result(result: dict[str, Any], detail: str) -> dict[str, Any]:
    """Project a full analyze result into the requested response detail level."""
    normalized = normalize_analyze_detail(detail)
    if normalized == "full":
        return result
    if normalized == "decision":
        return _base_decision_projection(result)
    return _standard_projection(result)
