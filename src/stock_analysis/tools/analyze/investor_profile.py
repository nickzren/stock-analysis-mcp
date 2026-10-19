"""Investor-style presets and default decision mode builders."""

from __future__ import annotations

from typing import Any

from stock_analysis.tools.analyze.decision import (
    CORE,
    LONG_TERM,
    MID_TERM,
    core_action,
    entry_allowed,
    most_restrictive,
)
from stock_analysis.tools.analyze.dislocation_framework import (
    classify_dislocation_opportunity,
)
from stock_analysis.utils.sizing_inputs import normalize_positive_number
from stock_analysis.utils.sizing_output import cap_sizing_scenario

_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "core": {
        "label": "Core compounder",
        "description": "Quality-first profile with tighter risk budgets and larger preference for durable businesses.",
        "risk_per_trade_pct": 0.5,
        "max_position_pct": 8.0,
        "starter_position_pct": 2.0,
        "add_on_confirmation_pct": 4.0,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Default profile that blends quality, valuation, and tactical entry discipline.",
        "risk_per_trade_pct": 1.0,
        "max_position_pct": 10.0,
        "starter_position_pct": 2.5,
        "add_on_confirmation_pct": 5.0,
    },
    "speculative": {
        "label": "Speculative asymmetric",
        "description": "Small-starter profile for solo investors willing to risk limited principal for higher upside.",
        "risk_per_trade_pct": 1.5,
        "max_position_pct": 6.0,
        "starter_position_pct": 1.0,
        "add_on_confirmation_pct": 2.5,
    },
}

_PROFILE_ALIASES = {
    "aggressive": "speculative",
    "asymmetric": "speculative",
    "default": "balanced",
    "quality": "core",
    "small_account": "speculative",
    "solo": "balanced",
}


def resolve_investor_profile(
    profile: str | None = None,
    account_size: float | str | None = None,
    risk_per_trade_pct: float | str | None = None,
    max_position_pct: float | str | None = None,
) -> dict[str, Any]:
    """Resolve a caller-supplied investor profile into a normalized config."""
    normalized_profile = (profile or "balanced").strip().lower()
    normalized_profile = _PROFILE_ALIASES.get(normalized_profile, normalized_profile)
    if normalized_profile not in _PROFILE_PRESETS:
        valid = ", ".join(sorted(_PROFILE_PRESETS))
        raise ValueError(f"Invalid profile '{profile}'. Expected one of: {valid}")

    preset = _PROFILE_PRESETS[normalized_profile]
    account_size = (
        normalize_positive_number(account_size, field="account_size")
        if account_size is not None else None
    )
    resolved_risk = (
        normalize_positive_number(risk_per_trade_pct, field="risk_per_trade_pct", maximum=100)
        if risk_per_trade_pct is not None else preset["risk_per_trade_pct"]
    )
    resolved_max_position = (
        normalize_positive_number(max_position_pct, field="max_position_pct", maximum=100)
        if max_position_pct is not None else preset["max_position_pct"]
    )

    account_bucket = "unknown"
    if account_size is not None:
        if account_size < 5_000:
            account_bucket = "micro"
        elif account_size < 25_000:
            account_bucket = "small"
        elif account_size < 100_000:
            account_bucket = "mid"
        else:
            account_bucket = "large"

    return {
        "profile": normalized_profile,
        "label": preset["label"],
        "description": preset["description"],
        "account_size": account_size,
        "account_size_status": "provided" if account_size is not None else "not_provided",
        "account_bucket": account_bucket,
        "small_account": bool(account_size is not None and account_size < 25_000),
        "risk_per_trade_pct": resolved_risk,
        "risk_per_trade_decimal": resolved_risk / 100,
        "max_position_pct": resolved_max_position,
        "limit_origins": {
            "risk_per_trade_pct": "caller" if risk_per_trade_pct is not None else "profile",
            "max_position_pct": "caller" if max_position_pct is not None else "profile",
        },
        "starter_position_pct": round(preset["starter_position_pct"], 2),
        "add_on_confirmation_pct": round(preset["add_on_confirmation_pct"], 2),
    }


# The speculative mode's own codes in the new-money vocabulary (plan §3 [D-3]); new_money caps them after.
_SPECULATIVE_ACTIONS = {"starter_then_add": "starter", "starter_position_only": "starter", "wait": "wait",
                        "watch": "wait"}


def _capped(own: str, action: str, cap: str) -> dict[str, str]:
    """A mode's action; capped_by names the canonical decision that tightened its own code (plan §7)."""
    return {"action": action, **({"capped_by": f"{cap}_{action}"} if action != own else {})}


def build_decision_modes(
    summary: dict[str, Any],
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
    decision_context: dict[str, Any],
    news_summary: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    events_summary: dict[str, Any],
    account_size: float | None = None,
    *,
    decision: dict[str, Any],
    conditions: dict[str, Any],
    why: list[str],
    canonical_sizing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build default Core / Balanced / Speculative decision blocks in the new-money vocabulary (plan §3 [D-3]):
    balanced is new_money; speculative its own code capped by new_money; core CORE[long_term_fit] capped by
    long_term.new_money. Mid-term sizes derive from the card's budget; core has no long-term allocation.
    The conditionals mirror the card's lists: core long-term, balanced and speculative mid-term."""
    horizon_fit = verdict.get("horizon_fit", {})
    long_term_fit = horizon_fit.get("long_term")
    long_term_gates = horizon_fit.get("long_term_gates") or []
    mid_term_fit = horizon_fit.get("mid_term")
    decomposed = verdict.get("decomposed", {})
    business_quality = decomposed.get("business_quality")
    risk_label = decomposed.get("risk")
    next_catalyst = events_summary.get("next_catalyst") or {}
    catalyst_intelligence = news_summary.get("catalyst_intelligence") or {}
    bullish_catalysts = [item.get("tag") for item in catalyst_intelligence.get("bullish", []) if item.get("tag")]
    bearish_catalysts = [item.get("tag") for item in catalyst_intelligence.get("bearish", []) if item.get("tag")]
    dip_assessment = dip_assessment or {}
    dip_info = dip_assessment.get("assessment") or {}
    dip_recommendation = dip_info.get("recommendation")
    dip_quality = dip_info.get("dip_quality")
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}
    dilution_analysis = burn_metrics.get("dilution_analysis") or {}

    def _format_tag(tag: str) -> str:
        return tag.replace("_", " ")

    def _position_block(mode: str, scenario: str) -> dict[str, Any] | None:
        if canonical_sizing is None:
            return None
        field = "starter_position_pct" if scenario == "starter" else "max_position_pct"
        capped = cap_sizing_scenario(
            canonical_sizing, scenario, _PROFILE_PRESETS[mode][field], f"{mode}_{scenario}_cap")
        if capped is None:
            return None
        return {
            **capped,
            "horizon": MID_TERM,
            "pct": capped["allocation_cap_pct"],
            "dollars": capped["position_dollars"],
            "shares": capped["whole_shares"],
        }

    dislocation_signals = classify_dislocation_opportunity(
        verdict=verdict,
        action_zones=action_zones,
        fundamentals_summary=fundamentals_summary,
        dip_assessment=dip_assessment,
    )
    is_dislocation_candidate = dislocation_signals["is_dislocation_candidate"]

    core_suitability: str
    if long_term_fit == "ok":
        core_suitability = "fit"
    elif long_term_fit == "caution":
        core_suitability = "caution"
    elif long_term_fit == "unknown":
        core_suitability = "needs_more_data"
    else:
        core_suitability = "avoid"

    core_why: list[str] = []
    if business_quality:
        core_why.append(f"business_quality={business_quality}")
    if risk_label:
        core_why.append(f"risk_regime={risk_label}")
    if long_term_gates:
        core_why.extend(_format_tag(gate) for gate in long_term_gates[:2])

    if is_dislocation_candidate and "price_broken_more_than_business" not in core_why:
        core_why.append("price_broken_more_than_business")

    balanced_suitability: str
    if mid_term_fit == "ok" and verdict.get("tilt") is None:
        balanced_suitability = "needs_more_data"  # the policy action is insufficient_data (WP9)
    elif mid_term_fit == "ok":
        balanced_suitability = "favorable" if verdict.get("tilt") == "bullish" else "workable"
    elif mid_term_fit == "caution":
        balanced_suitability = "caution"
    elif mid_term_fit == "unknown":
        balanced_suitability = "needs_more_data"
    else:
        balanced_suitability = "avoid"

    new_money_action = decision["new_money"]["action"]
    long_term_action = decision[LONG_TERM]["new_money"]["action"]
    balanced_why = list(why)
    if business_quality:
        balanced_why.append(f"business_quality={business_quality}")
    if risk_label:
        balanced_why.append(f"risk_regime={risk_label}")
    if is_dislocation_candidate:
        if new_money_action == "starter":
            balanced_suitability = "starter_only"
        balanced_why.append("price_broken_more_than_business")

    speculative_action = "watch"
    if dip_recommendation in {"do_not_catch_falling_knife", "wait_for_better_setup"}:
        speculative_action = "wait"
    elif dip_recommendation in {"small_speculative_position", "cautious_accumulation"}:
        speculative_action = "starter_position_only"
    elif mid_term_fit == "ok" and verdict.get("tilt") == "bullish":
        speculative_action = "starter_then_add"
    elif verdict.get("tilt") == "neutral":
        speculative_action = "starter_position_only"
    if is_dislocation_candidate and speculative_action in {"wait", "watch"}:
        speculative_action = "starter_position_only"
    speculative_own = _SPECULATIVE_ACTIONS[speculative_action]
    speculative = _capped(speculative_own, most_restrictive((speculative_own, new_money_action)), "new_money")
    core = _capped(CORE[long_term_fit], core_action(long_term_fit, long_term_action), "long_term")
    entry = entry_allowed(new_money_action)

    def _sizes(mode: str, action: str) -> dict[str, Any]:
        """starter_position iff entry and the mode's action admits it; max_position only under buy (§0 #21)."""
        starter = entry and entry_allowed(action)
        full = action == "buy" and new_money_action == "buy"
        return {"starter_position": _position_block(mode, "starter") if starter else None,
                "max_position": _position_block(mode, "full") if full else None}

    speculative_sizes = _sizes("speculative", speculative["action"])

    upside_case_parts: list[str] = []
    if bullish_catalysts:
        upside_case_parts.append(", ".join(_format_tag(tag) for tag in bullish_catalysts[:3]))
    if next_catalyst.get("status") == "available":
        catalyst_type = next_catalyst.get("type", "catalyst")
        days_until = next_catalyst.get("days_until")
        if days_until is not None and days_until <= 90:
            upside_case_parts.append(f"{catalyst_type} within {days_until}d")
    if action_zones.get("valuation_assessment", {}).get("gate") == "attractive":
        upside_case_parts.append("valuation support")

    failure_case_parts: list[str] = []
    if long_term_gates:
        failure_case_parts.extend(_format_tag(gate) for gate in long_term_gates[:3])
    if bearish_catalysts:
        failure_case_parts.extend(_format_tag(tag) for tag in bearish_catalysts[:2])
    dilution_pct = dilution_analysis.get("dilution_if_raised_today")
    if dilution_pct is not None:
        failure_case_parts.append(f"potential dilution ~{dilution_pct * 100:.0f}%")

    return {
        "core": {
            "label": "Core",
            "description": _PROFILE_PRESETS["core"]["description"],
            "suitability": core_suitability,
            **core,
            "why": core_why or None,
            "starter_position": None,
            "max_position": None,
            "required_before_full_size": list(conditions[LONG_TERM]["buy_only_if"]),
        },
        "balanced": {
            "label": "Balanced",
            "description": _PROFILE_PRESETS["balanced"]["description"],
            "suitability": balanced_suitability,
            "action": new_money_action,
            "why": balanced_why or None,
            **_sizes("balanced", new_money_action),
            "add_on_only_if": list(conditions[MID_TERM]["add_only_if"]),
            "long_term_view": long_term_action,
        },
        "speculative": {
            "label": "Speculative",
            "description": _PROFILE_PRESETS["speculative"]["description"],
            "suitability": dip_quality or ("event_driven" if bullish_catalysts else "unclear"),
            **speculative,
            **speculative_sizes,
            "add_on_only_if": list(conditions[MID_TERM]["add_only_if"]),
            "upside_case": ", ".join(upside_case_parts) if upside_case_parts else None,
            "failure_case": ", ".join(failure_case_parts) if failure_case_parts else None,
            "next_90d_catalysts": [_format_tag(tag) for tag in bullish_catalysts[:3]],
            "risk_flags": [_format_tag(tag) for tag in bearish_catalysts[:3]] or None,
            "capital_at_risk": None,
        },
    }


def build_investor_views(
    summary: dict[str, Any],
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
    decision_context: dict[str, Any],
    news_summary: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    events_summary: dict[str, Any],
    investor_profile: dict[str, Any],
    *,
    decision: dict[str, Any],
    conditions: dict[str, Any],
    why: list[str],
    canonical_sizing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for older internal callsites."""
    return build_decision_modes(
        summary=summary,
        verdict=verdict,
        action_zones=action_zones,
        dip_assessment=dip_assessment,
        decision_context=decision_context,
        news_summary=news_summary,
        fundamentals_summary=fundamentals_summary,
        events_summary=events_summary,
        account_size=investor_profile.get("account_size"),
        decision=decision,
        conditions=conditions,
        why=why,
        canonical_sizing=canonical_sizing,
    )
