"""Investor-style presets and default decision mode builders."""

from __future__ import annotations

from typing import Any

from stock_analysis.tools.analyze.action_zones import build_position_sizing_range
from stock_analysis.tools.analyze.dislocation_framework import (
    classify_dislocation_opportunity,
)
from stock_analysis.utils.helpers import format_compact_number

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
    account_size: float | None = None,
    risk_per_trade_pct: float | None = None,
    max_position_pct: float | None = None,
) -> dict[str, Any]:
    """Resolve a caller-supplied investor profile into a normalized config."""
    normalized_profile = (profile or "balanced").strip().lower()
    normalized_profile = _PROFILE_ALIASES.get(normalized_profile, normalized_profile)
    if normalized_profile not in _PROFILE_PRESETS:
        valid = ", ".join(sorted(_PROFILE_PRESETS))
        raise ValueError(f"Invalid profile '{profile}'. Expected one of: {valid}")

    if account_size is not None and account_size <= 0:
        raise ValueError("account_size must be greater than 0")
    if risk_per_trade_pct is not None and risk_per_trade_pct <= 0:
        raise ValueError("risk_per_trade_pct must be greater than 0")
    if max_position_pct is not None and max_position_pct <= 0:
        raise ValueError("max_position_pct must be greater than 0")

    preset = _PROFILE_PRESETS[normalized_profile]
    resolved_risk = risk_per_trade_pct if risk_per_trade_pct is not None else preset["risk_per_trade_pct"]
    resolved_max_position = (
        max_position_pct if max_position_pct is not None else preset["max_position_pct"]
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
        "account_size": round(account_size, 2) if account_size is not None else None,
        "account_size_status": "provided" if account_size is not None else "not_provided",
        "account_bucket": account_bucket,
        "small_account": bool(account_size is not None and account_size < 25_000),
        "risk_per_trade_pct": round(resolved_risk, 2),
        "risk_per_trade_decimal": round(resolved_risk / 100, 4),
        "max_position_pct": round(resolved_max_position, 2),
        "starter_position_pct": round(preset["starter_position_pct"], 2),
        "add_on_confirmation_pct": round(preset["add_on_confirmation_pct"], 2),
    }


def build_decision_modes(
    summary: dict[str, Any],
    verdict: dict[str, Any],
    policy_action: dict[str, Any],
    action_zones: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
    decision_context: dict[str, Any],
    news_summary: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    events_summary: dict[str, Any],
    account_size: float | None = None,
) -> dict[str, Any]:
    """Build default Core / Balanced / Speculative decision blocks."""
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
    shared_upgrade_conditions = list(policy_action.get("conditions_to_upgrade") or [])[:3]
    follow_up_conditions = []
    entry_timing = dip_assessment.get("entry_timing") or {}
    for item in entry_timing.get("wait_for", []) or []:
        if isinstance(item, dict):
            rationale = item.get("rationale")
            signal = item.get("signal")
            if rationale and signal:
                follow_up_conditions.append(f"{signal}: {rationale}")
            elif signal:
                follow_up_conditions.append(signal)
    if not follow_up_conditions:
        follow_up_conditions = shared_upgrade_conditions

    def _format_tag(tag: str) -> str:
        return tag.replace("_", " ")

    def _position_block(sizing: dict[str, Any], field: str) -> dict[str, Any]:
        use_pct = sizing.get(field)
        block: dict[str, Any] = {"pct": use_pct}
        if account_size is not None and use_pct is not None:
            dollars = round(account_size * use_pct / 100, 2)
            block["dollars"] = dollars
            block["shares"] = None
            price = summary.get("current_price")
            if price is not None and price > 0:
                block["shares"] = int(dollars / price)
        return block

    normalized_risk_label = "medium" if risk_label == "moderate" else risk_label
    stop_calculation = action_zones.get("stop_calculation")
    mode_profiles = {
        name: resolve_investor_profile(profile=name, account_size=account_size)
        for name in ("core", "balanced", "speculative")
    }
    mode_sizing = {
        name: build_position_sizing_range(
            current_price=summary.get("current_price"),
            stop_calculation=stop_calculation,
            regime_classification=normalized_risk_label,
            investor_profile=profile,
        )
        for name, profile in mode_profiles.items()
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

    if core_suitability == "fit":
        core_action = "build_position"
    elif core_suitability == "caution":
        core_action = "not_core_yet"
    elif core_suitability == "needs_more_data":
        core_action = "needs_more_data"
    else:
        core_action = "avoid"
    if is_dislocation_candidate and "price_broken_more_than_business" not in core_why:
        core_why.append("price_broken_more_than_business")

    balanced_suitability: str
    if mid_term_fit == "ok":
        balanced_suitability = "favorable" if verdict.get("tilt") == "bullish" else "workable"
    elif mid_term_fit == "caution":
        balanced_suitability = "caution"
    elif mid_term_fit == "unknown":
        balanced_suitability = "needs_more_data"
    else:
        balanced_suitability = "avoid"

    balanced_why = list(policy_action.get("rationale") or [])[:2]
    if business_quality and f"business_quality={business_quality}" not in balanced_why:
        balanced_why.append(f"business_quality={business_quality}")
    if risk_label and f"risk_regime={risk_label}" not in balanced_why:
        balanced_why.append(f"risk_regime={risk_label}")
    if is_dislocation_candidate:
        balanced_suitability = "starter_only"
        if "price_broken_more_than_business" not in balanced_why:
            balanced_why.append("price_broken_more_than_business")

    balanced_action = policy_action.get("mid_term")
    if is_dislocation_candidate and balanced_action in {
        "wait_for_entry",
        "speculative_small_position",
    }:
        balanced_action = "starter_position_only"

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
            "action": core_action,
            "why": core_why or None,
            "starter_position": _position_block(mode_sizing["core"], "starter_pct"),
            "max_position": _position_block(mode_sizing["core"], "max_pct"),
            "required_before_full_size": shared_upgrade_conditions or [],
        },
        "balanced": {
            "label": "Balanced",
            "description": _PROFILE_PRESETS["balanced"]["description"],
            "suitability": balanced_suitability,
            "action": balanced_action,
            "why": balanced_why or None,
            "starter_position": _position_block(mode_sizing["balanced"], "starter_pct"),
            "max_position": _position_block(mode_sizing["balanced"], "max_pct"),
            "add_on_only_if": shared_upgrade_conditions or [],
            "long_term_view": policy_action.get("long_term"),
        },
        "speculative": {
            "label": "Speculative",
            "description": _PROFILE_PRESETS["speculative"]["description"],
            "suitability": dip_quality or ("event_driven" if bullish_catalysts else "unclear"),
            "action": speculative_action,
            "starter_position": _position_block(mode_sizing["speculative"], "starter_pct"),
            "max_position": _position_block(mode_sizing["speculative"], "max_pct"),
            "add_on_only_if": follow_up_conditions or [],
            "upside_case": ", ".join(upside_case_parts) if upside_case_parts else None,
            "failure_case": ", ".join(failure_case_parts) if failure_case_parts else None,
            "next_90d_catalysts": [_format_tag(tag) for tag in bullish_catalysts[:3]],
            "risk_flags": [_format_tag(tag) for tag in bearish_catalysts[:3]] or None,
            "capital_at_risk": (
                format_compact_number(account_size) if account_size is not None else "account_size_not_provided"
            ),
        },
    }


def build_investor_views(
    summary: dict[str, Any],
    verdict: dict[str, Any],
    policy_action: dict[str, Any],
    action_zones: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
    decision_context: dict[str, Any],
    news_summary: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    events_summary: dict[str, Any],
    investor_profile: dict[str, Any],
) -> dict[str, Any]:
    """Backward-compatible wrapper for older internal callsites."""
    return build_decision_modes(
        summary=summary,
        verdict=verdict,
        policy_action=policy_action,
        action_zones=action_zones,
        dip_assessment=dip_assessment,
        decision_context=decision_context,
        news_summary=news_summary,
        fundamentals_summary=fundamentals_summary,
        events_summary=events_summary,
        account_size=investor_profile.get("account_size"),
    )
