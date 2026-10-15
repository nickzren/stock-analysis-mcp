"""Executive summary and policy action builders."""

from typing import Any

from stock_analysis.tools.analyze.conditions import candidate_factors
from stock_analysis.tools.analyze.dislocation_framework import (
    classify_dislocation_opportunity,
)
from stock_analysis.tools.analyze.signals import VOLATILITY_REGIME_THRESHOLDS
from stock_analysis.utils.helpers import fcf_label_from_cashflow

_ACTION_HUMANIZE: dict[str, str] = {
    # Long-term actions
    "avoid_core_position": "avoid as a core holding",
    "hold_watch": "hold and watch",
    "accumulate": "accumulate",
    "strong_buy": "strong buy",
    # Mid-term actions
    "wait_for_entry": "wait for a better entry",
    "speculative_small_position": "small starter position",
    "consider_entry": "consider entry",
    "hold": "hold",
    "reduce": "reduce",
    "exit": "exit",
    "small_position_with_stops": "small starter position",
}

# Gate ordering (fixed order for diff stability)
_GATE_ORDER: list[str] = [
    "unprofitable",
    "negative_fcf",
    "severe_revenue_decline",
    "burn_metrics_missing",
    "extreme_risk",
    "low_runway_confidence",
    "high_debt",
]


def _humanize_action(action: str) -> str:
    """Convert internal action enum to investor-friendly wording."""
    return _ACTION_HUMANIZE.get(action, action.replace("_", " "))


def _humanize_gates_with_numbers(
    gates: list[str],
    rev_yoy: float | None,
    net_margin: float | None,
    quarterly_burn: float | None,
    runway_years: float | None,
    vol: float | None,
    dd: float | None,
) -> str:
    """
    Convert gate list to investor-friendly string with numeric anchors.

    Fixed order ensures diff stability. Embeds the actual numbers so investors
    see the "why" immediately.
    """
    # Sort gates by fixed order (unknown gates go to end)
    def gate_sort_key(g: str) -> int:
        try:
            return _GATE_ORDER.index(g)
        except ValueError:
            return len(_GATE_ORDER)

    sorted_gates = sorted(gates, key=gate_sort_key)

    # Build humanized gate strings with numbers
    humanized: list[str] = []
    for gate in sorted_gates[:4]:  # Max 4 gates
        if gate == "unprofitable":
            if net_margin is not None:
                humanized.append(f"unprofitable (margin {net_margin * 100:.0f}%)")
            else:
                humanized.append("unprofitable")
        elif gate == "negative_fcf":
            if quarterly_burn is not None and quarterly_burn > 0:
                burn_str = f"burning cash (~${quarterly_burn / 1_000_000:.0f}M/q"
                # Add runway if available (time dimension investors care about)
                if runway_years is not None:
                    burn_str += f"; runway ~{runway_years:.1f}y"
                burn_str += ")"
                humanized.append(burn_str)
            else:
                humanized.append("negative FCF")
        elif gate == "severe_revenue_decline":
            if rev_yoy is not None:
                humanized.append(f"revenue {rev_yoy * 100:+.0f}% YoY")
            else:
                humanized.append("severe revenue decline")
        elif gate == "extreme_risk":
            risk_parts = []
            if vol is not None:
                risk_parts.append(f"vol {vol * 100:.0f}%")
            if dd is not None:
                risk_parts.append(f"dd {dd * 100:.0f}%")
            if risk_parts:
                humanized.append(f"extreme risk ({', '.join(risk_parts)})")
            else:
                humanized.append("extreme risk")
        elif gate == "burn_metrics_missing":
            humanized.append("burn metrics unavailable")
        elif gate == "low_runway_confidence":
            if runway_years is not None:
                humanized.append(f"runway ~{runway_years:.1f}y (low confidence)")
            else:
                humanized.append("low runway confidence")
        elif gate == "high_debt":
            humanized.append("high debt")
        else:
            # Unknown gate - just humanize the underscore
            humanized.append(gate.replace("_", " "))

    return "; ".join(humanized) if humanized else "multiple concerns"


def build_executive_summary(
    summary: dict[str, Any],
    technicals_summary: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    risk_summary: dict[str, Any],
    signals: dict[str, list[str]],
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    policy_action: dict[str, Any],
    news_summary: dict[str, Any] | None = None,
) -> str:
    """
    Build a MATERIALITY-FIRST deterministic executive summary from structured fields.

    DETERMINISM CONTRACT:
    - Lead sentence determined by materiality priority (policy gates > risk > fundamentals > technicals)
    - Fixed gate ordering for diff stability
    - Fixed rounding: percents 0-1dp, money in $M, runway 1dp
    - Humanized action names (avoid_core_position → "avoid as a core holding")
    - Numeric anchors embedded in gates (unprofitable → "unprofitable (margin -140%)")
    - Always includes: lead topic, counterbalancing factor, risk (if elevated), policy

    MATERIALITY PRIORITY (lead sentence topic):
    1. If long_term == "avoid" → lead with WHY (policy gates with numbers)
    2. Else if risk_regime in {extreme, high} → lead with RISK
    3. Else if fundamentals are weak/unprofitable → lead with FUNDAMENTALS
    4. Else → lead with TECHNICALS

    This ensures the summary highlights what actually matters for investment decisions.

    Returns:
        A 3-4 sentence narrative summary string.
    """
    name = summary.get("name", "This stock")
    # Use short name (first word or ticker-like)
    short_name = name.split(",")[0].split()[0] if name else "This stock"

    # === EXTRACT STRUCTURED FIELDS ===
    # Technicals
    trend = technicals_summary.get("trend", {})
    momentum = technicals_summary.get("momentum", {})
    returns = technicals_summary.get("returns", {})
    decomposed = verdict.get("decomposed", {})
    setup = decomposed.get("setup", "neutral")  # strong/weak/neutral

    # Fundamentals
    valuation = fundamentals_summary.get("valuation", {})
    profitability = fundamentals_summary.get("profitability", {})
    growth = fundamentals_summary.get("growth", {})
    cash_flow_summary = fundamentals_summary.get("cash_flow", {})
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}
    is_unprofitable = valuation.get("valuation_note") == "pe_not_meaningful"
    business_quality = decomposed.get("business_quality", "unknown")

    # Risk
    risk_regime = decomposed.get("risk", "unknown")  # extreme/high/medium/low
    vol = risk_summary.get("annualized_volatility")
    dd = risk_summary.get("max_drawdown_1y")

    # Policy (from policy_action - the source of truth)
    mid_term_action = policy_action.get("mid_term", "unknown")
    long_term_action = policy_action.get("long_term", "unknown")
    position_sizing = action_zones.get("position_sizing_range", {})
    suggested_pct = position_sizing.get("suggested_pct_range", [])

    # Horizon fit for materiality determination
    horizon_fit = verdict.get("horizon_fit", {})
    long_term_fit = horizon_fit.get("long_term", "unknown")
    long_term_gates = horizon_fit.get("long_term_gates") or []

    # === BUILD REUSABLE CLAUSES ===

    # Technicals clause
    tech_parts: list[str] = []
    if trend.get("above_sma200"):
        tech_parts.append("above SMA200")
    if trend.get("golden_cross"):
        tech_parts.append("golden cross")
    if momentum.get("macd_bullish"):
        tech_parts.append("MACD bullish")

    return_1m = returns.get("return_1m")
    return_3m = returns.get("return_3m")
    # Always include 1M if significant (>=10%), and 3M if available and significant (>=10%)
    # This avoids "1M pop" bias by showing both timeframes
    if return_1m is not None and abs(return_1m) >= 0.10:
        sign = "+" if return_1m >= 0 else ""
        tech_parts.append(f"1M {sign}{return_1m * 100:.1f}%")
    if return_3m is not None and abs(return_3m) >= 0.10:
        sign = "+" if return_3m >= 0 else ""
        tech_parts.append(f"3M {sign}{return_3m * 100:.1f}%")

    tech_str = ", ".join(tech_parts[:5]) if tech_parts else ""

    # Fundamentals clause parts
    fund_parts: list[str] = []
    rev_yoy = growth.get("revenue_yoy")
    if rev_yoy is not None:
        sign = "" if rev_yoy < 0 else "+"
        fund_parts.append(f"revenue_yoy {sign}{rev_yoy * 100:.1f}%")

    net_margin = profitability.get("net_margin")
    if net_margin is not None:
        fund_parts.append(f"net_margin {net_margin * 100:.1f}%")

    # `fcf_label` already covers the annualized FCF amount for burners, so don't
    # also append a manual `FCF $X M` clause derived from quarterly_burn — that
    # would duplicate the same information in `fund_parts` and crowd out runway.
    fcf_label = fcf_label_from_cashflow(cash_flow_summary)
    if fcf_label:
        fund_parts.append(fcf_label)
    quarterly_burn = burn_metrics.get("quarterly_fcf_burn")
    if quarterly_burn is not None and quarterly_burn > 0:
        fund_parts.append(f"burn ~${quarterly_burn / 1_000_000:.0f}M/quarter")

    runway_quarters = burn_metrics.get("cash_runway_quarters")
    runway_years = runway_quarters / 4 if runway_quarters else None
    if runway_years is not None:
        fund_parts.append(f"runway ~{runway_years:.1f} years")

    fund_str = "; ".join(fund_parts[:4]) if fund_parts else ""

    # Risk clause
    risk_details: list[str] = []
    if vol is not None:
        risk_details.append(f"vol {vol * 100:.1f}%")
    if dd is not None:
        risk_details.append(f"dd {dd * 100:.1f}%")
    risk_str = ", ".join(risk_details) if risk_details else ""

    # Policy clause (humanized actions)
    mid_term_human = _humanize_action(mid_term_action)
    long_term_human = _humanize_action(long_term_action)
    policy_parts: list[str] = [f"mid-term {mid_term_human}"]
    if len(suggested_pct) >= 2:
        policy_parts.append(f"size ~{suggested_pct[0]:.1f}-{suggested_pct[1]:.1f}%")
    policy_clause = "Policy: " + "; ".join(policy_parts) + "."

    # === DETERMINE LEAD TOPIC (MATERIALITY-FIRST) ===
    def _determine_lead_topic() -> str:
        # Priority 1: If long-term is "avoid", lead with policy gates
        if long_term_fit == "avoid":
            return "policy"
        # Priority 2: If risk is elevated, lead with risk
        if risk_regime in ("extreme", "high"):
            return "risk"
        # Priority 3: If fundamentals are weak/unprofitable, lead with fundamentals
        if business_quality in ("unprofitable", "poor", "weak") or is_unprofitable:
            return "fundamentals"
        # Priority 4: Default to technicals
        return "technicals"

    lead_topic = _determine_lead_topic()

    # === BUILD SENTENCES BASED ON LEAD TOPIC ===
    sentences: list[str] = []

    if lead_topic == "policy":
        # Lead: Why it's avoid (use humanized gates with numeric anchors)
        gates_str = _humanize_gates_with_numbers(
            gates=long_term_gates,
            rev_yoy=rev_yoy,
            net_margin=net_margin,
            quarterly_burn=quarterly_burn,
            runway_years=runway_years,
            vol=vol,
            dd=dd,
        )
        sentences.append(f"{short_name} is {long_term_human} long-term ({gates_str}).")

        # Counterbalance: mention technicals if strong
        if setup == "strong" and tech_str:
            sentences.append(f"Technicals are strong ({tech_str}).")
        elif tech_str:
            sentences.append(f"Technicals: {tech_str}.")

    elif lead_topic == "risk":
        # Lead: Risk regime
        if risk_str:
            sentences.append(f"{short_name} has {risk_regime} risk ({risk_str}).")
        else:
            sentences.append(f"{short_name} has {risk_regime} risk.")

        # Fundamentals
        if fund_str:
            if business_quality == "unprofitable" or is_unprofitable:
                sentences.append(f"Fundamentals: unprofitable ({fund_str}).")
            elif business_quality in ("poor", "weak"):
                sentences.append(f"Fundamentals are {business_quality} ({fund_str}).")
            else:
                sentences.append(f"Fundamentals: {fund_str}.")

        # Counterbalance: technicals if strong
        if setup == "strong" and tech_str:
            sentences.append(f"Technicals are strong ({tech_str}).")

    elif lead_topic == "fundamentals":
        # Lead: Fundamentals weakness
        if business_quality == "unprofitable" or is_unprofitable:
            if fund_str:
                sentences.append(f"{short_name} is unprofitable ({fund_str}).")
            else:
                sentences.append(f"{short_name} is unprofitable.")
        elif business_quality in ("poor", "weak"):
            if fund_str:
                sentences.append(f"{short_name} has {business_quality} fundamentals ({fund_str}).")
            else:
                sentences.append(f"{short_name} has {business_quality} fundamentals.")
        else:
            sentences.append(f"{short_name} fundamentals: {fund_str}.")

        # Risk if elevated
        if risk_regime in ("extreme", "high") and risk_str:
            sentences.append(f"Risk is {risk_regime} ({risk_str}).")

        # Counterbalance: technicals if strong
        if setup == "strong" and tech_str:
            sentences.append(f"Technicals are strong ({tech_str}).")

    else:  # technicals (default)
        # Lead: Technical setup
        if setup == "strong" and tech_str:
            sentences.append(f"{short_name} has a strong technical setup ({tech_str}).")
        elif setup == "weak":
            sentences.append(f"{short_name} has a weak technical setup.")
        else:
            if tech_str:
                sentences.append(f"{short_name} shows neutral technicals ({tech_str}).")
            else:
                sentences.append(f"{short_name} shows neutral technicals.")

        # Fundamentals
        if fund_str:
            if business_quality in ("good", "strong"):
                sentences.append(f"Fundamentals are solid ({fund_str}).")
            elif business_quality in ("unprofitable", "poor", "weak") or is_unprofitable:
                sentences.append(f"Fundamentals are weak ({fund_str}).")
            else:
                sentences.append(f"Fundamentals: {fund_str}.")

        # Risk if elevated
        if risk_regime in ("extreme", "high") and risk_str:
            sentences.append(f"Risk is {risk_regime} ({risk_str}).")

    # Always end with policy clause
    sentences.append(policy_clause)

    return " ".join(sentences)


def policy_codes(
    *,
    mid_term_fit: str,
    long_term_fit: str,
    long_term_gates: list[str],
    tilt: str | None,
    zone: str | None,
    valuation_gate: str | None,
    is_unprofitable: bool,
    dislocation: bool,
) -> tuple[str, str, list[str]]:
    """The pre-gate candidate codes (mid_term, long_term) and their rationale. Pure: the binding conditions
    replay it with one input cleared (plan §7e)."""
    rationale: list[str] = []

    # === MID-TERM ACTION (3-12 months) ===
    mid_term_action: str
    if mid_term_fit == "avoid":
        mid_term_action = "avoid"
        rationale.append("mid_term_fit=avoid")
    elif mid_term_fit == "caution":
        if tilt == "bullish":
            mid_term_action = "speculative_small_position"
            rationale.append("mid_term_fit=caution, tilt=bullish")
        elif dislocation:
            mid_term_action = "speculative_small_position"
            rationale.append("mid_term_fit=caution, dislocation=price_broken_more_than_business")
        elif tilt is None:
            mid_term_action = "insufficient_data"
            rationale.append("mid_term=caution, tilt=missing")
        else:
            mid_term_action = "wait_for_entry"
            rationale.append(f"mid_term_fit=caution, tilt={tilt}")
    elif mid_term_fit == "ok":
        if tilt is None:
            mid_term_action = "insufficient_data"
            rationale.append("mid_term=ok, tilt=missing")
        elif tilt == "bullish" and zone in ("strong_buy", "accumulate"):
            mid_term_action = "buy"
            rationale.append(f"mid_term=ok, tilt=bullish, zone={zone}")
        elif tilt == "bullish":
            mid_term_action = "hold_or_add"
            rationale.append(f"mid_term=ok, tilt=bullish, zone={zone}")
        elif tilt == "neutral":
            mid_term_action = "hold"
            rationale.append(f"mid_term=ok, tilt=neutral, zone={zone}")
        else:  # bearish
            mid_term_action = "hold_or_reduce"
            rationale.append(f"mid_term=ok, tilt=bearish, zone={zone}")
    else:  # unknown
        mid_term_action = "insufficient_data"
        rationale.append("mid_term_fit=unknown")

    # === LONG-TERM ACTION (1-5 years) ===
    long_term_action: str
    if long_term_fit == "avoid":
        long_term_action = "avoid_core_position"
        rationale.append(f"long_term_fit=avoid, gates={long_term_gates}")
    elif long_term_fit == "caution":
        if is_unprofitable:
            if valuation_gate == "attractive":
                long_term_action = "speculative_with_thesis"
                rationale.append("long_term=caution, unprofitable, valuation=attractive")
            else:
                long_term_action = "speculative_only"
                rationale.append("long_term=caution, unprofitable")
        else:
            if tilt == "bullish" or dislocation:
                long_term_action = "small_position_with_stops"
                if dislocation and tilt != "bullish":
                    rationale.append("long_term=caution, dislocation=price_broken_more_than_business")
                else:
                    rationale.append("long_term=caution, tilt=bullish")
            elif tilt is None:
                long_term_action = "insufficient_data"
                rationale.append("long_term=caution, tilt=missing")
            else:
                long_term_action = "wait_or_avoid"
                rationale.append(f"long_term=caution, tilt={tilt}")
    elif long_term_fit == "ok":
        if valuation_gate == "attractive":
            long_term_action = "accumulate"
            rationale.append(f"long_term=ok, valuation=attractive, zone={zone}")
        elif valuation_gate == "headwind":
            long_term_action = "hold_size_conservatively"
            rationale.append("long_term=ok, valuation=headwind")
        elif tilt is None:
            long_term_action = "insufficient_data"
            rationale.append("long_term=ok, tilt=missing")
        else:
            long_term_action = "hold_or_add" if tilt == "bullish" else "hold"
            rationale.append(f"long_term=ok, tilt={tilt}, zone={zone}")
    else:  # unknown
        long_term_action = "insufficient_data"
        rationale.append("long_term_fit=unknown")

    return mid_term_action, long_term_action, rationale


def build_policy_action(
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    decomposed: dict[str, Any] | None = None,
    risk_regime: dict[str, Any] | None = None,
    dip_assessment: dict[str, Any] | None = None,
    fundamentals_summary: dict[str, Any] | None = None,
    investor_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the pre-gate policy: the candidate decide() reads, emitted only as diagnostics.pre_gate_policy.

    Combines: tilt, horizon_fit, zone, valuation_gate. Includes the candidate's upgrade and downgrade
    conditions and, under inputs, the zone and valuation it read.

    Returns:
        Dict with mid_term/long_term codes, rationale, conditions and inputs.
    """
    # WP9: an absent or null tilt is missing, not neutral and not bearish; the
    # branches it would decide become insufficient_data.
    tilt = verdict.get("tilt")
    horizon_fit = verdict.get("horizon_fit", {})
    long_term_gates = horizon_fit.get("long_term_gates") or []

    zone = action_zones.get("current_zone")
    valuation_assessment = action_zones.get("valuation_assessment", {})
    valuation_gate = valuation_assessment.get("gate")
    valuation_basis = valuation_assessment.get("basis")
    is_unprofitable = valuation_assessment.get("is_unprofitable", False)
    fundamentals_summary = fundamentals_summary or {}

    risk_label = decomposed.get("risk") if decomposed else None
    dislocation_signals = classify_dislocation_opportunity(
        verdict=verdict,
        action_zones=action_zones,
        fundamentals_summary=fundamentals_summary,
        dip_assessment=dip_assessment,
    )
    mid_term_action, long_term_action, rationale = policy_codes(
        mid_term_fit=horizon_fit.get("mid_term", "unknown"),
        long_term_fit=horizon_fit.get("long_term", "unknown"),
        long_term_gates=long_term_gates,
        tilt=tilt,
        zone=zone,
        valuation_gate=valuation_gate,
        is_unprofitable=is_unprofitable,
        dislocation=dislocation_signals["is_dislocation_candidate"],
    )

    # === CONDITIONS TO UPGRADE ===
    # Each input holding the candidate back (plan §7e: decide which bind per horizon).
    conditions_to_upgrade = [
        text
        for factor in candidate_factors(
            decomposed=decomposed,
            valuation_assessment=valuation_assessment,
            zone_warnings=action_zones.get("zone_warnings") or [],
            risk_regime=risk_regime,
            long_term_gates=long_term_gates,
            leverage_status=(fundamentals_summary.get("health") or {}).get("leverage_status"),
        )
        for text in factor.texts
    ]

    # === CONDITIONS TO DOWNGRADE ===
    # What would make this worse?
    conditions_to_downgrade: list[str] = []

    if risk_label in ("low", "medium"):
        conditions_to_downgrade.append(
            f"volatility spike > {VOLATILITY_REGIME_THRESHOLDS['medium'] * 100:.0f}%"
        )
    stock_rules_apply = fundamentals_summary.get("status") != "not_applicable"
    if not is_unprofitable and stock_rules_apply:
        conditions_to_downgrade.append("guidance cut or profit warning")
        conditions_to_downgrade.append("net_margin turns negative")
    if valuation_gate == "attractive":
        conditions_to_downgrade.append("multiple expansion without earnings growth")

    return {
        "mid_term": mid_term_action,
        "long_term": long_term_action,
        "rationale": rationale,
        "conditions_to_upgrade": conditions_to_upgrade,
        "conditions_to_downgrade": conditions_to_downgrade,
        "inputs": {
            "zone": zone,
            "valuation_gate": valuation_gate,
            "valuation_basis": valuation_basis,
            "is_unprofitable": is_unprofitable if is_unprofitable else None,
        },
    }
