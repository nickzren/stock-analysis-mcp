"""Dislocation-style framework for broken price vs broken business analysis."""

from __future__ import annotations

from typing import Any

from stock_analysis.utils.helpers import format_compact_number, format_pct


def classify_dislocation_opportunity(
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return reusable dislocation signals for action overrides."""
    dip_assessment = dip_assessment or {}
    dip_depth = dip_assessment.get("dip_depth") or {}
    valuation_assessment = action_zones.get("valuation_assessment") or {}
    decomposed = verdict.get("decomposed") or {}
    growth = fundamentals_summary.get("growth") or {}
    profitability = fundamentals_summary.get("profitability") or {}
    health = fundamentals_summary.get("health") or {}
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}

    setup_status = _classify_setup(
        dip_depth.get("from_52w_high"),
        dip_depth.get("severity"),
    )
    business_integrity = _build_business_integrity(
        business_quality=decomposed.get("business_quality"),
        revenue_yoy=growth.get("revenue_yoy"),
        net_margin=profitability.get("net_margin"),
        fcf_positive=profitability.get("fcf_positive"),
    )
    balance_sheet_safety = _build_balance_sheet_safety(
        health=health,
        burn_metrics=burn_metrics,
    )
    mismatch_verdict = _build_mismatch_verdict(
        setup_status=setup_status,
        business_status=business_integrity["status"],
        balance_status=balance_sheet_safety["status"],
        valuation_gate=valuation_assessment.get("gate"),
    )

    return {
        "setup_status": setup_status,
        "business_status": business_integrity["status"],
        "balance_status": balance_sheet_safety["status"],
        "mismatch_status": mismatch_verdict["status"],
        "is_dislocation_candidate": (
            mismatch_verdict["status"] == "price_broken_more_than_business"
        ),
    }


def build_dislocation_framework(
    verdict: dict[str, Any],
    action_zones: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    dip_assessment: dict[str, Any] | None,
    decision_context: dict[str, Any],
    decision_modes: dict[str, Any],
    policy_action: dict[str, Any],
) -> dict[str, Any]:
    """Translate the full analysis into a dislocation-investing workflow."""
    dip_assessment = dip_assessment or {}
    dip_depth = dip_assessment.get("dip_depth") or {}
    valuation_assessment = action_zones.get("valuation_assessment") or {}
    thesis_checkpoints = decision_context.get("thesis_checkpoints") or {}
    dislocation_signals = classify_dislocation_opportunity(
        verdict=verdict,
        action_zones=action_zones,
        fundamentals_summary=fundamentals_summary,
        dip_assessment=dip_assessment,
    )

    drawdown = dip_depth.get("from_52w_high")
    severity = dip_depth.get("severity")
    setup_status = dislocation_signals["setup_status"]
    setup = {
        "status": setup_status,
        "drawdown": drawdown,
        "drawdown_label": _drawdown_label(drawdown),
        "severity": severity,
        "summary": _build_setup_summary(setup_status, drawdown),
    }

    growth = fundamentals_summary.get("growth") or {}
    profitability = fundamentals_summary.get("profitability") or {}
    health = fundamentals_summary.get("health") or {}
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}
    business_integrity = _build_business_integrity(
        business_quality=(verdict.get("decomposed") or {}).get("business_quality"),
        revenue_yoy=growth.get("revenue_yoy"),
        net_margin=profitability.get("net_margin"),
        fcf_positive=profitability.get("fcf_positive"),
    )

    balance_sheet_safety = _build_balance_sheet_safety(
        health=health,
        burn_metrics=burn_metrics,
    )

    thesis_integrity = _build_thesis_integrity(
        thesis_checkpoints=thesis_checkpoints,
        business_status=business_integrity["status"],
        balance_status=balance_sheet_safety["status"],
        long_term_fit=(verdict.get("horizon_fit") or {}).get("long_term"),
    )

    mismatch_verdict = _build_mismatch_verdict(
        setup_status=setup_status,
        business_status=business_integrity["status"],
        balance_status=balance_sheet_safety["status"],
        valuation_gate=valuation_assessment.get("gate"),
    )

    starter_actions = {"starter_position_only", "speculative_small_position"}
    can_start_now = any(
        (decision_modes.get(mode_name) or {}).get("action") in starter_actions
        for mode_name in ("balanced", "speculative")
    )
    add_only_if = list(policy_action.get("conditions_to_upgrade") or [])[:3]

    action = {
        "core": (decision_modes.get("core") or {}).get("action"),
        "balanced": (decision_modes.get("balanced") or {}).get("action"),
        "speculative": (decision_modes.get("speculative") or {}).get("action"),
        "can_start_now": can_start_now,
        "buy_now": "small_starter" if can_start_now else None,
        "buy_only_if": [] if can_start_now else add_only_if,
        "add_only_if": add_only_if if can_start_now else [],
        "sell_without_attachment_if": thesis_integrity.get("sell_without_attachment_if") or [],
    }

    summary = _build_framework_summary(
        setup=setup,
        business_integrity=business_integrity,
        balance_sheet_safety=balance_sheet_safety,
        mismatch_verdict=mismatch_verdict,
    )

    return {
        "setup": setup,
        "business_integrity": business_integrity,
        "balance_sheet_safety": balance_sheet_safety,
        "thesis_integrity": thesis_integrity,
        "mismatch_verdict": mismatch_verdict,
        "action": action,
        "summary": summary,
        "method": "dislocation_framework_v1",
    }


def _classify_setup(drawdown: float | None, severity: str | None) -> str:
    if drawdown is not None:
        if drawdown <= -0.40:
            return "present"
        if drawdown <= -0.20:
            return "weak"
        return "absent"
    if severity in {"deep", "extreme"}:
        return "present"
    if severity == "moderate":
        return "weak"
    return "absent"


def _drawdown_label(drawdown: float | None) -> str | None:
    pct = format_pct(drawdown, decimals=1)
    if pct is None:
        return None
    return f"{pct} from 52W high"


def _build_setup_summary(setup_status: str, drawdown: float | None) -> str:
    drawdown_label = _drawdown_label(drawdown)
    if setup_status == "present":
        if drawdown_label:
            return f"Large drawdown is present: {drawdown_label}."
        return "Large drawdown setup is present."
    if setup_status == "weak":
        if drawdown_label:
            return f"Some drawdown is present, but it is only {drawdown_label}."
        return "Some drawdown is present, but not enough for a strong dislocation setup."
    return "No meaningful drawdown setup yet."


def _metric_status(
    value: float | None,
    pass_if: float,
    watch_if: float,
) -> str:
    if value is None:
        return "unknown"
    if value >= pass_if:
        return "pass"
    if value >= watch_if:
        return "watch"
    return "fail"


def _build_business_integrity(
    business_quality: str | None,
    revenue_yoy: float | None,
    net_margin: float | None,
    fcf_positive: bool | None,
) -> dict[str, Any]:
    growth_status = _metric_status(revenue_yoy, pass_if=0.10, watch_if=0.0)

    if net_margin is None:
        profitability_status = "unknown"
    elif net_margin > 0:
        profitability_status = "pass"
    elif net_margin > -0.05:
        profitability_status = "watch"
    else:
        profitability_status = "fail"

    if fcf_positive is True:
        cash_flow_status = "pass"
    elif fcf_positive is False:
        cash_flow_status = "fail"
    else:
        cash_flow_status = "unknown"

    if business_quality in {"strong", "moderate"}:
        status = "intact"
    elif business_quality in {"mixed", "weak", "unprofitable"}:
        status = "mixed"
    elif business_quality == "poor" or (
        growth_status == "fail" and profitability_status == "fail"
    ):
        status = "broken"
    else:
        status = "mixed"

    growth_label = "growth still healthy" if growth_status == "pass" else "growth is not convincing"
    if profitability_status == "pass":
        profit_label = "profitability is proven"
    elif profitability_status == "watch":
        profit_label = "profitability is still weak"
    else:
        profit_label = "profitability is broken"

    if status == "intact":
        summary = f"Business still looks intact: {growth_label} and {profit_label}."
    elif status == "mixed":
        summary = f"Business is mixed: {growth_label}, but {profit_label}."
    else:
        summary = f"Business looks impaired: {growth_label} and {profit_label}."

    return {
        "status": status,
        "summary": summary,
        "checks": {
            "growth": growth_status,
            "profitability": profitability_status,
            "cash_flow": cash_flow_status,
            "durability": _durability_status(business_quality),
            "competitive_position": "unknown",
        },
    }


def _durability_status(business_quality: str | None) -> str:
    if business_quality in {"strong", "moderate"}:
        return "pass"
    if business_quality in {"mixed", "weak", "unprofitable"}:
        return "watch"
    if business_quality == "poor":
        return "fail"
    return "unknown"


def _build_balance_sheet_safety(
    health: dict[str, Any],
    burn_metrics: dict[str, Any],
) -> dict[str, Any]:
    net_cash_positive = health.get("net_cash_positive")
    debt_to_equity = health.get("debt_to_equity")
    liquidity = burn_metrics.get("liquidity")
    runway_quarters = burn_metrics.get("cash_runway_quarters")
    dilution_analysis = burn_metrics.get("dilution_analysis") or {}
    dilution_pct = dilution_analysis.get("dilution_if_raised_today")

    if net_cash_positive is True:
        net_cash_status = "pass"
    elif net_cash_positive is False:
        net_cash_status = "fail"
    else:
        net_cash_status = "unknown"

    if debt_to_equity is None:
        debt_status = "unknown"
    elif debt_to_equity <= 1.5:
        debt_status = "pass"
    elif debt_to_equity <= 3.0:
        debt_status = "watch"
    else:
        debt_status = "fail"

    if runway_quarters is None:
        runway_status = "pass" if burn_metrics.get("status") == "not_applicable" else "unknown"
    elif runway_quarters >= 8:
        runway_status = "pass"
    elif runway_quarters >= 4:
        runway_status = "watch"
    else:
        runway_status = "fail"

    if dilution_pct is None:
        dilution_status = "unknown"
    elif dilution_pct >= 0.20:
        dilution_status = "fail"
    elif dilution_pct >= 0.10:
        dilution_status = "watch"
    else:
        dilution_status = "pass"

    if runway_status == "fail" or net_cash_status == "fail":
        status = "unsafe"
    elif (
        net_cash_status == "pass"
        and debt_status in {"pass", "unknown"}
        and runway_status in {"pass", "unknown"}
        and dilution_status != "fail"
    ):
        status = "safe"
    else:
        status = "watch"

    liquidity_label = format_compact_number(liquidity)
    runway_label = f"{runway_quarters:.1f}q" if runway_quarters is not None else None

    if status == "safe":
        summary = "Balance sheet looks safe."
        if liquidity_label:
            summary = f"Balance sheet looks safe with liquidity around {liquidity_label}."
    elif status == "watch":
        summary = "Balance sheet is workable but needs monitoring."
        if runway_label:
            summary = f"Balance sheet needs monitoring; runway is about {runway_label}."
    else:
        summary = "Balance sheet risk is elevated enough to threaten the thesis."

    return {
        "status": status,
        "summary": summary,
        "checks": {
            "net_cash": net_cash_status,
            "debt_load": debt_status,
            "runway": runway_status,
            "dilution_risk": dilution_status,
        },
    }


def _build_thesis_integrity(
    thesis_checkpoints: dict[str, Any],
    business_status: str,
    balance_status: str,
    long_term_fit: str | None,
) -> dict[str, Any]:
    hold_thesis = thesis_checkpoints.get("hold_thesis")
    checkpoints = thesis_checkpoints.get("checkpoints") or []
    review_triggers = thesis_checkpoints.get("review_triggers") or []
    stop_triggers = thesis_checkpoints.get("thesis_stop_triggers") or []

    if business_status == "broken" or balance_status == "unsafe":
        status = "broken"
    elif business_status == "intact" and balance_status == "safe" and long_term_fit == "ok":
        status = "intact"
    else:
        status = "watch"

    must_remain_true = [
        item.get("target")
        for item in checkpoints
        if isinstance(item, dict) and item.get("target")
    ][:3]
    sell_without_attachment_if = [
        _format_stop_trigger(item)
        for item in stop_triggers
        if isinstance(item, dict)
    ]
    sell_without_attachment_if = [item for item in sell_without_attachment_if if item][:3]

    if status == "intact":
        summary = "The long-term thesis still looks intact on current evidence."
    elif status == "watch":
        summary = "The long-term thesis is still alive, but it needs proof at the next checkpoints."
    else:
        summary = "The long-term thesis looks damaged enough that the drawdown may be justified."

    return {
        "status": status,
        "hold_thesis": hold_thesis,
        "summary": summary,
        "what_must_remain_true": must_remain_true,
        "review_triggers": review_triggers[:4],
        "sell_without_attachment_if": sell_without_attachment_if,
    }


def _format_stop_trigger(trigger: dict[str, Any]) -> str | None:
    condition = trigger.get("condition")
    rationale = trigger.get("rationale")
    if condition and rationale:
        return f"{condition} ({rationale})"
    if condition:
        return str(condition)
    return None


def _build_mismatch_verdict(
    setup_status: str,
    business_status: str,
    balance_status: str,
    valuation_gate: str | None,
) -> dict[str, Any]:
    if business_status == "broken" or balance_status == "unsafe":
        status = "both_broken" if setup_status == "present" else "business_broken_more_than_price"
    elif setup_status == "present" and business_status == "intact" and balance_status == "safe":
        if valuation_gate == "attractive":
            status = "price_broken_more_than_business"
        elif valuation_gate == "headwind":
            status = "business_intact_but_not_cheap"
        else:
            status = "unclear"
    elif business_status == "intact" and valuation_gate == "headwind":
        status = "business_intact_but_not_cheap"
    else:
        status = "unclear"

    summary_map = {
        "price_broken_more_than_business": "Price looks more broken than the business.",
        "business_broken_more_than_price": "The business looks weaker than the drawdown alone suggests.",
        "both_broken": "Both price and business quality are damaged; this is not a clean mismatch.",
        "business_intact_but_not_cheap": "The business may be intact, but valuation still limits the mismatch.",
        "unclear": "There is some dislocation, but not yet a clean broken-price-only setup.",
    }

    return {
        "status": status,
        "summary": summary_map[status],
        "valuation_gate": valuation_gate,
    }


def _build_framework_summary(
    setup: dict[str, Any],
    business_integrity: dict[str, Any],
    balance_sheet_safety: dict[str, Any],
    mismatch_verdict: dict[str, Any],
) -> str:
    setup_summary = setup.get("summary") or "No drawdown read."
    business_summary = business_integrity.get("summary") or "Business quality unclear."
    balance_summary = balance_sheet_safety.get("summary") or "Balance sheet unclear."
    mismatch_summary = mismatch_verdict.get("summary") or "Mismatch unclear."
    return f"{setup_summary} {business_summary} {balance_summary} {mismatch_summary}"
