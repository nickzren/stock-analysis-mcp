"""Long-term ownership assessment from explicit business evidence, independent of entry rules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

ComponentState = Literal["evaluated", "not_evaluated", "not_applicable"]
Applicability = Literal["applicable", "not_applicable", "unverified"]


@dataclass(frozen=True)
class ComponentEvidence:
    state: ComponentState
    classification: str
    reason: str
    evidence: dict[str, Any]
    periods: dict[str, Any] | None = None
    periods_reason: str | None = "Input-specific periods unavailable."


@dataclass(frozen=True)
class HolderEvidence:
    applicability: Applicability
    business: ComponentEvidence
    cash_runway: ComponentEvidence
    watch_factors: tuple[dict[str, Any], ...] = ()


def _unavailable(state: ComponentState, reason: str) -> ComponentEvidence:
    return ComponentEvidence(state, "not_applicable" if state == "not_applicable" else "unknown", reason, {})


def _periods(values: dict[str, Any], cash_flow: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Only cash-flow input periods are supplied today; never borrow a composite/fetch date."""
    periods: dict[str, Any] = dict.fromkeys(values)
    for name, prefix in (("free_cash_flow_ttm", "free_cash_flow"), ("operating_cf_ttm", "operating_cf")):
        if values.get(name) is not None:
            metadata = {field: cash_flow.get(f"{prefix}_{field}")
                        for field in ("period", "period_end", "source", "ttm_status", "period_lag")}
            if metadata["period"] is not None or metadata["period_end"] is not None:
                periods[name] = metadata
    if not any(value is not None for value in periods.values()):
        return None, "Input-specific periods unavailable; fetch dates are not financial periods."
    return periods, "Null entries have no input-specific financial period; no composite period is inferred."


def build_holder_evidence(*, support: dict[str, Any], fundamentals_status: str,
                          fundamentals: dict[str, Any], source: dict[str, Any],
                          signals: dict[str, Any]) -> HolderEvidence:
    """Reuse the thesis classifier's exact assessability and unrounded component results."""
    instrument = (support.get("instrument") or {}).get("status")
    applicability: Applicability = "unverified"
    if instrument == "supported":
        applicability = "applicable"
    elif instrument == "unsupported":
        applicability = "not_applicable"
    missing = _unavailable("not_evaluated", "Fundamental source unavailable; required checks could not be run.")
    if fundamentals_status == "missing":
        return HolderEvidence(applicability, missing, missing)

    growth = fundamentals.get("growth") or {}
    profit = fundamentals.get("profitability") or {}
    health = fundamentals.get("health") or {}
    burn = fundamentals.get("burn_metrics") or {}
    cash_flow = source.get("cash_flow") or {}
    financial_health = source.get("financial_health") or {}
    business = signals.get("business_integrity") or {}
    balance = signals.get("balance_sheet_safety") or {}
    business_known = signals.get("business_evaluable") is True
    runway_known = signals.get("runway_evaluable") is True

    business_values = {"revenue_yoy": growth.get("revenue_yoy"), "net_margin": profit.get("net_margin"),
                       "fcf_positive": profit.get("fcf_positive"),
                       "free_cash_flow_ttm": cash_flow.get("free_cash_flow_ttm"),
                       "debt_to_equity": health.get("debt_to_equity")}
    business_class = (signals.get("business_status") or "unknown") if business_known else "unknown"
    business_reason = {
        "broken": "The existing business checks establish an impaired business; reassess continued ownership.",
        "intact": "The available business checks do not demonstrate a thesis failure.",
        "mixed": "Business evidence is mixed; the known concerns remain visible below.",
    }.get(business_class, "Required business evidence is unavailable; this is not a passed check.")
    business_check = ComponentEvidence(
        "evaluated" if business_known else "not_evaluated", business_class or "unknown", business_reason,
        business_values, *_periods(business_values, cash_flow))

    liquidity = financial_health.get("cash_and_st_investments")
    if liquidity is None:
        liquidity = financial_health.get("total_cash")
    runway_values = {"liquidity": liquidity, "free_cash_flow_ttm": cash_flow.get("free_cash_flow_ttm"),
                     "operating_cf_ttm": cash_flow.get("operating_cf_ttm"),
                     "cash_runway_quarters": burn.get("cash_runway_quarters"),
                     "runway_basis": burn.get("runway_basis")}
    runway_result = (balance.get("checks") or {}).get("runway") or "unknown"
    runway_class = "unknown"
    runway_state: ComponentState = "not_evaluated"
    runway_reason = "Required cash-runway evidence is unavailable; this is not a passed check."
    if runway_known:
        if burn.get("status") == "not_applicable" or not burn:
            runway_state, runway_class = "not_applicable", "not_applicable"
            runway_reason = ("Available cash-flow measures are not negative, so a cash-burn runway does not apply."
                             if burn else "The existing runway model does not apply under this business classification; "
                             "this does not assert positive free cash flow.")
        else:
            runway_state = "evaluated"
            runway_class = {"fail": "critical", "watch": "watch", "pass": "adequate"}.get(runway_result, "unknown")
            runway_reason = {"critical": "Cash runway is below four quarters; reassess continued ownership.",
                             "watch": "Cash runway is between four and eight quarters and needs review.",
                             "adequate": "Cash runway is at least eight quarters under the existing model."}.get(
                                 runway_class, runway_reason)
    runway_check = ComponentEvidence(runway_state, runway_class, runway_reason, runway_values,
                                     *_periods(runway_values, cash_flow))

    factors = []
    business_checks = business.get("checks") or {}
    for code, field, label in (("growth", "revenue_yoy", "Revenue growth"),
                               ("profitability", "net_margin", "Net margin"),
                               ("cash_flow", "fcf_positive", "Free cash flow")):
        if business_checks.get(code) in {"watch", "fail"} and business_values[field] is not None:
            factors.append({"code": code, "reason": f"{label} is a known concern.",
                            "evidence": {field: business_values[field]}})
    balance_values = {"runway_under_4q": {"cash_runway_quarters": burn.get("cash_runway_quarters")},
                      "runway_under_8q": {"cash_runway_quarters": burn.get("cash_runway_quarters")},
                      "net_debt": {"net_cash_positive": health.get("net_cash_positive")},
                      "negative_equity": {"leverage_status": health.get("leverage_status")},
                      "high_leverage": {"debt_to_equity": health.get("debt_to_equity")},
                      "dilution_risk": {"dilution_if_raised_today": (burn.get("dilution_analysis") or {}).get(
                          "dilution_if_raised_today")}}
    for code in balance.get("reasons") or []:
        if code in balance_values:
            factors.append({"code": code, "reason": code.replace("_", " "), "evidence": balance_values[code]})
    return HolderEvidence(applicability, business_check, runway_check, tuple(factors))


def _validated_component(check: ComponentEvidence, name: str, labels: tuple[str, ...]) -> ComponentEvidence:
    if check.state not in ("evaluated", "not_evaluated", "not_applicable"):
        return replace(check, state="not_evaluated",
                       reason=f"Unrecognized {name} state; this check was not evaluated.")
    if name == "business" and check.state == "not_applicable":
        return replace(check, state="not_evaluated",
                       reason="Business evidence is required for an applicable instrument; this check was not evaluated.")
    if check.state == "evaluated" and check.classification not in labels:
        return replace(check, state="not_evaluated",
                       reason=f"Unrecognized {name} classification; this check was not evaluated.")
    return check


def long_term_if_held(evidence: HolderEvidence | None) -> dict[str, Any]:
    """Applicability first, then proven component failures, missing checks and qualified hold."""
    if evidence is None or evidence.applicability != "applicable":
        unsupported = evidence is not None and evidence.applicability == "not_applicable"
        state: ComponentState = "not_applicable" if unsupported else "not_evaluated"
        reason = ("This tool does not assess long-term ownership of this instrument; its company rules do not apply."
                  if unsupported else "The instrument type must be confirmed before company rules can be applied.")
        check = _unavailable(state, reason)
        return {"action": "not_assessed" if unsupported else "insufficient_data",
                "basis": "unsupported_instrument" if unsupported else "applicability_unverified",
                "reasons": ["unsupported_instrument" if unsupported else "applicability_unverified"],
                "assessment_status": state, "checks": {"business": asdict(check), "cash_runway": asdict(check)},
                "watch_factors": [], "review_when": [reason]}

    business = _validated_component(evidence.business, "business", ("intact", "mixed", "broken"))
    runway = _validated_component(evidence.cash_runway, "cash runway", ("adequate", "watch", "critical"))
    failures = [code for hit, code in (
        (runway.state == "evaluated" and runway.classification == "critical", "runway_critical"),
        (business.state == "evaluated" and business.classification == "broken", "business_broken")) if hit]
    missing = [name for name, needed in (("business", business.state != "evaluated"),
               ("cash_runway", runway.state not in {"evaluated", "not_applicable"})) if needed]
    unknowns = [f"{name}_not_evaluated" for name in missing]
    if failures:
        action, basis, status = "review_exit", failures[0], "evaluated"
    elif unknowns:
        action, basis, status = "insufficient_data", "thesis_unknown", "not_evaluated"
    else:
        action, basis, status = "hold", "no_demonstrated_thesis_failure", "evaluated"
    checkpoints = [f"Reassess when the missing {name.replace('_', ' ')} evidence becomes available."
                   for name in missing]
    checkpoints.append("Reassess when reported results change the business or cash-runway evidence.")
    return {"action": action, "basis": basis, "reasons": failures + unknowns or [basis],
            "assessment_status": status, "checks": {"business": asdict(business), "cash_runway": asdict(runway)},
            "watch_factors": deepcopy(list(evidence.watch_factors)), "review_when": checkpoints}
