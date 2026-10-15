"""The pre-gate candidate's upgrade factors (plan §7e, BL-E1/BL-E2): each input that holds the candidate back,
its condition text and the cleared state the binding test replays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from stock_analysis.tools.analyze.action_zones import (
    PE_HEADWIND_ABOVE,
    PEG_HEADWIND_ABOVE,
    SALES_MULTIPLE_HEADWIND_ABOVE,
)
from stock_analysis.tools.analyze.signals import (
    BETA_REGIME_THRESHOLDS,
    DRAWDOWN_REGIME_THRESHOLDS,
    VOLATILITY_REGIME_THRESHOLDS,
)

_RISK_TARGET = {"extreme": "high", "high": "medium"}  # the regime one step down, whose thresholds must clear


@dataclass(frozen=True)
class Factor:
    id: str
    texts: tuple[str, ...]
    overrides: Mapping[str, Any]  # the cleared state: horizon-fit inputs, valuation_gate, is_unprofitable


@dataclass(frozen=True)
class CandidateInputs:
    """What the pre-gate candidate read, so a factor's cleared state can be replayed through it."""

    fit: Mapping[str, Any]  # verdict.horizon_fit_inputs
    tilt: str | None
    valuation_gate: str | None
    is_unprofitable: bool
    dislocation: bool
    uncapped_zone: str | None
    dip_type: str | None


def risk_text(level: str, factors: Mapping[str, Any] | None) -> str:
    """The risk condition naming the regime driver(s) at ``level`` (BL-E2)."""
    target = _RISK_TARGET[level]
    clauses = {
        "volatility": f"volatility < {VOLATILITY_REGIME_THRESHOLDS[target] * 100:.0f}%",
        "drawdown": f"max_drawdown_1y > {DRAWDOWN_REGIME_THRESHOLDS[target] * 100:.0f}%",
        "beta": f"beta < {BETA_REGIME_THRESHOLDS[target]}",
    }
    drivers = [text for name, text in clauses.items() if (factors or {}).get(name) == level]
    return f"risk_regime <= {target}" + (f" ({' and '.join(drivers)})" if drivers else "")


def valuation_headwind_texts(valuation_assessment: Mapping[str, Any], zone_warnings: list[str]) -> list[str]:
    """One condition per fired headwind metric, quoting the assessment's own threshold (BL-E1)."""
    sales = "EV/S" if valuation_assessment.get("basis") == "ev_to_sales" else "P/S"
    texts = {
        "valuation_extended_high_pe": f"P/E <= {PE_HEADWIND_ABOVE}x (currently headwind)",
        "valuation_extended_high_peg": f"PEG <= {PEG_HEADWIND_ABOVE} (currently headwind)",
        "valuation_extended_high_sales_multiple_unprofitable":
            f"{sales} <= {SALES_MULTIPLE_HEADWIND_ABOVE}x (currently headwind)",
    }
    return [text for code, text in texts.items() if code in zone_warnings]


def _valuation_unknown_text(valuation_assessment: Mapping[str, Any]) -> str:
    reasons = " ".join(valuation_assessment.get("reasons") or [])
    if "unverified" in reasons:
        return "valuation inputs on one verified currency basis"
    if "ev_nonpositive" in reasons:
        return "positive enterprise value"
    return "revenue data available for valuation"


def candidate_factors(
    *,
    decomposed: Mapping[str, Any] | None,
    valuation_assessment: Mapping[str, Any],
    zone_warnings: list[str],
    risk_regime: Mapping[str, Any] | None,
    long_term_gates: list[str],
    leverage_status: str | None,
) -> list[Factor]:
    """The closed factor list, in conditions_to_upgrade order (plan §7e)."""
    decomposed = decomposed or {}
    factors: list[Factor] = []
    risk = decomposed.get("risk")
    if risk in _RISK_TARGET:
        factors.append(Factor(f"risk_{risk}", (risk_text(risk, (risk_regime or {}).get("factors")),),
                              {"risk_label": _RISK_TARGET[risk]}))
    if valuation_assessment.get("is_unprofitable") or decomposed.get("business_quality_status") == "evaluated_unprofitable":
        factors.append(Factor(
            "unprofitable",
            ("profitMargins > 0 for 2 consecutive quarters", "freeCashflow > 0 for 2 consecutive quarters"),
            {"business_quality_label": "mixed", "business_quality_status": "evaluated", "is_unprofitable": False},
        ))
    gate = valuation_assessment.get("gate")
    if gate == "headwind":
        factors.append(Factor("valuation_headwind",
                              tuple(valuation_headwind_texts(valuation_assessment, zone_warnings)),
                              {"valuation_gate": "neutral"}))
    elif gate == "unknown":
        factors.append(Factor("valuation_unknown", (_valuation_unknown_text(valuation_assessment),),
                              {"valuation_gate": "neutral"}))
    quality = decomposed.get("business_quality")
    if quality == "poor":
        positives = (decomposed.get("business_quality_evidence") or {}).get("positives")
        factors.append(Factor(
            "business_poor",
            ("shareholder equity > 0" if leverage_status == "negative_equity" else "debt_to_equity < 1.5x",),
            {"business_quality_label": "moderate" if positives else "mixed"},
        ))
    elif quality == "weak":
        factors.append(Factor("business_weak", ("net_margin > 5%",), {"business_quality_label": "mixed"}))
    for gate_id, text, cleared in (
        ("burn_metrics_missing", "cash runway data available", {"burn_metrics_status": "available"}),
        ("low_runway_confidence", "cash runway > 8 quarters", {"runway_confidence": None}),
        ("severe_revenue_decline", "revenue_yoy > -20%", {"revenue_yoy": None}),
        ("negative_fcf", "free_cash_flow_ttm >= 0", {"fcf_ttm": None}),
    ):
        if gate_id in long_term_gates:
            factors.append(Factor(gate_id, (text,), cleared))
    return factors
