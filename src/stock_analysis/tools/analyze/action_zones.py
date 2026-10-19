"""Action zone computation with ATR-based price levels."""

from typing import Any

from stock_analysis.tools.analyze.signals import classify_unprofitability
from stock_analysis.utils.helpers import (
    format_level_distance_label,
    safe_round,
)
from stock_analysis.utils.valuation_basis import (
    instrument_supported,
    is_unverified,
    metric_reason,
    usable_metric,
)


def build_position_sizing_range(
    canonical_sizing: dict[str, Any] | None,
    regime_classification: str | None = None,
) -> dict[str, Any] | None:
    """Project legacy zone fields from the card's validated whole-share sizing."""
    if canonical_sizing is None or canonical_sizing.get("full") is None:
        return None
    assumptions, limits = canonical_sizing["assumptions"], canonical_sizing["limits"]
    starter, full = canonical_sizing["starter"], canonical_sizing["full"]
    account_size = assumptions["account_size"]
    normalized_regime = "medium" if regime_classification == "moderate" else regime_classification
    _, _, base = _base_sizing_range(normalized_regime)
    return {
        "suggested_pct_range": [starter["allocation_cap_pct"], full["allocation_cap_pct"]],
        "max_pct": full["allocation_cap_pct"],
        "starter_pct": starter["allocation_cap_pct"],
        "rationale": base["rationale"],
        "risk_per_trade_pct": limits["risk_per_trade_pct"],
        "profile_cap_pct": limits["max_position_pct"],
        "dollars_for_account": {
            "min": starter["position_dollars"], "max": full["position_dollars"],
            "portfolio_assumption": account_size,
        } if account_size is not None else None,
        "shares_range": {
            "min": starter["whole_shares"], "max": full["whole_shares"], "at_price": assumptions["entry_price"],
        } if account_size is not None else None,
        "stop_implied_max": {
            "pct": full["allocation_cap_pct"], "dollars_for_account": full["position_dollars"],
            "risk_per_trade_pct": limits["risk_per_trade_pct"],
            "stop_distance_pct": assumptions["stop_distance_fraction"] * 100,
        },
    }


def _base_sizing_range(regime_classification: str | None) -> tuple[float, float, dict[str, Any]]:
    """Build base position sizing range by risk regime."""
    if regime_classification == "extreme":
        pct_min, pct_max = 0.5, 3.0
        rationale = "extreme_risk_regime"
    elif regime_classification == "high":
        pct_min, pct_max = 2.0, 6.0
        rationale = "high_risk_regime"
    elif regime_classification == "medium":
        pct_min, pct_max = 3.0, 8.0
        rationale = "medium_risk_regime"
    else:
        pct_min, pct_max = 3.0, 10.0
        rationale = "low_risk_regime"

    return pct_min, pct_max, {
        "suggested_pct_range": [pct_min, pct_max],
        "max_pct": pct_max,
        "rationale": rationale,
    }


def _build_level_distance_views(
    current_price: float,
    levels: dict[str, float | None],
) -> tuple[
    dict[str, float | None],
    dict[str, float | None],
    dict[str, str | None],
    dict[str, str | None],
]:
    """Build distance and display views for action-zone levels."""
    distance_to_levels: dict[str, float | None] = {}
    price_vs_levels: dict[str, float | None] = {}

    for level_name, level_price in levels.items():
        if level_price is not None and current_price > 0:
            distance_to_levels[level_name] = round(
                (level_price - current_price) / current_price, 4
            )
        else:
            distance_to_levels[level_name] = None

        if level_price is not None and level_price != 0:
            price_vs_levels[level_name] = round(
                (current_price / level_price) - 1, 4
            )
        else:
            price_vs_levels[level_name] = None

    distance_labels = {
        level_name: format_level_distance_label(pct)
        for level_name, pct in distance_to_levels.items()
    }
    level_vs_current_labels = distance_labels.copy()
    return distance_to_levels, price_vs_levels, distance_labels, level_vs_current_labels


# Valuation headwind thresholds: the gate fires strictly above each (the condition text quotes them, BL-E1).
PE_HEADWIND_ABOVE = 50
PEG_HEADWIND_ABOVE = 3.0
SALES_MULTIPLE_HEADWIND_ABOVE = 10

# The neutral, diagnostic name of each uncapped zone classification (plan §7b): never an entry code.
PRICE_ZONES = {"strong_buy": "near_52w_low", "reduce": "near_52w_high", "accumulate": "below_sma200",
               "hold_bullish": "above_sma50", "hold_neutral": "at_or_below_sma50", "undetermined": "undetermined"}


def _classify_current_zone(
    current_price: float,
    levels: dict[str, float | None],
    sma_50: float | None,
    sma_200: float | None,
) -> str:
    """Classify the current price into an action zone.

    Level ordering invariant: strong_buy ≤ accumulate ≤ hold ≤ reduce.
    `reduce_above` is checked before the SMA200 fallback so a price that's both
    below SMA200 and above the `52w_high - 1·ATR` threshold doesn't get
    mis-classified as "accumulate".
    """
    strong_buy_level = levels.get("strong_buy_below")
    reduce_level = levels.get("reduce_above")

    if strong_buy_level is not None and current_price <= strong_buy_level:
        return "strong_buy"
    if reduce_level is not None and current_price >= reduce_level:
        return "reduce"
    if sma_200 and current_price < sma_200:
        return "accumulate"
    if sma_50 and current_price > sma_50:
        return "hold_bullish"
    if sma_50 and current_price <= sma_50:
        return "hold_neutral"
    return "undetermined"


def _build_action_levels(
    current_price: float,
    sma_200: float | None,
    week_52_low: float | None,
    week_52_high: float | None,
    atr_val: float | None,
    atr_pct: float | None,
    volatility_band: float,
    is_extreme_regime: bool,
    is_high_regime: bool,
) -> tuple[
    dict[str, float | None],
    dict[str, str | None],
    dict[str, Any] | None,
    float,
]:
    """Build ATR-adjusted action levels and stop metadata."""
    levels: dict[str, float | None] = {}
    basis: dict[str, str | None] = {}

    if week_52_low is not None:
        levels["strong_buy_below"] = round(week_52_low * (1 + volatility_band), 2)
        basis["strong_buy_below"] = "52w_low_plus_1atr" if atr_pct is not None else "52w_low_plus_2pct_fallback"
    else:
        levels["strong_buy_below"] = None
        basis["strong_buy_below"] = None

    if sma_200 is not None:
        levels["accumulate_near"] = round(sma_200, 2)
        basis["accumulate_near"] = "sma_200"
    else:
        levels["accumulate_near"] = None
        basis["accumulate_near"] = None

    if week_52_high is not None:
        levels["reduce_above"] = round(week_52_high * (1 - volatility_band), 2)
        basis["reduce_above"] = "52w_high_minus_1atr" if atr_pct is not None else "52w_high_minus_2pct_fallback"
    else:
        levels["reduce_above"] = None
        basis["reduce_above"] = None

    stop_multiple_used: float = 2.5 if is_extreme_regime else 2.0
    stop_min_multiple_required: float = 2.0 if is_high_regime else 1.5

    stop_calculation: dict[str, Any] | None = None
    if atr_val is not None:
        stop_price = round(current_price - (atr_val * stop_multiple_used), 2)
        levels["stop_loss"] = stop_price
        basis["stop_loss"] = f"current_minus_{stop_multiple_used}atr"

        stop_distance_pct = round(abs(stop_price - current_price) / current_price, 4)
        stop_calculation = {
            "stop_price": stop_price,
            "stop_distance_pct": stop_distance_pct,
            "atr_pct": safe_round(atr_pct, 4),
            "stop_multiple_used": stop_multiple_used,
            "min_multiple_required": stop_min_multiple_required,
        }
    else:
        levels["stop_loss"] = None
        basis["stop_loss"] = None

    return levels, basis, stop_calculation, stop_min_multiple_required


def _build_valuation_assessment(
    fund_data: dict[str, Any],
    signals: dict[str, list[str]] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Build the valuation gate used to temper action-zone guidance."""
    val = fund_data.get("valuation", {})
    yield_m = fund_data.get("yield_metrics", {})
    profit = fund_data.get("profitability", {})

    # WP4: only usable metrics (defined, applicable, verified basis) reach the gate.
    pe = usable_metric(val, "pe_trailing")
    peg = usable_metric(val, "peg_ratio")
    ps = usable_metric(val, "ps_trailing")
    ev_to_sales = usable_metric(val, "ev_to_sales")
    fcf_yield = usable_metric(yield_m, "fcf_yield")
    earnings_yield = usable_metric(yield_m, "earnings_yield")
    net_margin = profit.get("net_margin")

    bearish_signals = signals.get("bearish", []) if signals else []
    _unprofitable = classify_unprofitability(
        net_margin=net_margin,
        trailing_eps=val.get("trailing_eps"),
        signaled_unprofitable="unprofitable" in bearish_signals,
        pe_trailing=pe,
        pe_unverified=is_unverified(val, "pe_trailing"),
        has_negative_fcf_signal="negative_free_cash_flow" in bearish_signals,
    )
    is_unprofitable = (
        _unprofitable.by_margin
        or _unprofitable.by_eps
        or _unprofitable.by_signal
        or _unprofitable.by_no_pe_with_negative_fcf
    )

    valuation_gate: str = "neutral"
    valuation_gate_reasons: list[str] = []
    valuation_basis: str | None = None
    warnings: list[str] = []

    if is_unprofitable:
        sales_multiple: float | None = None
        # EV <= 0 (cash exceeds cap + debt): the sales multiple says nothing about
        # cheapness, so P/S is not a fallback either (R4.5).
        ev_nonpositive = metric_reason(val, "ev_to_sales") == "ev_nonpositive"
        if ev_to_sales is not None:
            sales_multiple = ev_to_sales
            valuation_basis = "ev_to_sales"
        elif ps is not None and not ev_nonpositive:
            sales_multiple = ps
            valuation_basis = "ps_trailing"

        if sales_multiple is not None:
            metric_label = "EV/S" if valuation_basis == "ev_to_sales" else "P/S"
            if 0 < sales_multiple < 3:
                valuation_gate = "attractive"
                valuation_gate_reasons.append(
                    f"{metric_label} {sales_multiple:.1f} reasonable for growth"
                )
            elif sales_multiple > SALES_MULTIPLE_HEADWIND_ABOVE:
                valuation_gate = "headwind"
                valuation_gate_reasons.append(
                    f"{metric_label} {sales_multiple:.1f} very elevated for unprofitable company"
                )
                warnings.append("valuation_extended_high_sales_multiple_unprofitable")
            else:
                valuation_gate_reasons.append(f"{metric_label} {sales_multiple:.1f} moderate")
        else:
            valuation_gate = "unknown"
            valuation_basis = "unknown"
            excluded = [
                f"{key}_{reason}"
                for key in ("ev_to_sales", "ps_trailing")
                if (reason := metric_reason(val, key))
            ]
            valuation_gate_reasons.extend(excluded or ["sales_multiple_unavailable_cannot_evaluate"])
            warnings.append("valuation_gate_unknown_sales_multiple_missing")
    else:
        if fcf_yield is not None and fcf_yield > 0.05:
            valuation_gate = "attractive"
            valuation_basis = "fcf_yield"
            valuation_gate_reasons.append(f"FCF yield {fcf_yield*100:.1f}% attractive")
        elif earnings_yield is not None and earnings_yield > 0.04:
            valuation_gate = "attractive"
            valuation_basis = "earnings_yield"
            valuation_gate_reasons.append(f"Earnings yield {earnings_yield*100:.1f}% reasonable")
        elif peg is not None and 0 < peg < 1.0:
            valuation_gate = "attractive"
            valuation_basis = "peg_ratio"
            valuation_gate_reasons.append(f"PEG {peg:.2f} suggests undervaluation")

        if pe is not None and pe > PE_HEADWIND_ABOVE:
            valuation_gate = "headwind"
            valuation_basis = "pe_trailing"
            valuation_gate_reasons.append(f"P/E {pe:.1f} very elevated")
            warnings.append("valuation_extended_high_pe")
        if peg is not None and peg > PEG_HEADWIND_ABOVE:
            if valuation_gate != "headwind":
                valuation_gate = "headwind"
            valuation_basis = "peg_ratio"
            valuation_gate_reasons.append(f"PEG {peg:.2f} suggests overvaluation")
            warnings.append("valuation_extended_high_peg")

        if valuation_gate == "neutral" and valuation_basis is None:
            if pe is not None:
                valuation_basis = "pe_trailing"
            elif peg is not None:
                valuation_basis = "peg_ratio"
            elif fcf_yield is not None:
                valuation_basis = "fcf_yield"
            elif ps is not None:
                valuation_basis = "ps_trailing"

        if (
            valuation_gate == "neutral"
            and pe is None
            and fcf_yield is None
            and earnings_yield is None
            and peg is None
            and ps is None
        ):
            valuation_gate = "unknown"
            valuation_basis = "unknown"
            excluded = [
                f"{key}_{reason}"
                for section, key in ((val, "pe_trailing"), (yield_m, "fcf_yield"), (val, "ps_trailing"))
                if (reason := metric_reason(section, key))
            ]
            valuation_gate_reasons.extend(excluded or ["insufficient_valuation_data"])
            warnings.append("valuation_gate_unknown_no_metrics")

    return {
        "gate": valuation_gate,
        "basis": valuation_basis,
        "reasons": valuation_gate_reasons if valuation_gate_reasons else None,
        "is_unprofitable": is_unprofitable,
    }, warnings


def _apply_valuation_gate_to_zone(
    current_zone: str | None,
    valuation_gate: str,
) -> tuple[str | None, str | None]:
    """Downgrade action zones when valuation is a headwind."""
    if valuation_gate != "headwind":
        return current_zone, None
    if current_zone == "strong_buy":
        return "accumulate", "zone_downgraded_valuation_headwind"
    if current_zone == "accumulate":
        return "hold_neutral", "zone_downgraded_valuation_headwind"
    return current_zone, None


def _missing_price_response() -> dict[str, Any]:
    """Build action-zone payload for missing current price."""
    return {
        "price_zone": None,
        "current_zone": None,
        "levels": {},
        "distance_to_levels": {},
        "price_vs_levels": {},
        "distance_labels": {},
        "level_vs_current_labels": {},
        "basis": {},
        "stop_calculation": None,
        "zone_warnings": ["missing_price"],
        "method": "atr_based_v1",
    }


def _append_stop_warning(
    zone_warnings: list[str],
    current_price: float,
    stop_price: float | None,
    atr_pct: float | None,
    stop_min_multiple_required: float,
) -> None:
    """Append stop-distance warning when the stop is too tight."""
    stop_distance = (stop_price - current_price) / current_price if stop_price is not None else None
    if stop_distance is not None and atr_pct is not None:
        stop_distance_pct_val = abs(stop_distance)
        if stop_distance_pct_val < (stop_min_multiple_required * atr_pct):
            zone_warnings.append("stop_too_tight")
    elif stop_distance is not None and stop_distance > -0.05:
        zone_warnings.append("stop_too_tight")


def _append_regime_warnings(
    zone_warnings: list[str],
    is_extreme_regime: bool,
    is_high_regime: bool,
) -> None:
    """Append the risk-regime warnings."""
    if is_extreme_regime:
        zone_warnings.append("extreme_risk_regime")
    elif is_high_regime:
        zone_warnings.append("high_risk_regime")


def build_action_zones(
    current_price: float | None,
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
    fund_data: dict[str, Any],
    risk_regime: dict[str, Any] | None = None,
    signals: dict[str, list[str]] | None = None,
    investor_profile: dict[str, Any] | None = None,
    support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build action zones with ATR-based price levels.

    ``support`` (WP4, D5): an unsupported instrument gets a ``not_applicable``
    valuation gate. Sizing is attached after the canonical decision is built.
    None keeps the pre-WP4 valuation behavior (direct callers and tests).

    Zones are volatility-adjusted using ATR, not arbitrary percentages.
    Risk regime affects zone interpretation and warnings.
    """
    if current_price is None:
        return _missing_price_response()

    # Extract risk regime classification
    regime_classification = (
        risk_regime.get("classification") if risk_regime else None
    )
    is_extreme_regime = regime_classification == "extreme"
    is_high_regime = regime_classification in ("high", "extreme")

    # Extract needed values
    ma = tech_data.get("moving_averages", {})
    price_pos = tech_data.get("price_position", {})
    atr_data = risk_data.get("atr", {})

    sma_50 = ma.get("sma_50")
    sma_200 = ma.get("sma_200")
    week_52_low = price_pos.get("week_52_low")
    week_52_high = price_pos.get("week_52_high")
    atr_val = atr_data.get("value")
    atr_pct = atr_data.get("as_pct_of_price")

    zone_warnings: list[str] = []

    # Use ATR for volatility-adjusted zones (default to 2% if missing)
    if atr_pct is not None:
        volatility_band = atr_pct
    else:
        volatility_band = 0.02
        zone_warnings.append("atr_unavailable_using_default")

    levels, basis, stop_calculation, stop_min_multiple_required = _build_action_levels(
        current_price=current_price,
        sma_200=sma_200,
        week_52_low=week_52_low,
        week_52_high=week_52_high,
        atr_val=atr_val,
        atr_pct=atr_pct,
        volatility_band=volatility_band,
        is_extreme_regime=is_extreme_regime,
        is_high_regime=is_high_regime,
    )

    (
        distance_to_levels,
        price_vs_levels,
        distance_labels,
        level_vs_current_labels,
    ) = _build_level_distance_views(current_price, levels)

    uncapped_zone = _classify_current_zone(
        current_price=current_price,
        levels=levels,
        sma_50=sma_50,
        sma_200=sma_200,
    )
    current_zone: str | None = uncapped_zone

    # Apply regime-aware zone capping
    # In extreme risk regime, cap "strong_buy" to "accumulate" - be more cautious
    if is_extreme_regime and current_zone == "strong_buy":
        current_zone = "accumulate"
        zone_warnings.append("zone_capped_due_to_extreme_risk")

    if support is not None and not instrument_supported(support):
        valuation_assessment: dict[str, Any] = {
            "gate": "not_applicable",
            "basis": None,
            "reasons": [(support.get("instrument") or {}).get("reason") or "unsupported_instrument"],
            "is_unprofitable": False,
        }
        valuation_warnings: list[str] = []
    else:
        valuation_assessment, valuation_warnings = _build_valuation_assessment(
            fund_data=fund_data,
            signals=signals,
        )
    zone_warnings.extend(valuation_warnings)

    current_zone, valuation_zone_warning = _apply_valuation_gate_to_zone(
        current_zone=current_zone,
        valuation_gate=valuation_assessment["gate"],
    )
    if valuation_zone_warning is not None:
        zone_warnings.append(valuation_zone_warning)

    _append_stop_warning(
        zone_warnings=zone_warnings,
        current_price=current_price,
        stop_price=levels.get("stop_loss"),
        atr_pct=atr_pct,
        stop_min_multiple_required=stop_min_multiple_required,
    )
    _append_regime_warnings(
        zone_warnings=zone_warnings,
        is_extreme_regime=is_extreme_regime,
        is_high_regime=is_high_regime,
    )

    return {
        "price_zone": PRICE_ZONES[uncapped_zone],
        "current_zone": current_zone,  # the capped label: a pre-gate candidate input, never emitted (plan §7b)
        "levels": levels,
        "distance_to_levels": distance_to_levels,
        "price_vs_levels": price_vs_levels,
        "distance_labels": distance_labels,
        "level_vs_current_labels": level_vs_current_labels,
        "basis": basis,
        "stop_calculation": stop_calculation,
        "position_sizing_range": None,
        "valuation_assessment": valuation_assessment,
        "zone_warnings": zone_warnings or [],
        "method": "atr_valuation_v2",
    }


def apply_dip_gates_to_action_zones(
    action_zones: dict[str, Any],
    dip_assessment: dict[str, Any],
    risk_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Cap the pre-gate candidate's zone input (current_zone) for a falling knife, an extended decline or an
    extreme risk regime, recording each cap in zone_warnings.
    """
    current_zone = action_zones.get("current_zone")
    zone_warnings = list(action_zones.get("zone_warnings") or [])
    dip_type = (dip_assessment.get("dip_classification") or {}).get("type")
    risk_label = risk_regime.get("classification") if risk_regime else None

    def _cap_zone(reason: str) -> None:
        nonlocal current_zone
        if current_zone in ("strong_buy", "accumulate"):
            current_zone = "hold_neutral"
        if reason not in zone_warnings:
            zone_warnings.append(reason)

    if dip_type == "falling_knife":
        _cap_zone("zone_capped_falling_knife")
    elif dip_type == "extended_decline" and current_zone == "strong_buy":
        current_zone = "accumulate"
        if "zone_capped_extended_decline" not in zone_warnings:
            zone_warnings.append("zone_capped_extended_decline")

    if risk_label == "extreme":
        _cap_zone("zone_capped_extreme_risk")

    action_zones["current_zone"] = current_zone
    action_zones["zone_warnings"] = zone_warnings
    return action_zones


def zone_after_caps(uncapped: str | None, *, extreme: bool, valuation_gate: str | None, dip_type: str | None) -> str | None:
    """The capped zone the pre-gate candidate reads, replayed from the uncapped label: the build-time extreme
    cap, the valuation headwind downgrade, then apply_dip_gates_to_action_zones (BL-E1 counterfactuals)."""
    zone = "accumulate" if extreme and uncapped == "strong_buy" else uncapped
    zone, _ = _apply_valuation_gate_to_zone(zone, valuation_gate or "")
    if dip_type == "falling_knife" and zone in ("strong_buy", "accumulate"):
        zone = "hold_neutral"
    elif dip_type == "extended_decline" and zone == "strong_buy":
        zone = "accumulate"
    if extreme and zone in ("strong_buy", "accumulate"):
        zone = "hold_neutral"
    return zone
