"""Action zone computation with ATR-based price levels."""

from typing import Any

from stock_mcp.utils.helpers import (
    format_level_distance_label,
)


def build_action_zones(
    current_price: float | None,
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
    fund_data: dict[str, Any],
    risk_regime: dict[str, Any] | None = None,
    signals: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Build action zones with ATR-based price levels.

    Zones are volatility-adjusted using ATR, not arbitrary percentages.
    Risk regime affects zone interpretation and warnings.
    """
    if current_price is None:
        return {
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

    # Calculate levels
    levels: dict[str, float | None] = {}
    basis: dict[str, str | None] = {}

    # Strong buy: near 52-week low + 1 ATR buffer
    if week_52_low is not None:
        levels["strong_buy_below"] = round(week_52_low * (1 + volatility_band), 2)
        basis["strong_buy_below"] = "52w_low_plus_1atr"
    else:
        levels["strong_buy_below"] = None
        basis["strong_buy_below"] = None

    # Accumulate: below SMA200 or approaching SMA200
    if sma_200 is not None:
        levels["accumulate_near"] = round(sma_200, 2)
        basis["accumulate_near"] = "sma_200"
    else:
        levels["accumulate_near"] = None
        basis["accumulate_near"] = None

    # Take profit: approaching 52-week high - 1 ATR buffer
    if week_52_high is not None:
        levels["reduce_above"] = round(week_52_high * (1 - volatility_band), 2)
        basis["reduce_above"] = "52w_high_minus_1atr"
    else:
        levels["reduce_above"] = None
        basis["reduce_above"] = None

    # Stop loss: 2 ATR below current price (2.5 ATR for extreme regime)
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
            "atr_pct": round(atr_pct, 4) if atr_pct else None,
            "stop_multiple_used": stop_multiple_used,
            "min_multiple_required": stop_min_multiple_required,
        }
    else:
        levels["stop_loss"] = None
        basis["stop_loss"] = None

    # Calculate distances as percentages from current price
    distance_to_levels: dict[str, float | None] = {}
    for level_name, level_price in levels.items():
        if level_price is not None and current_price > 0:
            distance_to_levels[level_name] = round(
                (level_price - current_price) / current_price, 4
            )
        else:
            distance_to_levels[level_name] = None

    # Calculate price vs level (negative means price below level)
    price_vs_levels: dict[str, float | None] = {}
    for level_name, level_price in levels.items():
        if level_price is not None and level_price != 0:
            price_vs_levels[level_name] = round(
                (current_price / level_price) - 1, 4
            )
        else:
            price_vs_levels[level_name] = None

    # Preformatted distance labels for clearer display (level vs current)
    distance_labels = {
        level_name: format_level_distance_label(pct)
        for level_name, pct in distance_to_levels.items()
    }

    # Separate field for renderer semantics (stop loss vs other levels)
    level_vs_current_labels = distance_labels.copy()

    # Determine current zone
    current_zone: str | None = None
    strong_buy_level = levels.get("strong_buy_below")
    reduce_level = levels.get("reduce_above")

    if strong_buy_level is not None and current_price <= strong_buy_level:
        current_zone = "strong_buy"
    elif sma_200 and current_price < sma_200:
        current_zone = "accumulate"
    elif reduce_level is not None and current_price >= reduce_level:
        current_zone = "reduce"
    elif sma_50 and current_price > sma_50:
        current_zone = "hold_bullish"
    elif sma_50 and current_price <= sma_50:
        current_zone = "hold_neutral"
    else:
        current_zone = "undetermined"

    # Apply regime-aware zone capping
    # In extreme risk regime, cap "strong_buy" to "accumulate" - be more cautious
    if is_extreme_regime and current_zone == "strong_buy":
        current_zone = "accumulate"
        zone_warnings.append("zone_capped_due_to_extreme_risk")

    # Valuation-aware accumulate gates for long-term investors
    val = fund_data.get("valuation", {})
    yield_m = fund_data.get("yield_metrics", {})
    profit = fund_data.get("profitability", {})

    pe = val.get("pe_trailing")
    peg = val.get("peg_ratio")
    ps = val.get("ps_trailing")
    ev_to_sales = val.get("ev_to_sales")  # Prefer over P/S when debt/cash material
    fcf_yield = yield_m.get("fcf_yield")
    earnings_yield = yield_m.get("earnings_yield")
    net_margin = profit.get("net_margin")

    # Determine if company is unprofitable (use P/S instead of P/E)
    # Use multiple signals for robust detection when data is sparse
    bearish_signals = signals.get("bearish", []) if signals else []
    is_signaled_unprofitable = "unprofitable" in bearish_signals
    has_negative_fcf = "negative_free_cash_flow" in bearish_signals

    # Detect unprofitability from EXPLICIT negative signals only
    # Never classify as unprofitable purely from missing data (that's "fundamentals_missing")
    # 1. Negative net margin (explicit)
    # 2. Signal system detected "unprofitable" (based on EPS or other metrics)
    # 3. Negative trailing EPS (explicit)
    # 4. No P/E + negative FCF (strong inference: negative earnings + burning cash)
    trailing_eps = val.get("trailing_eps")
    is_unprofitable = False
    if (net_margin is not None and net_margin < 0) or is_signaled_unprofitable:
        is_unprofitable = True
    elif trailing_eps is not None and trailing_eps <= 0:
        # Explicit negative or zero EPS
        is_unprofitable = True
    elif pe is None and has_negative_fcf:
        # No P/E (likely negative earnings) + negative FCF = strong inference
        is_unprofitable = True
    # NOTE: If pe=None and net_margin=None and trailing_eps=None, we DON'T assume unprofitable
    # That's a data gap, not evidence of losses. Let fundamentals_missing handle it.

    # Valuation gate: determines if valuation supports accumulation
    # attractive = valuation supports adding, neutral = ok to hold, headwind = valuation stretched
    valuation_gate: str = "neutral"
    valuation_gate_reasons: list[str] = []

    # Track which valuation metric drives the gate for auditability
    valuation_basis: str | None = None

    if is_unprofitable:
        # For unprofitable companies, prefer EV/S over P/S
        # EV/S is better when debt/cash position is material (adjusts for net debt)
        sales_multiple: float | None = None
        if ev_to_sales is not None:
            sales_multiple = ev_to_sales
            valuation_basis = "ev_to_sales"
        elif ps is not None:
            sales_multiple = ps
            valuation_basis = "ps_trailing"

        if sales_multiple is not None:
            metric_label = "EV/S" if valuation_basis == "ev_to_sales" else "P/S"
            if sales_multiple < 3:
                valuation_gate = "attractive"
                valuation_gate_reasons.append(
                    f"{metric_label} {sales_multiple:.1f} reasonable for growth"
                )
            elif sales_multiple > 10:
                valuation_gate = "headwind"
                valuation_gate_reasons.append(
                    f"{metric_label} {sales_multiple:.1f} very elevated for unprofitable company"
                )
                zone_warnings.append("valuation_extended_high_sales_multiple_unprofitable")
            else:
                valuation_gate_reasons.append(f"{metric_label} {sales_multiple:.1f} moderate")
        else:
            # Neither EV/S nor P/S available - cannot evaluate valuation
            valuation_gate = "unknown"
            valuation_basis = "unknown"
            valuation_gate_reasons.append("sales_multiple_unavailable_cannot_evaluate")
            zone_warnings.append("valuation_gate_unknown_sales_multiple_missing")
    else:
        # For profitable companies, use FCF yield / earnings yield / PEG
        if fcf_yield is not None and fcf_yield > 0.05:
            valuation_gate = "attractive"
            valuation_basis = "fcf_yield"
            valuation_gate_reasons.append(f"FCF yield {fcf_yield*100:.1f}% attractive")
        elif earnings_yield is not None and earnings_yield > 0.04:
            valuation_gate = "attractive"
            valuation_basis = "earnings_yield"
            valuation_gate_reasons.append(f"Earnings yield {earnings_yield*100:.1f}% reasonable")
        elif peg is not None and peg < 1.0:
            valuation_gate = "attractive"
            valuation_basis = "peg_ratio"
            valuation_gate_reasons.append(f"PEG {peg:.2f} suggests undervaluation")

        # Check for headwinds
        if pe is not None and pe > 50:
            valuation_gate = "headwind"
            valuation_basis = "pe_trailing"
            valuation_gate_reasons.append(f"P/E {pe:.1f} very elevated")
            zone_warnings.append("valuation_extended_high_pe")
        if peg is not None and peg > 3.0:
            if valuation_gate != "headwind":
                valuation_gate = "headwind"
            valuation_basis = "peg_ratio"
            valuation_gate_reasons.append(f"PEG {peg:.2f} suggests overvaluation")
            zone_warnings.append("valuation_extended_high_peg")

        # For neutral gate with profitable company, set basis to primary metric used
        if valuation_gate == "neutral" and valuation_basis is None:
            if pe is not None:
                valuation_basis = "pe_trailing"
            elif peg is not None:
                valuation_basis = "peg_ratio"
            elif fcf_yield is not None:
                valuation_basis = "fcf_yield"
            elif ps is not None:
                valuation_basis = "ps_trailing"

        # If we couldn't assess valuation at all (no P/E, no FCF yield, no earnings yield, no PEG)
        # AND P/S is also missing, gate is unknown
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
            valuation_gate_reasons.append("insufficient_valuation_data")
            zone_warnings.append("valuation_gate_unknown_no_metrics")

    # Apply valuation gate to zone recommendation
    # If valuation is headwind, downgrade strong_buy to accumulate, accumulate to hold
    if valuation_gate == "headwind":
        if current_zone == "strong_buy":
            current_zone = "accumulate"
            zone_warnings.append("zone_downgraded_valuation_headwind")
        elif current_zone == "accumulate":
            current_zone = "hold_neutral"
            zone_warnings.append("zone_downgraded_valuation_headwind")

    valuation_assessment = {
        "gate": valuation_gate,
        "basis": valuation_basis,  # Which metric drives the gate (ev_to_sales, ps_trailing, pe_trailing, etc.)
        "reasons": valuation_gate_reasons if valuation_gate_reasons else None,
        "is_unprofitable": is_unprofitable,
    }

    # Check for stop_too_tight warning - volatility-aware using ATR multiple
    stop_distance = distance_to_levels.get("stop_loss")
    if stop_distance is not None and atr_pct is not None:
        stop_distance_pct_val = abs(stop_distance)
        if stop_distance_pct_val < (stop_min_multiple_required * atr_pct):
            zone_warnings.append("stop_too_tight")
    elif stop_distance is not None and stop_distance > -0.05:
        # Fallback: fixed 5% if no ATR available
        zone_warnings.append("stop_too_tight")

    # Add regime-specific warnings
    if is_extreme_regime:
        zone_warnings.append("extreme_risk_regime:prefer_small_position")
    elif is_high_regime:
        zone_warnings.append("high_risk_regime:size_conservatively")

    # Position sizing range based on risk regime
    # These are suggested ranges as % of portfolio
    # Default portfolio value for dollar calculations
    default_portfolio_value = 50000.0

    position_sizing_range: dict[str, Any]
    if is_extreme_regime:
        pct_min, pct_max = 0.5, 3.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "extreme_risk_requires_minimal_exposure",
        }
    elif is_high_regime:
        pct_min, pct_max = 2.0, 6.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "high_risk_warrants_conservative_sizing",
        }
    elif regime_classification == "medium":
        pct_min, pct_max = 3.0, 8.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "moderate_risk_standard_sizing",
        }
    else:
        # low risk or unknown
        pct_min, pct_max = 3.0, 10.0
        position_sizing_range = {
            "suggested_pct_range": [pct_min, pct_max],
            "max_pct": pct_max,
            "rationale": "low_risk_allows_larger_positions",
        }

    # Add dollar amounts for default portfolio size
    dollar_min = round(default_portfolio_value * pct_min / 100, 0)
    dollar_max = round(default_portfolio_value * pct_max / 100, 0)
    position_sizing_range["dollars_for_50k"] = {
        "min": dollar_min,
        "max": dollar_max,
        "portfolio_assumption": default_portfolio_value,
    }

    # Add shares range at current price
    if current_price and current_price > 0:
        shares_min = int(dollar_min / current_price)
        shares_max = int(dollar_max / current_price)
        position_sizing_range["shares_range"] = {
            "min": shares_min,
            "max": shares_max,
            "at_price": current_price,
        }

    # Add stop-implied max size (risk 1% of portfolio per trade)
    stop_distance = stop_calculation.get("stop_distance_pct") if stop_calculation else None
    if stop_distance and stop_distance > 0:
        # 1% risk rule: max_position = (portfolio * 0.01) / stop_distance
        risk_pct = 1.0  # Risk 1% of portfolio on a single trade
        stop_implied_pct = round((risk_pct / stop_distance) * 100, 1)
        # Cap at the max_pct from risk regime
        if stop_implied_pct > pct_max:
            stop_implied_pct = pct_max
        stop_implied_dollars = round(default_portfolio_value * stop_implied_pct / 100, 0)
        position_sizing_range["stop_implied_max"] = {
            "pct": stop_implied_pct,
            "dollars_for_50k": stop_implied_dollars,
            "risk_per_trade_pct": risk_pct,
            "stop_distance_pct": round(stop_distance * 100, 1),
        }

    return {
        "current_zone": current_zone,
        "levels": levels,
        "distance_to_levels": distance_to_levels,
        "price_vs_levels": price_vs_levels,
        "distance_labels": distance_labels,
        "level_vs_current_labels": level_vs_current_labels,
        "basis": basis,
        "stop_calculation": stop_calculation,
        "position_sizing_range": position_sizing_range,
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
    Apply dip-aware guards to action zones to avoid conflicting guidance.
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
    elif dip_type == "extended_decline":
        if current_zone == "strong_buy":
            current_zone = "accumulate"
            if "zone_capped_extended_decline" not in zone_warnings:
                zone_warnings.append("zone_capped_extended_decline")

    if risk_label == "extreme":
        _cap_zone("zone_capped_extreme_risk")

    action_zones["current_zone"] = current_zone
    action_zones["zone_warnings"] = zone_warnings
    return action_zones
