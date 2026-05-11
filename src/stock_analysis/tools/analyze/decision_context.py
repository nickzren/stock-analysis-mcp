"""Decision context: triggers, thesis checkpoints, and actionable conditions."""

from typing import Any

from stock_analysis.tools.analyze.signals import (
    vol_threshold_for_improvement,
)
from stock_analysis.utils.helpers import fcf_label_from_cashflow, safe_float


def build_relative_performance(
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Build relative performance vs benchmark (SPY).

    Uses aligned data from risk_metrics beta calculation.
    """
    returns = tech_data.get("returns", {})
    beta_data = risk_data.get("beta", {})
    benchmark_returns = risk_data.get("benchmark_returns", {})

    # Get stock returns
    stock_return_1y = safe_float(returns.get("return_1y"))
    stock_return_3m = safe_float(returns.get("return_3m"))
    stock_return_1m = safe_float(returns.get("return_1m"))

    # Benchmark returns from risk_metrics benchmark fetch path
    benchmark_return_1y = safe_float(benchmark_returns.get("return_1y"))
    benchmark_return_3m = safe_float(benchmark_returns.get("return_3m"))
    benchmark_return_1m = safe_float(benchmark_returns.get("return_1m"))

    # Calculate alpha/outperformance when both series are available
    alpha_1y: float | None = None
    outperformed_1y: bool | None = None
    if stock_return_1y is not None and benchmark_return_1y is not None:
        alpha_1y = round(stock_return_1y - benchmark_return_1y, 4)
        outperformed_1y = bool(stock_return_1y > benchmark_return_1y)

    warnings: list[str] = []
    if benchmark_return_1y is None:
        warnings.append("benchmark_returns_not_available")
    if stock_return_1y is None:
        warnings.append("stock_return_1y_not_available")

    return {
        "stock_return_1y": stock_return_1y,
        "stock_return_3m": stock_return_3m,
        "stock_return_1m": stock_return_1m,
        "benchmark": "SPY",
        "benchmark_return_1y": benchmark_return_1y,
        "benchmark_return_3m": benchmark_return_3m,
        "benchmark_return_1m": benchmark_return_1m,
        "alpha_1y": alpha_1y,
        "outperformed_1y": outperformed_1y,
        "beta": beta_data.get("value"),
        "warnings": warnings or [],
    }


def _make_condition(
    id: str,
    condition: str,
    data_source: str,
    operator: str,
    target_value: float | None,
    current_value: float | None,
    threshold_str: str,
    current_str: str,
    next_update: str | None = None,
    priority: str | None = None,
) -> dict[str, Any]:
    """Build a machine-checkable condition dict."""
    result: dict[str, Any] = {
        "id": id,
        "condition": condition,
        "data_source": data_source,
        "operator": operator,
        "target_value": target_value,
        "current_value": current_value,
        "threshold": threshold_str,
        "current": current_str,
    }
    if next_update:
        result["next_update"] = next_update
    if priority:
        result["priority"] = priority
    return result


def _limit_list(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]] | None:
    """Return the first max_items entries, or None if the list is empty."""
    if not items:
        return None
    return items[:max_items]


def _simple_condition(
    condition: str,
    threshold: str,
    next_update: str | None,
) -> dict[str, Any]:
    """Build a legacy condition dict without machine-checkable fields."""
    return {
        "condition": condition,
        "threshold": threshold,
        "next_update": next_update,
    }


def _build_fundamental_conditions(
    *,
    is_unprofitable: bool,
    net_margin: float | None,
    fcf: float | None,
    revenue_yoy: float | None,
    next_earnings_date: str | None,
    bullish_list: list[str],
    bearish_list: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build fundamentals bullish/bearish monitor conditions."""
    fundamentals_bullish: list[dict[str, Any]] = []
    fundamentals_bearish: list[dict[str, Any]] = []

    if is_unprofitable:
        fundamentals_bullish.extend(
            [
                _make_condition(
                    id="returns_to_profitability",
                    condition="returns_to_profitability",
                    data_source="yfinance.info.profitMargins",
                    operator=">",
                    target_value=0.0,
                    current_value=net_margin,
                    threshold_str="net_margin > 0 for 2 consecutive quarters",
                    current_str=(
                        f"net_margin = {net_margin*100:.1f}%"
                        if net_margin is not None
                        else "net_margin = unknown"
                    ),
                    next_update=next_earnings_date,
                    priority="critical",
                ),
                _make_condition(
                    id="fcf_turns_positive",
                    condition="fcf_turns_positive",
                    data_source="yfinance.info.freeCashflow",
                    operator=">",
                    target_value=0.0,
                    current_value=fcf,
                    threshold_str="FCF > 0 for 2 consecutive quarters",
                    current_str=f"FCF = ${fcf/1e6:.0f}M" if fcf is not None else "FCF = unknown",
                    next_update=next_earnings_date,
                    priority="high",
                ),
                _make_condition(
                    id="revenue_stabilizes",
                    condition="revenue_stabilizes",
                    data_source="yfinance.info.revenueGrowth",
                    operator=">",
                    target_value=-0.10,
                    current_value=revenue_yoy,
                    threshold_str="revenue_yoy > -10%",
                    current_str=(
                        f"revenue_yoy = {revenue_yoy*100:.0f}%"
                        if revenue_yoy is not None
                        else "revenue_yoy = unknown"
                    ),
                    next_update=next_earnings_date,
                    priority="medium",
                ),
            ]
        )
        fundamentals_bearish.append(
            _make_condition(
                id="revenue_collapses",
                condition="revenue_collapses",
                data_source="yfinance.info.revenueGrowth",
                operator="<",
                target_value=-0.30,
                current_value=revenue_yoy,
                threshold_str="revenue_yoy < -30%",
                current_str=(
                    f"revenue_yoy = {revenue_yoy*100:.0f}%"
                    if revenue_yoy is not None
                    else "revenue_yoy = unknown"
                ),
                next_update=next_earnings_date,
            )
        )
        return fundamentals_bullish, fundamentals_bearish

    profitable_rules = (
        (
            bearish_list,
            fundamentals_bullish,
            "unprofitable",
            "returns_to_profitability",
            "net_margin > 0 for 2 consecutive quarters",
        ),
        (
            bullish_list,
            fundamentals_bearish,
            "profitable",
            "earnings_turn_negative",
            "net_margin < 0",
        ),
        (
            bearish_list,
            fundamentals_bullish,
            "negative_free_cash_flow",
            "fcf_turns_positive",
            "FCF > 0 for 2 consecutive quarters",
        ),
        (
            bullish_list,
            fundamentals_bearish,
            "positive_free_cash_flow",
            "fcf_turns_negative",
            "FCF < 0",
        ),
        (
            bullish_list,
            fundamentals_bearish,
            "high_growth",
            "growth_decelerates",
            "revenue_yoy < 15%",
        ),
        (
            bearish_list,
            fundamentals_bullish,
            "declining_growth",
            "growth_accelerates",
            "revenue_yoy > 10%",
        ),
    )
    for signal_list, output, signal, condition, threshold in profitable_rules:
        if signal in signal_list:
            output.append(_simple_condition(condition, threshold, next_earnings_date))

    return fundamentals_bullish, fundamentals_bearish


def build_decision_context(
    signals: dict[str, list[str]],
    tech_data: dict[str, Any],
    risk_data: dict[str, Any],
    events_data: dict[str, Any],
    fund_data: dict[str, Any],
    fundamentals_summary: dict[str, Any],
    action_zones: dict[str, Any],
    news_data: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    """
    Build multi-factor decision context explaining what would change the verdict.

    Organized by category with structured top triggers, next update schedules,
    and news catalyst keywords.
    """
    bullish_list = signals.get("bullish", [])
    bearish_list = signals.get("bearish", [])

    # Extract current values for concrete thresholds
    ma = tech_data.get("moving_averages", {})
    rsi_data = tech_data.get("rsi", {})
    current_price = tech_data.get("current_price")
    sma_200 = ma.get("sma_200")
    sma_50 = ma.get("sma_50")
    rsi_value = rsi_data.get("value")

    vol_data = risk_data.get("volatility", {})
    dd_data = risk_data.get("drawdown", {})
    annualized_vol = vol_data.get("annualized")
    max_dd = dd_data.get("max_1y")

    # Get earnings date for next_update
    earnings = events_data.get("earnings", {})
    next_earnings_date = earnings.get("next_date")
    days_until_earnings = earnings.get("days_until")

    # Get valuation assessment
    valuation_assessment = action_zones.get("valuation_assessment", {})
    valuation_gate = valuation_assessment.get("gate")
    is_unprofitable = valuation_assessment.get("is_unprofitable", False)

    # Get tilt from verdict for balancing triggers
    tilt = verdict.get("tilt", "neutral")

    # Get decomposed scores for top triggers
    decomposed = verdict.get("decomposed", {})
    setup_label = decomposed.get("setup")
    business_quality = decomposed.get("business_quality")
    business_quality_status = decomposed.get("business_quality_status", "unknown")
    risk_label = decomposed.get("risk")

    # Get component scores and weights for calculating score_delta
    components = verdict.get("components", {})
    weights_used = verdict.get("weights_used") or {}

    # Get valuation metrics for reasons
    val = fund_data.get("valuation", {})
    yield_m = fund_data.get("yield_metrics", {})
    profit = fund_data.get("profitability", {})
    cf = fund_data.get("cash_flow", {})
    growth = fund_data.get("growth", {})
    ps_trailing = val.get("ps_trailing")
    ps_source = val.get("ps_source")
    ev_to_sales = val.get("ev_to_sales")
    ev_to_sales_source = val.get("ev_to_sales_source")
    pe_trailing = val.get("pe_trailing")
    net_margin = profit.get("net_margin")
    fcf = cf.get("free_cash_flow_ttm")
    fcf_label = fcf_label_from_cashflow(cf)
    revenue_yoy = growth.get("revenue_yoy")

    # Get burn_metrics from fundamentals_summary (already computed with liquidity)
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}
    cash_runway_quarters = burn_metrics.get("cash_runway_quarters")
    runway_basis = burn_metrics.get("runway_basis")
    dilution_analysis = burn_metrics.get("dilution_analysis")

    # === STRUCTURED TOP TRIGGERS ===
    # Each trigger includes score_delta = component_score * weight_used
    # This shows actual contribution to final score, not just weight
    top_triggers: list[dict[str, Any]] = []

    # Helper to calculate score_delta from component
    def _calc_score_delta(category: str) -> float | None:
        """Calculate score contribution: component_score * weight_used.

        INVARIANT: sum of all score_deltas == score_raw (no rounding until display)
        """
        comp_score = components.get(category)
        weight = weights_used.get(category)
        if comp_score is not None and weight is not None:
            return comp_score * weight  # No rounding - preserve full precision
        return None

    def _aligned_score_delta(category: str, direction: str) -> float | None:
        """Only keep scoring drivers whose sign matches the trigger direction."""
        score_delta = _calc_score_delta(category)
        if score_delta is None:
            return None
        if direction == "bearish" and score_delta >= 0:
            return None
        if direction == "bullish" and score_delta <= 0:
            return None
        return score_delta

    def _make_trigger(
        id: str,
        category: str,
        direction: str,
        reason: str,
        *,
        scoring_key: str | None = None,
        next_update_earnings_date: str | None = None,
    ) -> dict[str, Any]:
        """Build a structured top/fallback trigger dict.

        `scoring_key` defaults to `category`; pass it to map e.g. 'valuation' triggers
        onto the 'fundamentals' scoring component.
        """
        key = scoring_key or category
        trigger: dict[str, Any] = {
            "id": id,
            "category": category,
            "direction": direction,
            "reason": reason,
            "component_score": components.get(key),
            "weight_used": weights_used.get(key),
            "score_delta": _aligned_score_delta(key, direction),
        }
        if next_update_earnings_date:
            trigger["next_update"] = {"event": "earnings", "date": next_update_earnings_date}
        return trigger

    # Add bearish triggers first (most important for risk awareness)
    # Collapse risk regime with supporting details to avoid double-counting
    if risk_label in ("high", "extreme"):
        # Build detailed reason with supporting factors
        risk_summary = verdict.get("inputs_used", {})
        vol_val = risk_summary.get("annualized_vol")
        dd_val = risk_summary.get("max_drawdown_1y")
        reason_parts = [f"risk_regime={risk_label}"]
        if vol_val is not None:
            reason_parts.append(f"vol={vol_val*100:.0f}%")
        if dd_val is not None:
            reason_parts.append(f"dd={dd_val*100:.0f}%")
        top_triggers.append(_make_trigger(
            "elevated_risk_regime",
            "risk",
            "bearish",
            " | ".join(reason_parts),
        ))

    if business_quality in ("poor", "unprofitable"):
        reason_parts = []
        if net_margin is not None and net_margin < 0:
            reason_parts.append(f"unprofitable (margin={net_margin*100:.0f}%)")
        if fcf is not None and fcf < 0:
            reason_parts.append("negative_fcf")
        reason = " and ".join(reason_parts) if reason_parts else f"business_quality={business_quality}"
        top_triggers.append(_make_trigger(
            "weak_business_quality",
            "fundamentals",
            "bearish",
            reason,
            next_update_earnings_date=next_earnings_date,
        ))

    if valuation_gate == "headwind":
        # Use the same metric that drove the valuation gate decision
        val_basis = valuation_assessment.get("basis")
        if is_unprofitable:
            # For unprofitable, prefer EV/S over P/S (same logic as gate)
            if val_basis == "ev_to_sales" and ev_to_sales is not None:
                reason = f"ev_to_sales={ev_to_sales:.1f}x (unprofitable, debt/cash-adjusted)"
            elif ps_trailing is not None:
                reason = f"ps_trailing={ps_trailing:.1f}x (unprofitable, source={ps_source or 'unknown'})"
            else:
                reason = "valuation_stretched (unprofitable)"
        elif pe_trailing is not None:
            reason = f"pe_trailing={pe_trailing:.1f}x"
        else:
            reason = "valuation_stretched"
        # Valuation uses fundamentals component
        top_triggers.append(_make_trigger(
            "valuation_headwind",
            "valuation",
            "bearish",
            reason,
            scoring_key="fundamentals",
        ))

    # Revenue decline trigger (severe decline is a major bearish signal)
    if revenue_yoy is not None and revenue_yoy < -0.20:
        top_triggers.append(_make_trigger(
            "severe_revenue_decline",
            "fundamentals",
            "bearish",
            f"revenue_yoy={revenue_yoy*100:.0f}% (severe decline)",
            next_update_earnings_date=next_earnings_date,
        ))

    # Dilution risk for unprofitable companies with low runway
    if is_unprofitable and cash_runway_quarters is not None and cash_runway_quarters < 8:
        runway_years = round(cash_runway_quarters / 4, 1)
        dilution_reason = f"cash_runway={runway_years}y ({runway_basis or 'fcf'})"
        if dilution_analysis:
            dilution_pct = dilution_analysis.get("dilution_if_raised_today")
            dilution_level = dilution_analysis.get("dilution_risk_level")
            if dilution_pct is not None:
                dilution_reason += f" - {dilution_pct*100:.0f}% dilution if raised ({dilution_level})"
        top_triggers.append(_make_trigger(
            "dilution_risk",
            "fundamentals",
            "bearish",
            dilution_reason,
        ))

    if setup_label in ("weak", "poor"):
        top_triggers.append(_make_trigger(
            "weak_technical_setup",
            "technicals",
            "bearish",
            f"setup={setup_label}",
        ))

    # === COLLECT BULLISH TRIGGERS (always, not just when no bearish) ===
    bullish_triggers: list[dict[str, Any]] = []

    if business_quality == "strong":
        reason = "profitable"
        if fcf_label:
            reason = f"{reason}, {fcf_label}"
        bullish_triggers.append(_make_trigger(
            "strong_business_quality",
            "fundamentals",
            "bullish",
            reason,
            next_update_earnings_date=next_earnings_date,
        ))
    elif business_quality == "moderate":
        bullish_triggers.append(_make_trigger(
            "moderate_business_quality",
            "fundamentals",
            "bullish",
            "profitable business with some growth",
            next_update_earnings_date=next_earnings_date,
        ))

    if valuation_gate == "attractive":
        # Use the same metric that drove the valuation gate decision
        val_basis = valuation_assessment.get("basis")
        if is_unprofitable:
            # For unprofitable, prefer EV/S over P/S (same logic as gate)
            if val_basis == "ev_to_sales" and ev_to_sales is not None:
                reason = f"ev_to_sales={ev_to_sales:.1f}x reasonable (debt/cash-adjusted)"
            elif ps_trailing is not None:
                reason = f"ps_trailing={ps_trailing:.1f}x reasonable (source={ps_source or 'unknown'})"
            else:
                reason = "valuation_attractive (unprofitable)"
        else:
            fcf_yield = yield_m.get("fcf_yield")
            earnings_yield = yield_m.get("earnings_yield")
            peg_ratio = val.get("peg_ratio")
            if val_basis == "fcf_yield" and fcf_yield is not None:
                reason = f"fcf_yield={fcf_yield*100:.1f}%"
            elif val_basis == "earnings_yield" and earnings_yield is not None:
                reason = f"earnings_yield={earnings_yield*100:.1f}%"
            elif val_basis == "peg_ratio" and peg_ratio is not None:
                reason = f"peg_ratio={peg_ratio:.2f}"
            elif val_basis == "pe_trailing" and pe_trailing is not None:
                reason = f"pe_trailing={pe_trailing:.1f}x"
            else:
                reason = "valuation_attractive"
        bullish_triggers.append(_make_trigger(
            "attractive_valuation",
            "valuation",
            "bullish",
            reason,
            scoring_key="fundamentals",
        ))

    if setup_label == "strong":
        bullish_triggers.append(_make_trigger(
            "strong_technical_setup",
            "technicals",
            "bullish",
            f"setup={setup_label}",
        ))
    elif setup_label == "moderate":
        bullish_triggers.append(_make_trigger(
            "moderate_technical_setup",
            "technicals",
            "bullish",
            f"setup={setup_label} (price above key MAs)",
        ))

    if risk_label == "low":
        bullish_triggers.append(_make_trigger(
            "favorable_risk_regime",
            "risk",
            "bullish",
            "risk_regime=low",
        ))
    elif risk_label == "moderate":
        bullish_triggers.append(_make_trigger(
            "acceptable_risk_regime",
            "risk",
            "bullish",
            "risk_regime=moderate (manageable)",
        ))

    # === ADD FALLBACK TRIGGERS ===
    # Only add these if we don't have enough primary bearish triggers
    # Note: vol/drawdown are already included in elevated_risk_regime reason, so don't duplicate
    fallback_triggers: list[dict[str, Any]] = []

    # Check if we already have elevated_risk_regime (which includes vol/dd details)
    has_risk_regime_trigger = any(t.get("id") == "elevated_risk_regime" for t in top_triggers)

    # Get risk metrics for fallback triggers (only used if no regime trigger)
    risk_summary = verdict.get("inputs_used", {})
    annualized_vol = risk_summary.get("annualized_vol")
    max_drawdown = risk_summary.get("max_drawdown_1y")

    # Only add vol/drawdown as separate triggers if risk regime is NOT already a trigger
    # This avoids double-counting the same risk factors
    if not has_risk_regime_trigger:
        if "very_high_volatility" in bearish_list and annualized_vol is not None:
            fallback_triggers.append(_make_trigger(
                "very_high_volatility",
                "risk",
                "bearish",
                f"volatility={annualized_vol*100:.0f}% (>60%)",
            ))
        elif "high_volatility" in bearish_list and annualized_vol is not None:
            fallback_triggers.append(_make_trigger(
                "high_volatility",
                "risk",
                "bearish",
                f"volatility={annualized_vol*100:.0f}% (>40%)",
            ))

        if "deep_drawdown" in bearish_list and max_drawdown is not None:
            fallback_triggers.append(_make_trigger(
                "deep_drawdown",
                "risk",
                "bearish",
                f"drawdown={max_drawdown*100:.0f}% (>35%)",
            ))

    # Burn metrics missing sub-trigger (for unprofitable companies) - always valuable
    burn_status = burn_metrics.get("status")
    if is_unprofitable and burn_status == "unavailable":
        fallback_triggers.append(_make_trigger(
            "runway_unknown",
            "fundamentals",
            "bearish",
            f"burn_metrics unavailable ({burn_metrics.get('status_reason', 'unknown')})",
        ))

    # Severe revenue decline - always a distinct concern worth showing
    if (
        revenue_yoy is not None
        and revenue_yoy < -0.20
        and not any(t.get("id") == "severe_revenue_decline" for t in top_triggers)
    ):
        fallback_triggers.append(_make_trigger(
            "severe_revenue_decline",
            "fundamentals",
            "bearish",
            f"revenue_yoy={revenue_yoy*100:.0f}% (severe decline)",
        ))

    # === BALANCE TRIGGERS BASED ON TILT ===
    # OPTION A: Only include triggers with actual score contribution
    # Triggers from components that didn't fire (score_delta=None) are excluded
    # This keeps Top Drivers = "scoring drivers only" for auditability

    # Filter to only scoring triggers (non-null score_delta)
    scoring_bearish = [
        t for t in top_triggers
        if t.get("direction") == "bearish" and t.get("score_delta") is not None
    ]
    scoring_bullish = [
        t for t in bullish_triggers
        if t.get("score_delta") is not None
    ]

    # Sort bearish by absolute score_delta (most impactful first)
    scoring_bearish.sort(key=lambda x: abs(x.get("score_delta") or 0), reverse=True)

    # Sort bullish by absolute score_delta
    scoring_bullish.sort(key=lambda x: abs(x.get("score_delta") or 0), reverse=True)

    # Reassign for downstream logic
    bearish_triggers = scoring_bearish
    bullish_triggers = scoring_bullish

    # Determine target counts based on tilt
    if tilt == "neutral":
        target_bearish = 2
        target_bullish = 1
    elif tilt == "bullish":
        target_bearish = 1
        target_bullish = 2
    elif tilt == "bearish":
        target_bearish = 2
        target_bullish = 1
    else:
        target_bearish = 2
        target_bullish = 1

    # Add fallback triggers if we don't have enough bearish
    # Avoid duplicates by checking IDs
    # Only include fallbacks with non-null score_delta (Option A: scoring drivers only)
    existing_ids = {t.get("id") for t in bearish_triggers}
    for fallback in fallback_triggers:
        if fallback.get("id") not in existing_ids and fallback.get("score_delta") is not None:
            bearish_triggers.append(fallback)
            existing_ids.add(fallback.get("id"))

    # Re-sort after adding sub-triggers
    bearish_triggers.sort(key=lambda x: abs(x.get("score_delta") or 0), reverse=True)

    # Build final triggers with target counts
    final_triggers: list[dict[str, Any]] = []
    final_triggers.extend(bearish_triggers[:target_bearish])
    final_triggers.extend(bullish_triggers[:target_bullish])

    # Track if we couldn't meet target count
    triggers_incomplete_reason: str | None = None
    if len(bearish_triggers) < target_bearish and len(bullish_triggers) < target_bullish:
        triggers_incomplete_reason = "insufficient_trigger_candidates"
    elif len(bearish_triggers) < target_bearish:
        triggers_incomplete_reason = "insufficient_bearish_candidates"
    elif len(bullish_triggers) < target_bullish:
        triggers_incomplete_reason = "insufficient_bullish_candidates"

    # If we have no triggers at all, just use whatever we collected
    if not final_triggers:
        final_triggers = bearish_triggers + bullish_triggers

    # Replace top_triggers with balanced set
    top_triggers = final_triggers

    # === FUNDAMENTALS CATEGORY ===
    # === MACHINE-CHECKABLE CONDITION BUILDER ===
    # Each condition has structured fields for automated monitoring:
    # - id: unique identifier
    # - data_source: yfinance field path (e.g., "yfinance.info.profitMargins")
    # - operator: <, <=, >, >=, ==, !=
    # - target_value: numeric threshold
    # - current_value: current numeric value (for comparison)
    # - condition: human-readable label
    # - threshold: human-readable threshold description
    # - current: human-readable current value
    # - next_update: when to check again

    fundamentals_bullish, fundamentals_bearish = _build_fundamental_conditions(
        is_unprofitable=is_unprofitable,
        net_margin=net_margin,
        fcf=fcf,
        revenue_yoy=revenue_yoy,
        next_earnings_date=next_earnings_date,
        bullish_list=bullish_list,
        bearish_list=bearish_list,
    )

    # === VALUATION CATEGORY ===
    valuation_bullish: list[dict[str, Any]] = []
    valuation_bearish: list[dict[str, Any]] = []

    # Get valuation metrics for thresholds (reuse val/yield_m from above)
    fcf_yield_val = yield_m.get("fcf_yield")
    pe_forward = val.get("pe_forward")
    peg_ratio = val.get("peg_ratio")

    # For unprofitable companies, prefer EV/S over P/S thresholds
    if is_unprofitable:
        # Use the same metric that drives the valuation gate
        val_basis = valuation_assessment.get("basis")
        if val_basis == "ev_to_sales" and ev_to_sales is not None:
            # EV/S preferred - adjusts for debt/cash position
            valuation_bullish.append(_make_condition(
                id="evs_contracts",
                condition="evs_contracts",
                data_source="computed.ev_to_sales",
                operator="<=",
                target_value=3.0,
                current_value=ev_to_sales,
                threshold_str="ev_to_sales <= 3",
                current_str=f"ev_to_sales = {ev_to_sales:.1f}",
            ))
            valuation_bearish.append(_make_condition(
                id="evs_expands",
                condition="evs_expands",
                data_source="computed.ev_to_sales",
                operator=">=",
                target_value=10.0,
                current_value=ev_to_sales,
                threshold_str="ev_to_sales >= 10",
                current_str=f"ev_to_sales = {ev_to_sales:.1f}",
            ))
        elif ps_trailing is not None:
            # Fallback to P/S if EV/S unavailable
            valuation_bullish.append(_make_condition(
                id="ps_contracts",
                condition="ps_contracts",
                data_source="yfinance.info.priceToSalesTrailing12Months",
                operator="<=",
                target_value=3.0,
                current_value=ps_trailing,
                threshold_str="ps_trailing <= 3",
                current_str=f"ps_trailing = {ps_trailing:.1f}",
            ))
            valuation_bearish.append(_make_condition(
                id="ps_expands",
                condition="ps_expands",
                data_source="yfinance.info.priceToSalesTrailing12Months",
                operator=">=",
                target_value=10.0,
                current_value=ps_trailing,
                threshold_str="ps_trailing >= 10",
                current_str=f"ps_trailing = {ps_trailing:.1f}",
            ))
    else:
        # For profitable companies, use P/E and FCF yield
        if valuation_gate == "headwind":
            if pe_forward is not None and pe_forward > 30:
                valuation_bullish.append({
                    "condition": "pe_contracts",
                    "threshold": "forward_pe < 25",
                    "current": f"forward_pe = {pe_forward:.1f}",
                })
            if fcf_yield_val is not None and fcf_yield_val > 0 and fcf_yield_val < 0.03:
                valuation_bullish.append({
                    "condition": "fcf_yield_expands",
                    "threshold": "fcf_yield > 4%",
                    "current": f"fcf_yield = {fcf_yield_val * 100:.1f}%",
                })
        elif valuation_gate == "attractive":
            if pe_forward is not None:
                valuation_bearish.append({
                    "condition": "pe_expands",
                    "threshold": "forward_pe > 30",
                    "current": f"forward_pe = {pe_forward:.1f}",
                })
            if fcf_yield_val is not None and fcf_yield_val > 0:
                valuation_bearish.append({
                    "condition": "fcf_yield_compresses",
                    "threshold": "fcf_yield < 3%",
                    "current": f"fcf_yield = {fcf_yield_val * 100:.1f}%",
                })

        if "high_pe" in bearish_list and pe_forward is not None:
            valuation_bullish.append({
                "condition": "valuation_normalizes",
                "threshold": "forward_pe < 25 or growth accelerates",
                "current": f"forward_pe = {pe_forward:.1f}",
            })
        if "low_peg" in bullish_list and peg_ratio is not None:
            valuation_bearish.append({
                "condition": "peg_expands",
                "threshold": "peg_ratio > 2.0",
                "current": f"peg_ratio = {peg_ratio:.2f}",
            })

    # === NEWS CATEGORY ===
    catalyst_intelligence = news_data.get("catalyst_intelligence") or {}
    news_triggers: dict[str, Any] = {
        "headline_triggers": {
            "bullish": [
                item.get("tag")
                for item in catalyst_intelligence.get("bullish", [])[:5]
                if item.get("tag")
            ],
            "bearish": [
                item.get("tag")
                for item in catalyst_intelligence.get("bearish", [])[:5]
                if item.get("tag")
            ],
        },
    }

    # Add recent sentiment if available
    sentiment = news_data.get("sentiment", {})
    if sentiment:
        news_triggers["current_sentiment"] = sentiment.get("overall")
        news_triggers["sentiment_confidence"] = sentiment.get("confidence")
    if catalyst_intelligence:
        news_triggers["catalyst_method"] = catalyst_intelligence.get("method")

    # === RISK CATEGORY ===
    risk_bullish: list[dict[str, Any]] = []
    risk_bearish: list[dict[str, Any]] = []

    # Thresholds aligned with risk_regime boundaries:
    # extreme: >60%, high: 40-60%, medium: 25-40%, low: <25%
    # bullish_if targets the next lower regime boundary
    vol_threshold = vol_threshold_for_improvement(risk_label, annualized_vol)
    if annualized_vol is not None and vol_threshold is not None and annualized_vol >= vol_threshold:
        risk_bullish.append({
            "condition": "volatility_decreases",
            "threshold": f"annualized_vol < {vol_threshold * 100:.0f}%",
            "current": f"{annualized_vol * 100:.1f}%",
        })

    if max_dd is not None and max_dd <= -0.50:
        risk_bullish.append({
            "condition": "drawdown_recovers",
            "threshold": "max_drawdown_1y > -50%",
            "current": f"{max_dd * 100:.1f}%",
        })

    if "deep_drawdown" in bearish_list and max_dd is not None:
        risk_bearish.append({
            "condition": "drawdown_worsens",
            "threshold": "max_drawdown_1y < -60%",
            "current": f"{max_dd * 100:.1f}%",
        })

    beta_data = risk_data.get("beta", {})
    beta_val = beta_data.get("value")
    if "high_beta" in bearish_list and beta_val is not None:
        risk_bullish.append({
            "condition": "beta_normalizes",
            "threshold": "beta < 1.3",
            "current": f"beta = {beta_val:.2f}",
        })

    # === TECHNICALS CATEGORY ===
    technicals_bullish: list[dict[str, Any]] = []
    technicals_bearish: list[dict[str, Any]] = []

    if "price_below_sma200" in bearish_list and sma_200 is not None:
        technicals_bullish.append({
            "condition": "price_reclaims_sma200",
            "threshold": f"close > ${sma_200:.2f} for 3 sessions",
            "current": f"${current_price:.2f}" if current_price else None,
        })
    if "price_above_sma200" in bullish_list and sma_200 is not None:
        technicals_bearish.append({
            "condition": "price_breaks_sma200",
            "threshold": f"close < ${sma_200:.2f} by >2% for 3 sessions",
            "current": f"${current_price:.2f}" if current_price else None,
        })

    if "death_cross" in bearish_list:
        technicals_bullish.append({
            "condition": "golden_cross_forms",
            "threshold": "SMA50 crosses above SMA200",
            "current": f"SMA50=${sma_50:.2f}, SMA200=${sma_200:.2f}" if sma_50 and sma_200 else None,
        })
    if "golden_cross" in bullish_list:
        technicals_bearish.append({
            "condition": "death_cross_forms",
            "threshold": "SMA50 crosses below SMA200",
            "current": f"SMA50=${sma_50:.2f}, SMA200=${sma_200:.2f}" if sma_50 and sma_200 else None,
        })

    if "rsi_overbought" in bearish_list:
        technicals_bullish.append({
            "condition": "rsi_normalizes",
            "threshold": "RSI < 70",
            "current": f"RSI = {rsi_value:.1f}" if rsi_value else None,
        })
    if "rsi_oversold" in bullish_list:
        technicals_bearish.append({
            "condition": "rsi_fails_to_recover",
            "threshold": "RSI stays < 30 for 10+ sessions",
            "current": f"RSI = {rsi_value:.1f}" if rsi_value else None,
        })

    if "weak_3m_momentum" in bearish_list:
        technicals_bullish.append({
            "condition": "momentum_reverses",
            "threshold": "3m_return > 0%",
        })
    if "strong_3m_momentum" in bullish_list:
        technicals_bearish.append({
            "condition": "momentum_reverses",
            "threshold": "3m_return < 0%",
        })

    # === NEXT CATALYST ===
    next_catalyst: dict[str, Any] | None = None
    if days_until_earnings is not None and days_until_earnings > 0 and next_earnings_date:
        next_catalyst = {
            "event": "earnings",
            "date": next_earnings_date,
            "days_until": days_until_earnings,
        }
    elif next_earnings_date:
        next_catalyst = {
            "event": "earnings",
            "date": next_earnings_date,
        }

    # === THESIS CHECKPOINTS (2-year framework for long-term investors) ===
    thesis_checkpoints = _build_thesis_checkpoints(
        is_unprofitable=is_unprofitable,
        net_margin=net_margin,
        fcf=fcf,
        cash_runway_quarters=cash_runway_quarters,
        revenue_yoy=growth.get("revenue_yoy") if growth else None,
        valuation_gate=valuation_gate,
        risk_label=risk_label,
        business_quality=business_quality,
        next_earnings_date=next_earnings_date,
    )

    # === APPLY PER-CATEGORY LIMITS FOR CLEANER OUTPUT ===
    max_per_category = 2

    # Build fundamentals status - separating data availability from valuation applicability
    # fundamentals_status: available/missing (did we get fundamental data?)
    # valuation_status: pe_valid/pe_not_meaningful (can we use P/E?)
    fundamentals_fetch_status: str
    fundamentals_status_explanation: str | None = None
    valuation_pe_status: str
    valuation_pe_explanation: str | None = None

    if business_quality_status == "data_missing":
        fundamentals_fetch_status = "missing"
        fundamentals_status_explanation = "Fundamental data unavailable - cannot assess business quality"
        valuation_pe_status = "unavailable"
    elif business_quality_status == "evaluated_unprofitable":
        # Fundamentals ARE available (revenue, margin, FCF) - it's just P/E that's not meaningful
        fundamentals_fetch_status = "available"
        valuation_pe_status = "not_meaningful"
        # Make explanation basis-aware (EV/S preferred over P/S)
        val_basis = valuation_assessment.get("basis")
        if valuation_gate == "unknown":
            valuation_pe_explanation = "P/E not meaningful (unprofitable), no sales multiple available"
        elif val_basis == "ev_to_sales":
            valuation_pe_explanation = "P/E not meaningful (unprofitable), using EV/S instead"
        elif val_basis == "ps_trailing":
            valuation_pe_explanation = "P/E not meaningful (unprofitable), using P/S instead"
        else:
            valuation_pe_explanation = "P/E not meaningful (unprofitable)"
    else:
        fundamentals_fetch_status = "available"
        valuation_pe_status = "valid"

    # EV/S status (preferred for unprofitable companies when debt/cash is material)
    valuation_evs_status: str
    valuation_evs_explanation: str | None = None
    if ev_to_sales is not None:
        valuation_evs_status = "available"
        if ev_to_sales_source == "computed":
            valuation_evs_explanation = "EV/S computed from enterprise_value / revenue_ttm"
    else:
        valuation_evs_status = "unavailable"

    # P/S status (fallback for unprofitable companies if EV/S unavailable)
    valuation_ps_status: str
    valuation_ps_explanation: str | None = None
    if ps_trailing is not None:
        valuation_ps_status = "available"
        if ps_source == "computed":
            valuation_ps_explanation = "P/S computed from market_cap / revenue_ttm"
    else:
        valuation_ps_status = "unavailable"
        if is_unprofitable and valuation_evs_status == "unavailable":
            valuation_ps_explanation = (
                "Neither EV/S nor P/S available - cannot assess valuation for unprofitable company"
            )

    # === HORIZON DRIVERS (policy gates, not score-based) ===
    # These explain why mid_term/long_term are "caution" or "avoid"
    # Separate from top_triggers which are scoring-only
    horizon_fit = verdict.get("horizon_fit", {})
    horizon_drivers: list[dict[str, Any]] = []

    burn_status = burn_metrics.get("status")

    # Long-term horizon drivers - ONLY emit drivers that fired in horizon_fit
    # This ensures 1:1 alignment between horizon_fit reasons and horizon_drivers
    burn_status_reason = burn_metrics.get("status_reason")
    long_term_gates = horizon_fit.get("long_term_gates") or []

    if horizon_fit.get("long_term") in ("caution", "avoid"):
        # Only emit drivers that are in long_term_gates (from horizon_fit)
        # Each gate in horizon_fit.long_term_gates gets a corresponding driver

        if "unprofitable" in long_term_gates:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "unprofitable",
                "reason": "unprofitable - requires thesis on path to profitability",
                "current": f"business_quality={business_quality}",
            })

        if "burn_metrics_missing" in long_term_gates:
            data_gaps_list = ["burn_metrics_unavailable"]
            if burn_status_reason:
                data_gaps_list.append(burn_status_reason)
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "burn_metrics_missing",
                "reason": f"unprofitable with burn metrics unavailable ({burn_status_reason or 'unknown'})",
                "data_gaps": data_gaps_list,
            })

        if "low_runway_confidence" in long_term_gates:
            # Low runway_confidence = runway < 2 years = dilution risk elevated
            runway_conf = burn_metrics.get("runway_confidence")
            runway_quarters = burn_metrics.get("cash_runway_quarters")
            runway_years = round(runway_quarters / 4, 1) if runway_quarters else None
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "low_runway_confidence",
                "reason": f"cash runway {runway_years}y - dilution risk elevated",
                "current": f"runway_confidence={runway_conf}, {runway_quarters:.1f}q" if runway_quarters else f"runway_confidence={runway_conf}",
            })

        if "extreme_risk" in long_term_gates:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "extreme_risk",
                "reason": "extreme risk unsuitable for core holdings",
                "current": f"risk_regime={risk_label}",
            })

        if "severe_revenue_decline" in long_term_gates and revenue_yoy is not None:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "severe_revenue_decline",
                "reason": f"revenue declining {abs(revenue_yoy)*100:.0f}% YoY",
                "current": f"{revenue_yoy*100:.0f}%",
            })

        if "negative_fcf" in long_term_gates and fcf is not None:
            horizon_drivers.append({
                "horizon": "long_term",
                "direction": "bearish",
                "gate": "negative_fcf",
                "reason": "negative free cash flow - burning cash",
                "current": f"FCF=${fcf/1e6:.0f}M",
            })

    # Mid-term horizon drivers
    if horizon_fit.get("mid_term") in ("caution", "avoid") and risk_label == "extreme":
        horizon_drivers.append({
            "horizon": "mid_term",
            "direction": "bearish",
            "gate": "extreme_risk",
            "reason": "extreme risk requires minimal position size",
            "current": f"risk_regime={risk_label}",
        })

    return {
        "top_triggers": top_triggers or [],
        "top_triggers_incomplete_reason": triggers_incomplete_reason,
        "horizon_drivers": horizon_drivers or [],
        "fundamentals": {
            "bullish_if": _limit_list(fundamentals_bullish, max_per_category),
            "bearish_if": _limit_list(fundamentals_bearish, max_per_category),
            "status": fundamentals_fetch_status,  # available/missing (data fetch status)
            "status_explanation": fundamentals_status_explanation,
            "business_quality": business_quality,  # strong/moderate/mixed/poor/unprofitable/weak or None
            "next_update": next_earnings_date,
            "check_frequency": "quarterly_earnings",
        },
        "valuation": {
            "bullish_if": _limit_list(valuation_bullish, 1),
            "bearish_if": _limit_list(valuation_bearish, 1),
            "current_gate": valuation_gate,
            "basis": valuation_assessment.get("basis"),  # ev_to_sales/ps_trailing/pe_trailing/etc.
            "pe_status": valuation_pe_status,  # valid/not_meaningful/unavailable
            "pe_explanation": valuation_pe_explanation,
            "evs_status": valuation_evs_status,  # available/unavailable (preferred for unprofitable)
            "evs_explanation": valuation_evs_explanation,
            "ps_status": valuation_ps_status,  # available/unavailable (fallback for unprofitable)
            "ps_explanation": valuation_ps_explanation,
            "is_unprofitable": is_unprofitable if is_unprofitable else None,
            "next_update": next_earnings_date,  # Multiples rerate after earnings
            "check_frequency": "quarterly_earnings",
        },
        "news": {
            **news_triggers,
            "check_frequency": "daily_or_weekly",
            "weight_note": "low_weight_unless_high_sample_size",
        },
        "risk": {
            "bullish_if": _limit_list(risk_bullish, max_per_category),
            "bearish_if": _limit_list(risk_bearish, max_per_category),
            "current_regime": risk_label,
            "check_frequency": "weekly",
            "regime_note": "risk_regime_shifts_slowly",
        },
        "technicals": {
            "bullish_if": _limit_list(technicals_bullish, max_per_category),
            "bearish_if": _limit_list(technicals_bearish, max_per_category),
            "check_frequency": "weekly_for_long_term",
        },
        "next_catalyst": next_catalyst,
        "thesis_checkpoints": thesis_checkpoints,
    }


def _build_thesis_checkpoints(
    is_unprofitable: bool,
    net_margin: float | None,
    fcf: float | None,
    cash_runway_quarters: float | None,
    revenue_yoy: float | None,
    valuation_gate: str | None,
    risk_label: str | None,
    business_quality: str | None,
    next_earnings_date: str | None = None,
) -> dict[str, Any]:
    """
    Build thesis checkpoints for 2-year investment framework.

    Returns milestones a long-term investor should monitor over the hold period.
    Each checkpoint includes:
    - data_source: which yfinance field to check
    - update_on: when this data updates (quarterly_earnings, weekly, etc.)
    - next_check: optional specific date for next check
    """
    checkpoints: list[dict[str, Any]] = []
    hold_thesis: str | None = None
    review_triggers: list[str] = []
    thesis_stop_triggers: list[dict[str, Any]] = []  # Non-price based exit triggers

    # Determine the investment thesis based on company profile
    if is_unprofitable:
        hold_thesis = "growth_to_profitability"

        # Unprofitable company checkpoints
        checkpoints.append({
            "id": "returns_to_profitability",
            "milestone": "path_to_profitability",
            "target": "achieve positive net margin within 6-8 quarters",
            "current": f"net_margin = {net_margin*100:.1f}%" if net_margin else "unknown",
            "data_source": "yfinance.info.profitMargins",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "critical",
        })

        if fcf is not None and fcf < 0:
            checkpoints.append({
                "id": "fcf_turns_positive",
                "milestone": "fcf_positive",
                "target": "achieve positive FCF within 4-6 quarters",
                "current": "fcf = negative",
                "data_source": "yfinance.info.freeCashflow",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": "high",
            })

        if cash_runway_quarters is not None:
            runway_status = "adequate" if cash_runway_quarters >= 8 else "limited"
            checkpoints.append({
                "id": "maintain_cash_runway",
                "milestone": "maintain_runway",
                "target": "maintain 2+ years cash runway without dilutive raise",
                "current": f"runway = {cash_runway_quarters/4:.1f} years ({runway_status})",
                "data_source": "computed from yfinance.info.{totalCash,freeCashflow,operatingCashflow}",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": "critical" if cash_runway_quarters < 8 else "medium",
            })

        if revenue_yoy is not None:
            # Adaptive growth checkpoint based on current revenue regime
            if revenue_yoy < -0.20:
                # Severe decline: goal is stabilization
                growth_target = "stabilize revenue (YoY > -10% or 2 sequential quarters of improvement)"
                growth_priority = "critical"
            elif revenue_yoy < -0.10:
                # Moderate decline: goal is re-acceleration
                growth_target = "return to positive growth (YoY > 0%)"
                growth_priority = "high"
            elif revenue_yoy < 0.10:
                # Flat/slow: goal is acceleration
                growth_target = "accelerate growth (YoY > 15%)"
                growth_priority = "high"
            else:
                # Already growing: goal is sustaining
                growth_target = "maintain 15%+ revenue growth while improving margins"
                growth_priority = "medium"

            checkpoints.append({
                "id": "revenue_trajectory_improves",
                "milestone": "revenue_trajectory",
                "target": growth_target,
                "current": f"revenue_yoy = {revenue_yoy*100:.1f}%",
                "data_source": "yfinance.info.revenueGrowth",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": growth_priority,
            })

        review_triggers = [
            "equity_raise_announced",
            "growth_decelerates_below_15%",
            "cash_runway_drops_below_4_quarters",
            "key_executive_departure",
        ]

        # Thesis stop triggers for unprofitable companies (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "runway_critical",
                "condition": "cash_runway < 4 quarters",
                "action": "exit_or_size_to_zero",
                "rationale": "dilutive raise imminent",
            },
            {
                "trigger": "profitability_path_broken",
                "condition": "net margin worsens 3 consecutive quarters",
                "action": "reassess_thesis",
                "rationale": "path to profitability not progressing",
            },
            {
                "trigger": "growth_collapsed",
                "condition": "revenue_yoy < -30% for 2 quarters",
                "action": "exit_or_reduce",
                "rationale": "growth thesis invalidated",
            },
        ]

    elif business_quality in ("strong", "moderate"):
        # Profitable company checkpoints
        if valuation_gate == "attractive":
            hold_thesis = "undervalued_quality"
            checkpoints.append({
                "id": "valuation_rerates",
                "milestone": "valuation_rerates",
                "target": "P/E or P/S expands toward sector median",
                "data_source": "yfinance.info.{trailingPE,priceToSalesTrailing12Months}",
                "update_on": "weekly",
                "next_check": None,  # Price-based, check weekly
                "priority": "medium",
            })
        else:
            hold_thesis = "quality_compounder"

        checkpoints.append({
            "id": "earnings_growth_continues",
            "milestone": "earnings_growth",
            "target": "maintain or accelerate earnings growth trajectory",
            "current": f"net_margin = {net_margin*100:.1f}%" if net_margin else "unknown",
            "data_source": "yfinance.info.{earningsGrowth,profitMargins}",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "high",
        })

        if fcf is not None and fcf > 0:
            checkpoints.append({
                "id": "capital_allocation_quality",
                "milestone": "capital_allocation",
                "target": "FCF deployed to buybacks, dividends, or accretive M&A",
                "current": "fcf_positive",
                "data_source": "yfinance.info.freeCashflow + SEC filings",
                "update_on": "quarterly_earnings",
                "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
                "priority": "medium",
            })

        review_triggers = [
            "margin_compression_2_consecutive_quarters",
            "guidance_cut",
            "competitive_threat_emerges",
            "valuation_exceeds_historical_range",
        ]

        # Thesis stop triggers for quality compounders (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "earnings_deterioration",
                "condition": "EPS declines 2 consecutive quarters (ex one-time)",
                "action": "reassess_position_size",
                "rationale": "compounder thesis requires earnings growth",
            },
            {
                "trigger": "margin_collapse",
                "condition": "operating margin drops >500bps YoY",
                "action": "investigate_and_reassess",
                "rationale": "competitive position may be weakening",
            },
            {
                "trigger": "capital_allocation_concern",
                "condition": "large dilutive acquisition or debt-funded buyback at peak",
                "action": "reassess_management_quality",
                "rationale": "poor capital allocation destroys long-term value",
            },
        ]

    elif business_quality == "mixed":
        hold_thesis = "turnaround_or_cyclical"
        checkpoints.append({
            "id": "operational_improvement",
            "milestone": "operational_improvement",
            "target": "margin expansion and/or revenue acceleration",
            "data_source": "yfinance.info.{operatingMargins,revenueGrowth}",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "high",
        })
        checkpoints.append({
            "id": "balance_sheet_strengthening",
            "milestone": "balance_sheet_strengthening",
            "target": "debt reduction or cash build",
            "data_source": "yfinance.info.{totalDebt,totalCash}",
            "update_on": "quarterly_earnings",
            "next_check": {"event": "earnings", "date": next_earnings_date} if next_earnings_date else None,
            "priority": "medium",
        })

        review_triggers = [
            "turnaround_thesis_invalidated",
            "industry_downturn_extends",
            "management_credibility_issue",
        ]

        # Thesis stop triggers for turnarounds (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "turnaround_stalled",
                "condition": "no margin improvement after 4 quarters",
                "action": "exit_position",
                "rationale": "turnaround thesis has failed",
            },
            {
                "trigger": "balance_sheet_worsening",
                "condition": "debt/equity increases or liquidity deteriorates",
                "action": "reduce_or_exit",
                "rationale": "financial stress increasing",
            },
        ]

    else:
        # Poor or unknown business quality
        hold_thesis = "speculative"
        checkpoints.append({
            "milestone": "thesis_validation",
            "target": "concrete evidence of business improvement",
            "priority": "critical",
        })

        review_triggers = [
            "no_improvement_after_4_quarters",
            "deteriorating_fundamentals",
        ]

        # Thesis stop triggers for speculative positions (non-price based exits)
        thesis_stop_triggers = [
            {
                "trigger": "thesis_invalidated",
                "condition": "no evidence of improvement after 4 quarters",
                "action": "exit_position",
                "rationale": "speculative thesis requires rapid validation",
            },
        ]

    # Add risk-related checkpoint if relevant
    if risk_label in ("high", "extreme"):
        checkpoints.append({
            "milestone": "risk_normalization",
            "target": "volatility decreases to moderate levels",
            "current": f"risk_regime = {risk_label}",
            "priority": "medium",
        })

    return {
        "hold_thesis": hold_thesis,
        "checkpoints": checkpoints[:4],  # Limit to top 4 checkpoints
        "review_triggers": review_triggers[:4],  # Limit to top 4 triggers
        "thesis_stop_triggers": thesis_stop_triggers[:3] if thesis_stop_triggers else None,
        "review_frequency": "quarterly_after_earnings",
    }
