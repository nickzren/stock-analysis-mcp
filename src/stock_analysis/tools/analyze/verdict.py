"""Verdict computation: scoring, confidence, and horizon fit."""

from typing import Any

from stock_analysis.tools.analyze.signals import (
    vol_threshold_for_improvement,
)
from stock_analysis.utils.helpers import format_fcf_label

# Weights for verdict scoring (mid/long-term investor bias)
# fundamentals > technicals > risk
# Note: news sentiment is shown but not scored (keyword-based sentiment unreliable for mid/long-term)
VERDICT_WEIGHTS = {
    "fundamentals": 0.45,
    "technicals": 0.30,
    "risk": 0.25,
}


def build_verdict(
    signals: dict[str, list[str]],
    fundamentals_data: dict[str, Any],
    risk_data: dict[str, Any],
    technicals_data: dict[str, Any],
    coverage: dict[str, bool],
    risk_regime: dict[str, Any] | None = None,
    fundamentals_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build verdict with score, tilt, confidence, and pros/cons.

    Returns score (None if insufficient coverage), tilt, confidence level,
    coverage status, component scores, decomposed scores (setup/business/risk),
    horizon fit assessment, and pros/cons lists.
    """
    if fundamentals_summary is None:
        fundamentals_summary = {}
    pros: list[str] = []
    cons: list[str] = []

    # Map signals to human-readable pros/cons
    signal_prose = {
        "price_above_sma200": "Trading above 200-day moving average (uptrend)",
        "golden_cross": "Golden cross pattern (bullish trend)",
        "rsi_oversold": "RSI indicates oversold conditions",
        "macd_bullish": "MACD showing bullish momentum",
        "strong_3m_momentum": "Strong 3-month price momentum",
        "high_revenue_growth": "High revenue growth (>20% YoY)",
        "net_cash_positive": "Net cash positive balance sheet",
        "low_debt": "Low debt-to-equity ratio",
        "profitable": "Currently profitable",
        "price_below_sma200": "Trading below 200-day moving average (downtrend)",
        "death_cross": "Death cross pattern (bearish trend)",
        "rsi_overbought": "RSI indicates overbought conditions",
        "macd_bearish": "MACD showing bearish momentum",
        "weak_3m_momentum": "Weak 3-month price momentum",
        "negative_revenue_growth": "Negative revenue growth",
        "unprofitable": "Currently unprofitable",
        "high_volatility": "High price volatility",
        "very_high_volatility": "Very high volatility (>60% annualized)",
        "deep_drawdown": "Deep drawdown (>50% from peak)",
        "insider_buying": "Insider buying activity (net positive 3M)",
        "insider_selling": "Insider selling activity (net negative 3M)",
        "high_put_call_ratio": "Elevated put/call ratio (>1.5, bearish options sentiment)",
        "elevated_implied_volatility": "Elevated implied volatility (>50%)",
        "below_bollinger_lower": "Price below lower Bollinger Band (oversold)",
        "above_bollinger_upper": "Price above upper Bollinger Band (overbought)",
        "bollinger_squeeze": "Bollinger Band squeeze (low volatility, breakout pending)",
    }

    # Build pros from bullish signals
    for sig in signals.get("bullish", []):
        if sig in ("positive_free_cash_flow", "negative_free_cash_flow"):
            continue
        prose = signal_prose.get(sig, sig.replace("_", " ").title())
        if prose not in pros:  # Deduplicate
            pros.append(prose)

    # Build cons from bearish signals
    for sig in signals.get("bearish", []):
        if sig in ("positive_free_cash_flow", "negative_free_cash_flow"):
            continue
        prose = signal_prose.get(sig, sig.replace("_", " ").title())
        if prose not in cons:  # Deduplicate
            cons.append(prose)

    cash_flow = fundamentals_data.get("cash_flow", {})
    fcf_value = cash_flow.get("free_cash_flow_ttm")
    fcf_period = cash_flow.get("free_cash_flow_period")
    fcf_period_end = cash_flow.get("free_cash_flow_period_end")
    fcf_currency = cash_flow.get("currency")
    fcf_label = format_fcf_label(fcf_value, fcf_period, fcf_currency, fcf_period_end)
    if fcf_label:
        if fcf_value is not None and fcf_value > 0:
            if fcf_label not in pros:
                pros.append(fcf_label)
        elif fcf_value is not None and fcf_value < 0:
            if fcf_label not in cons:
                cons.append(fcf_label)

    # Add fundamental-specific context (beyond signals)
    val = fundamentals_data.get("valuation", {})
    peg = val.get("peg_ratio")
    if peg is not None:
        if peg < 1.0:
            if "PEG ratio suggests undervaluation" not in pros:
                pros.append("PEG ratio suggests undervaluation")
        elif peg > 2.5:
            if "PEG ratio suggests overvaluation" not in cons:
                cons.append("PEG ratio suggests overvaluation")

    health = fundamentals_data.get("financial_health", {})
    debt_to_equity = health.get("debt_to_equity")
    if debt_to_equity is not None and debt_to_equity > 1.5:
        if "High debt levels (D/E > 1.5)" not in cons:
            cons.append("High debt levels (D/E > 1.5)")

    # Calculate component scores
    # Each category: (pros - cons) / max(total, 1) scaled to weight
    tech_signals = ["price_above_sma200", "price_below_sma200", "golden_cross",
                    "death_cross", "rsi_oversold", "rsi_overbought",
                    "macd_bullish", "macd_bearish", "strong_3m_momentum",
                    "weak_3m_momentum"]
    fund_signals = ["high_revenue_growth", "negative_revenue_growth",
                    "net_cash_positive", "low_debt", "profitable",
                    "unprofitable", "positive_free_cash_flow",
                    "negative_free_cash_flow"]
    risk_signals = ["high_volatility", "very_high_volatility", "deep_drawdown"]

    def calc_component_score(signal_list: list[str]) -> float | None:
        """Calculate normalized score for a signal category."""
        bullish_list = signals.get("bullish", [])
        bearish_list = signals.get("bearish", [])

        pos = sum(1 for s in signal_list if s in bullish_list)
        neg = sum(1 for s in signal_list if s in bearish_list)
        total = pos + neg

        if total == 0:
            return None  # No signals in this category
        return (pos - neg) / total

    tech_score = calc_component_score(tech_signals)
    fund_score = calc_component_score(fund_signals)
    risk_score = calc_component_score(risk_signals)

    # Build decomposed scores for investor clarity
    # setup_score: technicals + momentum (is the chart attractive?)
    # business_quality_score: profitability, FCF, growth, health (is the business good?)
    # risk_assessment: regime + volatility + drawdown (how dangerous is it?)

    # Setup score: purely technical/momentum driven
    setup_label: str | None = None
    if tech_score is not None:
        if tech_score > 0.3:
            setup_label = "strong"
        elif tech_score > 0:
            setup_label = "moderate"
        elif tech_score > -0.3:
            setup_label = "weak"
        else:
            setup_label = "poor"

    # Business quality score: fundamentals-driven
    # Requires profitability/FCF data to be meaningful
    business_quality_label: str | None = None
    business_quality_status: str = "unknown"

    profit = fundamentals_data.get("profitability", {})
    cf = fundamentals_data.get("cash_flow", {})
    health = fundamentals_data.get("financial_health", {})

    net_margin = profit.get("net_margin")
    fcf_positive = cf.get("rules", {}).get("positive_fcf", {}).get("triggered")
    fcf_value = cf.get("free_cash_flow_ttm")
    fcf_period = cf.get("free_cash_flow_period")
    fcf_period_end = cf.get("free_cash_flow_period_end")
    fcf_currency = cf.get("currency")
    fcf_value_label = format_fcf_label(
        fcf_value,
        fcf_period,
        fcf_currency,
        fcf_period_end,
    )
    low_debt = health.get("rules", {}).get("low_debt", {}).get("triggered")

    # Check if we can assess business quality
    # Use multiple signals to detect unprofitability when data is sparse
    valuation_data = fundamentals_data.get("valuation", {})
    pe_trailing = valuation_data.get("pe_trailing")

    # Check for unprofitability signals from signals list
    bearish_signals = signals.get("bearish", [])
    is_signaled_unprofitable = "unprofitable" in bearish_signals
    has_negative_fcf_signal = "negative_free_cash_flow" in bearish_signals

    # Check valuation_note from fundamentals_summary (set in analyze.py)
    # This catches cases where PE is not meaningful due to losses
    valuation_note = fundamentals_summary.get("valuation", {}).get("valuation_note")
    pe_not_meaningful = valuation_note == "pe_not_meaningful"

    # Get trailing EPS for explicit unprofitability check
    trailing_eps = valuation_data.get("trailing_eps")

    # Get FCF from cash_flow section
    fcf_ttm_val = cf.get("free_cash_flow_ttm")

    # Require explicit signals for "unprofitable" label:
    # 1. trailing_eps <= 0
    # 2. net_margin < 0
    # 3. free_cash_flow_ttm < 0 (only if other signals confirm)
    has_explicit_unprofitable_signal = (
        (net_margin is not None and net_margin < 0)
        or (trailing_eps is not None and trailing_eps <= 0)
        or is_signaled_unprofitable  # From fundamentals rules
        or pe_not_meaningful  # valuation_note indicates PE not meaningful
    )

    if not coverage.get("fundamentals", False):
        business_quality_status = "data_missing"
    elif has_explicit_unprofitable_signal:
        # Explicit signal confirms unprofitable - we HAVE evaluated business quality
        # (it's "unprofitable"), not that we couldn't evaluate it
        business_quality_status = "evaluated_unprofitable"
        business_quality_label = "unprofitable"
    elif fcf_ttm_val is not None and fcf_ttm_val < 0 and has_negative_fcf_signal:
        # Negative FCF confirmed by both value and signal
        # If profitability is positive, call this "mixed" (profitability strong, cash conversion weak)
        business_quality_status = "available"
        if net_margin is not None and net_margin > 0:
            business_quality_label = "mixed"
        else:
            business_quality_label = "weak"
    elif net_margin is None and pe_trailing is None and trailing_eps is None:
        # Insufficient data to assess profitability
        business_quality_status = "data_missing"
        business_quality_label = None
    else:
        business_quality_status = "evaluated"
        # Score based on profitability + FCF + health signals
        biz_positives = sum([
            net_margin is not None and net_margin > 0.10,  # Good margin
            fcf_positive is True,
            low_debt is True,
        ])
        biz_negatives = sum([
            net_margin is not None and net_margin < 0,
            fcf_positive is False,
            low_debt is False and health.get("debt_to_equity") is not None and health.get("debt_to_equity") > 1.5,
        ])

        if biz_positives >= 2 and biz_negatives == 0:
            business_quality_label = "strong"
        elif biz_positives >= 1 and biz_negatives <= 1:
            business_quality_label = "moderate"
        elif biz_negatives >= 2:
            business_quality_label = "poor"
        else:
            business_quality_label = "mixed"

    # Risk assessment from regime
    risk_label: str | None = None
    if risk_regime:
        regime_class = risk_regime.get("classification")
        if regime_class:
            risk_label = regime_class  # low/medium/high/extreme

    # Build business_quality_evidence to show basis for classification
    # This prevents confusion when classification doesn't match user expectations
    business_quality_evidence: dict[str, Any] = {}
    if business_quality_status == "evaluated_unprofitable":
        # Show what triggered "unprofitable" classification
        evidence_reasons: list[str] = []
        if net_margin is not None and net_margin < 0:
            evidence_reasons.append(f"net_margin={net_margin*100:.1f}%")
        if trailing_eps is not None and trailing_eps <= 0:
            evidence_reasons.append(f"trailing_eps=${trailing_eps:.2f}")
        if is_signaled_unprofitable:
            evidence_reasons.append("unprofitable_signal_fired")
        if pe_not_meaningful:
            evidence_reasons.append("pe_not_meaningful")
        business_quality_evidence = {
            "basis": "explicit_unprofitable_signals",
            "triggers": evidence_reasons if evidence_reasons else ["inferred_from_data"],
        }
    elif business_quality_status == "data_missing":
        # Show what data is missing
        missing_inputs: list[str] = []
        if net_margin is None:
            missing_inputs.append("net_margin")
        if pe_trailing is None:
            missing_inputs.append("pe_trailing")
        if trailing_eps is None:
            missing_inputs.append("trailing_eps")
        business_quality_evidence = {
            "basis": "insufficient_data",
            "missing_inputs": missing_inputs,
        }
    elif business_quality_status in ("evaluated", "available"):
        # Show what positives/negatives were counted
        evidence_positives: list[str] = []
        evidence_negatives: list[str] = []
        if net_margin is not None:
            if net_margin > 0.10:
                evidence_positives.append(f"net_margin={net_margin*100:.1f}%")
            elif net_margin < 0:
                evidence_negatives.append(f"net_margin={net_margin*100:.1f}%")
        if fcf_positive is True and fcf_value_label:
            evidence_positives.append(fcf_value_label)
        elif fcf_positive is False and fcf_value_label:
            evidence_negatives.append(fcf_value_label)
        if low_debt is True:
            evidence_positives.append("low_debt")
        elif low_debt is False and health.get("debt_to_equity") is not None:
            dte = health.get("debt_to_equity")
            if dte > 1.5:
                evidence_negatives.append(f"high_debt={dte:.1f}x")
        business_quality_evidence = {
            "basis": "fundamentals_scoring",
            "positives": evidence_positives,
            "negatives": evidence_negatives,
        }

    decomposed_scores = {
        "setup": setup_label,
        "business_quality": business_quality_label,
        "business_quality_status": business_quality_status,
        "business_quality_evidence": business_quality_evidence if business_quality_evidence else None,
        "risk": risk_label,
    }

    # Get revenue_yoy and fcf for horizon fit assessment
    growth_data = fundamentals_data.get("growth", {})
    cf_data = fundamentals_data.get("cash_flow", {})
    revenue_yoy_val = growth_data.get("revenue_yoy")
    fcf_ttm_val = cf_data.get("free_cash_flow_ttm")

    # Get burn_metrics for horizon fit
    burn_metrics = fundamentals_summary.get("burn_metrics") or {}

    # Build horizon fit assessment
    # mid_term (3-12 months): can lean on setup + regime-aware sizing
    # long_term (1-5 years): requires business quality + not extreme risk + runway
    horizon_fit = _build_horizon_fit(
        setup_label=setup_label,
        business_quality_label=business_quality_label,
        business_quality_status=business_quality_status,
        risk_label=risk_label,
        signals=signals,
        revenue_yoy=revenue_yoy_val,
        fcf_ttm=fcf_ttm_val,
        burn_metrics_status=burn_metrics.get("status"),
        runway_confidence=burn_metrics.get("runway_confidence"),
    )

    # Check minimum coverage for scoring
    required_coverage = ["price", "fundamentals"]
    has_minimum_coverage = all(coverage.get(k, False) for k in required_coverage)

    # Calculate weighted overall score
    score: float | None = None
    components: dict[str, float | None] = {
        "technicals": round(tech_score, 3) if tech_score is not None else None,
        "fundamentals": round(fund_score, 3) if fund_score is not None else None,
        "risk": round(risk_score, 3) if risk_score is not None else None,
    }

    # Explain why components are excluded from scoring
    # This prevents confusion when data is available but not used in scoring
    # Split "data availability" from "scoring usage" for investor clarity
    component_exclusions: dict[str, str] = {}
    if tech_score is None:
        component_exclusions["technicals"] = "no_technical_signals_fired"
    if fund_score is None:
        if not coverage.get("fundamentals", False):
            component_exclusions["fundamentals"] = "fundamentals_data_unavailable"
        elif business_quality_status == "evaluated_unprofitable":
            # Data is present, company evaluated as unprofitable, scored with unprofitable_signals_v1
            # This is NOT an exclusion - fundamentals WAS scored, just using different method
            # Only add to exclusions if no signals triggered
            component_exclusions["fundamentals"] = "no_fundamental_signals_fired"
        elif business_quality_status == "data_missing":
            # Coverage says we have fundamentals, but key inputs (margin, EPS, PE) are all None
            component_exclusions["fundamentals"] = "fundamentals_key_inputs_missing"
        else:
            # We have data and it's meaningful, but no signals triggered thresholds
            component_exclusions["fundamentals"] = "no_fundamental_signals_fired"
    if risk_score is None:
        if not coverage.get("price", False):
            component_exclusions["risk"] = "price_data_unavailable"
        else:
            component_exclusions["risk"] = "no_risk_signals_fired"

    # Track weights actually used (renormalized)
    weights_used: dict[str, float | None] = {}
    score_raw: float | None = None
    coverage_factor: float | None = None

    if has_minimum_coverage:
        # Weight available components
        weighted_sum = 0.0
        total_weight = 0.0

        if tech_score is not None:
            weighted_sum += tech_score * VERDICT_WEIGHTS["technicals"]
            total_weight += VERDICT_WEIGHTS["technicals"]
        if fund_score is not None:
            weighted_sum += fund_score * VERDICT_WEIGHTS["fundamentals"]
            total_weight += VERDICT_WEIGHTS["fundamentals"]
        if risk_score is not None:
            weighted_sum += risk_score * VERDICT_WEIGHTS["risk"]
            total_weight += VERDICT_WEIGHTS["risk"]

        if total_weight > 0:
            # Raw score from available components only (keep full precision)
            # INVARIANT: score_raw == sum(component_score * weight_used) for all components
            score_raw = weighted_sum / total_weight

            # Coverage factor: what fraction of full weights are present
            # Full weights = technicals + fundamentals + risk = 0.30 + 0.45 + 0.25 = 1.00
            full_weight = (
                VERDICT_WEIGHTS["technicals"]
                + VERDICT_WEIGHTS["fundamentals"]
                + VERDICT_WEIGHTS["risk"]
            )
            coverage_factor = total_weight / full_weight

            # Attenuate score by coverage factor to prevent partial-data overconfidence
            # This makes scores from partial data naturally smaller
            score = score_raw * coverage_factor

            # Record renormalized weights (what was actually used after renormalization)
            # Keep full precision - rounding happens only at display time
            if tech_score is not None:
                weights_used["technicals"] = VERDICT_WEIGHTS["technicals"] / total_weight
            if fund_score is not None:
                weights_used["fundamentals"] = VERDICT_WEIGHTS["fundamentals"] / total_weight
            if risk_score is not None:
                weights_used["risk"] = VERDICT_WEIGHTS["risk"] / total_weight

            # Invariant check: sum(weights_used) should ≈ 1.0
            weights_sum = sum(weights_used.values())
            if abs(weights_sum - 1.0) > 0.02:  # Allow small rounding error
                # This shouldn't happen, but log it for debugging
                pass  # weights_used sum check failed

    # Determine tilt and confidence
    tilt: str | None = None
    confidence: str = "low"

    # Build inputs_used for auditability (key numeric inputs that drove scoring)
    inputs_used: dict[str, float | None] = {}

    # Get key values from fundamentals
    vol = fundamentals_data.get("valuation", {})
    growth = fundamentals_data.get("growth", {})
    profit = fundamentals_data.get("profitability", {})
    yield_m = fundamentals_data.get("yield_metrics", {})
    health = fundamentals_data.get("financial_health", {})

    inputs_used["pe_trailing"] = vol.get("pe_trailing")
    inputs_used["peg_ratio"] = vol.get("peg_ratio")
    inputs_used["revenue_yoy"] = growth.get("revenue_yoy")
    inputs_used["net_margin"] = profit.get("net_margin")
    inputs_used["fcf_yield"] = yield_m.get("fcf_yield")
    inputs_used["debt_to_equity"] = health.get("debt_to_equity")

    # Get key values from risk data
    risk_vol = risk_data.get("volatility", {})
    risk_dd = risk_data.get("drawdown", {})
    risk_beta = risk_data.get("beta", {})

    inputs_used["annualized_vol"] = risk_vol.get("annualized")
    inputs_used["max_drawdown_1y"] = risk_dd.get("max_1y")
    inputs_used["beta"] = risk_beta.get("value")

    # Check for confidence-limiting factors
    bearish_list = signals.get("bearish", [])
    is_unprofitable = "unprofitable" in bearish_list
    has_negative_fcf = "negative_free_cash_flow" in bearish_list
    has_extreme_risk = "very_high_volatility" in bearish_list or "deep_drawdown" in bearish_list
    fundamentals_available = coverage.get("fundamentals", False)
    risk_available = coverage.get("risk", False)

    if score is not None:
        if score > 0.2:
            tilt = "bullish"
        elif score < -0.2:
            tilt = "bearish"
        else:
            tilt = "neutral"

        # Confidence based on score magnitude, coverage, AND data quality
        coverage_count = sum(1 for v in coverage.values() if v)

        # Start with score-based confidence
        if abs(score) > 0.5 and coverage_count >= 4:
            confidence = "high"
        elif abs(score) > 0.3 and coverage_count >= 3:
            confidence = "moderate"
        else:
            confidence = "low"

        # Cap confidence if critical data missing or company has red flags
        if not fundamentals_available:
            # Can't be high confidence without fundamentals
            if confidence == "high":
                confidence = "moderate"
        if not risk_available:
            # Can't be high confidence without risk data
            if confidence == "high":
                confidence = "moderate"
        if is_unprofitable or has_negative_fcf:
            # Unprofitable companies should not get high confidence bullish
            if tilt == "bullish" and confidence == "high":
                confidence = "moderate"
        if has_extreme_risk:
            # Extreme risk caps confidence
            if confidence == "high":
                confidence = "moderate"

    # Get additional data for condition-based confidence path
    risk_regime_class = risk_regime.get("classification") if risk_regime else None
    sma_200_val = technicals_data.get("moving_averages", {}).get("sma_200")
    current_price_val = technicals_data.get("current_price")
    max_dd_val = inputs_used.get("max_drawdown_1y")
    ann_vol_val = inputs_used.get("annualized_vol")

    # Build confidence path (what would upgrade/downgrade confidence)
    confidence_path = _build_confidence_path(
        confidence=confidence,
        tilt=tilt,
        fundamentals_available=fundamentals_available,
        risk_available=risk_available,
        is_unprofitable=is_unprofitable,
        has_negative_fcf=has_negative_fcf,
        has_extreme_risk=has_extreme_risk,
        coverage_count=sum(1 for v in coverage.values() if v),
        risk_regime=risk_regime_class,
        burn_metrics_status=burn_metrics.get("status"),
        sma_200=sma_200_val,
        current_price=current_price_val,
        max_drawdown=max_dd_val,
        annualized_vol=ann_vol_val,
    )

    # Build score_display for clean presentation
    # Pre-formatted strings to avoid rendering logic in consumers
    score_display: dict[str, str | None] = {}
    if score is not None and score_raw is not None and coverage_factor is not None:
        score_display["score"] = f"{score:.6f}"
        score_display["score_raw"] = f"{score_raw:.6f}"
        score_display["coverage_factor"] = f"{coverage_factor:.6f}"
        score_display["formula"] = f"{score:.6f} = {score_raw:.6f} × {coverage_factor:.6f}"

        # Build component breakdown for audit
        component_parts: list[str] = []
        if weights_used:
            for comp_name in ["technicals", "fundamentals", "risk"]:
                if comp_name in weights_used and components.get(comp_name) is not None:
                    comp_score = components[comp_name]
                    weight = weights_used[comp_name]
                    delta = comp_score * weight  # type: ignore[operator]
                    component_parts.append(f"{comp_name}={comp_score:.3f}×{weight:.6f}={delta:.6f}")
        score_display["component_breakdown"] = " + ".join(component_parts) if component_parts else None
    else:
        score_display = {"score": None, "score_raw": None, "coverage_factor": None, "formula": None, "component_breakdown": None}

    # Build structured coverage that separates "data availability" from "scoring usage"
    # This prevents confusion when coverage=True but component=None
    # Added: in_score_model (is this component part of the scoring model at all?)
    # Added: scoring_method (which scoring algorithm is used for this component?)
    structured_coverage: dict[str, dict[str, Any]] = {}

    # Map coverage keys to component names for exclusion lookup
    coverage_to_component = {
        "price": "technicals",  # price data feeds technicals + risk
        "fundamentals": "fundamentals",
        "risk": "risk",
        "news": None,  # news not used in scoring
        "events": None,  # events not used in scoring
    }

    # Determine scoring method for fundamentals based on business_quality_status
    fundamentals_scoring_method: str | None = None
    if business_quality_status == "evaluated_unprofitable":
        fundamentals_scoring_method = "unprofitable_signals_v1"
    elif business_quality_status in ("evaluated", "available"):
        fundamentals_scoring_method = "fundamentals_signals_v1"
    elif business_quality_status == "data_missing":
        fundamentals_scoring_method = None  # No method - data missing

    for cov_key, cov_fetched in coverage.items():
        component_name = coverage_to_component.get(cov_key)

        # Determine if this component is in the score model (by design, not by data availability)
        # News and events are NOT in the score model - they're contextual only
        in_score_model = component_name is not None

        # Determine scoring_method for each component
        scoring_method: str | None = None
        if component_name == "technicals":
            scoring_method = "technicals_signals_v1"
        elif component_name == "fundamentals":
            scoring_method = fundamentals_scoring_method
        elif component_name == "risk":
            scoring_method = "risk_signals_v1"
        # news and events: scoring_method = None (not scored)

        # Determine if used in score (only applicable to scored components)
        if not in_score_model:
            # Not a scored component (news, events) - explicitly False, not None
            used_in_score = False
            reason_excluded = "not_in_score_model"
        elif cov_fetched:
            # Data was fetched - check if actually used in score
            component_score = components.get(component_name)
            used_in_score = component_score is not None
            reason_excluded = component_exclusions.get(component_name) if not used_in_score else None
        else:
            # Data was not fetched
            used_in_score = False
            reason_excluded = f"{cov_key}_data_unavailable"

        structured_coverage[cov_key] = {
            "fetched": cov_fetched,
            "in_score_model": in_score_model,
            "scoring_method": scoring_method,
            "used_in_score": used_in_score,
            "reason_excluded": reason_excluded,
        }

    verdict = {
        "score": score,
        "score_raw": score_raw,
        "coverage_factor": coverage_factor,
        "score_display": score_display,
        "tilt": tilt,
        "confidence": confidence,
        "confidence_path": confidence_path,
        "coverage": structured_coverage,
        "components": components,
        "component_exclusions": component_exclusions if component_exclusions else None,
        "decomposed": decomposed_scores,
        "horizon_fit": horizon_fit,
        "weights_full": dict(VERDICT_WEIGHTS),  # Original weights for audit
        "weights_used": weights_used if weights_used else None,
        "inputs_used": inputs_used,
        "pros": pros[:5],  # Top 5
        "cons": cons[:5],  # Top 5
        "method": "weighted_signals_v2",
    }

    # Validate invariants (debug mode only, prevents disagreement between fields)
    _validate_verdict_invariants(verdict)

    return verdict


def _validate_verdict_invariants(verdict: dict[str, Any]) -> None:
    """
    Validate invariants between coverage, weights_used, components, and coverage_factor.

    Invariants enforced:
    1. If coverage.<component>.used_in_score == True, components.<component> must be non-null
    2. If components.<component> is None, coverage.<component>.used_in_score must be False/None
    3. weights_used keys must match components with non-null scores
    4. coverage_factor must equal sum of weights for used components / total weights

    Logs warnings for violations rather than raising (production-safe).
    """
    import logging
    logger = logging.getLogger(__name__)

    coverage = verdict.get("coverage", {})
    components = verdict.get("components", {})
    weights_used = verdict.get("weights_used") or {}
    coverage_factor = verdict.get("coverage_factor")

    # Map coverage keys to component names
    coverage_to_component = {
        "price": "technicals",
        "fundamentals": "fundamentals",
        "risk": "risk",
    }

    violations: list[str] = []

    # Invariant 1 & 2: coverage.used_in_score must agree with components
    for cov_key, comp_name in coverage_to_component.items():
        cov_data = coverage.get(cov_key, {})
        used_in_score = cov_data.get("used_in_score")
        comp_score = components.get(comp_name)

        if used_in_score is True and comp_score is None:
            violations.append(
                f"coverage.{cov_key}.used_in_score=True but components.{comp_name}=None"
            )
        if comp_score is not None and used_in_score is False:
            violations.append(
                f"components.{comp_name}={comp_score} but coverage.{cov_key}.used_in_score=False"
            )

    # Invariant 3: weights_used keys must match non-null component scores
    for comp_name, comp_score in components.items():
        if comp_score is not None:
            if comp_name not in weights_used:
                violations.append(
                    f"components.{comp_name}={comp_score} but {comp_name} not in weights_used"
                )
        else:
            if comp_name in weights_used:
                violations.append(
                    f"components.{comp_name}=None but {comp_name} in weights_used"
                )

    # Invariant 4: coverage_factor consistency (if calculable)
    # NOTE: weights_used is RENORMALIZED to sum to 1.0 across used components.
    # coverage_factor reflects the fraction of full weights actually present.
    if coverage_factor is not None and weights_used:
        weights_full = verdict.get("weights_full") or {}
        full_weight = sum(float(w) for w in weights_full.values() if w is not None)
        used_full_weight = sum(
            float(weights_full.get(comp_name, 0.0) or 0.0)
            for comp_name in weights_used
        )
        expected_coverage_factor = (
            used_full_weight / full_weight
            if full_weight > 0
            else None
        )

        if expected_coverage_factor is not None and abs(expected_coverage_factor - coverage_factor) > 0.0001:
            violations.append(
                f"coverage_factor={coverage_factor} but expected_coverage_factor={expected_coverage_factor}"
            )

    if violations:
        for v in violations:
            logger.warning(f"Verdict invariant violation: {v}")


def _build_confidence_path(
    confidence: str,
    tilt: str | None,
    fundamentals_available: bool,
    risk_available: bool,
    is_unprofitable: bool,
    has_negative_fcf: bool,
    has_extreme_risk: bool,
    coverage_count: int,
    risk_regime: str | None = None,
    burn_metrics_status: str | None = None,
    sma_200: float | None = None,
    current_price: float | None = None,
    max_drawdown: float | None = None,
    annualized_vol: float | None = None,
) -> dict[str, Any]:
    """
    Build confidence upgrade/downgrade path.

    Shows what would change the current confidence level up or down.
    Uses condition-based triggers instead of score-based.
    """
    upgrade_conditions: list[dict[str, Any]] = []
    downgrade_conditions: list[dict[str, Any]] = []
    current_blockers: list[str] = []
    note: str | None = None

    # Identify current blockers that cap confidence
    if not fundamentals_available:
        current_blockers.append("fundamentals_missing")
    if not risk_available:
        current_blockers.append("risk_data_missing")
    if is_unprofitable:
        current_blockers.append("unprofitable")
    if has_negative_fcf:
        current_blockers.append("negative_fcf")
    if has_extreme_risk:
        current_blockers.append("extreme_risk")
    if burn_metrics_status == "unavailable" and is_unprofitable:
        current_blockers.append("burn_metrics_missing")

    # Build condition-based upgrade conditions
    if confidence == "low":
        # For low confidence, focus on data availability and risk improvement
        if has_extreme_risk:
            added_specific = False
            if risk_regime == "extreme":
                if max_drawdown is not None and max_drawdown <= -0.50:
                    upgrade_conditions.append({
                        "condition": "drawdown_recovers",
                        "threshold": "max_drawdown_1y > -50%",
                        "current": f"{max_drawdown*100:.0f}%",
                    })
                    added_specific = True
                vol_threshold = vol_threshold_for_improvement(risk_regime, annualized_vol)
                if annualized_vol is not None and vol_threshold is not None and annualized_vol >= vol_threshold:
                    upgrade_conditions.append({
                        "condition": "volatility_decreases",
                        "threshold": f"annualized_vol < {vol_threshold * 100:.0f}%",
                        "current": f"{annualized_vol*100:.0f}%",
                    })
                    added_specific = True
            if not added_specific:
                upgrade_conditions.append({
                    "condition": "risk_regime_improves",
                    "threshold": "risk_regime <= high",
                    "current": f"risk_regime={risk_regime}",
                })
        if is_unprofitable:
            upgrade_conditions.append({
                "condition": "company_turns_profitable",
                "threshold": "net_margin > 0 for 2 quarters",
            })
        if has_negative_fcf:
            upgrade_conditions.append({
                "condition": "fcf_turns_positive",
                "threshold": "FCF > 0 for 2 quarters",
            })
        if burn_metrics_status == "unavailable" and is_unprofitable:
            upgrade_conditions.append({
                "condition": "burn_metrics_available",
                "threshold": "liquidity and cash flow data populated",
            })
        if not has_extreme_risk:
            vol_threshold = vol_threshold_for_improvement(risk_regime, annualized_vol)
            if annualized_vol is not None and vol_threshold is not None and annualized_vol >= vol_threshold:
                upgrade_conditions.append({
                    "condition": "volatility_decreases",
                    "threshold": f"annualized_vol < {vol_threshold * 100:.0f}%",
                    "current": f"{annualized_vol*100:.0f}%",
                })

    elif confidence == "moderate":
        # For moderate confidence, focus on quality improvements
        if is_unprofitable:
            upgrade_conditions.append({
                "condition": "achieves_profitability",
                "threshold": "net_margin > 5% sustained",
            })
        if has_extreme_risk:
            upgrade_conditions.append({
                "condition": "risk_normalizes",
                "threshold": "risk_regime <= medium",
                "current": f"risk_regime={risk_regime}",
            })
        if max_drawdown is not None and max_drawdown < -0.35:
            upgrade_conditions.append({
                "condition": "drawdown_recovers",
                "threshold": "max_drawdown_1y > -30%",
                "current": f"{max_drawdown*100:.0f}%",
            })

    elif confidence == "high":
        note = "already at highest level"

    # Build condition-based downgrade conditions
    if confidence == "high":
        downgrade_conditions.append({
            "condition": "earnings_miss_or_guidance_cut",
            "threshold": "EPS misses by >10% or guidance lowered",
        })
        if sma_200 is not None and current_price is not None:
            downgrade_conditions.append({
                "condition": "breaks_sma200",
                "threshold": f"price < ${sma_200:.2f} for 3 sessions",
                "current": f"${current_price:.2f}",
            })
        downgrade_conditions.append({
            "condition": "fcf_turns_negative",
            "threshold": "FCF < 0 for 2 quarters",
        })

    elif confidence == "moderate":
        if sma_200 is not None and current_price is not None:
            downgrade_conditions.append({
                "condition": "breaks_sma200",
                "threshold": f"price < ${sma_200:.2f} by >5%",
                "current": f"${current_price:.2f}",
            })
        if max_drawdown is not None:
            downgrade_conditions.append({
                "condition": "drawdown_worsens",
                "threshold": "max_drawdown_1y < -60%",
                "current": f"{max_drawdown*100:.0f}%",
            })
        downgrade_conditions.append({
            "condition": "risk_regime_worsens",
            "threshold": "risk_regime becomes extreme",
            "current": f"risk_regime={risk_regime}",
        })

    elif confidence == "low":
        # Low confidence can't really downgrade further
        note = "confidence already low"

    return {
        "current": confidence,
        "upgrade_if": upgrade_conditions[:3] if upgrade_conditions else None,
        "downgrade_if": downgrade_conditions[:3] if downgrade_conditions else None,
        "current_blockers": current_blockers if current_blockers else None,
        "note": note,
    }


def _build_horizon_fit(
    setup_label: str | None,
    business_quality_label: str | None,
    business_quality_status: str,
    risk_label: str | None,
    signals: dict[str, list[str]],
    revenue_yoy: float | None = None,
    fcf_ttm: float | None = None,
    burn_metrics_status: str | None = None,
    runway_confidence: str | None = None,
) -> dict[str, Any]:
    """
    Build horizon fit assessment for mid-term and long-term investors.

    mid_term (3-12 months): can lean on setup + regime-aware sizing
    long_term (1-5 years): requires business quality + not extreme risk + runway

    Investment policy gates:
    - Long-term = avoid if: extreme risk AND revenue_yoy < -20% AND fcf < 0
    - Long-term = caution if: unprofitable with declining revenue
    - Long-term = caution if: unprofitable with low runway_confidence
    - Long-term = ok only if: business quality moderate+ AND risk <= high
    - Long-term = data_missing if: unprofitable AND burn_metrics unavailable
    """
    mid_term_fit: str = "unknown"
    long_term_fit: str = "unknown"
    reasons: list[str] = []
    data_gaps: list[str] = []

    has_extreme_risk = risk_label == "extreme"
    has_high_risk = risk_label in ("high", "extreme")
    has_severe_revenue_decline = revenue_yoy is not None and revenue_yoy < -0.20
    has_negative_fcf = fcf_ttm is not None and fcf_ttm < 0
    is_unprofitable = business_quality_label == "unprofitable"
    burn_metrics_missing = burn_metrics_status == "unavailable"

    # Track data gaps for unprofitable companies
    if is_unprofitable and burn_metrics_missing:
        data_gaps.append("burn_metrics_unavailable")

    # Mid-term assessment (3-12 months)
    # Can work with setup alone, but risk regime matters for sizing
    if setup_label is None:
        mid_term_fit = "unknown"
        reasons.append("mid_term: insufficient technical data")
    elif has_extreme_risk:
        mid_term_fit = "caution"
        reasons.append("mid_term: extreme risk requires minimal position size")
    elif setup_label in ("strong", "moderate"):
        if has_high_risk:
            mid_term_fit = "ok"
            reasons.append("mid_term: setup favorable but size conservatively due to high risk")
        else:
            mid_term_fit = "ok"
            reasons.append("mid_term: setup supports position with standard sizing")
    elif setup_label == "weak":
        mid_term_fit = "caution"
        reasons.append("mid_term: weak setup suggests waiting for better entry")
    else:  # poor
        mid_term_fit = "avoid"
        reasons.append("mid_term: poor setup with negative technicals")

    # Long-term assessment (1-5 years)
    # Investment policy gates based on fundamentals + risk + trend
    # Accumulate ALL applicable concerns for 1:1 alignment with horizon_drivers
    long_term_concerns: list[str] = []
    long_term_gates: list[str] = []  # Track which gates fired for drivers

    if business_quality_status == "data_missing":
        long_term_fit = "unknown"
        reasons.append("long_term: fundamentals unavailable")
    else:
        # Collect all applicable concerns - each concern maps to a gate for 1:1 alignment
        if is_unprofitable or business_quality_status == "evaluated_unprofitable":
            long_term_concerns.append("unprofitable")
            long_term_gates.append("unprofitable")  # Explicit gate for unprofitability
            if burn_metrics_missing:
                long_term_concerns.append("burn_metrics_missing")
                long_term_gates.append("burn_metrics_missing")
                data_gaps.append("runway_assessment_blocked")
            elif runway_confidence == "low":
                # Burn metrics available but runway < 2 years = dilution risk
                long_term_concerns.append("low_runway_confidence")
                long_term_gates.append("low_runway_confidence")

        if has_extreme_risk:
            long_term_concerns.append("extreme_risk")
            long_term_gates.append("extreme_risk")

        if has_severe_revenue_decline and revenue_yoy is not None:
            long_term_concerns.append(f"revenue_decline_{abs(revenue_yoy)*100:.0f}pct")
            long_term_gates.append("severe_revenue_decline")

        if has_negative_fcf:
            long_term_concerns.append("negative_fcf")
            long_term_gates.append("negative_fcf")

        # Determine fit level based on severity
        if business_quality_label == "poor":
            long_term_fit = "avoid"
            reasons.append("long_term: poor business quality")
        elif has_extreme_risk and has_severe_revenue_decline and has_negative_fcf:
            # Triple threat = structural issues
            long_term_fit = "avoid"
            reasons.append(f"long_term: {' + '.join(long_term_concerns)}")
        elif is_unprofitable and revenue_yoy is not None and revenue_yoy < -0.30:
            # Unprofitable with collapsing revenue
            long_term_fit = "avoid"
            reasons.append(f"long_term: {' + '.join(long_term_concerns)}")
        elif long_term_concerns:
            # Any concerns = caution
            long_term_fit = "caution"
            reasons.append(f"long_term: {' + '.join(long_term_concerns)}")
        elif business_quality_label in ("strong", "moderate"):
            if has_high_risk:
                long_term_fit = "ok"
                reasons.append("long_term: business quality supports holding despite elevated risk")
            else:
                long_term_fit = "ok"
                reasons.append("long_term: business quality and risk support long-term holding")
        elif business_quality_label == "mixed":
            long_term_fit = "caution"
            reasons.append("long_term: mixed fundamentals suggest smaller position")
        else:
            long_term_fit = "unknown"
            reasons.append("long_term: insufficient data for assessment")

    return {
        "mid_term": mid_term_fit,
        "long_term": long_term_fit,
        "reasons": reasons,
        "data_gaps": data_gaps or [],
        "long_term_gates": long_term_gates or [],  # For horizon_drivers alignment
    }
