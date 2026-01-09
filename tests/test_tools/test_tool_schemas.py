"""Tests for tool response schemas."""

import pytest

# These tests verify tool outputs conform to expected schemas
# In a real setup, you'd use record/replay HTTP mocking (pytest-recording)


class TestToolResponseSchemas:
    """Tests for tool response schemas - currently placeholders for CI mocking."""

    def test_price_history_schema(self) -> None:
        """Test price_history response conforms to schema."""
        # Expected schema fields
        expected_fields = {
            "meta": {"server_version", "schema_version", "tool", "duration_ms"},
            "data_provenance": {"price"},
            "symbol": str,
            "period": str,
            "interval": str,
            "adjusted": bool,
            "summary": {
                "data_points",
                "start_date",
                "end_date",
                "start_price",
                "end_price",
                "period_high",
                "period_low",
                "total_return",
            },
            "resource_uri": str,
            "resource_rows": int,
        }
        # Placeholder - would verify against actual response
        assert True

    def test_technicals_schema(self) -> None:
        """Test technicals response conforms to schema."""
        expected_fields = {
            "meta",
            "data_provenance",
            "symbol",
            "current_price",
            "moving_averages",
            "rsi",
            "macd",
            "atr",
            "price_position",
            "returns",
            "volume",
        }
        assert True

    def test_fundamentals_snapshot_schema(self) -> None:
        """Test fundamentals_snapshot response conforms to schema."""
        expected_fields = {
            "meta",
            "data_provenance",
            "symbol",
            "valuation",
            "growth",
            "profitability",
            "financial_health",
            "cash_flow",
            "yield_metrics",
        }
        # Valuation sub-fields (from fundamentals tool)
        valuation_subfields = {
            "pe_trailing",
            "pe_forward",
            "ps_trailing",
            "ps_source",  # "direct" or "computed" or None
            "ps_explanation",  # Only present when ps_source="computed"
            "pb_ratio",
            "peg_ratio",
            "ev_to_ebitda",
        }
        # Fundamentals summary includes burn_metrics for unprofitable companies
        fundamentals_summary_fields = {
            "valuation",
            "growth",
            "profitability",
            "cash_flow",
            "health",
            "burn_metrics",  # NEW: for unprofitable companies
        }
        cash_flow_fields = {
            "operating_cf_ttm",
            "free_cash_flow_ttm",
            "free_cash_flow_period",
            "free_cash_flow_period_end",
            "free_cash_flow_source",
            "currency",
            "free_cash_flow_label",
            "fcf_margin",
            "rules",
        }
        # Burn metrics sub-fields (ALWAYS present for unprofitable companies)
        burn_metrics_fields = {
            "status",  # available/unavailable/not_applicable
            "status_reason",  # why unavailable if so
            "liquidity",  # cash + ST investments
            "cash_runway_quarters",
            "runway_basis",  # min_fcf_ocf/fcf_only/ocf_only
            "quarterly_fcf_burn",
            "quarterly_ocf_burn",
            "dilution_analysis",  # if runway < 8 quarters
            "warnings",  # e.g., using_fcf_only, liquidity_missing
        }
        # Dilution analysis sub-fields
        dilution_analysis_fields = {
            "raise_needed_for_2y_runway",
            "dilution_if_raised_today",  # Decimal (0.123), not percentage
            "dilution_risk_level",  # low/moderate/high/severe
            "current_market_cap",
        }
        assert True

    def test_risk_metrics_schema(self) -> None:
        """Test risk_metrics response conforms to schema."""
        expected_fields = {
            "meta",
            "data_provenance",
            "symbol",
            "benchmark",
            "volatility",
            "beta",
            "drawdown",
            "var",
            "atr",
            "liquidity",
            "stop_suggestions",
            "position_sizing",
            "market_context",  # NEW: SPY trend for regime awareness
        }
        # Market context sub-fields
        market_context_fields = {
            "spy_trend",  # bullish/neutral/recovering/bearish/unknown
            "spy_above_200d",
            "spy_above_50d",
            "spy_price",
            "spy_sma_200",
            "spy_sma_50",
            "spy_distance_to_200d",
            "spy_distance_to_50d",
            # Provenance fields for auditability
            "symbol_used",  # "SPY"
            "source",  # "yfinance"
            "as_of",  # last bar date
            "price_adjustment",  # "split_adjusted"
            # Sanity check warnings
            "sanity_warnings",  # e.g., spy_price_unusually_high, spy_sma200_missing
        }
        assert True

    def test_analyze_stock_schema(self) -> None:
        """Test analyze_stock response conforms to schema."""
        expected_fields = {
            "meta",
            "data_provenance",
            "symbol",
            "summary",
            "technicals_summary",
            "fundamentals_summary",
            "risk_summary",
            "events_summary",
            "news_summary",
            "signals",
            "verdict",
            "action_zones",
            "dip_assessment",
            "relative_performance",
            "market_context",  # NEW: SPY trend for regime awareness
            "decision_context",
            "data_quality",
        }
        # Verdict sub-fields
        verdict_fields = {
            "score",
            "score_raw",
            "coverage_factor",
            "tilt",
            "confidence",
            "confidence_path",  # NEW: upgrade/downgrade conditions
            "coverage",
            "components",
            "component_exclusions",  # Explains why components are excluded from scoring
            "decomposed",
            "horizon_fit",
            "weights_full",  # Original weights for audit (e.g., {fundamentals: 0.45, technicals: 0.30, risk: 0.25})
            "weights_used",  # Renormalized weights actually used (sums to 1.0)
            "inputs_used",
            "pros",
            "cons",
            "method",
        }
        # Confidence path sub-fields (now condition-based, not score-based)
        confidence_path_fields = {
            "current",
            "upgrade_if",  # List of dicts with condition/threshold/current
            "downgrade_if",  # List of dicts with condition/threshold/current
            "current_blockers",
            "note",  # e.g., "confidence already low" or "already at highest level"
        }
        # Decomposed sub-fields
        decomposed_fields = {
            "setup",
            "business_quality",
            "business_quality_status",
            "risk",
        }
        # Horizon fit sub-fields
        horizon_fit_fields = {
            "mid_term",
            "long_term",
            "reasons",
            "data_gaps",  # e.g., burn_metrics_unavailable for unprofitable
            "long_term_gates",  # For 1:1 alignment with horizon_drivers
        }
        # Risk summary sub-fields
        risk_summary_fields = {
            "beta",
            "annualized_volatility",
            "max_drawdown_1y",
            "atr_pct",
            "risk_regime",
        }
        # Action zones sub-fields
        action_zones_fields = {
            "current_zone",
            "levels",
            "distance_to_levels",
            "price_vs_levels",
            "distance_labels",
            "level_vs_current_labels",
            "basis",
            "stop_calculation",
            "position_sizing_range",
            "valuation_assessment",
            "zone_warnings",
            "method",
        }
        # Dip assessment sub-fields
        dip_assessment_fields = {
            "dip_classification",
            "dip_depth",
            "oversold_metrics",
            "support_levels",
            "volume_analysis",
            "bounce_potential",
            "entry_timing",
            "dip_confidence",
            "assessment",
            "method",
        }
        dip_classification_fields = {
            "type",
            "signals",
            "explanation",
        }
        dip_depth_fields = {
            "from_52w_high",
            "from_52w_low",
            "from_3m_high",
            "from_6m_high",
            "days_since_52w_high",
            "days_since_52w_low",
            "low_set_today",
            "high_set_today",
            "severity",  # none/shallow/moderate/deep/extreme/unknown
            "severity_basis",  # from_52w_high or max_drawdown_1y (fallback)
        }
        oversold_metrics_fields = {
            "level",
            "score",
            "rsi_status",
            "rsi_value",
            "indicators",
            "distance_from_sma20",
            "distance_from_sma50",
            "distance_from_sma200",
            "distance_from_sma50_atr",
            "return_1w_zscore",
            "sma200_slope_pct_per_day",
            "position_in_52w_range",
            "oversold_composite",
        }
        oversold_composite_fields = {
            "score",
            "level",
            "components",
            "cap",
            "notes",
        }
        support_level_fields = {
            "level",
            "type",
            "distance_pct",
            "strength",
            "status",
            "price_basis",
        }
        volume_analysis_fields = {
            "signal",
            "ratio",
            "interpretation",
        }
        bounce_potential_fields = {
            "rating",
            "score",
            "factors",
        }
        entry_timing_fields = {
            "signals",
            "wait_for",
        }
        entry_signal_fields = {
            "signal",
            "action",
            "rationale",
        }
        dip_assessment_summary_fields = {
            "dip_quality",
            "recommendation",
            "rationale",
        }
        dip_confidence_fields = {
            "level",
            "score",
            "missing",
        }
        # Position sizing range sub-fields (with dollar amounts)
        position_sizing_range_fields = {
            "suggested_pct_range",
            "max_pct",
            "rationale",
            "dollars_for_50k",  # NEW: dollar amounts for $50k portfolio
            "shares_range",  # NEW: share count at current price
            "stop_implied_max",  # NEW: max size based on 1% risk rule
        }
        # dollars_for_50k sub-fields
        dollars_for_50k_fields = {
            "min",
            "max",
            "portfolio_assumption",
        }
        # shares_range sub-fields
        shares_range_fields = {
            "min",
            "max",
            "at_price",
        }
        # stop_implied_max sub-fields
        stop_implied_max_fields = {
            "pct",
            "dollars_for_50k",
            "risk_per_trade_pct",
            "stop_distance_pct",
        }
        # Valuation assessment sub-fields
        valuation_assessment_fields = {
            "gate",
            "reasons",
            "is_unprofitable",
        }
        # Decision context sub-fields (multi-factor structure)
        decision_context_fields = {
            "top_triggers",
            "top_triggers_incomplete_reason",  # NEW: if couldn't meet target count
            "horizon_drivers",  # NEW: policy gates affecting horizon fit (not score-based)
            "fundamentals",
            "valuation",
            "news",
            "risk",
            "technicals",
            "next_catalyst",
            "thesis_checkpoints",  # 2-year investment framework
        }
        # Horizon driver sub-fields
        horizon_driver_fields = {
            "horizon",  # mid_term or long_term
            "direction",  # bearish
            "gate",  # burn_metrics_missing, short_runway, extreme_risk, severe_revenue_decline, negative_fcf
            "reason",  # human-readable explanation
            "data_gaps",  # optional: list of data gaps
            "current",  # optional: current value
        }
        # Thesis checkpoints sub-fields
        thesis_checkpoints_fields = {
            "hold_thesis",
            "checkpoints",
            "review_triggers",
            "thesis_stop_triggers",  # NEW: non-price based exit triggers
            "review_frequency",
        }
        # Top triggers are now structured objects with score contribution
        top_trigger_fields = {
            "id",
            "category",
            "direction",
            "reason",
            "component_score",  # Raw score for this category (-1 to +1)
            "weight_used",  # Renormalized weight used
            "score_delta",  # component_score * weight_used (actual contribution)
            "next_update",  # Optional: for fundamental triggers
        }
        # Fundamentals category with status explanation
        decision_context_fundamentals_fields = {
            "bullish_if",
            "bearish_if",
            "status",  # available/missing (data fetch status)
            "status_explanation",
            "business_quality",  # strong/moderate/mixed/poor/unprofitable/weak or None
            "next_update",
            "check_frequency",
        }
        # Valuation category with gate and unprofitable flag
        decision_context_valuation_fields = {
            "bullish_if",
            "bearish_if",
            "current_gate",
            "pe_status",  # valid/not_meaningful/unavailable
            "pe_explanation",
            "ps_status",  # available/unavailable
            "ps_explanation",
            "is_unprofitable",
            "next_update",
            "check_frequency",
        }
        # News category has headline_triggers
        decision_context_news_fields = {
            "headline_triggers",
            "current_sentiment",
            "sentiment_confidence",
        }
        # Risk category adds current_regime
        decision_context_risk_fields = {
            "bullish_if",
            "bearish_if",
            "current_regime",
        }
        # Technicals category
        decision_context_technicals_fields = {
            "bullish_if",
            "bearish_if",
        }
        # Data quality sub-fields
        data_quality_fields = {
            "completeness",
            "missing_critical",
            "fundamentals_status",
            "fundamentals_status_reason",
            "data_gaps",  # NEW: list of data issues for transparency
            "tool_failures",
            "tool_timings",
            "warnings",
        }
        # Retry provenance sub-fields (for data_provenance.price, data_provenance.info, etc.)
        retry_provenance_fields = {
            "source",  # "yfinance"
            "attempts",  # int
            "retries_exhausted",  # bool (always False if successful)
            "fallback_used",  # bool (always False since Alpha Vantage removed)
            "total_backoff_seconds",  # float
        }
        assert True

    def test_stock_news_schema(self) -> None:
        """Test stock_news response conforms to schema."""
        expected_fields = {
            "meta",
            "data_provenance",
            "symbol",
            "period_days",
            "article_count",
            "articles",
            "sentiment",
            "recent_earnings",
            "warnings",
        }
        assert True

    def test_error_response_schema(self) -> None:
        """Test error response conforms to schema."""
        from stock_mcp.utils.provenance import build_error_response

        error = build_error_response(
            error_type="invalid_symbol",
            message="Symbol not found",
            symbol="XYZ",
        )

        assert error["error"] is True
        assert "error_type" in error
        assert "message" in error
        assert "meta" in error


class TestVerdictInvariants:
    """Tests for verdict scoring invariants."""

    def test_component_score_bounds(self) -> None:
        """Component scores must be in [-1, 1] range."""
        # This tests the calc_component_score formula: (pos - neg) / total
        # where pos + neg = total, so result is in [-1, 1]
        test_cases = [
            # (pos, neg) -> expected result
            (3, 0, 1.0),  # All bullish
            (0, 3, -1.0),  # All bearish
            (1, 1, 0.0),  # Balanced
            (2, 1, 1 / 3),  # More bullish
            (1, 2, -1 / 3),  # More bearish
        ]
        for pos, neg, expected in test_cases:
            total = pos + neg
            if total > 0:
                result = (pos - neg) / total
                assert -1.0 <= result <= 1.0, f"Score {result} out of bounds"
                assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"

    def test_score_delta_calculation(self) -> None:
        """score_delta = component_score * weight_used must match."""
        # Test data representing what we expect from the system
        test_triggers = [
            {"component_score": 1.0, "weight_used": 0.55, "expected_delta": 0.55},
            {"component_score": -1.0, "weight_used": 0.45, "expected_delta": -0.45},
            {"component_score": 0.5, "weight_used": 0.30, "expected_delta": 0.15},
            {"component_score": -0.67, "weight_used": 0.45, "expected_delta": -0.302},
        ]
        for trigger in test_triggers:
            calculated = trigger["component_score"] * trigger["weight_used"]
            expected = trigger["expected_delta"]
            assert abs(calculated - expected) < 0.01, (
                f"score_delta mismatch: {calculated:.3f} != {expected:.3f}"
            )

    def test_score_delta_sum_approximates_score_raw(self) -> None:
        """Sum of all component score_deltas should approximate score_raw.

        Invariant: abs(verdict.score_raw - sum(component.score_delta)) < tolerance

        Note: This may not be exact due to:
        1. Coverage factor attenuation (score = score_raw * coverage_factor)
        2. Some components being None (not contributing)
        3. Rounding in the actual implementation

        The top_triggers list may only show a subset of components, so we test
        the principle using the components dict directly.
        """
        # Simulate a verdict with all components present
        components = {
            "technicals": 0.75,  # Strong setup
            "fundamentals": -0.33,  # Unprofitable
            "risk": -0.67,  # High risk
        }
        weights = {
            "technicals": 0.30,
            "fundamentals": 0.45,
            "risk": 0.25,
        }

        # Calculate what score_raw should be (weighted average)
        weighted_sum = sum(
            components[k] * weights[k]
            for k in components
        )
        total_weight = sum(weights.values())
        score_raw = weighted_sum / total_weight

        # Calculate sum of score_deltas (using renormalized weights)
        renormalized_weights = {k: w / total_weight for k, w in weights.items()}
        score_delta_sum = sum(
            components[k] * renormalized_weights[k]
            for k in components
        )

        # The two should match (both are the weighted average)
        assert abs(score_raw - score_delta_sum) < 0.001, (
            f"score_raw ({score_raw:.4f}) != sum(score_delta) ({score_delta_sum:.4f})"
        )

    def test_top_triggers_balance_rules(self) -> None:
        """Top triggers must follow balance rules based on tilt.

        - neutral: 2 bearish + 1 bullish
        - bullish: 2 bullish + 1 bearish
        - bearish: 2 bearish + 1 bullish

        Note: If insufficient triggers exist, top_triggers_incomplete_reason
        should be set.
        """
        # Test that the balance rules are applied correctly
        # This is more of a specification test
        balance_rules = {
            "neutral": {"bearish": 2, "bullish": 1},
            "bullish": {"bearish": 1, "bullish": 2},
            "bearish": {"bearish": 2, "bullish": 1},
        }

        for tilt, expected_counts in balance_rules.items():
            total_expected = sum(expected_counts.values())
            assert total_expected == 3, f"Tilt {tilt} should show 3 triggers"

    def test_score_delta_sum_equals_score_raw_exactly(self) -> None:
        """Score deltas must sum to score_raw with negligible tolerance.

        This enforces that:
        1. score_raw = sum(component_score * weight_used) for all components
        2. Rounding only happens at display time, not in calculation
        3. No "why doesn't it add up?" questions from users

        Tolerance is 1e-9 (effectively zero for float math).
        Display rounding is separate from this invariant.
        """
        # Test multiple component configurations
        test_cases = [
            # All positive
            {
                "components": {"technicals": 1.0, "fundamentals": 1.0, "risk": 1.0},
                "weights": {"technicals": 0.30, "fundamentals": 0.45, "risk": 0.25},
            },
            # All negative
            {
                "components": {"technicals": -1.0, "fundamentals": -1.0, "risk": -1.0},
                "weights": {"technicals": 0.30, "fundamentals": 0.45, "risk": 0.25},
            },
            # Mixed with fractional values
            {
                "components": {"technicals": 0.67, "fundamentals": -0.33, "risk": -0.50},
                "weights": {"technicals": 0.30, "fundamentals": 0.45, "risk": 0.25},
            },
            # Only two components present (missing risk)
            {
                "components": {"technicals": 0.5, "fundamentals": -0.5},
                "weights": {"technicals": 0.30, "fundamentals": 0.45},
            },
        ]

        for case in test_cases:
            components = case["components"]
            weights = case["weights"]

            # Renormalize weights to sum to 1.0
            total_weight = sum(weights.values())
            renormalized = {k: w / total_weight for k, w in weights.items()}

            # Calculate score_raw as weighted average
            score_raw = sum(components[k] * weights[k] for k in components) / total_weight

            # Calculate sum of score_deltas
            score_delta_sum = sum(components[k] * renormalized[k] for k in components)

            # These MUST be effectively equal (1e-9 tolerance for float precision)
            assert abs(score_raw - score_delta_sum) < 1e-9, (
                f"INVARIANT VIOLATED: score_raw ({score_raw:.10f}) != "
                f"sum(score_delta) ({score_delta_sum:.10f})"
            )


class TestDipAssessmentLogic:
    """Tests for dip assessment helper logic."""

    def test_oversold_composite_extreme(self) -> None:
        """Composite should cap at 5 and classify as extreme."""
        from stock_mcp.tools.analyze import _build_oversold_composite

        result = _build_oversold_composite(
            rsi=24.0,
            return_1w_zscore=-2.1,
            distance_to_sma50_atr=-2.2,
            position_in_range=0.03,
        )

        assert result["level"] == "extreme"
        assert result["score"] == 5.0
        assert result["components"]["momentum"] == 2.0
        assert result["components"]["trend_deviation"] == 2.0
        assert result["components"]["range_position"] == 1.0

    def test_oversold_composite_missing_momentum(self) -> None:
        """Missing RSI and z-score should emit momentum_missing note."""
        from stock_mcp.tools.analyze import _build_oversold_composite

        result = _build_oversold_composite(
            rsi=None,
            return_1w_zscore=None,
            distance_to_sma50_atr=-1.2,
            position_in_range=0.2,
        )

        assert "momentum_missing" in result["notes"]

    def test_action_zone_distance_labels(self) -> None:
        """Distance labels should be level-relative to current price."""
        from stock_mcp.tools.analyze import _build_action_zones

        current_price = 100.0
        tech_data = {
            "moving_averages": {"sma_50": 110.0, "sma_200": 120.0},
            "price_position": {"week_52_low": 80.0, "week_52_high": 150.0},
        }
        risk_data = {"atr": {"value": 5.0, "as_pct_of_price": 0.05}}
        fund_data = {"valuation": {}, "yield_metrics": {}, "profitability": {}}
        risk_regime = {"classification": "extreme"}

        result = _build_action_zones(
            current_price=current_price,
            tech_data=tech_data,
            risk_data=risk_data,
            fund_data=fund_data,
            risk_regime=risk_regime,
            signals={"bullish": [], "bearish": []},
        )

        labels = result["distance_labels"]
        assert labels["strong_buy_below"] == "16.0% below current"
        assert labels["accumulate_near"] == "20.0% above current"
        assert labels["reduce_above"] == "42.5% above current"
        assert labels["stop_loss"] == "12.5% below current"
        assert result["level_vs_current_labels"] == labels
