"""Minimal tests for human-readable section summary builders.

These builders are intentionally lightweight and expected to evolve.
Tests focus on: returns a string/None as expected, includes key signals,
and doesn't crash on missing/None inputs.
"""

from stock_mcp.tools.analyze.summaries import (
    build_analyst_summary_text,
    build_dip_summary_text,
    build_fundamentals_summary_text,
    build_news_summary_text,
    build_ownership_summary_text,
    build_risk_summary_text,
    build_short_interest_summary_text,
    build_technicals_summary_text,
)


def test_technicals_summary_downtrend_mentions_downtrend_or_below() -> None:
    technicals_summary = {
        "trend": {
            "above_sma50": False,
            "above_sma200": False,
            "golden_cross": False,
            "death_cross": True,
        },
        "momentum": {
            "rsi": 26.0,
            "rsi_overbought": False,
            "rsi_oversold": True,
            "macd_bullish": False,
        },
        "returns": {"return_1m": -0.12, "return_3m": -0.05, "return_1y": None},
        "position_in_52w_range": 0.1,
    }
    text = build_technicals_summary_text(technicals_summary)
    assert isinstance(text, str)
    lower = text.lower()
    assert ("downtrend" in lower) or ("below" in lower)
    assert "oversold" in lower


def test_technicals_summary_missing_data_does_not_crash() -> None:
    text = build_technicals_summary_text({})
    assert isinstance(text, str)


def test_fundamentals_summary_unprofitable_mentions_unprofitable() -> None:
    fundamentals_summary = {
        "valuation": {"valuation_note": "pe_not_meaningful", "ps_trailing": 2.3},
        "growth": {"revenue_yoy": 0.10, "eps_yoy": None},
        "profitability": {"gross_margin": 0.60, "net_margin": -0.20},
        "cash_flow": {"free_cash_flow_label": "FCF (TTM): +$103M"},
    }
    text = build_fundamentals_summary_text(fundamentals_summary)
    assert isinstance(text, str)
    assert "unprofitable" in text.lower()


def test_fundamentals_summary_empty_returns_none() -> None:
    assert build_fundamentals_summary_text({}) is None


def test_risk_summary_extreme_mentions_risk_regime() -> None:
    risk_summary = {
        "risk_regime": {"classification": "extreme"},
        "annualized_volatility": 0.62,
        "max_drawdown_1y": -0.50,
        "beta": 2.0,
    }
    text = build_risk_summary_text(risk_summary)
    assert isinstance(text, str)
    assert "risk regime" in text.lower()
    assert "extreme" in text.lower()


def test_news_summary_accepts_string_confidence() -> None:
    news_summary = {
        "article_count": 3,
        "sentiment": {"overall": "negative", "confidence": "moderate"},
    }
    text = build_news_summary_text(news_summary)
    assert isinstance(text, str)
    assert "sentiment" in text.lower()


def test_dip_summary_mentions_type() -> None:
    dip_assessment = {
        "dip_classification": {"type": "falling_knife"},
        "dip_depth": {"severity": "deep", "from_52w_high": "-45.2%"},
    }
    text = build_dip_summary_text(dip_assessment)
    assert isinstance(text, str)
    assert "falling_knife" in text


def test_ownership_short_interest_and_analyst_summaries_do_not_crash() -> None:
    ownership = {"insider_pct": 0.006, "institutional_pct": 0.75, "float_shares": 15_400_000_000}
    short_interest = {"short_pct_of_float": 0.012, "days_to_cover": 1.5, "short_change_mom": -0.05}
    analyst = {
        "rating": "buy",
        "rating_score": 2.1,
        "num_analysts": 35,
        "price_target_mean": 225.0,
        "upside_to_mean_target": 0.15,
    }

    ownership_text = build_ownership_summary_text(ownership)
    assert ownership_text is None or isinstance(ownership_text, str)

    short_text = build_short_interest_summary_text(short_interest)
    assert short_text is None or isinstance(short_text, str)

    analyst_text = build_analyst_summary_text(analyst, current_price=195.0, currency="USD")
    assert analyst_text is None or isinstance(analyst_text, str)
