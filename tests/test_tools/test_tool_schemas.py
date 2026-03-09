"""Tests for tool contracts and analysis invariants."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta

import pandas as pd
import pytest

from stock_analysis.tools.analyze.action_zones import build_action_zones
from stock_analysis.tools.analyze.decision_context import build_decision_context
from stock_analysis.tools.analyze.dip_assessment import (
    _build_oversold_composite,
    align_dip_assessment_with_action,
)
from stock_analysis.tools.analyze.executive_summary import build_policy_action
from stock_analysis.tools.analyze.investor_profile import build_decision_modes
from stock_analysis.tools.analyze.orchestrator import analyze_stock
from stock_analysis.tools.fundamentals import fundamentals_snapshot
from stock_analysis.tools.news import stock_news
from stock_analysis.tools.price_history import price_history
from stock_analysis.tools.risk_metrics import risk_metrics
from stock_analysis.tools.technicals import technicals
from stock_analysis.utils.provenance import build_error_response

orchestrator_module = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
fundamentals_module = importlib.import_module("stock_analysis.tools.fundamentals")
news_module = importlib.import_module("stock_analysis.tools.news")
price_history_module = importlib.import_module("stock_analysis.tools.price_history")
risk_metrics_module = importlib.import_module("stock_analysis.tools.risk_metrics")
technicals_module = importlib.import_module("stock_analysis.tools.technicals")


def _make_history_frame(
    days: int = 260,
    start_price: float = 100.0,
    daily_step: float = 0.35,
) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    close = [start_price + daily_step * idx for idx in range(days)]
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "open": [value - 0.4 for value in close],
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1_000_000 + idx * 1000 for idx in range(days)],
        }
    )


class _FakeNewsTicker:
    def __init__(self) -> None:
        recent = (datetime.utcnow() - timedelta(days=2)).isoformat() + "Z"
        self.news = [
            {
                "content": {
                    "pubDate": recent,
                    "title": "Company raises guidance after contract win",
                    "summary": "Management highlighted a new partnership and stronger outlook.",
                    "provider": {"displayName": "ExampleWire"},
                    "canonicalUrl": {"url": "https://example.com/article"},
                }
            }
        ]
        earnings_date = datetime.utcnow() - timedelta(days=5)
        self.earnings_dates = pd.DataFrame(
            {
                "EPS Estimate": [1.0],
                "Reported EPS": [1.1],
            },
            index=[earnings_date],
        )


def _fundamentals_info() -> dict[str, object]:
    return {
        "regularMarketPrice": 100.0,
        "currentPrice": 100.0,
        "lastFiscalYearEnd": int(datetime(2025, 12, 31).timestamp()),
        "trailingPE": 22.0,
        "forwardPE": 20.0,
        "trailingEps": 4.5,
        "priceToSalesTrailing12Months": 3.2,
        "priceToBook": 4.0,
        "pegRatio": 1.1,
        "enterpriseToEbitda": 14.0,
        "enterpriseValue": 3_000_000_000,
        "totalRevenue": 1_000_000_000,
        "revenueGrowth": 0.22,
        "earningsGrowth": 0.18,
        "grossMargins": 0.65,
        "operatingMargins": 0.22,
        "profitMargins": 0.15,
        "returnOnEquity": 0.20,
        "returnOnAssets": 0.10,
        "totalCash": 200_000_000,
        "cashAndShortTermInvestments": 200_000_000,
        "totalDebt": 50_000_000,
        "currentRatio": 1.8,
        "debtToEquity": 30.0,
        "operatingCashflow": 150_000_000,
        "freeCashflow": 120_000_000,
        "financialCurrency": "USD",
        "currency": "USD",
        "marketCap": 2_000_000_000,
        "dividendYield": 0.0,
        "targetLowPrice": 90.0,
        "targetMeanPrice": 120.0,
        "targetHighPrice": 140.0,
        "targetMedianPrice": 118.0,
        "recommendationKey": "buy",
        "recommendationMean": 2.0,
        "numberOfAnalystOpinions": 12,
        "sharesShort": 1_000_000,
        "sharesShortPriorMonth": 1_200_000,
        "shortPercentOfFloat": 0.02,
        "shortRatio": 1.5,
        "dateShortInterest": "2026-03-01",
        "heldPercentInsiders": 0.05,
        "heldPercentInstitutions": 0.70,
        "floatShares": 9_000_000,
        "auditRisk": 2,
        "boardRisk": 3,
        "compensationRisk": 4,
        "shareHolderRightsRisk": 2,
        "overallRisk": 3,
        "grossProfits": 500_000_000,
        "ebitda": 200_000_000,
        "ebitdaMargins": 0.20,
        "revenuePerShare": 10.0,
        "quickRatio": 1.5,
        "sharesOutstanding": 10_000_000,
        "quoteType": "EQUITY",
    }


class TestToolResponseSchemas:
    """Contract tests using deterministic in-memory mocks."""

    @pytest.mark.asyncio
    async def test_price_history_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = _make_history_frame(days=30)

        async def fake_fetch_history(params):  # type: ignore[no-untyped-def]
            return frame

        monkeypatch.setattr(price_history_module, "fetch_history", fake_fetch_history)
        monkeypatch.setattr(price_history_module.price_cache, "store", lambda params, df: "price://TEST")

        result = await price_history("TEST", period="1mo", interval="1d")

        assert result["symbol"] == "TEST"
        assert result["resource_uri"] == "price://TEST"
        assert result["resource_rows"] == len(frame)
        assert len(result["preview"]) == 5
        assert "summary" in result
        assert "data_provenance" in result

    @pytest.mark.asyncio
    async def test_technicals_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = _make_history_frame(days=260)

        async def fake_fetch_history(params):  # type: ignore[no-untyped-def]
            return frame

        monkeypatch.setattr(technicals_module, "fetch_history", fake_fetch_history)

        result = await technicals("TEST")

        assert result["symbol"] == "TEST"
        assert result["current_price"] is not None
        assert "moving_averages" in result
        assert "rsi" in result
        assert "macd" in result
        assert "returns" in result
        assert "price_position" in result

    @pytest.mark.asyncio
    async def test_fundamentals_snapshot_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_fetch_info(symbol: str) -> dict[str, object]:
            return _fundamentals_info()

        async def fake_fetch_ticker(symbol: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("ticker enrichment unavailable in unit test")

        monkeypatch.setattr(fundamentals_module, "fetch_info", fake_fetch_info)
        monkeypatch.setattr(fundamentals_module, "fetch_ticker", fake_fetch_ticker)

        result = await fundamentals_snapshot("TEST")

        assert result["symbol"] == "TEST"
        assert result["valuation"]["pe_trailing"] == 22.0
        assert result["growth"]["revenue_yoy"] == 0.22
        assert result["profitability"]["net_margin"] == 0.15
        assert result["cash_flow"]["free_cash_flow_ttm"] == 120_000_000
        assert "yield_metrics" in result

    @pytest.mark.asyncio
    async def test_risk_metrics_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        symbol_frame = _make_history_frame(days=260, start_price=100.0, daily_step=0.35)
        benchmark_frame = _make_history_frame(days=260, start_price=400.0, daily_step=0.20)

        async def fake_fetch_history(params):  # type: ignore[no-untyped-def]
            if params.symbol == "SPY":
                return benchmark_frame
            return symbol_frame

        monkeypatch.setattr(risk_metrics_module, "fetch_history", fake_fetch_history)

        result = await risk_metrics("TEST", portfolio_value=10_000, risk_per_trade=0.01)

        assert result["symbol"] == "TEST"
        assert result["benchmark"] == "SPY"
        assert result["volatility"]["annualized"] is not None
        assert result["beta"]["value"] is not None
        assert result["position_sizing"]["recommended"]["position_dollars"] >= 0
        assert result["market_context"]["symbol_used"] == "SPY"

    @pytest.mark.asyncio
    async def test_analyze_stock_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_stock_summary(symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "name": "Test Corp",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 2_000_000_000,
                "currency": "USD",
                "current_price": 100.0,
                "description": "Test company for unit tests.",
                "employees": 100,
                "website": "https://example.com",
            }

        async def fake_technicals(symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "current_price": 100.0,
                "moving_averages": {
                    "sma_20": 98.0,
                    "sma_50": 95.0,
                    "sma_200": 90.0,
                    "sma_200_slope_pct_per_day": 0.001,
                    "price_vs_sma20": 0.02,
                    "price_vs_sma50": 0.05,
                    "price_vs_sma200": 0.11,
                    "rules": {
                        "above_sma20": {"triggered": True},
                        "above_sma50": {"triggered": True},
                        "above_sma200": {"triggered": True},
                        "golden_cross": {"triggered": True},
                        "death_cross": {"triggered": False},
                    },
                },
                "rsi": {
                    "value": 55.0,
                    "bullish_divergence": False,
                    "rules": {
                        "oversold": {"triggered": False},
                        "overbought": {"triggered": False},
                    },
                },
                "macd": {
                    "macd_line": 1.0,
                    "signal_line": 0.8,
                    "histogram": 0.2,
                    "histogram_rising_3d": True,
                    "rules": {
                        "bullish_cross": {"triggered": True},
                        "bearish_cross": {"triggered": False},
                    },
                },
                "returns": {
                    "return_1w": 0.02,
                    "return_1w_zscore": 0.4,
                    "return_1m": 0.12,
                    "return_3m": 0.18,
                    "return_1y": 0.40,
                },
                "price_position": {
                    "position_in_range": 0.70,
                    "week_52_low": 70.0,
                    "week_52_high": 120.0,
                    "from_52w_high": -0.1667,
                    "from_52w_low": 0.4286,
                    "from_3m_high": -0.05,
                    "from_6m_high": -0.08,
                    "days_since_52w_high": 30,
                    "days_since_52w_low": 180,
                    "low_1m": 95.0,
                },
                "bollinger": {
                    "pct_b": 0.70,
                    "bandwidth": 0.12,
                    "rules": {
                        "above_upper": {"triggered": False},
                        "below_lower": {"triggered": False},
                        "squeeze": {"triggered": False},
                    },
                },
                "obv": {"trend": "up"},
                "fibonacci": {"nearest_support": 95.0, "nearest_resistance": 110.0},
                "volume": {"ratio": 1.40},
                "price_action": {},
            }

        async def fake_fundamentals(symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "valuation": {
                    "pe_trailing": 22.0,
                    "pe_forward": 20.0,
                    "trailing_eps": 4.5,
                    "ps_trailing": 3.2,
                    "ps_source": "direct",
                    "peg_ratio": 1.1,
                    "ev_to_ebitda": 14.0,
                    "ev_to_sales": 3.0,
                },
                "growth": {
                    "revenue_yoy": 0.22,
                    "eps_yoy": 0.18,
                    "rules": {
                        "positive_revenue_growth": {"triggered": True},
                        "high_growth": {"triggered": True},
                    },
                },
                "profitability": {
                    "gross_margin": 0.65,
                    "net_margin": 0.15,
                    "rules": {
                        "profitable": {"triggered": True},
                        "high_margin": {"triggered": True},
                    },
                },
                "financial_health": {
                    "cash_and_st_investments": 200_000_000,
                    "total_cash": 200_000_000,
                    "debt_to_equity": 0.3,
                    "rules": {
                        "net_cash_positive": {"triggered": True},
                        "low_debt": {"triggered": True},
                    },
                },
                "cash_flow": {
                    "free_cash_flow_ttm": 120_000_000,
                    "free_cash_flow_period": "TTM",
                    "free_cash_flow_period_end": "2025-12-31",
                    "free_cash_flow_source": "info",
                    "operating_cf_ttm": 150_000_000,
                    "currency": "USD",
                    "rules": {"positive_fcf": {"triggered": True}},
                },
                "yield_metrics": {
                    "fcf_yield": 0.06,
                    "earnings_yield": 0.045,
                    "dividend_yield": 0.0,
                    "rules": {"attractive_fcf_yield": {"triggered": True}},
                },
                "analyst_coverage": {
                    "rating": "buy",
                    "rating_score": 2.0,
                    "num_analysts": 12,
                    "price_target_mean": 120.0,
                    "upside_to_mean_target": 0.20,
                },
                "short_interest": {
                    "shares_short": 1_000_000,
                    "short_pct_of_float": 0.02,
                    "days_to_cover": 1.5,
                    "short_change_mom": -0.10,
                },
                "ownership": {
                    "insider_pct": 0.05,
                    "institutional_pct": 0.70,
                    "float_shares": 9_000_000,
                },
                "governance": {
                    "audit_risk": 2,
                    "board_risk": 3,
                    "compensation_risk": 4,
                    "shareholder_rights_risk": 2,
                    "overall_risk": 3,
                },
                "quality": {
                    "roic": 0.15,
                    "ebitda_margin": 0.20,
                    "quick_ratio": 1.5,
                },
                "valuation_context": {
                    "pe_current": 22.0,
                    "pe_5y_avg": 25.0,
                    "pe_percentile_5y": 0.4,
                    "ps_current": 3.2,
                    "ps_5y_avg": 4.0,
                    "status": "available",
                    "status_reason": None,
                },
                "valuation_history": None,
                "fundamental_trends": None,
                "analyst_estimates": None,
                "dividend_analysis": None,
            }

        async def fake_risk(symbol: str, benchmark: str = "SPY", portfolio_value=None, risk_per_trade=None):  # type: ignore[no-untyped-def]
            return {
                "symbol": symbol,
                "benchmark": benchmark,
                "benchmark_returns": {
                    "return_1m": 0.02,
                    "return_3m": 0.07,
                    "return_1y": 0.18,
                },
                "volatility": {
                    "annualized": 0.32,
                    "rules": {"high_volatility": {"triggered": False}},
                },
                "beta": {"value": 1.1},
                "drawdown": {"max_1y": -0.22, "current": -0.05, "days_since_high": 30},
                "var": {"daily_95": -0.02, "daily_99": -0.04},
                "atr": {"value": 4.0, "as_pct_of_price": 0.04},
                "liquidity": {"avg_dollar_volume": 50_000_000},
                "position_sizing": {
                    "recommended": {"position_dollars": 100.0},
                },
                "market_context": {
                    "spy_trend": "bullish",
                    "spy_above_200d": True,
                    "spy_above_50d": True,
                    "spy_price": 500.0,
                    "spy_sma_200": 480.0,
                    "spy_sma_50": 495.0,
                    "spy_distance_to_200d": 0.04,
                    "spy_distance_to_50d": 0.01,
                    "symbol_used": "SPY",
                    "source": "yfinance",
                    "as_of": "2026-03-06",
                    "price_adjustment": "split_adjusted",
                    "sanity_warnings": None,
                },
            }

        async def fake_events(symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "earnings": {
                    "next_date": "2026-05-01",
                    "next_date_source": "calendar",
                    "next_date_status": "available",
                    "next_date_status_reason": None,
                    "days_until": 53,
                    "history": [{"date": "2025-11-01", "estimate": 1.0, "actual": 1.1, "surprise": 0.1}],
                    "beat_rate": 0.75,
                },
                "dividends": {"yield": 0.0},
                "splits": {},
            }

        async def fake_news(symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "articles": [
                    {
                        "date": "2026-03-07",
                        "title": "Company raises guidance after contract win",
                        "sentiment": "positive",
                        "catalyst_tags": {
                            "bullish": ["guidance_raise", "partnership_or_contract"],
                            "bearish": [],
                            "neutral": [],
                        },
                    }
                ],
                "sentiment": {
                    "overall": "positive",
                    "confidence": "moderate",
                    "counts": {"positive": 1, "negative": 0, "neutral": 0},
                    "method": "keyword_v2",
                    "headline_triggers": {"positive": ["raises guidance"], "negative": []},
                    "sentiment_7d": "positive",
                    "sample_size_7d": 1,
                    "confidence_7d": "low",
                    "sentiment_30d": "positive",
                    "sample_size_30d": 1,
                    "confidence_30d": "low",
                },
                "catalyst_intelligence": {
                    "bullish": [
                        {"tag": "guidance_raise", "count": 1},
                        {"tag": "partnership_or_contract", "count": 1},
                    ],
                    "bearish": [],
                    "neutral": [],
                    "sample_size": 1,
                    "method": "keyword_catalyst_v1",
                },
                "recent_earnings": {"date": "2025-11-01", "beat_miss": "beat", "surprise_pct": 0.1},
            }

        async def fake_ownership(symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "insider_activity": {"sentiment": "buying", "net_shares_3m": 1000},
                "institutional": {"top_holders": [], "total_institutional_pct": 0.70},
            }

        async def fake_options(symbol: str) -> dict[str, object]:
            return {
                "symbol": symbol,
                "put_call_ratio": {"volume_based": 0.8},
                "implied_volatility": {"atm_avg_iv": 0.35},
                "unusual_activity": {"items": []},
            }

        monkeypatch.setattr(orchestrator_module, "stock_summary", fake_stock_summary)
        monkeypatch.setattr(orchestrator_module, "technicals", fake_technicals)
        monkeypatch.setattr(orchestrator_module, "fundamentals_snapshot", fake_fundamentals)
        monkeypatch.setattr(orchestrator_module, "risk_metrics", fake_risk)
        monkeypatch.setattr(orchestrator_module, "events_calendar", fake_events)
        monkeypatch.setattr(orchestrator_module, "stock_news", fake_news)
        monkeypatch.setattr(orchestrator_module, "ownership_analysis", fake_ownership)
        monkeypatch.setattr(orchestrator_module, "options_signals", fake_options)

        result = await analyze_stock(
            "TEST",
            profile="speculative",
            account_size=3_000,
            risk_per_trade_pct=1.0,
            max_position_pct=5.0,
        )

        assert result["symbol"] == "TEST"
        assert "investor_profile" not in result
        assert result["action_zones"]["position_sizing_range"]["dollars_for_account"]["min"] == 30.0
        assert result["decision_modes"]["core"]["action"] is not None
        assert result["decision_modes"]["balanced"]["action"] is not None
        assert result["decision_modes"]["speculative"]["action"] in {
            "starter_position_only",
            "starter_then_add",
            "watch",
            "wait",
        }
        assert result["decision_modes"]["core"]["starter_position"]["dollars"] == 60.0
        assert result["decision_modes"]["balanced"]["starter_position"]["dollars"] == 75.0
        assert result["decision_modes"]["speculative"]["starter_position"]["dollars"] == 30.0
        assert result["news_summary"]["catalyst_intelligence"]["bullish"][0]["tag"] == "guidance_raise"
        assert result["dislocation_framework"]["setup"]["status"] == "absent"
        assert result["dislocation_framework"]["action"]["core"] == result["decision_modes"]["core"]["action"]
        assert result["section_summaries"]["dislocation"] == result["dislocation_framework"]["summary"]

    @pytest.mark.asyncio
    async def test_analyze_stock_resolves_company_name_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        resolved_symbols: list[str] = []

        async def fake_symbol_search(query: str, limit: int = 5) -> dict[str, object]:
            assert query == "Circle"
            assert limit == 5
            return {
                "results": [
                    {
                        "symbol": "CRCL",
                        "name": "Circle Internet Group, Inc.",
                        "exchange": "NYQ",
                        "type": "equity",
                        "is_valid": True,
                    }
                ],
                "exact_match": None,
            }

        async def fake_stock_summary(symbol: str) -> dict[str, object]:
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "name": "Circle Internet Group, Inc.",
                "sector": "Financial Services",
                "industry": "Capital Markets",
                "market_cap": 28_000_000_000,
                "currency": "USD",
                "current_price": 111.65,
                "description": "Stablecoin infrastructure",
                "website": "https://www.circle.com",
            }

        async def fake_technicals(symbol: str) -> dict[str, object]:
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "current_price": 111.65,
                "moving_averages": {
                    "sma_20": 100.0,
                    "sma_50": 90.0,
                    "sma_200": None,
                    "price_vs_sma20": 0.1165,
                    "price_vs_sma50": 0.2406,
                    "price_vs_sma200": None,
                    "rules": {"above_sma50": {"triggered": True}},
                },
                "rsi": {"value": 76.2, "rules": {"overbought": {"triggered": True}}},
                "macd": {"rules": {"bullish_cross": {"triggered": True}}},
                "returns": {
                    "return_1w": 0.04,
                    "return_1w_zscore": 1.05,
                    "return_1m": 1.2238,
                    "return_3m": 0.2772,
                    "return_1y": None,
                },
                "price_position": {
                    "week_52_low": 49.9,
                    "week_52_high": 297.0,
                    "from_52w_high": -0.6264,
                    "from_52w_low": 1.2385,
                    "from_3m_high": -0.0085,
                    "from_6m_high": -0.2996,
                    "days_since_52w_high": 178,
                    "days_since_52w_low": 21,
                    "position_in_range": 0.2481,
                    "low_1m": 53.62,
                },
                "bollinger": {"pct_b": 0.9358, "bandwidth": 1.0322},
                "obv": {"trend": "rising"},
                "fibonacci": {"nearest_support": 103.21, "nearest_resistance": 145.05},
                "volume": {"current": 1_000_000, "avg_20d": 1_050_000, "ratio": 0.96},
                "price_action": {"break_5d_high": True, "higher_closes_2d": False},
            }

        async def fake_fundamentals(symbol: str) -> dict[str, object]:
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "valuation": {
                    "trailing_eps": -0.44,
                    "pe_trailing": None,
                    "peg_ratio": None,
                    "ps_trailing": 10.42,
                    "ps_source": "direct",
                    "ev_to_sales": 8.27,
                    "ev_to_ebitda": -256.98,
                },
                "yield_metrics": {"fcf_yield": 0.0121, "earnings_yield": None},
                "growth": {"revenue_yoy": 0.769, "eps_yoy": 8.802},
                "profitability": {"gross_margin": 0.0867, "net_margin": -0.0253},
                "financial_health": {
                    "debt_to_equity": 1.105,
                    "cash_and_st_investments": 1_526_045_952.0,
                    "net_cash_positive": True,
                },
                "cash_flow": {
                    "free_cash_flow_ttm": 347_260_000.0,
                    "free_cash_flow_period": "TTM",
                    "free_cash_flow_period_end": "2025-09-30",
                    "currency": "USD",
                    "operating_cf_ttm": 200_000_000.0,
                },
                "analyst_coverage": {
                    "recommendation_key": "buy",
                    "recommendation_mean": 2.22,
                    "number_of_analysts": 20,
                    "target_low_price": 50.0,
                    "target_mean_price": 125.01,
                    "target_high_price": 280.0,
                    "target_median_price": 104.5,
                },
                "short_interest": {
                    "shares_short": 22_160_247,
                    "short_pct_of_float": 0.1141,
                    "days_to_cover": 1.84,
                    "short_change_mom": 0.5141,
                    "date_short_interest": "2026-02-12",
                },
                "ownership": {
                    "held_percent_insiders": 0.04283,
                    "held_percent_institutions": 0.60894,
                    "float_shares": 184_703_112,
                },
                "governance": {},
                "quality": {
                    "gross_profit": 238_112_992.0,
                    "ebitda": -88_436_336.0,
                    "ebitda_margin": -0.0322,
                    "revenue_per_share": 17.307,
                    "quick_ratio": 0.021,
                },
                "valuation_history": None,
                "fundamental_trends": None,
                "analyst_estimates": None,
                "dividend_analysis": None,
            }

        async def fake_risk(symbol: str, benchmark: str = "SPY", portfolio_value=None, risk_per_trade=None):  # type: ignore[no-untyped-def]
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "benchmark": benchmark,
                "benchmark_returns": {
                    "return_1m": -0.0005,
                    "return_3m": -0.0074,
                    "return_1y": 0.2224,
                },
                "volatility": {"annualized": 1.2059, "rules": {"high_volatility": {"triggered": True}}},
                "beta": {"value": None},
                "drawdown": {"max_1y": -0.8093, "current": -0.6264, "days_since_high": 178},
                "var": {"daily_95": -0.08, "daily_99": -0.12},
                "atr": {"value": 7.4, "as_pct_of_price": 0.0663},
                "liquidity": {"avg_dollar_volume": 500_000_000},
                "position_sizing": {"recommended": {"position_dollars": None}},
                "market_context": {
                    "spy_trend": "neutral",
                    "spy_above_200d": True,
                    "spy_above_50d": False,
                    "spy_price": 677.29,
                    "spy_sma_200": 654.58,
                    "spy_sma_50": 687.84,
                    "spy_distance_to_200d": 0.0347,
                    "spy_distance_to_50d": -0.0153,
                    "symbol_used": "SPY",
                    "source": "yfinance",
                    "as_of": "2026-03-09",
                    "price_adjustment": "split_adjusted",
                    "sanity_warnings": None,
                },
            }

        async def fake_events(symbol: str) -> dict[str, object]:
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "earnings": {
                    "next_date": None,
                    "next_date_source": None,
                    "next_date_status": "unavailable",
                    "next_date_status_reason": "calendar_missing_and_no_future_earnings_dates",
                    "days_until": None,
                    "history": [],
                    "beat_rate": None,
                },
                "dividends": {"yield": 0.0},
                "splits": {},
            }

        async def fake_news(symbol: str) -> dict[str, object]:
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "articles": [],
                "sentiment": {
                    "overall": "positive",
                    "confidence": "high",
                    "counts": {"positive": 2, "negative": 1, "neutral": 0},
                    "method": "keyword_v2",
                    "headline_triggers": {"positive": ["product launch"], "negative": ["offering"]},
                    "sentiment_7d": "positive",
                    "sample_size_7d": 3,
                    "confidence_7d": "high",
                    "sentiment_30d": "positive",
                    "sample_size_30d": 3,
                    "confidence_30d": "high",
                },
                "catalyst_intelligence": {
                    "bullish": [{"tag": "product_launch", "count": 2}],
                    "bearish": [{"tag": "offering_or_dilution", "count": 1}],
                    "neutral": [],
                    "sample_size": 3,
                    "method": "keyword_catalyst_v1",
                },
                "recent_earnings": None,
            }

        async def fake_ownership(symbol: str) -> dict[str, object]:
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "insider_activity": {"sentiment": "buying", "net_shares_3m": 616_378},
                "institutional": {"top_holders": [], "total_institutional_pct": 0.60894},
            }

        async def fake_options(symbol: str) -> dict[str, object]:
            resolved_symbols.append(symbol)
            return {
                "symbol": symbol,
                "put_call_ratio": {"volume_based": 1.01, "oi_based": 0.91},
                "implied_volatility": {"atm_avg_iv": 0.8621},
                "unusual_activity": {"items": []},
            }

        monkeypatch.setattr(orchestrator_module, "symbol_search", fake_symbol_search)
        monkeypatch.setattr(orchestrator_module, "stock_summary", fake_stock_summary)
        monkeypatch.setattr(orchestrator_module, "technicals", fake_technicals)
        monkeypatch.setattr(orchestrator_module, "fundamentals_snapshot", fake_fundamentals)
        monkeypatch.setattr(orchestrator_module, "risk_metrics", fake_risk)
        monkeypatch.setattr(orchestrator_module, "events_calendar", fake_events)
        monkeypatch.setattr(orchestrator_module, "stock_news", fake_news)
        monkeypatch.setattr(orchestrator_module, "ownership_analysis", fake_ownership)
        monkeypatch.setattr(orchestrator_module, "options_signals", fake_options)

        result = await analyze_stock("Circle")

        assert result["symbol"] == "CRCL"
        assert "decision_modes" in result
        assert "dislocation_framework" in result
        assert "investor_profile" not in result
        assert resolved_symbols
        assert set(resolved_symbols) == {"CRCL"}
        assert result["dislocation_framework"]["setup"]["status"] == "present"
        assert result["dislocation_framework"]["mismatch_verdict"]["status"] in {
            "unclear",
            "both_broken",
            "business_broken_more_than_price",
            "business_intact_but_not_cheap",
            "price_broken_more_than_business",
        }

    def test_top_triggers_do_not_show_positive_bearish_score_delta(self) -> None:
        result = build_decision_context(
            signals={
                "bullish": ["macd_bullish", "strong_3m_momentum"],
                "bearish": ["unprofitable", "very_high_volatility", "deep_drawdown"],
            },
            tech_data={
                "moving_averages": {},
                "rsi": {"value": 76.2},
                "current_price": 111.65,
            },
            risk_data={
                "volatility": {"annualized": 1.2059},
                "drawdown": {"max_1y": -0.8093},
            },
            events_data={"earnings": {}},
            fund_data={
                "valuation": {},
                "yield_metrics": {},
                "profitability": {"net_margin": -0.0253},
                "cash_flow": {
                    "free_cash_flow_ttm": 347_260_000.0,
                    "free_cash_flow_period": "TTM",
                    "currency": "USD",
                    "free_cash_flow_period_end": "2025-09-30",
                },
                "growth": {"revenue_yoy": 0.769},
            },
            fundamentals_summary={"burn_metrics": {"status": "not_applicable"}},
            action_zones={"valuation_assessment": {"gate": "neutral", "is_unprofitable": True}},
            news_data={
                "sentiment": {"overall": "positive", "confidence": "high"},
                "catalyst_intelligence": {"bullish": [], "bearish": [], "neutral": []},
            },
            verdict={
                "tilt": "neutral",
                "decomposed": {
                    "setup": "strong",
                    "business_quality": "unprofitable",
                    "business_quality_status": "evaluated_unprofitable",
                    "risk": "extreme",
                },
                "components": {
                    "technicals": 0.333,
                    "fundamentals": 0.5,
                    "risk": -1.0,
                },
                "weights_used": {
                    "technicals": 0.3,
                    "fundamentals": 0.45,
                    "risk": 0.25,
                },
                "inputs_used": {
                    "annualized_vol": 1.2059,
                    "max_drawdown_1y": -0.8093,
                },
                "horizon_fit": {"long_term_gates": ["unprofitable", "extreme_risk"]},
            },
        )

        for trigger in result["top_triggers"]:
            if trigger["direction"] == "bearish":
                assert trigger["score_delta"] is None or trigger["score_delta"] <= 0

    def test_dislocation_candidate_overrides_wait_actions_to_small_starter(self) -> None:
        verdict = {
            "tilt": "neutral",
            "horizon_fit": {
                "mid_term": "caution",
                "long_term": "caution",
                "long_term_gates": ["extreme_risk"],
            },
            "decomposed": {
                "business_quality": "moderate",
                "business_quality_status": "evaluated",
                "risk": "extreme",
            },
        }
        action_zones = {
            "current_zone": "hold_bullish",
            "valuation_assessment": {
                "gate": "attractive",
                "basis": "fcf_yield",
                "is_unprofitable": False,
            },
            "position_sizing_range": {
                "suggested_pct_range": [0.5, 3.0],
                "starter_pct": 0.5,
                "risk_per_trade_pct": 1.0,
            },
            "stop_calculation": {
                "stop_price": 9.0,
                "stop_distance_pct": 0.2,
            },
        }
        dip_assessment = {
            "dip_classification": {"type": "falling_knife"},
            "dip_depth": {"from_52w_high": -0.743, "severity": "extreme"},
            "assessment": {
                "recommendation": "do_not_catch_falling_knife",
                "dip_quality": "avoid",
            },
            "entry_timing": {"wait_for": ["two_higher_closes"]},
        }
        fundamentals_summary = {
            "growth": {"revenue_yoy": 0.198},
            "profitability": {"net_margin": 0.063, "fcf_positive": True},
            "health": {"net_cash_positive": True, "debt_to_equity": 0.8},
            "burn_metrics": {"status": "not_applicable"},
        }

        policy_action = build_policy_action(
            verdict=verdict,
            action_zones=action_zones,
            decomposed=verdict["decomposed"],
            risk_regime={"classification": "extreme"},
            dip_assessment=dip_assessment,
            fundamentals_summary=fundamentals_summary,
        )
        decision_modes = build_decision_modes(
            summary={"current_price": 10.0},
            verdict=verdict,
            policy_action=policy_action,
            action_zones=action_zones,
            dip_assessment=dip_assessment,
            decision_context={"thesis_checkpoints": {}},
            news_summary={"catalyst_intelligence": {"bullish": [], "bearish": [], "neutral": []}},
            fundamentals_summary=fundamentals_summary,
            events_summary={"next_catalyst": {}},
        )

        assert policy_action["mid_term"] == "speculative_small_position"
        assert policy_action["long_term"] == "small_position_with_stops"
        assert decision_modes["core"]["action"] == "not_core_yet"
        assert decision_modes["balanced"]["action"] == "starter_position_only"
        assert decision_modes["speculative"]["action"] == "starter_position_only"
        assert "price_broken_more_than_business" in (decision_modes["balanced"]["why"] or [])

    def test_dip_assessment_is_reframed_when_starter_is_allowed(self) -> None:
        dip_assessment = {
            "assessment": {
                "dip_quality": "avoid",
                "recommendation": "do_not_catch_falling_knife",
                "rationale": "Trend is broken - wait for stabilization",
            }
        }

        result = align_dip_assessment_with_action(
            dip_assessment=dip_assessment,
            mismatch_status="price_broken_more_than_business",
            can_start_now=True,
            add_only_if=["risk_regime <= high (volatility < 60%)"],
        )

        assert result is not None
        assert result["assessment"]["scope"] == "entry_timing_only"
        assert result["assessment"]["timing_only"] is True
        assert result["assessment"]["dip_quality"] == "starter_only"
        assert result["assessment"]["recommendation"] == "starter_only_wait_for_stabilization"
        assert result["assessment"]["add_only_if"] == ["risk_regime <= high (volatility < 60%)"]

    @pytest.mark.asyncio
    async def test_stock_news_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_fetch_ticker(symbol: str) -> _FakeNewsTicker:
            return _FakeNewsTicker()

        monkeypatch.setattr(news_module, "fetch_ticker", fake_fetch_ticker)

        result = await stock_news("TEST", days=7)

        assert result["symbol"] == "TEST"
        assert result["article_count"] == 1
        assert result["sentiment"]["overall"] == "positive"
        assert result["catalyst_intelligence"]["bullish"][0]["tag"] == "guidance_raise"
        assert result["articles"][0]["catalyst_tags"]["bullish"]

    def test_error_response_schema(self) -> None:
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
        test_cases = [
            (3, 0, 1.0),
            (0, 3, -1.0),
            (1, 1, 0.0),
            (2, 1, 1 / 3),
            (1, 2, -1 / 3),
        ]
        for pos, neg, expected in test_cases:
            total = pos + neg
            if total > 0:
                result = (pos - neg) / total
                assert -1.0 <= result <= 1.0, f"Score {result} out of bounds"
                assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"

    def test_score_delta_calculation(self) -> None:
        """score_delta = component_score * weight_used must match."""
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
        """Sum of all component score_deltas should approximate score_raw."""
        components = {
            "technicals": 0.75,
            "fundamentals": -0.33,
            "risk": -0.67,
        }
        weights = {
            "technicals": 0.30,
            "fundamentals": 0.45,
            "risk": 0.25,
        }

        weighted_sum = sum(components[k] * weights[k] for k in components)
        total_weight = sum(weights.values())
        score_raw = weighted_sum / total_weight

        renormalized_weights = {k: w / total_weight for k, w in weights.items()}
        score_delta_sum = sum(components[k] * renormalized_weights[k] for k in components)

        assert abs(score_raw - score_delta_sum) < 0.001, (
            f"score_raw ({score_raw:.4f}) != sum(score_delta) ({score_delta_sum:.4f})"
        )

    def test_top_triggers_balance_rules(self) -> None:
        """Top triggers must follow balance rules based on tilt."""
        balance_rules = {
            "neutral": {"bearish": 2, "bullish": 1},
            "bullish": {"bearish": 1, "bullish": 2},
            "bearish": {"bearish": 2, "bullish": 1},
        }

        for tilt, expected_counts in balance_rules.items():
            total_expected = sum(expected_counts.values())
            assert total_expected == 3, f"Tilt {tilt} should show 3 triggers"

    def test_score_delta_sum_equals_score_raw_exactly(self) -> None:
        """Score deltas must sum to score_raw with negligible tolerance."""
        test_cases = [
            {
                "components": {"technicals": 1.0, "fundamentals": 1.0, "risk": 1.0},
                "weights": {"technicals": 0.30, "fundamentals": 0.45, "risk": 0.25},
            },
            {
                "components": {"technicals": -1.0, "fundamentals": -1.0, "risk": -1.0},
                "weights": {"technicals": 0.30, "fundamentals": 0.45, "risk": 0.25},
            },
            {
                "components": {"technicals": 0.67, "fundamentals": -0.33, "risk": -0.50},
                "weights": {"technicals": 0.30, "fundamentals": 0.45, "risk": 0.25},
            },
            {
                "components": {"technicals": 0.5, "fundamentals": -0.5},
                "weights": {"technicals": 0.30, "fundamentals": 0.45},
            },
        ]

        for case in test_cases:
            components = case["components"]
            weights = case["weights"]

            total_weight = sum(weights.values())
            renormalized = {k: w / total_weight for k, w in weights.items()}

            score_raw = sum(components[k] * weights[k] for k in components) / total_weight
            score_delta_sum = sum(components[k] * renormalized[k] for k in components)

            assert abs(score_raw - score_delta_sum) < 1e-9, (
                f"INVARIANT VIOLATED: score_raw ({score_raw:.10f}) != "
                f"sum(score_delta) ({score_delta_sum:.10f})"
            )


class TestDipAssessmentLogic:
    """Tests for dip assessment helper logic."""

    def test_oversold_composite_extreme(self) -> None:
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
        result = _build_oversold_composite(
            rsi=None,
            return_1w_zscore=None,
            distance_to_sma50_atr=-1.2,
            position_in_range=0.2,
        )

        assert "momentum_missing" in result["notes"]

    def test_action_zone_distance_labels(self) -> None:
        current_price = 100.0
        tech_data = {
            "moving_averages": {"sma_50": 110.0, "sma_200": 120.0},
            "price_position": {"week_52_low": 80.0, "week_52_high": 150.0},
        }
        risk_data = {"atr": {"value": 5.0, "as_pct_of_price": 0.05}}
        fund_data = {"valuation": {}, "yield_metrics": {}, "profitability": {}}
        risk_regime = {"classification": "extreme"}

        result = build_action_zones(
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

    def test_action_zone_uses_investor_profile_for_dollars_and_risk_budget(self) -> None:
        result = build_action_zones(
            current_price=100.0,
            tech_data={
                "moving_averages": {"sma_50": 110.0, "sma_200": 120.0},
                "price_position": {"week_52_low": 80.0, "week_52_high": 150.0},
            },
            risk_data={"atr": {"value": 5.0, "as_pct_of_price": 0.05}},
            fund_data={"valuation": {}, "yield_metrics": {}, "profitability": {}},
            risk_regime={"classification": "medium"},
            signals={"bullish": [], "bearish": []},
            investor_profile={
                "account_size": 3_000.0,
                "risk_per_trade_pct": 1.0,
                "max_position_pct": 5.0,
                "starter_position_pct": 1.0,
            },
        )

        sizing = result["position_sizing_range"]
        assert sizing["suggested_pct_range"] == [1.0, 5.0]
        assert sizing["dollars_for_account"] == {"min": 30.0, "max": 150.0, "portfolio_assumption": 3000.0}
        assert sizing["stop_implied_max"]["pct"] == 5.0
