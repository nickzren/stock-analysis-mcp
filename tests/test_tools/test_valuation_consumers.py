"""Step 1 WP4: consumers honor applicability (R4.2, R4.4, R4.5, R4.6, R4.12).

analyze runs the real fundamentals_snapshot on synthetic info, then every detail
tier is swept for the gate, pros/cons, prose and conditions.
"""

import importlib
import json
from typing import Any

import pytest

import stock_analysis.tools.fundamentals as fundamentals
from stock_analysis.tools.analyze.action_zones import _build_valuation_assessment
from stock_analysis.tools.analyze.projection import project_analyze_result

orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
compare = importlib.import_module("stock_analysis.tools.compare")

SYMBOL = "EXMP"
BASE_INFO: dict[str, Any] = {
    "quoteType": "EQUITY",
    "currency": "USD",
    "financialCurrency": "USD",
    "regularMarketPrice": 100.0,
    "currentPrice": 100.0,
    "marketCap": 1e10,
    "sharesOutstanding": 1e8,
    "totalRevenue": 1e9,
    "grossMargins": 0.5,
    "revenueGrowth": 0.1,
}
UNPROFITABLE_NEGATIVE_EV = {
    **BASE_INFO, "profitMargins": -0.2, "trailingEps": -1.0, "enterpriseValue": -50.0, "totalRevenue": 100.0,
    "freeCashflow": -1e7, "operatingCashflow": -5e6, "marketCap": None,  # no P/S to fall back on
}
# P/E 40 (earnings yield 2.5%): only the PEG could make these gates attractive.
NEGATIVE_PEG = {**BASE_INFO, "profitMargins": 0.15, "trailingEps": 2.5, "trailingPE": 40.0, "pegRatio": -1.2}
NEGATIVE_EQUITY = {**BASE_INFO, "profitMargins": 0.15, "trailingEps": 2.5, "trailingPE": 40.0, "priceToBook": -2.0}
TSM_LIKE = {**BASE_INFO, "financialCurrency": "TWD", "profitMargins": 0.4, "trailingEps": 13.76,
            "trailingPE": 32.86, "priceToSalesTrailing12Months": 0.53, "priceToBook": 93.0}

FORBIDDEN = ("reasonable for growth", "PEG ratio suggests undervaluation", "Low debt", "suggests undervaluation")


def fund_data(info: dict[str, Any]) -> dict[str, Any]:
    valuation = fundamentals._build_valuation(info)
    return {
        "valuation": valuation,
        "yield_metrics": fundamentals._build_yield_metrics(info, valuation, None, info.get("marketCap"), 100.0),
        "profitability": fundamentals._build_profitability(info),
        "financial_health": fundamentals._build_financial_health(info),
    }


@pytest.fixture
def analyze(monkeypatch: pytest.MonkeyPatch) -> Any:
    async def run(info: dict[str, Any]) -> dict[str, Any]:
        async def fake_info(symbol: str) -> dict[str, Any]:
            return dict(info)

        async def no_ticker(symbol: str) -> Any:
            raise ConnectionError("no ticker in this test")

        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "name": "Example", "current_price": 100.0, "currency": "USD",
                    "quote_type": "EQUITY", "market_cap": 1e10}

        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(fundamentals, "fetch_info", fake_info)
        monkeypatch.setattr(fundamentals, "fetch_ticker", no_ticker)
        monkeypatch.setattr(orchestrator, "stock_summary", summary)
        for name in ("technicals", "risk_metrics", "events_calendar", "stock_news",
                     "ownership_analysis", "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty)
        return await orchestrator.analyze_stock(SYMBOL)

    return run


class TestValuationGate:
    def test_r4_5_negative_ev_is_never_an_attractive_sales_multiple(self) -> None:
        gate, _ = _build_valuation_assessment(fund_data(UNPROFITABLE_NEGATIVE_EV), {"bearish": ["unprofitable"]})

        assert gate["is_unprofitable"] is True
        assert gate["gate"] == "unknown"
        assert gate["reasons"] == ["ev_to_sales_ev_nonpositive"]

    def test_r4_5_plan_example_is_not_attractive_through_ps(self) -> None:
        """Cap 100 + debt 0 - cash 150 = EV -50 on sales 100: P/S 1.0 must not stand in."""
        info = {**UNPROFITABLE_NEGATIVE_EV, "marketCap": 100.0}

        gate, _ = _build_valuation_assessment(fund_data(info), {"bearish": ["unprofitable"]})

        assert gate["gate"] == "unknown"
        assert gate["reasons"] == ["ev_to_sales_ev_nonpositive"]

    @pytest.mark.parametrize("revenue", [None, 0.0], ids=["revenue_missing", "revenue_zero"])
    def test_r4_5_negative_ev_rules_out_ps_without_revenue(self, revenue: float | None) -> None:
        """No revenue means no EV/S, but EV <= 0 still bars P/S from standing in."""
        info = {**UNPROFITABLE_NEGATIVE_EV, "totalRevenue": revenue, "priceToSalesTrailing12Months": 0.5}

        gate, _ = _build_valuation_assessment(fund_data(info), {"bearish": ["unprofitable"]})

        assert gate["gate"] == "unknown"
        assert gate["reasons"] == ["ev_to_sales_ev_nonpositive"]

    def test_ps_is_the_fallback_when_ev_sales_is_absent(self) -> None:
        info = {**UNPROFITABLE_NEGATIVE_EV, "enterpriseValue": None, "priceToSalesTrailing12Months": 12.0}

        gate, _ = _build_valuation_assessment(fund_data(info), {"bearish": ["unprofitable"]})

        assert (gate["gate"], gate["basis"]) == ("headwind", "ps_trailing")

    def test_all_unverified_ratios_name_their_basis(self) -> None:
        info = {**TSM_LIKE, "freeCashflow": 7e11}

        gate, _ = _build_valuation_assessment(fund_data(info), {})

        assert gate["gate"] == "unknown"
        assert gate["reasons"] == ["pe_trailing_unverified_currency", "ps_trailing_unverified_currency"]

    def test_r4_4_negative_peg_is_never_attractive(self) -> None:
        gate, _ = _build_valuation_assessment(fund_data(NEGATIVE_PEG), {})

        assert gate["gate"] != "attractive"
        assert "PEG" not in " ".join(gate["reasons"] or [])

    def test_r4_6_unverified_pe_does_not_flip_a_profitable_filer_to_unprofitable(self) -> None:
        info = {**TSM_LIKE, "freeCashflow": -1e9}

        gate, _ = _build_valuation_assessment(fund_data(info), {"bearish": ["negative_free_cash_flow"]})

        assert gate["is_unprofitable"] is False


class TestConditionsFollowTheGate:
    async def test_r4_5_negative_ev_conditions_name_only_what_the_gate_lacks(self, analyze: Any) -> None:
        result = await analyze({**UNPROFITABLE_NEGATIVE_EV, "marketCap": 100.0})

        assert result["action_zones"]["valuation_assessment"]["gate"] == "unknown"
        upgrades = result["diagnostics"]["pre_gate_policy"]["conditions_to_upgrade"]
        assert "positive enterprise value" in upgrades
        assert not any("P/S" in condition for condition in upgrades)
        valuation = result["decision_context"]["valuation"]
        conditions = (valuation["bullish_if"] or []) + (valuation["bearish_if"] or [])
        assert not [c for c in conditions if "ps_" in json.dumps(c)]

    def test_rerate_milestone_names_the_metric_that_made_the_gate_attractive(self) -> None:
        from stock_analysis.tools.analyze.decision_context import _build_thesis_checkpoints

        thesis = _build_thesis_checkpoints(
            is_unprofitable=False, net_margin=0.4, fcf=None, cash_runway_quarters=None, revenue_yoy=0.2,
            valuation_gate="attractive", risk_label="medium", business_quality="strong", valuation_basis="peg_ratio",
        )

        milestone = next(c for c in thesis["checkpoints"] if c["id"] == "valuation_rerates")
        assert milestone["target"] == "PEG rises toward sector median"
        assert "P/E" not in milestone["target"] and "P/S" not in milestone["target"]


class TestAnalyzeTiers:
    @pytest.mark.parametrize(
        "info", [UNPROFITABLE_NEGATIVE_EV, NEGATIVE_PEG, NEGATIVE_EQUITY], ids=["r4_5", "r4_4", "r4_2"]
    )
    async def test_r4_12_no_tier_reads_an_inapplicable_value_as_good(self, analyze: Any, info: dict[str, Any]) -> None:
        result = await analyze(info)

        assert result["action_zones"]["valuation_assessment"]["gate"] != "attractive"
        for detail in ("decision", "standard", "full"):
            text = json.dumps(project_analyze_result(result, detail))
            for phrase in FORBIDDEN:
                assert phrase not in text, (detail, phrase)

    async def test_r4_2_negative_equity_fails_the_balance_sheet_debt_check(self, analyze: Any) -> None:
        result = await analyze(NEGATIVE_EQUITY)

        assert result["fundamentals_summary"]["health"]["leverage_status"] == "negative_equity"
        balance_sheet = result["dislocation_framework"]["balance_sheet_safety"]
        assert balance_sheet["checks"]["debt_load"] == "fail"
        assert balance_sheet["status"] != "safe"

    async def test_r4_6_unverified_ratios_are_named_not_called_not_meaningful(self, analyze: Any) -> None:
        result = await analyze(TSM_LIKE)

        valuation = result["fundamentals_summary"]["valuation"]
        assert valuation["pe_trailing"] is None
        assert valuation["metric_status"]["pe_trailing"]["basis_status"] == "unverified_currency"
        assert result["data_quality"]["fundamentals_status_reason"] == "pe_unverified_currency"
        assert "pe_not_meaningful" not in result["data_quality"]["data_gaps"]


class TestCompare:
    @pytest.mark.parametrize(
        ("metric", "best_info", "bad_info"),
        [
            ("peg_ratio", {**NEGATIVE_PEG, "pegRatio": 1.5}, NEGATIVE_PEG),
            ("debt_to_equity", {**NEGATIVE_EQUITY, "debtToEquity": 80.0, "priceToBook": 3.0}, NEGATIVE_EQUITY),
            ("debt_to_equity", {**NEGATIVE_EQUITY, "debtToEquity": 80.0, "priceToBook": 3.0},
             {**NEGATIVE_EQUITY, "debtToEquity": -40.0, "priceToBook": 3.0}),
        ],
        ids=["negative_peg", "de_missing_negative_pb", "negative_de"],
    )
    def test_r4_2_r4_4_an_inapplicable_value_is_never_ranked_best(
        self, metric: str, best_info: dict[str, Any], bad_info: dict[str, Any]
    ) -> None:
        good = compare._extract_metrics(fund_data(best_info) | {"growth": {}}, {})
        bad = compare._extract_metrics(fund_data(bad_info) | {"growth": {}}, {})

        ranked = compare._rank_metric(
            [{"symbol": "GOOD", "value": good[metric]}, {"symbol": "BAD", "value": bad[metric]}], metric
        )

        assert ranked[0]["symbol"] == "GOOD" and bad[metric] is None

    def test_r4_3_compare_reads_the_fractional_forward_yield(self) -> None:
        info = {**NEGATIVE_PEG, "dividendRate": 0.5}

        metrics = compare._extract_metrics(fund_data(info) | {"growth": {}}, {})

        assert metrics["dividend_yield"] == 0.005
