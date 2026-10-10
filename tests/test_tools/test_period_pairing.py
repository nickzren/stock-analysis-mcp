"""WP6: period pairing and contiguous TTM (R6.1-R6.6)."""

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

fundamentals = importlib.import_module("stock_analysis.tools.fundamentals")
orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")
project_analyze_result = importlib.import_module("stock_analysis.tools.analyze.projection").project_analyze_result

NAN = float("nan")
QUARTERS = ["2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31", "2024-09-30"]
YEARS = ["2025-12-31", "2024-12-31", "2023-12-31"]
# Costco-style 12/12/12/16-week fiscal quarters (the 16-week quarter ends 2025-08-31).
COSTCO_QUARTERS = ["2025-11-23", "2025-08-31", "2025-05-11", "2025-02-16", "2024-11-24", "2024-09-01",
                   "2024-05-12"]


def statement(columns: list[str], **rows: list[float]) -> pd.DataFrame:
    labels = {"fcf": "Free Cash Flow", "ocf": "Operating Cash Flow", "capex": "Capital Expenditure"}
    return pd.DataFrame(list(rows.values()), index=[labels[k] for k in rows],
                        columns=pd.to_datetime(columns))


def resolve(quarterly: pd.DataFrame | None, yearly: pd.DataFrame | None,
            *, ocf: float | None = 5.0, fcf: float | None = 3.0) -> dict[str, Any]:
    return fundamentals._resolve_cashflow(quarterly, yearly, ocf, fcf, "FY 2025")  # type: ignore[no-any-return]


class TestR62CompleteTtm:
    def test_four_consecutive_quarters_sum_and_take_the_newest_label(self) -> None:
        q = statement(QUARTERS, fcf=[10.0, 20.0, 30.0, 40.0, 999.0], ocf=[15.0, 25.0, 35.0, 45.0, 999.0])

        cf = resolve(q, None)

        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_period"], cf["free_cash_flow_period_end"]) == (
            100.0, "TTM", "2025-09-30")
        assert (cf["free_cash_flow_source"], cf["free_cash_flow_ttm_status"], cf["free_cash_flow_period_lag"]) == (
            "cashflow_quarterly", "complete", None)
        assert (cf["operating_cf_ttm"], cf["operating_cf_period"], cf["operating_cf_period_end"]) == (
            120.0, "TTM", "2025-09-30")
        assert cf["operating_cf_ttm_status"] == "complete"

    def test_52_53_week_quarters_are_consecutive(self) -> None:
        q = statement(["2025-06-28", "2025-03-29", "2024-12-28", "2024-09-28"], fcf=[1.0, 2.0, 3.0, 4.0])
        assert resolve(q, None)["free_cash_flow_ttm"] == 10.0

    @pytest.mark.parametrize("newest", range(4), ids=["q4", "q3", "q2", "q1"])
    def test_12_12_12_16_week_calendars_are_consecutive(self, newest: int) -> None:
        q = statement(COSTCO_QUARTERS[newest:newest + 4], fcf=[1.0, 2.0, 3.0, 4.0])
        cf = resolve(q, None)
        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_ttm_status"]) == (10.0, "complete")

    def test_a_17_week_quarter_is_consecutive(self) -> None:
        q = statement(["2026-01-03", "2025-09-06", "2025-06-14", "2025-03-22"], fcf=[1.0, 2.0, 3.0, 4.0])
        assert resolve(q, None)["free_cash_flow_ttm_status"] == "complete"

    def test_monthly_columns_are_not_quarters(self) -> None:
        q = statement(["2025-09-30", "2025-08-31", "2025-07-31", "2025-06-30"], fcf=[1.0, 2.0, 3.0, 4.0])
        assert resolve(q, None)["free_cash_flow_ttm_status"] == "ttm_incomplete"

    def test_skipping_a_12_week_quarter_is_still_a_gap(self) -> None:
        q = statement(["2025-08-31", "2025-02-16", "2024-11-24", "2024-09-01"], fcf=[1.0, 2.0, 3.0, 4.0])
        assert resolve(q, None)["free_cash_flow_ttm_status"] == "ttm_incomplete"


class TestR61IncompleteTtm:
    def test_a_missing_quarter_value_falls_back_to_a_labelled_annual(self) -> None:
        q = statement(QUARTERS, fcf=[10.0, 20.0, NAN, 40.0, 50.0])
        y = statement(YEARS, fcf=[300.0, 200.0, 100.0])

        cf = resolve(q, y)

        assert cf["free_cash_flow_ttm_status"] == "ttm_incomplete"
        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_period"], cf["free_cash_flow_period_end"]) == (
            300.0, "FY 2025", "2025-12-31")
        assert (cf["free_cash_flow_source"], cf["free_cash_flow_period_lag"]) == ("cashflow_yearly", 0)

    def test_an_infinite_quarter_is_incomplete(self) -> None:
        q = statement(QUARTERS[:4], fcf=[10.0, np.inf, 30.0, 40.0])
        assert resolve(q, None)["free_cash_flow_ttm_status"] == "ttm_incomplete"

    def test_without_an_annual_value_the_info_figure_is_used_and_still_flagged(self) -> None:
        q = statement(QUARTERS, fcf=[10.0, 20.0, NAN, 40.0, 50.0])

        cf = resolve(q, None)

        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_period"], cf["free_cash_flow_source"]) == (
            3.0, "TTM", "info")
        assert cf["free_cash_flow_ttm_status"] == "ttm_incomplete"


class TestR63MissingQuarterColumn:
    def test_a_gap_between_columns_is_incomplete(self) -> None:
        q = statement(["2025-09-30", "2025-06-30", "2024-12-31", "2024-09-30"], fcf=[1.0, 2.0, 3.0, 4.0])
        assert resolve(q, None)["free_cash_flow_ttm_status"] == "ttm_incomplete"

    def test_fewer_than_four_quarters_is_incomplete(self) -> None:
        q = statement(QUARTERS[:3], fcf=[1.0, 2.0, 3.0])
        assert resolve(q, None)["free_cash_flow_ttm_status"] == "ttm_incomplete"

    @pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
    def test_undated_columns_cannot_prove_consecutive(self) -> None:
        q = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=["Free Cash Flow"], columns=["a", "b", "c", "d"])
        assert resolve(q, None)["free_cash_flow_ttm_status"] == "ttm_incomplete"


class TestR64AnnualPeriodLag:
    def test_an_older_annual_value_keeps_its_own_period(self) -> None:
        y = statement(YEARS, fcf=[NAN, 100.0, 90.0])

        cf = resolve(None, y)

        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_period"], cf["free_cash_flow_period_end"]) == (
            100.0, "FY 2024", "2024-12-31")
        assert cf["free_cash_flow_period_lag"] == 1
        assert cf["free_cash_flow_ttm_status"] == "unavailable"

    def test_operating_cash_flow_is_labelled_the_same_way(self) -> None:
        y = statement(YEARS, ocf=[NAN, NAN, 70.0])

        cf = resolve(None, y)

        assert (cf["operating_cf_ttm"], cf["operating_cf_period"], cf["operating_cf_period_end"]) == (
            70.0, "FY 2023", "2023-12-31")
        assert (cf["operating_cf_source"], cf["operating_cf_period_lag"]) == ("cashflow_yearly", 2)


class TestR65SamePeriodReconstruction:
    def test_ocf_and_capex_from_different_years_do_not_make_fcf(self) -> None:
        y = statement(YEARS[:2], ocf=[500.0, NAN], capex=[NAN, -200.0])

        cf = resolve(None, y)

        assert cf["free_cash_flow_source"] == "info"
        assert cf["free_cash_flow_ttm"] == 3.0

    def test_same_year_ocf_and_capex_make_fcf_for_that_year(self) -> None:
        y = statement(YEARS, ocf=[NAN, 500.0, 400.0], capex=[NAN, -200.0, -100.0])

        cf = resolve(None, y)

        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_period_end"], cf["free_cash_flow_period_lag"]) == (
            300.0, "2024-12-31", 1)

    def test_the_newest_annual_value_wins_between_the_fcf_row_and_a_reconstruction(self) -> None:
        y = statement(YEARS, fcf=[NAN, 250.0, NAN], ocf=[600.0, 500.0, NAN], capex=[-100.0, -200.0, NAN])
        cf = resolve(None, y)
        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_period"], cf["free_cash_flow_period_end"],
                cf["free_cash_flow_period_lag"]) == (500.0, "FY 2025", "2025-12-31", 0)

    def test_quarterly_reconstruction_needs_both_complete_over_the_same_quarters(self) -> None:
        complete = statement(QUARTERS[:4], ocf=[10.0, 10.0, 10.0, 10.0], capex=[-1.0, -1.0, -1.0, -1.0])
        assert resolve(complete, None)["free_cash_flow_ttm"] == 36.0
        gap = statement(QUARTERS[:4], ocf=[10.0, 10.0, 10.0, 10.0], capex=[-1.0, NAN, -1.0, -1.0])
        cf = resolve(gap, None)
        assert (cf["free_cash_flow_source"], cf["free_cash_flow_ttm_status"]) == ("info", "ttm_incomplete")


class TestNoStatements:
    def test_info_figures_keep_their_ttm_label(self) -> None:
        cf = resolve(None, None, ocf=1.0, fcf=2.0)
        assert (cf["operating_cf_ttm"], cf["operating_cf_period"], cf["operating_cf_source"]) == (1.0, "TTM", "info")
        assert (cf["free_cash_flow_ttm"], cf["free_cash_flow_period"], cf["free_cash_flow_source"]) == (
            2.0, "TTM", "info")
        assert cf["free_cash_flow_ttm_status"] == "unavailable"

    def test_nothing_at_all(self) -> None:
        cf = resolve(None, None, ocf=None, fcf=None)
        assert cf["free_cash_flow_ttm"] is None and cf["free_cash_flow_period"] is None


def income(columns: list[str], net_income: list[float], revenue: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame([net_income, revenue or [1000.0] * len(columns)], index=["Net Income", "Total Revenue"],
                        columns=pd.to_datetime(columns))


class TestR66RollingHistory:
    INFO = {"currency": "USD", "financialCurrency": "USD", "regularMarketPrice": 100.0, "marketCap": 1e8,
            "sharesOutstanding": 1e6}
    PRICES = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2023-01-02", "2025-10-10"))

    def test_a_non_consecutive_window_is_flagged_not_summed(self) -> None:
        cols = ["2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31", "2024-06-30", "2024-03-31", "2023-12-31"]

        history = fundamentals._compute_valuation_history(income(cols, [100.0] * 7), self.PRICES, self.INFO)

        assert [e["quarter"] for e in history["pe_history"]] == ["2025-09-30"]
        assert history["non_consecutive_windows"] == ["2025-06-30", "2025-03-31", "2024-12-31"]

    def test_an_unverified_basis_reports_windows_as_unchecked(self) -> None:
        cols = ["2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31"]
        prices = pd.DataFrame({"Close": [100.0]}, index=pd.DatetimeIndex(["2025-09-30"]))
        info = {**self.INFO, "financialCurrency": "TWD"}

        history = fundamentals._compute_valuation_history(income(cols, [100.0] * 4), prices, info)

        assert (history["status"], history["non_consecutive_windows"]) == ("unverified_currency", None)

    def test_consecutive_history_flags_nothing(self) -> None:
        cols = [d.strftime("%Y-%m-%d") for d in reversed(pd.date_range("2024-03-31", periods=6, freq="QE"))]

        history = fundamentals._compute_valuation_history(income(cols, [100.0] * 6), self.PRICES, self.INFO)

        assert len(history["pe_history"]) == 3
        assert history["non_consecutive_windows"] == []


class TestTrendPairing:
    def test_comparisons_need_an_unbroken_chain_of_quarters(self) -> None:
        cols = ["2025-09-30", "2025-06-30", "2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31"]
        trends = fundamentals._compute_fundamental_trends(income(cols, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                                                                 [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]))

        latest, second, third = trends["quarterly_data"][:3]
        assert latest["net_income_qoq"] == -0.5  # 2025-06-30 is the previous quarter
        assert "net_income_qoq" not in second  # 2024-12-31 is two quarters back
        assert "net_income_yoy" not in latest  # the year back crosses the gap
        assert third["net_income_qoq"] == pytest.approx(-0.25)
        assert trends["margin_trend"] == {"gross": None, "operating": None, "net": None}
        assert trends["quarter_gaps"] == ["2025-06-30"]

    def test_a_12_12_12_16_week_calendar_has_no_gaps(self) -> None:
        trends = fundamentals._compute_fundamental_trends(income(COSTCO_QUARTERS[:5], [20.0, 10.0, 10.0, 10.0, 10.0],
                                                                 [100.0] * 5))
        assert trends["quarter_gaps"] == []
        assert trends["quarterly_data"][0]["net_income_yoy"] == 1.0
        assert trends["margin_trend"]["net"] == "expanding"

    def test_an_unbroken_chain_compares_as_before(self) -> None:
        cols = [d.strftime("%Y-%m-%d") for d in reversed(pd.date_range("2024-06-30", periods=5, freq="QE"))]
        trends = fundamentals._compute_fundamental_trends(income(cols, [20.0, 10.0, 10.0, 10.0, 10.0],
                                                                 [100.0] * 5))

        assert trends["quarterly_data"][0]["net_income_yoy"] == 1.0
        assert trends["margin_trend"]["net"] == "expanding"
        assert trends["quarter_gaps"] == []


class TestAnalyzeDisclosure:
    CASH_FLOW = {"free_cash_flow_ttm": 100.0, "free_cash_flow_period": "FY 2024",
                 "free_cash_flow_period_end": "2024-12-31", "free_cash_flow_source": "cashflow_yearly",
                 "free_cash_flow_ttm_status": "ttm_incomplete", "free_cash_flow_period_lag": 1, "currency": "USD"}

    async def test_analyze_keeps_the_ttm_status_and_lag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_fundamentals(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "valuation": {}, "cash_flow": self.CASH_FLOW}

        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "current_price": 100.0, "currency": "USD", "quote_type": "EQUITY"}

        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "fundamentals_snapshot", fake_fundamentals)
        monkeypatch.setattr(orchestrator, "stock_summary", summary)
        for name in ("technicals", "risk_metrics", "events_calendar", "stock_news", "ownership_analysis",
                     "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty)
        result = await orchestrator.analyze_stock("ACME")

        for detail in ("standard", "full"):
            cash_flow = project_analyze_result(result, detail)["fundamentals_summary"]["cash_flow"]
            assert (cash_flow["free_cash_flow_ttm_status"], cash_flow["free_cash_flow_period_lag"]) == (
                "ttm_incomplete", 1), detail

    def test_the_fundamentals_summary_carries_the_ttm_status_and_lag(self) -> None:
        summary = orchestrator._fundamentals_cash_flow_summary(self.CASH_FLOW)
        assert (summary["free_cash_flow_ttm_status"], summary["free_cash_flow_period_lag"]) == ("ttm_incomplete", 1)
        assert summary["free_cash_flow_label"] == "FCF (FY 2024): +$100 (end 2024-12-31)"
