"""Step 1 WP4: units, domain, basis and support at the producers (R4.1-R4.11)."""

import importlib
from typing import Any

import pandas as pd
import pytest

import stock_analysis.tools.events as events
import stock_analysis.tools.fundamentals as fundamentals
from stock_analysis.utils.valuation_basis import classify_support, dividend_yields, usable_metric
from tests.support.fake_source import FakeSource
from tests.support.fake_ticker import FakeTicker

# tools/__init__ re-exports a function named like this module, shadowing it.
stock_summary = importlib.import_module("stock_analysis.tools.stock_summary")

USD = {"currency": "USD", "financialCurrency": "USD"}


def valuation(**info: Any) -> dict[str, Any]:
    return fundamentals._build_valuation({**USD, "regularMarketPrice": 100.0, **info})


def with_cache_state(fetch: Any) -> Any:
    async def fetch_with_cache_state(symbol: str) -> tuple[dict[str, Any], str, bool]:
        return await fetch(symbol), "2026-10-09T12:00:00Z", False

    return fetch_with_cache_state


def status(section: dict[str, Any], key: str) -> tuple[Any, Any, Any]:
    entry = section["metric_status"][key]
    return entry["basis_status"], entry["scoring_applicable"], entry["reason"]


class TestDebtToEquity:
    @pytest.mark.parametrize(
        ("percent", "ratio"), [(0, 0.0), (5, 0.05), (9.99, 0.0999), (10, 0.10), (10.5, 0.105), (150, 1.5)]
    )
    def test_r4_1_always_percent_to_ratio(self, percent: float, ratio: float) -> None:
        health = fundamentals._build_financial_health({"debtToEquity": percent})

        assert health["debt_to_equity"] == pytest.approx(ratio)
        assert health["leverage_status"] == "reported"

    @pytest.mark.parametrize(
        "info",
        [{"priceToBook": -2.0}, {"debtToEquity": -40.0, "priceToBook": 3.0}],
        ids=["de_missing_negative_pb", "negative_de"],
    )
    def test_r4_2_negative_equity_is_never_low_debt(self, info: dict[str, Any]) -> None:
        health = fundamentals._build_financial_health(info)

        assert health["leverage_status"] == "negative_equity"
        assert health["rules"]["low_debt"]["triggered"] is False


class TestPriceEarnings:
    def test_r4_10_forward_pe_survives_negative_trailing_eps(self) -> None:
        section = valuation(trailingPE=None, trailingEps=-2.0, forwardPE=30.0, forwardEps=3.0)

        assert (section["pe_trailing"], section["pe_forward"]) == (None, 30.0)
        assert status(section, "pe_trailing") == ("verified", False, "undefined_eps_nonpositive")
        assert status(section, "pe_forward") == ("verified", True, None)

    def test_forward_pe_is_undefined_on_its_own_negative_eps(self) -> None:
        section = valuation(trailingPE=20.0, trailingEps=5.0, forwardPE=-8.0, forwardEps=-1.74)

        assert (section["pe_trailing"], section["pe_forward"]) == (20.0, None)
        assert status(section, "pe_forward")[2] == "undefined_eps_nonpositive"

    def test_pe_without_its_eps_is_unverified(self) -> None:
        section = valuation(trailingPE=20.0)

        assert section["pe_trailing"] is None
        assert status(section, "pe_trailing") == ("unverified", False, "eps_basis_unknown")


class TestPeg:
    def test_r4_4_nonpositive_peg_is_kept_but_never_scored(self) -> None:
        section = valuation(pegRatio=-1.2)

        assert section["peg_ratio"] == -1.2
        assert status(section, "peg_ratio") == ("provider", False, "peg_nonpositive")
        assert usable_metric(section, "peg_ratio") is None

    def test_r4_11_provider_peg_ignores_unrelated_growth(self) -> None:
        section = valuation(pegRatio=1.5, earningsGrowth=-0.3)

        assert usable_metric(section, "peg_ratio") == 1.5
        assert status(section, "peg_ratio") == ("provider", True, None)


class TestEnterpriseValue:
    def test_r4_5_negative_ev_keeps_ev_sales_unscored(self) -> None:
        section = valuation(enterpriseValue=-50.0, totalRevenue=100.0)

        assert (section["enterprise_value"], section["ev_to_sales"]) == (-50.0, -0.5)
        assert status(section, "ev_to_sales") == ("verified", False, "ev_nonpositive")
        assert usable_metric(section, "ev_to_sales") is None

    def test_r4_5_ebitda_nonpositive_makes_ev_ebitda_undefined(self) -> None:
        section = valuation(enterpriseToEbitda=-7.7, ebitda=-2.7e9, enterpriseValue=2e10)

        assert section["ev_to_ebitda"] is None
        assert status(section, "ev_to_ebitda")[2] == "undefined_ebitda_nonpositive"

    def test_ev_ebitda_with_negative_ev_is_kept_unscored(self) -> None:
        section = valuation(enterpriseToEbitda=-0.5, ebitda=100.0, enterpriseValue=-50.0)

        assert section["ev_to_ebitda"] == -0.5
        assert status(section, "ev_to_ebitda") == ("verified", False, "ev_nonpositive")


class TestCurrencyBasis:
    TSM = {
        "currency": "USD",
        "financialCurrency": "TWD",
        "regularMarketPrice": 452.185,
        "trailingPE": 32.86,
        "trailingEps": 13.76,
        "priceToSalesTrailing12Months": 0.528,
        "priceToBook": 93.09,
        "enterpriseValue": 1.718e13,
        "totalRevenue": 4.44e12,
        "enterpriseToEbitda": 5.425,
        "ebitda": 3.2e12,
        "marketCap": 2.345e12,
        "sharesOutstanding": 5.186e9,
    }

    def test_r4_6_cross_currency_ratios_are_null_with_their_basis(self) -> None:
        section = fundamentals._build_valuation(self.TSM)

        for key in ("ps_trailing", "pb_ratio", "ev_to_sales", "ev_to_ebitda", "pe_trailing"):
            assert section[key] is None, key
            assert status(section, key)[0] == "unverified_currency", key

    def test_minor_unit_price_is_not_its_major_currency(self) -> None:
        section = fundamentals._build_valuation({**self.TSM, "currency": "GBp", "financialCurrency": "GBP"})

        assert section["ps_trailing"] is None
        assert status(section, "ps_trailing")[0] == "unverified_currency"

    def test_pb_needs_consistent_share_units(self) -> None:
        adr_like = {**USD, "regularMarketPrice": 100.0, "priceToBook": 3.0,
                    "marketCap": 1e9, "sharesOutstanding": 5e7}  # price × shares = 5e9 ≠ 1e9

        section = fundamentals._build_valuation(adr_like)

        assert section["pb_ratio"] is None
        assert status(section, "pb_ratio")[0] == "unverified_share_units"

    def test_r4_2_negative_pb_is_kept_unscored(self) -> None:
        section = valuation(priceToBook=-2.0, marketCap=1e10, sharesOutstanding=1e8)

        assert section["pb_ratio"] == -2.0
        assert status(section, "pb_ratio") == ("verified", False, "pb_nonpositive")


class TestDividendYield:
    INFO = {**USD, "regularMarketPrice": 100.0, "currentPrice": 100.0, "dividendRate": 0.5,
            "trailingAnnualDividendRate": 0.48, "dividendYield": 0.5, "quoteType": "EQUITY",
            "marketCap": 1e10, "sharesOutstanding": 1e8, "totalRevenue": 1e9}

    async def test_r4_3_one_sub_one_percent_yield_on_all_three_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_info(symbol: str) -> dict[str, Any]:
            return dict(self.INFO)

        async def fake_ticker(symbol: str) -> FakeTicker:
            return FakeTicker(symbol, splits=None)

        monkeypatch.setattr(stock_summary, "fetch_info_with_cache_state", with_cache_state(fake_info))
        monkeypatch.setattr(events, "fetch_info", fake_info)
        monkeypatch.setattr(events, "fetch_ticker", fake_ticker)
        monkeypatch.setattr(events, "read_calendar_events", FakeSource.empty("calendar", {}))
        monkeypatch.setattr(events, "read_earnings_page", FakeSource.empty("earnings_page", []))

        summary = await stock_summary.stock_summary("EXMP")
        dividends = (await events.events_calendar("EXMP"))["dividends"]
        yields = fundamentals._build_yield_metrics(
            self.INFO, fundamentals._build_valuation(self.INFO), None, 1e10, 100.0
        )

        assert summary["dividend_yield"] == dividends["yield"] == yields["dividend_yield"] == 0.005
        assert summary["dividend_yield_basis"] == dividends["yield_basis"] == yields["dividend_yield_basis"] == "forward"
        assert summary["dividend_yield_trailing"] == dividends["yield_trailing"] == 0.0048

    def test_non_payer_is_zero_with_its_basis(self) -> None:
        info = {**USD, "trailingAnnualDividendRate": 0.0, "dividendYield": 0.0}

        yields = fundamentals._build_yield_metrics(info, valuation(), None, None, 100.0)

        assert (yields["dividend_yield"], yields["dividend_yield_basis"]) == (0.0, "no_dividend")

    def test_no_dividend_data_is_null(self) -> None:
        yields = fundamentals._build_yield_metrics(dict(USD), valuation(), None, None, 100.0)

        assert (yields["dividend_yield"], yields["dividend_yield_basis"]) == (None, None)

    def test_trailing_yield_needs_one_currency(self) -> None:
        tsm = {**TestCurrencyBasis.TSM, "dividendRate": 3.77, "trailingAnnualDividendRate": 26.0}

        yields = fundamentals._build_yield_metrics(tsm, fundamentals._build_valuation(tsm), None, None, 452.185)

        assert yields["dividend_yield"] == pytest.approx(0.0083, abs=1e-4)  # forward: USD per ADR
        assert yields["dividend_yield_trailing"] is None  # TWD over a USD price
        assert yields["metric_status"]["dividend_yield_trailing"]["basis_status"] == "unverified_currency"


class TestBasisRules:
    def test_pe_absent_with_positive_eps_is_absent_not_unverified(self) -> None:
        section = valuation(trailingEps=2.0)

        assert section["pe_trailing"] is None
        assert "pe_trailing" not in section["metric_status"]

    def test_valuation_history_needs_one_currency(self) -> None:
        income = pd.DataFrame({pd.Timestamp("2026-06-30"): [1.0]}, index=["Net Income"])
        prices = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2026-06-30"]))

        history = fundamentals._compute_valuation_history(income, prices, dict(TestCurrencyBasis.TSM))

        assert (history["status"], history["pe_history"]) == ("unverified_currency", None)

    def test_trailing_dividend_and_fcf_payout_need_share_units(self) -> None:
        multi_class = {**USD, "regularMarketPrice": 100.0, "marketCap": 1e6, "sharesOutstanding": 5000.0,
                       "dividendRate": 2.0, "trailingAnnualDividendRate": 2.0, "trailingEps": 8.0, "trailingPE": 12.5}

        yields = fundamentals._build_yield_metrics(
            multi_class, fundamentals._build_valuation(multi_class), 40_000.0, 1e6, 100.0
        )

        assert yields["dividend_yield"] == 0.02  # forward: per share over price, no share count
        assert (yields["dividend_yield_trailing"], yields["fcf_payout_ratio"]) == (None, None)
        assert yields["metric_status"]["dividend_yield_trailing"]["basis_status"] == "unverified_share_units"
        assert yields["metric_status"]["fcf_payout_ratio"]["basis_status"] == "unverified_share_units"

    @pytest.mark.parametrize(
        ("info", "price", "expected"),
        [({"currency": "GBp", "dividendRate": 5.0}, 2500.0, (None, "unverified_currency")),
         ({"currency": None, "dividendRate": 5.0}, 2500.0, (None, "unverified_currency")),
         ({"currency": "USD", "dividendRate": 1.0}, None, (None, None))],
        ids=["minor_unit", "unknown_currency", "missing_price"],
    )
    def test_forward_yield_basis(self, info: dict[str, Any], price: Any, expected: tuple[Any, Any]) -> None:
        result = dividend_yields(info, price)

        assert (result["yield"], result["basis"]) == expected

    def test_dividend_safety_skips_cross_currency_fcf_coverage(self) -> None:
        tsm = {**TestCurrencyBasis.TSM, "freeCashflow": 7.3e11, "dividendRate": 3.77, "payoutRatio": 0.7}
        dividends = pd.Series([1.0, 1.0, 1.0, 1.0, 1.1],
                              index=pd.DatetimeIndex(["2021-06-01", "2022-06-01", "2023-06-01", "2024-06-01",
                                                      "2025-06-01"]))

        analysis = fundamentals._analyze_dividend_history(dividends, tsm)

        # Payout band only: the TWD-FCF / USD-dividend coverage (about 37x, score 100) is skipped.
        assert analysis["safety_score"] == fundamentals._payout_ratio_score(0.7) < 100

    def test_r4_9_sufficient_history_under_5y_leaves_5y_fields_null(self) -> None:
        quarters = pd.date_range("2023-03-31", periods=12, freq="QE")
        income = pd.DataFrame([[100.0] * 12, [1000.0] * 12], index=["Net Income", "Total Revenue"],
                              columns=list(reversed(quarters)))
        days = pd.bdate_range("2023-01-02", "2026-01-10")
        prices = pd.DataFrame({"Close": [50.0 + i * 0.05 for i in range(len(days))]}, index=days)
        info = {**USD, "regularMarketPrice": 100.0, "marketCap": 1e8, "sharesOutstanding": 1e6,
                "trailingPE": 250.0, "trailingEps": 0.4}

        section = fundamentals._build_valuation(info)
        history = fundamentals._compute_valuation_history(income, prices, info, section)

        assert history["status"] == "available" and history["history_points"] >= 8
        assert history["history_span_years"] < 5
        assert history["pe_stats"]["current_percentile"] is not None  # current P/E ranked against history

    async def test_p3_each_metric_needs_its_own_five_year_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Six years of P/S but only ~2.5 years of positive earnings: no pe_5y_avg."""
        quarters = pd.date_range("2019-03-31", periods=28, freq="QE")
        net_income = [-50.0] * 16 + [100.0] * 12
        income = pd.DataFrame([list(reversed(net_income)), [1000.0] * 28], index=["Net Income", "Total Revenue"],
                              columns=list(reversed(quarters)))
        days = pd.bdate_range("2018-06-01", "2026-01-10")
        prices = pd.DataFrame({"Close": [50.0 + i * 0.02 for i in range(len(days))]}, index=days)
        info = {**USD, "regularMarketPrice": 100.0, "marketCap": 1e8, "sharesOutstanding": 1e6,
                "trailingPE": 250.0, "trailingEps": 0.4, "priceToSalesTrailing12Months": 25.0}

        async def fake_info(symbol: str) -> dict[str, Any]:
            return dict(info)

        async def fake_ticker(symbol: str) -> FakeTicker:
            return FakeTicker(symbol)

        async def statements(ticker: Any) -> dict[str, Any]:
            return {"quarterly_income_stmt": income, "price_history_5y": prices}

        monkeypatch.setattr(fundamentals, "fetch_info", fake_info)
        monkeypatch.setattr(fundamentals, "fetch_ticker", fake_ticker)
        monkeypatch.setattr(fundamentals, "_read_statements", statements)

        snapshot = await fundamentals.fundamentals_snapshot("EXMP")
        context, history = snapshot["valuation_context"], snapshot["valuation_history"]

        assert history["ps_stats"]["span_years"] >= 5 > history["pe_stats"]["span_years"]
        assert context["ps_5y_avg"] is not None
        assert (context["pe_5y_avg"], context["pe_percentile_5y"]) == (None, None)

    async def test_r4_9_valuation_context_names_the_short_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        quarters = pd.date_range("2023-03-31", periods=12, freq="QE")
        income = pd.DataFrame([[100.0] * 12, [1000.0] * 12], index=["Net Income", "Total Revenue"],
                              columns=list(reversed(quarters)))
        days = pd.bdate_range("2023-01-02", "2026-01-10")
        prices = pd.DataFrame({"Close": [50.0 + i * 0.05 for i in range(len(days))]}, index=days)
        info = {**USD, "regularMarketPrice": 100.0, "marketCap": 1e8, "sharesOutstanding": 1e6,
                "trailingPE": 250.0, "trailingEps": 0.4}

        async def fake_info(symbol: str) -> dict[str, Any]:
            return dict(info)

        async def fake_ticker(symbol: str) -> FakeTicker:
            return FakeTicker(symbol)

        async def statements(ticker: Any) -> dict[str, Any]:
            return {"quarterly_income_stmt": income, "price_history_5y": prices}

        monkeypatch.setattr(fundamentals, "fetch_info", fake_info)
        monkeypatch.setattr(fundamentals, "fetch_ticker", fake_ticker)
        monkeypatch.setattr(fundamentals, "_read_statements", statements)

        context = (await fundamentals.fundamentals_snapshot("EXMP"))["valuation_context"]

        assert (context["pe_5y_avg"], context["pe_percentile_5y"], context["ps_5y_avg"]) == (None, None, None)
        assert context["status_reason"] == "history_span_under_5y"


class TestSupport:
    @pytest.mark.parametrize(
        ("quote_type", "currency", "symbol", "name", "instrument", "currency_status"),
        [
            ("EQUITY", "USD", "MSFT", "Microsoft", "supported", "supported"),
            ("EQUITY", "USD", "TSM", "Taiwan Semiconductor ADR", "supported", "supported"),
            ("ETF", "USD", "SPY", "SPDR S&P 500", "unsupported", "supported"),
            ("EQUITY", "GBp", "VOD.L", "Vodafone", "supported", "unsupported"),
            ("EQUITY", "USD", "BAC-PL", "Bank of America", "unsupported", "supported"),
            ("EQUITY", "USD", "XYZ", "XYZ 6.5% Pfd", "unsupported", "supported"),
            ("EQUITY", "USD", "BRK-B", "Berkshire Hathaway", "supported", "supported"),
            ("EQUITY", "USD", "PFBC", "Preferred Bank", "supported", "supported"),
            ("EQUITY", "USD", "TSM", "Taiwan Semiconductor Manufacturing Company Limited American Depositary Shares",
             "supported", "supported"),
            ("EQUITY", "USD", "XYZ", "XYZ Corp 6.50% Non-Cumulative Preferred Stock, Series A",
             "unsupported", "supported"),
            (None, None, "EXMP", None, "unverified", "unverified"),
        ],
    )
    def test_classify(
        self, quote_type: Any, currency: Any, symbol: str, name: Any, instrument: str, currency_status: str
    ) -> None:
        support = classify_support(quote_type=quote_type, currency=currency, symbol=symbol, name=name)

        assert (support["instrument"]["status"], support["currency"]["status"]) == (instrument, currency_status)

    async def test_stock_summary_no_longer_defaults_to_usd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_info(symbol: str) -> dict[str, Any]:
            return {"regularMarketPrice": 10.0, "quoteType": "EQUITY"}

        monkeypatch.setattr(stock_summary, "fetch_info_with_cache_state", with_cache_state(fake_info))

        summary = await stock_summary.stock_summary("EXMP")

        assert summary["currency"] is None
        assert summary["support"]["currency"]["status"] == "unverified"
