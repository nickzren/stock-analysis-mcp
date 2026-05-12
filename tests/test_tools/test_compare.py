"""Tests for compare_stocks metric schema and composite ranking."""

import pytest

from stock_analysis.tools import compare as compare_module
from stock_analysis.tools.compare import compare_stocks


def _fund_payload(*, pe: float, ps: float, eps_yoy: float, net_margin: float) -> dict:
    """Minimal fundamentals snapshot shape used by _extract_metrics."""
    return {
        "valuation": {"pe_trailing": pe, "ps_trailing": ps, "peg_ratio": None},
        "growth": {"revenue_yoy": None, "eps_yoy": eps_yoy},
        "profitability": {"net_margin": net_margin, "gross_margin": None, "roe": None},
        "financial_health": {"debt_to_equity": None},
        "yield_metrics": {"dividend_yield": None, "fcf_yield": None},
    }


def _tech_payload(*, rsi_value: float | None, return_1m: float | None) -> dict:
    return {
        "price_position": {"position_in_range": None},
        "returns": {"return_1m": return_1m, "return_3m": None, "return_1y": None},
        "rsi": {"value": rsi_value},
    }


def _summary_payload(*, name: str, sector: str = "Tech", market_cap: int = 1_000_000_000) -> dict:
    return {"name": name, "sector": sector, "market_cap": market_cap}


def _install_compare_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    summaries: dict[str, dict],
    funds: dict[str, dict],
    techs: dict[str, dict],
) -> None:
    """Patch the compare module's tool entry points to return pre-built payloads."""
    async def fake_stock_summary(symbol: str) -> dict:
        return summaries.get(symbol, {"error": True, "message": f"{symbol} not found"})

    async def fake_fundamentals_snapshot(symbol: str) -> dict:
        return funds.get(symbol, {"error": True, "message": f"{symbol} not found"})

    async def fake_technicals(symbol: str) -> dict:
        return techs.get(symbol, {"error": True, "message": f"{symbol} not found"})

    monkeypatch.setattr(compare_module, "stock_summary", fake_stock_summary)
    monkeypatch.setattr(compare_module, "fundamentals_snapshot", fake_fundamentals_snapshot)
    monkeypatch.setattr(compare_module, "technicals", fake_technicals)


class TestCompareSchema:
    """compare_stocks must emit a consistent per-metric `{value, rank}` schema."""

    @pytest.mark.asyncio
    async def test_rsi_emitted_with_rank_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """rsi is populated but unranked; schema must still be {value, rank}."""
        _install_compare_mocks(
            monkeypatch,
            summaries={
                "AAA": _summary_payload(name="AAA Corp"),
                "BBB": _summary_payload(name="BBB Corp"),
            },
            funds={
                "AAA": _fund_payload(pe=15.0, ps=2.0, eps_yoy=0.10, net_margin=0.15),
                "BBB": _fund_payload(pe=25.0, ps=3.5, eps_yoy=0.05, net_margin=0.08),
            },
            techs={
                "AAA": _tech_payload(rsi_value=45.0, return_1m=0.02),
                "BBB": _tech_payload(rsi_value=72.0, return_1m=-0.01),
            },
        )

        result = await compare_stocks(["AAA", "BBB"])

        assert "error" not in result or not result.get("error")
        # Every metric in every symbol must be a {value, rank} dict — no raw scalars
        for sym in result["symbols"]:
            metrics = result["comparison"][sym]["metrics"]
            for metric_name, metric_payload in metrics.items():
                assert isinstance(metric_payload, dict), (
                    f"{sym}.{metric_name} should be dict, got {type(metric_payload).__name__}"
                )
                assert "value" in metric_payload
                assert "rank" in metric_payload
        # rsi specifically: value populated, rank is None (unranked)
        assert result["comparison"]["AAA"]["metrics"]["rsi"] == {"value": 45.0, "rank": None}
        assert result["comparison"]["BBB"]["metrics"]["rsi"] == {"value": 72.0, "rank": None}

    @pytest.mark.asyncio
    async def test_rsi_excluded_from_composite_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Composite rank average must not include rsi (since rank is None)."""
        _install_compare_mocks(
            monkeypatch,
            summaries={
                "AAA": _summary_payload(name="AAA Corp"),
                "BBB": _summary_payload(name="BBB Corp"),
            },
            funds={
                "AAA": _fund_payload(pe=15.0, ps=2.0, eps_yoy=0.10, net_margin=0.15),
                "BBB": _fund_payload(pe=25.0, ps=3.5, eps_yoy=0.05, net_margin=0.08),
            },
            techs={
                "AAA": _tech_payload(rsi_value=45.0, return_1m=0.02),
                "BBB": _tech_payload(rsi_value=72.0, return_1m=-0.01),
            },
        )

        result = await compare_stocks(["AAA", "BBB"])

        # Both symbols should have composite_rank (not None)
        assert result["comparison"]["AAA"]["composite_rank"] is not None
        assert result["comparison"]["BBB"]["composite_rank"] is not None
        # Composite positions assigned
        assert {result["comparison"]["AAA"]["composite_position"],
                result["comparison"]["BBB"]["composite_position"]} == {1, 2}


class TestCompareErrorPaths:
    """compare_stocks edge cases that previously had no test coverage."""

    @pytest.mark.asyncio
    async def test_too_few_symbols_returns_error(self) -> None:
        result = await compare_stocks(["AAA"])
        assert result.get("error") is True
        assert result.get("error_type") == "invalid_parameters"

    @pytest.mark.asyncio
    async def test_too_many_symbols_returns_error(self) -> None:
        result = await compare_stocks(["A", "B", "C", "D", "E", "F"])
        assert result.get("error") is True
        assert result.get("error_type") == "invalid_parameters"

    @pytest.mark.asyncio
    async def test_one_valid_one_failing_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When only 1 of 2 symbols succeeds, the response is an insufficient_data error."""
        _install_compare_mocks(
            monkeypatch,
            summaries={
                "AAA": _summary_payload(name="AAA Corp"),
                # BBB intentionally absent → fake returns {"error": True, ...}
            },
            funds={
                "AAA": _fund_payload(pe=15.0, ps=2.0, eps_yoy=0.10, net_margin=0.15),
            },
            techs={
                "AAA": _tech_payload(rsi_value=45.0, return_1m=0.02),
            },
        )

        result = await compare_stocks(["AAA", "BBB"])

        assert result.get("error") is True
        assert result.get("error_type") == "insufficient_data"
