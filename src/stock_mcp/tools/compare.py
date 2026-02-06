"""Stock comparison tool."""

import asyncio
from time import perf_counter
from typing import Any

from stock_mcp.tools.fundamentals import fundamentals_snapshot
from stock_mcp.tools.stock_summary import stock_summary
from stock_mcp.tools.technicals import technicals
from stock_mcp.utils.helpers import safe_float, safe_round
from stock_mcp.utils.provenance import build_error_response, build_meta

# Metrics where lower values are better (rank 1 = lowest)
_LOWER_IS_BETTER = {"pe_trailing", "ps_trailing", "peg_ratio", "debt_to_equity"}

# Metrics where higher values are better (rank 1 = highest)
_HIGHER_IS_BETTER = {
    "revenue_yoy",
    "eps_yoy",
    "net_margin",
    "gross_margin",
    "roe",
    "return_1m",
    "return_3m",
    "return_1y",
    "dividend_yield",
    "fcf_yield",
    "position_in_52w_range",
}

_ALL_METRICS = _LOWER_IS_BETTER | _HIGHER_IS_BETTER


async def compare_stocks(symbols: list[str]) -> dict[str, Any]:
    """
    Compare 2-5 stocks side-by-side with ranked metrics.

    Args:
        symbols: List of 2-5 stock ticker symbols

    Returns:
        Dict with comparison table, per-metric rankings, and composite ranks
    """
    start_time = perf_counter()

    if not symbols or len(symbols) < 2 or len(symbols) > 5:
        return build_error_response(
            error_type="invalid_parameters",
            message="Must provide between 2 and 5 symbols",
        )

    symbols = [s.upper().strip() for s in symbols]

    # Fetch all data in parallel: summary + fundamentals + technicals per symbol
    summary_tasks = [stock_summary(s) for s in symbols]
    fundamental_tasks = [fundamentals_snapshot(s) for s in symbols]
    technical_tasks = [technicals(s) for s in symbols]

    all_results = await asyncio.gather(
        *summary_tasks, *fundamental_tasks, *technical_tasks,
        return_exceptions=True,
    )

    n = len(symbols)
    summary_results = all_results[:n]
    fundamental_results = all_results[n : 2 * n]
    technical_results = all_results[2 * n : 3 * n]

    # Build per-symbol data, skipping failures
    warnings: list[str] = []
    symbol_data: dict[str, dict[str, Any]] = {}

    for i, sym in enumerate(symbols):
        summ = summary_results[i]
        fund = fundamental_results[i]
        tech = technical_results[i]

        if isinstance(summ, Exception):
            warnings.append(f"{sym}: summary fetch failed: {summ}")
            continue
        if isinstance(fund, Exception):
            warnings.append(f"{sym}: fundamentals fetch failed: {fund}")
            continue
        if isinstance(tech, Exception):
            warnings.append(f"{sym}: technicals fetch failed: {tech}")
            continue

        if summ.get("error") or fund.get("error") or tech.get("error"):
            failed = [
                r.get("message", "unknown error")
                for r in (summ, fund, tech)
                if r.get("error")
            ]
            warnings.append(f"{sym}: {'; '.join(failed)}")
            continue

        metrics = _extract_metrics(fund, tech)
        symbol_data[sym] = {
            "name": summ.get("name"),
            "sector": summ.get("sector"),
            "market_cap": summ.get("market_cap"),
            "metrics": metrics,
        }

    if len(symbol_data) < 2:
        duration_ms = (perf_counter() - start_time) * 1000
        return build_error_response(
            error_type="insufficient_data",
            message=f"Need at least 2 valid symbols for comparison, got {len(symbol_data)}",
        )

    # Rank each metric across symbols
    valid_symbols = list(symbol_data.keys())
    metric_rankings: dict[str, list[dict[str, Any]]] = {}

    for metric in _ALL_METRICS:
        entries: list[dict[str, Any]] = []
        for sym in valid_symbols:
            value = symbol_data[sym]["metrics"].get(metric)
            entries.append({"symbol": sym, "value": value})

        ranked = _rank_metric(entries, metric)
        metric_rankings[metric] = ranked

        # Write ranks back into symbol_data
        for item in ranked:
            sym = item["symbol"]
            symbol_data[sym]["metrics"][metric] = {
                "value": item["value"],
                "rank": item["rank"],
            }

    # Composite ranking: average rank across all metrics with data
    for sym in valid_symbols:
        ranks = [
            symbol_data[sym]["metrics"][m]["rank"]
            for m in _ALL_METRICS
            if m in symbol_data[sym]["metrics"]
            and symbol_data[sym]["metrics"][m]["value"] is not None
        ]
        symbol_data[sym]["composite_rank"] = (
            safe_round(sum(ranks) / len(ranks), 2) if ranks else None
        )

    # Assign composite position (1 = best = lowest composite_rank)
    ranked_symbols = sorted(
        valid_symbols,
        key=lambda s: (
            symbol_data[s]["composite_rank"]
            if symbol_data[s]["composite_rank"] is not None
            else float("inf")
        ),
    )
    for pos, sym in enumerate(ranked_symbols, start=1):
        symbol_data[sym]["composite_position"] = pos

    duration_ms = (perf_counter() - start_time) * 1000

    return {
        "meta": build_meta("compare_stocks", duration_ms),
        "symbols": valid_symbols,
        "comparison": symbol_data,
        "metric_rankings": metric_rankings,
        "warnings": warnings if warnings else None,
    }


def _extract_metrics(
    fund: dict[str, Any],
    tech: dict[str, Any],
) -> dict[str, float | None]:
    """Extract comparable metrics from fundamentals and technicals results."""
    valuation = fund.get("valuation", {})
    growth = fund.get("growth", {})
    profitability = fund.get("profitability", {})
    health = fund.get("financial_health", {})
    yield_metrics = fund.get("yield_metrics", {})
    price_position = tech.get("price_position", {})
    returns = tech.get("returns", {})
    rsi = tech.get("rsi", {})

    return {
        # Valuation
        "pe_trailing": safe_float(valuation.get("pe_trailing")),
        "ps_trailing": safe_float(valuation.get("ps_trailing")),
        "peg_ratio": safe_float(valuation.get("peg_ratio")),
        # Growth
        "revenue_yoy": safe_float(growth.get("revenue_yoy")),
        "eps_yoy": safe_float(growth.get("eps_yoy")),
        # Profitability
        "net_margin": safe_float(profitability.get("net_margin")),
        "gross_margin": safe_float(profitability.get("gross_margin")),
        "roe": safe_float(profitability.get("roe")),
        # Health
        "debt_to_equity": safe_float(health.get("debt_to_equity")),
        # Risk
        "position_in_52w_range": safe_float(price_position.get("position_in_range")),
        # Technicals
        "return_1m": safe_float(returns.get("return_1m")),
        "return_3m": safe_float(returns.get("return_3m")),
        "return_1y": safe_float(returns.get("return_1y")),
        "rsi": safe_float(rsi.get("value")),
        # Yield
        "dividend_yield": safe_float(yield_metrics.get("dividend_yield")),
        "fcf_yield": safe_float(yield_metrics.get("fcf_yield")),
    }


def _rank_metric(
    entries: list[dict[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    """Rank entries for a single metric, direction-aware. None gets worst rank."""
    reverse = metric in _HIGHER_IS_BETTER
    num_entries = len(entries)

    has_value = [e for e in entries if e["value"] is not None]
    no_value = [e for e in entries if e["value"] is None]

    has_value.sort(key=lambda e: e["value"], reverse=reverse)

    ranked: list[dict[str, Any]] = []
    for rank, entry in enumerate(has_value, start=1):
        ranked.append({**entry, "rank": rank})

    worst_rank = num_entries
    for entry in no_value:
        ranked.append({**entry, "rank": worst_rank})

    return ranked
