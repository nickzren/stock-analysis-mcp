"""Canonical holder assessment with optional, stateless personal context."""

from datetime import datetime
from time import perf_counter
from typing import Any, cast
from zoneinfo import ZoneInfo

from stock_analysis.tools.analyze.orchestrator import _analyze_with_context
from stock_analysis.tools.position_context import build_position_context
from stock_analysis.utils.helpers import safe_round, valid_price
from stock_analysis.utils.position_inputs import normalize_position_inputs
from stock_analysis.utils.provenance import build_error_response, build_meta

_ET = ZoneInfo("America/New_York")


async def analyze_position(
    symbol: str,
    cost_basis: float | str | None = None,
    purchase_date: str | None = None,
    shares: float | str | None = None,
    horizon: str = "mid_term",
    portfolio_value: float | str | None = None,
    max_holding_pct: float | str | None = None,
    *,
    _now: datetime | None = None,
) -> dict[str, Any]:
    """One stock analysis, with personal calculations kept outside its decision."""
    started = perf_counter()
    now = (_now or datetime.now(_ET)).astimezone(_ET)
    today = now.astimezone(_ET).date()
    try:
        inputs = normalize_position_inputs(
            cost_basis, purchase_date, shares, horizon, portfolio_value, max_holding_pct,
            today=today,
        )
    except ValueError as exc:
        return build_error_response("invalid_parameters", str(exc), symbol=symbol)

    try:
        analysis, fetched = await _analyze_with_context(symbol, _now=now)
    except ValueError as exc:
        return build_error_response("invalid_symbol", str(exc), symbol=symbol)
    decision = analysis["decision_card"]
    chosen = decision if inputs.horizon == "mid_term" else decision["long_term"]
    personal = build_position_context(
        inputs=inputs, summary=analysis["summary"], quote_type=fetched["quote_type"], today=today,
    )
    holder = chosen["if_held"]
    concentration = personal["concentration_check"]
    addition = _addition_context(holder, chosen["new_money"], concentration["status"])
    card = {
        "horizon": inputs.horizon,
        "holder_assessment": holder,
        "new_money": chosen["new_money"],
        "context_status": personal["context_status"],
        "review_status": personal["review_status"],
        "concentration_check": concentration,
        "addition_context": addition,
    }
    fit = analysis["verdict"]["horizon_fit"]
    suitability = {
        "horizon": inputs.horizon,
        "fit": fit[inputs.horizon],
        "reasons": [reason for reason in fit["reasons"] if reason.startswith(inputs.horizon + ":")],
        "note": "Suitability is separate from the holder assessment; hold is not an endorsement.",
    }
    technical = fetched["technicals"]
    ma, rsi, macd = (technical.get(key) or {} for key in ("moving_averages", "rsi", "macd"))
    result = {
        "meta": build_meta("analyze_position", (perf_counter() - started) * 1000),
        "data_provenance": analysis["data_provenance"],
        "data_quality": analysis["data_quality"],
        "symbol": analysis["symbol"],
        "position_card": card,
        "position": personal["position"],
        "holding": personal["holding"],
        "assumptions": personal["assumptions"],
        "suitability": suitability,
        "tax": {"status": "not_assessed", "reason": "Personal tax treatment and sale quantities are not assessed."},
        "diagnostics": {"technicals": {
            "quote_currency": analysis["summary"].get("currency"),
            "current_signals": _build_current_signals(ma, rsi, macd, technical.get("returns") or {}),
            "flags": _build_technical_flags(ma, rsi, macd),
            "reference_levels": _build_reference_levels(valid_price(analysis["summary"]["current_price"]), technical),
            "note": "Unmonitored technical context, not holder stops or instructions to exit.",
        }},
    }
    for key in ("error", "error_type", "message"):
        if key in analysis:
            result[key] = analysis[key]
    return result


def _addition_context(holder: dict[str, Any], new_money: dict[str, Any], allocation: str) -> str:
    if holder["add_allowed"]:
        if allocation == "above_limit":
            return ("The stock model permits considering new money, but this holding already exceeds your own "
                    "ceiling; no remaining add capacity has been assessed.")
        if allocation == "not_evaluated":
            return "The stock model permits considering new money, but the requested allocation and add capacity have not been checked."
        return "The stock model permits considering new money; no remaining add capacity has been assessed."
    blockers = ", ".join(gate["id"] for gate in new_money["blocked_by"])
    concern = {
        "above_limit": " The holding also exceeds your own allocation ceiling.",
        "not_evaluated": " The requested allocation and add capacity have not been checked.",
    }.get(allocation, "")
    blocker_text = f" ({blockers})" if blockers else ""
    message = f"Adding is not allowed by the stock model: {new_money['action']}"
    return message + blocker_text + "." + concern


def _build_current_signals(
    ma: dict[str, Any],
    rsi: dict[str, Any],
    macd: dict[str, Any],
    returns: dict[str, Any],
) -> dict[str, list[str]]:
    """Build bullish and bearish signal labels from technical rule flags."""
    bullish: list[str] = []
    bearish: list[str] = []

    if _get_rule_triggered(ma, "above_sma50") is True:
        bullish.append("above_sma50")
    if _get_rule_triggered(ma, "above_sma200") is True:
        bullish.append("above_sma200")
    if _get_rule_triggered(ma, "golden_cross") is True:
        bullish.append("golden_cross")
    if _get_rule_triggered(rsi, "oversold") is True:
        bullish.append("rsi_oversold")
    if _get_rule_triggered(macd, "bullish_cross") is True:
        bullish.append("macd_bullish")

    ret_1m = returns.get("return_1m")
    if ret_1m is not None and ret_1m > 0.05:
        bullish.append("positive_1m_momentum")

    if _get_rule_triggered(ma, "above_sma50") is False:
        bearish.append("below_sma50")
    if _get_rule_triggered(ma, "above_sma200") is False:
        bearish.append("below_sma200")
    if _get_rule_triggered(ma, "death_cross") is True:
        bearish.append("death_cross")
    if _get_rule_triggered(rsi, "overbought") is True:
        bearish.append("rsi_overbought")
    if _get_rule_triggered(macd, "bearish_cross") is True:
        bearish.append("macd_bearish")

    if ret_1m is not None and ret_1m < -0.05:
        bearish.append("negative_1m_momentum")

    return {"bullish": bullish, "bearish": bearish}


def _build_technical_flags(
    ma: dict[str, Any],
    rsi: dict[str, Any],
    macd: dict[str, Any],
) -> dict[str, bool | None]:
    """Retain raw technical flags as diagnostics, never an exit assessment."""
    broke_sma50 = _invert_nullable(_get_rule_triggered(ma, "above_sma50"))
    broke_sma200 = _invert_nullable(_get_rule_triggered(ma, "above_sma200"))
    death_cross = _get_rule_triggered(ma, "death_cross")
    rsi_overbought = _get_rule_triggered(rsi, "overbought")
    macd_bearish = _get_rule_triggered(macd, "bearish_cross")

    return {
        "broke_sma50": broke_sma50,
        "broke_sma200": broke_sma200,
        "death_cross": death_cross,
        "rsi_overbought": rsi_overbought,
        "macd_bearish": macd_bearish,
    }


def _build_reference_levels(
    current_price: float | None,
    tech_result: dict[str, Any],
) -> dict[str, float | None]:
    """Unmonitored technical reference levels, not holder stops."""
    ma = tech_result.get("moving_averages", {})
    atr = tech_result.get("atr", {})
    price_pos = tech_result.get("price_position", {})

    atr_val = atr.get("value")
    atr_1x_below = current_price - atr_val if current_price is not None and atr_val else None

    return {
        "sma_50": ma.get("sma_50"),
        "sma_200": ma.get("sma_200"),
        "atr_1x_below": safe_round(atr_1x_below, 2),
        "recent_low_1m": price_pos.get("low_1m"),
    }


def _get_rule_triggered(data: dict[str, Any], rule_name: str) -> bool | None:
    """Extract triggered value from a rules dict."""
    rules = data.get("rules", {})
    rule = rules.get(rule_name, {})
    return cast(bool | None, rule.get("triggered"))


def _invert_nullable(value: bool | None) -> bool | None:
    """Invert a boolean while preserving None."""
    if value is None:
        return None
    return not value
