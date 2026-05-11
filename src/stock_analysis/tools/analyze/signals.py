"""Signal generation and risk regime classification."""

from dataclasses import dataclass, field
from typing import Any

TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class UnprofitabilityResult:
    """Structured unprofitability classification.

    Centralizes the per-trigger reasons so each caller can decide which subset of
    triggers it cares about. This pins down the (intentional) drift between
    orchestrator/action_zones/verdict in one place rather than scattered conditionals.

    Each call site historically uses a different combination:
      - orchestrator: by_margin | by_eps
      - action_zones: by_margin | by_eps | by_signal | by_no_pe_with_negative_fcf
      - verdict (business quality): by_margin | by_eps | by_signal | by_pe_not_meaningful
    """

    triggers: frozenset[str] = field(default_factory=frozenset)

    @property
    def by_margin(self) -> bool:
        return "net_margin_negative" in self.triggers

    @property
    def by_eps(self) -> bool:
        return "trailing_eps_nonpositive" in self.triggers

    @property
    def by_signal(self) -> bool:
        return "unprofitable_signal" in self.triggers

    @property
    def by_pe_not_meaningful(self) -> bool:
        return "pe_not_meaningful" in self.triggers

    @property
    def by_no_pe_with_negative_fcf(self) -> bool:
        return "no_pe_with_negative_fcf" in self.triggers

    @property
    def any(self) -> bool:
        """True if any trigger fired (most permissive view)."""
        return bool(self.triggers)


def classify_unprofitability(
    *,
    net_margin: float | None = None,
    trailing_eps: float | None = None,
    signaled_unprofitable: bool = False,
    pe_not_meaningful: bool = False,
    pe_trailing: float | None = None,
    has_negative_fcf_signal: bool = False,
) -> UnprofitabilityResult:
    """Classify whether a company exhibits unprofitability signals.

    All inputs are optional; pass only what's available. The result lets each call
    site read a specific subset of triggers via `by_*` properties.

    Triggers:
      - net_margin_negative: net_margin is not None and net_margin < 0
      - trailing_eps_nonpositive: trailing_eps is not None and trailing_eps <= 0
      - unprofitable_signal: caller passed signaled_unprofitable=True
      - pe_not_meaningful: caller passed pe_not_meaningful=True
      - no_pe_with_negative_fcf: pe_trailing is None and has_negative_fcf_signal=True
    """
    triggers: set[str] = set()
    if net_margin is not None and net_margin < 0:
        triggers.add("net_margin_negative")
    if trailing_eps is not None and trailing_eps <= 0:
        triggers.add("trailing_eps_nonpositive")
    if signaled_unprofitable:
        triggers.add("unprofitable_signal")
    if pe_not_meaningful:
        triggers.add("pe_not_meaningful")
    if pe_trailing is None and has_negative_fcf_signal:
        triggers.add("no_pe_with_negative_fcf")
    return UnprofitabilityResult(triggers=frozenset(triggers))


# Volatility regime thresholds
VOLATILITY_REGIME_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.40,
    "high": 0.60,
}


def vol_threshold_for_improvement(
    risk_regime: str | None,
    annualized_vol: float | None,
) -> float | None:
    """Return volatility threshold that improves the current regime."""
    if risk_regime == "extreme":
        return VOLATILITY_REGIME_THRESHOLDS["high"]
    if risk_regime == "high":
        return VOLATILITY_REGIME_THRESHOLDS["medium"]
    if risk_regime == "medium":
        return VOLATILITY_REGIME_THRESHOLDS["low"]
    if annualized_vol is None:
        return None
    if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["high"]:
        return VOLATILITY_REGIME_THRESHOLDS["high"]
    if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["medium"]:
        return VOLATILITY_REGIME_THRESHOLDS["medium"]
    if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["low"]:
        return VOLATILITY_REGIME_THRESHOLDS["low"]
    return None


def get_rule_triggered(data: dict[str, Any], rule_name: str) -> bool | None:
    """Extract triggered value from a rules dict."""
    rules = data.get("rules", {})
    rule = rules.get(rule_name, {})
    triggered = rule.get("triggered")
    if triggered is None:
        return None
    return bool(triggered)


def classify_risk_regime(
    annualized_vol: float | None,
    max_drawdown: float | None,
    beta_val: float | None,
) -> dict[str, Any]:
    """
    Classify overall risk regime based on volatility, drawdown, and beta.

    Returns a dict with classification (low/medium/high/extreme), factors, and thresholds.

    Risk thresholds:
    - Volatility: low <25%, medium 25-40%, high 40-60%, extreme >60%
    - Drawdown: low >-20%, medium -20% to -35%, high -35% to -50%, extreme <-50%
    - Beta: low <0.8, medium 0.8-1.2, high 1.2-1.8, extreme >1.8
    """
    factors: dict[str, str | None] = {}
    scores: list[int] = []  # 0=low, 1=medium, 2=high, 3=extreme

    # Classify volatility
    if annualized_vol is not None:
        if annualized_vol < VOLATILITY_REGIME_THRESHOLDS["low"]:
            factors["volatility"] = "low"
            scores.append(0)
        elif annualized_vol < VOLATILITY_REGIME_THRESHOLDS["medium"]:
            factors["volatility"] = "medium"
            scores.append(1)
        elif annualized_vol < VOLATILITY_REGIME_THRESHOLDS["high"]:
            factors["volatility"] = "high"
            scores.append(2)
        else:
            factors["volatility"] = "extreme"
            scores.append(3)
    else:
        factors["volatility"] = None

    # Classify drawdown (note: drawdown is negative)
    if max_drawdown is not None:
        if max_drawdown > -0.20:
            factors["drawdown"] = "low"
            scores.append(0)
        elif max_drawdown > -0.35:
            factors["drawdown"] = "medium"
            scores.append(1)
        elif max_drawdown > -0.50:
            factors["drawdown"] = "high"
            scores.append(2)
        else:
            factors["drawdown"] = "extreme"
            scores.append(3)
    else:
        factors["drawdown"] = None

    # Classify beta
    if beta_val is not None:
        if beta_val < 0.8:
            factors["beta"] = "low"
            scores.append(0)
        elif beta_val < 1.2:
            factors["beta"] = "medium"
            scores.append(1)
        elif beta_val < 1.8:
            factors["beta"] = "high"
            scores.append(2)
        else:
            factors["beta"] = "extreme"
            scores.append(3)
    else:
        factors["beta"] = None

    # Overall classification: use max score (most conservative)
    classification: str | None = None
    if scores:
        max_score = max(scores)
        classification = ["low", "medium", "high", "extreme"][max_score]

    return {
        "classification": classification,
        "factors": factors,
        "thresholds": {
            "volatility": {
                "low": f"<{VOLATILITY_REGIME_THRESHOLDS['low'] * 100:.0f}%",
                "medium": (
                    f"{VOLATILITY_REGIME_THRESHOLDS['low'] * 100:.0f}-"
                    f"{VOLATILITY_REGIME_THRESHOLDS['medium'] * 100:.0f}%"
                ),
                "high": (
                    f"{VOLATILITY_REGIME_THRESHOLDS['medium'] * 100:.0f}-"
                    f"{VOLATILITY_REGIME_THRESHOLDS['high'] * 100:.0f}%"
                ),
                "extreme": f">{VOLATILITY_REGIME_THRESHOLDS['high'] * 100:.0f}%",
            },
            "drawdown": {"low": ">-20%", "medium": "-20% to -35%", "high": "-35% to -50%", "extreme": "<-50%"},
            "beta": {"low": "<0.8", "medium": "0.8-1.2", "high": "1.2-1.8", "extreme": ">1.8"},
        },
    }


def generate_signals(
    technicals: dict[str, Any],
    fundamentals: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, list[str]]:
    """Generate bullish/bearish/neutral signals from all available data."""
    bullish: list[str] = []
    bearish: list[str] = []
    neutral: list[str] = []

    ma = technicals.get("moving_averages", {})
    rsi_data = technicals.get("rsi", {})
    macd_data = technicals.get("macd", {})
    returns = technicals.get("returns", {})

    growth = fundamentals.get("growth", {})
    profit = fundamentals.get("profitability", {})
    health = fundamentals.get("financial_health", {})
    cf = fundamentals.get("cash_flow", {})

    vol = risk.get("volatility", {})
    beta_data = risk.get("beta", {})

    # Technical signals
    if get_rule_triggered(ma, "above_sma200") is True:
        bullish.append("price_above_sma200")
    elif get_rule_triggered(ma, "above_sma200") is False:
        bearish.append("price_below_sma200")

    if get_rule_triggered(ma, "golden_cross") is True:
        bullish.append("golden_cross")
    elif get_rule_triggered(ma, "death_cross") is True:
        bearish.append("death_cross")

    if get_rule_triggered(rsi_data, "oversold") is True:
        bullish.append("rsi_oversold")
    elif get_rule_triggered(rsi_data, "overbought") is True:
        bearish.append("rsi_overbought")

    if get_rule_triggered(macd_data, "bullish_cross") is True:
        bullish.append("macd_bullish")
    elif get_rule_triggered(macd_data, "bearish_cross") is True:
        bearish.append("macd_bearish")

    # Returns-based signals
    ret_3m = returns.get("return_3m")
    if ret_3m is not None:
        if ret_3m > 0.15:
            bullish.append("strong_3m_momentum")
        elif ret_3m < -0.15:
            bearish.append("weak_3m_momentum")
        else:
            neutral.append("moderate_3m_momentum")

    # Growth signals
    if get_rule_triggered(growth, "high_growth") is True:
        bullish.append("high_revenue_growth")
    elif get_rule_triggered(growth, "positive_revenue_growth") is False:
        bearish.append("negative_revenue_growth")

    # Value/health signals
    if get_rule_triggered(health, "net_cash_positive") is True:
        bullish.append("net_cash_positive")
    if get_rule_triggered(health, "low_debt") is True:
        bullish.append("low_debt")

    # Profitability signals
    if get_rule_triggered(profit, "profitable") is True:
        bullish.append("profitable")
    elif get_rule_triggered(profit, "profitable") is False:
        bearish.append("unprofitable")

    if get_rule_triggered(cf, "positive_fcf") is True:
        bullish.append("positive_free_cash_flow")
    elif get_rule_triggered(cf, "positive_fcf") is False:
        bearish.append("negative_free_cash_flow")

    # Risk signals
    annualized_vol = vol.get("annualized")
    if annualized_vol is not None:
        if annualized_vol >= VOLATILITY_REGIME_THRESHOLDS["high"]:
            bearish.append("very_high_volatility")
        elif get_rule_triggered(vol, "high_volatility") is True:
            bearish.append("high_volatility")

    # Drawdown signals
    dd = risk.get("drawdown", {})
    max_dd = dd.get("max_1y")
    if max_dd is not None and max_dd < -0.50:
        bearish.append("deep_drawdown")

    beta_val = beta_data.get("value")
    if beta_val is not None:
        if beta_val > 1.5:
            neutral.append("very_high_beta")
        elif beta_val < 0.5:
            neutral.append("low_beta_defensive")

    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
    }
