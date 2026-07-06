"""All tunable thresholds for the trade-setup card. Values pinned by the design doc."""

from stock_analysis.utils.freshness import (
    FRESHNESS_CEILING_MINUTES as FRESHNESS_CEILING_MINUTES,
)

PROBE_PERIOD = "1d"
PROBE_INTERVAL = "5m"

PULLBACK_MIN_OFF_HIGH = 0.03
PULLBACK_MAX_OFF_HIGH = 0.08
PULLBACK_RSI_MIN = 35.0
PULLBACK_RSI_MAX = 55.0

BREAKOUT_MAX_BELOW_HIGH = 0.03
BREAKOUT_VOLUME_RATIO_MIN = 1.5
BREAKOUT_BANDWIDTH_PCTILE_MAX = 0.25
BREAKOUT_STOP_LEVEL_BUFFER = 0.01  # stop sits 1% under the breakout level

MEANREV_RSI_MAX = 30.0
MEANREV_ZSCORE_MAX = -2.0
MEANREV_SUPPORT_SMA200_FACTOR = 0.90

STOP_ATR_MULT = 2.0
MEANREV_STOP_ATR_MULT = 2.5

TIME_STOP_TRADING_DAYS = {
    "pullback_in_uptrend": 10,
    "breakout": 10,
    "oversold_mean_reversion": 5,
}

SETUP_PRIORITY = ["pullback_in_uptrend", "breakout", "oversold_mean_reversion"]

PDT_ACCOUNT_MIN = 25_000.0
DEFAULT_REVIEW_TRADING_DAYS = 10

# Tools whose failure makes the trade card unable to gate/detect safely.
TRADE_CRITICAL_TOOLS = frozenset({"technicals", "risk_metrics"})
