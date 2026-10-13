"""The card's liquidity basis, shared by risk_metrics and scan stage 1 (D-11)."""

from __future__ import annotations

import pandas as pd

# risk_metrics needs this many daily bars; below it the card has no liquidity basis.
RISK_METRICS_MIN_BARS = 50


def mean_dollar_volume(volume: pd.Series, last_close: float | None) -> float | None:
    """Mean volume over the frame x the last close."""
    if last_close is None or volume.isna().all():
        return None
    return float(volume.mean()) * last_close
