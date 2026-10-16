"""One offline scan promotion, retaining the real screen, card and plan builders.

Reuses swing_matrix's synthetic history, summary and event stubs. Technicals and
risk metrics are real; no options are fetched. The scan's public row has no size,
so the recording labels the captured intermediate card separately. State lives
in a temporary directory and the caller supplies the provider-transport guard
(the sizing recorder and pytest both do).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from stock_analysis.tools import watchlist
from tests.support import swing_matrix as sm

LIMITS = {"account_size": 10_000.0, "risk_per_trade_pct": 0.25, "max_position_pct": 8.25}


async def build_scan_case() -> dict[str, Any]:
    """Run first-scan promotion and capture the real card with its forwarded limits."""
    captured: list[dict[str, Any]] = []
    real_card = watchlist.analyze_trade_setup

    async def capture_card(symbol: str, **kwargs: Any) -> dict[str, Any]:
        card = await real_card(symbol, **kwargs)
        captured.append({"symbol": symbol, "limits": {key: kwargs[key] for key in LIMITS},
                         "clock": kwargs["_now"].isoformat(),
                         "action": card["action"], "setup": card["setup"], "plan": card["plan"]})
        return card

    with TemporaryDirectory(prefix="sizing-scan-") as directory, pytest.MonkeyPatch.context() as patch:
        data_dir = Path(directory)
        patch.setenv("STOCK_ANALYSIS_DATA_DIR", directory)
        sm.install(patch, sm.SCAN_SCENARIOS["fresh_breakout"])
        patch.setattr(watchlist, "analyze_trade_setup", capture_card)
        await watchlist.manage_watchlist("add", [sm.SYMBOL], _today=sm.PRE_MARKET.date(), data_dir=data_dir)
        scan = await watchlist.scan_watchlist(**LIMITS, _now=sm.PRE_MARKET, data_dir=data_dir)
    (card,) = captured
    return {"inputs": dict(LIMITS), "clock": sm.PRE_MARKET.isoformat(),
            "scan": scan, "promoted_card": card}
