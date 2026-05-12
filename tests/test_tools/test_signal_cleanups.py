"""Regression tests for the cluster of low-risk truthiness/sentiment fixes."""

import pandas as pd

from stock_analysis.tools.analyze.action_zones import _classify_current_zone
from stock_analysis.tools.options_signals import _compute_put_call_ratio
from stock_analysis.tools.ownership import _sign_shares


class TestOwnershipUnknownTxnDirection:
    """_sign_shares returns 0 for unknown transaction types to avoid sentiment inflation."""

    def test_known_sale(self) -> None:
        assert _sign_shares(100, "Sale of common stock") == -100

    def test_known_purchase(self) -> None:
        assert _sign_shares(100, "Purchase of common stock") == 100

    def test_option_exercise_returns_zero(self) -> None:
        """Unknown text (e.g., option exercise) returns 0 — not raw positive shares."""
        assert _sign_shares(100, "Exercise of Stock Option") == 0

    def test_gift_returns_zero(self) -> None:
        assert _sign_shares(50, "Gift") == 0

    def test_conversion_returns_zero(self) -> None:
        assert _sign_shares(75, "Conversion") == 0

    def test_empty_text_returns_zero(self) -> None:
        assert _sign_shares(100, "") == 0


class TestPutCallRatioZeroVolume:
    """Zero volume on one side must not be treated as missing."""

    def test_zero_call_volume_yields_no_volume_ratio(self) -> None:
        """call_volume == 0 means we can't divide, but call_volume is *known*; we
        emit volume_ratio=None for the zero-denominator case, distinct from a
        missing column. The key contract is that the check uses `is not None`
        followed by `> 0`, not a truthy check that confuses zero with missing.
        """
        calls = pd.DataFrame({"volume": [0, 0, 0], "openInterest": [100, 200, 300]})
        puts = pd.DataFrame({"volume": [50, 50, 50], "openInterest": [200, 200, 200]})
        warnings: list[str] = []

        result = _compute_put_call_ratio(calls, puts, warnings)

        # Zero call_volume → division impossible → volume_ratio is None
        assert result["volume_based"] is None
        # OI ratio is computable: 600 / 600 = 1.0
        assert result["oi_based"] is not None

    def test_normal_volumes_compute_ratio(self) -> None:
        calls = pd.DataFrame({"volume": [100, 200, 300], "openInterest": [100, 200, 300]})
        puts = pd.DataFrame({"volume": [200, 200, 200], "openInterest": [200, 200, 200]})
        warnings: list[str] = []

        result = _compute_put_call_ratio(calls, puts, warnings)

        # put 600 / call 600 = 1.0
        assert result["volume_based"] == 1.0
        assert result["oi_based"] == 1.0


class TestActionZoneOrdering:
    """Reduce-zone check must precede the SMA200 accumulate fallback."""

    def test_above_reduce_level_returns_reduce_not_accumulate(self) -> None:
        """Price below SMA200 but at/above reduce_above must be `reduce`."""
        # Scenario: 52w_high - 1·ATR puts reduce_above at 95.0; price is 96
        # (below SMA200 of 100 but above the reduce threshold).
        zone = _classify_current_zone(
            current_price=96.0,
            levels={"strong_buy_below": 80.0, "reduce_above": 95.0},
            sma_50=92.0,
            sma_200=100.0,
        )
        assert zone == "reduce"

    def test_below_sma200_below_reduce_is_accumulate(self) -> None:
        zone = _classify_current_zone(
            current_price=90.0,
            levels={"strong_buy_below": 80.0, "reduce_above": 110.0},
            sma_50=92.0,
            sma_200=100.0,
        )
        assert zone == "accumulate"

    def test_strong_buy_takes_priority(self) -> None:
        zone = _classify_current_zone(
            current_price=75.0,
            levels={"strong_buy_below": 80.0, "reduce_above": 110.0},
            sma_50=92.0,
            sma_200=100.0,
        )
        assert zone == "strong_buy"
