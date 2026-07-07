"""Tests for the pure expected-move module."""

from datetime import date

import pandas as pd

from stock_analysis.utils.expected_move import (
    compute_expected_move,
    select_event_expiration,
)

TODAY = date(2026, 7, 7)
EARNINGS = date(2026, 7, 29)


class TestSelectEventExpiration:
    def test_first_expiration_on_or_after_earnings(self) -> None:
        exps = ["2026-07-17", "2026-07-31", "2026-08-21"]
        assert select_event_expiration(exps, EARNINGS, TODAY) == "2026-07-31"

    def test_expiration_exactly_on_earnings_qualifies(self) -> None:
        exps = ["2026-07-29", "2026-08-21"]
        assert select_event_expiration(exps, EARNINGS, TODAY) == "2026-07-29"

    def test_none_when_no_expiration_spans_event(self) -> None:
        assert select_event_expiration(["2026-07-10", "2026-07-17"], EARNINGS, TODAY) is None

    def test_expired_entries_skipped(self) -> None:
        exps = ["2026-01-16", "2026-07-31"]
        assert select_event_expiration(exps, EARNINGS, TODAY) == "2026-07-31"

    def test_unparseable_entries_skipped(self) -> None:
        exps = ["garbage", "2026-07-31"]
        assert select_event_expiration(exps, EARNINGS, TODAY) == "2026-07-31"

    def test_unsorted_input_still_first_chronologically(self) -> None:
        exps = ["2026-08-21", "2026-07-31"]
        assert select_event_expiration(exps, EARNINGS, TODAY) == "2026-07-31"

    def test_empty_is_none(self) -> None:
        assert select_event_expiration([], EARNINGS, TODAY) is None


def chain(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["strike", "bid", "ask", "lastPrice"])


class TestComputeExpectedMove:
    def test_mid_pricing_both_legs(self) -> None:
        calls = chain([{"strike": 100.0, "bid": 3.0, "ask": 3.4, "lastPrice": 3.1}])
        puts = chain([{"strike": 100.0, "bid": 2.6, "ask": 3.0, "lastPrice": 2.9}])
        result = compute_expected_move(calls, puts, 101.0)
        assert result == {"pct": round((3.2 + 2.8) / 101.0, 4), "dollars": 6.0,
                          "strike": 100.0, "basis": "atm_straddle_mid"}

    def test_last_fallback_when_either_leg_lacks_quotes(self) -> None:
        calls = chain([{"strike": 100.0, "bid": 0.0, "ask": 0.0, "lastPrice": 3.1}])
        puts = chain([{"strike": 100.0, "bid": 2.6, "ask": 3.0, "lastPrice": 2.9}])
        result = compute_expected_move(calls, puts, 100.0)
        assert result is not None
        assert result["basis"] == "atm_straddle_last"
        assert result["dollars"] == round(3.1 + 2.8, 2)

    def test_unpriceable_leg_is_none(self) -> None:
        calls = chain([{"strike": 100.0, "bid": 0.0, "ask": 0.0, "lastPrice": 0.0}])
        puts = chain([{"strike": 100.0, "bid": 2.6, "ask": 3.0, "lastPrice": 2.9}])
        assert compute_expected_move(calls, puts, 100.0) is None

    def test_atm_is_nearest_common_strike_tie_lower(self) -> None:
        calls = chain([{"strike": 95.0, "bid": 1, "ask": 1.2, "lastPrice": 1},
                       {"strike": 105.0, "bid": 1, "ask": 1.2, "lastPrice": 1}])
        puts = chain([{"strike": 95.0, "bid": 1, "ask": 1.2, "lastPrice": 1},
                      {"strike": 105.0, "bid": 1, "ask": 1.2, "lastPrice": 1}])
        result = compute_expected_move(calls, puts, 100.0)  # equidistant -> 95
        assert result is not None and result["strike"] == 95.0

    def test_no_common_strike_is_none(self) -> None:
        calls = chain([{"strike": 95.0, "bid": 1, "ask": 1.2, "lastPrice": 1}])
        puts = chain([{"strike": 105.0, "bid": 1, "ask": 1.2, "lastPrice": 1}])
        assert compute_expected_move(calls, puts, 100.0) is None

    def test_degenerate_inputs_are_none(self) -> None:
        good = chain([{"strike": 100.0, "bid": 1, "ask": 1.2, "lastPrice": 1}])
        assert compute_expected_move(None, good, 100.0) is None
        assert compute_expected_move(good, chain([]), 100.0) is None
        assert compute_expected_move(good, good, 0.0) is None
        assert compute_expected_move(good, good, None) is None

    def test_non_numeric_quotes_are_none_not_an_exception(self) -> None:
        calls = chain([{"strike": 100.0, "bid": "N/A", "ask": "N/A", "lastPrice": "N/A"}])
        puts = chain([{"strike": 100.0, "bid": 2.6, "ask": 3.0, "lastPrice": 2.9}])
        assert compute_expected_move(calls, puts, 100.0) is None
