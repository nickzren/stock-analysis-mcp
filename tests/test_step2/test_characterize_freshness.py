"""Step 2 WP2.0: freshness characterization, FR-C1-FR-C5.

Annex: docs/plans/step2-v2-research/freshness.md, "Corrected proposal" > "Tests". These
record today's behavior (0609fde); they are not red-first. Each test is ported from an annex
probe (probe_calendar.py, probe_price_basis.py, probe_technicals_fields.py,
probe_risk_nan_latest.py, v3_corrected_gate.py). Every clock is pinned. FR-C5 was flipped by
WP2.1 and now pins PD D-12 (one valid-price predicate: finite and > 0).
"""

import importlib
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

import stock_analysis.tools.trade_setup.orchestrator as trade_orch
from stock_analysis.data.cache_manager import classify_session
from stock_analysis.tools.analyze.gates import check_liquidity
from stock_analysis.utils.freshness import build_freshness, freshness_blockers
from stock_analysis.utils.market_calendar import last_completed_session
from stock_analysis.utils.price_basis import select_price
from stock_analysis.utils.validators import FetchParams
from tests.support import decision_matrix as dm
from tests.support.freshness_frames import REGULAR, at, daily, five_min
from tests.test_tools.test_setup_detection import make_technicals

tech_mod = importlib.import_module("stock_analysis.tools.technicals")
risk_mod = importlib.import_module("stock_analysis.tools.risk_metrics")


def history(frames: dict[str, pd.DataFrame]) -> Any:
    """A fetch_history stand-in serving one frame per interval."""

    async def fake_history(params: FetchParams) -> pd.DataFrame:
        return frames[params.interval]

    return fake_history


class TestExpectedSession:
    # (now, session, E): the annex C1 instants (probe_calendar.py).
    CASES = {
        "overnight": (at(2026, 10, 13, 3), "closed", date(2026, 10, 12)),
        "pre_market": (at(2026, 10, 13, 8), "pre_market", date(2026, 10, 12)),
        "open_0930": (at(2026, 10, 13, 9, 30), "regular", date(2026, 10, 12)),
        "regular_1559": (at(2026, 10, 13, 15, 59), "regular", date(2026, 10, 12)),
        "close_1600": (at(2026, 10, 13, 16), "after_hours", date(2026, 10, 13)),
        "evening_2000": (at(2026, 10, 13, 20), "closed", date(2026, 10, 13)),
        "saturday": (at(2026, 10, 17, 12), "closed", date(2026, 10, 16)),
        "sunday_0200": (at(2026, 10, 18, 2), "closed", date(2026, 10, 16)),
        "monday_0200": (at(2026, 10, 19, 2), "closed", date(2026, 10, 16)),
        "monday_pre_open": (at(2026, 10, 19, 8), "pre_market", date(2026, 10, 16)),
        "thanksgiving": (at(2026, 11, 26, 11), "closed", date(2026, 11, 25)),
        "day_after_0200": (at(2026, 11, 27, 2), "closed", date(2026, 11, 25)),
        "day_after_0800": (at(2026, 11, 27, 8), "pre_market", date(2026, 11, 25)),
        "day_after_1259": (at(2026, 11, 27, 12, 59), "regular", date(2026, 11, 25)),
        "day_after_1300": (at(2026, 11, 27, 13), "after_hours", date(2026, 11, 27)),
        "day_after_1700": (at(2026, 11, 27, 17), "closed", date(2026, 11, 27)),
        "clock_only_2031": (at(2031, 1, 1, 11), "regular", date(2030, 12, 31)),
    }

    @pytest.mark.parametrize(("now", "session", "expected"), CASES.values(), ids=CASES.keys())
    def test_last_completed_session_is_the_cards_expected_date(
        self, now: datetime, session: str, expected: date
    ) -> None:
        """FR-C1: E = last_completed_session(now) equals the card's expected daily date
        (previous_trading_day in session, most_recent_trading_day off-hours) at every instant.
        G: holds before and after 2a; WP2.1's legs measure against E, the card keeps its dates."""
        assert classify_session(now) == session
        card = build_freshness(intraday_df=None, daily_df=None, session=session, now=now)

        assert last_completed_session(now) == expected
        assert card["daily_expected_date"] == expected.isoformat()


def bar(minutes_ago: float, close: float = 101.0) -> dict[str, Any]:
    """technicals' intraday_quote block for a 5m bar ``minutes_ago`` before REGULAR."""
    bar_time = (REGULAR.astimezone(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    return {"close": close, "bar_time": bar_time, "fetched_at": None, "cache_hit": False}


class TestInSessionInfoFallback:
    SESSION_CLOSE = {"close": 100.0, "session_date": "2026-10-12", "fetched_at": None,
                     "cache_hit": False}
    INFO = {"price": 99.0, "quote_time": None}

    @pytest.mark.parametrize(
        ("quote", "rejected"),
        [
            pytest.param(None, "intraday_quote_unavailable", id="unavailable"),
            pytest.param(bar(40), "intraday_quote_stale", id="stale_40m"),
            pytest.param(bar(-10), "intraday_future_timestamp", id="future_10m"),
        ],
    )
    def test_the_info_fallback_drops_the_intraday_reason(
        self, quote: dict[str, Any] | None, rejected: str
    ) -> None:
        """FR-C2: in session a daily close keeps the intraday rejection reason, but with no
        valid close the info fallback reports fallback_reason=no_valid_daily_close.
        G: holds before and after 2a; leg Q keys the info fallback on price_source."""
        _, daily_basis = select_price(now=REGULAR, intraday_quote=quote,
                                      session_close=self.SESSION_CLOSE, info_quote=self.INFO)
        price, info_basis = select_price(now=REGULAR, intraday_quote=quote, session_close=None,
                                         info_quote=self.INFO)

        assert (daily_basis["price_source"], daily_basis["fallback_reason"]) == (
            "daily_close", rejected)
        assert (price, info_basis["price_source"], info_basis["fallback_reason"]) == (
            99.0, "info_quote_fallback", "no_valid_daily_close")

    async def test_analyze_reports_it_in_the_price_provenance(self) -> None:
        """FR-C2: analyze in session, with a stale 5m quote and no completed-session close,
        prices off the info quote with fallback_reason=no_valid_daily_close, not the stale reason.
        G: holds before and after 2a; WP2.1 reads this provenance, it does not rewrite it."""
        technicals = {**dm.STRONG, "session_close": None, "intraday_quote": bar(40)}

        result = await dm.run("S1_hold_or_add_clean", now=REGULAR, technicals=technicals)

        provenance = result["summary"]["price_provenance"]
        assert (provenance["price_source"], provenance["fallback_reason"]) == (
            "info_quote_fallback", "no_valid_daily_close")


class TestTechnicalsInvalidLatestRow:
    @pytest.mark.parametrize(
        ("last_close", "latest_bar_invalid", "current_price"),
        [
            pytest.param(None, True, None, id="R4_nan_forming_row"),
            pytest.param(0.0, False, 0.0, id="R10_zero_close"),
        ],
    )
    async def test_bar_validity_checks_finiteness_only(
        self, monkeypatch: pytest.MonkeyPatch, last_close: float | None,
        latest_bar_invalid: bool, current_price: float | None,
    ) -> None:
        """FR-C3: a NaN last row gives latest_bar_invalid=True and current_price None; a 0.0
        last row gives latest_bar_invalid=False and current_price 0.0. Both give the prior row
        as last_bar_date.
        G: holds before and after 2a; WP2.1's leg H reads daily_history (finite and > 0)."""
        monkeypatch.setattr(tech_mod, "fetch_history", history({
            "1d": daily("2026-10-13", last_close), "5m": five_min(REGULAR, [(2, 101.0)]),
        }))

        result = await tech_mod.technicals("ACME", _now=REGULAR, intraday_quote=True)

        assert (
            result["bar_validity"]["latest_bar_invalid"],
            result["current_price"],
            result["data_provenance"]["price"]["last_bar_date"],
        ) == (latest_bar_invalid, current_price, "2026-10-12")


class TestRiskMetricsEmptyLatestBar:
    @pytest.mark.parametrize(
        ("last_close", "avg_dollar_volume", "gate_ids"),
        [
            pytest.param(101.0, 101_000_000.0, [], id="valid_latest_bar"),
            pytest.param(None, None, ["liquidity_missing"], id="empty_latest_bar"),
        ],
    )
    async def test_an_empty_latest_bar_reads_as_liquidity_missing(
        self, monkeypatch: pytest.MonkeyPatch, last_close: float | None,
        avg_dollar_volume: float | None, gate_ids: list[str],
    ) -> None:
        """FR-C4: an empty latest bar in risk_metrics' frame gives avg_dollar_volume None, so
        check_liquidity fires liquidity_missing.
        G: holds before and after 2a; D-5c (WP2.2) changes the gate's outcome, not its firing."""
        frames = {"1d": daily("2026-10-12", last_close)}
        monkeypatch.setattr(risk_mod, "fetch_history", history(frames))

        result = await risk_mod.risk_metrics("ACME")

        _, missing, blockers = check_liquidity(result)
        assert result["liquidity"]["avg_dollar_volume"] == avg_dollar_volume
        assert (missing, [b["id"] for b in blockers]) == (avg_dollar_volume is None, gate_ids)


# Truth-table rows with a 0.0 5m close; the daily frame ends at E (Mon 2026-10-12).
ZERO_CLOSE_ROWS = [
    pytest.param([(6, 101.0), (1, 0.0)], id="R16_last_row_zero"),
    pytest.param([(1, 0.0)], id="R17_only_bar_zero"),
    pytest.param([(1, -1.0)], id="only_bar_negative"),
]


class TestCardRejectsAZeroClose:
    @pytest.mark.parametrize("bars", ZERO_CLOSE_ROWS)
    def test_build_freshness_blocks_it(self, bars: list[tuple[float, float]]) -> None:
        """FR-C5: the card's freshness reads the raw last 5m row with valid_price (finite and > 0),
        so a 0.0 or negative close a minute old is unverifiable with its own reason, nonpositive_close.
        PD D-12 (flipped by WP2.1)."""
        freshness = build_freshness(intraday_df=five_min(REGULAR, bars),
                                    daily_df=daily("2026-10-12"), session="regular", now=REGULAR)

        assert (freshness["basis"], freshness["stale"], freshness["reason_code"],
                freshness["quote_age_seconds"]) == ("unverifiable", True, "nonpositive_close", None)
        assert [b["id"] for b in freshness_blockers(freshness)] == ["freshness_unverifiable"]

    @pytest.mark.parametrize("bars", ZERO_CLOSE_ROWS)
    async def test_the_orchestrator_never_prices_the_card_at_zero(
        self, monkeypatch: pytest.MonkeyPatch, bars: list[tuple[float, float]]
    ) -> None:
        """FR-C5: analyze_trade_setup never takes a 0.0 or negative 5m close as actionable_price (it
        falls back to the daily close, 101.0), and the card waits for data with freshness_unverifiable
        (nonpositive_close) instead of showing no_setup. PD D-12 (flipped by WP2.1)."""
        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "name": "Acme", "currency": "USD", "quote_type": "EQUITY",
                    "current_price": 101.0}

        async def technicals(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return make_technicals()

        async def risk(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"liquidity": {"avg_dollar_volume": 50_000_000}}

        async def events(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"earnings": {"days_until": 40, "next_date": "2026-11-22"}}

        seen: dict[str, Any] = {}
        real_card = trade_orch.build_trade_setup_card

        def spy_card(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return real_card(**kwargs)

        monkeypatch.setattr(trade_orch, "stock_summary", summary)
        monkeypatch.setattr(trade_orch, "technicals", technicals)
        monkeypatch.setattr(trade_orch, "risk_metrics", risk)
        monkeypatch.setattr(trade_orch, "events_calendar", events)
        monkeypatch.setattr(trade_orch, "fetch_history", history({
            "1d": daily("2026-10-12"), "5m": five_min(REGULAR, bars),
        }))
        monkeypatch.setattr(trade_orch, "build_trade_setup_card", spy_card)

        card = await trade_orch.analyze_trade_setup("ACME", _now=REGULAR)

        assert seen["actionable_price"] == 101.0
        assert (card["action"], [b["id"] for b in card["blockers"]], card["freshness"]["reason_code"]) == (
            "wait_for_data", ["freshness_unverifiable"], "nonpositive_close")
