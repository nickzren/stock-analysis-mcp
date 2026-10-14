"""WP7: one canonical watchlist snapshot and a consistent detect_changes diff (R7.1-R7.4)."""

import copy
import importlib
from typing import Any

import pytest

from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.utils.normalize import SNAPSHOT_VERSION, build_watchlist_snapshot
from tests.support.legacy_snapshot import (
    as_0609fde,
    legacy_compact_snapshot,
    legacy_enriched_snapshot,
)

diff_module = importlib.import_module("stock_analysis.tools.diff_analysis")
orchestrator = importlib.import_module("stock_analysis.tools.analyze.orchestrator")


def raw_result(**changes: Any) -> dict[str, Any]:
    """A minimal analyze result carrying every field the snapshot reads."""
    raw: dict[str, Any] = {
        "symbol": "ACME",
        "summary": {
            "current_price": 100.0, "market_cap": 1e10, "sector": "Industrials",
            "support": {"instrument": {"status": "supported"}, "currency": {"status": "supported"}},
            "price_provenance": {
                "price_source": "daily_close", "price_as_of": "2026-10-09", "as_of_kind": "session_close",
                "fetched_at": "2026-10-10T12:00:00Z", "cache_hit": False,
                "staleness": {"unit": "sessions", "sessions_behind": 0, "stale": False},
            },
        },
        "verdict": {"tilt": "bullish", "confidence": "medium", "score": 0.4,
                    "horizon_fit": {"mid_term": "good", "long_term": "fair"}},
        "diagnostics": {"pre_gate_policy": {"mid_term": "hold_or_add", "long_term": "accumulate",
                                            "inputs": {"zone": "accumulate"}}},
        "decision_card": {"action_now": "buy", "hard_gates": {"blocking": [{"id": "earnings_blackout"}]}},
        "risk_summary": {"risk_regime": {"classification": "medium"}, "annualized_volatility": 0.3},
        "events_summary": {"next_catalyst": {"type": "earnings", "status": "available", "date": "2026-10-30",
                                             "days_until": 18, "source": "calendar"}},
        "action_zones": {"valuation_assessment": {"gate": "neutral", "is_unprofitable": False}},
        "signals": {"bullish": ["above_sma200", "golden_cross"], "bearish": ["high_put_call_ratio"],
                    "neutral": []},
        "data_quality": {"component_freshness": {"price": {"as_of": "2026-10-10T12:00:00Z", "stale": False}}},
    }
    for path, value in changes.items():
        node = raw
        keys = path.split("__")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    raw["watchlist_snapshot"] = build_watchlist_snapshot(raw)
    return raw


@pytest.fixture
def detect(monkeypatch: pytest.MonkeyPatch) -> Any:
    calls: list[str] = []

    async def run(previous: Any, **changes: Any) -> dict[str, Any]:
        current = raw_result(**changes)

        async def fake_analyze(symbol: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(symbol)
            return current

        monkeypatch.setattr(diff_module, "analyze_stock", fake_analyze)
        response: dict[str, Any] = await diff_module.what_changed("ACME", previous_snapshot=previous)
        response["_analyzed"] = list(calls)
        return response

    return run


BASE = raw_result()["watchlist_snapshot"]

R71_CHANGES = {
    "action_now": ({"decision_card__action_now": "wait"}, "action_now: buy -> wait"),
    "gate_added": ({"decision_card__hard_gates__blocking": [{"id": "earnings_blackout"}, {"id": "falling_knife"}]},
                   "+hard_gate: falling_knife"),
    "gate_removed": ({"decision_card__hard_gates__blocking": []}, "-hard_gate: earnings_blackout"),
    "zone": ({"diagnostics__pre_gate_policy__inputs__zone": "hold_neutral"}, "zone: accumulate -> hold_neutral"),
    "horizon_fit": ({"verdict__horizon_fit__mid_term": "poor"}, "mid_term_fit: good -> poor"),
    "signal_added": ({"signals__bullish": ["above_sma200", "golden_cross", "insider_buying"]}, "+bullish: insider_buying"),
    "signal_removed": ({"signals__bearish": []}, "-bearish: high_put_call_ratio"),
    "price_move": ({"summary__current_price": 107.0}, "price +7.0%"),
}
CODEX_CHANGES = {
    "support": ({"summary__support__currency__status": "unverified"}, "support_currency: supported -> unverified"),
    "valuation": ({"action_zones__valuation_assessment__gate": "stretched"}, "valuation_gate: neutral -> stretched"),
    "confidence": ({"verdict__confidence": "low"}, "confidence: medium -> low"),
    "price_source": ({"summary__price_provenance__price_source": "info_quote_fallback"},
                     "price_source: daily_close -> info_quote_fallback"),
}


class TestR71MaterialChanges:
    async def test_unchanged_data_has_no_material_change_and_the_same_hash(self, detect: Any) -> None:
        response = await detect(copy.deepcopy(BASE))

        assert response["material_changes"] == []
        assert response["hash_changed"] is False
        assert response["not_comparable"] == []

    @pytest.mark.parametrize("case", [*R71_CHANGES, *CODEX_CHANGES])
    async def test_each_decision_change_is_material_and_changes_the_hash(self, detect: Any, case: str) -> None:
        changes, line = {**R71_CHANGES, **CODEX_CHANGES}[case]

        response = await detect(copy.deepcopy(BASE), **changes)

        assert line in response["material_changes"]
        assert response["hash_changed"] is True
        assert response["current_snapshot"]["snapshot_hash"] != BASE["snapshot_hash"]

    async def test_a_small_price_move_changes_the_hash_but_is_not_material(self, detect: Any) -> None:
        response = await detect(copy.deepcopy(BASE), summary__current_price=101.0)
        assert (response["material_changes"], response["hash_changed"]) == ([], True)

    @pytest.mark.parametrize(
        "changes",
        [
            {"summary__price_provenance__fetched_at": "2026-10-11T09:00:00Z"},
            {"summary__price_provenance__cache_hit": True},
            {"summary__price_provenance__staleness__age_hours": 5.0},
            {"events_summary__next_catalyst__days_until": 17},
            {"data_quality__component_freshness__price__as_of": "2026-10-11T09:00:00Z"},
            {"summary__price_provenance__staleness__stale": True},
            {"action_zones__valuation_assessment__is_unprofitable": None},
        ],
        ids=["fetched_at", "cache_hit", "staleness_age", "days_until", "fetch_date", "stale_by_age",
             "unprofitable_lossy"],
    )
    async def test_fetch_times_cache_hits_and_ages_change_nothing(self, detect: Any, changes: dict[str, Any]) -> None:
        response = await detect(copy.deepcopy(BASE), **changes)
        assert response["material_changes"] == []
        assert response["hash_changed"] is False  # is_unprofitable reads False as None (plan §7a)

    async def test_a_version_change_alone_is_not_material(self, detect: Any) -> None:
        previous = {**copy.deepcopy(BASE), "snapshot_version": "1.1.0"}
        response = await detect(previous)
        assert response["material_changes"] == []

    async def test_crossing_the_close_from_bar_to_daily_close_is_not_material(self, detect: Any) -> None:
        previous = {**copy.deepcopy(BASE), "price_source": "intraday_quote"}
        response = await detect(previous)
        assert response["material_changes"] == []
        assert response["changes"]["price_source"] == {
            "previous": "intraday_quote", "current": "daily_close", "changed": True, "material": False}

    async def test_a_newer_intraday_bar_time_is_not_material(self, detect: Any) -> None:
        response = await detect(copy.deepcopy(BASE), summary__price_provenance__price_as_of="2026-10-10")
        assert response["material_changes"] == []


WINDOW = {"start": "2026-10-12", "end": "2026-10-16"}


class TestCatalystHash:
    async def test_a_rescheduled_date_changes_the_hash_without_an_alert(self, detect: Any) -> None:
        response = await detect(copy.deepcopy(BASE), events_summary__next_catalyst__date="2026-11-10")
        assert (response["hash_changed"], response["material_changes"]) == (True, [])

    async def test_losing_the_date_changes_the_hash(self, detect: Any) -> None:
        unavailable = {"type": "earnings", "status": "unavailable", "reason": "calendar_data_unavailable"}
        response = await detect(copy.deepcopy(BASE), events_summary__next_catalyst=unavailable)
        assert response["hash_changed"] is True

    async def test_inside_a_window_the_advancing_date_is_not_data(self, detect: Any) -> None:
        catalyst = {"type": "earnings", "status": "available", "date": "2026-10-13", "days_until": 0,
                    "source": "calendar", "window": WINDOW}
        previous = raw_result(events_summary__next_catalyst=catalyst)["watchlist_snapshot"]
        response = await detect(previous, events_summary__next_catalyst={**catalyst, "date": "2026-10-14"})
        assert (response["hash_changed"], response["material_changes"]) == (False, [])
        assert "date" not in response["current_snapshot"]["next_catalyst"]

    async def test_moving_the_window_changes_the_hash(self, detect: Any) -> None:
        catalyst = {"type": "earnings", "status": "available", "date": "2026-10-13", "source": "calendar",
                    "window": WINDOW}
        previous = raw_result(events_summary__next_catalyst=catalyst)["watchlist_snapshot"]
        moved = {**catalyst, "window": {"start": "2026-10-19", "end": "2026-10-23"}}
        response = await detect(previous, events_summary__next_catalyst=moved)
        assert response["hash_changed"] is True

    async def test_analyze_reports_the_window_behind_a_derived_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def events(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "earnings": {
                "next_date": "2026-10-13", "next_date_status": "available", "next_date_source": "calendar",
                "days_until": 0, "next_window": {"source": "calendar", **WINDOW, "window_basis": "assumed"}}}

        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "events_calendar", events)
        for name in ("stock_summary", "technicals", "fundamentals_snapshot", "risk_metrics", "stock_news",
                     "ownership_analysis", "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty)
        result = await orchestrator.analyze_stock("ACME")

        assert result["events_summary"]["next_catalyst"]["window"] == WINDOW
        assert result["watchlist_snapshot"]["next_catalyst"] == {
            "type": "earnings", "status": "available", "source": "calendar", "window": WINDOW}


class TestDiffEdges:
    @pytest.mark.parametrize(
        ("changes", "material"),
        [({"summary__current_price": 1.47}, False), ({"summary__current_price": 1.4701}, True),
         ({"verdict__score": 0.40}, False), ({"verdict__score": 0.4001}, True)],
        ids=["price_exactly_5pct", "price_over_5pct", "score_exactly_0_10", "score_over_0_10"],
    )
    async def test_exact_thresholds_are_not_crossed_by_float_error(
        self, detect: Any, changes: dict[str, Any], material: bool,
    ) -> None:
        previous = {**copy.deepcopy(BASE), "price": 1.40, "score": 0.30}
        response = await detect(previous, **{"summary__current_price": 1.40, "verdict__score": 0.30, **changes})
        assert bool(response["material_changes"]) is material

    @pytest.mark.parametrize(
        ("previous", "changes", "line", "sentence"),
        [
            ({"price": None}, {}, "price: None -> 100.0", "Price None -> 100.0."),
            ({}, {"summary__current_price": None}, "price: 100.0 -> None", "Price 100.0 -> None."),
            ({"score": None}, {}, "score: None -> 0.4", "Score None -> 0.4."),
        ],
        ids=["price_appears", "price_disappears", "score_appears"],
    )
    async def test_a_value_appearing_or_disappearing_is_reported(
        self, detect: Any, previous: dict[str, Any], changes: dict[str, Any], line: str, sentence: str,
    ) -> None:
        response = await detect({**copy.deepcopy(BASE), **previous}, **changes)
        assert line in response["material_changes"]
        assert sentence in response["summary"]

    async def test_a_move_just_over_the_threshold_is_material(self, detect: Any) -> None:
        response = await detect(copy.deepcopy(BASE), summary__current_price=105.004)
        assert response["changes"]["price"]["material"] is True
        assert "price +5.0%" in response["material_changes"]

    async def test_nothing_in_common_is_not_reported_as_no_change(self, detect: Any) -> None:
        response = await detect({"snapshot_version": "2.0.0", "symbol": "ACME"})
        assert response["material_changes"] == []
        assert response["summary"].startswith("ACME: Nothing to compare")

    async def test_hashes_from_another_version_are_not_compared(self, detect: Any) -> None:
        response = await detect({**copy.deepcopy(BASE), "snapshot_version": "1.1.0"})
        assert response["hash_changed"] is None

    async def test_signal_changes_carry_a_changed_flag(self, detect: Any) -> None:
        response = await detect({**copy.deepcopy(BASE), "bearish_signals": None}, signals__bearish=[])
        signals = response["changes"]["signals"]
        assert (signals["bearish_changed"], signals["bullish_changed"]) == (True, False)


# Fields only a 2.0.0 snapshot carries, and the legacy signal lists, which are ambiguous.
NEW_FIELDS = ["as_of_kind", "price_as_of", "price_source", "price_stale", "support_currency", "support_instrument"]
LEGACY_SIGNALS = ["bearish_signals", "bullish_signals"]


class TestR72LegacySnapshots:
    """Previous snapshots made by the frozen 1.1.0 producers (tests/support/legacy_snapshot.py)."""

    async def test_an_old_compact_snapshot_has_no_phantom_changes(self, detect: Any) -> None:
        response = await detect(legacy_compact_snapshot(as_0609fde(raw_result())))

        assert response["material_changes"] == []
        assert response["not_comparable"] == sorted(
            [*LEGACY_SIGNALS, "long_term_fit", "mid_term_fit", "zone", *NEW_FIELDS])
        assert response["hash_changed"] is None

    async def test_an_old_enriched_snapshot_has_no_phantom_changes(self, detect: Any) -> None:
        response = await detect(legacy_enriched_snapshot(as_0609fde(raw_result())))

        assert response["material_changes"] == []
        assert response["not_comparable"] == sorted([*LEGACY_SIGNALS, *NEW_FIELDS])

    async def test_a_failed_source_tool_does_not_turn_old_signal_lists_into_alerts(self, detect: Any) -> None:
        failed = {"tool_failures": [{"tool": "ownership_analysis", "error": "data_unavailable"}],
                  "component_freshness": {"price": {"as_of": "2026-10-10T12:00:00Z", "stale": False}}}
        previous = legacy_enriched_snapshot(as_0609fde(raw_result(data_quality=failed)))

        response = await detect(previous, data_quality=failed)

        assert previous["bullish_signals"] == ["above_sma200", "golden_cross"]  # the old producer's list
        assert response["current_snapshot"]["bullish_signals"] is None
        assert response["material_changes"] == []

    async def test_a_1_0_0_snapshot_without_decision_fields_compares_cleanly(self, detect: Any) -> None:
        old = legacy_compact_snapshot(as_0609fde(raw_result()))
        del old["decision_action_now"], old["hard_gate_ids"]
        response = await detect({**old, "snapshot_version": "1.0.0"})

        assert response["material_changes"] == []
        assert {"decision_action_now", "hard_gate_ids"} <= set(response["not_comparable"])

    async def test_real_changes_still_show_against_an_old_snapshot(self, detect: Any) -> None:
        response = await detect(legacy_compact_snapshot(as_0609fde(raw_result())), decision_card__action_now="avoid")
        assert response["material_changes"] == ["action_now: buy -> avoid"]


class TestR73AbsentVersusNull:
    async def test_a_null_that_becomes_a_value_is_a_change(self, detect: Any) -> None:
        previous = {**copy.deepcopy(BASE), "zone": None}
        response = await detect(previous)
        assert "zone: None -> accumulate" in response["material_changes"]

    async def test_an_absent_field_is_not_comparable(self, detect: Any) -> None:
        previous = copy.deepcopy(BASE)
        del previous["zone"]
        response = await detect(previous)
        assert response["material_changes"] == []
        assert response["not_comparable"] == ["zone"]
        assert "action_zone" not in response["changes"]

    async def test_a_null_signal_list_that_becomes_known_is_one_change_not_new_signals(self, detect: Any) -> None:
        previous = {**copy.deepcopy(BASE), "bullish_signals": None}
        response = await detect(previous)
        assert response["material_changes"] == ["bullish_signals: None -> ['above_sma200', 'golden_cross']"]
        assert response["changes"]["signals"]["bullish_changed"] is True

    async def test_a_list_that_becomes_unknown_removes_nothing(self, detect: Any) -> None:
        failed = {"tool_failures": [{"tool": "technicals", "error": "TimeoutError"}],
                  "component_freshness": {"price": {"as_of": "2026-10-10T12:00:00Z", "stale": False}}}
        response = await detect(copy.deepcopy(BASE), data_quality=failed)
        assert not [line for line in response["material_changes"] if line.startswith(("-bullish", "-bearish"))]
        assert "bullish_signals: ['above_sma200', 'golden_cross'] -> None" in response["material_changes"]

    async def test_a_null_list_that_becomes_known_empty_is_a_change(self, detect: Any) -> None:
        previous = {**copy.deepcopy(BASE), "hard_gate_ids": None}
        response = await detect(previous, decision_card__hard_gates__blocking=[])
        assert response["material_changes"] == ["hard_gate_ids: None -> []"]
        assert response["changes"]["hard_gates"]["changed"] is True

    async def test_a_known_list_that_becomes_null_is_a_change(self, detect: Any) -> None:
        response = await detect(copy.deepcopy(BASE), signals__bearish=None)
        assert "bearish_signals: ['high_put_call_ratio'] -> None" in response["material_changes"]

    def test_the_snapshot_keeps_unknown_lists_null_before_normalization(self) -> None:
        raw = raw_result(decision_card=None, signals={"bullish": ["golden_cross"]})
        snapshot = raw["watchlist_snapshot"]
        assert snapshot["hard_gate_ids"] is None
        assert (snapshot["bullish_signals"], snapshot["bearish_signals"]) == (["golden_cross"], None)

    @pytest.mark.parametrize(
        ("failure", "known"),
        [({"tool": "technicals", "error": "data_unavailable"}, False),
         ({"tool": "options_signals", "error": "TimeoutError"}, False),
         ({"tool": "options_signals", "error": "no_options"}, True),
         ({"tool": "stock_news", "error": "news_source_failed"}, True)],
        ids=["technicals_failed", "options_failed", "no_options_is_known", "news_feeds_no_signal"],
    )
    def test_signals_are_unknown_when_a_source_tool_failed(self, failure: dict[str, Any], known: bool) -> None:
        snapshot = raw_result(data_quality={"tool_failures": [failure]})["watchlist_snapshot"]
        assert (snapshot["bullish_signals"] is not None) is known
        assert (snapshot["bearish_signals"] is not None) is known

    @pytest.mark.parametrize(("latest_bar_invalid", "known"), [(True, False), (False, True)])
    def test_an_empty_latest_daily_bar_makes_the_signals_unknown(self, latest_bar_invalid: bool, known: bool) -> None:
        """That day the SMA rules have no value, so their signals are unknown, not removed."""
        data_quality = {"bar_validity": {"latest_bar_invalid": latest_bar_invalid}}
        snapshot = raw_result(data_quality=data_quality)["watchlist_snapshot"]
        assert (snapshot["bullish_signals"] is not None) is known
        assert (snapshot["bearish_signals"] is not None) is known

    @pytest.mark.parametrize(
        ("endpoints", "known"), [(["return_3m"], False), (["return_1m", "return_1y"], True), ([], True)]
    )
    def test_a_3m_return_lost_to_an_invalid_bar_makes_the_signals_unknown(
        self, endpoints: list[str], known: bool
    ) -> None:
        """return_3m feeds strong/weak_3m_momentum; the other returns feed no signal."""
        data_quality = {"bar_validity": {"latest_bar_invalid": False, "invalid_return_endpoints": endpoints}}
        snapshot = raw_result(data_quality=data_quality)["watchlist_snapshot"]
        assert (snapshot["bullish_signals"] is not None) is known
        assert (snapshot["bearish_signals"] is not None) is known

    @pytest.mark.parametrize(
        ("warning", "known"),
        [("insider_transactions_error", False), ("insider_transactions_columns_unrecognized", False),
         ("insider_transactions_unavailable", True), ("institutional_holders_error", True)],
    )
    def test_a_failed_insider_read_makes_the_signals_unknown(self, warning: str, known: bool) -> None:
        snapshot = raw_result(ownership_flow={"warnings": [warning]})["watchlist_snapshot"]
        assert (snapshot["bullish_signals"] is not None) is known

    async def test_a_failed_insider_read_is_not_insider_buying_stopping(self, detect: Any) -> None:
        with_insider = ["above_sma200", "golden_cross", "insider_buying"]
        previous = raw_result(signals__bullish=with_insider)["watchlist_snapshot"]
        # The 503 leaves insider sentiment neutral, so analyze drops insider_buying.
        response = await detect(previous, ownership_flow={"warnings": ["insider_transactions_error"]})
        assert "-bullish: insider_buying" not in response["material_changes"]
        assert response["current_snapshot"]["bullish_signals"] is None

    def test_known_empty_lists_stay_empty(self) -> None:
        snapshot = raw_result(decision_card__hard_gates__blocking=[], signals__bearish=[])["watchlist_snapshot"]
        assert (snapshot["hard_gate_ids"], snapshot["bearish_signals"]) == ([], [])


class TestR74Validation:
    @pytest.mark.parametrize(
        ("previous", "problem"),
        [
            ({k: v for k, v in BASE.items() if k != "symbol"}, "symbol"),
            ({**BASE, "symbol": 42}, "symbol"),
            ({k: v for k, v in BASE.items() if k != "snapshot_version"}, "snapshot_version"),
            ({**BASE, "snapshot_version": "9.0.0"}, "snapshot_version"),
            ({**BASE, "snapshot_version": ["2.0.0"]}, "snapshot_version"),
            ({**BASE, "snapshot_version": {"v": 2}}, "snapshot_version"),
            ({**BASE, "price": 10**400}, "price"),
            ({**BASE, "bullish_signals": "golden_cross"}, "bullish_signals"),
            ({**BASE, "hard_gate_ids": [1]}, "hard_gate_ids"),
            ({**BASE, "price": "100"}, "price"),
            ({**BASE, "price": True}, "price"),
            ({**BASE, "price_stale": "no"}, "price_stale"),
            ({**BASE, "next_catalyst": "soon"}, "next_catalyst"),
            ({**BASE, "next_catalyst": {"type": ["earnings"], "date": "2026-10-30"}}, "next_catalyst.type"),
            ({**BASE, "next_catalyst": {"type": "earnings", "date": 20261030}}, "next_catalyst.date"),
            ({**BASE, "next_catalyst": {"type": "earnings", "window": {"start": [], "end": "2026-10-16"}}},
             "next_catalyst.window.start"),
            ({**BASE, "next_catalyst": {"type": "earnings", "window": {"start": "2026-10-12", "end": 23}}},
             "next_catalyst.window.end"),
            ("not a snapshot", "object"),
        ],
        ids=["missing_symbol", "symbol_not_a_string", "missing_version", "unknown_version", "version_list",
             "version_dict", "price_too_big", "signals_not_a_list",
             "gate_not_a_string", "price_string", "price_bool", "stale_not_bool", "catalyst_not_object",
             "catalyst_type_list", "catalyst_date_int", "window_start_list", "window_end_int", "not_dict"],
    )
    async def test_bad_snapshots_get_a_structured_error_before_any_analysis(
        self, detect: Any, previous: Any, problem: str,
    ) -> None:
        response = await detect(previous)

        assert (response["error"], response["error_type"]) == (True, "invalid_snapshot")
        assert any(problem in p for p in response["problems"])
        assert response["_analyzed"] == []

    async def test_another_symbol_s_snapshot_is_rejected_against_the_resolved_ticker(self, detect: Any) -> None:
        response = await detect({**copy.deepcopy(BASE), "symbol": "OTHER"})

        assert (response["error"], response["error_type"]) == (True, "invalid_snapshot")
        assert response["problems"] == ["symbol 'OTHER' does not match ACME"]

    async def test_a_company_name_round_trips_its_own_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        current = raw_result()

        async def fake_analyze(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return current  # analyze resolves "Acme Corp" to ACME

        monkeypatch.setattr(diff_module, "analyze_stock", fake_analyze)
        response = await diff_module.what_changed("Acme Corp", previous_snapshot=copy.deepcopy(BASE))

        assert "error" not in response
        assert response["material_changes"] == []

    @pytest.mark.parametrize("window", [{}, {"start": None, "end": None}, {"start": "2026-10-12"}],
                             ids=["empty", "null_bounds", "absent_end"])
    async def test_absent_or_null_window_bounds_stay_compatible(self, detect: Any, window: dict[str, Any]) -> None:
        catalyst = {"type": "earnings", "status": "available", "source": "calendar", "window": window}
        response = await detect({**copy.deepcopy(BASE), "next_catalyst": catalyst})
        assert "error" not in response

    async def test_symbol_case_and_whitespace_do_not_matter(self, detect: Any) -> None:
        response = await detect({**copy.deepcopy(BASE), "symbol": " acme "})
        assert "error" not in response


class TestOddSections:
    @pytest.mark.parametrize(
        "changes",
        [{"summary__price_provenance": "x"}, {"summary__price_provenance__staleness": "x"},
         {"summary__support": "x"}, {"summary__support__instrument": "x"}, {"verdict__horizon_fit": "x"},
         {"summary": None}, {"events_summary": None}, {"risk_summary": None}, {"risk_summary__risk_regime": "x"},
         {"action_zones": None}, {"data_quality": "x"}],
    )
    def test_a_section_that_is_not_a_dict_reads_as_empty(self, changes: dict[str, Any]) -> None:
        snapshot = raw_result(**changes)["watchlist_snapshot"]
        assert snapshot["snapshot_version"] == "2.0.0"


class TestOneSnapshot:
    def test_the_snapshot_is_versioned_and_complete(self) -> None:
        snapshot = BASE
        assert snapshot["snapshot_version"] == SNAPSHOT_VERSION == "2.0.0"
        assert (snapshot["zone"], snapshot["mid_term_fit"], snapshot["long_term_fit"]) == (
            "accumulate", "good", "fair")
        assert (snapshot["support_instrument"], snapshot["support_currency"]) == ("supported", "supported")
        assert (snapshot["price_source"], snapshot["price_as_of"], snapshot["as_of_kind"], snapshot["price_stale"]) == (
            "daily_close", "2026-10-09", "session_close", False)
        assert "days_until" not in snapshot["next_catalyst"]
        assert not {"fetched_at", "cache_hit"} & set(snapshot)

    async def test_detect_changes_uses_analyze_s_snapshot_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        current = raw_result()
        current["watchlist_snapshot"] = {**current["watchlist_snapshot"], "marker": "from analyze"}

        async def fake_analyze(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return current

        monkeypatch.setattr(diff_module, "analyze_stock", fake_analyze)
        response = await diff_module.what_changed("ACME")

        assert response["current_snapshot"] is current["watchlist_snapshot"]

    async def test_every_tier_carries_the_identical_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def summary(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol, "current_price": 100.0, "currency": "USD", "quote_type": "EQUITY"}

        async def empty(symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {"symbol": symbol}

        monkeypatch.setattr(orchestrator, "stock_summary", summary)
        for name in ("technicals", "fundamentals_snapshot", "risk_metrics", "events_calendar", "stock_news",
                     "ownership_analysis", "options_signals"):
            monkeypatch.setattr(orchestrator, name, empty)
        result = await orchestrator.analyze_stock("ACME")

        snapshots = [project_analyze_result(result, detail)["watchlist_snapshot"]
                     for detail in ("decision", "standard", "full")]
        assert snapshots[0] == snapshots[1] == snapshots[2] == result["watchlist_snapshot"]
        assert snapshots[0]["snapshot_version"] == "2.0.0" and "zone" in snapshots[0]
