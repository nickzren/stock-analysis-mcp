"""WP7: one canonical watchlist snapshot and a consistent detect_changes diff (R7.1-R7.4)."""

import copy
import importlib
from typing import Any

import pytest

from stock_analysis.tools.analyze.projection import project_analyze_result
from stock_analysis.utils.normalize import SNAPSHOT_VERSION, build_watchlist_snapshot

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
        "policy_action": {"mid_term": "hold_or_add", "long_term": "accumulate", "valuation_gate": "neutral",
                          "is_unprofitable": False},
        "decision_card": {"action_now": "buy", "hard_gates": {"blocking": [{"id": "earnings_blackout"}]}},
        "risk_summary": {"risk_regime": {"classification": "medium"}, "annualized_volatility": 0.3},
        "events_summary": {"next_catalyst": {"type": "earnings", "status": "available", "date": "2026-10-30",
                                             "days_until": 18, "source": "calendar"}},
        "action_zones": {"current_zone": "accumulate"},
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
    "zone": ({"action_zones__current_zone": "hold_neutral"}, "zone: accumulate -> hold_neutral"),
    "horizon_fit": ({"verdict__horizon_fit__mid_term": "poor"}, "mid_term_fit: good -> poor"),
    "signal_added": ({"signals__bullish": ["above_sma200", "golden_cross", "insider_buying"]}, "+bullish: insider_buying"),
    "signal_removed": ({"signals__bearish": []}, "-bearish: high_put_call_ratio"),
    "price_move": ({"summary__current_price": 107.0}, "price +7.0%"),
}
CODEX_CHANGES = {
    "support": ({"summary__support__currency__status": "unverified"}, "support_currency: supported -> unverified"),
    "valuation": ({"policy_action__valuation_gate": "stretched"}, "valuation_gate: neutral -> stretched"),
    "confidence": ({"verdict__confidence": "low"}, "confidence: medium -> low"),
    "price_source": ({"summary__price_provenance__price_source": "info_quote_fallback"},
                     "price_source: daily_close -> info_quote_fallback"),
    "price_stale": ({"summary__price_provenance__staleness__stale": True}, "price_stale: False -> True"),
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
        ],
        ids=["fetched_at", "cache_hit", "staleness_age", "days_until", "fetch_date"],
    )
    async def test_fetch_times_cache_hits_and_ages_change_nothing(self, detect: Any, changes: dict[str, Any]) -> None:
        response = await detect(copy.deepcopy(BASE), **changes)
        assert (response["material_changes"], response["hash_changed"]) == ([], False)

    async def test_a_version_change_alone_is_not_material(self, detect: Any) -> None:
        previous = {**copy.deepcopy(BASE), "snapshot_version": "1.1.0"}
        response = await detect(previous)
        assert response["material_changes"] == []

    async def test_a_newer_intraday_bar_time_is_not_material(self, detect: Any) -> None:
        response = await detect(copy.deepcopy(BASE), summary__price_provenance__price_as_of="2026-10-10")
        assert response["material_changes"] == []


def legacy_compact(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The 1.1.0 watchlist_snapshot analyze returned before WP7."""
    keys = ("symbol", "price", "market_cap", "sector", "tilt", "confidence", "score", "action_mid_term",
            "action_long_term", "valuation_gate", "is_unprofitable", "decision_action_now", "hard_gate_ids",
            "risk_regime", "volatility_ann", "next_catalyst", "snapshot_as_of_date")
    old = {key: copy.deepcopy(snapshot[key]) for key in keys}
    old["next_catalyst"]["days_until"] = 21
    return {"snapshot_version": "1.1.0", **old, "snapshot_hash": "0123456789abcdef"}


def legacy_enriched(snapshot: dict[str, Any]) -> dict[str, Any]:
    """The 1.1.0 current_snapshot detect_changes returned before WP7."""
    extra = ("zone", "mid_term_fit", "long_term_fit", "bullish_signals", "bearish_signals")
    return {**legacy_compact(snapshot), **{key: copy.deepcopy(snapshot[key]) for key in extra}}


NEW_FIELDS = ["as_of_kind", "price_as_of", "price_source", "price_stale", "support_currency", "support_instrument"]


class TestR72LegacySnapshots:
    async def test_an_old_compact_snapshot_has_no_phantom_changes(self, detect: Any) -> None:
        response = await detect(legacy_compact(BASE))

        assert response["material_changes"] == []
        assert response["not_comparable"] == sorted(
            ["bearish_signals", "bullish_signals", "long_term_fit", "mid_term_fit", "zone", *NEW_FIELDS])

    async def test_an_old_enriched_snapshot_has_no_phantom_changes(self, detect: Any) -> None:
        response = await detect(legacy_enriched(BASE))

        assert response["material_changes"] == []
        assert response["not_comparable"] == NEW_FIELDS

    async def test_real_changes_still_show_against_an_old_snapshot(self, detect: Any) -> None:
        response = await detect(legacy_compact(BASE), decision_card__action_now="avoid")
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

    async def test_a_null_signal_list_that_gains_signals_reports_them(self, detect: Any) -> None:
        previous = {**copy.deepcopy(BASE), "bullish_signals": None}
        response = await detect(previous)
        assert {"+bullish: above_sma200", "+bullish: golden_cross",
                "bullish_signals: None -> ['above_sma200', 'golden_cross']"} <= set(response["material_changes"])

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

    def test_known_empty_lists_stay_empty(self) -> None:
        snapshot = raw_result(decision_card__hard_gates__blocking=[], signals__bearish=[])["watchlist_snapshot"]
        assert (snapshot["hard_gate_ids"], snapshot["bearish_signals"]) == ([], [])


class TestR74Validation:
    @pytest.mark.parametrize(
        ("previous", "problem"),
        [
            ({**BASE, "symbol": "OTHER"}, "symbol"),
            ({k: v for k, v in BASE.items() if k != "symbol"}, "symbol"),
            ({k: v for k, v in BASE.items() if k != "snapshot_version"}, "snapshot_version"),
            ({**BASE, "snapshot_version": "9.0.0"}, "snapshot_version"),
            ({**BASE, "bullish_signals": "golden_cross"}, "bullish_signals"),
            ({**BASE, "hard_gate_ids": [1]}, "hard_gate_ids"),
            ({**BASE, "price": "100"}, "price"),
            ({**BASE, "price": True}, "price"),
            ({**BASE, "price_stale": "no"}, "price_stale"),
            ({**BASE, "next_catalyst": "soon"}, "next_catalyst"),
            ("not a snapshot", "object"),
        ],
        ids=["other_symbol", "missing_symbol", "missing_version", "unknown_version", "signals_not_a_list",
             "gate_not_a_string", "price_string", "price_bool", "stale_not_bool", "catalyst_not_object", "not_dict"],
    )
    async def test_bad_snapshots_get_a_structured_error_before_any_analysis(
        self, detect: Any, previous: Any, problem: str,
    ) -> None:
        response = await detect(previous)

        assert (response["error"], response["error_type"]) == (True, "invalid_snapshot")
        assert any(problem in p for p in response["problems"])
        assert response["_analyzed"] == []

    async def test_symbol_case_and_whitespace_do_not_matter(self, detect: Any) -> None:
        response = await detect({**copy.deepcopy(BASE), "symbol": " acme "})
        assert "error" not in response


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

    async def test_detect_changes_uses_analyze_s_snapshot_as_is(self, detect: Any) -> None:
        response = await detect(None)
        assert response["current_snapshot"] == raw_result()["watchlist_snapshot"]

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
