"""Step 2 WP2.3: the closed field-class registry (plan §7 "Field classes") and the lints it drives.

Every leaf of an analyze result at every tier matches exactly one FIELD_REGISTRY row (OC-RF1). Classes:
A actionable (from decide(); declares its horizon), S sizing (subset of A), C conditional, D diagnostic,
N our narrative, P pre-gate (full tier only, under diagnostics), T third-party, E static config echo.
Globs are root-anchored: `prefix.**` matches `prefix` itself (a null or empty leaf) and everything under
it; `*` is one key; `[*]` is a list item. The most specific row wins: more literal segments, then an exact
row over a `.**` row. watchlist_snapshot (snapshot 3.0.0, plan §8) is registered like every other root.
The WP2.5 live smoke calls registry_violations() on live payloads.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from stock_analysis.tools.analyze.decision import ENTRY_ACTIONS, ENTRY_CODES, FULL_ENTRY_CODES

A, S, C, D, N, P, T, E = "A", "S", "C", "D", "N", "P", "T", "E"
CLASSES = (A, S, C, D, N, P, T, E)
MID, HOLDER, LONG = "mid_term_new_money", "holder", "long_term"
ALL = frozenset({"decision", "standard", "full"})
SF = frozenset({"standard", "full"})
FULL = frozenset({"full"})
STANDARD = frozenset({"standard"})


@dataclass(frozen=True)
class Field:
    glob: str
    cls: str
    horizon: str | None = None
    tiers: frozenset[str] = ALL


R = Field
FIELD_REGISTRY: tuple[Field, ...] = (
    # root
    R("symbol", D), R("meta.**", D), R("summary.**", D), R("data_quality.**", D),
    R("error", D), R("error_type", D), R("message", D),
    R("data_provenance.**", D, tiers=FULL), R("company_profile.**", D, tiers=FULL),
    R("company_profile.description", T, tiers=FULL),
    # decision_card: closed key by key (plan §2)
    R("decision_card.horizon", D),
    R("decision_card.new_money.**", A, MID), R("decision_card.new_money.earliest_reassessment.**", C, MID),
    R("decision_card.if_held.**", A, HOLDER),
    R("decision_card.long_term.**", A, LONG),
    R("decision_card.long_term.new_money.earliest_reassessment.**", C, LONG),
    R("decision_card.long_term.new_money.conditions.**", C, LONG),
    R("decision_card.hard_gates.**", D), R("decision_card.hard_gates.blocking.**", A, MID),
    R("decision_card.action_now", A, MID),
    R("decision_card.why.**", A, MID), R("decision_card.exit_triggers.**", C, HOLDER),
    R("decision_card.data_flags.**", D),
    R("decision_card.sizing.**", S, MID), R("decision_card.entry.**", S, MID),
    R("decision_card.exit.**", A, HOLDER), R("decision_card.conditions.**", C, MID),
    R("decision_card.next_review.**", C, MID), R("decision_card.confidence", D),
    # pre-gate (plan §2, §7)
    R("diagnostics.**", P, tiers=FULL),
    # verdict and summaries
    R("verdict.**", D),
    R("verdict.confidence_path.upgrade_if.**", C, tiers=FULL), R("verdict.confidence_path.downgrade_if.**", C, tiers=FULL),
    R("executive_summary", N, tiers=SF),
    R("section_summaries.**", D, tiers=SF),
    R("section_summaries.analyst_view", T, tiers=SF), R("section_summaries.news", T, tiers=SF),
    # action zones
    R("action_zones.price_zone", D, tiers=SF), R("action_zones.levels.**", D, tiers=SF),
    R("action_zones.stop_calculation.**", D, tiers=SF), R("action_zones.valuation_assessment.**", D, tiers=SF),
    R("action_zones.position_sizing_range.**", S, MID, SF), R("action_zones.position_sizing_range.rationale", D, tiers=SF),
    R("action_zones.basis.**", D, tiers=FULL), R("action_zones.distance_to_levels.**", D, tiers=FULL),
    R("action_zones.price_vs_levels.**", D, tiers=FULL), R("action_zones.distance_labels.**", D, tiers=FULL),
    R("action_zones.level_vs_current_labels.**", D, tiers=FULL), R("action_zones.zone_warnings.**", D, tiers=FULL),
    R("action_zones.method", D, tiers=FULL),
    # decision modes: closed key by key
    R("decision_modes.*.label", E, tiers=FULL), R("decision_modes.*.description", E, tiers=FULL),
    R("decision_modes.core.action", A, LONG, FULL), R("decision_modes.balanced.action", A, MID, FULL),
    R("decision_modes.speculative.action", A, MID, FULL), R("decision_modes.balanced.long_term_view", A, LONG, FULL),
    R("decision_modes.core.suitability", D, tiers=FULL), R("decision_modes.balanced.suitability", A, MID, FULL),
    R("decision_modes.speculative.suitability", A, MID, FULL),
    R("decision_modes.*.starter_position.**", S, MID, FULL), R("decision_modes.*.max_position.**", S, MID, FULL),
    R("decision_modes.speculative.capital_at_risk", E, tiers=FULL),
    R("decision_modes.core.required_before_full_size.**", C, LONG, FULL),
    R("decision_modes.balanced.add_on_only_if.**", C, MID, FULL),
    R("decision_modes.speculative.add_on_only_if.**", C, MID, FULL),
    R("decision_modes.balanced.why.**", A, MID, FULL), R("decision_modes.core.why.**", D, tiers=FULL),
    R("decision_modes.speculative.upside_case", D, tiers=FULL), R("decision_modes.speculative.failure_case", D, tiers=FULL),
    R("decision_modes.speculative.next_90d_catalysts.**", D, tiers=FULL),
    R("decision_modes.speculative.risk_flags.**", D, tiers=FULL),
    R("decision_modes.*.capped_by.**", D, tiers=FULL),
    # dislocation
    R("dislocation_framework.**", D, tiers=FULL), R("dislocation_framework.action.**", A, MID, FULL),
    R("dislocation_framework.action.core", A, LONG, FULL),
    R("dislocation_framework.action.buy_only_if.**", C, MID, FULL),
    R("dislocation_framework.action.add_only_if.**", C, MID, FULL),
    R("dislocation_framework.action.sell_without_attachment_if.**", C, HOLDER, FULL),
    R("dislocation_framework.action.capped_by.**", D, tiers=FULL),
    R("dislocation_framework.thesis_integrity.**", C, HOLDER, FULL),
    R("dislocation_check.**", D, tiers=STANDARD), R("dislocation_check.decision.**", A, MID, STANDARD),
    # dip
    R("dip_assessment.**", D, tiers=FULL),
    R("dip_assessment.assessment.recommendation", A, MID, FULL), R("dip_assessment.assessment.dip_quality", A, MID, FULL),
    R("dip_assessment.assessment.add_only_if.**", C, MID, FULL),
    R("dip_assessment.assessment.capped_by.**", D, tiers=FULL),
    R("dip_assessment.entry_timing.signals[*].action", A, MID, FULL),
    R("dip_assessment.entry_timing.signals[*].capped_by.**", D, tiers=FULL),
    R("dip_assessment.entry_timing.wait_for.**", C, MID, FULL),
    # decision context
    R("decision_context.**", D, tiers=FULL),
    R("decision_context.*.bullish_if.**", C, tiers=FULL), R("decision_context.*.bearish_if.**", C, tiers=FULL),
    R("decision_context.thesis_checkpoints.thesis_stop_triggers.**", C, HOLDER, FULL),
    R("decision_context.news.**", T, tiers=FULL),
    # other diagnostics
    R("technicals_summary.**", D, tiers=FULL), R("fundamentals_summary.**", D, tiers=SF),
    R("risk_summary.**", D, tiers=SF), R("events_summary.**", D, tiers=SF), R("signals.**", D, tiers=FULL),
    R("market_context.**", D, tiers=FULL), R("relative_performance.**", D, tiers=SF),
    R("sector_comparison.**", D, tiers=SF), R("quality.**", D, tiers=FULL), R("valuation_context.**", D, tiers=FULL),
    # watchlist_snapshot 3.0.0 (plan §8): closed key by key; the canonical actions and the fired gate ids are A
    R("watchlist_snapshot.new_money_action", A, MID), R("watchlist_snapshot.if_held_action", A, HOLDER),
    R("watchlist_snapshot.long_term_action", A, LONG), R("watchlist_snapshot.hard_gate_ids.**", A, MID),
    *(R(f"watchlist_snapshot.{key}", D) for key in (
        "snapshot_version", "symbol", "snapshot_hash", "snapshot_as_of_date", "price", "market_cap", "sector", "tilt",
        "confidence", "score", "mid_term_fit", "long_term_fit", "valuation_gate", "is_unprofitable", "price_zone",
        "risk_regime", "volatility_ann", "support_instrument", "support_currency", "price_source", "price_as_of",
        "as_of_kind", "price_stale")),
    *(R(f"watchlist_snapshot.{key}.**", D) for key in ("bullish_signals", "bearish_signals", "next_catalyst")),
    # third party (plan §10 WP2.3 carry-forward)
    R("analyst_coverage.**", T, tiers=SF), R("fundamentals_summary.analyst_estimates.**", T, tiers=SF),
    R("news_summary.**", T, tiers=SF), R("ownership.**", T, tiers=SF), R("short_interest.**", T, tiers=SF),
    R("governance.**", T, tiers=FULL), R("ownership_flow.**", T, tiers=FULL), R("options_signals.**", T, tiers=FULL),
)
UNEXERCISED_ROWS: dict[str, str] = {}  # rows no matrix run reaches, with the reason (OC-RF1b asserts the set)


def _pattern(glob: str) -> re.Pattern[str]:
    subtree = glob.endswith(".**")
    base = glob[:-3] if subtree else glob
    pieces = []
    for piece in re.split(r"(\[\*\])", base):
        if piece == "[*]":
            pieces.append(r"\[\*\]")
        else:
            pieces.append(r"\.".join(r"[^.\[\]]+" if seg == "*" else re.escape(seg) for seg in piece.split(".")))
    return re.compile("".join(pieces) + (r"(?:(?:\.|\[\*\]).*)?" if subtree else ""))


def _specificity(glob: str) -> tuple[int, bool]:
    base = glob[:-3] if glob.endswith(".**") else glob
    return sum(seg not in ("", "*") for seg in base.replace("[*]", "").split(".")), not glob.endswith(".**")


_ROWS = [(field, _pattern(field.glob), _specificity(field.glob)) for field in FIELD_REGISTRY]


class UnregisteredPathError(LookupError):
    pass


class AmbiguousPathError(LookupError):
    pass


def classify(path: str) -> Field:
    hits = [(spec, field) for field, pattern, spec in _ROWS if pattern.fullmatch(path)]
    if not hits:
        raise UnregisteredPathError(path)
    best = max(spec for spec, _ in hits)
    winners = [field for spec, field in hits if spec == best]
    if len(winners) > 1:
        raise AmbiguousPathError(f"{path}: {[f.glob for f in winners]}")
    return winners[0]


def leaves(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Every (path, leaf); list indices collapse to [*]; an empty dict or list is itself a leaf."""
    if isinstance(node, dict) and node:
        return [x for key, value in node.items() for x in leaves(value, f"{path}.{key}" if path else key)]
    if isinstance(node, list) and node:
        return [x for value in node for x in leaves(value, f"{path}[*]")]
    return [(path, node)]


def classified(payload: Mapping[str, Any]) -> list[tuple[str, Any, Field]]:
    """(path, value, row) for every registered leaf; OC-RF1 reports the rest."""
    out = []
    for path, value in leaves(dict(payload)):
        try:
            out.append((path, value, classify(path)))
        except LookupError:
            continue
    return out


def registry_violations(payload: Mapping[str, Any], tier: str) -> list[str]:
    """OC-RF1 for one projected payload: unregistered, ambiguous and wrong-tier leaves."""
    problems = []
    for path, _ in leaves(dict(payload)):
        try:
            field = classify(path)
        except LookupError as error:
            problems.append(f"{type(error).__name__}: {error}")
            continue
        if tier not in field.tiers:
            problems.append(f"{path} at {tier} (row {field.glob}: {sorted(field.tiers)})")
    return problems


ADVICE_PHRASES = (
    "size ~", "hold or add", "accumulate", "small starter position", "strong buy", "consider entry",
    "favorable for dip buying", "good entry point", "acceptable entry", "starter position acceptable",
    "small position only", "smaller position", "pullback to buy", "standard sizing", "size conservatively",
    "minimal position size", "_sizing", "_exposure", "larger_positions", "prefer_small_position", "size_conservatively",
)
REASSESSMENT_DISCLAIMER = "not a buy date"  # plan §6/§9: the reassessment date's mandated negation, not advice
_WORD = re.compile(r"[a-z][a-z0-9_]*")
PREDICATE_TOKEN = re.compile(r"(?i)(?<![a-z])(buy|add|accumulate|starter)(?![a-z])")
IMPERATIVE = re.compile(r"(?i)^(buy|add|accumulate|start|enter|sell|exit|reduce|trim|hold|wait|avoid|consider)\b")
CAPPED_BY = re.compile(r"(new_money|long_term)_(buy|starter|wait|wait_for_data|insufficient_data|watch|avoid)")


def advice(value: Any) -> list[str]:
    """The advice phrases and entry codes a string carries (R-LEX)."""
    if not isinstance(value, str):
        return []
    low = value.lower().replace(REASSESSMENT_DISCLAIMER, "")
    return [p for p in ADVICE_PHRASES if p in low] + sorted({w for w in _WORD.findall(low) if w in ENTRY_CODES})


SENTENCE_END = re.compile(r"(?<=\.) ")


def segments(executive_summary: str) -> list[str]:
    """The summary's sentences; the first three are the horizon segments (plan §7 class N)."""
    return SENTENCE_END.split(executive_summary)


def holds(horizon: str, card: Mapping[str, Any]) -> bool:
    """The horizon's entry predicate (plan §7): entry_allowed, add_allowed, lt_entry."""
    if horizon == MID:
        return card["new_money"]["action"] in ENTRY_ACTIONS
    if horizon == HOLDER:
        return bool(card["if_held"]["add_allowed"])
    return card["long_term"]["new_money"]["action"] in ENTRY_ACTIONS


STARTER_ONLY_LABELS = ("dip_assessment.assessment.dip_quality", "decision_modes.balanced.suitability",
                       "decision_modes.speculative.suitability")


def actionable_violations(payload: Mapping[str, Any], card: Mapping[str, Any]) -> list[tuple[str, Any, str]]:
    """OC-RF4 over one projected payload; card is decision_card (identical at every tier)."""
    found = []
    for path, value, field in classified(payload):
        if field.cls == A and field.horizon is not None and advice(value) and not holds(field.horizon, card):
            found.append((path, value, f"entry wording without {field.horizon}"))
        if (field.cls == A and field.horizon == MID and isinstance(value, str) and value in FULL_ENTRY_CODES
                and card["new_money"]["action"] != "buy"):
            found.append((path, value, "full-entry code without new_money buy"))
        if path in STARTER_ONLY_LABELS and value == "starter_only" and card["new_money"]["action"] != "starter":
            found.append((path, value, "starter_only without new_money starter"))
        if re.search(r"(^|\.)capped_by(\.|$)", path) and value != {}:  # an empty capped_by map caps nothing
            match = CAPPED_BY.fullmatch(value) if isinstance(value, str) else None
            current = None if match is None else (card["new_money"] if match[1] == "new_money"
                                                  else card["long_term"]["new_money"])["action"]
            if match is None or match[2] != current:
                found.append((path, value, "capped_by names another action"))
    return found


def diagnostic_violations(payload: Mapping[str, Any]) -> list[tuple[str, Any, list[str]]]:
    """R-LEX-2: D values carry no entry code and no advice phrase."""
    return [(p, v, advice(v)) for p, v, f in classified(payload) if f.cls == D and advice(v)]


def conditional_violations(payload: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """R-LEX-3: C string values are predicates (no buy/add/accumulate/starter token, no leading imperative)."""
    return [(p, v) for p, v, f in classified(payload) if f.cls == C and isinstance(v, str)
            and (PREDICATE_TOKEN.search(v) or IMPERATIVE.match(v))]


def freshness_violations(payload: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The restated freshness invariant: no A, S, N or C leaf carries an entry code or advice phrase (S: no
    value at all). P and T are exempt; D is covered by R-LEX-2 unconditionally."""
    found = []
    for path, value, field in classified(payload):
        cls = field.cls
        if (cls == S and value is not None) or (cls in (A, N, C) and advice(value)):
            found.append((path, value))
    return found
