"""Public versions, release pins, historical decisions and the schema-7 sizing migration."""

import re
from pathlib import Path

from stock_analysis import SCHEMA_VERSION
from stock_analysis.tools.analyze.decision import NEW_MONEY_ACTIONS
from stock_analysis.tools.analyze.gates import ANALYZE_GATE_IDS
from stock_analysis.utils.normalize import SNAPSHOT_FIELDS, SNAPSHOT_VERSION
from stock_analysis.utils.position_sizing import SizingAssumptions, SizingLimits, SizingScenario
from stock_analysis.utils.provenance import build_meta
from tests.support.output_contract import CLASSES

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ("In `watchlist_snapshot`, `new_money_action` and `if_held_action` are the mid-term decision and "
            "`long_term_action` the long-term one; `scan_watchlist` state holds swing-card actions.")
SCHEMA_7 = '`meta.schema_version` goes from `"6"` to `"7"`'
SCHEMA_6 = '`meta.schema_version` goes from `"5"` to `"6"`'
PRICE_ZONE_ALERTS = ("`price_zone` follows the uncapped bands, so a capped name (falling knife, extended decline, "
                     "extreme risk, valuation headwind) can now report zone moves the capped `zone` did not; they are context, not advice.")
SCHEMA_6_NAMED = ("policy_action", "sizing_guidance", "decision_card.horizon_fit", "decision_card.rationale",
                  "decision_card.entry.current_zone", "action_zones.current_zone", "unavailable_reason",
                  "decision_card.action_now")


def _read(name: str) -> str:
    return (ROOT / name).read_text()


def _flat(text: str) -> str:
    return " ".join(text.split())


def _sections(markdown: str, level: str) -> dict[str, str]:
    """{heading: body} for every `level` heading, in order."""
    parts = re.split(rf"^{level} (.+)$", markdown, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _schema_6_section() -> str:
    found = [body for body in _sections(_read("CHANGELOG.md"), "##").values() if SCHEMA_6 in body]
    assert len(found) == 1, "exactly one CHANGELOG section introduces schema 6"
    return found[0]


def test_schema_version_is_7() -> None:
    """Sizing changes advance the response contract without advancing the snapshot."""
    assert SCHEMA_VERSION == "7"
    assert build_meta("analyze_stock")["schema_version"] == "7"


def test_docs_state_the_current_versions_and_the_release_pin() -> None:
    """T2 (docs pin): the newest CHANGELOG section, the README and AGENTS.md state the code's schema and snapshot
    versions; the README targets the newest versioned release and labels unreleased content iff the CHANGELOG's newest
    section is Unreleased. Holds before WP2.4's versions, after them and after the release. G."""
    changelog = _sections(_read("CHANGELOG.md"), "##")
    newest_heading, newest = next(iter(changelog.items()))
    released = next(m.group(1) for h in changelog if (m := re.match(r"(\d+\.\d+\.\d+) \(", h)))
    readme = _read("README.md")
    unreleased = next((text for heading, text in _sections(readme, "##").items()
                       if heading.startswith("Unreleased")), None)
    agents = _flat(_read("AGENTS.md"))

    assert f'`"{SCHEMA_VERSION}"`' in newest and f'`"{SNAPSHOT_VERSION}"`' in newest
    assert set(re.findall(r"stock-analysis-mcp@([^\s`\"')]+)", readme)) == {f"v{released}"}
    assert (unreleased is not None) == newest_heading.startswith("Unreleased")
    assert f"schema **{SCHEMA_VERSION}**" in (unreleased or readme)
    assert f"**{SNAPSHOT_VERSION}**" in (unreleased or readme)
    assert f'now "{SCHEMA_VERSION}"' in agents and f'now "{SNAPSHOT_VERSION}"' in agents


def test_the_schema_6_changelog_names_every_removal() -> None:
    """T3 (schema-6 inventory, §13): exactly one CHANGELOG section introduces schema 6, and it names every removed
    or moved key, the alias and every snapshot field 3.0.0 retires or adds. RF."""
    section = _schema_6_section()
    snapshot_delta = sorted(SNAPSHOT_FIELDS["2.0.0"] ^ SNAPSHOT_FIELDS["3.0.0"])

    assert [key for key in (*SCHEMA_6_NAMED, *snapshot_delta) if f"`{key}`" not in section] == []


def test_the_schema_6_changelog_says_price_zone_alerts_on_capped_names() -> None:
    """Review fix (integrated plan §6 decision 9): the schema-6 section says `price_zone` can alert on a capped name
    whose capped `zone` stayed still, and that those moves are context, not advice. RF."""
    assert PRICE_ZONE_ALERTS in _flat(_schema_6_section())


def test_readme_and_agents_carry_the_decision_tables_and_horizons() -> None:
    """T4 (§9 tables, BL-F3, S-C16 docs clause): README and AGENTS.md carry a row per new-money action and per field
    class, the README names every analyze gate and says `profile` is sizing-only, and README, AGENTS.md and the
    schema-6 CHANGELOG section label the snapshot mid-term / long-term and the scan state swing. RF."""
    readme = _flat(_read("README.md"))
    assert [g for g in ANALYZE_GATE_IDS if f"`{g}`" not in readme] == []
    assert "changes sizing only, never an action" in readme
    for name in ("README.md", "AGENTS.md"):
        text = _read(name)
        assert [a for a in NEW_MONEY_ACTIONS if f"| `{a}` |" not in text] == [], name
        assert [c for c in CLASSES if f"| {c} |" not in text] == [], name
        assert HORIZONS in _flat(text), name
    assert HORIZONS in _flat(_schema_6_section())


def test_schema_7_migration_names_canonical_fields_and_legacy_adapters() -> None:
    """The current migration covers the canonical shape and every legacy sizing adapter."""
    newest = next(iter(_sections(_read("CHANGELOG.md"), "##").values()))
    assert SCHEMA_7 in newest
    for shape in (SizingAssumptions, SizingLimits, SizingScenario):
        assert [field for field in shape.__annotations__ if f"`{field}`" not in newest] == []
    legacy = (
        "decision_card.sizing", "starter_pct", "full_pct_cap", "starter_dollars", "full_dollars",
        "whole_shares", "fractional_shares", "at_price", "unavailable_reason",
        "action_zones.position_sizing_range", "suggested_pct_range", "max_pct", "profile_cap_pct",
        "dollars_for_account", "portfolio_assumption", "shares_range", "stop_implied_max",
        "decision_modes", "starter_position", "max_position", "pct", "dollars", "shares",
        "capital_at_risk", "plan", "max_loss_dollars", "position_pct", "distance_pct",
        "stop_distance_pct", "buy_zone_low", "buy_zone_high", "entry_band_unavailable",
    )
    assert [field for field in legacy if f"`{field}`" not in newest] == []
    assert "numeric_range_unavailable" in newest
    assert "below_one_whole_share" in newest
    assert "percentage_only" in newest


def test_current_guidance_explains_null_sizing_and_conditional_loss() -> None:
    """A buy can lack a size; availability and planning loss stay distinct from the action."""
    readme = _read("README.md")
    for text in (readme, _read("AGENTS.md")):
        assert "decision_card.sizing_status" in text
        assert "its reason" in text
        assert "percentage_only" in text
        assert "below_one_whole_share" in text
    assert "Analyze HOOD" in readme
    assert "not a guarantee that every suggestion fits the risk budget" not in _flat(readme)
    assert "The planning stop is not monitored" in _flat(readme)
    assert "gaps and slippage can increase the loss" in _flat(readme)
    assert "pending the next" not in _read("AGENTS.md")
