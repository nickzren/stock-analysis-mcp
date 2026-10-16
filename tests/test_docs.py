"""WP2.4: schema 6 and the docs that describe it (plan §10 WP2.4, D-13; §9 tables; §8 S-C16; BL-F3).

The version pin; the docs' versions and the README's release pin; the CHANGELOG section that introduces schema 6
naming every removal; the README / AGENTS.md action and field-class tables and the snapshot-vs-scan horizon
labels.
"""

import re
from pathlib import Path

from stock_analysis import SCHEMA_VERSION
from stock_analysis.tools.analyze.decision import NEW_MONEY_ACTIONS
from stock_analysis.tools.analyze.gates import ANALYZE_GATE_IDS
from stock_analysis.utils.normalize import SNAPSHOT_FIELDS, SNAPSHOT_VERSION
from stock_analysis.utils.provenance import build_meta
from tests.support.output_contract import CLASSES

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ("In `watchlist_snapshot`, `new_money_action` and `if_held_action` are the mid-term decision and "
            "`long_term_action` the long-term one; `scan_watchlist` state holds swing-card actions.")
SCHEMA_6 = '`meta.schema_version` goes from `"5"` to `"6"`'
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


def test_schema_version_is_6() -> None:
    """T1 (schema 6, D-13): SCHEMA_VERSION and every response's meta.schema_version are "6". RF."""
    assert SCHEMA_VERSION == "6"
    assert build_meta("analyze_stock")["schema_version"] == "6"


def test_docs_state_the_current_versions_and_the_release_pin() -> None:
    """T2 (docs pin): the newest CHANGELOG section, the README and AGENTS.md state the code's schema and snapshot
    versions; the README installs the newest released tag and labels unreleased content iff the CHANGELOG's newest
    section is Unreleased. Holds before WP2.4's versions, after them and after the release. G."""
    changelog = _sections(_read("CHANGELOG.md"), "##")
    newest_heading, newest = next(iter(changelog.items()))
    released = next(m.group(1) for h in changelog if (m := re.match(r"(\d+\.\d+\.\d+) \(", h)))
    readme = _read("README.md")
    unreleased = _sections(readme, "##").get("Unreleased (main)")
    agents = _flat(_read("AGENTS.md"))

    assert f'`"{SCHEMA_VERSION}"`' in newest and f'`"{SNAPSHOT_VERSION}"`' in newest
    assert set(re.findall(r"stock-analysis-mcp@(\S+) ", readme)) == {f"v{released}"}
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
