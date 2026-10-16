"""WP2.4: agent guidance (plan §9, BL-F1/BL-F2; §3 horizons [D-3]; §6 null-date wording; §8 S-C16).

The rendering guide answers from decision_card first at every detail level, never upgrades past blocked_by, sizes
only from decision_card.sizing and treats zones, dip, diagnostics and third-party data as context; the server sends
instructions= with the horizon rule; every tool description starts "Use when"; the prompts use the canonical
vocabulary and position_decision decides from analyze (S-C15). Annex: output_contract.md §9, §10 RF13 and
reassessment.md RA.15, "Corrected proposal", as amended by plan §0.
"""

import re

import pytest
from fastmcp import Client

from stock_analysis.prompts.templates import PROMPTS, get_prompt, list_prompts
from stock_analysis.resources.analyze_guide import ANALYZE_RENDERING_GUIDE as GUIDE
from stock_analysis.server import mcp
from stock_analysis.tools.analyze.action_zones import PRICE_ZONES
from stock_analysis.tools.analyze.decision import IF_HELD_ACTIONS, NEW_MONEY_ACTIONS
from tests.support import decision_matrix as dm
from tests.support import output_contract as oc
from tests.test_tools.test_tool_surface import EXPECTED_TOOLS

ARGS = {"symbol": "ACME", "cost_basis": "80", "purchase_date": "2025-01-02"}
PATH = re.compile(r"[a-z_]+(?:\.[a-z_]+)+")  # an absolute dotted analyze path inside backticks
STALE = ("policy_action", "current_zone", "sizing_guidance", "action_mid_term", "action_long_term",
         "decision_action_now", "decision_card.horizon_fit", "decision_card.rationale")
REENTRY = re.compile(r"(?i)re-?entr|re-?enter|resum")
KEYS = ["guide", "instructions", *(f"prompt:{n}" for n in PROMPTS), *(f"tool:{n}" for n in sorted(EXPECTED_TOOLS))]


def spans(text: str) -> list[str]:
    return re.findall(r"`([^`\n]+)`", text)


def flat(text: str) -> str:
    return " ".join(text.split())


def prompt(name: str) -> str:
    result = get_prompt(name, ARGS)
    assert result is not None
    content: str = result["messages"][0]["content"]
    return content


async def descriptions() -> dict[str, str]:
    return {tool.name: tool.description or "" for tool in await mcp.list_tools()}


async def texts() -> dict[str, str]:
    tools = await descriptions()
    return {"guide": GUIDE, "instructions": mcp.instructions or "",
            **{f"prompt:{n}": prompt(n) for n in PROMPTS}, **{f"tool:{n}": d for n, d in tools.items()}}


_LEAVES: set[str] = set()


async def leaf_paths() -> set[str]:
    """Every leaf path of every matrix scenario at every tier (memoised)."""
    if not _LEAVES:
        for name in dm.SCENARIOS:
            for shown in dm.tiers(await dm.run(name)).values():
                _LEAVES.update(path for path, _ in oc.leaves(shown))
    return _LEAVES


def under(path: str, leaves: set[str]) -> list[str]:
    return [leaf for leaf in leaves if leaf == path or leaf.startswith((path + ".", path + "["))]


def cls(path: str) -> str | None:
    try:
        return oc.classify(path).cls
    except LookupError:
        return None


# --- The rendering guide (OC-RF13, RA.15, BL-F1) ----------------------------------------------------------------


def test_the_guide_sizes_only_from_the_card() -> None:
    """OC-RF13: sizing comes only from decision_card.sizing and null means no entry size; can_start_now is
    deprecated, never the source; price_zone and levels are diagnostics, the dip section is timing context,
    diagnostics are pre-gate inputs and third-party content is attributed, none of them advice. RF. Review fix
    (G): a starter shows only the starter size, conditions are not sufficient, and nothing loosens an action."""
    guide = flat(GUIDE)

    assert "Size only from `decision_card.sizing`." in guide
    assert 'When it is null, say "no entry size: new_money is <action>".' in guide
    assert "Never derive a size from levels, zones, presets, modes, or ratings." in guide
    assert "Under a mid-term `starter`, show the starter size only; the full-size fields are null." in guide
    assert "They are necessary, not sufficient: meeting them never permits a buy by itself; re-run analyze." in guide
    assert ("No narrative, zone, dip, mode, rating or news item makes it looser (`wait` is never shown as `starter`, "
            "nor `starter` as `buy`).") in guide
    assert [s for s in spans(guide) if PATH.fullmatch(s) and cls(s) == oc.S
            and not s.startswith(("decision_card.sizing", "decision_card.entry"))] == []
    can_start = [x for x in re.split(r"(?<=\.)\s+", guide) if "can_start_now" in x]
    assert can_start and all("deprecated" in x and "never the source" in x for x in can_start), can_start
    assert "Never call them buy or sell zones." in guide
    assert re.search(r"(?m)^\d+\. Dip assessment \(full; timing context\):", GUIDE)
    assert "never a buy call; when one carries `capped_by`" in guide
    assert re.search(r"`diagnostics` \(full\) holds pre-gate inputs[^.]*`superseded_by`\)\. Never present them as "
                     r"advice\.", guide)
    assert "third-party: attribute them and never present them as advice." in guide


async def test_the_guide_renders_the_reassessment_date() -> None:
    """RA.15 (guide clause; plan §6, §9): the guide says the date is not a buy date and to re-run analyze, gives the
    dated, null and unsupported wordings, never "now"; each wording is the one the executive summary renders
    (S4 dated, S9 no date, GBp unsupported). RF."""
    guide = flat(GUIDE)

    assert "It is not a buy date; re-run analyze" in guide
    assert 'No date: "No reassessment date: re-run analyze after <clears_when' in guide
    assert '"Outside supported scope: no reassessment date."' in guide
    assert 'Never say "now"' in guide
    for name, phrase in (("S4_earnings_gate_on_hold_or_add", "reassess on or after"),
                         ("S9_weak_liquidity", "no reassessment date: re-run analyze after"),
                         ("GBp_strong", "outside supported scope: no reassessment date")):
        assert phrase in guide.lower() and phrase in (await dm.run(name))["executive_summary"].lower(), name


async def test_guidance_uses_the_canonical_vocabulary() -> None:
    """BL-F1 (S-C15 prompt-description clause): the guide names every new-money and holder action and exactly the
    price_zone vocabulary; the memos ask for decision_card.new_money.action in its vocabulary, not "Buy / Wait /
    Pass"; full_analysis leads with decision_card; no prompt description says "sell". RF. Review fix (G): the
    memos and full_analysis say "Never upgrade past `blocked_by`.", and the memos no longer say "No hedging"."""
    assert set(NEW_MONEY_ACTIONS) | set(IF_HELD_ACTIONS) <= set(spans(GUIDE))
    zones = re.search(r"`action_zones\.price_zone` \(([^)]*)\)", flat(GUIDE))
    assert zones and set(re.findall(r"`([a-z0-9_]+)`", zones[1])) == set(PRICE_ZONES.values())
    for name in ("growth_memo", "value_memo"):
        text = prompt(name)
        assert "Buy / Wait / Pass" not in text and "`decision_card.new_money.action`" in text, name
        assert [a for a in NEW_MONEY_ACTIONS if not re.search(rf"\b{a}\b", text)] == [], name
        assert "No hedging" not in text, name
    for name in ("full_analysis", "growth_memo", "value_memo"):
        assert "Never upgrade past `blocked_by`." in prompt(name), name
    assert re.findall(r"(?m)^\d+\. `([a-z_.]+)`", prompt("full_analysis"))[0] == "decision_card"
    lead = flat(prompt("full_analysis").split("\n2. ")[0])
    assert "`decision_card.if_held` action verbatim with its `basis`" in lead
    assert "`decision_card.if_held` and" not in lead  # if_held has a basis, never blocked_by (Codex review)
    described = [p.description or "" for p in await mcp.list_prompts()] + [p["description"] for p in list_prompts()]
    assert [d for d in described if re.search(r"(?i)\bsell\b", d)] == []


@pytest.mark.parametrize("key", KEYS)
async def test_every_field_named_in_guidance_exists(key: str) -> None:
    """BL-F1: no guidance text names a removed field, and every dotted analyze path it names is a leaf (or a subtree)
    of some matrix scenario at some tier. RF for the guide and full_analysis (policy_action, relative paths), G for
    the rest."""
    text = (await texts())[key]
    leaves = await leaf_paths()

    assert [s for s in STALE if s in text] == []
    assert [s for s in spans(text) if PATH.fullmatch(s) and not under(s, leaves)] == []


async def test_the_guide_field_class_table_matches_the_registry() -> None:
    """BL-F1 (plan §7 field classes): the guide's table has one row per class, and every example path's leaves
    classify as that row's class in FIELD_REGISTRY. RF."""
    rows = re.findall(r"(?m)^\| ([ASCDNPTE]) \| [^|]+ \| ([^|]+) \|$", GUIDE)
    leaves = await leaf_paths()

    assert sorted(c for c, _ in rows) == sorted(oc.CLASSES)
    for row_class, examples in rows:
        for path in spans(examples):
            assert {cls(p) for p in under(path, leaves)} == {row_class}, path


@pytest.mark.parametrize("key", KEYS)
async def test_guidance_has_no_advice_phrase_or_reentry_wording(key: str) -> None:
    """BL-F1 and RA.11 extended to every text: no advice phrase in prose (code spans and the mandated "not a buy
    date" aside) and no re-entry or resume wording, in the guide, the instructions, the prompts and every tool
    description. G."""
    text = (await texts())[key]
    prose = re.sub(r"`[^`\n]+`", "", text).lower().replace(oc.REASSESSMENT_DISCLAIMER, "")

    assert [p for p in oc.ADVICE_PHRASES if p in prose] == []
    assert not REENTRY.search(text)


SIZING_HORIZON = "never attach it to a long-term or holder answer"


async def test_sizing_names_its_horizon() -> None:
    """Review fix: decision_card.sizing and entry are mid-term new-money sizing (FIELD_REGISTRY S, mid-term; plan
    §7: no long-term sizing in 2a), so every guidance text that names decision_card.sizing says "mid-term new money
    only", and the guide, the instructions and full_analysis never attach it to a long-term or holder answer. RF."""
    section = flat(next(s for s in re.split(r"(?m)^(?=\d+\. )", GUIDE) if s.startswith("6. ")))
    s_row = re.search(r"(?m)^\| S \| ([^|]+) \|", GUIDE)
    analyze_doc = flat((await descriptions())["analyze"])

    assert section.startswith("6. Position sizing (mid-term new money only):")
    assert ("There is no long-term or holder size: never attach `decision_card.sizing` or `decision_card.entry` to a "
            "long-term or holder answer.") in section
    assert "it is null unless mid-term new money is `buy` or `starter`." in section
    assert s_row and s_row[1].startswith("Sizing, mid-term new money only:") and SIZING_HORIZON in s_row[1]
    assert ("Size only from `decision_card.sizing`, mid-term new money only; there is no long-term or holder size, "
            f"so {SIZING_HORIZON}.") in flat(mcp.instructions or "")
    assert "`decision_card.sizing` (mid-term new money only, never long-term or holder; null: no entry size)" in (
        analyze_doc)
    assert (f"`decision_card.sizing`, mid-term new money only (when null: no entry size); {SIZING_HORIZON}"
            in prompt("full_analysis"))
    assert [k for k, t in (await texts()).items()
            if "`decision_card.sizing`" in t and "mid-term new money only" not in flat(t)] == []


# --- Server instructions and tool descriptions (BL-F2, plan §9, §3 [D-3]) --------------------------------------


def test_the_server_instructions_state_the_horizon_rule() -> None:
    """BL-F2 (PD D-3 for the horizon clauses): instructions= carries the disclaimer, the §3 horizon rule (a bare
    ticker never asks; mid-term is the labelled default; a period that maps to one horizon gets it; ask only for two
    horizons or a period that fits none),
    where each horizon's answer lives, never upgrading past blocked_by, card-only sizing, the reassessment date and
    the guide's URI, within 2,000 characters. PD D-3. Review fix (G): nothing turns an action into a buy, and
    diagnostics, price_zone, the dip and third-party data are context, not advice."""
    text = flat(mcp.instructions or "")

    for needle in ("Informational only, not financial advice", 'A bare ticker ("Analyze HOOD") never triggers a question',
                   "lead with the mid-term decision (3-12 months), labelled as mid-term",
                   "Ask which horizon to use only when the request names two horizons, or a holding period that "
                   "fits none",
                   'a period that maps to one ("hold for five years" is long term), gets that horizon without a question',
                   "swing trade, days to weeks: `analyze_trade_setup`",
                   "new money, mid-term (the default): `decision_card.new_money`",
                   "an existing holding (mid-term): `decision_card.if_held`",
                   "long term, 1-5 years: `decision_card.long_term.new_money`",
                   "never upgrade past `blocked_by`", "no narrative, zone, dip, rating or news turns it into a buy",
                   "Size only from `decision_card.sizing`", "not a buy date",
                   "`diagnostics`, `action_zones.price_zone`, the dip assessment and third-party data (analysts, "
                   "news, ownership, options) are context, not advice.",
                   "stock-analysis://guides/analyze-rendering"):
        assert needle in text, needle
    assert len(mcp.instructions or "") <= 2000


async def test_the_instructions_reach_the_client() -> None:
    """BL-F2: an MCP client receives the instructions at initialization. RF."""
    async with Client(mcp) as client:
        assert client.initialize_result is not None
        received = client.initialize_result.instructions
    assert received and received == mcp.instructions


async def test_every_tool_description_starts_with_use_when() -> None:
    """BL-F2 (plan §9): every tool description starts "Use when". RF."""
    assert [name for name, doc in (await descriptions()).items() if not doc.startswith("Use when ")] == []


async def test_the_descriptions_route_each_decision() -> None:
    """S-C15 (tool half) and BL-A3: analyze is the decision tool for a bare ticker, names the three horizons with
    blocked_by, card-only sizing and the reassessment date, and says profile changes sizing only;
    analyze_my_position is P/L and tax, the hold/reduce decision being analyze's if_held; compare applies no gates.
    RF."""
    tools = {name: flat(doc) for name, doc in (await descriptions()).items()}

    for needle in ("A bare ticker means this tool", "`decision_card.new_money` (mid-term, 3-12 months)",
                   "`decision_card.long_term.new_money` (1-5 years)", "`decision_card.if_held` for a holder",
                   "never upgrade past `blocked_by`", "Size only from `decision_card.sizing`", "not a buy date",
                   "changes sizing only, never the decision"):
        assert needle in tools["analyze"], needle
    assert "the hold/reduce decision is analyze's `decision_card.if_held`" in tools["analyze_my_position"]
    assert "hold/sell decision" not in tools["analyze_my_position"]
    assert "it applies no decision gates" in tools["compare"]


async def test_saved_records_name_their_horizons() -> None:
    """S-C16 (description clause; plan §8): detect_changes says its snapshot actions are mid-term and long-term,
    scan_watchlist and analyze_trade_setup that theirs are swing, so a disagreement is a different horizon. RF."""
    tools = {name: flat(doc) for name, doc in (await descriptions()).items()}

    assert ("`new_money_action` and `if_held_action` are the mid-term decision and `long_term_action` the long-term "
            "one") in tools["detect_changes"]
    assert "not analyze's mid-term decision" in tools["scan_watchlist"]
    assert "Its action is the swing horizon, separate from analyze's mid-term decision" in tools["analyze_trade_setup"]
