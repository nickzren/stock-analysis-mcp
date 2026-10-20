# Holder and position characterization

Frozen synthetic observations from v0.4.0 production code at `c9fba4a` (MCP SDK
1.30.0, schema 7, snapshot 3.0.0). The baseline package changed tests only. It does not
endorse the legacy tax assumptions or claim that investment recommendations work.
The separate current capture includes the long-term holder core and
stateless position context, summary guidance and the coordinated schema 8 / snapshot 4.0.0 migration.

- `before.md`: frozen stock/position table.
- `after.json` / `after.md`: current stock and position outputs. Existing stock decisions/sizing
  match the baseline. The new `decision_card.long_term.if_held`, long-term summary
  segment, and snapshot 4.0.0 version/hash/new holder action are the explicit stock
  changes; all pre-existing snapshot data stay equal. Position output changes as below.
- `before.json`: source inputs, source-call counts, selected price/provenance,
  full decision cards/sizing, verdict and business evidence, modes/zones,
  summaries and **complete** stock snapshots; position responses omit only
  execution duration.
- `tests/support/holder_matrix.py`: scenarios and offline current-output recorder.
- `tests/test_step3a/`: observations, immutable hashes, recorder safeguards and
  real in-process MCP-client input tests.

## Current position migration

Position inputs are optional and validated before reads. One full analysis supplies
its canonical holder/new-money assessment, selected price and raw diagnostics.
Valid calls now invoke each of the eight source tools once; invalid personal
inputs invoke none. Cost/date/shares/account/ceiling never change stock decisions,
entry sizes or complete stock snapshots. The position request clock is Eastern
Time (`2026-10-09T20:00:00-04:00` for these rows).

USD share-priced positions have descriptive P&L and an allocation check against
only an explicit ceiling. Missing/foreign-currency context stays unavailable;
stale/fallback prices are labelled historical and cannot verify allocation. ETFs
can have personal numbers while long-term ownership is `not_assessed`. Known
broken businesses now differ from sound businesses even on an identical chart.
Tax estimates and the 365-day tax category are removed. Raw technical flags and
unmonitored reference levels move under `diagnostics.technicals`, without an
active sell-signal count. Current captures omit execution duration and per-source
timings (including failed reads); separate evidence records offline timing, not live network performance.

The scenarios below describe the **frozen baseline**, not the new handler.

## Baseline coverage and limits

There are 32 stock cases and 25 position cases. All prices, companies, accounts
and holdings are synthetic. Even rows named `etf_spy` / `etf_vti` use `ACME` as
the symbol with representative ETF metadata; they are not live SPY/VTI analyses.
The ADR row checks the name/classification path only, not statement-currency
conversion or a verified ADR ratio. The neutral-valuation row is not a case of
extreme overvaluation.

Stock cases reuse the existing decision-matrix tool stubs. They cover sound and
broken business on the same chart, missing evidence, confirmed negative FCF,
critical/unknown/noncritical runway, a non-burner without cash/runway numbers,
net debt, negative equity, entry gates, ETF/preferred/unknown classification,
non-USD prices, profiles and differing horizons. The hold-with-avoid-fit case
retains the actual reasons: extreme risk, declining revenue and negative FCF.
The old partly empty PROFITABLE fixture is explicitly labelled missing evidence,
not a clean or assessable business.

The business-failure-plus-unknown-runway and critical-runway-plus-unknown-business
combinations are not claimed as end-to-end baseline rows. Bounded probes did not
produce them through today's coupled classifier: the unprofitability evidence
that enables runway calculation also makes the business component assessable.
The pure-component tests cover these injected states separately; they are not
claims about provider payloads or observed investment performance.

Position cases record gains/losses, optional/fractional/zero/negative inputs,
365/366-day and leap-date boundaries, future/invalid dates, missing/nonpositive
prices, stale or fallback closes, technical errors and a changing ATR reference.
The baseline tool reads only technicals, so identical charts give identical
position results for a broken business, a stock, an ETF and a non-USD holding.
One paired case gives analyze a selected daily close of $100 while the position
tool uses the same technical input's $105 current price.

Source counts are **tool invocations**, not Yahoo HTTP requests or latency
measurements. Analyze calls each of its eight stubbed source tools once; a valid
position call uses one technicals invocation. Invalid date syntax makes zero
reads. The later full-analysis position path must inventory added work and
measure timing separately.

Analyze uses `2026-10-09T20:00:00-04:00`. Position deliberately uses a simulated
naive host-local `2026-10-09T20:00:00`; this does not establish timezone-correct
behavior. The client boundary tests use their own stated fixed October 19 clock.

The real MCP tests preserve the distinction between published schema validation,
framework coercion, application errors and JSON output sanitization. Numeric
strings and booleans can reach this old handler despite the numeric schema;
negative values, future dates and text `nan`/`inf`/overflow are accepted.
Nonfinite outputs become null. Python NaN/Inf supplied to this client instead
arrive as null, so they do not prove server-side nonfinite rejection. Missing
cost/date and future context fields (`horizon`, `portfolio_value`,
`max_holding_pct`) are rejected before data reads. These are baseline observations,
not the new input contract.

## Approved changes, kept separate from observations

| Current observation | Later approved design; package start still separate |
|---|---|
| Only mid-term `if_held`; long-term object contains new money only | Add a separate long-term holder assessment. Preserve current mid-term/new-money decisions and entry sizing. |
| ETF/preferred coverage differs from unknown classification | Long-term known inapplicable instruments get `not_assessed`; unknown classification gets `insufficient_data`. Independently valid USD per-share personal math remains available. |
| Hold can coexist with an avoid suitability fit | Show both and their actual reasons. No automatic long-term exit from chart signals or entry gates. |
| Position requires cost/date and reads technicals alone | Optional validated personal context; one full analysis provides stock assessment and the same selected price/provenance for P&L. |
| No allocation check or portfolio/ceiling input | Explicit user ceiling only. Allocation status never substitutes for the holder action; no default ceiling or automatic trim quantity. |
| Boolean/negative/nonfinite values or future dates can be accepted | Reject invalid supplied values before data calls; retain valid decimal strings, fractional owned shares and genuine zero cost. |
| Fixed 37%/20% tax rates and day-based tax classification | Retire tax estimates/classification; retain descriptive purchase date/elapsed days. |
| Chart sell flags and moving ATR reference | Move raw signals to diagnostics; do not treat a recomputed reference as an anchored or monitored holder stop. |
| Snapshot 3.0.0 has no long-term holder field | Add the field together with snapshot 4.0.0 in the later schema/guidance package; no personal data in stock snapshots. |

This table is not a second decision implementation. No future action, allocation
status or tax behavior is fabricated in the frozen observations. Step 2c and
saved holdings remain deferred.

## Reproduce

```sh
uv run --frozen --offline pytest tests/test_step3a -q -p no:cacheprovider
capture_dir=$(mktemp -d)
uv run --frozen --offline python -m tests.support.holder_matrix --after --output-dir "$capture_dir"
```

Current captures use `baseline_code` as a reference. The frozen JSON uses
`recorded_code` to identify the production source that actually produced it.
Tests compare all old content exactly after removing only the new long-term
holder object. The current Markdown adds its own holder action/basis column;
the frozen renderer stays byte-identical.

Baseline generation is permanently disabled (`--baseline`, or omitting `--after`,
fails before any work). The recorder never writes before files, refuses to relabel
old evidence as current, and preflights both after files before writing; differing
existing evidence is not overwritten. Identical writes are idempotent. HTTP
requests through the two application transports are blocked, including swallowed
attempts. No real market data or user state is read.

The before files have literal SHA-256 pins independent of generated current
outputs. Keep them and all earlier decision/sizing fixtures byte-identical.
As later approved changes land, replace the relevant current-behavior assertions
and write separate after evidence; never regenerate or hand-edit the baseline to
make tests pass. App versions and inputs stay unchanged. Only the new holder object is added in
the current holder-core package; position behavior is still the legacy baseline.
