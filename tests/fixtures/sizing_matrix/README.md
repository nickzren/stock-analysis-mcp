# Sizing characterization baseline

These synthetic, offline recordings capture released v0.3.0 (`7d581f1`),
including its known sizing defects. They are observations, not desired behavior
or investment recommendations. Production sizing is unchanged by this package.

- `before.md`: compact table for review.
- `before.json`: exact card, zone and mode sizing; summaries; all three decisions;
  complete snapshots; and response-tier sizing/entry fields.
- `tests/support/sizing_matrix.py`: minimal inputs and reproducible recorder.
- `tests/test_step2b/`: characterization and real MCP-client boundary tests.

The recorder reuses the Step 2a offline tool stubs and fixed clock. It adds
assessable fundamentals to synthetic buy scenarios. The source stubs are inputs,
not recordings of Yahoo responses. No prices, account data or holdings are read
from a real user or provider.

`inputs` and `limit_origins` are fixture metadata. `resolved_profile` records
the real profile helper, not a top-level analyze response field. Report-only
implied loss uses the displayed allocation dollars and stop; v0.3.0 does not
emit this loss for analyze, and its displayed dollars need not match whole
shares. The table deliberately exposes that discrepancy. Missing or invalid
stops show no derived loss rather than an invented estimate.

## Coverage

| Cases | Observed behavior / purpose |
|---|---|
| Core, balanced, speculative; narrow/wide stops | Defaults can exceed the displayed risk budget; profile changes leave decisions/snapshots unchanged. |
| Risk only, cap only (tight/wide), both | Caller limits and profile defaults are distinguished; stop-implied caps are diagnostic only. |
| Tiny risk / tiny position cap | Risk rounds to zero then falls back to 1%; a 0.06% maximum becomes 0.1%. |
| Small/no account | Fractional dollar amounts can accompany zero whole shares; absent accounts keep amounts null. |
| Missing/negative stop, inverted band | Sizing survives; missing ATR's fallback is labelled ATR; inverted bounds are emitted. |
| Starter / earnings-blocked | Existing visibility controls: starter hides full sizing; blocked new money hides both. |
| Speculative-mode override | Mode sizing exceeds the canonical card; `capital_at_risk` echoes the entire account. |
| Swing: fractional / zero / no account | Dollars and reported loss use fractional shares rather than the displayed whole count. |
| MCP boundary tests | Number-only schemas, coercion, malformed rejection, omission/null defaults, and validation after company lookup. |

Boundary tests use the real FastMCP client and registered tools with downstream
handlers replaced. They distinguish framework coercion from application
acceptance; direct validator and real analyze-wrapper cases cover the latter.
They do not claim a production fix or end-to-end acceptance of nonfinite inputs.

## Reproduce safely

From the repository root:

```sh
uv run --frozen --offline pytest tests/test_step2b -q -p no:cacheprovider
capture_dir=$(mktemp -d)
uv run --frozen --offline python -m tests.support.sizing_matrix --output-dir "$capture_dir"
diff -u tests/fixtures/sizing_matrix/before.json "$capture_dir/before.json"
diff -u tests/fixtures/sizing_matrix/before.md "$capture_dir/before.md"
```

The generator blocks the two HTTP transports used by the app, including swallowed
attempts. It refuses all writes if either existing target would differ. Identical
files are left untouched. It never writes the Step 2a decision matrix.

Keep these before files frozen. Later packages must add separate after evidence
and deliberately replace defect pins as fixes land; do not bless a changed result
by overwriting the baseline. Check every snapshot field and action against this
recording before keeping snapshot 3.0.0. The source of this package is tests and
test support only; the schema remains 6 and the package remains 0.3.0.
