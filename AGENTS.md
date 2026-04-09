# Repository Instructions

## Purpose
- This repository provides the `stock-analysis` MCP server for single-stock analysis.
- The public contract is the MCP tool surface and the JSON schemas returned by those tools.

## Key Paths
- `src/stock_analysis/server.py`: MCP server entrypoint
- `src/stock_analysis/tools/`: tool implementations
- `src/stock_analysis/data/`: yfinance client and caching
- `src/stock_analysis/resources/`: MCP resources
- `tests/`: schema, invariants, and utility coverage
- `README.md`: install, usage, and public-facing behavior

## Working Rules
- Keep changes consistent with the existing MCP tool surface unless the task explicitly changes it.
- Preserve response-shape stability for `analyze`, `compare`, and other public tools. If a schema changes, update tests and docs in the same change.
- Prefer extending existing analysis and normalization code over adding duplicate logic paths.
- Surface missing, stale, or low-confidence data through provenance and data-quality fields instead of hiding it.
- Keep tests deterministic. Do not hardcode live market values or rely on current dates unless the test is explicitly about date handling.
- Treat output as informational software, not financial advice. Do not weaken existing disclaimers.

## Validation
- Install dev dependencies with `uv pip install -e ".[dev]"`.
- Run `uv run pytest` after behavior changes.
- Run `uv run ruff check` after code edits.
- Run `uv run mypy` when touching typed Python code or schemas. `mypy` is available via the `dev` extra in `pyproject.toml`.

## Test Characteristics
- The suite is unit-test focused and should not require network access.
- Keep the suite fast enough for routine local validation.

## When Changing Analysis Logic
- Update schema and invariant tests under `tests/test_schemas.py` and `tests/test_tools/` when score math, section ordering, or required fields change.
- Keep score math internally consistent and auditable.
- If you change user-visible output semantics, update the relevant examples in `README.md`.

## Common Schema Gotchas
- `technicals` returns `current_price` at the top level, not under a `summary` object.
- `analyze` returns price, name, sector, market cap, and currency under `summary`, not as duplicated top-level fields.
- `options_signals` returns put/call data at `put_call_ratio.volume_based` and `put_call_ratio.oi_based`, not under `summary.*`.
- In `analyze`, raw options output is nested under `options_signals`, while derived narrative fields appear elsewhere such as `section_summaries`, `signals`, and `decision_context`.
- `risk_metrics` keeps volatility at `volatility.annualized`; the synthesized `analyze` output exposes the normalized view under `risk_summary.annualized_volatility`.

## Multi-Agent Note
- `AGENTS.md` is the canonical shared instruction file for this repo.
- `CLAUDE.md` should stay as a thin wrapper and only contain Claude-specific deltas.
