# Sizing: current output table

Generated from synthetic offline scenarios by `tests.support.sizing_matrix`. Do not edit by hand.
Observed current outputs, including remaining limitations; not investment recommendations.
Details, full snapshots and limit origins: `after.json`. The frozen baseline remains `before.json`.
The implied loss follows the same whole-share amounts as the emitted planned loss.
Sizing is conditional on a sale at the planning stop; it is not a monitored exit or a guaranteed loss cap.

Risk/maximum below are the effective caller/profile values; both are enforced.

| Case | Resolved risk / maximum (%), origin | New money / held | Stop | Starter / full (%) | Starter / full ($) | Starter whole shares | Implied loss starter / full ($) |
|---|---|---|---|---|---|---|---|
| core_narrow | 0.5 (profile) / 8 (profile) | buy / hold | 96.0 | 2.0 / 8.0 | 200.0 / 800.0 | 2 | 8 / 32 |
| core_wide | 0.5 (profile) / 8 (profile) | buy / hold | 80.0 | 2.0 / 2.5 | 200.0 / 200.0 | 2 | 40 / 40 |
| balanced_narrow | 1 (profile) / 10 (profile) | buy / hold | 96.0 | 2.5 / 8.0 | 200.0 / 800.0 | 2 | 8 / 32 |
| balanced_wide | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 5.0 | 200.0 / 500.0 | 2 | 40 / 100 |
| speculative_narrow | 1.5 (profile) / 6 (profile) | buy / hold | 96.0 | 1.0 / 6.0 | 100.0 / 600.0 | 1 | 4 / 24 |
| speculative_wide | 1.5 (profile) / 6 (profile) | buy / hold | 80.0 | 1.0 / 6.0 | 100.0 / 600.0 | 1 | 20 / 120 |
| balanced_low_wide | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 5.0 | 200.0 / 500.0 | 2 | 40 / 100 |
| balanced_high_wide | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.0 / 5.0 | 200.0 / 500.0 | 2 | 40 / 100 |
| balanced_extreme_wide | 1 (profile) / 10 (profile) | starter / hold | 62.5 | 0.5 / None | 0.0 / None | 0 | 0 / — |
| rounded_stop_warning | 1 (profile) / 10 (profile) | buy / hold | 95.0 | 2.0 / 6.0 | 198.0 / 594.0 | 2 | 8 / 24 |
| missing_atr_pct | 1 (profile) / 10 (profile) | buy / hold | 96.0 | 2.5 / 8.0 | 200.0 / 800.0 | 2 | 8 / 32 |
| speculative_binding | 1.5 (profile) / 6 (profile) | buy / hold | 70.0 | 1.0 / 5.0 | 100.0 / 500.0 | 1 | 30 / 150 |
| risk_only | 0.25 (caller) / 10 (profile) | buy / hold | 80.0 | 1.25 / 1.25 | 100.0 / 100.0 | 1 | 20 / 20 |
| cap_only_tight | 1 (profile) / 2 (caller) | buy / hold | 80.0 | 2.0 / 2.0 | 200.0 / 200.0 | 2 | 40 / 40 |
| cap_only_wide | 1 (profile) / 8 (caller) | buy / hold | 80.0 | 2.5 / 5.0 | 200.0 / 500.0 | 2 | 40 / 100 |
| both_overrides | 0.25 (caller) / 0.5 (caller) | buy / hold | 80.0 | 0.5 / 0.5 | 0.0 / 0.0 | 0 | 0 / 0 |
| tiny_risk | 0.001 (caller) / 10 (profile) | buy / hold | 80.0 | 0.005 / 0.005 | 0.0 / 0.0 | 0 | 0 / 0 |
| tiny_cap | 1 (profile) / 0.06 (caller) | buy / hold | 80.0 | 0.06 / 0.06 | 0.0 / 0.0 | 0 | 0 / 0 |
| small_account | 1 (profile) / 10 (profile) | buy / hold | 48.0 | 2.5 / 8.0 | 0.0 / 50.0 | 0 | 0 / 2 |
| no_account | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 5.0 | None / None | None | — / — |
| no_atr | 1 (profile) / 10 (profile) | buy / hold | None | None / None | None / None | None | — / — |
| negative_stop | 1 (profile) / 10 (profile) | starter / hold | -0.5 | None / None | None / None | None | — / — |
| inverted_band | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 5.0 | 200.0 / 500.0 | 2 | 40 / 100 |
| starter | 1 (profile) / 10 (profile) | starter / hold | 80.0 | 2.5 / None | 200.0 / None | 2 | 40 / — |
| earnings_blocked | 1 (profile) / 10 (profile) | wait / hold | 80.0 | None / None | None / None | None | — / — |
| speculative_mode_override | 1 (profile) / 0.5 (caller) | buy / hold | 56.0 | 0.5 / 0.5 | 0.0 / 0.0 | 0 | 0 / 0 |

## Swing outputs

| Case | Whole shares | Fractional shares | Position dollars | Reported max_loss_dollars |
|---|---|---|---|---|
| fractional_amounts | 8 | None | 800.0 | 24.0 |
| zero_whole_shares | 0 | None | 0.0 | 0.0 |
| no_account | None | None | None | None |

## Scan-promoted card

The public scan row contains no size. These values come from the captured real card
at the scan fixture's own clock, 2026-03-10T08:00:00-04:00, with synthetic data and temporary state.

| Action | Whole shares | Fractional shares | Position dollars | Reported max_loss_dollars |
|---|---|---|---|---|
| enter_on_trigger | 7 | None | 783.72 | 7.84 |

## Changed numerical fields

Every changed pre-existing numerical sizing field is listed below; duplicated detail tiers are omitted.
New canonical scenarios expose ceilings, actual whole-share amounts and conditional loss separately.
Core sizes are null because core is long-term; starter-only and blocked actions hide inapplicable sizes.
Missing/invalid stops suppress estimates; inverted entry bands lose both bounds. Fractional estimates are null.
The policy decisions and every saved snapshot field are checked against the frozen baseline.

| Path | Before | After | Reason |
|---|---|---|---|
| `analyze.balanced_extreme_wide.card.sizing.fractional_shares` | 0.5 | None | Retired fractional estimate |
| `analyze.balanced_extreme_wide.card.sizing.starter_dollars` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_extreme_wide.modes.balanced.starter_position.dollars` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.card.sizing.fractional_shares` | 2.0 | None | Retired fractional estimate |
| `analyze.balanced_high_wide.card.sizing.full_dollars` | 600.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.card.sizing.full_pct_cap` | 6.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.balanced.max_position.dollars` | 600.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.balanced.max_position.pct` | 6.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.balanced.max_position.shares` | 6 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.core.max_position.dollars` | 600.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.core.max_position.pct` | 6.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.core.max_position.shares` | 6 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.zones.position_sizing_range.dollars_for_account.max` | 600.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.zones.position_sizing_range.max_pct` | 6.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.zones.position_sizing_range.shares_range.max` | 6 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_high_wide.zones.position_sizing_range.suggested_pct_range[1]` | 6.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.balanced_low_wide.card.sizing.full_dollars` | 1000.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.card.sizing.full_pct_cap` | 10.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.card.sizing.starter_dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.balanced.max_position.dollars` | 1000.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.balanced.max_position.pct` | 10.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.balanced.max_position.shares` | 10 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.zones.position_sizing_range.dollars_for_account.max` | 1000.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.zones.position_sizing_range.max_pct` | 10.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.zones.position_sizing_range.shares_range.max` | 10 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_low_wide.zones.position_sizing_range.suggested_pct_range[1]` | 10.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.balanced_narrow.card.sizing.starter_dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_narrow.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.balanced_wide.card.sizing.full_dollars` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.card.sizing.full_pct_cap` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.card.sizing.starter_dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.balanced.max_position.dollars` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.balanced.max_position.pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.balanced.max_position.shares` | 8 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.zones.position_sizing_range.dollars_for_account.max` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.zones.position_sizing_range.max_pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.zones.position_sizing_range.shares_range.max` | 8 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.balanced_wide.zones.position_sizing_range.suggested_pct_range[1]` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.card.sizing.fractional_shares` | 0.5 | None | Retired fractional estimate |
| `analyze.both_overrides.card.sizing.full_dollars` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.card.sizing.starter_dollars` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.balanced.max_position.dollars` | 800.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.balanced.max_position.pct` | 8.0 | 0.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.balanced.max_position.shares` | 8 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.balanced.starter_position.dollars` | 250.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.balanced.starter_position.pct` | 2.5 | 0.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.balanced.starter_position.shares` | 2 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.zones.position_sizing_range.dollars_for_account.max` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.zones.position_sizing_range.dollars_for_account.min` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.both_overrides.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.card.sizing.fractional_shares` | 2.0 | None | Retired fractional estimate |
| `analyze.cap_only_tight.modes.balanced.max_position.dollars` | 800.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.balanced.max_position.pct` | 8.0 | 2.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.balanced.max_position.shares` | 8 | 2 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.balanced.starter_position.pct` | 2.5 | 2.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_tight.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.cap_only_wide.card.sizing.full_dollars` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.card.sizing.full_pct_cap` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.card.sizing.starter_dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.balanced.max_position.dollars` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.balanced.max_position.pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.balanced.max_position.shares` | 8 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.zones.position_sizing_range.dollars_for_account.max` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.zones.position_sizing_range.max_pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.zones.position_sizing_range.shares_range.max` | 8 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.cap_only_wide.zones.position_sizing_range.suggested_pct_range[1]` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.card.sizing.fractional_shares` | 2.0 | None | Retired fractional estimate |
| `analyze.core_narrow.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.modes.balanced.starter_position.pct` | 2.5 | 2.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_narrow.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.card.sizing.fractional_shares` | 2.0 | None | Retired fractional estimate |
| `analyze.core_wide.card.sizing.full_dollars` | 800.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.card.sizing.full_pct_cap` | 8.0 | 2.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.balanced.max_position.dollars` | 800.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.balanced.max_position.pct` | 8.0 | 2.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.balanced.max_position.shares` | 8 | 2 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.balanced.starter_position.pct` | 2.5 | 2.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.zones.position_sizing_range.dollars_for_account.max` | 800.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.zones.position_sizing_range.max_pct` | 8.0 | 2.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.zones.position_sizing_range.shares_range.max` | 8 | 2 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.core_wide.zones.position_sizing_range.suggested_pct_range[1]` | 8.0 | 2.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.card.entry.buy_zone_high` | 101.0 | None | Validated entry band |
| `analyze.inverted_band.card.entry.buy_zone_low` | 115.5 | None | Validated entry band |
| `analyze.inverted_band.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.inverted_band.card.sizing.full_dollars` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.card.sizing.full_pct_cap` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.card.sizing.starter_dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.balanced.max_position.dollars` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.balanced.max_position.pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.balanced.max_position.shares` | 8 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.zones.position_sizing_range.dollars_for_account.max` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.zones.position_sizing_range.max_pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.zones.position_sizing_range.shares_range.max` | 8 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.inverted_band.zones.position_sizing_range.suggested_pct_range[1]` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.missing_atr_pct.card.sizing.starter_dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.missing_atr_pct.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.negative_stop.card.entry.buy_zone_high` | 4.04 | None | Validated entry band |
| `analyze.negative_stop.card.entry.buy_zone_low` | 4.06 | None | Validated entry band |
| `analyze.negative_stop.card.sizing.at_price` | 4.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.negative_stop.card.sizing.fractional_shares` | 12.5 | None | Retired fractional estimate |
| `analyze.negative_stop.card.sizing.starter_dollars` | 50.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.negative_stop.card.sizing.starter_pct` | 0.5 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.negative_stop.card.sizing.whole_shares` | 12 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.negative_stop.modes.balanced.starter_position.dollars` | 50.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.negative_stop.modes.balanced.starter_position.pct` | 0.5 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.negative_stop.modes.balanced.starter_position.shares` | 12 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_account.card.sizing.full_pct_cap` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_account.modes.balanced.max_position.pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_account.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_account.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_account.zones.position_sizing_range.max_pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_account.zones.position_sizing_range.suggested_pct_range[1]` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.card.sizing.at_price` | 100.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.no_atr.card.sizing.full_dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.card.sizing.full_pct_cap` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.card.sizing.starter_dollars` | 250.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.card.sizing.starter_pct` | 2.5 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.card.sizing.whole_shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.balanced.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.balanced.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.balanced.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.balanced.starter_position.dollars` | 250.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.balanced.starter_position.pct` | 2.5 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.balanced.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.dollars_for_account.max` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.dollars_for_account.min` | 250.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.dollars_for_account.portfolio_assumption` | 10000.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.max_pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.profile_cap_pct` | 10.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.risk_per_trade_pct` | 1.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.shares_range.at_price` | 100.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.shares_range.max` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.shares_range.min` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.no_atr.zones.position_sizing_range.starter_pct` | 2.5 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.risk_only.card.sizing.full_dollars` | 800.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.card.sizing.full_pct_cap` | 8.0 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.card.sizing.starter_dollars` | 250.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.card.sizing.starter_pct` | 2.5 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.card.sizing.whole_shares` | 2 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.balanced.max_position.dollars` | 800.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.balanced.max_position.pct` | 8.0 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.balanced.max_position.shares` | 8 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.balanced.starter_position.dollars` | 250.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.balanced.starter_position.pct` | 2.5 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.balanced.starter_position.shares` | 2 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.dollars_for_account.max` | 800.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.max_pct` | 8.0 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.shares_range.max` | 8 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.shares_range.min` | 2 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.starter_pct` | 2.5 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 120.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.stop_implied_max.pct` | 1.2 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.suggested_pct_range[0]` | 2.5 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.risk_only.zones.position_sizing_range.suggested_pct_range[1]` | 8.0 | 1.25 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.card.sizing.fractional_shares` | 2.0202 | None | Retired fractional estimate |
| `analyze.rounded_stop_warning.card.sizing.full_dollars` | 600.0 | 594.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.card.sizing.starter_dollars` | 200.0 | 198.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.balanced.max_position.dollars` | 600.0 | 594.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.balanced.starter_position.dollars` | 200.0 | 198.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.core.max_position.dollars` | 600.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.core.max_position.pct` | 6.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.core.max_position.shares` | 6 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.modes.speculative.starter_position.dollars` | 100.0 | 99.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.zones.position_sizing_range.dollars_for_account.max` | 600.0 | 594.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.zones.position_sizing_range.dollars_for_account.min` | 200.0 | 198.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 600.0 | 594.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.rounded_stop_warning.zones.position_sizing_range.stop_implied_max.stop_distance_pct` | 4.0 | 4.04040404040404 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.card.sizing.fractional_shares` | 0.5 | None | Retired fractional estimate |
| `analyze.small_account.card.sizing.full_dollars` | 80.0 | 50.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.card.sizing.starter_dollars` | 25.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.balanced.max_position.dollars` | 80.0 | 50.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.balanced.starter_position.dollars` | 25.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.core.max_position.dollars` | 80.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.core.max_position.shares` | 1 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.core.starter_position.dollars` | 20.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.core.starter_position.shares` | 0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.modes.speculative.starter_position.dollars` | 10.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.zones.position_sizing_range.dollars_for_account.max` | 80.0 | 50.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.zones.position_sizing_range.dollars_for_account.min` | 25.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.small_account.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 80.0 | 50.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.card.sizing.fractional_shares` | 1.0 | None | Retired fractional estimate |
| `analyze.speculative_binding.card.sizing.full_dollars` | 600.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.card.sizing.full_pct_cap` | 6.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.balanced.max_position.dollars` | 800.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.balanced.max_position.pct` | 8.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.balanced.max_position.shares` | 8 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.balanced.starter_position.dollars` | 250.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.balanced.starter_position.pct` | 2.5 | 1.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.balanced.starter_position.shares` | 2 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.zones.position_sizing_range.dollars_for_account.max` | 600.0 | 500.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.zones.position_sizing_range.max_pct` | 6.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.zones.position_sizing_range.shares_range.max` | 6 | 5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_binding.zones.position_sizing_range.suggested_pct_range[1]` | 6.0 | 5.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.card.sizing.fractional_shares` | 0.8333 | None | Retired fractional estimate |
| `analyze.speculative_mode_override.card.sizing.full_dollars` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.card.sizing.starter_dollars` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.balanced.max_position.dollars` | 800.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.balanced.max_position.pct` | 8.0 | 0.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.balanced.max_position.shares` | 13 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.balanced.starter_position.dollars` | 250.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.balanced.starter_position.pct` | 2.5 | 0.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.balanced.starter_position.shares` | 4 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.core.max_position.shares` | 13 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.core.starter_position.shares` | 3 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.speculative.starter_position.dollars` | 100.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.speculative.starter_position.pct` | 1.0 | 0.5 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.modes.speculative.starter_position.shares` | 1 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.zones.position_sizing_range.dollars_for_account.max` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.zones.position_sizing_range.dollars_for_account.min` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 50.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_mode_override.zones.position_sizing_range.stop_implied_max.stop_distance_pct` | 6.7 | 6.666666666666665 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.card.sizing.fractional_shares` | 1.0 | None | Retired fractional estimate |
| `analyze.speculative_narrow.modes.balanced.max_position.dollars` | 800.0 | 600.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.balanced.max_position.pct` | 8.0 | 6.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.balanced.max_position.shares` | 8 | 6 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.balanced.starter_position.dollars` | 250.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.balanced.starter_position.pct` | 2.5 | 1.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.balanced.starter_position.shares` | 2 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_narrow.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.card.sizing.fractional_shares` | 1.0 | None | Retired fractional estimate |
| `analyze.speculative_wide.modes.balanced.max_position.dollars` | 800.0 | 600.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.balanced.max_position.pct` | 8.0 | 6.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.balanced.max_position.shares` | 8 | 6 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.balanced.starter_position.dollars` | 250.0 | 100.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.balanced.starter_position.pct` | 2.5 | 1.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.balanced.starter_position.shares` | 2 | 1 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.speculative_wide.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.starter.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.starter.card.sizing.starter_dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.starter.modes.balanced.starter_position.dollars` | 250.0 | 200.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.starter.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.starter.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.starter.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.card.sizing.fractional_shares` | 0.1 | None | Retired fractional estimate |
| `analyze.tiny_cap.card.sizing.full_dollars` | 10.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.card.sizing.full_pct_cap` | 0.1 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.card.sizing.starter_dollars` | 10.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.card.sizing.starter_pct` | 0.1 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.balanced.max_position.dollars` | 800.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.balanced.max_position.pct` | 8.0 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.balanced.max_position.shares` | 8 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.balanced.starter_position.dollars` | 250.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.balanced.starter_position.pct` | 2.5 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.balanced.starter_position.shares` | 2 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.dollars_for_account.max` | 10.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.dollars_for_account.min` | 10.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.max_pct` | 0.1 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.starter_pct` | 0.1 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 10.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.stop_implied_max.pct` | 0.1 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.suggested_pct_range[0]` | 0.1 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_cap.zones.position_sizing_range.suggested_pct_range[1]` | 0.1 | 0.06 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.card.sizing.fractional_shares` | 2.5 | None | Retired fractional estimate |
| `analyze.tiny_risk.card.sizing.full_dollars` | 800.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.card.sizing.full_pct_cap` | 8.0 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.card.sizing.starter_dollars` | 250.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.card.sizing.starter_pct` | 2.5 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.card.sizing.whole_shares` | 2 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.balanced.max_position.dollars` | 800.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.balanced.max_position.pct` | 8.0 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.balanced.max_position.shares` | 8 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.balanced.starter_position.dollars` | 250.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.balanced.starter_position.pct` | 2.5 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.balanced.starter_position.shares` | 2 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.core.max_position.dollars` | 800.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.core.max_position.pct` | 8.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.core.max_position.shares` | 8 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.core.starter_position.dollars` | 200.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.core.starter_position.pct` | 2.0 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.modes.core.starter_position.shares` | 2 | None | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.dollars_for_account.max` | 800.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.dollars_for_account.min` | 250.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.max_pct` | 8.0 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.risk_per_trade_pct` | 1.0 | 0.001 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.shares_range.max` | 8 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.shares_range.min` | 2 | 0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.starter_pct` | 2.5 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.stop_implied_max.dollars_for_account` | 500.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.stop_implied_max.pct` | 5.0 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.stop_implied_max.risk_per_trade_pct` | 1.0 | 0.001 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.suggested_pct_range[0]` | 2.5 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `analyze.tiny_risk.zones.position_sizing_range.suggested_pct_range[1]` | 8.0 | 0.005 | Risk/cap enforcement and consistent whole-share amounts |
| `scan.promoted_card.plan.fractional_shares` | 7.3687 | None | Retired fractional estimate |
| `scan.promoted_card.plan.max_loss_dollars` | 8.25 | 7.84 | Risk/cap enforcement and consistent whole-share amounts |
| `scan.promoted_card.plan.position_dollars` | 825.0 | 783.72 | Risk/cap enforcement and consistent whole-share amounts |
| `scan.promoted_card.plan.position_pct` | 8.25 | 7.8372 | Risk/cap enforcement and consistent whole-share amounts |
| `swing.fractional_amounts.plan.fractional_shares` | 8.25 | None | Retired fractional estimate |
| `swing.fractional_amounts.plan.max_loss_dollars` | 24.75 | 24.0 | Risk/cap enforcement and consistent whole-share amounts |
| `swing.fractional_amounts.plan.position_dollars` | 825.0 | 800.0 | Risk/cap enforcement and consistent whole-share amounts |
| `swing.fractional_amounts.plan.position_pct` | 8.25 | 8.0 | Risk/cap enforcement and consistent whole-share amounts |
| `swing.zero_whole_shares.plan.fractional_shares` | 0.8 | None | Retired fractional estimate |
| `swing.zero_whole_shares.plan.max_loss_dollars` | 2.4 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `swing.zero_whole_shares.plan.position_dollars` | 80.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
| `swing.zero_whole_shares.plan.position_pct` | 8.0 | 0.0 | Risk/cap enforcement and consistent whole-share amounts |
