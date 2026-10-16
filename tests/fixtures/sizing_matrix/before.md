# Sizing: v0.3.0 before table

Generated from synthetic offline scenarios by `tests.support.sizing_matrix`. Do not edit by hand.
This records defects, not recommendations. Details, full snapshots and limit origins: `before.json`.
The loss column is computed for this report from the displayed dollars / price × stop distance;
it is not a field emitted by v0.3.0 and can disagree with its whole-share count.

Risk/maximum below are the resolved profile-helper values, not necessarily the raw input.
For `tiny_risk`, the supplied 0.001% rounds to 0 here, then sizing falls back to 1%.

| Case | Resolved risk / maximum (%), origin | New money / held | Stop | Starter / full (%) | Starter / full ($) | Starter whole shares | Implied loss starter / full ($) |
|---|---|---|---|---|---|---|---|
| core_narrow | 0.5 (profile) / 8 (profile) | buy / hold | 96.0 | 2.0 / 8.0 | 200.0 / 800.0 | 2 | 8 / 32 |
| core_wide | 0.5 (profile) / 8 (profile) | buy / hold | 80.0 | 2.0 / 8.0 | 200.0 / 800.0 | 2 | 40 / 160 |
| balanced_narrow | 1 (profile) / 10 (profile) | buy / hold | 96.0 | 2.5 / 8.0 | 250.0 / 800.0 | 2 | 10 / 32 |
| balanced_wide | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 8.0 | 250.0 / 800.0 | 2 | 50 / 160 |
| speculative_narrow | 1.5 (profile) / 6 (profile) | buy / hold | 96.0 | 1.0 / 6.0 | 100.0 / 600.0 | 1 | 4 / 24 |
| speculative_wide | 1.5 (profile) / 6 (profile) | buy / hold | 80.0 | 1.0 / 6.0 | 100.0 / 600.0 | 1 | 20 / 120 |
| balanced_low_wide | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 10.0 | 250.0 / 1000.0 | 2 | 50 / 200 |
| balanced_high_wide | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.0 / 6.0 | 200.0 / 600.0 | 2 | 40 / 120 |
| balanced_extreme_wide | 1 (profile) / 10 (profile) | starter / hold | 62.5 | 0.5 / None | 50.0 / None | 0 | 18.75 / — |
| rounded_stop_warning | 1 (profile) / 10 (profile) | buy / hold | 95.0 | 2.0 / 6.0 | 200.0 / 600.0 | 2 | 8.08081 / 24.2424 |
| missing_atr_pct | 1 (profile) / 10 (profile) | buy / hold | 96.0 | 2.5 / 8.0 | 250.0 / 800.0 | 2 | 10 / 32 |
| speculative_binding | 1.5 (profile) / 6 (profile) | buy / hold | 70.0 | 1.0 / 6.0 | 100.0 / 600.0 | 1 | 30 / 180 |
| risk_only | 0.25 (caller) / 10 (profile) | buy / hold | 80.0 | 2.5 / 8.0 | 250.0 / 800.0 | 2 | 50 / 160 |
| cap_only_tight | 1 (profile) / 2 (caller) | buy / hold | 80.0 | 2.0 / 2.0 | 200.0 / 200.0 | 2 | 40 / 40 |
| cap_only_wide | 1 (profile) / 8 (caller) | buy / hold | 80.0 | 2.5 / 8.0 | 250.0 / 800.0 | 2 | 50 / 160 |
| both_overrides | 0.25 (caller) / 0.5 (caller) | buy / hold | 80.0 | 0.5 / 0.5 | 50.0 / 50.0 | 0 | 10 / 10 |
| tiny_risk | 0 (caller) / 10 (profile) | buy / hold | 80.0 | 2.5 / 8.0 | 250.0 / 800.0 | 2 | 50 / 160 |
| tiny_cap | 1 (profile) / 0.06 (caller) | buy / hold | 80.0 | 0.1 / 0.1 | 10.0 / 10.0 | 0 | 2 / 2 |
| small_account | 1 (profile) / 10 (profile) | buy / hold | 48.0 | 2.5 / 8.0 | 25.0 / 80.0 | 0 | 1 / 3.2 |
| no_account | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 8.0 | None / None | None | — / — |
| no_atr | 1 (profile) / 10 (profile) | buy / hold | None | 2.5 / 8.0 | 250.0 / 800.0 | 2 | — / — |
| negative_stop | 1 (profile) / 10 (profile) | starter / hold | -0.5 | 0.5 / None | 50.0 / None | 12 | — / — |
| inverted_band | 1 (profile) / 10 (profile) | buy / hold | 80.0 | 2.5 / 8.0 | 250.0 / 800.0 | 2 | 50 / 160 |
| starter | 1 (profile) / 10 (profile) | starter / hold | 80.0 | 2.5 / None | 250.0 / None | 2 | 50 / — |
| earnings_blocked | 1 (profile) / 10 (profile) | wait / hold | 80.0 | None / None | None / None | None | — / — |
| speculative_mode_override | 1 (profile) / 0.5 (caller) | buy / hold | 56.0 | 0.5 / 0.5 | 50.0 / 50.0 | 0 | 3.33333 / 3.33333 |

## Swing quantity mismatch

| Case | Whole shares | Fractional shares | Position dollars | Reported max_loss_dollars |
|---|---|---|---|---|
| fractional_amounts | 8 | 8.25 | 825.0 | 24.75 |
| zero_whole_shares | 0 | 0.8 | 80.0 | 2.4 |
| no_account | None | None | None | None |

## Scan-promoted card

The public scan row contains no size. These values come from the captured real card
at the scan fixture's own clock, 2026-03-10T08:00:00-04:00, with synthetic data and temporary state.

| Action | Whole shares | Fractional shares | Position dollars | Reported max_loss_dollars |
|---|---|---|---|---|
| enter_on_trigger | 7 | 7.3687 | 825.0 | 8.25 |
