# 16 Binary Label V2 Round 2 Audit Report

## 1. Executive Verdict

`16_binary_label_v2` remains the accepted quote-risk proxy candidate under the current proxy execution framework. Round 2 added score-bucket toxic lift, stricter matched-random baselines, per-session stability, baseline comparison, and validation/test threshold sensitivity diagnostics.

Main frozen claim:

- Model: F8 side-specific binary cancel model, validation-selected 10% cancel-rate threshold.
- Test net PnL: `132,590`.
- Test evaluation policy: frozen validation thresholds applied once to test.
- Empirical p-value definition: empirical p-value = fraction of matched random trials whose PnL >= model PnL.

The model beats the main strict random hurdles:

| Random baseline | Mean PnL | 95p PnL | 99p PnL | Empirical p-value |
| --- | ---: | ---: | ---: | ---: |
| random_v1, count+side | 63,785.00 | 63,785.00 | 63,785.00 | 0.0000 |
| random_v2, +session | 53,294.70 | 99,545.00 | 117,525.55 | 0.0020 |
| random_v3, +spread bucket | 25,046.09 | 77,561.75 | 99,320.75 | 0.0002 |

`random_v4` is produced as the strictest diagnostic with event-rate matching and hierarchical fallback. It is not marked diagnostic-only by the overlap rule because max `overlap_with_model_rate` is `0.3217`, below the `0.50` audit threshold; the primary statistical hurdle remains `v1/v2/v3`.

## 2. Reproducibility And Leakage Controls

Core reproducibility and leakage artifacts:

- `results/reproducibility_manifest.md`
- `results/leakage_audit.md`
- `results/binary_label_definition.md`
- `results/threshold_selection_report.md`
- `results/random_matching_health_report.csv`

Leakage controls:

- Causal z-scores use past-only `shift(1).rolling(...)` in `02_preprocess_quote_eligible.py:78`.
- No centered rolling windows are used.
- Future information is used only for labels, not features.
- Train balancing is isolated from validation/test distributions.
- Threshold selection writes `selection_split='val'` and `test_used=False`.
- Test backtest rejects non-validation threshold files and stamps `test_evaluation_count=1`.

The p-value is reported only as: empirical p-value = fraction of matched random trials whose PnL >= model PnL.

## 3. Quote-Eligible Universe And Labels

Quote-eligible rows remove invalid bid/ask, non-positive spreads, missing or non-positive depth, first `100` ticks and last `10` ticks per session, and non-finite feature rows.

Waterfall retention stayed near complete:

| Split | Raw rows | Quote-eligible rows | Retention |
| --- | ---: | ---: | ---: |
| train | 3,593,426 | 3,592,764 | 99.9816% |
| val | 563,593 | 563,477 | 99.9794% |
| test | 1,166,541 | 1,166,321 | 99.9811% |

Side-specific proxy labels:

- `bid_fill_proxy = bid_trade_hits_quote OR bid_price_cross_hits_quote`
- `ask_fill_proxy = ask_trade_hits_quote OR ask_price_cross_hits_quote`
- `bid_toxic = 1` iff `bid_fill_proxy == 1` and `future_mid_return_H_ticks <= -0.5`
- `ask_toxic = 1` iff `ask_fill_proxy == 1` and `future_mid_return_H_ticks >= +0.5`

The filter preserved hard events. On test, bid base toxic rate is `0.6643%`; ask base toxic rate is `0.7070%`.

## 4. Score Bucket Toxic Lift

Round 2 bucketed test scores into deciles plus top 10%, 5%, and 1% tails. Monotonicity is logged, not used as an automated pass/fail. Both bid and ask deciles were monotonic non-decreasing in this run.

Minimum requirement passed: top 10%, 5%, and 1% toxic rates are all above the base toxic rate for both sides.

| Side | Bucket | Toxic rate | Base toxic rate | Lift |
| --- | --- | ---: | ---: | ---: |
| bid | top 10% | 2.2755% | 0.6643% | 3.43x |
| bid | top 5% | 2.9134% | 0.6643% | 4.39x |
| bid | top 1% | 4.5010% | 0.6643% | 6.78x |
| ask | top 10% | 2.5199% | 0.7070% | 3.56x |
| ask | top 5% | 3.1929% | 0.7070% | 4.52x |
| ask | top 1% | 4.9640% | 0.7070% | 7.02x |

Artifacts:

- `results/score_bucket_toxic_lift.csv`
- `results/score_distribution_pos_neg.png`

## 5. Strict Matched Random Baselines

`15_strict_random_baseline.py` ran `5000` trials each for:

- `random_v1`: count + side
- `random_v2`: count + side + session
- `random_v3`: count + side + session + spread bucket
- `random_v4`: count + side + session + spread bucket + event-rate bucket with hierarchical fallback

The v4 fallback protocol is implemented as:

`v4_exact -> v4_coarse_eventrate -> v4_coarse_spread_eventrate -> v4_fallback_v3`

Health report summary:

- `random_v4` exact buckets: `37`
- `random_v4` coarse event-rate fallbacks: `9`
- `random_v4` coarse spread/event-rate fallbacks: `5`
- Replacement used: `False` for all logged rows
- Max v4 overlap with model: `0.3217`

Strict random results:

| Method | Mean | Std | 90p | 95p | 99p | Empirical p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_v1 | 63,785.00 | 0.00 | 63,785.00 | 63,785.00 | 63,785.00 | 0.0000 |
| random_v2 | 53,294.70 | 27,898.78 | 89,276.00 | 99,545.00 | 117,525.55 | 0.0020 |
| random_v3 | 25,046.09 | 30,991.80 | 66,366.50 | 77,561.75 | 99,320.75 | 0.0002 |
| random_v4 | 32,694.21 | 31,110.89 | 72,834.50 | 85,263.50 | 105,306.10 | 0.0002 |

Artifacts:

- `results/strict_random_baseline_distribution.csv`
- `results/strict_random_baseline_trials.csv`
- `results/strict_random_baseline_session_trials.csv`
- `results/random_matching_health_report.csv`

## 6. Per-Session Stability And Baseline Comparison

Per-session stability is positive but should be stated carefully. Against `random_v3`, both sessions beat random mean and 95p. The Day session also beats 99p; the Night session beats 95p but not 99p.

| Session | Model PnL | Share of total | random_v3 mean | random_v3 95p | random_v3 99p |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260127Day | 41,355.00 | 31.19% | -25,919.06 | 11,913.25 | 29,626.40 |
| 20260127Night | 91,235.00 | 68.81% | 50,965.15 | 88,391.25 | 104,116.20 |

This means the total `132,590` is not purely one outlier session, but session-level 99p dominance is not uniform.

Baseline comparison:

| Method | Test PnL | Improvement vs no-cancel | Trade count | Cancel count | PnL/trade | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_cancel | 63,785.00 | 0.00 | 661.0 | 0.0 | 96.50 | 76,230.00 |
| 015_smoothed | 170,625.00 | 106,840.00 | 705.0 | NA | 242.02 | 25,420.00 |
| 016_old_binary | 266,045.00 | 202,260.00 | 77.0 | 1,318,395.0 | 3,455.13 | 37,800.00 |
| 016_v2_binary | 132,590.00 | 68,805.00 | 474.0 | 333,425.0 | 279.73 | 69,390.00 |
| matched_random_v1 | 63,785.00 | 0.00 | 661.0 | 333,425.0 | 96.50 | 64,185.00 |
| matched_random_v3 | 25,046.09 | -38,738.91 | 482.3 | 333,425.0 | 51.93 | 74,298.33 |

The historical `016_old_binary` row is not a valid main claim because it was selected on NEW_test. The `015_smoothed` row remains a strong legacy reference, but it lacks quote-level cancel count in its artifact, so `cancel_count` is left as `NA` rather than guessed.

Artifacts:

- `results/pnl_by_session_model_vs_random.csv`
- `results/session_pnl_bar_model_vs_random.png`
- `results/baseline_comparison.csv`

## 7. Threshold Sensitivity

Validation-only sensitivity was run for target cancel rates `5%`, `7.5%`, `10%`, `12.5%`, `15%`, and `20%`. Only the validation-selected `10%` row has `is_main_claim=True` and `used_for_model_selection=True` in the validation sensitivity table.

Validation sensitivity:

| Target | Bid threshold | Ask threshold | Val PnL | Main claim |
| --- | ---: | ---: | ---: | --- |
| 5.0% | 0.0243065841 | 0.0315669775 | -52,065.00 | False |
| 7.5% | 0.0190258846 | 0.0249061901 | -16,100.00 | False |
| 10.0% | 0.0158583950 | 0.0209005475 | -25,460.00 | True |
| 12.5% | 0.0136965485 | 0.0181953143 | -34,200.00 | False |
| 15.0% | 0.0121350000 | 0.0162341781 | -36,480.00 | False |
| 20.0% | 0.0100187901 | 0.0135385729 | -45,440.00 | False |

Test threshold sensitivity is explicitly diagnostic only. Every row has `is_main_claim=False` and `used_for_model_selection=False`.

DIAGNOSTIC ONLY — NOT USED FOR MODEL SELECTION

| Val-selected target | Test bid cancel rate | Test ask cancel rate | Test PnL |
| --- | ---: | ---: | ---: |
| 5.0% | 7.5948% | 7.1385% | 119,855.00 |
| 7.5% | 11.2557% | 10.5180% | 121,470.00 |
| 10.0% | 14.7584% | 13.8294% | 132,590.00 |
| 12.5% | 18.2309% | 17.0711% | 132,750.00 |
| 15.0% | 21.7028% | 20.2281% | 107,540.00 |
| 20.0% | 28.5657% | 26.4125% | 76,470.00 |

The 12.5% diagnostic row is slightly higher on test, but it is not a model-selection result and must not be used to revise the main threshold. The frozen accepted claim remains the validation-selected 10% threshold.

Artifacts:

- `results/threshold_sensitivity_val.csv`
- `results/threshold_sensitivity_val_thresholds.json`
- `results/threshold_sensitivity_val.md`
- `results/threshold_sensitivity_test_diagnostic.csv`
- `results/threshold_sensitivity_test_diagnostic.md`

## 8. Conservative Conclusion

`16_binary_label_v2` passes Round 2 under a stronger statistical audit: score buckets show toxic lift, strict random v1/v2/v3 95p and 99p hurdles are beaten by the frozen test PnL, v4 hierarchical fallback is logged with health metrics, per-session stability is supportive though not uniformly 99p-dominant, and threshold sensitivity is clearly separated into validation selection versus diagnostic-only test reporting.

The correct interpretation remains conservative. The model is a side-specific quote-risk proxy signal built without true `aggressor_side` and without queue-position ground truth. It is not a claim of exchange-realized fill toxicity or deployable HFT profitability.

16_binary_label_v2 has passed a stronger statistical audit under the current proxy execution framework. It is now a valid candidate for 018 queue-aware sensitivity analysis. It is not yet proof of real-world HFT profitability or true exchange fill toxicity.
