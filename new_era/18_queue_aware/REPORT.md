# 018 Queue-Aware Stress Test Report

## 1. Purpose (Not a profitability test)

018 is a stress audit of frozen candidate cancel policies. It is not a production profitability claim and it does not add features, change thresholds, or select a new model on test outcomes. The purpose is to ask whether defensive cancel edge survives when queue/fill assumptions become less favorable.

## 2. Pre-018 Freeze

Both frozen candidates passed the 13-item freeze checklist in `pre_018_model_freeze_report.csv`. The common test sessions are `20260127Day` and `20260127Night`; the common quote-eligible universe has `1,166,321` rows. The stress engine also reproduces the frozen base PnL exactly in `stress_reproducibility_check.csv`.

## 3. Stress Definitions

The moderate setting used for the main reporting claim is `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, and `cancel_delay_ticks=0`.

- `neutral_fill_haircut`: uniformly keeps only a fixed fraction of realized maker fills.
- `good_fill_haircut`: preferentially removes favorable fills and is reserved for pessimistic appendix analysis.
- `toxic_multiplier`: increases adverse fill damage after a fill.
- `cancel_delay_ticks`: delays cancel effectiveness by event ticks.

Fixed random masks are recorded in `random_mask_manifest.json`.

## 4. Base PnL

- no_cancel: `63,785`
- matched random_v3 mean: `25,046.086`
- 015_common_universe: `90,845`
- 016_v2_binary: `132,590`

The original historical 015 result remains `170,625`; the common-universe 018 result is `90,845` because the evaluation universe was changed for apples-to-apples comparison.

## 5. Moderate Stress Result

Moderate stress makes absolute profitability fail, but relative defense survives.

- no_cancel moderate PnL: `-97,270`
- random_v3_mean moderate PnL: `-85,569.435`
- random_v3_95p moderate PnL: `-37,463.335`
- 015_common moderate PnL: `-52,345`; gap vs no-cancel `44,925`; gap vs random_v3_mean `33,224.435`; gap vs random_v3_95p `-14,881.665`.
- 016_v2_binary moderate PnL: `-38,490`; gap vs no-cancel `58,780`; gap vs random_v3_mean `47,079.435`; gap vs random_v3_95p `-1,026.665`.

The relative pass is against no-cancel and random_v3_mean. Neither candidate clears the random_v3_95p hurdle under moderate stress.

Decision status: both candidates are `Partial Pass`.

## 6. Heatmap / Break-even Frontier

Report-ready figures:

- `queue_stress_heatmap_015_common.png`
- `queue_stress_heatmap_016_v2_binary.png`
- `queue_stress_gap_vs_no_cancel_015.png`
- `queue_stress_gap_vs_no_cancel_016.png`
- `queue_stress_gap_vs_random_v3_015.png`
- `queue_stress_gap_vs_random_v3_016.png`

The break-even table is in `break_even_frontier.csv` and the narrative summary is in `break_even_frontier.md`. The main visual message is that the green absolute-PnL region is fragile under moderate queue/fill stress, while positive relative gaps can remain.

## 7. Delay Sensitivity (FPGA framing caveat)

Delay sensitivity is used only to motivate latency risk-control. Do not claim FPGA necessity is proven.

The delay sweep remains diagnostic: 016_v2_binary declines from `132,590` at delay 0 to `81,520` at delay 10, while 015_common is non-monotone but still latency-sensitive. This supports studying low-latency risk-control paths, not claiming a proven production FPGA requirement.

## 8. Final Interpretation

Under moderate queue/fill stress, both 015_common and 016_v2_binary fail the absolute profitability criterion but retain relative defensive value against no-cancel and matched random baselines. Therefore, 018 is a Partial Pass...

This result should be presented as a conservative robustness audit: the cancel policies are not proof of real-world HFT profitability, but they do preserve defensive signal value under a harsher proxy execution framework. The next step is queue-aware sensitivity and execution realism, not model expansion.
