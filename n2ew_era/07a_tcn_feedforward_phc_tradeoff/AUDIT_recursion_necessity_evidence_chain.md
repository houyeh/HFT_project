status: mixed
finding: recurrence_not_required_by_current_diagnostics_but_tcn_hls_latency_not_yet_optimized
# Audit: Recursion Necessity Evidence Chain

## Scope and claim boundary

This audit formalizes the connection between Phase 05A memory ablations and Phase 07A feedforward TCN diagnostics. No NEW5 data was used. No 05A or 07A checkpoint was retrained. NEW3 and NEW4 numbers below are frozen post-hoc diagnostic outputs, not model-selection inputs. No implementation, P&R, bitstream, or on-board output was generated.

Verification labels used in this file:

| label | meaning |
| --- | --- |
| `training-eval` | Existing frozen model evaluation on the named split. |
| `matched-random-tested` | Existing selected-policy diagnostic against matched random baseline. |
| `pre-synthesis MAC estimate, not HLS-verified` | 07A analytical MAC-rate latency estimate only. |
| `HLS CSim` | 07B C++ HLS simulation against the prepared 200-vector reference set. |
| `HLS csynth estimate` | Vitis HLS C-synthesis report. This is not RTL cosim, implementation, routed timing, or board evidence. |
| `routed P&R` | Existing PHC-lite routed 100MHz result. |

## Evidence 1: learned recurrence parameters are not the dominant source

Source: `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md`.

The frozen PHC-lite v1 checkpoint was evaluated under inference-only counterfactuals. Fixed decay 0.9 stayed close to baseline, but disabling memory or corrupting time order collapsed performance.

| ablation | split | action PnL delta vs PHC baseline | verification |
| --- | --- | ---: | --- |
| fixed_decay_0p9 | val | +470 | training-eval, post-hoc ablation |
| fixed_decay_0p9 | new3 | +460 | training-eval, frozen diagnostic |
| fixed_decay_0p9 | new4 | -7,505 | training-eval, frozen diagnostic |
| memory_off_decay_0 | val | -803,235 | training-eval, post-hoc ablation |
| memory_off_decay_0 | new3 | -1,899,740 | training-eval, frozen diagnostic |
| memory_off_decay_0 | new4 | -781,695 | training-eval, frozen diagnostic |
| reverse_sequence | val | -2,764,500 | training-eval, post-hoc ablation |
| reverse_sequence | new3 | -6,947,680 | training-eval, frozen diagnostic |
| reverse_sequence | new4 | -2,530,160 | training-eval, frozen diagnostic |
| deterministic_shuffle_timesteps | val | -2,569,590 | training-eval, post-hoc ablation |
| deterministic_shuffle_timesteps | new3 | -6,518,720 | training-eval, frozen diagnostic |
| deterministic_shuffle_timesteps | new4 | -2,314,520 | training-eval, frozen diagnostic |

Readout: the evidence supports "sequence order plus gated temporal state is important." It does not support a stronger claim that every learned per-channel decay parameter is essential.

## Evidence 2: frozen feedforward TCN matches or exceeds the recurrent teacher

Source: regenerated `n2ew_era/07a_tcn_feedforward_phc_tradeoff/TCN_FEEDFORWARD_PHC_TRADEOFF_REPORT.md` and `results/tcn_vs_phc_tradeoff_summary.csv`.

The final report bug was fixed: selected PHC and TCN rows now include `matched_random_pass=True`, `empirical_p_value_fraction_random_ge_model=0`, and `< 0.001` wording from the existing matched-random diagnostics. The former "Seed stability summary missing" issue is also fixed; the regenerated seed summary reads `matched_random_pass_all=True`, `empirical_p_value_max=0`, and `seed_count=3` for every variant/split row.

| model | split | action PnL delta vs PHC | causal-fill PnL delta vs PHC | matched-random result | latency label |
| --- | --- | ---: | ---: | --- | --- |
| tcn_prepad_scratch_16feat | val | -6,280 | +1,885 | pass, empirical p=0 (`< 0.001`) | 32.656 us, pre-synthesis MAC estimate, not HLS-verified |
| tcn_prepad_scratch_16feat | new3 | -15,590 | +76,390 | pass, empirical p=0 (`< 0.001`) | 32.656 us, pre-synthesis MAC estimate, not HLS-verified |
| tcn_prepad_scratch_16feat | new4 | +32,785 | +5,150 | pass, empirical p=0 (`< 0.001`) | 32.656 us, pre-synthesis MAC estimate, not HLS-verified |
| tcn_prepad_distilled_16feat | val | +44,380 | +3,340 | pass, empirical p=0 (`< 0.001`) | 32.656 us, pre-synthesis MAC estimate, not HLS-verified |
| tcn_prepad_distilled_16feat | new3 | +69,715 | +98,925 | pass, empirical p=0 (`< 0.001`) | 32.656 us, pre-synthesis MAC estimate, not HLS-verified |
| tcn_prepad_distilled_16feat | new4 | +51,960 | +11,955 | pass, empirical p=0 (`< 0.001`) | 32.656 us, pre-synthesis MAC estimate, not HLS-verified |

Seed stability source: `results/tcn_prepad_seed_variance_summary.csv`.

| variant | split | action PnL mean | action PnL std | causal-fill PnL mean | causal-fill PnL std | matched_random_pass_all | empirical_p_value_max | seed_count |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| scratch | val | 1,903,000.00 | 17,945.25 | 166,205.00 | 5,208.29 | True | 0 | 3 |
| scratch | new3 | 4,939,000.00 | 58,052.19 | 563,015.00 | 8,926.80 | True | 0 | 3 |
| scratch | new4 | 1,763,775.00 | 31,853.02 | 129,323.33 | 3,755.00 | True | 0 | 3 |
| distilled | val | 1,944,405.00 | 24,331.98 | 175,273.33 | 1,854.86 | True | 0 | 3 |
| distilled | new3 | 5,055,366.67 | 44,211.77 | 578,973.33 | 8,408.57 | True | 0 | 3 |
| distilled | new4 | 1,801,591.67 | 11,186.57 | 130,381.67 | 4,146.58 | True | 0 | 3 |

## Cross-split pooled test

Source: `results/tcn_vs_phc_pooled_significance.csv`.

The exact sign tests and exact split-level bootstrap CIs use the three frozen diagnostic splits as the resampling units. With only three split-level observations, the sign test has low power; it is used here to avoid visual inference from three point estimates.

| model | metric | n splits | positive deltas | delta sum | delta mean | exact sign-test p | exact bootstrap CI95 for mean | formal readout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| scratch | action PnL delta | 3 | 1 | 10,915 | 3,638.33 | 1.00 | [-13,572.83, 24,320.92] | does not reject equal median at 5% |
| scratch | causal-fill PnL delta | 3 | 3 | 83,425 | 27,808.33 | 0.25 | [2,592.42, 60,954.67] | does not reject equal median at 5% |
| distilled | action PnL delta | 3 | 3 | 166,055 | 55,351.67 | 0.25 | [46,022.33, 65,868.08] | does not reject equal median at 5% |
| distilled | causal-fill PnL delta | 3 | 3 | 114,220 | 38,073.33 | 0.25 | [5,206.58, 80,081.50] | does not reject equal median at 5% |

Formal statistical readout: scratch action PnL is within observed split-level noise because its bootstrap CI includes zero and the sign test p-value is 1.00. The distilled rows and causal-fill rows are directionally positive, but with only three split units the exact sign test still does not establish superiority at 5%. The defensible claim is "no evidence of feedforward inferiority under these frozen diagnostics," not "formally proven TCN superiority."

## Part B: 07B HLS CSim and csynth status

Source phase: `n2ew_era/07b_tcn_prepad_manual_hls/`.

CSim was executed against the prepared 200-vector test sets. The Vitis `vitis-run --mode hls --csim` wrapper hung after reporting `mingw32-make: 'csim.exe' is up to date`, so the generated `csim.exe` was run directly with the same configured argv from each `hls_config.cfg`. C synthesis was then rerun with `v++ -c --mode hls --config hls_config.cfg --work_dir hls_component` for both variants. No cosim, implementation, P&R, bitstream, or board test was run.

Collector source: `n2ew_era/07b_tcn_prepad_manual_hls/03_collect_hls_results.py`.

| variant | CSim status | rows | failed rows | RMSE readout | max abs diff readout | sign agreement joint | CSim output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| scratch | pass | 200 / 200 | 0 | 0.085448 | 0.342693 | 1.000000 | `n2ew_era/07b_tcn_prepad_manual_hls/results/tcn_prepad_scratch_csim_out.csv` |
| distilled | pass | 200 / 200 | 0 | 0.305065 | 0.571899 | 1.000000 | `n2ew_era/07b_tcn_prepad_manual_hls/results/tcn_prepad_distilled_csim_out.csv` |

| variant | csynth status | latency cycles | latency at 100MHz | interval cycles | pipeline | LUT | FF | DSP | BRAM_18K | report |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| scratch | present, rerun 2026-07-28 | 1,370,385 | 13,703.85 us | 1,370,386 | no | 25,275 | 17,306 | 34 | 63 | `n2ew_era/07b_tcn_prepad_manual_hls/hls_component_scratch/hls_component/hls/syn/report/tcn_prepad_infer_csynth.rpt` |
| distilled | present, rerun 2026-07-28 | 1,370,385 | 13,703.85 us | 1,370,386 | no | 25,619 | 18,325 | 48 | 63 | `n2ew_era/07b_tcn_prepad_manual_hls/hls_component_distilled/hls_component/hls/syn/report/tcn_prepad_infer_csynth.rpt` |

Comparison against existing latency claims:

| artifact | latency | verification level | note |
| --- | ---: | --- | --- |
| PHC-lite v8 | 83.16 us | routed P&R at 100MHz | Existing Phase 06F/06O routed result. |
| TCN 07A analytical estimate | 32.656 us | pre-synthesis MAC estimate, not HLS-verified | Must not be cited as measured FPGA latency. |
| TCN 07B scratch manual HLS | 13,703.85 us | HLS csynth estimate only | Much slower than the analytical MAC estimate in the current non-pipelined full-window implementation. |
| TCN 07B distilled manual HLS | 13,703.85 us | HLS csynth estimate only | Same latency as scratch; resource mix differs because weights differ. |

The 07B manual HLS result does not validate the 32.656 us analytical estimate. It shows that the prepared full-window C++ implementation synthesizes and passes CSim, but its current schedule is not a pipelined streaming TCN datapath.

## Evidence-chain conclusion

The current evidence supports a limited but useful claim: this task appears to need causal sequence order and a stateful or windowed temporal computation, but the learned recurrent decay mechanism itself is not established as necessary. Phase 05A shows fixed decay is nearly tied with learned decay while memory-off and time-order corruption collapse performance. Phase 07A shows a fully feedforward TCN can match or exceed the recurrent PHC teacher in frozen split diagnostics, with matched-random evidence and three-seed stability.

The honest wording is therefore: recurrence is not proven necessary for the trading signal under the current diagnostics, and feedforward temporal architectures remain a credible path toward pipelineable hardware. The stronger wording is not allowed: the 32.656 us TCN number is only a pre-synthesis MAC estimate, not HLS-verified, and the 07B manual HLS implementation currently csynthesizes to 1,370,385 cycles, or 13,703.85 us at 100MHz. More hardware work is needed before claiming a fast TCN FPGA implementation.
