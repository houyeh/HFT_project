# EVIDENCE PACK

Status: `evidence_pack_compiled_from_existing_artifacts_only`

Generated: 2026-09-24. This is a read-only synthesis of existing Markdown reports, AUDIT files, and `results/*.csv` under `n2ew_era/`. No experiment, training, HLS/Vivado run, or NEW5 re-evaluation was performed. Values retain their original validation tier and scope. When two phases use the same split name with different scopes, both scopes are stated rather than silently merged.

## Table 1 | Dataset and splits (6 rows)

`event_count` is the TXF-only, boundary-kept event count used by the PHC sequence pipeline except for `test` (the older Phase 01 protocol) and NEW5 (the frozen one-shot scope). Registry counts are included in `scope_and_dedup` so raw coverage is not confused with model-ready rows.

| split | source group / role | sessions | event_count | time range | dedup / validity result | scope_and_dedup | source_path |
|---|---|---:|---:|---|---|---|---|
| train | NEW2 / development train | 3 | 2,470,559 | 2026-01-21 to 2026-01-22 | registry: no excluded duplicate | Registry raw: 10,035,946 rows, 16 files; PHC windows: 77,194 | `n2ew_era/04a1_dataset_registry_patch/results/dataset_registry_expanded.csv`; `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/results/phc_lite_sequence_manifest.csv` |
| val | NEW1 / development validation | 6 | 2,655,369 | 2026-01-23 to 2026-01-27 | one duplicate physical path excluded at registry freeze | Registry corrected raw: 9,956,601 rows, 17 files; PHC windows: 82,960 | `n2ew_era/04a1_dataset_registry_patch/results/dataset_registry_expanded.csv`; `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/results/phc_lite_sequence_manifest.csv` |
| test | Phase 01 frozen test | 2 | 1,166,321 | 2026-01-27 day/night | no overlap claim beyond frozen Phase 01 manifest | This older protocol uses 20260127 day/night; later PHC reports place these sessions in NEW1/val. Keep the protocol label with every citation. | `n2ew_era/01_full_l2_action_universe_builder/results/build_parts_manifest.csv`; `n2ew_era/01_full_l2_action_universe_builder/FULL_L2_ACTION_UNIVERSE_REPORT.md` |
| NEW3 | frozen OOS diagnostic | 6 TXFA6 | 5,656,822 | 2026-01-13 to 2026-03-30 | all 6 sessions: `valid_frozen_oos`; no session-id or timestamp-hash overlap | Registry has 7 sessions/8,243,070 raw rows through 2026-05-15; the model-ready TXFA6 scope has 6 sessions and 176,754 windows. | `n2ew_era/02e_final_non_ml_feasibility_decision/results/new3_dedup_audit.csv`; `n2ew_era/04b_txf_only_final_development_rerun/AUDIT_state_count_vs_target_valid_rows.md`; `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/results/phc_lite_sequence_manifest.csv` |
| NEW4 | frozen OOS diagnostic | 4 TXFA6 | 2,196,280 | 2026-01-02 to 2026-01-12 | registry: no excluded duplicate | Registry raw: 2,944,345 rows, 7 files; PHC windows: 68,620 | `n2ew_era/04a1_dataset_registry_patch/results/dataset_registry_expanded.csv`; `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/results/phc_lite_sequence_manifest.csv` |
| NEW5 | one-shot final holdout | 1 (20260128 TXFB6) | 815,912 | 2026-01-28 | frozen contract says `NEW5 20260128 only`; no tuning | Registry contains 2 sessions/1,403,447 rows through 2026-05-18, but the final one-shot contract intentionally evaluates only 20260128. | `n2ew_era/04c_new5_one_shot_final_holdout/NEW5_TXF_ACTION_UNIVERSE_REPORT.md`; `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |

## Table 2 | Non-ML baselines and proxy-environment failure evidence (10 rows)

The old-bid decomposition is at H20. `total_pnl = half_spread_advantage_pnl + mid_price_drift_pnl - maker_fee_total`.

| evidence | split | selected_count | total_pnl | half_spread | drift | fee / significance | source_path |
|---|---|---:|---:|---:|---:|---|---|
| old_bid_only decomposition | val | 563,457 | 116,710,045 | 125,160,100 | 1,800 | maker fee 8,451,855 | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv` |
| old_bid_only decomposition | test | 1,166,281 | 245,095,685 | 261,056,500 | 1,533,400 | maker fee 17,494,215 | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv` |
| old_bid_only decomposition | NEW3 | 5,657,242 | 1,432,466,070 | 1,515,320,700 | 2,004,000 | maker fee 84,858,630; positive rate 90.344977%; half-spread share of positive components 99.867925% | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv` |
| old_bid_only decomposition | NEW4 | 2,196,560 | 534,754,500 | 566,640,400 | 1,062,500 | maker fee 32,948,400 | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv` |
| final non-ML classification | all frozen splits | — | — | — | — | `signal_found_but_strategy_not_proven`; “The proxy environment was too weak.” | `n2ew_era/02e_final_non_ml_feasibility_decision/FINAL_NON_ML_FEASIBILITY_DECISION_REPORT.md` |
| side_by_queue_imbalance_abs_0p1 matched random | test / NEW4 / NEW3 | 971,600 / 1,799,304 / 144,745 | 233,869,570 / 495,871,045 / 1,333,714,860 | — | — | empirical p=0.0 (`<0.001`), 1,000 trials, beats 95th percentile on each reported split | `n2ew_era/02d_final_strict_baseline_audit_package/results/exact_random_oos_summary.csv`; `n2ew_era/02e_final_non_ml_feasibility_decision/results/new3_frozen_oos_results.csv` |
| H20 fee sensitivity: old_bid_only | val | 563,477 | 116,710,045 at 1x fee | — | — | adjusted PnL at 1x/2x/3x/5x = 116,710,045 / 108,257,890 / 99,805,735 / 82,901,425; break-even fee multiplier 14.808318 | `n2ew_era/02b_strict_baseline_and_excess_pnl/results/fee_sensitivity.csv` |
| H20 fee sensitivity: old_bid_only | NEW4 | 2,196,640 | 534,754,500 at 1x fee | — | — | adjusted PnL at 1x/2x/3x/5x = 534,754,500 / 501,804,900 / 468,855,300 / 402,956,100; break-even fee multiplier 17.229469 | `n2ew_era/02b_strict_baseline_and_excess_pnl/results/fee_sensitivity.csv` |
| H20 fee sensitivity: side_by_queue_imbalance_abs_0p1 | val | 468,627 | 111,460,580 at 1x fee | — | — | adjusted PnL at 1x/2x/3x/5x = 111,460,580 / 104,431,175 / 97,401,770 / 83,342,960; break-even fee multiplier 16.856332 | `n2ew_era/02b_strict_baseline_and_excess_pnl/results/fee_sensitivity.csv` |
| H20 fee sensitivity: side_by_queue_imbalance_abs_0p1 | NEW4 | 1,799,304 | 495,871,045 at 1x fee | — | — | adjusted PnL at 1x/2x/3x/5x = 495,871,045 / 468,881,485 / 441,891,925 / 387,912,805; break-even fee multiplier 19.372698 | `n2ew_era/02b_strict_baseline_and_excess_pnl/results/fee_sensitivity.csv` |

## Table 3 | NEW5 one-shot holdout (13 rows)

### Frozen contract (9 rows)

| contract_item | expected | actual | pass | source_path |
|---|---|---|---|---|
| product_filter | `^TXF[A-Z0-9]*$` | `^TXF[A-Z0-9]*$` | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| target | `excess_pnl_H20` | `excess_pnl_H20` | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| model_id | `ml_hgb_regressor_shallow_convergence_selected` | same | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| hgb_max_iter | 120 | 120 | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| hgb_actual_n_iter | 98 | 98 | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| threshold | 0.22419440373778343 | 0.22419440373778343 | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| queue_rule | `side_by_queue_imbalance_abs_0p1` | same | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| fill_proxy | `conservative_l2_stress, H20, inventory_cap=1` | same | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |
| holdout_scope | `NEW5 20260128 only` | same | True | `n2ew_era/04c_new5_one_shot_final_holdout/results/frozen_contract_check.csv` |

### Sequential causal-fill results (4 rows)

| policy | sequential_excess_pnl | fill_count | new5_used_for_tuning | source_path |
|---|---:|---:|---|---|
| no_quote | 0 | 0 | False | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` |
| old_bid_only_spread_ge_1 | -553,365 | 30,212 | False | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` |
| side_by_queue_imbalance_abs_0p1 | 722,745 | 39,600 | False | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` |
| ml_hgb_regressor_shallow_convergence_selected | 893,555 | 40,526 | False | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` |

## Table 4 | Model and ablations (14 rows)

| evaluation_track | item / variant | split | parameters / hidden / numeric | PnL delta vs baseline | interpretation | source_path |
|---|---|---|---|---:|---|---|
| TXF production | PHC-STP v2 dense16 | all | 1,731 params; 16 features; hidden_dim=16 | — | 05C4 source architecture used by the TXF hardware frontier and board path | `n2ew_era/05f_phc_diagonal_ssm/PHC_DIAGONAL_SSM_ABLATION_REPORT.md`; `n2ew_era/05e_phc_hidden8_ablation/PHC_HIDDEN8_ABLATION_REPORT.md` |
| FI-2010 benchmark | PHC v3 hidden16 fair512 | FI-2010 Setup 2 | 2,132 params; input window 100x40; hidden_dim=16 | — | public-benchmark classifier; not the TXF hardware-frontier checkpoint | `n2ew_era/08a_fi2010_public_benchmark/REPORT_STAGE1.md`; `n2ew_era/08a_fi2010_public_benchmark/F1_DEFINITION_RECONCILIATION.md` |
| TXF production | HLS numeric format | all | input/weight `ap_fixed<24,12>`; accumulator `ap_fixed<40,16>` | — | 0 saturated exported values | `n2ew_era/06f_phc_lite_manual_hls/PHC_WEIGHT_EXPORT_REPORT.md`; `n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.hpp` |
| TXF production | fixed_decay_0p9 | val | — | +470 | essentially matches learned decay | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | fixed_decay_0p9 | NEW3 | — | +460 | essentially matches | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | fixed_decay_0p9 | NEW4 | — | -7,505 | small regression | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | memory_off_decay_0 | val | — | -803,235 | memory removal materially hurts | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | memory_off_decay_0 | NEW3 | — | -1,899,740 | material regression | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | memory_off_decay_0 | NEW4 | — | -781,695 | material regression | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | reverse_sequence | val | — | -2,764,500 | temporal order matters | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | reverse_sequence | NEW3 | — | -6,947,680 | temporal order matters | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | reverse_sequence | NEW4 | — | -2,530,160 | temporal order matters | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | shuffle_timesteps | val / NEW3 | — | -2,569,590 / -6,518,720 | deterministic shuffle destroys signal | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |
| TXF production | shuffle_timesteps | NEW4 | — | -2,314,520 | deterministic shuffle destroys signal | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` |

## Table 5 | Latency-resource-value frontier (8 rows)

Percentages use Zynq-7020 totals: LUT 53,200, FF 106,400, DSP 220, BRAM_18K 280. Routed percentages are used when routed results exist; the over-budget row uses csynth estimates. The val/NEW3/NEW4 task-value cells are software `action_pnl`, i.e. aliases of `total_excess_pnl_H20`: selected-action H20 markout sums. They are not causal-fill sequential PnL and are not FPGA-measured PnL.

| design | latency_us | cycles | freq_MHz | WNS_ns | TNS_ns | LUT% | FF% | DSP% | BRAM% | verification_tier | model_source | val / NEW3 / NEW4 action-level H20 markout sum | source_path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 06F v8 | 83.160 | 8,316 | 100 | +0.020 | 0.000 | 14.20 | 7.72 | 21.36 | 2.14 | full cosim + routed P&R met | 05c4 PHC-STP v2 seed2 | 2,037,925 / 5,237,410 / 1,871,685 | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` |
| 06L unroll16 | 25.733 | 2,316 | 90.001 | +0.151 | 0.000 | 27.87 | 18.61 | 56.82 | 0.00 | full cosim + routed P&R met | 05c4 PHC-STP v2 seed2 | 2,037,925 / 5,237,410 / 1,871,685 | `n2ew_era/06l_phc_lite_pipeline_budget_sweep/AUDIT_unroll16_pnr_timing.md` |
| 06U V1 @90 | 24.778 RTL (24.733 csynth) | 2,230 RTL | 90 | +0.352 | 0.000 | 28.61 | 19.46 | 68.64 | 0.00 | full 25,600-vector cosim + routed P&R met | 05c4 PHC-STP v2 seed2 | 2,037,925 / 5,237,410 / 1,871,685 | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` |
| 06U V1 @100 | 22.360 RTL (22.320 csynth) | 2,236 RTL | 100 | +0.004 | 0.000 | 28.34 | 19.73 | 68.64 | 0.00 | 1,000-vector cosim + routed P&R met | 05c4 PHC-STP v2 seed2 | 2,037,925 / 5,237,410 / 1,871,685 | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` |
| 06U V2 hidden8 | 12.020 RTL | 1,202 RTL | 100 | +0.445 | 0.000 | 22.87 | 13.68 | 59.55 | 0.00 | 1,000-vector cosim + routed P&R met | 05e hidden8 seed1, median-val source | 1,908,845 / 5,032,255 / 1,755,610 | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` |
| 06U V3 diagonal unroll8, no local partition | 5.540 RTL | 554 RTL | 100 | N/A | N/A | 53.71 csynth | 14.49 csynth | 30.00 csynth | 0.00 csynth | CSim pass; RTL cosim failed 268/1,000; no P&R | 05f diagonal SSM seed2 | 1,771,525 / 4,727,425 / 1,642,680 | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` |
| 06U V3 diagonal | 5.300 RTL | 530 RTL | 100 | +0.130 | 0.000 | 21.33 | 11.99 | 89.09 | 0.00 | 1,000-vector cosim + routed P&R met | 05f diagonal SSM seed2 | 1,771,525 / 4,727,425 / 1,642,680 | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` |
| 06U V3 diagonal unroll16 (over budget) | 3.830 csynth | 383 | 100 | N/A | N/A | 94.39 | 48.85 | 205.45 | 0.00 | csim + csynth only; DSP over budget; no cosim/P&R | 05f diagonal SSM seed2 | 1,771,525 / 4,727,425 / 1,642,680 | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` |

Power artifacts do exist: routed total on-chip power is 0.357 W for 06F v8 and 0.564 W for 06L unroll16@90 MHz, both with Medium confidence. They are not added as a frontier column because equivalent power reports are not present for every 06U row. Sources: `n2ew_era/06f_phc_lite_manual_hls/hls_component/hls_component/hls/impl/verilog/report/phc_step_power_routed.rpt` and `n2ew_era/06l_phc_lite_pipeline_budget_sweep/pnr_unroll16_90mhz/hls_component/hls_component/hls/impl/verilog/direct_phc_step_reports_baseline_90mhz/phc_step_power_routed.rpt`.

## Table 6 | TCN feed-forward controls (10 rows)

| evidence | variant / split | action-level H20 markout sum or cycles | delta vs PHC | matched-random / timing / resources | source_path |
|---|---|---:|---:|---|---|
| 07A selected seed | scratch seed2 / val | 1,920,890 | -6,280 | p=0.0 (`<0.001`) | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_prepad_oos_eval.csv`; `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_prepad_matched_random_summary.csv` |
| 07A selected seed | scratch seed2 / NEW3 | 5,005,925 | -15,590 | p=0.0 | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_prepad_oos_eval.csv`; `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_prepad_matched_random_summary.csv` |
| 07A selected seed | scratch seed2 / NEW4 | 1,795,280 | +32,785 | p=0.0 | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_prepad_oos_eval.csv`; `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_prepad_matched_random_summary.csv` |
| 07A selected seed | distilled seed1 / val | 1,971,550 | action +44,380; causal-fill +3,340 | p=0.0 (`<0.001`) vs matched random | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv`; `n2ew_era/07a_tcn_feedforward_phc_tradeoff/AUDIT_recursion_necessity_evidence_chain.md` |
| 07A selected seed | distilled seed1 / NEW3 | 5,091,230 | action +69,715; causal-fill +98,925 | p=0.0 (`<0.001`) vs matched random | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv`; `n2ew_era/07a_tcn_feedforward_phc_tradeoff/AUDIT_recursion_necessity_evidence_chain.md` |
| 07A selected seed | distilled seed1 / NEW4 | 1,814,455 | action +51,960; causal-fill +11,955 | p=0.0 (`<0.001`) vs matched random | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv`; `n2ew_era/07a_tcn_feedforward_phc_tradeoff/AUDIT_recursion_necessity_evidence_chain.md` |
| 07C2 unroll scaling | shared_u1 scratch/distilled | 16,510 latency; 16,511 interval | — | csim+csynth; scratch max diff 0.149958, RMSE 0.076498; distilled 0.420075/0.308342 | `n2ew_era/07c2_tcn_streaming_closure/results/stage2_closure_hls_summary.csv` |
| 07C2B unroll scaling | shared_u4 scratch/distilled | 7,438 latency; 7,439 interval | — | csim+csynth; 1008 event calls | `n2ew_era/07c2b_tcn_streaming_speed/results/stage2_speed_unroll_scan.csv` |
| 07C2D routed | scratch shared_u4 @90 | 7,438 cycles; interval 7,439 | — | WNS +0.746; LUT 31.39%, FF 14.36%, DSP 51.36%, BRAM_18K 23.57%; 1008-event cosim max diff 0.149958, RMSE 0.076498 | `n2ew_era/07c2d_tcn_streaming_iface_slim/results/iface_slim_pnr_summary.csv`; `n2ew_era/07c2_tcn_streaming_closure/results/stage2_closure_hls_summary.csv` |
| 07C2D routed | distilled shared_u4 @100 | 7,438 cycles; interval 7,439 | — | WNS +0.187; LUT 32.00%, FF 14.44%, DSP 49.09%, BRAM_18K 23.57%; 1008-event cosim max diff 0.420075, RMSE 0.308342 | `n2ew_era/07c2d_tcn_streaming_iface_slim/results/iface_slim_pnr_summary.csv`; `n2ew_era/07c2_tcn_streaming_closure/results/stage2_closure_hls_summary.csv` |

Required quotation from `n2ew_era/07c2d_tcn_streaming_iface_slim/AUDIT_tcn_streaming_iface_slim_pnr.md`:

> Latency/interval are still the HLS streaming-step latency and interval objects from csynth. They should not be ranked directly against PHC recurrent single-step latency or naive TCN whole-window recompute latency.

The 07A “3 seeds” stability evidence is aggregate, while the six PnL rows above are the selected scratch seed2/distilled seed1 checkpoints. Existing report aggregates: scratch PnL standard deviations val/NEW3/NEW4 = 17,945.3 / 58,052.2 / 31,853.0; distilled = 24,332.0 / 44,211.8 / 11,186.6; matched-random p remained 0.0.

The pooled three-split exact sign tests are not tests of “TCN beats random.” They test the TCN-minus-PHC split-level deltas and yield p=1.00 for scratch action PnL and p=0.25 for scratch causal-fill, distilled action, and distilled causal-fill. With n=3, none establishes TCN superiority at 5%. Source: `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_pooled_significance.csv`; `n2ew_era/07a_tcn_feedforward_phc_tradeoff/AUDIT_recursion_necessity_evidence_chain.md`.

## Table 7 | Board validation (7 rows)

Tier boundary: `25.733 us` is `routed (derived from post-route timing closure)`, not an on-silicon measurement. JTAG programming, LED/HDMI observation, and replay equivalence are `on-board (functional + bit-level equivalence only)`. No independent cycle-counter or ILA latency measurement exists.

| evidence | final status / metric | source_path |
|---|---|---|
| JTAG detection/programming | detected `arm_dap_0`, `xc7z020_1`; pre- and post-done-latch bitstreams programmed successfully | `n2ew_era/06s2_pynq_z2_wrapper_bringup/WRAPPER_BRINGUP_STATUS.md`; `n2ew_era/06s2_pynq_z2_wrapper_bringup/logs/program_pynq_z2_demo_done_latch_fix_retry.log` |
| LED observation | first bitstream: LD3 heartbeat blinking, LD1 pass lit, LD0 not human-visible because done was one 90 MHz cycle; post-latch visual confirmation remained pending in status report | `n2ew_era/06s2_pynq_z2_wrapper_bringup/WRAPPER_BRINGUP_STATUS.md` |
| baseline 1-LSB drift | 4,096 rows; 27 drift rows; max 1 LSB; 32 channel deltas (11 negative, 21 positive); 0 decision flips; minimum margin among drift rows 72 LSB; minimum over all rows 1 LSB | `n2ew_era/06s2_pynq_z2_wrapper_bringup/results/wrapper_1lsb_drift_summary.csv`; `n2ew_era/06s2_pynq_z2_wrapper_bringup/1LSB_DRIFT_ROOT_CAUSE_AND_IMPACT.md` |
| S_DRAIN/drain2 experiment | identical to baseline: 27 drift rows, same direction counts, 0 decision flips | `n2ew_era/06s2_pynq_z2_wrapper_bringup/results/wrapper_1lsb_drift_summary.csv` |
| comparator/XSim | positive replay passes under documented 1-LSB tolerance; negative mutation is detected | `n2ew_era/06s2_pynq_z2_wrapper_bringup/WRAPPER_BRINGUP_STATUS.md` |
| numeric HDMI branch | `visual_failed_final_abandoned`; attempt 3 met timing (+0.100 ns), programmed, briefly showed 4095/-0.012/-0.136/decision0, then flickered/black | `n2ew_era/06s2_pynq_z2_wrapper_bringup/AUDIT_hdmi_numeric_display_bringup.md` |
| actual demo bitstream | recommended and reprogrammed stable solid-color HDMI: `results/phc_demo_wrapper_hdmi.bit`; LED-only fallback: `results/phc_demo_wrapper.bit`; no ILA `.ltx`, so live ILA capture is not claimed | `n2ew_era/06s2_pynq_z2_wrapper_bringup/AUDIT_hdmi_numeric_display_bringup.md`; `n2ew_era/06s2_pynq_z2_wrapper_bringup/WRAPPER_BRINGUP_STATUS.md` |

## Table 8 | FI-2010 public benchmark (9 rows)

Weighted-F1 is the metric definition directly comparable with DeepLOB Table II; macro-F1 remains the project's class-balanced metric.

| model / check | horizon | seeds | parameters | test macro-F1 % | test weighted-F1 % | accuracy / precision / recall / F1 % | source_path |
|---|---:|---:|---:|---:|---:|---|---|
| PHC v3 hidden16 fair512 | 50 | 3 | 2,132 | 63.97 +/- 1.45 | 66.50 +/- 1.15 | accuracy 66.45 +/- 1.07 | `n2ew_era/08a_fi2010_public_benchmark/results/stage1_summary_v3_fair512_weighted_recompute.csv` |
| PHC v3 hidden64 | 50 | 1 | 23,876 | 64.43 | 66.32 | accuracy 66.12 | `n2ew_era/08a_fi2010_public_benchmark/results/stage1_summary_v3_hidden64_capacity_weighted_recompute.csv` |
| TCN small raw | 50 | 1 | 2,344 | 61.23 | 64.02 | accuracy 63.92 | `n2ew_era/08a_fi2010_public_benchmark/results/tcn_fi2010_summary_weighted_recompute.csv` |
| TCN scratch width16 | 50 | 1 | 8,547 | 65.20 | 67.60 | accuracy 67.64 | `n2ew_era/08a_fi2010_public_benchmark/results/tcn_fi2010_summary_weighted_recompute.csv` |
| DeepLOB Table II | 10 | published | 60,000 (paper Table III, rounded) | N/A | 83.40 | 84.47 / 84.00 / 84.47 / 83.40 | `n2ew_era/08a_fi2010_public_benchmark/results/published_comparisons.csv`; `n2ew_era/08a_fi2010_public_benchmark/F1_DEFINITION_RECONCILIATION.md` |
| DeepLOB Table II | 20 | published | 60,000 (paper Table III, rounded) | N/A | 72.82 | 74.85 / 74.06 / 74.85 / 72.82 | `n2ew_era/08a_fi2010_public_benchmark/results/published_comparisons.csv`; `n2ew_era/08a_fi2010_public_benchmark/F1_DEFINITION_RECONCILIATION.md` |
| DeepLOB Table II | 50 | published | 60,000 (paper Table III, rounded) | N/A | 80.35 | 80.51 / 80.38 / 80.51 / 80.35 | `n2ew_era/08a_fi2010_public_benchmark/results/published_comparisons.csv`; `n2ew_era/08a_fi2010_public_benchmark/F1_DEFINITION_RECONCILIATION.md` |
| DeepLOB definition check | 10/20/50 | published | — | — | Table II F1 interpreted weighted | Accuracy equals Recall at every horizon | `n2ew_era/08a_fi2010_public_benchmark/F1_DEFINITION_RECONCILIATION.md` |
| sklearn recompute check | PHC/TCN rows above | existing checkpoints | — | — | — | max `abs(weighted_recall - accuracy)` = 0.0 on val and test | `n2ew_era/08a_fi2010_public_benchmark/F1_DEFINITION_RECONCILIATION.md` |

## Table 9 | Financial meaning of latency (14 rows)

| evidence | split / latency / k | result | boundary | source_path |
|---|---|---|---|---|
| TXF resolution | NEW1-NEW4 | minimum non-zero timestamp step = 1 ms; same-timestamp adjacent share 67.874085% | `P(dt<23.16us)` and `P(dt<83.16us)` are not directly measurable | `n2ew_era/06p_event_interarrival_latency_budget/AUDIT_event_interarrival_latency_budget.md` |
| TXF PnL sensitivity | val/NEW3/NEW4 | `resolution_limited_for_sub_ms_latency_references` | 23.16us vs 83.16us cannot be economically distinguished by this proxy | `n2ew_era/06q_latency_pnl_sensitivity/AUDIT_latency_pnl_sensitivity.md` |
| LOBSTER stale exposure | 25.733us | 2.233353% aggregate | five tickers, one 2012 NASDAQ day; book-change probability, not fill/PnL | `n2ew_era/06r_lobster_us_latency_budget/AUDIT_lobster_us_latency_budget.md` |
| LOBSTER stale exposure | 83.16us | 3.712161% aggregate | legacy 06F FPGA reference | `n2ew_era/06r_lobster_us_latency_budget/AUDIT_lobster_us_latency_budget.md` |
| LOBSTER stale exposure | 262.6us | 7.198588% aggregate | corrected official CPU comparator: 06F N=4 round1 p99 | `n2ew_era/06r_lobster_us_latency_budget/AUDIT_lobster_us_latency_budget.md`; `n2ew_era/06r_lobster_us_latency_budget/results/stale_exposure_summary_06f_corrected.csv` |
| C(k), PHC | val, k=1 | -30.366298 PnL/decision mean delta | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | val, k=2 | -39.936266 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | val, k=5 | -44.085657 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | NEW3, k=1 | -30.678260 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | NEW3, k=2 | -42.355034 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | NEW3, k=5 | -49.312529 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | NEW4, k=1 | -28.810284 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | NEW4, k=2 | -40.683173 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |
| C(k), PHC | NEW4, k=5 | -46.987467 | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` |

Required 06P conclusion (verbatim):

> Because the effective timestamp granularity is 1ms, the requested probabilities `P(delta_t < 23.16us)` and `P(delta_t < 83.16us)` are **not directly measurable** from this dataset.

Required 06Q conclusion (verbatim):

> However, the project still cannot claim a measured economic distinction between `23.16us` and `83.16us`, because both are below the verified 1ms timestamp resolution.

Required 06R dataset boundary: this is a five-ticker, single-day 2012 NASDAQ free-sample pilot, not TXF; stale exposure is book-change probability, not execution impact, fill probability, or PnL.

Required 06T boundary (verbatim):

> This is an event-index paired-label diagnostic and a hypothetical scale estimate, not a causal latency-to-PnL measurement.

## A | 🔴 Missing items

| id | status | missing evidence | consequence | source_path checked |
|---|---|---|---|---|
| M1 | [MISSING] | Board-measured inference latency (cycle counter/ILA/readback) for the programmed demo bitstream | Routed/cosim latency is available, but no physical-board latency observation can be claimed. The current bitstream has no `.ltx`. | `n2ew_era/06s2_pynq_z2_wrapper_bringup/WRAPPER_BRINGUP_STATUS.md` |
| M2 | [MISSING] | Exchange/broker derivation for `MAKER_FEE=15.0` | The fee is an inherited one-way proxy assumption; tax and exact exchange/broker schedules are not verified. Table 2 provides a `1x/2x/3x/5x` sensitivity sweep and break-even multipliers, but that does not certify the base constant. | `n2ew_era/01_full_l2_action_universe_builder/FULL_L2_ACTION_UNIVERSE_REPORT.md`; `n2ew_era/01_full_l2_action_universe_builder/common_phase01.py`; `n2ew_era/02b_strict_baseline_and_excess_pnl/results/fee_sensitivity.csv` |

Special checks that are **not missing**:

- Power: `report_power` outputs exist. 06F routed total/dynamic/static = 0.357/0.251/0.107 W; 06L unroll16@90 = 0.564/0.454/0.110 W; both confidence `Medium`. Coverage is incomplete across the full frontier.
- `excess_pnl_H20` unit definition: the repo defines `POINT_VALUE=200.0` TWD-like proxy units and `MAKER_FEE=15.0`; formulas are `(future_mid-best_bid)*200-15` for bid and `(best_ask-future_mid)*200-15` for ask. It is explicitly an inherited TXF-like proxy, not an exchange-certified monetary accounting definition.

Missing-item count: **2**.

## B | ⚠️ Mutually inconsistent or stale records

| id | issue | record A | record B | disposition |
|---|---|---|---|---|
| C1 | 06F run status | `PHC_HLS_SYNTHESIS_REPORT.md`: “setup_ready_simulation_not_run” and csim/csynth/cosim `not_run` | later reports/artifacts contain full csim, csynth, cosim and routed P&R evidence | Treat the synthesis report as an early setup snapshot, not final status. Sources: `n2ew_era/06f_phc_lite_manual_hls/PHC_HLS_SYNTHESIS_REPORT.md`; `n2ew_era/06l_phc_lite_pipeline_budget_sweep/AUDIT_unroll16_pnr_timing.md` |
| C2 | 06S2 board status | `FPGA_DEMO_SETUP_GUIDE.md`: `bitstream_ready_board_test_pending`, “板子未接上” | `WRAPPER_BRINGUP_STATUS.md`: hardware detected and programmed before/after latch fix | Later hardware status supersedes the guide preflight text. |
| C3 | 06S2 P&R CSV tier | `wrapper_pnr_summary.csv`: `routed_pnr_met_no_board_test` | later JTAG logs/status show successful programming | CSV was not refreshed after board bring-up; retain as P&R-time provenance only. |
| C4 | 06R official CPU comparator | corrected main body uses 262.6us -> 7.198588% and labels 646.318/6299.416us stress-only | an appended legacy section still says “@636.7us (CPU p99)” and reports 10.32%/+6.61pp as if current | The legacy appended prose is stale and must not be used. Official value is 262.6us/7.198588%. Source: `n2ew_era/06r_lobster_us_latency_budget/AUDIT_lobster_us_latency_budget.md`. |
| C5 | NEW3/NEW5 session counts | registry: NEW3=7 sessions, NEW5=2 sessions | model/frozen scopes: NEW3 TXFA6=6 sessions; NEW5 one-shot=1 session | This is a scope mismatch rather than arithmetic error. Every citation must state registry scope vs TXF/model/frozen scope. Sources: `n2ew_era/04a1_dataset_registry_patch/results/dataset_registry_expanded.csv`; Table 1 sources. |

Contradiction/stale-record count: **5**.

## Row-count manifest

| table | rows |
|---|---:|
| Table 1 | 6 |
| Table 2 | 10 |
| Table 3 | 13 |
| Table 4 | 14 |
| Table 5 | 8 |
| Table 6 | 10 |
| Table 7 | 7 |
| Table 8 | 9 |
| Table 9 | 14 |
