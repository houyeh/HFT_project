# Validation Methodology Soundness Audit

Scope: red-team methodology audit only. I read code and reports; I did not rerun experiments, modify existing code, or rewrite existing reports.

Labels:

- `沒問題`: the design/wording is internally consistent from the inspected evidence.
- `需要加註警語`: not fatal, but current wording can be over-read unless the caveat travels with the result.
- `是真正的漏洞`: the current method can invalidate a claimed comparison unless corrected or explicitly quarantined.

## 1. 洩漏檢查

### 1.1 05A/05C scaler and NEW split convention

Verdict: `沒問題`.

Evidence:

- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:256` and `:267` fit the scaler by iterating `part_paths("train")` only.
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:292-294` records `normalization_fit_split="train"` and `new3/new4_used_for_scaler=False`.
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:1447` states the split contract: `NEW2 -> train`, `NEW1 -> val`, `NEW3/NEW4` diagnostic OOS, `NEW5` not loaded.
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:1463-1467` says the feature scaler is fit on train only and unchanged for val/NEW3/NEW4; no NEW5 for scaler, architecture, early stopping, thresholding, or diagnostics.
- Downstream 05C scripts reuse 05A payloads/scaler rather than refitting: examples include `05c1/.../common_05c1.py:94-97`, `05c2/.../common_05c2.py:90-93`, `05c3/.../common_05c3.py:149-152`, and `05c4/.../common_05c4.py:161-164`.
- Downstream threshold/seed selection is validation-only: examples include `05c1/.../common_05c1.py:278-287`, `05c2/.../common_05c2.py:278-286`, `05c3/.../common_05c3.py:353-366`, and `05c4/.../common_05c4.py:383-420`.
- `n2ew_era/05c4_phc_stp_cosine_lr/11_train_val_label_distribution_gap.py:21-34` only allows `train`/`val` and rejects NEW5 paths; `:391-395` rejects any scaler not marked train-only or marked NEW5-used.

Red-team readout: I did not find a scaler fit on validation/OOS, a NEW1/NEW2 reversal, or a downstream 05C script that silently refits normalization on validation/OOS.

### 1.2 FI-2010 Stage1 split protocol

Verdict: `沒問題` for Stage1 training/evaluation itself.

Evidence:

- `n2ew_era/08a_fi2010_public_benchmark/01_prepare_fi2010_stage1.py:20-35` uses official `NoAuction/1.NoAuction_Zscore`, `Train_Dst_NoAuction_ZScore_CF_7.txt`, and test `CF_7/8/9`.
- `n2ew_era/08a_fi2010_public_benchmark/01_prepare_fi2010_stage1.py:153-163` splits the training file into first 80% train and tail 20% validation, while test labels come from concatenated test files.
- `n2ew_era/08a_fi2010_public_benchmark/02_train_phc_lite_stage1.py:259` builds train/val only from `train_features_all`; `:343-345` selects checkpoints on `val_metrics["macro_f1"]`; `:381-395` evaluates val/test only after loading the best checkpoint.
- `n2ew_era/08a_fi2010_public_benchmark/REPORT_STAGE1.md:32` and `:42-43` describe the same Setup 2 protocol and say test is touched only after checkpoint selection.

Red-team readout: Stage1 train/val/test mechanics look consistent with Setup 2 and do not use test for early stopping.

### 1.3 FI-2010 Stage2 representative checkpoint selection

Verdict: `是真正的漏洞`.

Evidence:

- `n2ew_era/08a_fi2010_public_benchmark/05_prepare_stage2_hls.py:85-94` reads `stage1_metrics_v2.csv`, filters k=50/raw candidates, sorts by `test_macro_f1`, and selects the median row.
- `n2ew_era/08a_fi2010_public_benchmark/05_prepare_stage2_hls.py:105-113` writes `rank_by_test_macro_f1_for_k50_raw` and `selection_rule = "k=50 raw primary horizon; median seed by Stage 1 v2 test macro F1"`.
- `n2ew_era/08a_fi2010_public_benchmark/results/stage2_checkpoint_selection.csv:2-4` records the same test-ranked seed table, with seed 1 selected.
- `n2ew_era/08a_fi2010_public_benchmark/REPORT_STAGE2.md:8` states the selected checkpoint was chosen as median by Stage1 v2 test macro F1, and `:22-23` reports that selected checkpoint's val/test macro-F1.

Red-team readout: even though "median seed" is less cherry-picked than "best seed", it still uses the test set to choose a representative checkpoint. This contradicts the Stage1 test-isolation wording and makes the Stage2 checkpoint test-selected. The clean fix is to reselect the representative checkpoint by validation-only information or a pre-registered seed rule, and mark prior Stage2 artifacts as test-selected if they are cited.

## 2. 統計檢定有效性

### 2.1 Matched-random null

Verdict: mixed: 04B/05A/07A `沒問題`; 03B `需要加註警語`.

Evidence for the stronger nulls:

- `n2ew_era/04b_txf_only_final_development_rerun/run_phase04b_pipeline.py:709-765` samples exact matched random trials by required count within `(session, side)` without replacement when possible; the output labels `matching="exact_count_side_session"`.
- `n2ew_era/04b_txf_only_final_development_rerun/TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT.md:37-46` reports the exact-count/session/side matching and warns that p-value 0 should be read as `<0.001`, not exactly zero.
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:854-899` samples by selected count, side, session, and spread bucket, with session-side fallback and replacement flags.
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:902-918` computes the empirical random-ge-model fraction and records the matching label.
- `n2ew_era/07a_tcn_feedforward_phc_tradeoff/common_07a.py:577-589` reuses the 05A matched-random function for TCN comparisons.

Evidence for the warning:

- `n2ew_era/03b_simple_ml_action_value_baselines/common_phase03b.py:502-516` implements `random_normal_trials()` as a finite-population normal approximation to the random sum.
- `n2ew_era/03b_simple_ml_action_value_baselines/05_matched_random_and_baseline_hurdles.py:39-60` applies that approximation by `(session_id, side)` group instead of actually resampling individual actions.
- `n2ew_era/03b_simple_ml_action_value_baselines/SIMPLE_ML_ACTION_VALUE_REPORT.md:69-77` labels the method as `finite_population_normal_1000_trials_matched_count_session_side` and says it is a signal-value hurdle, not strategy tuning.

Red-team readout: 04B/05A/07A are reasonable no-skill nulls because they preserve at least action count, side, and session exposure; 05A/07A additionally preserve spread bucket. 03B is not a strawman, but its "1000 trials" are simulated draws from a normal approximation, not exact random action draws. It should not be cited with the same evidentiary strength as exact/permutation-style matched random.

Suggested warning text:

> 03B matched-random is a finite-population normal approximation matched by selected count, session, and side. It is a no-skill signal hurdle, not an exact permutation/randomization test; compare its p-values and 95p hurdles separately from the later exact/resampling matched-random diagnostics.

### 2.2 Sign test under n=3 split units

Verdict: `沒問題`.

Evidence:

- `n2ew_era/07a_tcn_feedforward_phc_tradeoff/common_07a.py:1024-1028` writes that exact sign tests/bootstrap CIs use the three frozen diagnostic splits and have low power.
- `n2ew_era/07a_tcn_feedforward_phc_tradeoff/TCN_FEEDFORWARD_PHC_TRADEOFF_REPORT.md:37` carries the same low-power warning in the final report.
- `n2ew_era/07a_tcn_feedforward_phc_tradeoff/AUDIT_recursion_necessity_evidence_chain.md:73-82` explicitly says that, with only three split units, the exact sign test does not establish superiority at 5%; the defensible claim is no evidence of feedforward inferiority, not formal TCN superiority.

Red-team readout: this limitation is visible at the points where the sign-test result is interpreted. I did not find a competing report claiming p<0.05 sign-test significance from the n=3 split-level test.

## 3. 定義一致性

### 3.1 `excess PnL`

Verdict: `沒問題`.

Evidence:

- `n2ew_era/02f_excess_markout_and_causal_fill_proxy/EXCESS_MARKOUT_CAUSAL_FILL_REPORT.md:11-16` defines excess PnL as future-mid movement relative to `mid_t` minus fee, with no-quote equal to zero.
- `n2ew_era/04b_txf_only_final_development_rerun/run_phase04b_pipeline.py:336-342` computes raw bid/ask PnL, half-spread components, and excess bid/ask PnL separately.
- `n2ew_era/04b_txf_only_final_development_rerun/run_phase04b_pipeline.py:655-672` audits `raw_pnl_H20 = half_spread_component_H20 + excess_pnl_H20` and records `excess_pnl_H20_confirmed_main_target`.
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:234-236` maps bid/ask action rows to `bid_excess_pnl_H20` and `ask_excess_pnl_H20`; `:1455` says raw PnL, excess PnL, future mid, and positive labels are forbidden input columns.
- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:1540-1542` says causal-fill uses the same conservative L2 proxy family and that fees are embedded in excess labels.

Red-team readout: raw markout and excess markout are separated consistently. I did not find evidence that later 05A/05C metrics silently switch excess PnL to a different baseline.

### 3.2 `MAKER_FEE=15` / `POINT_VALUE=200` caveat propagation

Verdict: `需要加註警語`.

Evidence that the caveat exists:

- `n2ew_era/01_full_l2_action_universe_builder/04_write_phase01_report.py:104-106` writes point value/maker fee and states they are inherited proxy assumptions, not exchange-certified values.
- `n2ew_era/03a6_root_cause_feature_and_contract_closure/COST_UNIT_CONTRACT.md:3-7` states `point_value=200`, `maker_fee=15`, no fee/tax/round-trip certification, and proxy PnL units.
- `n2ew_era/02b_strict_baseline_and_excess_pnl/AUDIT_fee_sensitivity_figure.md:21` and `:135` say the fee trace remains unresolved as a real fee/tax/rebate derivation.

Evidence that the caveat does not travel with every citation:

- `n2ew_era/02e_final_non_ml_feasibility_decision/OLD_BID_ONLY_INTERPRETATION.md:6-7` cites `POINT_VALUE` and `MAKER_FEE` in the formula without a nearby proxy/not-exchange-certified caveat.
- `n2ew_era/04a2_product_filter_and_loader_contract_closure/PHASE04A2_PRODUCT_FILTER_AND_LOADER_CLOSURE_REPORT.md:13` mentions inherited proxy point value and maker fee, but the table row does not carry the full "not exchange-certified/no fee-tax-rebate decomposition" warning.

Red-team readout: the project has the right master caveat, but not every local citation is self-contained. This is a presentation risk, not evidence of inconsistent computation.

Suggested warning text:

> All PnL/excess-PnL values that use `POINT_VALUE=200` and `MAKER_FEE=15` are inherited proxy units, not exchange-certified values. The repository does not document a fee/tax/rebate/round-trip decomposition, so these numbers should not be presented as verified exchange economics.

### 3.3 06T stale k-count vs CPU/FPGA demo scoreboard

Verdict: `是真正的漏洞` if the demo scoreboard is used to support the 06T staleness-cost claim.

06T evidence:

- `n2ew_era/06t_event_shift_staleness_cost/AUDIT_event_shift_staleness_cost.md:17-18` defines a book-change event as best bid/ask change, and k shift as the kth future top-of-book price change after event `t`.
- `n2ew_era/06t_event_shift_staleness_cost/00_event_shift_staleness_cost.py:172-188` constructs `top_change_positions` from best bid/ask changes.
- `n2ew_era/06t_event_shift_staleness_cost/00_event_shift_staleness_cost.py:344-356` shifts selected decisions by `target_positions_after_k_book_changes(...)` and compares shifted same-side labels.
- `n2ew_era/06t_event_shift_staleness_cost/00_event_shift_staleness_cost.py:480-504` writes synthesis rows using `k_used_as_C` and a borrowed stale-exposure delta.

Demo scoreboard evidence:

- `n2ew_era/06s2_pynq_z2_wrapper_bringup/00_build_demo_replay_staleness.py:249-274` computes several event-count variants from `timestamp + latency_us`: `stale_all_events`, `stale_top_change_events`, `stale_strict_time_events`, and `stale_strict_time_top_change_events`.
- `n2ew_era/06s2_pynq_z2_wrapper_bringup/00_build_demo_replay_staleness.py:367-369` plots cumulative `stale_strict_time_events` as the scoreboard's visible cumulative stale-events line.
- `n2ew_era/06s2_pynq_z2_wrapper_bringup/DEMO_REPLAY_STALENESS_REPORT.md:41-43` states that `stale_strict_time_events` is the default scoreboard definition; top-change variants are exported but not the default visual score.

Red-team readout: these are not the same quantity. 06T's k is "kth future top-of-book change used for repriced label decay"; the demo's default score is "number of strictly later timestamped L2 events inside the latency window." The demo does export top-change variants, but the default visual scoreboard is not 06T's k-count. A demo that uses strict timestamp counts to narrate 06T C(k) cost is methodologically invalid until the definitions are aligned.

Required correction:

- Either make the demo's primary score use the same top-of-book-change k logic as 06T, or explicitly rename the strict timestamp scoreboard as a separate visualization metric that does not quantify 06T C(k) staleness cost.

## 4. 驗證層級是否被混用

Verdict: `沒問題`.

Evidence:

- `n2ew_era/NEW_TEAMMATE_BRIEFING_REPORT.md:30` says HLS/FPGA is validated to logic/timing level but has not been run on the physical board.
- `n2ew_era/NEW_TEAMMATE_BRIEFING_REPORT.md:34` says only PHC-lite seed2 has csim -> csynth -> cosim -> Vivado P&R timing-met evidence.
- `n2ew_era/NEW_TEAMMATE_BRIEFING_REPORT.md:352-362` separates csynth estimate, P&R measured resources, routed timing, cosim mismatch, and power estimate.
- `n2ew_era/NEW_TEAMMATE_BRIEFING_REPORT.md:366` warns the WNS margin is thin and must be cited with risk context; `:376` says PYNQ-Z2 board measurement is not done; `:473` repeats that the final step is not complete.
- `n2ew_era/PHC_HLS_Reviewer_Briefing.md:566-571` explicitly says board testing is future work; `:577-580` labels "FPGA inference 83 us" as a csynth number with P&R timing confirmation and says CPU-speedup claims are high-risk; `:590-594` forbids production/PnL overclaims.
- `n2ew_era/08a_fi2010_public_benchmark/REPORT_STAGE2.md:1` labels Stage2 as `csim_and_csynth_complete_no_cosim_no_pnr`; `:76-78` distinguishes routed TXF evidence from FI-2010 `csim_pass + csynth_only`; `:86-87` says no cosim/P&R/bitstream/board and no FPGA PnL proof.
- `n2ew_era/07a_tcn_feedforward_phc_tradeoff/TCN_FEEDFORWARD_PHC_TRADEOFF_REPORT.md:14-19` marks TCN latency as pre-synthesis MAC-rate estimates, not csynth/cosim/P&R validated; `:25-33` keeps TCN rows labelled `pre-synthesis MAC estimate only, not HLS-verified`.

Red-team readout: I did not find a current report that presents csim-only, csynth, routed P&R, and board measurement as interchangeable evidence. The hardware claims are generally separated by verification level.

## 5. 閾值/checkpoint 選擇是否循環

Verdict: `是真正的漏洞` for FI-2010 Stage2 checkpoint selection; `沒問題` for inspected 05A/05C threshold selection.

05A/05C evidence:

- `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/common_05a.py:777-788` records validation-only threshold selection for PHC-lite.
- `n2ew_era/05c4_phc_stp_cosine_lr/common_05c4.py:383-420` searches seed/threshold on `val` and records `selection_split="val"` with `new3/new4/new5_used_for_selection=False`.
- 05C1/05C2/05C3 show the same pattern at `05c1/.../common_05c1.py:278-287`, `05c2/.../common_05c2.py:278-286`, and `05c3/.../common_05c3.py:353-366`.

FI-2010 Stage2 evidence:

- `n2ew_era/08a_fi2010_public_benchmark/05_prepare_stage2_hls.py:93-94` ranks candidates by `test_macro_f1` and selects the median.
- `n2ew_era/08a_fi2010_public_benchmark/REPORT_STAGE2.md:8` makes the test-median selection rule explicit; `:22-23` then reports the selected checkpoint's val/test macro-F1.
- `n2ew_era/08a_fi2010_public_benchmark/06_collect_stage2_hls_results.py:208-225` generates the same text/table in the Stage2 report.

Red-team readout: the test-selected Stage2 checkpoint is not a training-time threshold leak, but it is still a checkpoint-selection leak. If later analysis uses this selected checkpoint to support an F1/resource/latency narrative, it must state that the checkpoint was selected using test macro-F1. The clean method is validation-only or pre-registered seed selection.

## 結論清單

### 無問題

- 05A scaler/normalization fit is train-only, and 05C downstream scripts reuse 05A scaler/payloads rather than refitting on validation/OOS.
- 05A/05C NEW2 -> train, NEW1 -> val, NEW3/NEW4 diagnostic OOS, NEW5-not-used convention is consistently documented and enforced in inspected scripts.
- FI-2010 Stage1 data split and checkpoint selection are consistent with train/validation/test separation.
- 04B/05A/07A matched-random procedures are reasonable no-skill controls for their stated scope.
- 07A sign-test limitations under n=3 split units are clearly stated at the relevant interpretation points.
- `excess_pnl_H20` is consistently defined as half-spread-removed future-mid excess markout with fee embedded, and later 05A/05C metrics use that target.
- Hardware verification levels are generally not mixed: TXF routed-P&R/cosim, FI-2010 csim+csynth-only, TCN pre-synthesis estimate, and no-board status are visibly separated.

### 需要加註警語

- 03B matched-random uses a finite-population normal approximation, not exact random action resampling.
  - Suggested warning: `03B matched-random is a finite-population normal approximation matched by selected count, session, and side. It is a no-skill signal hurdle, not an exact permutation/randomization test.`
- `POINT_VALUE=200` and `MAKER_FEE=15` have master caveats, but not every local citation carries the full proxy/not-exchange-certified warning.
  - Suggested warning: `All PnL/excess-PnL values that use POINT_VALUE=200 and MAKER_FEE=15 are inherited proxy units, not exchange-certified values; no fee/tax/rebate/round-trip decomposition is documented.`

### 真正需要修正的漏洞

- FI-2010 Stage2 selects the representative checkpoint by Stage1 v2 `test_macro_f1` median. This violates clean test isolation for the selected Stage2 checkpoint.
  - Required action: reselect by validation-only information or a pre-registered seed rule; mark existing Stage2 artifacts/reports as test-selected if cited.
- 06T stale C(k) and the 06S2 demo scoreboard default are not the same definition. 06T uses kth future top-of-book change for repriced-label decay; the demo default plots strict timestamp stale-event counts.
  - Required action: align the demo's primary scoreboard with 06T top-change k logic, or explicitly quarantine the strict timestamp scoreboard as a separate demo metric that does not quantify 06T C(k) staleness cost.
