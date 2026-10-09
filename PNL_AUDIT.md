# PnL Quantity Audit

Status: `resolved_with_source_correction`

This audit is report-layer work only. It did not retrain a model, recompute a PnL series, rerun a simulator, or read NEW5 raw data. Existing CSVs and reports are treated as immutable provenance.

## 1. Quantity Dictionary

All monetary-looking values below are **TWD proxies** produced with `POINT_VALUE=200` and `MAKER_FEE=15`; they are not exchange-certified accounting records.

| field | plain-language name | exact definition | unit | typical source | baseline/model |
|---|---|---|---|---|---|
| `excess_pnl_H20` | per-action H20 excess markout | Bid or ask H20 future-mid markout multiplied by `POINT_VALUE`, less maker fee; one value per candidate action | TWD proxy/action label | `02f_excess_markout_and_causal_fill_proxy/common_phase02f.py:143-144` | either |
| `total_excess_pnl_H20` | selected-action H20 markout sum | Sum of `excess_pnl_H20` for every selected bid/ask decision; it does not apply the sequential fill mask or inventory cap | TWD proxy | `05a.../common_05a.py:523-556`; `05c4.../phc_stp_v2_oos_eval.csv` | either |
| `action_pnl` | action-level PnL alias | Renamed `total_excess_pnl_H20` used in model-comparison/frontier summaries | TWD proxy | `05c4.../common_05c4.py:646,788` | model or baseline |
| `sequential_excess_pnl` | causal-fill sequential PnL | Sum of filled action labels after deterministic conservative fill decisions and sequential inventory-cap enforcement | TWD proxy | `05a.../common_05a.py:1069`; `05c4.../phc_stp_v2_causal_fill_eval.csv` | either |
| `causal_sequential_excess_pnl` | explicit causal sequential PnL alias | Distillation-table name for the same sequential conservative-fill quantity | TWD proxy | `07d.../distillation_depth_lag_summary.csv` | model |
| `causal_fill_pnl` | causal-fill PnL alias | Renamed `sequential_excess_pnl` in 05c4 comparison tables | TWD proxy | `05c4.../common_05c4.py:778-789` | model |
| `phc_causal_fill_pnl` | PHC causal-fill reference | PHC row's `sequential_excess_pnl`, copied beside a candidate for retention comparison | TWD proxy | `07d.../00_run_decision_distillation.py:164-187` | model reference |
| `simple_rule_causal_fill_pnl` | simple-rule causal-fill reference | Queue-imbalance rule's `sequential_excess_pnl`, copied beside a candidate | TWD proxy | `07d.../00_run_decision_distillation.py:164-187` | baseline |
| `avg_trade_excess_pnl` | average filled-trade excess PnL | `sequential_excess_pnl / fill_count`, or zero when no fills occur | TWD proxy/fill | `05c4.../common_05c4.py:581-588` | either |

### Frontier quantity determination

The 06U signal-quality triplet `2,037,925 / 5,237,410 / 1,871,685` is `action_pnl`, an alias of `total_excess_pnl_H20`, for val/NEW3/NEW4. The values are copied directly from `phc_stp_v2_oos_eval.csv`; they are **not** sequential causal-fill results. Sources: `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv`, `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_oos_eval.csv`, and `n2ew_era/05c4_phc_stp_cosine_lr/common_05c4.py:646,788-789`.

## 2. Authoritative 05C4 Reconciliation

| split | action-level `total_excess_pnl_H20` | PHC-STP sequential PnL | fill count | average/fill | PHC-lite v1 sequential PnL | queue-rule sequential PnL | source |
|---|---:|---:|---:|---:|---:|---:|---|
| val | 2,037,925 | 171,835 | 6,891 | 24.936149 | 170,310 | 103,260 | `phc_stp_v2_oos_eval.csv`; `phc_stp_v2_causal_fill_eval.csv`; `phc_stp_v2_causal_fill_vs_v1.csv` |
| NEW3 | 5,237,410 | 611,095 | 17,907 | 34.126040 | 488,870 | 303,700 | same files |
| NEW4 | 1,871,685 | 129,160 | 4,237 | 30.483833 | 121,765 | 72,415 | same files |

The verified NEW3 difference `611,095 - 488,870 = 122,225`, or `+25.0015%`, is versus **PHC-lite v1**, not versus the queue rule. Relative to the actual queue-rule sequential value `303,700`, the difference is `307,395`, or approximately `+101.22%`. The task background's statement that `488,870` is the queue baseline is therefore contradicted by the authoritative CSV and was not propagated into the corrected documents.

## 3. Report-Layer Inventory

Rows group adjacent occurrences when a table or paragraph uses one quantity consistently. Line numbers refer to the corrected files.

| file | line(s) | value(s) | prior/current label | correct quantity | source path | change? | correction |
|---|---:|---|---|---|---|---|---|
| `__專題成果報告_中文初稿_20261001.md` | 20, 24, 66 | frontier task-value description and unchanged-model claim | generic economic value -> action-level H20 markout sum | `total_excess_pnl_H20` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | explicitly distinguish from sequential causal-fill PnL |
| same | 27 | +3,340 / +98,925 / +11,955 | TCN relative PnL | TCN-minus-PHC causal-fill `sequential_excess_pnl` deltas | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv` | no | already causal-fill |
| same | 197-219 | 1,432,466,070; 90.344977%; 99.867925% | naive old-bid PnL/decomposition | NEW3 action-level H20 markout and decomposition | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv` | no numeric change | retain naive-markout qualifier |
| same | 238-243 | 893,555 / 722,745 / 0 / -553,365; delta 170,810 | NEW5 sequential excess PnL | `sequential_excess_pnl` | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` | no | already correct |
| same | 309-314 | +470/+460/-7,505; -803,235/-1,899,740/-781,695; -2,764,500/-6,947,680/-2,530,160; -2,569,590/-6,518,720/-2,314,520 | ablation PnL changes | action-level `total_excess_pnl_H20` deltas | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` | yes (label) | header now identifies action-level H20 markout |
| same | 330-334 | queue 1,237,145/2,954,770/1,003,910; HGB 1,671,675/4,010,580/1,395,420; PHC 2,037,925/5,237,410/1,871,685 | model task PnL | action-level `total_excess_pnl_H20` | `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_vs_v1_hgb_queue.csv`; `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_oos_eval.csv` | yes (label) | section/table units corrected |
| same | 348-356 | 5,237,410; 5,032,255; 4,727,425 | frontier task PnL | NEW3 `action_pnl` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | table header corrected |
| same | 370-374 | -205,155; -304,830; -509,985 | frontier model-change PnL | differences in NEW3 `action_pnl` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | change column corrected |
| same | 414-417 | 5,237,410; 37.1; 34.126; 17,907 | ambiguous PnL -> action-level plus causal comparison | action sum/selected action; causal average/fill | `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_oos_eval.csv`; `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_causal_fill_eval.csv` | yes (label) | both denominators named |
| same | 431-433 | +3,340 / +98,925 / +11,955 | causal-fill delta | candidate-minus-PHC sequential causal-fill PnL | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv` | no | already correct |
| same | 494, 664 | -9.74% | signal-quality/task-value reduction | relative NEW3 `action_pnl` reduction | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | explicitly says action-level H20 markout |
| `__figures/make_fig6_1_frontier.py` | 20-32, 82, 123-126 | 5,237,410; 5,032,255; 4,727,425 | NEW3 excess PnL -> action-level H20 markout sum | `action_pnl` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | ylabel, in-chart wording, and footnote corrected; no point changed |
| `demo_narrative_and_script_20260808.md` | 82 | 170,310 / 488,870 / 121,765 | presented as project candidate result | PHC-lite v1 causal-fill baseline | `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_causal_fill_eval.csv` | yes (attribution) | replaced candidate triplet with 171,835 / 611,095 / 129,160; retained old values as v1 provenance; named queue NEW3 as 303,700 |
| `demo_narrative_and_script_20260808.md` | 200 | 893,555 | NEW5 HGB result | HGB `sequential_excess_pnl` | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` | no | already correct |
| `EVIDENCE_PACK.md` | Table 2, line 28 | 1,432,466,070 and decomposition | old-bid PnL | naive action-level H20 decomposition | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv` | no | already qualified |
| same | Table 3, lines 53-60 | -553,365 / 722,745 / 893,555 | sequential causal-fill | `sequential_excess_pnl` | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` | no | already correct |
| same | Table 4 | ablation deltas | PnL delta | action-level H20 markout delta | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` | no | table context retained |
| same | Table 5, lines 83-96 | 2,037,925 / 5,237,410 / 1,871,685 and model variants | task-value/PnL -> action-level | `action_pnl` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | explicit alias note added |
| same | Table 6, lines 100-119 | TCN action and causal deltas | mixed but column-identifiable | separate `action_pnl` and causal-fill deltas | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv` | no | header now stresses action-level where relevant |
| same | Table 9, lines 160-182 | negative C(k) deltas | PnL/decision diagnostic | repriced-label paired diagnostic, not causal execution | `n2ew_era/06t_event_shift_staleness_cost/results/staleness_cost_by_k.csv` | no | boundary retained |
| `REPORT_OUTLINE.md` | Section 4 | naive/NEW5 values | naive versus sequential | correct quantities on different splits/simulators | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv`; `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` | no | special-evidence qualifier retained |
| same | Section 6/Figure 1 | frontier triplets | task value -> action-level H20 markout sum | `action_pnl` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | caption requirement corrected |
| same | Section 7 | +3,340 / +98,925 / +11,955 | causal-fill deltas | candidate minus PHC `sequential_excess_pnl` | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv` | no | already correct |
| `REPORT_SECTION_04.md` | 35-39 | 1,432,466,070; 90.344977%; 99.867925%; -553,365; 722,745; 893,555 | naive markout and sequential contrast | correct | `n2ew_era/02e_final_non_ml_feasibility_decision/results/old_bid_only_pnl_decomposition.csv`; `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` | no | no edit required |
| `PROJECT_STORY_RETROSPECTIVE.md` | NEW5 passage, lines 80-89 | 893,555 / 722,745 / 0 / -553,365; 170,810 | NEW5 causal-fill PnL | `sequential_excess_pnl` | `n2ew_era/04c_new5_one_shot_final_holdout/results/new5_causal_fill_sequential_eval.csv` | no | already correct |
| same | ablation passage, lines 114 onward | +470 / -803,235 / -2,764,500 / -2,569,590 | ablation PnL change | action-level H20 markout deltas | `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/AUDIT_learned_memory_vs_fixed_window.md` | yes (label) | action-level label added |
| same | frontier passages, lines 185-198, 366 | 5,237,410 and -9.74% | task value -> action-level H20 markout | `action_pnl` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | sequential exclusion added |
| same | TCN passage, lines 251-255 | +3,340 / +98,925 / +11,955 | causal-fill PnL | `sequential_excess_pnl` delta | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv` | no | already correct |
| `TEAM_HANDOFF_AND_FINALS_PLAN_20260812.md` | lines 13, 27, 168-178, 228 | 2,037,925 / 5,237,410 / 1,871,685; hidden8/diagonal triplets; -9.74% | PnL -> action-level H20 markout | `action_pnl` | `n2ew_era/06u_phc_latency_breakthrough/results/06u_latency_resource_summary.csv` | yes (label) | table, prose, and figure instruction corrected |
| same | line 70 | +3,340 / +98,925 / +11,955 | causal-fill PnL | causal-fill delta | `n2ew_era/07a_tcn_feedforward_phc_tradeoff/results/tcn_vs_phc_tradeoff_summary.csv` | no | already correct |

## 4. Correction Checklist

### (a) Pure label errors

- [x] Chinese report introduction and frontier narrative identify `action_pnl`/`total_excess_pnl_H20` rather than sequential PnL.
- [x] Chinese report Section 5.4 table title and units identify action-level H20 markout.
- [x] Chinese report Section 6 frontier table and Figure 6-1 caption identify action-level H20 markout.
- [x] Figure generator y-axis and footnote distinguish markout sum from causal-fill sequential PnL.
- [x] `EVIDENCE_PACK.md` Table 5 identifies its task-value columns as `action_pnl`.
- [x] `REPORT_OUTLINE.md` Section 6 and Figure 1 specify the correct quantity.
- [x] `PROJECT_STORY_RETROSPECTIVE.md` frontier references specify the correct quantity.
- [x] `TEAM_HANDOFF_AND_FINALS_PLAN_20260812.md` frontier references specify the correct quantity.

### (b) Attribution errors

- [x] `demo_narrative_and_script_20260808.md` no longer attributes the PHC-lite v1 causal-fill triplet to the deployed 05C4 PHC-STP candidate.
- [x] The prompt-level claim that NEW3 `488,870` is the queue baseline is rejected; the CSV identifies `488,870` as PHC-lite v1 and `303,700` as the queue rule.

### (c) Requires owner judgment

- [ ] Whether the final report should retain both action-level and causal-fill tables, or use only one as the principal task-value axis. Both are now correctly labeled, but choosing the principal reporting quantity is an editorial decision.

## 5. Audit Counts

- Distinct report/figure label-error groups corrected: **8**.
- Distinct attribution errors corrected or rejected: **2** (one repository document, one task-background assertion).
- `[UNRESOLVED]` quantity identities: **0**.
- Figure regeneration: **complete**. The corrected script regenerated `__figures/fig6_1_latency_value_frontier.png` and `.svg` without changing any plotted value.
