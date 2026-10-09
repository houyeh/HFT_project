status: mixed
finding: figure_generated_but_maker_fee_15_has_no_derivation_document
# AUDIT Fee Sensitivity Figure And Maker Fee Trace

## Scope

This audit has two independent parts:

- Part A is a read-only repository trace for `MAKER_FEE = 15.0`.
- Part B is a pure visualization of the existing `results/fee_sensitivity.csv`.

No `point_value`, `maker_fee`, model, policy-selection logic, or baseline pipeline was modified. `01_strict_baseline_excess_pnl.py` was not rerun.

## Part A: Maker Fee Trace

Repository search covered `n2ew_era/`, `new_era/`, `data/`, repo markdown/json/README files, and older meeting/report material with keywords including `maker_fee`, `MAKER_FEE`, `maker_fee_twd`, `FEE_PER_TRADE`, `fee_sensitivity`, `手續費`, and `期交稅`. HLS generated artifacts and large duplicated experiment-output directories were excluded from the final human-readable table after confirming they were not source documents for the fee assumption.

### Conclusion

目前 codebase 中沒有找到 maker_fee=15 的推導文件,這是一個未附來源的假設常數,建議在正式報告中將其明確標註為『假設值』,或請作者補充實際手續費/稅費/回扣的拆解計算。

The codebase does contain several useful caveats: the current Phase 01 and Phase 03A6 documents explicitly call `maker_fee=15` an inherited proxy assumption and state that it is not exchange-certified. I did not find a documented decomposition into exchange fee, tax, broker fee, and maker rebate.

### Direct Phase 01/02 Evidence

| file:line | snippet / relevance |
| --- | --- |
| `n2ew_era/01_full_l2_action_universe_builder/common_phase01.py:22` | Defines `MAKER_FEE = 15.0`. |
| `n2ew_era/01_full_l2_action_universe_builder/data/build_config.json:25` | Stores `"maker_fee": 15.0`. |
| `n2ew_era/01_full_l2_action_universe_builder/data/build_config.json:30-31` | Stores label formulae subtracting `maker_fee` for bid/ask. |
| `n2ew_era/01_full_l2_action_universe_builder/01_build_full_l2_action_universe.py:343-344` | Computes `pnl = (...) * POINT_VALUE - MAKER_FEE`. |
| `n2ew_era/01_full_l2_action_universe_builder/01_build_full_l2_action_universe.py:434,439-440` | Writes `maker_fee` and label formulae into `build_config.json`. |
| `n2ew_era/01_full_l2_action_universe_builder/04_write_phase01_report.py:104-106` | Report generator writes point value, maker fee, and inherited-proxy caveat. |
| `n2ew_era/01_full_l2_action_universe_builder/04_write_phase01_report.py:112-113` | Report generator writes bid/ask formulas with `maker_fee`. |
| `n2ew_era/01_full_l2_action_universe_builder/FULL_L2_ACTION_UNIVERSE_REPORT.md:48` | Reports `Point value: 200.0. Maker fee: 15.0`. |
| `n2ew_era/01_full_l2_action_universe_builder/FULL_L2_ACTION_UNIVERSE_REPORT.md:50` | Says `MAKER_FEE = 15.0` is inherited from prior constants and that current source does not verify tax/exchange/broker schedule. |
| `n2ew_era/01_full_l2_action_universe_builder/FULL_L2_ACTION_UNIVERSE_REPORT.md:56-57` | Documents bid/ask PnL formulae subtracting `maker_fee`. |
| `n2ew_era/02b_strict_baseline_and_excess_pnl/01_strict_baseline_excess_pnl.py:29` | Defines local `MAKER_FEE = 15.0` for strict baseline fee sensitivity. |
| `n2ew_era/02b_strict_baseline_and_excess_pnl/01_strict_baseline_excess_pnl.py:349-370` | Existing fee sensitivity logic subtracts `(fee_multiplier - 1) * MAKER_FEE * selected_count` and writes `break_even_fee_multiplier`, `positive_after_fee`, and `eligible_for_final_selection_at_1x`. |
| `n2ew_era/02b_strict_baseline_and_excess_pnl/01_strict_baseline_excess_pnl.py:478` | Report text states that `fee_sensitivity.csv` is an adjustment of Phase 01 labels, not a rerun. |
| `n2ew_era/02b_strict_baseline_and_excess_pnl/results/fee_sensitivity.csv:1` | Existing CSV fields used for Part B. |
| `n2ew_era/02b_strict_baseline_and_excess_pnl/STRICT_BASELINE_EXCESS_PNL_REPORT.md:68` | Existing report explains fee sensitivity adjustment formula. |

### Later n2ew_era Contract Evidence

| file:line | snippet / relevance |
| --- | --- |
| `n2ew_era/02f_excess_markout_and_causal_fill_proxy/common_phase02f.py:38` | Defines `MAKER_FEE = 15.0`. |
| `n2ew_era/02f_excess_markout_and_causal_fill_proxy/common_phase02f.py:143-144` | Excess markout labels subtract `MAKER_FEE`. |
| `n2ew_era/02f_excess_markout_and_causal_fill_proxy/common_phase02f.py:276-277` | Fee multiplier stress uses `MAKER_FEE * (fee_multiplier - 1.0)`. |
| `n2ew_era/02f_excess_markout_and_causal_fill_proxy/results/excess_label_definition.json:2-6` | Records formulae and `"maker_fee": 15.0`. |
| `n2ew_era/03a6_root_cause_feature_and_contract_closure/COST_UNIT_CONTRACT.md:3` | States `point_value = 200.0` and `maker_fee = 15.0` are inherited proxy assumptions. |
| `n2ew_era/03a6_root_cause_feature_and_contract_closure/COST_UNIT_CONTRACT.md:5` | States current files do not certify whether `maker_fee=15` includes exchange fees, tax, broker fees, or round-trip costs. |
| `n2ew_era/03a6_root_cause_feature_and_contract_closure/COST_UNIT_CONTRACT.md:7` | States these are proxy PnL units, not exchange-certified values. |
| `n2ew_era/03a6_root_cause_feature_and_contract_closure/results/cost_unit_product_audit.csv:1-20` | Repeats `maker_fee=15.0`, `exchange_certified=False` across audited splits/sessions. |
| `n2ew_era/03b_simple_ml_action_value_baselines/common_phase03b.py:40` | Defines `MAKER_FEE = 15.0` for ML/action-value baseline phase. |
| `n2ew_era/04a2_product_filter_and_loader_contract_closure/PRODUCT_UNIVERSE_CONTRACT.md:5` | Says `POINT_VALUE=200` and `MAKER_FEE=15` are TXF-compatible project assumptions, not exchange-certified. |
| `n2ew_era/04a2_product_filter_and_loader_contract_closure/PRODUCT_UNIVERSE_CONTRACT.md:11` | Freezes TXF family only and warns against pooling products without normalization. |
| `n2ew_era/04b_txf_only_final_development_rerun/run_phase04b_pipeline.py:45` | Defines `MAKER_FEE = 15.0`. |
| `n2ew_era/04b_txf_only_final_development_rerun/run_phase04b_pipeline.py:339-342` | Computes raw/excess bid/ask PnL subtracting `MAKER_FEE`. |

### Older new_era Sources

| file:line | snippet / relevance |
| --- | --- |
| `new_era/models_playaround_15/config.py:75-76` | Defines `maker_fee_twd: float = 15.0` and `taker_fee_twd: float = 50.0`. No derivation found. |
| `new_era/models_playaround_15/experiment_matrix.py:51-52` | Experiment config sets `maker_fee_twd=15.0`, `taker_fee_twd=50.0`. |
| `new_era/18_queue_aware/queue_stress_core.py:25` | Defines `MAKER_FEE = 15.0`. This is cited by the Phase 01 report as an inherited source. |
| `new_era/18_queue_aware/queue_stress_core.py:160,169` | Adds `MAKER_FEE` on simulated bid/ask fills. |
| `new_era/18_queue_aware/01_pre_018_freeze_check.py:173` | Freeze check records consistency: `maker_fee=15.0 in 016 helpers and 018 stress core`. |
| `new_era/18_queue_aware/pre_018_model_freeze_report.csv:6,19` | Freeze report records `maker_fee_consistency` as `maker_fee=15.0`, but not a fee/tax derivation. |
| `new_era/16_binary_label_v2/round2_audit_utils.py:15` | Defines `MAKER_FEE = 15.0`. |
| `new_era/16_binary_label_v2/10_robust_random_baseline.py:19` | Defines `MAKER_FEE = 15.0`. |
| `new_era/16_binary_label_v2/12_pnl_decomposition.py:19` | Defines `MAKER_FEE = 15.0`. |
| `new_era/10_walk_forward_mm_backtest.py:62` | Defines `MAKER_FEE_TWD = 15.0`. |
| `new_era/11_mm_backtest_30ticks.py:52` | Defines `FEE_PER_TRADE = 15` with comment `Maker (掛單方) 手續費較低`; no external basis. |
| `new_era/tcn_pipeline_utils.py:34` | Defines `MAKER_FEE_TWD = 15`. |

### Related But Not Supporting The `15` Derivation

| file:line | snippet / relevance |
| --- | --- |
| `new_era/08_backtest_pnl.py:125` | Uses `FEE_PER_TRADE_TWD = 50` with comment that it assumes one-way fee plus tax. This is an older different assumption, not a derivation of maker fee 15. |
| `new_era/008_Backtest_Report.md:18` | States `FEE_PER_TRADE_TWD = 50`, round trip 100 TWD; not the current maker fee 15. |
| `new_era/008_model_examine.md:30` | Mentions one-way 50 TWD fee in older active-trading backtest. |
| `new_era/execution_log.md:137,167` | Mentions one-way 50 TWD fee in older notes. |
| `new_era/009_MM_Backtest_Report.md:10` | Qualitatively mentions TXF fee/tax thin margins; no numerical 15 derivation. |
| `meeting/w13/___Nancy_meeting_deck/02_backtest_strategy_result/slide.md:6` | Meeting slide lists `Maker Fee = 15 TWD, Taker Fee = 50 TWD, Point Value = 200`; no source breakdown. |

## Part B: Fee Sensitivity Visualization

Input CSV:

`C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\results\fee_sensitivity.csv`

Filter:

```text
split == "val"
eligible_for_final_selection_at_1x == True
```

Plot grouping:

```text
one line = policy_id + horizon
```

This grouping avoids merging the same policy name across different horizons. The visualized dataset contains:

| item | value |
| --- | ---: |
| filtered rows | 368 |
| plotted lines (`policy_id + horizon`) | 92 |
| unique policy_id values | 20 |
| horizons | 1, 5, 10, 20, 50 |
| fee multipliers | 1, 2, 3, 5 |
| maker_fee values represented | 15, 30, 45, 75 |
| minimum `adjusted_total_pnl` in plotted rows | 157,525 |
| all plotted adjusted PnL values positive | True |

Output figure:

`C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\figures\fee_sensitivity_by_multiplier.png`

Derived plot summary:

`C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\02b_strict_baseline_and_excess_pnl\results\fee_sensitivity_val_eligible_plot_summary.csv`

The figure includes one line for every `policy_id + horizon`, an `adjusted_total_pnl = 0` horizontal reference line, and x-axis labels mapping fee multipliers to maker fees `15/30/45/75`.

## Honest Conclusion

The requested figure was generated from the existing `fee_sensitivity.csv` only; no model, grid, or strategy-selection computation was rerun.

The maker-fee trace remains unresolved as a real-world fee/tax/rebate derivation. The best supported statement is: `maker_fee=15` is an inherited proxy one-way maker-fee assumption used consistently across the current proxy PnL pipeline, with existing project documents explicitly warning that it is not exchange-certified and does not document whether it includes exchange fees, tax, broker fees, rebates, or round-trip costs.
