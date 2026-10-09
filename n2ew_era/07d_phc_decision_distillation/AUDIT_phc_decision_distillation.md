# PHC Decision Distillation Audit

Status: `resolved_negative`

## Scope And Guardrails

- Phase: `07d_phc_decision_distillation`.
- Teacher: frozen `05c4_phc_stp_v2_cosine_lr` seed=2 checkpoint selected on validation.
- Fit split: `train`; selection split: `val`; OOS diagnostic splits: `NEW3`, `NEW4`.
- NEW5 is not read, loaded, or used.
- The distillation target is the frozen teacher's two raw action-value scores, not realized PnL labels.
- Matched-random significance uses the existing 05A action-level `matched_random_for_split` routine; causal-fill PnL uses the existing conservative L2 stress sequential simulator with `inventory_cap=1`.
- This is CPU/Python analysis only. It does not perform HLS, csim, csynth, cosim, P&R, bitstream, or board work.

## Selected Development-Only Distilled Rule

- Selected by: `val_causal_sequential_excess_pnl` on `val` only.
- Selected policy: `phc_stp_v2_distilled_tree_current_t_depth8`.
- Variant: `current_t`; depth: `8`; leaves: `251`.
- One-line conclusion: phc_stp_v2_distilled_tree_current_t_depth8 selected on val keeps at least 33.3% of the PHC-over-simple causal-fill advantage across val/NEW3/NEW4 (direct PHC PnL ratio min 68.1%).

| split | policy_id                                  | variant   | depth | leaf_nodes | score_r2_vs_teacher_uniform | causal_sequential_excess_pnl | phc_causal_fill_pnl | simple_rule_causal_fill_pnl | causal_pnl_retention_vs_phc | causal_advantage_retention_vs_simple_to_phc | total_excess_pnl_H20 | matched_random_95p | beats_random_95p | matched_random_p_value |
| ----- | ------------------------------------------ | --------- | ----- | ---------- | --------------------------- | ---------------------------- | ------------------- | --------------------------- | --------------------------- | ------------------------------------------- | -------------------- | ------------------ | ---------------- | ---------------------- |
| val   | phc_stp_v2_distilled_tree_current_t_depth8 | current_t | 8     | 251        | 0.643872                    | 143555                       | 171835              | 103260                      | 0.835424                    | 0.587605                                    | 1.62703e+06          | -846998            | True             | 0                      |
| new3  | phc_stp_v2_distilled_tree_current_t_depth8 | current_t | 8     | 251        | 0.660781                    | 415950                       | 611095              | 303700                      | 0.680663                    | 0.365165                                    | 4.0603e+06           | -1.8735e+06        | True             | 0                      |
| new4  | phc_stp_v2_distilled_tree_current_t_depth8 | current_t | 8     | 251        | 0.635537                    | 91285                        | 129160              | 72415                       | 0.706759                    | 0.33254                                     | 1.37666e+06          | -706445            | True             | 0                      |

## Claim Verdict

Claim under test: `PHC-lite decisions can be distilled into human-readable shallow rules (roughly depth4-6 / 15-60 leaves) while preserving most of the PnL edge over the simple rule.`

Verdict: Not supported as stated: the best validation-selected tree still needs 251 leaves and keeps only 33.3% of the PHC-over-simple causal-fill advantage at the weakest split. The compact depth4-6 family is useful but not yet a 'mostly preserves the edge' replacement.

Future-work verdict: worth keeping as a concrete compression/interpretability direction, but not worth rushing into demo hardware before a second-stage rule search. The current evidence is diagnostic, not deployment-ready.

## Best Compact Depth4-6 Candidate

For the user-requested shallow range, the best validation candidate keeps at least 33.1% of PHC-over-simple advantage across val/NEW3/NEW4 (direct PHC PnL ratio min 66.3%).

| split | policy_id                                          | variant           | depth | leaf_nodes | score_r2_vs_teacher_uniform | causal_sequential_excess_pnl | phc_causal_fill_pnl | simple_rule_causal_fill_pnl | causal_pnl_retention_vs_phc | causal_advantage_retention_vs_simple_to_phc | total_excess_pnl_H20 | matched_random_95p | beats_random_95p | matched_random_p_value |
| ----- | -------------------------------------------------- | ----------------- | ----- | ---------- | --------------------------- | ---------------------------- | ------------------- | --------------------------- | --------------------------- | ------------------------------------------- | -------------------- | ------------------ | ---------------- | ---------------------- |
| val   | phc_stp_v2_distilled_tree_lag2_t_to_tminus2_depth6 | lag2_t_to_tminus2 | 6     | 64         | 0.634601                    | 139140                       | 171835              | 103260                      | 0.80973                     | 0.523223                                    | 1.61934e+06          | -903670            | True             | 0                      |
| new3  | phc_stp_v2_distilled_tree_lag2_t_to_tminus2_depth6 | lag2_t_to_tminus2 | 6     | 64         | 0.645583                    | 405445                       | 611095              | 303700                      | 0.663473                    | 0.330991                                    | 4.12043e+06          | -1.94771e+06       | True             | 0                      |
| new4  | phc_stp_v2_distilled_tree_lag2_t_to_tminus2_depth6 | lag2_t_to_tminus2 | 6     | 64         | 0.618792                    | 156375                       | 129160              | 72415                       | 1.21071                     | 1.4796                                      | 1.3858e+06           | -728779            | True             | 0                      |

## Development Top Candidates

These rows are sorted by validation causal-fill PnL. NEW3/NEW4 were not used for this ranking.

| split | policy_id                                          | variant           | depth | leaf_nodes | score_r2_vs_teacher_uniform | causal_sequential_excess_pnl | phc_causal_fill_pnl | simple_rule_causal_fill_pnl | causal_pnl_retention_vs_phc | causal_advantage_retention_vs_simple_to_phc | total_excess_pnl_H20 | matched_random_95p | beats_random_95p | matched_random_p_value |
| ----- | -------------------------------------------------- | ----------------- | ----- | ---------- | --------------------------- | ---------------------------- | ------------------- | --------------------------- | --------------------------- | ------------------------------------------- | -------------------- | ------------------ | ---------------- | ---------------------- |
| val   | phc_stp_v2_distilled_tree_current_t_depth8         | current_t         | 8     | 251        | 0.643872                    | 143555                       | 171835              | 103260                      | 0.835424                    | 0.587605                                    | 1.62703e+06          | -846998            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_lag1_t_tminus1_depth8    | lag1_t_tminus1    | 8     | 255        | 0.681535                    | 140860                       | 171835              | 103260                      | 0.81974                     | 0.548305                                    | 1.71012e+06          | -898878            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_lag2_t_to_tminus2_depth6 | lag2_t_to_tminus2 | 6     | 64         | 0.634601                    | 139140                       | 171835              | 103260                      | 0.80973                     | 0.523223                                    | 1.61934e+06          | -903670            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_lag3_t_to_tminus3_depth8 | lag3_t_to_tminus3 | 8     | 255        | 0.682649                    | 138925                       | 171835              | 103260                      | 0.808479                    | 0.520087                                    | 1.70669e+06          | -911724            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_lag3_t_to_tminus3_depth6 | lag3_t_to_tminus3 | 6     | 64         | 0.634512                    | 138780                       | 171835              | 103260                      | 0.807635                    | 0.517973                                    | 1.61782e+06          | -903884            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_lag2_t_to_tminus2_depth8 | lag2_t_to_tminus2 | 8     | 255        | 0.683665                    | 138710                       | 171835              | 103260                      | 0.807228                    | 0.516952                                    | 1.71054e+06          | -908886            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_current_t_depth6         | current_t         | 6     | 64         | 0.616692                    | 137075                       | 171835              | 103260                      | 0.797713                    | 0.49311                                     | 1.58484e+06          | -850534            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_current_t_depth5         | current_t         | 5     | 32         | 0.589975                    | 136930                       | 171835              | 103260                      | 0.796869                    | 0.490995                                    | 1.58314e+06          | -885782            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_lag1_t_tminus1_depth6    | lag1_t_tminus1    | 6     | 64         | 0.633534                    | 136335                       | 171835              | 103260                      | 0.793406                    | 0.482319                                    | 1.6257e+06           | -870715            | True             | 0                      |
| val   | phc_stp_v2_distilled_tree_lag2_t_to_tminus2_depth5 | lag2_t_to_tminus2 | 5     | 32         | 0.599609                    | 129760                       | 171835              | 103260                      | 0.755143                    | 0.386438                                    | 1.56245e+06          | -805874            | True             | 0                      |

## Score-Distillation Fit Quality

| variant           | lags    | depth | feature_count | train_rows | val_rows | leaf_nodes | node_count | max_depth_observed | r2_train_uniform | r2_val_uniform | r2_train_bid | r2_train_ask | r2_val_bid | r2_val_ask | fit_split | selection_split | new3_used_for_selection | new4_used_for_selection | new5_used_for_selection |
| ----------------- | ------- | ----- | ------------- | ---------- | -------- | ---------- | ---------- | ------------------ | ---------------- | -------------- | ------------ | ------------ | ---------- | ---------- | --------- | --------------- | ----------------------- | ----------------------- | ----------------------- |
| current_t         | 0       | 2     | 16            | 77194      | 82960    | 4          | 7          | 2                  | 0.42375          | 0.40381        | 0.42136      | 0.42613      | 0.40281    | 0.40481    | train     | val             | False                   | False                   | False                   |
| current_t         | 0       | 3     | 16            | 77194      | 82960    | 8          | 15         | 3                  | 0.49731          | 0.48538        | 0.49317      | 0.50144      | 0.48216    | 0.48861    | train     | val             | False                   | False                   | False                   |
| current_t         | 0       | 4     | 16            | 77194      | 82960    | 16         | 31         | 4                  | 0.55004          | 0.54526        | 0.54751      | 0.55258      | 0.54354    | 0.54699    | train     | val             | False                   | False                   | False                   |
| current_t         | 0       | 5     | 16            | 77194      | 82960    | 32         | 63         | 5                  | 0.59866          | 0.58997        | 0.59664      | 0.60067      | 0.58862    | 0.59133    | train     | val             | False                   | False                   | False                   |
| current_t         | 0       | 6     | 16            | 77194      | 82960    | 64         | 127        | 6                  | 0.62694          | 0.61669        | 0.6253       | 0.62858      | 0.61556    | 0.61782    | train     | val             | False                   | False                   | False                   |
| current_t         | 0       | 8     | 16            | 77194      | 82960    | 251        | 501        | 8                  | 0.66331          | 0.64387        | 0.66274      | 0.66389      | 0.64385    | 0.64388    | train     | val             | False                   | False                   | False                   |
| lag1_t_tminus1    | 0,1     | 2     | 32            | 77194      | 82960    | 4          | 7          | 2                  | 0.42375          | 0.40381        | 0.42136      | 0.42613      | 0.40281    | 0.40481    | train     | val             | False                   | False                   | False                   |
| lag1_t_tminus1    | 0,1     | 3     | 32            | 77194      | 82960    | 8          | 15         | 3                  | 0.50265          | 0.49324        | 0.49856      | 0.50674      | 0.49029    | 0.49619    | train     | val             | False                   | False                   | False                   |
| lag1_t_tminus1    | 0,1     | 4     | 32            | 77194      | 82960    | 16         | 31         | 4                  | 0.56507          | 0.55852        | 0.56166      | 0.56848      | 0.55616    | 0.56089    | train     | val             | False                   | False                   | False                   |
| lag1_t_tminus1    | 0,1     | 5     | 32            | 77194      | 82960    | 32         | 63         | 5                  | 0.60735          | 0.59854        | 0.60485      | 0.60985      | 0.59659    | 0.60047    | train     | val             | False                   | False                   | False                   |
| lag1_t_tminus1    | 0,1     | 6     | 32            | 77194      | 82960    | 64         | 127        | 6                  | 0.64043          | 0.63353        | 0.6386       | 0.64226      | 0.63215    | 0.63492    | train     | val             | False                   | False                   | False                   |
| lag1_t_tminus1    | 0,1     | 8     | 32            | 77194      | 82960    | 255        | 509        | 8                  | 0.69599          | 0.68153        | 0.69461      | 0.69737      | 0.68039    | 0.68268    | train     | val             | False                   | False                   | False                   |
| lag2_t_to_tminus2 | 0,1,2   | 2     | 48            | 77194      | 82960    | 4          | 7          | 2                  | 0.42375          | 0.40381        | 0.42136      | 0.42613      | 0.40281    | 0.40481    | train     | val             | False                   | False                   | False                   |
| lag2_t_to_tminus2 | 0,1,2   | 3     | 48            | 77194      | 82960    | 8          | 15         | 3                  | 0.50265          | 0.49324        | 0.49856      | 0.50674      | 0.49029    | 0.49619    | train     | val             | False                   | False                   | False                   |
| lag2_t_to_tminus2 | 0,1,2   | 4     | 48            | 77194      | 82960    | 16         | 31         | 4                  | 0.56507          | 0.55852        | 0.56166      | 0.56848      | 0.55616    | 0.56089    | train     | val             | False                   | False                   | False                   |
| lag2_t_to_tminus2 | 0,1,2   | 5     | 48            | 77194      | 82960    | 32         | 63         | 5                  | 0.60751          | 0.59961        | 0.60472      | 0.61031      | 0.59754    | 0.60168    | train     | val             | False                   | False                   | False                   |
| lag2_t_to_tminus2 | 0,1,2   | 6     | 48            | 77194      | 82960    | 64         | 127        | 6                  | 0.64199          | 0.6346         | 0.64011      | 0.64387      | 0.63313    | 0.63607    | train     | val             | False                   | False                   | False                   |
| lag2_t_to_tminus2 | 0,1,2   | 8     | 48            | 77194      | 82960    | 255        | 509        | 8                  | 0.69798          | 0.68367        | 0.69666      | 0.69931      | 0.68253    | 0.6848     | train     | val             | False                   | False                   | False                   |
| lag3_t_to_tminus3 | 0,1,2,3 | 2     | 64            | 77194      | 82960    | 4          | 7          | 2                  | 0.42375          | 0.40381        | 0.42136      | 0.42613      | 0.40281    | 0.40481    | train     | val             | False                   | False                   | False                   |
| lag3_t_to_tminus3 | 0,1,2,3 | 3     | 64            | 77194      | 82960    | 8          | 15         | 3                  | 0.50265          | 0.49324        | 0.49856      | 0.50674      | 0.49029    | 0.49619    | train     | val             | False                   | False                   | False                   |
| lag3_t_to_tminus3 | 0,1,2,3 | 4     | 64            | 77194      | 82960    | 16         | 31         | 4                  | 0.56507          | 0.55852        | 0.56166      | 0.56848      | 0.55616    | 0.56089    | train     | val             | False                   | False                   | False                   |
| lag3_t_to_tminus3 | 0,1,2,3 | 5     | 64            | 77194      | 82960    | 32         | 63         | 5                  | 0.60753          | 0.59965        | 0.60472      | 0.61033      | 0.59756    | 0.60172    | train     | val             | False                   | False                   | False                   |
| lag3_t_to_tminus3 | 0,1,2,3 | 6     | 64            | 77194      | 82960    | 64         | 127        | 6                  | 0.64203          | 0.63451        | 0.64012      | 0.64392      | 0.633      | 0.63603    | train     | val             | False                   | False                   | False                   |
| lag3_t_to_tminus3 | 0,1,2,3 | 8     | 64            | 77194      | 82960    | 255        | 509        | 8                  | 0.69712          | 0.68265        | 0.6958       | 0.69844      | 0.6814     | 0.6839     | train     | val             | False                   | False                   | False                   |

## Human-Readable Rule Example

The tree below predicts `[teacher_bid_score, teacher_ask_score]`. The full exported text is also saved at `results/selected_distilled_tree_rules.txt`.

```text
|-- t__queue_imbalance <= -0.22040
|  |-- t__imbalance_change_past <= 0.15210
|  |  |-- t__queue_imbalance <= -1.12305
|  |  |  |-- t__imbalance_change_past <= -1.33350
|  |  |  |  |-- t__spread_change_past <= -1.00394
|  |  |  |  |  |-- t__spread_ticks <= -1.14404
|  |  |  |  |  |  |-- t__inter_arrival_ms <= -0.00307
|  |  |  |  |  |  |  |-- t__log1p_ask_sz_1 <= 2.57910
|  |  |  |  |  |  |  |  |-- value: [25.90569, -55.88587]
|  |  |  |  |  |  |  |-- t__log1p_ask_sz_1 >  2.57910
|  |  |  |  |  |  |  |  |-- value: [-95.85951, 70.34934]
|  |  |  |  |  |  |-- t__inter_arrival_ms >  -0.00307
|  |  |  |  |  |  |  |-- t__imbalance_change_past <= -4.54883
|  |  |  |  |  |  |  |  |-- value: [-78.22089, 42.99699]
|  |  |  |  |  |  |  |-- t__imbalance_change_past >  -4.54883
|  |  |  |  |  |  |  |  |-- value: [-10.32764, -17.68274]
|  |  |  |  |  |-- t__spread_ticks >  -1.14404
|  |  |  |  |  |  |-- t__top_depth <= -0.24115
|  |  |  |  |  |  |  |-- t__clipped_inter_arrival_ms <= 0.33850
|  |  |  |  |  |  |  |  |-- value: [-16.56998, -11.93618]
|  |  |  |  |  |  |  |-- t__clipped_inter_arrival_ms >  0.33850
|  |  |  |  |  |  |  |  |-- value: [-35.33958, 6.00651]
|  |  |  |  |  |  |-- t__top_depth >  -0.24115
|  |  |  |  |  |  |  |-- t__clipped_inter_arrival_ms <= 0.66870
|  |  |  |  |  |  |  |  |-- value: [-35.48309, 6.15675]
|  |  |  |  |  |  |  |-- t__clipped_inter_arrival_ms >  0.66870
|  |  |  |  |  |  |  |  |-- value: [-55.83897, 25.97111]
|  |  |  |  |-- t__spread_change_past >  -1.00394
|  |  |  |  |  |-- t__bid_sz_1 <= -0.66846
|  |  |  |  |  |  |-- t__spread_change_past <= 0.80466
|  |  |  |  |  |  |  |-- t__deep_imbalance <= -1.86621
|  |  |  |  |  |  |  |  |-- value: [-139.64833, 108.53777]
|  |  |  |  |  |  |  |-- t__deep_imbalance >  -1.86621
|  |  |  |  |  |  |  |  |-- value: [-82.87386, 51.23694]
|  |  |  |  |  |  |-- t__spread_change_past >  0.80466
|  |  |  |  |  |  |  |-- t__spread_change_past <= 2.61328
|  |  |  |  |  |  |  |  |-- value: [-53.39414, 26.11504]
|  |  |  |  |  |  |  |-- t__spread_change_past >  2.61328
|  |  |  |  |  |  |  |  |-- value: [-71.12695, 42.44173]
|  |  |  |  |  |-- t__bid_sz_1 >  -0.66846
|  |  |  |  |  |  |-- t__log1p_top_depth <= 1.72363
|  |  |  |  |  |  |  |-- t__spread_ticks <= 0.00366
|  |  |  |  |  |  |  |  |-- value: [-51.08076, 24.68113]
|  |  |  |  |  |  |  |-- t__spread_ticks >  0.00366
|  |  |  |  |  |  |  |  |-- value: [-35.13615, 6.52942]
|  |  |  |  |  |  |-- t__log1p_top_depth >  1.72363
|  |  |  |  |  |  |  |-- t__imbalance_change_past <= -3.42578
|  |  |  |  |  |  |  |  |-- value: [-30.12421, 1.19306]
|  |  |  |  |  |  |  |-- t__imbalance_change_past >  -3.42578
|  |  |  |  |  |  |  |  |-- value: [-4.72928, -24.11929]
|  |  |  |-- t__imbalance_change_past >  -1.33350
|  |  |  |  |-- t__log1p_bid_sz_1 <= -0.30096
|  |  |  |  |  |-- t__imbalance_change_past <= -0.03502
|  |  |  |  |  |  |-- t__spread_change_past <= -1.00394
|  |  |  |  |  |  |  |-- t__volatility_past_50 <= 1.15820
|  |  |  |  |  |  |  |  |-- value: [-44.34794, 17.26916]
|  |  |  |  |  |  |  |-- t__volatility_past_50 >  1.15820
|  |  |  |  |  |  |  |  |-- value: [-14.57599, -15.27232]
|  |  |  |  |  |  |-- t__spread_change_past >  -1.00394
|  |  |  |  |  |  |  |-- t__deep_imbalance <= -0.50830
|  |  |  |  |  |  |  |  |-- value: [-108.78330, 78.13597]
|  |  |  |  |  |  |  |-- t__deep_imbalance >  -0.50830
|  |  |  |  |  |  |  |  |-- value: [-92.66259, 62.58182]
|  |  |  |  |  |-- t__imbalance_change_past >  -0.03502
|  |  |  |  |  |  |-- t__log1p_bid_sz_1 <= -0.92993
|  |  |  |  |  |  |  |-- t__deep_imbalance <= -0.57983
|  |  |  |  |  |  |  |  |-- value: [-88.28603, 57.59492]
|  |  |  |  |  |  |  |-- t__deep_imbalance >  -0.57983
|  |  |  |  |  |  |  |  |-- value: [-71.50407, 41.04804]
|  |  |  |  |  |  |-- t__log1p_bid_sz_1 >  -0.92993
|  |  |  |  |  |  |  |-- t__spread_ticks <= -1.14404
|  |  |  |  |  |  |  |  |-- value: [-39.46149, 14.63451]
|  |  |  |  |  |  |  |-- t__spread_ticks >  -1.14404
|  |  |  |  |  |  |  |  |-- value: [-62.32609, 32.51196]
|  |  |  |  |-- t__log1p_bid_sz_1 >  -0.30096
|  |  |  |  |  |-- t__imbalance_change_past <= -0.08289
|  |  |  |  |  |  |-- t__top_depth <= 2.00146
|  |  |  |  |  |  |  |-- t__spread_ticks <= -1.14404
|  |  |  |  |  |  |  |  |-- value: [-25.67747, -2.36740]
|  |  |  |  |  |  |  |-- t__spread_ticks >  -1.14404
|  |  |  |  |  |  |  |  |-- value: [-77.78740, 47.34968]
|  |  |  |  |  |  |-- t__top_depth >  2.00146
|  |  |  |  |  |  |  |-- t__event_rate_past_50 <= 0.94434
|  |  |  |  |  |  |  |  |-- value: [-33.26017, 3.34894]
|  |  |  |  |  |  |  |-- t__event_rate_past_50 >  0.94434
|  |  |  |  |  |  |  |  |-- value: [-61.84726, 33.76346]
|  |  |  |  |  |-- t__imbalance_change_past >  -0.08289
|  |  |  |  |  |  |-- t__spread_ticks <= 0.00366
|  |  |  |  |  |  |  |-- t__queue_imbalance <= -1.63184
|  |  |  |  |  |  |  |  |-- value: [-29.31162, 3.61855]
|  |  |  |  |  |  |  |-- t__queue_imbalance >  -1.63184
|  |  |  |  |  |  |  |  |-- value: [-9.39797, -19.06949]
|  |  |  |  |  |  |-- t__spread_ticks >  0.00366
|  |  |  |  |  |  |  |-- t__zscore_volatility_past_50_by_train_stats <= 0.70557
|  |  |  |  |  |  |  |  |-- value: [-43.45230, 11.89212]
|  |  |  |  |  |  |  |-- t__zscore_volatility_past_50_by_train_stats >  0.70557
|  |  |  |  |  |  |  |  |-- value: [-30.17839, 1.99715]
|  |  |-- t__queue_imbalance >  -1.12305
|  |  |  |-- t__imbalance_change_past <= -0.19824
|  |  |  |  |-- t__imbalance_change_past <= -1.44043
|  |  |  |  |  |-- t__spread_change_past <= -1.00394
|  |  |  |  |  |  |-- t__imbalance_change_past <= -5.08594
|  |  |  |  |  |  |  |-- value: [-167.82549, 120.58586]
|  |  |  |  |  |  |-- t__imbalance_change_past >  -5.08594
|  |  |  |  |  |  |  |-- t__spread_ticks <= -1.14404
|  |  |  |  |  |  |  |  |-- value: [27.10246, -53.99420]
|  |  |  |  |  |  |  |-- t__spread_ticks >  -1.14404
|  |  |  |  |  |  |  |  |-- value: [5.99673, -33.22094]
|  |  |  |  |  |-- t__spread_change_past >  -1.00394
|  |  |  |  |  |  |-- t__spread_change_past <= 0.80466
|  |  |  |  |  |  |  |-- t__zscore_volatility_past_50_by_train_stats <= 2.55762
|  |  |  |  |  |  |  |  |-- value: [-63.92244, 31.89427]
|  |  |  |  |  |  |  |-- t__zscore_volatility_past_50_by_train_stats >  2.55762
|  |  |  |  |  |  |  |  |-- value: [7.99531, -43.68142]
|  |  |  |  |  |  |-- t__spread_change_past >  0.80466
|  |  |  |  |  |  |  |-- t__log1p_bid_sz_1 <= 0.16254
|  |  |  |  |  |  |  |  |-- value: [-30.10397, 1.30502]
|  |  |  |  |  |  |  |-- t__log1p_bid_sz_1 >  0.16254
|  |  |  |  |  |  |  |  |-- value: [-12.76675, -16.76214]
|  |  |  |  |-- t__imbalance_change_past >  -1.44043
... [full tree saved in results/selected_distilled_tree_rules.txt]
```

## Interpretation

- A high score R2 alone is not treated as success; the selected rule is judged through the existing policy threshold and causal-fill simulator.
- `causal_advantage_retention_vs_simple_to_phc = (tree - simple_rule) / (PHC - simple_rule)`, so 100% means matching the PHC teacher's improvement over the simple queue-imbalance rule.
- If this phase is positive or mixed, the next step would be a separate hardware feasibility phase for the selected shallow tree. This report itself makes no FPGA latency or PnL-improvement claim.

## Baseline Values Used

| split | PHC causal-fill | simple-rule causal-fill | PHC action PnL | simple-rule action PnL |
|---|---:|---:|---:|---:|
| val | 171835 | 103260 | 2.03792e+06 | 1.23714e+06 |
| new3 | 611095 | 303700 | 5.23741e+06 | 2.95477e+06 |
| new4 | 129160 | 72415 | 1.87168e+06 | 1.00391e+06 |

## Forbidden Claims

- This does not prove FPGA PnL improvement.
- This does not prove real-world profitability.
- This does not use NEW5.
- This does not retrain or modify the PHC-lite checkpoint.
- This does not select depth or hyperparameters on NEW3/NEW4.
- Matched-random trials per split/variant: `1000`.
