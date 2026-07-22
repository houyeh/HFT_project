# Frozen Contract Verification

Phase 04C hard-fails before NEW5 evaluation if any frozen contract item mismatches.

| contract_item     | expected                                      | actual                                        | pass   |
|:------------------|:----------------------------------------------|:----------------------------------------------|:-------|
| product_filter    | ^TXF[A-Z0-9]*$                                | ^TXF[A-Z0-9]*$                                | True   |
| target            | excess_pnl_H20                                | excess_pnl_H20                                | True   |
| model_id          | ml_hgb_regressor_shallow_convergence_selected | ml_hgb_regressor_shallow_convergence_selected | True   |
| hgb_max_iter      | 120                                           | 120                                           | True   |
| hgb_actual_n_iter | 98                                            | 98                                            | True   |
| threshold         | 0.22419440373778343                           | 0.22419440373778343                           | True   |
| queue_rule        | side_by_queue_imbalance_abs_0p1               | side_by_queue_imbalance_abs_0p1               | True   |
| fill_proxy        | conservative_l2_stress, H20, inventory_cap=1  | conservative_l2_stress, H20, inventory_cap=1  | True   |
| holdout_scope     | NEW5 20260128 only                            | NEW5 20260128 only                            | True   |

Overall pass: `True`.
