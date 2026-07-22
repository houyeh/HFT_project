status: resolved
finding: positive
# Audit: Evaluation Methodology External Validation

## 9.1 Val-selected causal-fill policies checked on frozen splits
| phase   | policy                                        | selection_basis                                |   val_causal_fill_pnl | frozen_split   |   frozen_causal_fill_pnl | val_positive   | frozen_positive   | sign_agree_positive   | note                                    |
|:--------|:----------------------------------------------|:-----------------------------------------------|----------------------:|:---------------|-------------------------:|:---------------|:------------------|:----------------------|:----------------------------------------|
| 04B     | side_by_queue_imbalance_abs_0p1               | development val / final candidate scorecard    |           2.04026e+06 | NEW3           |              5.73154e+06 | True           | True              | True                  | TXF-only non-ML baseline                |
| 04B     | side_by_queue_imbalance_abs_0p1               | development val / final candidate scorecard    |           2.04026e+06 | NEW4           |              2.07526e+06 | True           | True              | True                  | TXF-only non-ML baseline                |
| 04B.1   | ml_hgb_regressor_shallow                      | validation threshold + convergence closure     |           2.8635e+06  | NEW3           |              7.65278e+06 | True           | True              | True                  | NEW3/NEW4 diagnostic OOS                |
| 04B.1   | ml_hgb_regressor_shallow                      | validation threshold + convergence closure     |           2.8635e+06  | NEW4           |              2.79185e+06 | True           | True              | True                  | NEW3/NEW4 diagnostic OOS                |
| 04C     | ml_hgb_regressor_shallow_convergence_selected | frozen contract from 04B.1                     |           2.8635e+06  | NEW5 one-shot  |         893555           | True           | True              | True                  | NEW5 final holdout; not used for tuning |
| 05A     | phc_lite_v1                                   | validation epoch/threshold only                |      170310           | NEW3           |         488870           | True           | True              | True                  | sequence-window universe                |
| 05A     | phc_lite_v1                                   | validation epoch/threshold only                |      170310           | NEW4           |         121765           | True           | True              | True                  | sequence-window universe                |
| 05C4    | phc_stp_v2_cosine_lr                          | validation seed/threshold + six adoption gates |      171835           | NEW3           |         611095           | True           | True              | True                  | NEW5 not used                           |
| 05C4    | phc_stp_v2_cosine_lr                          | validation seed/threshold + six adoption gates |      171835           | NEW4           |         129160           | True           | True              | True                  | NEW5 not used                           |

## 9.2 Sign consistency

- Policy/frozen-split observations: `9`
- Fraction where validation positive sign matched frozen positive sign: `1.000`
- Fraction of frozen observations with positive causal-fill PnL: `1.000`

This is a small, correlated sample of project phases, not a formal statistical proof. It does, however, show that the causal-fill sequential criterion did not immediately collapse on NEW3/NEW4, and the HGB candidate also stayed positive on the locked NEW5 one-shot holdout.

## 9.3 Answer for defense use

> The n2ew_era methodology has positive empirical support, not proof. Across the audited val-selected policies, frozen NEW3/NEW4 checks and the 04C NEW5 one-shot HGB holdout preserved positive causal-fill sequential PnL. The sample is still small and phase-correlated, so the claim should be limited to: the evaluation protocol appears useful for filtering proxy strategies, but it is not yet an independently proven guarantee of live profitability.
