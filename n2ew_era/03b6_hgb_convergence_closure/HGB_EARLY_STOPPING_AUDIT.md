# HGB Early Stopping Audit

Explicit early stopping was configured with `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`, and `tol=1e-7`. The internal validation fraction is drawn from the training split only; frozen OOS splits are not used.

| model_id                     |   max_iter |   n_iter | terminated_before_cap   |   tol |   n_iter_no_change |   validation_fraction |   train_rmse |   val_rmse |   train_pr_auc_positive_excess |   val_pr_auc_positive_excess |   val_spearman |   selected_threshold |   val_policy_total_excess_pnl | selection_split   | test_new3_new4_used_for_selection   |
|:-----------------------------|-----------:|---------:|:------------------------|------:|-------------------:|----------------------:|-------------:|-----------:|-------------------------------:|-----------------------------:|---------------:|---------------------:|------------------------------:|:------------------|:------------------------------------|
| hgb_regressor_early_stop_320 |        320 |      121 | True                    | 1e-07 |                 10 |                   0.1 |      147.398 |    239.657 |                       0.386254 |                     0.374895 |       0.297661 |                    0 |                   1.23607e+07 | val               | False                               |
| hgb_regressor_early_stop_500 |        500 |      121 | True                    | 1e-07 |                 10 |                   0.1 |      147.398 |    239.657 |                       0.386254 |                     0.374895 |       0.297661 |                    0 |                   1.23607e+07 | val               | False                               |
| hgb_regressor_early_stop_800 |        800 |      121 | True                    | 1e-07 |                 10 |                   0.1 |      147.398 |    239.657 |                       0.386254 |                     0.374895 |       0.297661 |                    0 |                   1.23607e+07 | val               | False                               |

Staged internal scores are written to `results/hgb_early_stopping_staged_scores.csv`.
