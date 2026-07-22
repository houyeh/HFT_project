status: resolved
# Audit: Why HGB Early-Stopping Runs All Stop at `n_iter=121`

The three runs differ only in requested `max_iter` (320/500/800). All use `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=10`, and `tol=1e-7` on the training split only. Therefore once the same internal validation curve stops improving, all three caps terminate at the same `n_iter=121`.

## Early-Stopping Audit
| model_id                     |   max_iter |   n_iter | terminated_before_cap   |   tol |   n_iter_no_change |   validation_fraction |   train_rmse |   val_rmse |   train_pr_auc_positive_excess |   val_pr_auc_positive_excess |   val_spearman |   selected_threshold |   val_policy_total_excess_pnl | selection_split   | test_new3_new4_used_for_selection   |
|:-----------------------------|-----------:|---------:|:------------------------|------:|-------------------:|----------------------:|-------------:|-----------:|-------------------------------:|-----------------------------:|---------------:|---------------------:|------------------------------:|:------------------|:------------------------------------|
| hgb_regressor_early_stop_320 |        320 |      121 | True                    | 1e-07 |                 10 |                   0.1 |      147.398 |    239.657 |                       0.386254 |                     0.374895 |       0.297661 |                    0 |                   1.23607e+07 | val               | False                               |
| hgb_regressor_early_stop_500 |        500 |      121 | True                    | 1e-07 |                 10 |                   0.1 |      147.398 |    239.657 |                       0.386254 |                     0.374895 |       0.297661 |                    0 |                   1.23607e+07 | val               | False                               |
| hgb_regressor_early_stop_800 |        800 |      121 | True                    | 1e-07 |                 10 |                   0.1 |      147.398 |    239.657 |                       0.386254 |                     0.374895 |       0.297661 |                    0 |                   1.23607e+07 | val               | False                               |

## Last 12 staged scores per run
| model_id                     |   iteration |   train_score |   validation_score |
|:-----------------------------|------------:|--------------:|-------------------:|
| hgb_regressor_early_stop_320 |         110 |      -10970.4 |           -10042.5 |
| hgb_regressor_early_stop_320 |         111 |      -10968   |           -10041.2 |
| hgb_regressor_early_stop_320 |         112 |      -10967.6 |           -10041.5 |
| hgb_regressor_early_stop_320 |         113 |      -10966.8 |           -10041.8 |
| hgb_regressor_early_stop_320 |         114 |      -10962.3 |           -10042.3 |
| hgb_regressor_early_stop_320 |         115 |      -10961.9 |           -10042.5 |
| hgb_regressor_early_stop_320 |         116 |      -10961.4 |           -10042.7 |
| hgb_regressor_early_stop_320 |         117 |      -10959.7 |           -10041.7 |
| hgb_regressor_early_stop_320 |         118 |      -10959.3 |           -10042.1 |
| hgb_regressor_early_stop_320 |         119 |      -10955.2 |           -10043.4 |
| hgb_regressor_early_stop_320 |         120 |      -10954.9 |           -10043.7 |
| hgb_regressor_early_stop_320 |         121 |      -10954.3 |           -10043.1 |
| hgb_regressor_early_stop_500 |         110 |      -10970.4 |           -10042.5 |
| hgb_regressor_early_stop_500 |         111 |      -10968   |           -10041.2 |
| hgb_regressor_early_stop_500 |         112 |      -10967.6 |           -10041.5 |
| hgb_regressor_early_stop_500 |         113 |      -10966.8 |           -10041.8 |
| hgb_regressor_early_stop_500 |         114 |      -10962.3 |           -10042.3 |
| hgb_regressor_early_stop_500 |         115 |      -10961.9 |           -10042.5 |
| hgb_regressor_early_stop_500 |         116 |      -10961.4 |           -10042.7 |
| hgb_regressor_early_stop_500 |         117 |      -10959.7 |           -10041.7 |
| hgb_regressor_early_stop_500 |         118 |      -10959.3 |           -10042.1 |
| hgb_regressor_early_stop_500 |         119 |      -10955.2 |           -10043.4 |
| hgb_regressor_early_stop_500 |         120 |      -10954.9 |           -10043.7 |
| hgb_regressor_early_stop_500 |         121 |      -10954.3 |           -10043.1 |
| hgb_regressor_early_stop_800 |         110 |      -10970.4 |           -10042.5 |
| hgb_regressor_early_stop_800 |         111 |      -10968   |           -10041.2 |
| hgb_regressor_early_stop_800 |         112 |      -10967.6 |           -10041.5 |
| hgb_regressor_early_stop_800 |         113 |      -10966.8 |           -10041.8 |
| hgb_regressor_early_stop_800 |         114 |      -10962.3 |           -10042.3 |
| hgb_regressor_early_stop_800 |         115 |      -10961.9 |           -10042.5 |
| hgb_regressor_early_stop_800 |         116 |      -10961.4 |           -10042.7 |
| hgb_regressor_early_stop_800 |         117 |      -10959.7 |           -10041.7 |
| hgb_regressor_early_stop_800 |         118 |      -10959.3 |           -10042.1 |
| hgb_regressor_early_stop_800 |         119 |      -10955.2 |           -10043.4 |
| hgb_regressor_early_stop_800 |         120 |      -10954.9 |           -10043.7 |
| hgb_regressor_early_stop_800 |         121 |      -10954.3 |           -10043.1 |

## Fixed-budget sweep context
| model_id               |   max_iter |   n_iter | hit_iteration_cap   |   val_policy_total_excess_pnl | test_new3_new4_used_for_selection   |
|:-----------------------|-----------:|---------:|:--------------------|------------------------------:|:------------------------------------|
| hgb_regressor_iter_40  |         40 |       40 | True                |                   1.12557e+07 | False                               |
| hgb_regressor_iter_80  |         80 |       80 | True                |                   1.2231e+07  | False                               |
| hgb_regressor_iter_120 |        120 |      120 | True                |                   1.2314e+07  | False                               |
| hgb_regressor_iter_160 |        160 |      160 | True                |                   1.23676e+07 | False                               |
| hgb_regressor_iter_240 |        240 |      240 | True                |                   1.25512e+07 | False                               |
| hgb_regressor_iter_320 |        320 |      320 | True                |                   1.25915e+07 | False                               |

Conclusion: `n_iter=121` is expected and reproducible under identical sklearn early-stopping settings. It is not evidence that frozen OOS splits were used, and it is not a hidden hard-coded 121 cap.
