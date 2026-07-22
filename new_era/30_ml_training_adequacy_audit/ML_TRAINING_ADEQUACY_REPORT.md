# ML Training Adequacy Report

## 1. Purpose

Phase 030 audits whether the residual Quote-EV ML veto failed in Phase 028 because the model was too shallow or because the residual signal is not learnable from the available tabular causal features.

## 2. GPU Training Pack

Pack location: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\30_ml_training_adequacy_audit\export\ml_training_pack`.

- `train`: rows `820`, negative EV rate `0.276`, severe <= -5000 rate `0.211`, quote EV sum `-1,586,875`.
- `val`: rows `134`, negative EV rate `0.127`, severe <= -5000 rate `0.022`, quote EV sum `1,218,095`.
- `test`: rows `419`, negative EV rate `0.038`, severe <= -5000 rate `0.000`, quote EV sum `2,312,590`.
- `new4`: rows `714`, negative EV rate `0.144`, severe <= -5000 rate `0.050`, quote EV sum `6,348,770`.

The pack includes `sequence_index.parquet`, `README_GPU_TRAINING.md`, and `environment.yml`. Sequence rows are pointers only; full temporal windows are not expanded into RAM.

## 3. Pipeline Audit

- `phase_028_location`: `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\28_residual_quote_ev_veto` (read_only_reference).
- `phase_028_feature_count`: `10` (causal_tabular_features).
- `phase_028_model_family`: `logistic_l2_balanced,small_mlp_8` (shallow_baseline).
- `phase_028_validation_selection`: `validation_only` (pass).
- `phase_028_selected_veto_count`: `0` (no_veto_selected).
- `torch_available_locally`: `True` (available).
- `phase_030_max_veto_rate`: `<=5%` (locked).

## 4. Residual Universe Difficulty

- `train`: samples `820`, negative count `226`, <= -5000 count `173`, top-10 loss share `0.131`.
- `val`: samples `134`, negative count `17`, <= -5000 count `3`, top-10 loss share `0.836`.
- `test`: samples `419`, negative count `16`, <= -5000 count `0`, top-10 loss share `0.903`.
- `new4`: samples `714`, negative count `103`, <= -5000 count `36`, top-10 loss share `0.371`.

## 5. Existing Phase 028 Threshold Diagnostic

- `logistic_l2_balanced` threshold `0.833388`: val after-veto PnL `1,218,095`, veto count `0`.
- `small_mlp_8` threshold `0.621284`: val after-veto PnL `1,218,095`, veto count `0`.
- `small_mlp_8` threshold `0.621283`: val after-veto PnL `1,218,095`, veto count `1`.
- `logistic_l2_balanced` threshold `0.833387`: val after-veto PnL `1,218,060`, veto count `1`.
- `logistic_l2_balanced` threshold `0.689187`: val after-veto PnL `1,218,060`, veto count `2`.

## 6. Expanded Model Families

The expanded audit includes calibrated logistic, histogram gradient boosting, sklearn 64-32 MLP, and PyTorch 64-32 MLP classifier/regressor variants. PyTorch was run locally on CPU; the exported pack is intended for external RTX sequence training.

- `calibrated_logistic_sigmoid` (classification_negative_ev): val after-veto PnL `1,227,410`, veto count `1`, threshold `0.344667`.
- `hist_gradient_boosting_classifier` (classification_negative_ev): val after-veto PnL `1,218,095`, veto count `0`, threshold `0.772883`.
- `sklearn_mlp_64_32_classifier` (classification_negative_ev): val after-veto PnL `1,218,095`, veto count `0`, threshold `0.438027`.
- `ridge_quote_ev_regressor` (regression_quote_ev): val after-veto PnL `1,218,095`, veto count `0`, threshold `8005.171987`.
- `hist_gradient_boosting_regressor` (regression_quote_ev): val after-veto PnL `1,218,095`, veto count `0`, threshold `9850.178346`.
- `sklearn_mlp_64_32_regressor` (regression_quote_ev): val after-veto PnL `1,218,095`, veto count `0`, threshold `18.437602`.
- `torch_mlp_64_32_classifier_cpu` (classification_negative_ev): val after-veto PnL `1,218,095`, veto count `0`, threshold `0.998318`.
- `torch_mlp_64_32_regressor_cpu` (regression_quote_ev): val after-veto PnL `1,218,095`, veto count `0`, threshold `52064.101564`.

Selected validation model: `calibrated_logistic_sigmoid` with threshold `0.344667`.

- `val` PR-AUC `0.2688`, ROC-AUC `0.6511`, Spearman `0.1742`.
- `test` PR-AUC `0.0865`, ROC-AUC `0.7303`, Spearman `0.1529`.
- `new4` PR-AUC `0.1302`, ROC-AUC `0.4642`, Spearman `-0.0436`.

## 7. Final Incremental Value Test

- `val`: baseline `1,218,095`, ML after-veto `1,227,410`, incremental `9,315`, veto rate `0.007`, random 95p `1,223,810`, accepted `True`.
- `test`: baseline `2,312,590`, ML after-veto `2,312,590`, incremental `-0`, veto rate `0.000`, random 95p `2,312,590`, accepted `False`.
- `new4`: baseline `6,348,770`, ML after-veto `6,341,385`, incremental `-7,385`, veto rate `0.003`, random 95p `6,348,770`, accepted `False`.

## 8. Conclusion

Expanded tabular models did not unlock incremental value. This does not prove the residual signal is impossible, but it does show that the current small tabular residual universe is not adequate; the remaining test is external sequence TCN training using the exported GPU pack.

The useful-ML rule is strict: validation-selected low-veto ML must improve the frozen baseline and beat matched random veto 95p. Test and NEW4 are not used for model or threshold tuning.
