# ML Evaluation Contract

Primary targets:

- Regression: `excess_pnl_H20`
- Binary classification: `positive_excess_H20 = excess_pnl_H20 > 0`

Validation split is strictly for model family selection, hyperparameter selection, calibration, and threshold/action-policy selection.

Test, NEW3, and NEW4 are strictly frozen out-of-sample. They must not be used for feature selection, target choice, threshold selection, model selection, hyperparameter tuning, or early stopping.

Future labels are forbidden as model inputs. Raw PnL is not the main target because raw markout contains mechanical half-spread capture.

Required frozen evaluations:

- Excess markout H20 PnL
- Positive excess classification metrics
- Matched random 95p comparison
- Sequential excess inventory proxy with causal fill
- Split-by-side and split-by-session reporting
