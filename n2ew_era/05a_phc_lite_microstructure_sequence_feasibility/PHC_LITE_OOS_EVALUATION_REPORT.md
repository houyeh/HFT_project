# PHC-lite OOS Evaluation Report

Evaluation is on the sequence target-state universe only. NEW5 is not loaded.

Comparison caveat: the frozen Phase 04B.1 HGB artifact was loaded under a different local sklearn version than the one that created it, which emitted an `InconsistentVersionWarning`. PHC-lite training/evaluation does not depend on that pickle; only the HGB comparison row carries this reproducibility caveat.

| split   |         phc |       queue |         hgb | beats_queue   | beats_hgb   |
|:--------|------------:|------------:|------------:|:--------------|:------------|
| val     | 1.92717e+06 | 1.23714e+06 | 1.67168e+06 | True          | True        |
| new3    | 5.02152e+06 | 2.95477e+06 | 4.01058e+06 | True          | True        |
| new4    | 1.7625e+06  | 1.00391e+06 | 1.39542e+06 | True          | True        |

Success criterion is positive val/NEW3/NEW4 PnL, beating queue rule, and being competitive with the frozen Phase 04B.1 HGB. This report is diagnostic and does not claim real-world profitability.
