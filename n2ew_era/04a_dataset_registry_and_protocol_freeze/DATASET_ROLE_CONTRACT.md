# Dataset Role Contract

| Dataset group | Frozen role | Allowed usage | Forbidden usage |
|---|---|---|---|
| `NEW1` + `NEW2` | `development` | feature engineering, model fitting, validation selection inside development protocol | claiming final OOS robustness by itself |
| `NEW3` + `NEW4` | `used_oos_diagnostic` | robust diagnostics and stress checks after development selection is frozen | tuning, thresholding, model selection |
| `NEW5` | `final_holdout_locked` | one-shot final evaluation only after final development is complete | any tuning, thresholding, model selection, repeated probing, or PnL evaluation before final lock release |

`NEW5` is locked as the final untouched holdout. Phase 04A performed metadata/schema checks only.
