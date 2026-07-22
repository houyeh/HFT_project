# Audit: PHC-lite Train Loss Greater Than Validation Loss

Status: resolved with caveat.

## Question

The PHC-lite training curve shows train loss higher than validation loss. This should not be hand-waved as automatically reasonable. The audit checks whether this is caused by train/eval-mode differences, regularization, BatchNorm-like behavior, or split difficulty.

## Code-Path Findings

| check | finding | evidence |
|---|---|---|
| Is dropout active? | No, the Phase 05A default checkpoint uses `dropout=0.0`. | `common_05a.py:617-626`, checkpoint field `dropout`; `models/phc_lite.py` has `nn.Dropout(dropout)`. |
| Is BatchNorm used? | No. PHC-lite uses `LayerNorm`, which does not keep train/eval running statistics like BatchNorm. | `models/phc_lite.py` encoder and loop norm definitions. |
| Is validation loss computed in eval mode? | Yes. `predict_scores()` calls `model.eval()`. | `common_05a.py:502-511`. |
| Is eval-mode train loss already logged? | Yes. `train_huber_loss` is recomputed on the full train split using `predict_scores()` in eval mode; it is separate from the mini-batch `train_loss`. | `common_05a.py:666-681`. |

## Loss Check

At the best validation-policy epoch:

| metric | value |
|---|---:|
| best_epoch | 48 |
| train_loss from training mini-batches | 0.2376149375 |
| train_huber_loss recomputed in eval mode | 0.2373948842 |
| val_huber_loss recomputed in eval mode | 0.2041162550 |
| eval-mode train minus val | 0.0332786292 |
| train-loop loss minus eval-mode train loss | 0.0002200533 |

At the last epoch:

| metric | value |
|---|---:|
| epoch | 50 |
| train_loss from training mini-batches | 0.2375180894 |
| train_huber_loss recomputed in eval mode | 0.2376231253 |
| val_huber_loss recomputed in eval mode | 0.2045303434 |
| eval-mode train minus val | 0.0330927819 |
| train-loop loss minus eval-mode train loss | -0.0001050359 |

Interpretation: the gap remains after using eval mode on the train split. Therefore the explanation is not dropout, train/eval mode, or BatchNorm behavior.

## Split Difficulty Check

Weighted target distribution from `results/phc_lite_sequence_manifest.csv`:

| split | side | windows | target_mean | target_std | positive_rate |
|---|---|---:|---:|---:|---:|
| train | bid | 77194 | -14.587856 | 205.718162 | 0.358357 |
| train | ask | 77194 | -15.411755 | 205.718158 | 0.351543 |
| val | bid | 82960 | -14.212150 | 149.123093 | 0.349867 |
| val | ask | 82960 | -15.786403 | 149.123101 | 0.343925 |
| new3 | bid | 176754 | -14.678961 | 156.031676 | 0.366781 |
| new3 | ask | 176754 | -15.320530 | 156.031677 | 0.367053 |
| new4 | bid | 68620 | -14.559458 | 158.149236 | 0.366263 |
| new4 | ask | 68620 | -15.439668 | 158.149232 | 0.365243 |

Train target standard deviation is about `205.7`, while validation is about `149.1`. That is roughly `38%` higher label dispersion in train than validation. Because the Huber target is `excess_pnl_H20 / target_scale`, a noisier/higher-dispersion train target can produce a higher train loss even under eval mode.

## Conclusion

The correct explanation is:

`train loss > val loss` in Phase 05A is not primarily caused by dropout or BatchNorm-style train/eval differences. The model uses dropout configured as `0.0`, no BatchNorm, and the eval-mode train loss remains higher than validation loss. The more defensible explanation is split difficulty: the train split has materially higher target dispersion than the validation split, so the train labels are harder/noisier under the same Huber objective.

Safe presentation wording:

> We checked this explicitly. The apparent inversion is not a dropout artifact: recomputing train loss in eval mode still gives train higher than validation. The validation split has substantially lower target dispersion, so it is an easier split under the Huber objective. We therefore do not interpret lower validation loss as over-generalization; we treat it as a split-regime difficulty difference and rely on NEW3/NEW4 OOS plus matched-random and causal-fill checks for robustness.

