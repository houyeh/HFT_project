# FI-2010 Decay Parameter Survey

status: decay_zero_is_universal_in_08a_v2_checkpoints_but_fixed_by_implementation

## Scope

This survey loads existing Stage 1 v2 FI-2010 checkpoints from `models/`. It does not retrain, modify checkpoints, or use TXF NEW1-NEW5 data.

## Result

| item | value |
| --- | ---: |
| checkpoint files surveyed | 22 |
| primary horizon/seed policies covered | 15 |
| all checkpoint decay_abs_max == 0 | True |
| all checkpoint alpha_abs == 0 | True |
| primary policies decay_abs_max == 0 | True |
| primary policies alpha_abs == 0 | True |
| decay source key(s) | decay |
| sigmoid applied to decay | False |

## Honest Conclusion

The zero-decay / zero-alpha behavior is universal across the surveyed Stage 1 v2 FI-2010 checkpoints, including all 15 primary horizon/seed policies. It is not a one-off peculiarity of `k=50/seed=1/raw`.

`decay_parameter_survey.csv` intentionally includes every existing `*_v2.pt` file, including raw and weighted variants. `decay_parameter_survey_primary_15.csv` is the compact 5-horizon x 3-seed primary-policy view.

However, this should not be described as training convergence to zero. The FI-2010 implementation stores `decay` and `alpha_scalar` as non-trainable buffers initialized to zero, unlike the TXF 05A model where `decay_param` is learnable and the actual memory coefficient is `sigmoid(decay_param)`. Therefore the correct statement is:

> In the FI-2010 Stage 1 v2 implementation, recurrent memory and the within-step message loop are disabled by construction for all surveyed checkpoints.

This supports using the Stage 2 HLS result as a fixed step-form classifier feasibility check, but it is not evidence that FI-2010 training learned to suppress recurrence.
