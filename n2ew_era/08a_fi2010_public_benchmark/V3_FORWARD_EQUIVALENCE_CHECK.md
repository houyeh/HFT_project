# V3 Forward Equivalence Check

This check compares the vectorized full-window `forward()` path against repeated calls to `forward_step()` over the same 100-timestep input window.

- max_abs_diff: `1.78813934e-07`
- threshold: `1.0e-05`
- pass: `True`
- trainable_parameters: `2132`
- decay_min: `0.899999976`
- decay_max: `0.899999976`
- alpha_scalar: `0.25`

Interpretation: a pass means the v3 CPU-efficient `forward()` implementation preserves the true recurrent step semantics up to floating-point accumulation noise.
