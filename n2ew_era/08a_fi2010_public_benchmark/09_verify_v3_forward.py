from __future__ import annotations

from pathlib import Path

import torch

from fi2010_phc_lite import PHCLiteFI2010Classifier, count_trainable_parameters

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "V3_FORWARD_EQUIVALENCE_CHECK.md"
ATOL = 1e-5


def main() -> None:
    torch.manual_seed(0)
    model = PHCLiteFI2010Classifier()
    x = torch.randn(8, 100, 40)
    y_fast_vectorized = model(x)

    state = x.new_zeros((x.shape[0], model.hidden_dim))
    y_step = x.new_zeros((x.shape[0], model.num_classes))
    for t in range(x.shape[1]):
        y_step, state = model.forward_step(x[:, t, :], state)

    max_abs_diff = float((y_fast_vectorized - y_step).detach().abs().max())
    trainable_parameters = count_trainable_parameters(model)
    decay = model.decay.detach()
    decay_min = float(decay.min())
    decay_max = float(decay.max())
    alpha_scalar = float(model.alpha_scalar.detach())
    passed = max_abs_diff < ATOL
    rows = {
        "max_abs_diff": max_abs_diff,
        "threshold": ATOL,
        "passed": passed,
        "trainable_parameters": trainable_parameters,
        "decay_min": decay_min,
        "decay_max": decay_max,
        "alpha_scalar": alpha_scalar,
    }
    print(rows, flush=True)
    OUT.write_text(
        "\n".join(
            [
                "# V3 Forward Equivalence Check",
                "",
                "This check compares the vectorized full-window `forward()` path against repeated calls to `forward_step()` over the same 100-timestep input window.",
                "",
                f"- max_abs_diff: `{max_abs_diff:.9g}`",
                f"- threshold: `{ATOL:.1e}`",
                f"- pass: `{passed}`",
                f"- trainable_parameters: `{trainable_parameters}`",
                f"- decay_min: `{decay_min:.9g}`",
                f"- decay_max: `{decay_max:.9g}`",
                f"- alpha_scalar: `{alpha_scalar:.9g}`",
                "",
                "Interpretation: a pass means the v3 CPU-efficient `forward()` implementation preserves the true recurrent step semantics up to floating-point accumulation noise.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
