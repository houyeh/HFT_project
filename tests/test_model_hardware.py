"""Smoke tests for hardware profiling and FX PTQ utilities."""

import torch

from src.models.hardware import profile_model_macs, quantize_model_fx_ptq
from src.models.tcn import TCNClassifier


def _build_model() -> TCNClassifier:
    return TCNClassifier(
        num_inputs=5,
        num_classes=3,
        channel_list=[4, 4],
        kernel_size=3,
        dropout=0.0,
    )


def test_profile_model_macs_reports_positive_counts():
    model = _build_model().eval()
    example = torch.randn(2, 16, 5)

    profile = profile_model_macs(model, example)

    assert profile["total_params"] > 0
    assert profile["total_macs"] > 0
    assert profile["weight_bytes_int8"] < profile["weight_bytes_fp32"]
    assert len(profile["module_rows"]) >= 1


def test_quantize_model_fx_ptq_runs_forward():
    model = _build_model().eval()
    example = torch.randn(2, 16, 5)
    calibration = [torch.randn(2, 16, 5) for _ in range(2)]

    quantized = quantize_model_fx_ptq(model, example, calibration)
    output = quantized(example)

    assert output.shape == (2, 3)
