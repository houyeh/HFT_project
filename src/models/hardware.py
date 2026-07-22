"""Utilities for MAC profiling and post-training quantization experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def _num_weight_params(module: nn.Module) -> int:
    if getattr(module, "weight", None) is None:
        return 0
    return int(module.weight.numel())


def _module_macs(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> int:
    if isinstance(module, nn.Conv1d):
        batch_size = int(output.shape[0])
        out_channels = int(output.shape[1])
        out_steps = int(output.shape[2])
        kernel_size = int(module.kernel_size[0])
        in_channels = int(module.in_channels)
        groups = int(module.groups)
        return batch_size * out_channels * out_steps * (in_channels // groups) * kernel_size

    if isinstance(module, nn.Linear):
        batch_size = int(output.shape[0]) if output.ndim > 1 else 1
        return batch_size * int(module.in_features) * int(module.out_features)

    return 0


def profile_model_macs(model: nn.Module, example_input: torch.Tensor) -> dict:
    handles = []
    rows: list[dict] = []

    def hook(name: str):
        def _capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            macs = _module_macs(module, inputs, output)
            if macs <= 0:
                return
            rows.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "weight_params": _num_weight_params(module),
                    "macs": int(macs),
                }
            )

        return _capture

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(example_input)

    for handle in handles:
        handle.remove()

    total_params = sum(int(p.numel()) for p in model.parameters())
    total_trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total_macs = sum(int(row["macs"]) for row in rows)
    return {
        "total_params": total_params,
        "trainable_params": total_trainable_params,
        "total_macs": int(total_macs),
        "weight_bytes_fp32": int(total_params * 4),
        "weight_bytes_int8": int(total_params),
        "module_rows": rows,
    }


def _observer_batches(calibration_batches: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    cached: list[torch.Tensor] = []
    for batch in calibration_batches:
        cached.append(batch.detach().cpu())
    if not cached:
        raise ValueError("Calibration batches are empty.")
    return cached


def quantize_model_fx_ptq(
    model: nn.Module,
    example_input: torch.Tensor,
    calibration_batches: Iterable[torch.Tensor],
    backend: str | None = None,
) -> nn.Module:
    try:
        import torch.ao.quantization.quantize_fx as quantize_fx
    except ImportError as exc:
        raise RuntimeError("FX PTQ is not available in this PyTorch build.") from exc

    supported_engines = list(torch.backends.quantized.supported_engines)
    selected_backend = backend or torch.backends.quantized.engine
    if selected_backend not in supported_engines:
        if not supported_engines:
            raise RuntimeError("No quantized backend is available in this PyTorch build.")
        selected_backend = supported_engines[0]

    torch.backends.quantized.engine = selected_backend
    model = model.eval().cpu()
    example_input = example_input.detach().cpu()
    calibration_cache = _observer_batches(calibration_batches)

    qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(
        torch.ao.quantization.get_default_qconfig(selected_backend)
    )
    prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_input)
    with torch.no_grad():
        for batch in calibration_cache:
            prepared(batch)
    quantized = quantize_fx.convert_fx(prepared)
    return quantized.eval()
