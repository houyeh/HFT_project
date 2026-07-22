"""Conversion experiments that document why the TCN was rewritten by hand for HLS."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import hls4ml

from src.models.tcn import build_default_model


class ChannelFirstMeanWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.tcn = base.tcn
        self.fc = base.fc

    def forward(self, x):
        x = self.tcn(x)
        x = x.mean(dim=2)
        x = self.fc(x)
        return x


class ChannelFirstPoolWrapper(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.tcn = base.tcn
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = base.fc

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


def try_convert(model: nn.Module, variant: str, input_shape: tuple[int, ...], output_dir: Path) -> dict:
    report = {"variant": variant, "input_shape": list(input_shape)}
    try:
        config = hls4ml.utils.config_from_pytorch_model(
            model,
            input_shape=input_shape,
            granularity="name",
            backend="Vivado",
        )
        with (output_dir / f"{variant}_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            input_shape=input_shape,
            output_dir=str(output_dir / variant),
            project_name=f"tcn_{variant}",
            backend="Vivado",
            hls_config=config,
        )
        hls_model.write()
        report["conversion_succeeded"] = True
    except Exception as exc:  # noqa: BLE001
        report["conversion_succeeded"] = False
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    return report


def main() -> None:
    output_dir = Path("artifacts/04_hardware_handoff/hls4ml_attempt")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = build_default_model(num_inputs=12)
    state_dict = torch.load("artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt", map_location="cpu")
    base.load_state_dict(state_dict)
    base.eval()

    attempts = [
        ("original", base, (100, 12)),
        ("channel_first_mean", ChannelFirstMeanWrapper(base).eval(), (12, 100)),
        ("channel_first_pool", ChannelFirstPoolWrapper(base).eval(), (12, 100)),
    ]

    reports = [try_convert(model, variant, input_shape, output_dir) for variant, model, input_shape in attempts]
    with (output_dir / "attempt_report.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
