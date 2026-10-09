from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07l_ultrafast_policy_sweep"
RESULTS = ROOT / "results"
HLS_SRC = ROOT / "hls_src"
SWEEP = ROOT / "00_run_ultrafast_policy_sweep.py"
BITS = 12
INTEGER_BITS = 6
DATA_FRAC = BITS - INTEGER_BITS
ACC_FRAC = 16


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def format_array(name: str, values: np.ndarray) -> list[str]:
    if values.ndim == 1:
        return [f"static const int {name}[{len(values)}] = {{" + ", ".join(map(str, values.tolist())) + "};"]
    lines = [f"static const int {name}[{values.shape[0]}][{values.shape[1]}] = {{"]
    lines.extend("    {" + ", ".join(map(str, row.tolist())) + "}," for row in values)
    lines.append("};")
    return lines


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    RESULTS.mkdir(parents=True, exist_ok=True)
    HLS_SRC.mkdir(parents=True, exist_ok=True)
    phase = load_module("phase07l_export", SWEEP)
    selected = __import__("json").loads(
        (phase.PHASE07G / "results" / "tiny_mlp_selected.json").read_text(encoding="utf-8")
    )
    mean = np.asarray(selected["feature_mean"], dtype=np.float64)
    scale = np.asarray(selected["feature_scale"], dtype=np.float64)
    model = phase.QATTinyMLP(BITS, INTEGER_BITS)
    checkpoint = torch.load(
        ROOT / "models" / "qat_h8_q12_6.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    w1, _ = phase.quantize_int(model.fc1.weight.detach().numpy(), BITS, INTEGER_BITS)
    b1, _ = phase.quantize_int(model.fc1.bias.detach().numpy(), BITS, INTEGER_BITS)
    w2, _ = phase.quantize_int(model.fc2.weight.detach().numpy(), BITS, INTEGER_BITS)
    b2, _ = phase.quantize_int(model.fc2.bias.detach().numpy(), BITS, INTEGER_BITS)
    lines = ["#pragma once", "", "// Q12.6 AP_TRN/AP_SAT constants from the frozen QAT champion."]
    for name, values in (("QAT_W1", w1), ("QAT_B1", b1), ("QAT_W2", w2), ("QAT_B2", b2)):
        lines.extend(format_array(name, values))
        lines.append("")
    (HLS_SRC / "qat12_weights.hpp").write_text("\n".join(lines), encoding="ascii")

    payload = phase.helpers.load_payload("val")
    raw, _ = phase.helpers.base_features(payload)
    x = ((raw[:25_600] - mean) / scale).astype(np.float64)
    xq, input_saturation = phase.quantize_int(x, BITS, INTEGER_BITS)
    logits, audit = phase.qat_integer_predict(model, x)
    # qat_integer_predict returns real logits whose integer scale is 2*DATA_FRAC.
    logits_product_raw = np.rint(logits * float(1 << (2 * DATA_FRAC))).astype(np.int64)
    logits_acc_raw = logits_product_raw << (ACC_FRAC - 2 * DATA_FRAC)
    action = np.argmax(logits_acc_raw, axis=1).astype(np.int64)
    columns = {"row_id": np.arange(len(xq), dtype=np.int64)}
    columns.update({f"x{i}_q": xq[:, i] for i in range(16)})
    columns.update({f"logit{o}_q": logits_acc_raw[:, o] for o in range(3)})
    columns["action"] = action
    output = RESULTS / "qat12_test_vectors.csv"
    pd.DataFrame(columns).to_csv(output, index=False)
    print(f"[OK] vectors={len(xq)} input_saturation_count={input_saturation}")
    print(f"[OK] audit={audit}")
    print(f"[OK] weights={HLS_SRC / 'qat12_weights.hpp'} vectors={output}")


if __name__ == "__main__":
    main()
