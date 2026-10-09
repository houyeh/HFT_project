from __future__ import annotations

import argparse
import csv
from pathlib import Path


# --- 判準（只看 epoch 1 的 train_loss；epoch 2/3 的發散是正常的，不可當判準）---
GATE_METRIC = "train_loss"
GATE_EPOCH = 1
PASS_REL = 1e-3          # 前四位有效數字吻合 -> 移植正確
INVESTIGATE_REL = 1e-1   # 介於兩者之間 -> 人工判斷
# -----------------------------------------------------------------------

METRICS = [
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
]


def _emit_verdict(output_path: Path, verdict: str, cpu_v, gpu_v, rel, note: str) -> None:
    """一律 exit 0：此腳本在 sbatch 中位於 SU 會計之前，非零離開會讓 set -e 砍掉作業。"""
    line = "=" * 68
    print(line, flush=True)
    print(f"  PORT CHECK VERDICT: {verdict}", flush=True)
    if cpu_v is not None:
        print(f"  epoch {GATE_EPOCH} {GATE_METRIC}:  CPU={cpu_v!r}  GPU={gpu_v!r}", flush=True)
        print(f"  relative |delta| = {rel:.3e}   (PASS < {PASS_REL:g};"
              f"  INVESTIGATE < {INVESTIGATE_REL:g})", flush=True)
    print(f"  {note}", flush=True)
    print(line, flush=True)
    vp = output_path.parent / "port_check_verdict.txt"
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(
        f"verdict={verdict}\ngate_metric=epoch{GATE_EPOCH}:{GATE_METRIC}\n"
        f"cpu_value={cpu_v}\ngpu_value={gpu_v}\nrelative_abs_delta={rel}\nnote={note}\n",
        encoding="utf-8",
    )


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["epoch"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPU probe epochs with the CPU fair512 reference.")
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gpu = read_rows(args.gpu)
    cpu = read_rows(args.cpu)
    common = sorted(set(gpu) & set(cpu))
    if not common:
        _emit_verdict(args.output, "INCOMPLETE", None, None, None,
                      f"No overlapping epochs. gpu={sorted(gpu)}, cpu={sorted(cpu)}")
        return
    if common != [1, 2, 3]:
        print(f"[WARN] probe did not produce the full 3 epochs; comparing {common}", flush=True)

    rows: list[dict[str, object]] = []
    for epoch in common:
        for metric in METRICS:
            cpu_value = float(cpu[epoch][metric])
            gpu_value = float(gpu[epoch][metric])
            abs_delta = abs(gpu_value - cpu_value)
            rows.append(
                {
                    "epoch": epoch,
                    "metric": metric,
                    "cpu_value": cpu_value,
                    "gpu_value": gpu_value,
                    "signed_delta_gpu_minus_cpu": gpu_value - cpu_value,
                    "abs_delta": abs_delta,
                    "relative_abs_delta": abs_delta / max(abs(cpu_value), 1e-15),
                    "gpu_epoch_wall_seconds": gpu[epoch].get("epoch_wall_seconds", ""),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if GATE_EPOCH in common:
        cpu_v = float(cpu[GATE_EPOCH][GATE_METRIC])
        gpu_v = float(gpu[GATE_EPOCH][GATE_METRIC])
        rel = abs(gpu_v - cpu_v) / max(abs(cpu_v), 1e-15)
        if rel < PASS_REL:
            _emit_verdict(args.output, "PASS", cpu_v, gpu_v, rel,
                          "移植正確，可核可 Step 1 九個 run 並行啟動。")
        elif rel < INVESTIGATE_REL:
            _emit_verdict(args.output, "INVESTIGATE", cpu_v, gpu_v, rel,
                          "差異大於預期但未到結構性等級。先人工檢查資料順序與 batch 組成，不要直接開 Step 1。")
        else:
            _emit_verdict(args.output, "FAIL", cpu_v, gpu_v, rel,
                          "結構性差異（資料順序／初始化／LR／batch）。停止，不要開 Step 1。")
    else:
        _emit_verdict(args.output, "INCOMPLETE", None, None, None,
                      f"缺少 epoch {GATE_EPOCH}，無法判定。probe 可能被 watchdog 截斷。")


if __name__ == "__main__":
    main()
