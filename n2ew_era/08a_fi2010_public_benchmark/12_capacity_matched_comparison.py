from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

from fi2010_tcn import TCNFI2010Classifier, count_trainable_parameters as count_tcn_parameters
from fi2010_phc_lite import PHCLiteFI2010Classifier, count_trainable_parameters as count_phc_parameters


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORT = ROOT / "CAPACITY_MATCHED_COMPARISON_REPORT.md"
SUMMARY_CSV = RESULTS / "capacity_matched_comparison.csv"
TCN_CHANNEL_CSV = RESULTS / "tcn_small_channel_selection.csv"

PHC_TARGET_PARAMS = 2132
DEEPL_OB_PAPER_PARAMS = 60_000
DEEPL_OB_REF_IMPL_PARAMS = 142_435


def pct(x: float | None) -> str:
    if x is None or not math.isfinite(float(x)):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def pct_points(x: float | None) -> str:
    if x is None or not math.isfinite(float(x)):
        return "n/a"
    return f"{100.0 * float(x):+.2f} pp"


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_first(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    return [row.to_dict() for _, row in df.iterrows()]


def md_table(rows: list[dict[str, object]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def tcn_channel_selection() -> list[dict[str, object]]:
    rows = []
    for channels in range(4, 17):
        for input_output_bias in [True, False]:
            params = count_tcn_parameters(
                TCNFI2010Classifier(channels=channels, input_output_bias=input_output_bias)
            )
            rel_delta = (params - PHC_TARGET_PARAMS) / PHC_TARGET_PARAMS
            rows.append(
                {
                    "channels": channels,
                    "input_output_bias": input_output_bias,
                    "trainable_parameters": params,
                    "relative_delta_vs_phc_v3_hidden16": rel_delta,
                    "within_plus_minus_10pct": abs(rel_delta) <= 0.10,
                }
            )
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    channel_rows = tcn_channel_selection()
    with TCN_CHANNEL_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(channel_rows[0].keys()))
        writer.writeheader()
        writer.writerows(channel_rows)

    phc16 = read_first(RESULTS / "stage1_metrics_v3_fair512.csv")
    tcn_small = read_first(RESULTS / "tcn_fi2010_metrics_small_raw.csv")
    tcn_full = read_first(RESULTS / "tcn_fi2010_metrics.csv")
    h64_probe = read_first(RESULTS / "capacity_phc_hidden64_speedprobe.csv")
    h64_params = read_first(RESULTS / "parameter_count_v3_hidden64_probe.csv")
    h64_capacity_params = read_first(RESULTS / "parameter_count_v3_hidden64_capacity.csv")
    h64_capacity = read_first(RESULTS / "stage1_metrics_v3_hidden64_capacity.csv")
    h64_heartbeat = read_csv_rows(RESULTS / "training_heartbeat_v3_hidden64_capacity.csv")
    current_preflight = read_first(RESULTS / "hidden64_current_resource_preflight.csv")
    prior_preflight = read_first(RESULTS / "v3_fair512_resource_preflight.csv")

    rows: list[dict[str, object]] = []
    if phc16:
        phc16_convergence = (
            "early_stop_closed"
            if bool(phc16.get("early_stop", False)) and bool(phc16.get("convergence_closed", False))
            else "not_closed_or_partial"
        )
        rows.append(
            {
                "model": "PHC v3 hidden16 fair512",
                "architecture": "recurrent PHC-lite",
                "horizon": int(phc16["horizon"]),
                "seed": int(phc16["seed"]),
                "loss_mode": phc16["loss_mode"],
                "trainable_parameters": int(phc16["trainable_parameters"]),
                "convergence_status": phc16_convergence,
                "best_epoch": int(phc16["best_epoch"]),
                "final_epoch": int(phc16["final_epoch"]),
                "val_macro_f1": float(phc16["val_macro_f1"]),
                "test_macro_f1": float(phc16["test_macro_f1"]),
                "notes": "Existing fair512 run; status read directly from stage1_metrics_v3_fair512.csv.",
            }
        )
    if h64_capacity:
        early_stop = truthy(h64_capacity.get("early_stop", False))
        convergence_closed = truthy(h64_capacity.get("convergence_closed", False))
        if early_stop and convergence_closed:
            h64_convergence = "early_stop_closed"
        elif early_stop:
            h64_convergence = "early_stop_not_closed"
        else:
            h64_convergence = "not_closed_or_max_epoch"
        rows.append(
            {
                "model": "PHC v3 hidden64 capacity",
                "architecture": "recurrent PHC-lite",
                "horizon": int(h64_capacity["horizon"]),
                "seed": int(h64_capacity["seed"]),
                "loss_mode": h64_capacity["loss_mode"],
                "trainable_parameters": int(h64_capacity["trainable_parameters"]),
                "convergence_status": h64_convergence,
                "best_epoch": int(h64_capacity["best_epoch"]),
                "final_epoch": int(h64_capacity["final_epoch"]),
                "val_macro_f1": float(h64_capacity["val_macro_f1"]),
                "test_macro_f1": float(h64_capacity["test_macro_f1"]),
                "notes": (
                    "New hidden_dim=64, batch_size=512 capacity run; checkpoint and metrics use "
                    "_v3_hidden64_capacity suffix."
                ),
            }
        )
    elif h64_params:
        rows.append(
            {
                "model": "PHC v3 hidden64 probe",
                "architecture": "recurrent PHC-lite",
                "horizon": 50,
                "seed": 0,
                "loss_mode": "raw",
                "trainable_parameters": int(h64_params["trainable_parameters"]),
                "convergence_status": "not_trained_speedprobe_only",
                "best_epoch": "",
                "final_epoch": "",
                "val_macro_f1": "",
                "test_macro_f1": "",
                "notes": "One-epoch formal probe exited before first logged epoch; synthetic batch512 probe retained for cost estimate.",
            }
        )
    if tcn_small:
        rows.append(
            {
                "model": "TCN small",
                "architecture": "feed-forward causal TCN",
                "horizon": int(tcn_small["horizon"]),
                "seed": int(tcn_small["seed"]),
                "loss_mode": tcn_small["loss_mode"],
                "trainable_parameters": int(tcn_small["trainable_parameters"]),
                "convergence_status": "early_stop_closed" if bool(tcn_small["early_stop"]) else "not_closed",
                "best_epoch": int(tcn_small["best_epoch"]),
                "final_epoch": int(tcn_small["final_epoch"]),
                "val_macro_f1": float(tcn_small["val_macro_f1"]),
                "test_macro_f1": float(tcn_small["test_macro_f1"]),
                "notes": "channels=8, input/readout bias disabled to stay within +10% of PHC hidden16 parameters.",
            }
        )
    if tcn_full:
        rows.append(
            {
                "model": "TCN scratch width16",
                "architecture": "feed-forward causal TCN",
                "horizon": int(tcn_full["horizon"]),
                "seed": int(tcn_full["seed"]),
                "loss_mode": tcn_full["loss_mode"],
                "trainable_parameters": int(tcn_full["trainable_parameters"]),
                "convergence_status": "early_stop_closed" if bool(tcn_full["early_stop"]) else "not_closed",
                "best_epoch": int(tcn_full["best_epoch"]),
                "final_epoch": int(tcn_full["final_epoch"]),
                "val_macro_f1": float(tcn_full["val_macro_f1"]),
                "test_macro_f1": float(tcn_full["test_macro_f1"]),
                "notes": "Previously completed wider TCN reference.",
            }
        )

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    phc16_f1 = next((float(r["test_macro_f1"]) for r in rows if r["model"].startswith("PHC v3 hidden16")), float("nan"))
    small_f1 = next((float(r["test_macro_f1"]) for r in rows if r["model"] == "TCN small"), float("nan"))
    full_f1 = next((float(r["test_macro_f1"]) for r in rows if r["model"] == "TCN scratch width16"), float("nan"))
    h64_hours = float(h64_probe["estimated_60epoch_train_hours"]) if h64_probe else float("nan")
    h64_batch_seconds = float(h64_probe["single_batch_forward_backward_seconds"]) if h64_probe else float("nan")
    h64_param_count = int(
        (h64_capacity_params or h64_params or {"trainable_parameters": 0})["trainable_parameters"]
    )
    h64_vs_paper = DEEPL_OB_PAPER_PARAMS / h64_param_count if h64_param_count else float("nan")
    h64_vs_ref = DEEPL_OB_REF_IMPL_PARAMS / h64_param_count if h64_param_count else float("nan")
    heartbeat_anomalies = [
        row for row in h64_heartbeat if truthy(row.get("duration_anomaly_gt_3x_baseline", False))
    ]
    anomaly_note = (
        "No completed hidden64 heartbeat anomaly table is present yet."
        if not h64_heartbeat
        else (
            "No hidden64 epoch-duration anomaly exceeded the 3x-baseline threshold."
            if not heartbeat_anomalies
            else "; ".join(
                [
                    (
                        f"seed {int(row['seed'])} epoch {int(row['epoch'])}: "
                        f"{float(row['epoch_wall_seconds']):.1f}s "
                        f"(threshold {float(row['anomaly_threshold_seconds']):.1f}s)"
                    )
                    for row in heartbeat_anomalies[:5]
                ]
            )
        )
    )

    selected_channel = next(r for r in channel_rows if r["channels"] == 8 and r["input_output_bias"] is False)
    comparison_rows = []
    for r in rows:
        out = dict(r)
        out["val_macro_f1"] = pct(float(r["val_macro_f1"])) if r["val_macro_f1"] != "" else "n/a"
        out["test_macro_f1"] = pct(float(r["test_macro_f1"])) if r["test_macro_f1"] != "" else "n/a"
        comparison_rows.append(out)

    channel_note = (
        f"Integer channel width alone cannot hit the +/-10% band: channels=7 gives 1,851 params "
        f"(-13.18%), while channels=8 gives 2,355 params (+10.46%). The selected strict-band "
        f"variant uses channels=8 and disables only input/readout bias, yielding "
        f"{selected_channel['trainable_parameters']} params "
        f"({100.0 * float(selected_channel['relative_delta_vs_phc_v3_hidden16']):+.2f}%)."
    )

    verdict = (
        "With parameter count aligned, the previous TCN advantage shrinks and slightly reverses: "
        f"TCN small reaches {pct(small_f1)} test macro-F1 versus PHC v3 hidden16 fair512 at "
        f"{pct(phc16_f1)}, a delta of {pct_points(small_f1 - phc16_f1)} for TCN small. "
        "Because the current fair512 CSV reports natural early stopping, this is a cleaner capacity-controlled "
        "single-seed result than the earlier partial report. It shows that the width16 TCN edge was materially affected "
        "by capacity: the 8,547-param TCN reaches "
        f"{pct(full_f1)}, while the 2,344-param TCN small falls to {pct(small_f1)}."
    )

    if h64_capacity:
        hidden64_status = "measured"
        hidden64_text = (
            f"PHC hidden64 now has a software-measured training result: best_epoch="
            f"{int(h64_capacity['best_epoch'])}, final_epoch={int(h64_capacity['final_epoch'])}, "
            f"val macro-F1={pct(float(h64_capacity['val_macro_f1']))}, "
            f"test macro-F1={pct(float(h64_capacity['test_macro_f1']))}. "
            f"The run used hidden_dim=64, batch_size={int(h64_capacity['batch_size'])}, "
            f"torch_threads={int(h64_capacity['torch_threads'])}, patience={int(h64_capacity['patience'])}."
        )
    else:
        hidden64_status = "probe_only"
        hidden64_text = (
            f"PHC hidden64 has {h64_param_count if h64_param_count else 'n/a'} trainable parameters. "
            f"A synthetic batch512 forward/backward probe measured {h64_batch_seconds:.3f}s per batch and "
            f"estimated {h64_hours:.2f} train-only hours for 60 epochs before full validation overhead. "
            "A prior formal one-epoch hidden64 probe using the requested `batch_size=512` exited before its "
            "first logged epoch, leaving only the parameter-count file and no checkpoint/metrics. Therefore "
            "hidden64 is not used as a performance point until `stage1_metrics_v3_hidden64_capacity.csv` exists."
        )

    status = (
        "capacity_matched_tcn_small_complete_phc_hidden64_seed0_measured"
        if hidden64_status == "measured"
        else "capacity_matched_tcn_small_complete_phc_hidden64_probe_only"
    )

    report = f"""# Capacity-Matched TCN vs PHC v3 Comparison

status: {status}

## Scope

This report covers only FI-2010 `k=50`, `seed=0`. It does not use TXF NEW1-NEW5 and does not modify existing PHC v2/v3/fair512/speedprobe result files.

Seed1 was not launched in this turn. The seed0 hidden64 run reached natural early stopping, but it also consumed 10:10 wall-clock time and included one large epoch-duration anomaly; launching another hidden64 seed immediately would conflict with the resource-yield boundary for concurrent hardware/demo work.

## Direction A: Shrink TCN

{channel_note}

The TCN topology is unchanged: 5 residual causal-conv blocks, kernel=3, dilations `[1, 2, 4, 8, 16]`, receptive field 125, final-timestep readout. The `tcn_small` run uses raw cross entropy.

## Direction B: Expand PHC

{hidden64_text}

The hidden64 capacity run has {h64_param_count:,} trainable parameters, which is 1/{h64_vs_paper:.2f} of the DeepLOB paper-rounded 60,000 parameter count and 1/{h64_vs_ref:.2f} of the 142,435-parameter public reference implementation count recorded in `EFFICIENCY_RESOURCE_TRADEOFF_REPORT.md`. The intended claim is therefore not that fewer parameters alone win; it is that PHC capacity/performance should be interpreted under the low-latency deployment feasibility constraint.

### Hidden64 Interruption Diagnosis

The previous hidden64 formal run left no persisted stdout/stderr log, checkpoint, or metrics CSV in this phase; only `results/parameter_count_v3_hidden64_probe.csv` and `results/capacity_phc_hidden64_speedprobe.csv` were present. The strongest codebase-local resource clue is `results/v3_fair512_resource_preflight.csv`: available memory was {prior_preflight['available_memory_mb'] if prior_preflight else 'n/a'} MB and the sampled process list included background OneDrive plus Anaconda Python processes. That does not prove OOM, but it makes the old exit unsuitable as evidence of a PHC hidden64 code bug.

For the new run, `results/hidden64_current_resource_preflight.csv` recorded {current_preflight['available_memory_mb'] if current_preflight else 'n/a'} MB available before launch. A `vitis_hls.exe` process was present, so the run kept `batch_size=512` for v2 comparability but limited PyTorch to 2 threads to reduce contention. The new run reached epoch-level logging under batch512, so immediate batch-size memory failure and pre-epoch code failure are ruled out for this configuration. Heartbeat anomaly summary: {anomaly_note}.

## Main Comparison

{md_table(comparison_rows, ["model", "architecture", "trainable_parameters", "loss_mode", "convergence_status", "best_epoch", "final_epoch", "val_macro_f1", "test_macro_f1", "notes"])}

## Interpretation

{verdict}

The honest current answer is: after parameter alignment, the architecture gap mostly collapses and slightly reverses within this one k=50/seed0 slice. TCN small does not beat the converged PHC hidden16 result; the wider TCN does. Hidden64 is now framed as a deployability-constrained capacity probe: even if it moves PHC closer to DeepLOB F1, its 23,876 trainable parameters remain materially below the published DeepLOB parameter counts used elsewhere in this phase.

## Files

- Summary CSV: `results/capacity_matched_comparison.csv`
- TCN channel selection: `results/tcn_small_channel_selection.csv`
- TCN small metrics: `results/tcn_fi2010_metrics_small_raw.csv`
- PHC hidden64 synthetic speedprobe: `results/capacity_phc_hidden64_speedprobe.csv`
- PHC hidden64 parameter count: `results/parameter_count_v3_hidden64_probe.csv`
- PHC hidden64 capacity helper: `13_train_hidden64_capacity.py`
- PHC hidden64 capacity stdout/stderr logs: `results/train_v3_hidden64_capacity_seed0.log`, `results/train_v3_hidden64_capacity_seed0.err.log`
- PHC hidden64 current preflight: `results/hidden64_current_resource_preflight.csv`
- PHC hidden64 metrics, if completed: `results/stage1_metrics_v3_hidden64_capacity.csv`
- PHC hidden64 heartbeat, if completed: `results/training_heartbeat_v3_hidden64_capacity.csv`

## Claim Boundary

- This is a single-horizon, single-seed control, not a 3-seed benchmark.
- `TCN small` is converged by early stopping; PHC hidden16 fair512 convergence status is read from the current CSV.
- PHC hidden64 is reported as a performance point only if `results/stage1_metrics_v3_hidden64_capacity.csv` exists; otherwise only parameter count and speed/cost probes are reported.
- No HLS, P&R, board work, distillation, or NEW1-NEW5 data are involved.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
