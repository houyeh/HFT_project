from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

VARIANTS = {
    "scratch": ROOT / "hls_component_stage2_scratch",
    "distilled": ROOT / "hls_component_stage2_distilled",
}

RESOURCE_BUDGET = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}

BLOCK_LENGTHS = {
    "block0": 248,
    "block1": 240,
    "block2": 224,
    "block3": 192,
    "block4": 128,
}

BLOCK_START_EVENTS = {
    "block0": 4,
    "block1": 12,
    "block2": 28,
    "block3": 60,
    "block4": 124,
}

NAIVE_FULL_TCN_US = 13_703.85
TCN_MAC_ESTIMATE_US = 32.656
PHC_LITE_ROUTED_US = 83.16


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.8g}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def find_first(base: Path, pattern: str) -> Path | None:
    matches = sorted(base.rglob(pattern))
    return matches[0] if matches else None


def parse_csim_state(component: Path) -> str:
    summary = component / "hls_component" / "hls_component.hlsrun_csim_summary"
    if not summary.exists():
        return "missing"
    text = summary.read_text(errors="ignore")
    if "CS_PASSED" in text:
        return "CS_PASSED"
    if "CS_FAILED" in text:
        return "CS_FAILED"
    if "CS_RUNNING" in text:
        return "CS_RUNNING_OR_INCOMPLETE"
    return "present_unknown"


def load_block0_baseline_ratios() -> dict[str, float]:
    path = RESULTS / "stage2_preflight_block0_error_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(row["variant"]): float(row["last_over_first_quarter_rmse"]) for _, row in df.iterrows()}


def parse_block_csim(variant: str, baseline_ratio: float | None) -> list[dict[str, Any]]:
    path = RESULTS / f"stage2_streaming_{variant}_block_csim_out.csv"
    if not path.exists():
        return [
            {
                "variant": variant,
                "stage": "all_blocks",
                "csim_block_output": str(path),
                "block_csim_status": "missing",
            }
        ]
    df = pd.read_csv(path)
    diff_cols = [c for c in df.columns if c.startswith("abs_diff_c")]
    rows: list[dict[str, Any]] = []
    for stage, g_stage in df.groupby("stage", sort=True):
        vals = g_stage[diff_cols].to_numpy(dtype=np.float64)
        by_t_rows: list[dict[str, Any]] = []
        for out_t, g_t in g_stage.groupby("out_t", sort=True):
            t_vals = g_t[diff_cols].to_numpy(dtype=np.float64)
            by_t_rows.append(
                {
                    "variant": variant,
                    "stage": stage,
                    "out_t": int(out_t),
                    "rmse": float(math.sqrt(float((t_vals * t_vals).mean()))),
                    "max_abs_diff": float(t_vals.max()),
                }
            )
        by_t = pd.DataFrame(by_t_rows)
        length = int(BLOCK_LENGTHS.get(str(stage), by_t["out_t"].nunique()))
        first_q = by_t[by_t["out_t"] < max(1, length // 4)]
        last_q = by_t[by_t["out_t"] >= (3 * length) // 4]
        post10 = by_t[by_t["out_t"] >= 10]
        real_data_start = max(0, 124 - int(BLOCK_START_EVENTS.get(str(stage), 124)))
        post_real = by_t[by_t["out_t"] >= real_data_start + 10]
        if post_real.empty:
            post_real = post10
        slope = float(np.polyfit(post10["out_t"].to_numpy(dtype=np.float64), post10["rmse"].to_numpy(dtype=np.float64), 1)[0]) if len(post10) > 1 else float("nan")
        post_real_slope = float(np.polyfit(post_real["out_t"].to_numpy(dtype=np.float64), post_real["rmse"].to_numpy(dtype=np.float64), 1)[0]) if len(post_real) > 1 else float("nan")
        ratio = float(last_q["rmse"].mean() / first_q["rmse"].mean()) if not first_q.empty and float(first_q["rmse"].mean()) > 0 else float("nan")
        baseline_gate = float(baseline_ratio * 2.0) if baseline_ratio and math.isfinite(baseline_ratio) else float("nan")
        ratio_warning = bool(math.isfinite(baseline_gate) and ratio > baseline_gate)
        abs_gate_pass = bool(float(vals.max()) <= 1.0 and math.sqrt(float((vals * vals).mean())) <= 0.305)
        post_real_gate_pass = bool(float(post_real["rmse"].max()) <= 0.1 and (not math.isfinite(post_real_slope) or abs(post_real_slope) <= 1.0e-4))
        rows.append(
            {
                "variant": variant,
                "stage": stage,
                "block_csim_status": "present",
                "csim_block_output": str(path),
                "rows": int(len(g_stage)),
                "timesteps": int(by_t["out_t"].nunique()),
                "compared_values": int(vals.size),
                "max_abs_diff": float(vals.max()),
                "mean_abs_diff": float(vals.mean()),
                "rmse": float(math.sqrt(float((vals * vals).mean()))),
                "first_quarter_mean_rmse": float(first_q["rmse"].mean()),
                "last_quarter_mean_rmse": float(last_q["rmse"].mean()),
                "last_over_first_quarter_rmse": ratio,
                "post10_rmse_max": float(post10["rmse"].max()) if not post10.empty else float("nan"),
                "post10_rmse_slope_per_timestep": slope,
                "real_data_start_out_t": real_data_start,
                "post_real10_rmse_max": float(post_real["rmse"].max()) if not post_real.empty else float("nan"),
                "post_real10_rmse_median": float(post_real["rmse"].median()) if not post_real.empty else float("nan"),
                "post_real10_rmse_slope_per_timestep": post_real_slope,
                "block0_preflight_ratio": baseline_ratio,
                "ratio_warning": ratio_warning,
                "ratio_warning_basis": f">2x block0 preflight ratio ({baseline_gate:.6g})" if math.isfinite(baseline_gate) else "baseline missing",
                "stage2_block_gate_pass": bool(abs_gate_pass and post_real_gate_pass),
            }
        )
    return rows


def parse_readout_csim(variant: str) -> dict[str, Any]:
    path = RESULTS / f"stage2_streaming_{variant}_readout_csim_out.csv"
    if not path.exists():
        return {"variant": variant, "readout_csim_status": "missing", "readout_csim_output": str(path)}
    df = pd.read_csv(path)
    diff_cols = [c for c in df.columns if c.startswith("abs_diff_readout_")]
    vals = df[diff_cols].to_numpy(dtype=np.float64)
    return {
        "variant": variant,
        "readout_csim_status": "present",
        "readout_csim_output": str(path),
        "readout_rows": int(len(df)),
        "readout_max_abs_diff": float(vals.max()),
        "readout_mean_abs_diff": float(vals.mean()),
        "readout_rmse": float(math.sqrt(float((vals * vals).mean()))),
        "sign_agreement_readout_0": float(pd.to_numeric(df["sign_agree_0"]).mean()),
        "sign_agreement_readout_1": float(pd.to_numeric(df["sign_agree_1"]).mean()),
        "sign_agreement_joint": float(pd.to_numeric(df["sign_agree_joint"]).mean()),
        "readout_gate_pass": bool(float(vals.max()) <= 1.0 and float(pd.to_numeric(df["sign_agree_joint"]).mean()) == 1.0),
    }


def parse_csynth(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"csynth_status": "missing"}
    text = path.read_text(errors="ignore")
    out: dict[str, Any] = {"csynth_status": "present", "csynth_report": str(path)}
    latency_section_match = re.search(
        r"\+ Latency:\s*\* Summary:\s*(.*?)(?:\+ Detail:|== Utilization Estimates)",
        text,
        re.S,
    )
    if latency_section_match:
        for line in latency_section_match.group(1).splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 7 and cells[0].isdigit() and cells[1].isdigit():
                out["latency_min_cycles"] = int(cells[0])
                out["latency_max_cycles"] = int(cells[1])
                out["latency_absolute_min_reported"] = cells[2]
                out["latency_absolute_max_reported"] = cells[3]
                out["ii_min_cycles"] = int(cells[4])
                out["ii_max_cycles"] = int(cells[5])
                out["pipeline_type"] = cells[6]
                out["latency_us_100mhz"] = float(out["latency_max_cycles"]) * 0.01
                out["interval_us_100mhz"] = float(out["ii_max_cycles"]) * 0.01
                break
    util_section = text.split("== Utilization Estimates", 1)
    if len(util_section) > 1:
        for line in util_section[1].splitlines():
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 5 and cells[0] == "Total":
                names = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
                for name, value in zip(names, cells[1:6]):
                    try:
                        out[name] = int(value)
                    except ValueError:
                        pass
                break
    for key, budget in RESOURCE_BUDGET.items():
        if key in out:
            out[f"{key}_pct"] = float(out[key]) / budget * 100.0
    if all(key in out for key in RESOURCE_BUDGET):
        out["resource_fit_zynq7020"] = bool(all(float(out[key]) <= budget for key, budget in RESOURCE_BUDGET.items()))
    return out


def parse_csynth_failure(variant: str, component: Path) -> dict[str, Any]:
    logs = [
        component / "hls_component" / "logs" / "hls_compile.log",
        RESULTS / f"stage2_{variant}_vpp_csynth_stdout.log",
        RESULTS / f"stage2_{variant}_vpp_csynth_stderr.log",
    ]
    errors: list[str] = []
    for log_path in logs:
        if not log_path.exists():
            continue
        for line in log_path.read_text(errors="ignore").splitlines():
            if "ERROR:" in line:
                errors.append(f"{log_path}: {line.strip()}")
    partial_reports = list((component / "hls_component" / "hls" / "syn" / "report").glob("*_csynth.rpt")) if (component / "hls_component" / "hls" / "syn" / "report").exists() else []
    if errors:
        return {
            "csynth_status": "failed_before_top_report",
            "csynth_failure_error": " | ".join(errors[-4:]),
            "csynth_partial_reports_count": len(partial_reports),
            "csynth_failure_log": str(component / "hls_component" / "logs" / "hls_compile.log"),
        }
    return {}


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = load_block0_baseline_ratios()
    block_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    synth_rows: list[dict[str, Any]] = []
    for variant, component in VARIANTS.items():
        block_rows.extend(parse_block_csim(variant, baseline.get(variant)))
        readout = parse_readout_csim(variant)
        readout["component_csim_state"] = parse_csim_state(component)
        readout_rows.append(readout)
        synth_row = {"variant": variant}
        synth_row.update(parse_csynth(find_first(component, "tcn_streaming_step_csynth.rpt")))
        if synth_row.get("csynth_status") == "missing":
            synth_row.update(parse_csynth_failure(variant, component))
        synth_rows.append(synth_row)
    return pd.DataFrame(block_rows), pd.DataFrame(readout_rows), pd.DataFrame(synth_rows)


def write_report(blocks: pd.DataFrame, readout: pd.DataFrame, synth: pd.DataFrame) -> None:
    comparison = pd.DataFrame(
        [
            {
                "artifact": "07b naive full TCN",
                "latency": "13,703.85 us",
                "verification": "HLS csynth estimate",
                "note": "Full-window recomputation baseline.",
            },
            {
                "artifact": "07a TCN analytical estimate",
                "latency": f"{TCN_MAC_ESTIMATE_US:.3f} us",
                "verification": "pre-synthesis MAC estimate, not HLS-verified",
                "note": "Analytical MAC-rate estimate only.",
            },
            {
                "artifact": "PHC-lite v8",
                "latency": f"{PHC_LITE_ROUTED_US:.2f} us",
                "verification": "routed P&R at 100MHz",
                "note": "Existing PHC baseline.",
            },
        ]
    )
    csim_ok = bool(readout.get("component_csim_state", pd.Series(dtype=str)).eq("CS_PASSED").all()) if not readout.empty else False
    any_failed_synth = bool(synth["csynth_status"].astype(str).str.contains("failed", na=False).any()) if "csynth_status" in synth.columns else False
    all_synth_present = bool(synth["csynth_status"].eq("present").all()) if "csynth_status" in synth.columns else False
    if all_synth_present:
        status = "stage2_complete"
    elif csim_ok and any_failed_synth:
        status = "stage2_mixed_csim_passed_csynth_partial_failure"
    else:
        status = "stage2_partial"
    lines = [
        f"status: {status}",
        "finding: five_block_streaming_csim_passed_csynth_collected" if status == "stage2_complete" else "finding: five_block_streaming_csim_passed_but_csynth_not_fully_available",
        "# Stage 2 Full Streaming HLS Report",
        "",
        "## Scope",
        "",
        "This report covers the new 07c Stage 2 streaming HLS implementation only. It does not modify 07a, 07b, Stage 0 outputs, or Stage 1 outputs. It does not use NEW5. It stops at CSim/csynth; no cosim, implementation, P&R, bitstream, or board result is claimed.",
        "",
        "Stage 2 uses `TCN_STAGE2_CHANNEL_UNROLL=4` to keep the five-block implementation resource-feasible; this differs from the Stage 1 single-block diagnostic that used the inherited unroll-16 default.",
        "",
        "## Preflight",
        "",
        f"- Preflight report: `{ROOT / 'STAGE2_PREFLIGHT_BLOCK0_ERROR_DIAGNOSTIC_REPORT.md'}`",
        f"- Preflight curve CSV: `{RESULTS / 'stage2_preflight_block0_timestep_rmse.csv'}`",
        "",
        "## Per-Block CSim Diagnostics",
        "",
        md_table(blocks),
        "",
        "## Final Readout CSim Diagnostics",
        "",
        md_table(readout),
        "",
        "## Csynth Summary",
        "",
        md_table(synth),
        "",
        "## Baseline Comparison",
        "",
        md_table(comparison),
        "",
        "## Stage Boundary",
        "",
        "This stage intentionally stops after CSim/csynth. No Stage 2 cosim, Vivado implementation, P&R, routed timing, or board-level claim is made here.",
    ]
    (ROOT / "STAGE2_STREAMING_FULL_HLS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    blocks, readout, synth = collect()
    blocks.to_csv(RESULTS / "stage2_streaming_block_csim_summary.csv", index=False)
    readout.to_csv(RESULTS / "stage2_streaming_readout_csim_summary.csv", index=False)
    synth.to_csv(RESULTS / "stage2_streaming_csynth_summary.csv", index=False)
    with (RESULTS / "stage2_streaming_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "blocks": blocks.to_dict(orient="records"),
                "readout": readout.to_dict(orient="records"),
                "csynth": synth.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
    write_report(blocks, readout, synth)
    print(f"wrote {RESULTS / 'stage2_streaming_block_csim_summary.csv'}")
    print(f"wrote {RESULTS / 'stage2_streaming_readout_csim_summary.csv'}")
    print(f"wrote {RESULTS / 'stage2_streaming_csynth_summary.csv'}")
    print(f"wrote {ROOT / 'STAGE2_STREAMING_FULL_HLS_REPORT.md'}")


if __name__ == "__main__":
    main()
