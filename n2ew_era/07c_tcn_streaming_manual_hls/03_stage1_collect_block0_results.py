from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

VARIANTS = {
    "scratch": ROOT / "hls_component_block0_scratch",
    "distilled": ROOT / "hls_component_block0_distilled",
}

RESOURCE_BUDGET = {
    "DSP": 220,
    "LUT": 53200,
    "FF": 106400,
    "BRAM_18K": 280,
}
NAIVE_BLOCK0_LATENCY_CYCLES = 319_689
NAIVE_BLOCK0_OUTPUT_TIMESTEPS = 248
NAIVE_BLOCK0_AMORTIZED_CYCLES = NAIVE_BLOCK0_LATENCY_CYCLES / NAIVE_BLOCK0_OUTPUT_TIMESTEPS
PHC_LITE_ROUTED_US = 83.16
NAIVE_FULL_TCN_US = 13_703.85
TCN_MAC_ESTIMATE_US = 32.656


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


def parse_csim(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"csim_status": "missing", "csim_output": str(path)}
    df = pd.read_csv(path)
    diff_cols = [c for c in df.columns if c.startswith("abs_diff_c")]
    values = df[diff_cols].to_numpy(dtype=float)
    by_t = df.groupby("out_t", as_index=False).agg(
        rmse_timestep_mean=("rmse_timestep", "mean"),
        max_abs_diff_timestep_max=("max_abs_diff", "max"),
    )
    first_q = df[df["out_t"] < 62]
    last_q = df[df["out_t"] >= 186]
    first_rmse = float(math.sqrt((first_q[diff_cols].to_numpy(dtype=float) ** 2).mean())) if not first_q.empty else float("nan")
    last_rmse = float(math.sqrt((last_q[diff_cols].to_numpy(dtype=float) ** 2).mean())) if not last_q.empty else float("nan")
    return {
        "csim_status": "present",
        "csim_output": str(path),
        "csim_rows": int(len(df)),
        "samples": int(df["sample_idx"].nunique()) if "sample_idx" in df.columns else 0,
        "valid_timesteps_per_sample": int(df["out_t"].nunique()) if "out_t" in df.columns else 0,
        "compared_values": int(values.size),
        "max_abs_diff": float(values.max()) if values.size else float("nan"),
        "mean_abs_diff": float(abs(values).mean()) if values.size else float("nan"),
        "rmse": float(math.sqrt(float((values * values).mean()))) if values.size else float("nan"),
        "valid_alignment_all_true": bool(df["valid"].eq(1).all()) if "valid" in df.columns else False,
        "first_quarter_rmse": first_rmse,
        "last_quarter_rmse": last_rmse,
        "last_minus_first_quarter_rmse": float(last_rmse - first_rmse) if math.isfinite(first_rmse) and math.isfinite(last_rmse) else float("nan"),
        "last_over_first_quarter_rmse": float(last_rmse / first_rmse) if first_rmse > 0 else float("nan"),
        "max_timestep_mean_rmse": float(by_t["rmse_timestep_mean"].max()) if not by_t.empty else float("nan"),
        "max_timestep_max_abs_diff": float(by_t["max_abs_diff_timestep_max"].max()) if not by_t.empty else float("nan"),
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
    if "latency_max_cycles" in out:
        latency = float(out["latency_max_cycles"])
        out["speedup_vs_naive_block0_total"] = NAIVE_BLOCK0_LATENCY_CYCLES / latency
        out["speedup_vs_naive_block0_amortized"] = NAIVE_BLOCK0_AMORTIZED_CYCLES / latency
    if "ii_max_cycles" in out:
        interval = float(out["ii_max_cycles"])
        out["throughput_speedup_vs_naive_block0_total"] = NAIVE_BLOCK0_LATENCY_CYCLES / interval
        out["throughput_speedup_vs_naive_block0_amortized"] = NAIVE_BLOCK0_AMORTIZED_CYCLES / interval
    return out


def collect() -> pd.DataFrame:
    rows = []
    for variant, component in VARIANTS.items():
        row: dict[str, Any] = {"variant": variant}
        row.update(parse_csim(RESULTS / f"stage1_block0_{variant}_csim_out.csv"))
        row.update(parse_csynth(find_first(component, "tcn_block0_stream_step_csynth.rpt")))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame) -> None:
    comparison = pd.DataFrame(
        [
            {
                "artifact": "07b naive full TCN",
                "latency": "13,703.85 us",
                "verification": "HLS csynth estimate",
                "note": "Full-window recomputation baseline.",
            },
            {
                "artifact": "07b naive block0",
                "latency": f"{NAIVE_BLOCK0_LATENCY_CYCLES:,} cycles total; {NAIVE_BLOCK0_AMORTIZED_CYCLES:.2f} cycles/output timestep amortized",
                "verification": "HLS csynth instance estimate",
                "note": "Report row grp_p_anonymous_namespace_tcn_block_252_1_s_fu_442.",
            },
            {
                "artifact": "07a TCN analytical estimate",
                "latency": f"{TCN_MAC_ESTIMATE_US:.3f} us",
                "verification": "pre-synthesis MAC estimate, not HLS-verified",
                "note": "Not a measured FPGA number.",
            },
            {
                "artifact": "PHC-lite v8",
                "latency": f"{PHC_LITE_ROUTED_US:.2f} us",
                "verification": "routed P&R at 100MHz",
                "note": "Existing PHC baseline.",
            },
        ]
    )
    lines = [
        "status: stage1_block0_complete_stop_for_review" if summary["csynth_status"].eq("present").all() else "status: stage1_block0_partial",
        "finding: block0_streaming_hls_measured_against_stage0_reference",
        "# Stage 1 Block0 Streaming HLS Report",
        "",
        "## Scope",
        "",
        "This stage implements only block0 (`dilation=1`) as a streaming HLS step function. It does not modify 07a, 07b, or Stage 0 reference outputs. It does not use NEW5. It does not implement blocks 1-4, cosim, implementation, P&R, bitstream, or board tests.",
        "",
        "CSim compares every valid block0 timestep and all 16 channels against the Stage 0 Python streaming reference. The csynth `latency_max_cycles` column is one event-step top-function call latency; `ii_max_cycles` is the conservative repeated-call event interval when invoking the step function back-to-back.",
        "",
        "## CSim and Csynth Summary",
        "",
        md_table(summary),
        "",
        "## Baseline Comparison",
        "",
        md_table(comparison),
        "",
        "## Naive Block0 Improvement",
        "",
    ]
    if "latency_max_cycles" in summary.columns:
        compact = summary[
            [
                "variant",
                "latency_max_cycles",
                "ii_max_cycles",
                "latency_us_100mhz",
                "interval_us_100mhz",
                "pipeline_type",
                "speedup_vs_naive_block0_total",
                "speedup_vs_naive_block0_amortized",
                "throughput_speedup_vs_naive_block0_total",
                "throughput_speedup_vs_naive_block0_amortized",
                "csynth_report",
            ]
        ].copy()
        lines += [md_table(compact), ""]
    else:
        lines += ["Csynth latency missing.", ""]
    lines += [
        "## Error-Stability Readout",
        "",
        "The `last_minus_first_quarter_rmse` and `last_over_first_quarter_rmse` columns are a simple check for timestep-wise error growth. They are diagnostic only; Stage 1 did not introduce a new tuned threshold.",
        "",
        "## Stage Boundary",
        "",
        "STOP HERE. Blocks 1-4 are intentionally not implemented until this Stage 1 block0 result is reviewed.",
    ]
    (ROOT / "STAGE1_BLOCK0_STREAMING_HLS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = collect()
    summary.to_csv(RESULTS / "stage1_block0_hls_summary.csv", index=False)
    with (RESULTS / "stage1_block0_hls_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(orient="records"), handle, indent=2)
    write_report(summary)
    print(f"wrote {RESULTS / 'stage1_block0_hls_summary.csv'}")
    print(f"wrote {ROOT / 'STAGE1_BLOCK0_STREAMING_HLS_REPORT.md'}")


if __name__ == "__main__":
    main()
