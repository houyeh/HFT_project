from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
PHASE06F_RESULTS = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_phase02f import (  # noqa: E402
    FillNorms,
    RESULTS,
    SELECTED_POLICY_ID,
    iter_excess_event_frames,
    latency_aware_fill_probabilities,
    policy_decision,
    policy_set,
    stable_uniform,
    write_markdown,
)

SPLIT = "val"
HORIZON = 20
FILL_MODEL = "conservative_l2_stress"
FPGA_LATENCY_US = 83.16
CPU_RAW_CSV = PHASE06F_RESULTS / "cpu_baseline_timing_libtorch_raw.csv"
FILL_NORMS_JSON = RESULTS / "causal_fill_proxy_norms.json"
DIFF_EPS = 1e-12
INDISTINGUISHABLE_RATE = 0.001


def read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_fill_norms() -> FillNorms:
    payload = read_json(FILL_NORMS_JSON)
    return FillNorms(
        top_depth_median=float(payload["top_depth_median"]),
        bid_sz_1_median=float(payload["bid_sz_1_median"]),
        ask_sz_1_median=float(payload["ask_sz_1_median"]),
    )


def load_cpu_latency_samples() -> np.ndarray:
    raw = pd.read_csv(CPU_RAW_CSV)
    samples = pd.to_numeric(raw["latency_us"], errors="coerce").dropna().to_numpy(dtype=float)
    samples = samples[np.isfinite(samples) & (samples >= 0.0)]
    if len(samples) == 0:
        raise RuntimeError(f"No usable latency samples in {CPU_RAW_CSV}")
    return samples


def empirical_latency_for_frame(frame: pd.DataFrame, decision: np.ndarray, samples: np.ndarray, salt_suffix: str) -> np.ndarray:
    u = stable_uniform(frame["session_id"], frame["event_index"], decision, salt=f"02f|val_prob_equiv|{salt_suffix}")
    idx = np.minimum((u * len(samples)).astype(np.int64), len(samples) - 1)
    return samples[idx]


def main() -> None:
    norms = load_fill_norms()
    cpu_samples = load_cpu_latency_samples()
    policy = policy_set()[SELECTED_POLICY_ID]

    selected_total = 0
    different_total = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    part_rows = []

    for part_idx, frame in enumerate(iter_excess_event_frames(SPLIT, horizons=[HORIZON])):
        decision = policy_decision(policy, frame).astype(np.int8)
        selected = decision != 0
        cpu_latency = empirical_latency_for_frame(frame, decision, cpu_samples, salt_suffix=f"{SPLIT}|part{part_idx}")
        probs_cpu, meta_cpu = latency_aware_fill_probabilities(frame, decision, FILL_MODEL, norms, cpu_latency)
        probs_fpga, meta_fpga = latency_aware_fill_probabilities(frame, decision, FILL_MODEL, norms, FPGA_LATENCY_US)
        diff = np.abs(probs_cpu[selected] - probs_fpga[selected])
        selected_count = int(selected.sum())
        different_count = int((diff > DIFF_EPS).sum())
        selected_total += selected_count
        different_total += different_count
        if len(diff):
            max_abs_diff = max(max_abs_diff, float(np.max(diff)))
            sum_abs_diff += float(np.sum(diff))
        part_rows.append(
            {
                "split": SPLIT,
                "part_index": int(part_idx),
                "selected_action_count": selected_count,
                "different_probability_count": different_count,
                "different_probability_rate": float(different_count / selected_count) if selected_count else 0.0,
                "max_abs_probability_diff": float(np.max(diff)) if len(diff) else 0.0,
                "mean_abs_probability_diff": float(np.mean(diff)) if len(diff) else 0.0,
                "cpu_mean_event_shift": float(meta_cpu["mean_events_to_skip_selected"]),
                "fpga_mean_event_shift": float(meta_fpga["mean_events_to_skip_selected"]),
                "cpu_latency_no_fill_count": int(meta_cpu["latency_no_fill_count"]),
                "fpga_latency_no_fill_count": int(meta_fpga["latency_no_fill_count"]),
            }
        )

    different_rate = float(different_total / selected_total) if selected_total else 0.0
    mean_abs_diff = float(sum_abs_diff / selected_total) if selected_total else 0.0
    summary = {
        "split": SPLIT,
        "policy_id": SELECTED_POLICY_ID,
        "fill_model": FILL_MODEL,
        "selected_action_count": int(selected_total),
        "different_probability_count": int(different_total),
        "different_probability_rate": different_rate,
        "max_abs_probability_diff": float(max_abs_diff),
        "mean_abs_probability_diff": mean_abs_diff,
        "diff_epsilon": DIFF_EPS,
        "indistinguishable_threshold_rate": INDISTINGUISHABLE_RATE,
        "proxy_indistinguishable": bool(different_rate < INDISTINGUISHABLE_RATE),
        "cpu_raw_latency_source": str(CPU_RAW_CSV),
        "fpga_latency_us": FPGA_LATENCY_US,
    }

    part_df = pd.DataFrame(part_rows)
    pd.DataFrame([summary]).to_csv(RESULTS / "val_latency_fill_probability_equivalence_summary.csv", index=False)
    part_df.to_csv(RESULTS / "val_latency_fill_probability_equivalence_by_part.csv", index=False)

    conclusion = (
        "The current proxy cannot meaningfully distinguish CPU and FPGA latency at this latency/data-resolution scale. Running n=1000 matched-random trials will not change that conclusion unless the latency scale or data resolution changes."
        if summary["proxy_indistinguishable"]
        else "The current proxy shows observable CPU-vs-FPGA fill probability differences on val; a full PnL/matched-random run may be worth considering."
    )

    lines = [
        "# Val Latency Fill Probability Equivalence Check",
        "",
        "Scope: val split only, selected decisions only. This does not run PnL, matched random, or NEW5.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| selected_action_count | {summary['selected_action_count']} |",
        f"| different_probability_count | {summary['different_probability_count']} |",
        f"| different_probability_rate | {summary['different_probability_rate']:.8%} |",
        f"| max_abs_probability_diff | {summary['max_abs_probability_diff']:.12g} |",
        f"| mean_abs_probability_diff | {summary['mean_abs_probability_diff']:.12g} |",
        f"| diff_epsilon | {DIFF_EPS:.1e} |",
        "",
        f"Conclusion: {conclusion}",
        "",
        "If the probability vectors are effectively identical, a full n=1000 matched-random run would mainly confirm the same result at much higher compute cost.",
    ]
    write_markdown(ROOT / "VAL_LATENCY_FILL_PROBABILITY_EQUIVALENCE.md", lines)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(conclusion)


if __name__ == "__main__":
    main()
