from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
PHASE06F_RESULTS = PROJECT_ROOT / "n2ew_era" / "06f_phc_lite_manual_hls" / "results"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_phase02f import (  # noqa: E402
    ACTION_SLOT_ASK,
    ACTION_SLOT_BID,
    FillNorms,
    HORIZONS,
    MAKER_FEE,
    RESULTS,
    SELECTED_POLICY_ID,
    compute_train_fill_norms,
    deterministic_latency_aware_fill_mask,
    ensure_dirs,
    iter_excess_event_frames,
    policy_decision,
    policy_set,
    spread_bucket_values,
    stable_uniform,
    write_markdown,
)

HORIZON = 20
EVAL_SPLITS = ["val", "new3", "new4"]
FILL_MODEL = "conservative_l2_stress"
INVENTORY_CAP = 1
FEE_MULTIPLIER = 1.0
FPGA_LATENCY_US = 83.16
ORIGINAL_CPU_SUMMARY = PHASE06F_RESULTS / "cpu_baseline_timing_libtorch.json"
CPU_RAW_SUMMARY = PHASE06F_RESULTS / "cpu_baseline_timing_libtorch_raw.json"
CPU_RAW_CSV = PHASE06F_RESULTS / "cpu_baseline_timing_libtorch_raw.csv"
DRIFT_REL_TOL = 0.10
FILL_NORMS_JSON = RESULTS / "causal_fill_proxy_norms.json"


def load_sequential_module():
    path = ROOT / "04_sequential_excess_inventory.py"
    spec = importlib.util.spec_from_file_location("phase02f_sequential_inventory", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEQ = load_sequential_module()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_latency_samples(samples: np.ndarray) -> dict[str, float]:
    ordered = np.sort(samples.astype(float))
    p99_idx = min(len(ordered) - 1, int(0.99 * len(ordered)))
    return {
        "mean_us": float(np.mean(samples)),
        "median_us": float(np.median(samples)),
        "p99_us": float(ordered[p99_idx]),
        "min_us": float(np.min(samples)),
        "max_us": float(np.max(samples)),
    }


def compare_cpu_summaries() -> tuple[dict[str, Any], bool]:
    original = read_json(ORIGINAL_CPU_SUMMARY)
    raw_summary = read_json(CPU_RAW_SUMMARY)
    rows = []
    consistent = True
    for key in ["mean_us", "median_us", "p99_us", "min_us"]:
        old = float(original[key])
        new = float(raw_summary[key])
        rel = (new - old) / old if old else math.nan
        rows.append({"metric": key, "original": old, "raw_rerun": new, "relative_delta": rel})
        if abs(rel) > DRIFT_REL_TOL:
            consistent = False
    payload = {
        "original_summary_path": str(ORIGINAL_CPU_SUMMARY),
        "raw_summary_path": str(CPU_RAW_SUMMARY),
        "raw_csv_path": str(CPU_RAW_CSV),
        "relative_tolerance": DRIFT_REL_TOL,
        "consistent_with_original": bool(consistent),
        "rows": rows,
    }
    return payload, bool(consistent)


def write_blocker_report(comparison: dict[str, Any]) -> None:
    lines = [
        "# Latency-Aware Fill Proxy Blocker",
        "",
        "The latency-aware CPU scenario requires empirical raw CPU latency samples. Raw samples were recaptured, but the rerun summary does not match the earlier LibTorch CPU summary within the configured 10% relative tolerance.",
        "",
        "| metric | original | raw_rerun | relative_delta |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| {row['metric']} | {row['original']:.6g} | {row['raw_rerun']:.6g} | {row['relative_delta'] * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "This script refuses to run the CPU-vs-FPGA PnL comparison by default because mixing the older summary with the newly captured empirical raw distribution would weaken the latency-tail claim.",
            "",
            "To proceed deliberately with the newly captured raw empirical distribution, rerun with `--allow-raw-drift`. The generated report will keep this caveat explicit.",
        ]
    )
    write_markdown(RESULTS / "latency_aware_fill_blocker_report.md", lines)


def load_cpu_latency_samples() -> np.ndarray:
    if not CPU_RAW_CSV.exists():
        raise FileNotFoundError(f"CPU raw latency CSV not found: {CPU_RAW_CSV}")
    raw = pd.read_csv(CPU_RAW_CSV)
    if "latency_us" not in raw.columns:
        raise ValueError(f"{CPU_RAW_CSV} must contain latency_us")
    samples = pd.to_numeric(raw["latency_us"], errors="coerce").dropna().to_numpy(dtype=float)
    samples = samples[np.isfinite(samples) & (samples >= 0.0)]
    if len(samples) < 1000:
        raise ValueError(f"Need at least 1000 raw latency samples; got {len(samples)}")
    return samples


def load_fill_norms() -> FillNorms:
    if FILL_NORMS_JSON.exists():
        payload = read_json(FILL_NORMS_JSON)
        return FillNorms(
            top_depth_median=float(payload["top_depth_median"]),
            bid_sz_1_median=float(payload["bid_sz_1_median"]),
            ask_sz_1_median=float(payload["ask_sz_1_median"]),
        )
    return compute_train_fill_norms()


def empirical_latency_for_frame(frame: pd.DataFrame, decision: np.ndarray, samples: np.ndarray, salt_suffix: str) -> np.ndarray:
    u = stable_uniform(frame["session_id"], frame["event_index"], decision, salt=f"02f|cpu_latency_empirical|{salt_suffix}")
    idx = np.minimum((u * len(samples)).astype(np.int64), len(samples) - 1)
    return samples[idx]


def base_excess_pnl(frame: pd.DataFrame, decision: np.ndarray) -> np.ndarray:
    bid = frame[f"bid_excess_pnl_H{HORIZON}"].to_numpy(dtype=float)
    ask = frame[f"ask_excess_pnl_H{HORIZON}"].to_numpy(dtype=float)
    out = np.zeros(len(frame), dtype=float)
    bid_mask = decision == (ACTION_SLOT_BID + 1)
    ask_mask = decision == (ACTION_SLOT_ASK + 1)
    out[bid_mask] = bid[bid_mask]
    out[ask_mask] = ask[ask_mask]
    return np.nan_to_num(out, nan=0.0)


def simulate_frame(
    frame: pd.DataFrame,
    decision: np.ndarray,
    norms: Any,
    scenario: str,
    cpu_samples: np.ndarray,
    salt_suffix: str,
) -> dict[str, Any]:
    if scenario == "cpu_empirical":
        latency = empirical_latency_for_frame(frame, decision, cpu_samples, salt_suffix=salt_suffix)
    elif scenario == "fpga_fixed_83p16us":
        latency = FPGA_LATENCY_US
    else:
        raise ValueError(scenario)
    fills, probs, meta = deterministic_latency_aware_fill_mask(frame, decision, FILL_MODEL, norms, latency, salt_suffix=salt_suffix)
    pnl = base_excess_pnl(frame, decision)
    total, fills_n, forced, max_inv, max_dd, wins, losses = SEQ.simulate_scheduled_excess_numba(
        decision.astype(np.int8),
        fills.astype(np.bool_),
        pnl.astype(np.float64),
        HORIZON,
        INVENTORY_CAP,
        FEE_MULTIPLIER,
        float(MAKER_FEE),
    )
    selected_count = int((decision != 0).sum())
    return {
        "sequential_excess_pnl": float(total),
        "selected_action_count": selected_count,
        "fill_count": int(fills_n),
        "no_fill_count": int(max(selected_count - fills_n, 0)),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_inv),
        "max_drawdown": float(max_dd),
        "win_count": int(wins),
        "loss_count": int(losses),
        "fill_prob_mean_selected": float(np.mean(probs[decision != 0])) if selected_count else 0.0,
        **meta,
    }


def sample_matched_random_decision(frame: pd.DataFrame, reference_decision: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(len(frame), dtype=np.int8)
    buckets = spread_bucket_values(frame["spread_ticks"])
    for bucket in np.unique(buckets):
        group_idx = np.flatnonzero(buckets == bucket)
        ref_group = reference_decision[group_idx]
        bid_count = int(np.sum(ref_group == (ACTION_SLOT_BID + 1)))
        ask_count = int(np.sum(ref_group == (ACTION_SLOT_ASK + 1)))
        total = bid_count + ask_count
        if total <= 0:
            continue
        if total > len(group_idx):
            total = len(group_idx)
        chosen = rng.choice(group_idx, size=total, replace=False)
        side_labels = np.array([ACTION_SLOT_BID + 1] * bid_count + [ACTION_SLOT_ASK + 1] * ask_count, dtype=np.int8)
        if len(side_labels) > len(chosen):
            side_labels = side_labels[: len(chosen)]
        rng.shuffle(side_labels)
        out[chosen[: len(side_labels)]] = side_labels
    return out


def aggregate_dicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    total_pnl = sum(float(r["sequential_excess_pnl"]) for r in rows)
    selected = sum(int(r["selected_action_count"]) for r in rows)
    fills = sum(int(r["fill_count"]) for r in rows)
    latency_no_fill = sum(int(r["latency_no_fill_count"]) for r in rows)
    wins = sum(int(r["win_count"]) for r in rows)
    losses = sum(int(r["loss_count"]) for r in rows)
    return {
        "sequential_excess_pnl": float(total_pnl),
        "selected_action_count": int(selected),
        "fill_count": int(fills),
        "no_fill_count": int(max(selected - fills, 0)),
        "latency_no_fill_count": int(latency_no_fill),
        "latency_no_fill_rate": float(latency_no_fill / selected) if selected else 0.0,
        "forced_no_quote_count": int(sum(int(r["forced_no_quote_count"]) for r in rows)),
        "max_abs_inventory": int(max(int(r["max_abs_inventory"]) for r in rows)),
        "max_drawdown": float(max(float(r["max_drawdown"]) for r in rows)),
        "win_rate": float(wins / fills) if fills else 0.0,
        "loss_rate": float(losses / fills) if fills else 0.0,
        "fill_rate_after_latency": float(fills / selected) if selected else 0.0,
        "mean_events_to_skip_selected": float(np.average([r["mean_events_to_skip_selected"] for r in rows], weights=[max(r["selected_count"], 1) for r in rows])),
        "p99_events_to_skip_selected_max_session": float(max(float(r["p99_events_to_skip_selected"]) for r in rows)),
        "max_events_to_skip_selected": int(max(int(r["max_events_to_skip_selected"]) for r in rows)),
    }


def run_policy_split(split: str, scenario: str, norms: Any, cpu_samples: np.ndarray) -> tuple[dict[str, Any], list[tuple[pd.DataFrame, np.ndarray]]]:
    policy = policy_set()[SELECTED_POLICY_ID]
    rows: list[dict[str, Any]] = []
    cached: list[tuple[pd.DataFrame, np.ndarray]] = []
    for part_idx, frame in enumerate(iter_excess_event_frames(split, horizons=[HORIZON])):
        decision = policy_decision(policy, frame).astype(np.int8)
        cached.append((frame, decision))
        rows.append(simulate_frame(frame, decision, norms, scenario, cpu_samples, salt_suffix=f"{split}|policy|{part_idx}"))
    agg = aggregate_dicts(rows)
    agg.update(
        {
            "split": split,
            "policy_id": SELECTED_POLICY_ID,
            "scenario": scenario,
            "horizon": HORIZON,
            "fill_model": FILL_MODEL,
            "inventory_cap": INVENTORY_CAP,
            "fee_multiplier": FEE_MULTIPLIER,
        }
    )
    return agg, cached


def run_random_trials(
    split: str,
    scenario: str,
    cached: list[tuple[pd.DataFrame, np.ndarray]],
    norms: Any,
    cpu_samples: np.ndarray,
    trials: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trial_rows = []
    for trial in range(trials):
        rng = np.random.default_rng(abs(hash((split, scenario, trial, "02f_latency_random"))) % (2**32))
        frame_rows = []
        for part_idx, (frame, reference_decision) in enumerate(cached):
            random_decision = sample_matched_random_decision(frame, reference_decision, rng)
            frame_rows.append(
                simulate_frame(
                    frame,
                    random_decision,
                    norms,
                    scenario,
                    cpu_samples,
                    salt_suffix=f"{split}|random|{scenario}|trial{trial}|part{part_idx}",
                )
            )
        agg = aggregate_dicts(frame_rows)
        agg.update({"split": split, "scenario": scenario, "trial": int(trial)})
        trial_rows.append(agg)
    trials_df = pd.DataFrame(trial_rows)
    return trials_df, {
        "matched_random_mean": float(trials_df["sequential_excess_pnl"].mean()) if len(trials_df) else 0.0,
        "matched_random_95p": float(trials_df["sequential_excess_pnl"].quantile(0.95)) if len(trials_df) else 0.0,
        "matched_random_max": float(trials_df["sequential_excess_pnl"].max()) if len(trials_df) else 0.0,
    }


def write_report(
    comparison: dict[str, Any],
    latency_summary: dict[str, float],
    policy_df: pd.DataFrame,
    random_summary_df: pd.DataFrame,
    allow_raw_drift: bool,
) -> None:
    lines = [
        "# Latency-Aware Fill Proxy Report",
        "",
        "This is a proxy model for latency effects on fill quality. It approximates latency-driven adverse selection by shifting the order-book state used for fill probability forward by a latency-dependent number of events. It is not validated exchange fill data, not true queue replay, and not proof of real-world profitability. If true tick-level order acknowledgements, queue priority, and exchange matching data become available, they should replace this proxy.",
        "",
        "Decision features are unchanged: policies still decide from the original event. Only the book state used for fill probability is shifted after the decision latency.",
        "",
        f"CPU raw drift allowed: `{allow_raw_drift}`.",
        f"CPU raw latency source: `{CPU_RAW_CSV}`.",
        f"CPU raw summary: mean={latency_summary['mean_us']:.4f}us, median={latency_summary['median_us']:.4f}us, p99={latency_summary['p99_us']:.4f}us, min={latency_summary['min_us']:.4f}us, max={latency_summary['max_us']:.4f}us.",
        f"FPGA latency scenario: fixed {FPGA_LATENCY_US:.2f}us.",
        "",
        "## CPU Summary Drift Check",
        "",
        "| metric | original | raw_rerun | relative_delta |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(f"| {row['metric']} | {row['original']:.6g} | {row['raw_rerun']:.6g} | {row['relative_delta'] * 100:.2f}% |")
    max_trials = int(policy_df["random_trials"].max()) if "random_trials" in policy_df.columns and len(policy_df) else 0
    if max_trials < 1000:
        lines.extend(
            [
                "",
                "## Matched Random Status",
                "",
                f"This report was generated with `{max_trials}` matched-random trial(s). It is a pipeline smoke test, not the final n=1000 matched-random significance run requested for presentation use.",
                "",
                "Run `python 07_latency_aware_fill_proxy_analysis.py --allow-raw-drift --trials 1000` to produce the full matched-random output after accepting the recaptured CPU latency distribution caveat.",
            ]
        )
    lines.extend(["", "## Policy Results", "", policy_df.to_markdown(index=False), "", "## Matched Random Summary", "", random_summary_df.to_markdown(index=False)])
    write_markdown(ROOT / "LATENCY_AWARE_FILL_PROXY_REPORT.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-raw-drift", action="store_true", help="Proceed even if the recaptured raw CPU summary differs from the older summary.")
    parser.add_argument("--trials", type=int, default=1000)
    args = parser.parse_args()

    ensure_dirs()
    comparison, consistent = compare_cpu_summaries()
    (RESULTS / "latency_aware_cpu_summary_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    if not consistent and not args.allow_raw_drift:
        write_blocker_report(comparison)
        raise SystemExit("CPU raw latency summary differs from original. See results/latency_aware_fill_blocker_report.md or rerun with --allow-raw-drift.")

    cpu_samples = load_cpu_latency_samples()
    latency_summary = summarize_latency_samples(cpu_samples)
    norms = load_fill_norms()

    policy_rows = []
    random_summary_rows = []
    random_trial_frames = []
    for split in EVAL_SPLITS:
        for scenario in ["cpu_empirical", "fpga_fixed_83p16us"]:
            policy_row, cached = run_policy_split(split, scenario, norms, cpu_samples)
            trials_df, random_stats = run_random_trials(split, scenario, cached, norms, cpu_samples, args.trials)
            policy_pnl = float(policy_row["sequential_excess_pnl"])
            policy_row.update(
                {
                    **random_stats,
                    "beats_matched_random_95p": bool(policy_pnl > random_stats["matched_random_95p"]),
                    "empirical_p_value": float((trials_df["sequential_excess_pnl"] >= policy_pnl).mean()) if len(trials_df) else math.nan,
                    "random_trials": int(args.trials),
                }
            )
            policy_rows.append(policy_row)
            random_summary_rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    **random_stats,
                    "policy_pnl": policy_pnl,
                    "beats_matched_random_95p": bool(policy_pnl > random_stats["matched_random_95p"]),
                    "empirical_p_value": policy_row["empirical_p_value"],
                    "random_trials": int(args.trials),
                }
            )
            trials_df["policy_pnl"] = policy_pnl
            random_trial_frames.append(trials_df)
            print(f"complete split={split} scenario={scenario} policy_pnl={policy_pnl:.2f} random95={random_stats['matched_random_95p']:.2f}", flush=True)

    policy_df = pd.DataFrame(policy_rows)
    random_summary_df = pd.DataFrame(random_summary_rows)
    random_trials_df = pd.concat(random_trial_frames, ignore_index=True) if random_trial_frames else pd.DataFrame()
    policy_df.to_csv(RESULTS / "latency_aware_fill_pnl_comparison.csv", index=False)
    random_summary_df.to_csv(RESULTS / "latency_aware_matched_random_summary.csv", index=False)
    random_trials_df.to_csv(RESULTS / "latency_aware_matched_random_trials.csv", index=False)
    write_report(comparison, latency_summary, policy_df, random_summary_df, allow_raw_drift=bool(args.allow_raw_drift))
    print(policy_df[["split", "scenario", "sequential_excess_pnl", "fill_count", "latency_no_fill_count", "latency_no_fill_rate", "empirical_p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
