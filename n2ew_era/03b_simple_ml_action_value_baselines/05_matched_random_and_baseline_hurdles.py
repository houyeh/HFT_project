from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase03b import (
    FROZEN_SPLITS,
    RANDOM_TRIALS,
    RESULTS,
    TARGET,
    build_feature_frame,
    ensure_dirs,
    part_paths,
    random_normal_trials,
    read_json,
    selected_actions_path,
)


def candidate_group_values(split: str) -> dict[tuple[str, str], list[np.ndarray]]:
    groups: dict[tuple[str, str], list[np.ndarray]] = {}
    for path in part_paths(split):
        frame = build_feature_frame(path, include_targets=True)
        quote = frame[frame["action_slot"].astype(int).isin([0, 1])]
        for side, group in quote.groupby("action_side"):
            key = (str(group["session_id"].iloc[0]), str(side))
            groups.setdefault(key, []).append(group[TARGET].to_numpy(dtype=float))
    return groups


def selected_requirements(selected: pd.DataFrame) -> dict[tuple[str, str], int]:
    sel = selected[selected["decision"].astype(int) != 0].copy()
    req: dict[tuple[str, str], int] = {}
    for (session_id, side), group in sel.groupby(["session_id", "selected_side"]):
        req[(str(session_id), str(side))] = int(len(group))
    return req


def run_random_trials(split: str, rng: np.random.Generator) -> pd.DataFrame:
    selected = pd.read_parquet(selected_actions_path(split))
    req = selected_requirements(selected)
    groups = candidate_group_values(split)
    trials = np.zeros(RANDOM_TRIALS, dtype=float)
    health_rows = []
    for key, chunks in groups.items():
        values = np.concatenate(chunks)
        k = req.get(key, 0)
        group_trials = random_normal_trials(values, k, RANDOM_TRIALS, rng)
        trials += group_trials
        health_rows.append(
            {
                "split": split,
                "session_id": key[0],
                "side": key[1],
                "required_count": k,
                "candidate_pool_size": int(len(values)),
                "method": "finite_population_normal_1000_trials_matched_session_side",
            }
        )
    pd.DataFrame(health_rows).to_csv(RESULTS / f"matched_random_health_{split}.csv", index=False)
    return pd.DataFrame({"split": split, "trial": np.arange(RANDOM_TRIALS), "random_total_excess_pnl": trials})


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(20260601)
    trial_frames = []
    for split in FROZEN_SPLITS:
        trial_frames.append(run_random_trials(split, rng))
        print(f"random trials complete split={split}")
    trials = pd.concat(trial_frames, ignore_index=True)
    trials.to_csv(RESULTS / "matched_random_trials.csv", index=False)

    ml = pd.read_csv(RESULTS / "frozen_oos_evaluation.csv")
    base = pd.read_csv(RESULTS / "frozen_oos_baseline_controls.csv")
    selected_policy = read_json(RESULTS / "selected_phase03b_policy.json")
    rows = []
    for split, group in trials.groupby("split"):
        model_pnl = float(ml.loc[ml["split"].eq(split), "total_excess_pnl"].iloc[0])
        random_values = group["random_total_excess_pnl"].to_numpy(dtype=float)
        queue = base[(base["split"].eq(split)) & (base["policy_id"].eq("side_by_queue_imbalance_abs_0p1"))]
        old = base[(base["split"].eq(split)) & (base["policy_id"].eq("old_bid_only_spread_ge_1"))]
        queue_pnl = float(queue["total_excess_pnl"].iloc[0]) if not queue.empty else float("nan")
        old_pnl = float(old["total_excess_pnl"].iloc[0]) if not old.empty else float("nan")
        rows.append(
            {
                "split": split,
                "policy_id": selected_policy["policy_id"],
                "model_id": selected_policy["model_id"],
                "model_total_excess_pnl": model_pnl,
                "queue_rule_total_excess_pnl": queue_pnl,
                "old_bid_only_total_excess_pnl": old_pnl,
                "matched_random_mean": float(np.mean(random_values)),
                "matched_random_95p": float(np.quantile(random_values, 0.95)),
                "matched_random_99p": float(np.quantile(random_values, 0.99)),
                "empirical_p_value_fraction_random_ge_model": float(np.mean(random_values >= model_pnl)),
                "beats_matched_random_95p": bool(model_pnl > np.quantile(random_values, 0.95)),
                "beats_queue_imbalance_rule": bool(model_pnl > queue_pnl),
                "random_trials": len(random_values),
                "random_method": "finite_population_normal_1000_trials_matched_count_session_side",
                "test_new3_new4_used_for_selection": False,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "ml_hurdle_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

