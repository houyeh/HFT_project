from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase02 import RESULTS_DIR, RANDOM_SEED, iter_event_frames, selected_policy_entries


TRIALS = 1000
RANDOM_SPLITS = ["val", "test", "new4"]


def load_pnl_arrays(split: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    bids: list[np.ndarray] = []
    asks: list[np.ndarray] = []
    for frame in iter_event_frames(split, horizons=[horizon]):
        bids.append(np.nan_to_num(frame[f"bid_pnl_H{horizon}"].to_numpy(dtype=float), nan=0.0))
        asks.append(np.nan_to_num(frame[f"ask_pnl_H{horizon}"].to_numpy(dtype=float), nan=0.0))
    return np.concatenate(bids), np.concatenate(asks)


def random_uniform_trials(bid: np.ndarray, ask: np.ndarray, trials: int, rng: np.random.Generator) -> np.ndarray:
    n = len(bid)
    out = np.zeros(trials, dtype=float)
    for i in range(trials):
        choice = rng.integers(0, 3, size=n, dtype=np.int8)
        out[i] = float(bid[choice == 1].sum() + ask[choice == 2].sum())
    return out


def random_matched_count_trials(
    bid: np.ndarray,
    ask: np.ndarray,
    selected_count: int,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(bid)
    k = max(0, min(int(selected_count), n))
    out = np.zeros(trials, dtype=float)
    if k == 0:
        return out
    all_idx = np.arange(n, dtype=np.int64)
    for i in range(trials):
        if k == n:
            idx = all_idx
        elif k > n // 2:
            exclude = rng.choice(n, size=n - k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[exclude] = False
            idx = all_idx[mask]
        else:
            idx = rng.choice(n, size=k, replace=False)
        side = rng.integers(0, 2, size=len(idx), dtype=np.int8)
        out[i] = float(bid[idx[side == 0]].sum() + ask[idx[side == 1]].sum())
    return out


def summarize(split: str, policy_id: str, horizon: int, baseline_name: str, model_pnl: float, selected_count: int, trials: np.ndarray) -> dict:
    return {
        "split": split,
        "policy_id": policy_id,
        "horizon": int(horizon),
        "random_baseline": baseline_name,
        "trials": int(len(trials)),
        "model_pnl": float(model_pnl),
        "model_selected_action_count": int(selected_count),
        "random_mean": float(np.mean(trials)),
        "random_std": float(np.std(trials, ddof=1)),
        "random_90p": float(np.quantile(trials, 0.90)),
        "random_95p": float(np.quantile(trials, 0.95)),
        "random_99p": float(np.quantile(trials, 0.99)),
        "empirical_p_value_random_ge_model": float(np.mean(trials >= float(model_pnl))),
        "beats_random_95p": bool(float(model_pnl) > float(np.quantile(trials, 0.95))),
    }


def main() -> None:
    grid = pd.read_csv(RESULTS_DIR / "baseline_grid_all_splits.csv")
    main_policy = selected_policy_entries()[0]
    policy_id = main_policy["policy_id"]
    horizon = int(main_policy["selected_horizon"])
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows: list[dict] = []
    trial_rows: list[dict] = []

    for split in RANDOM_SPLITS:
        row = grid[
            (grid["split"].astype(str) == split)
            & (grid["policy_id"].astype(str) == policy_id)
            & (grid["horizon"].astype(int) == horizon)
        ].iloc[0]
        model_pnl = float(row["total_pnl"])
        selected_count = int(row["selected_action_count"])
        bid, ask = load_pnl_arrays(split, horizon)

        random_sets = {
            "random_action_uniform": random_uniform_trials(bid, ask, TRIALS, rng),
            "random_quote_matched_count": random_matched_count_trials(bid, ask, selected_count, TRIALS, rng),
        }
        for name, values in random_sets.items():
            summary_rows.append(summarize(split, policy_id, horizon, name, model_pnl, selected_count, values))
            for trial_id, pnl in enumerate(values):
                trial_rows.append({"split": split, "random_baseline": name, "trial_id": trial_id, "random_pnl": float(pnl)})
        print(f"random baselines complete split={split} horizon=H{horizon}")

    summary = pd.DataFrame(summary_rows)
    trials = pd.DataFrame(trial_rows)
    summary.to_csv(RESULTS_DIR / "random_baseline_summary.csv", index=False)
    trials.to_csv(RESULTS_DIR / "random_baseline_trials.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {RESULTS_DIR / 'random_baseline_summary.csv'}")


if __name__ == "__main__":
    main()
