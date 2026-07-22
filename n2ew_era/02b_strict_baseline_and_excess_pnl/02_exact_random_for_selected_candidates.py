from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE02_DIR = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
if str(PHASE02_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE02_DIR))

from common_phase02 import RANDOM_SEED, iter_event_frames, policy_catalog, policy_decision  # noqa: E402


ROOT = PROJECT_ROOT / "n2ew_era" / "02b_strict_baseline_and_excess_pnl"
RESULTS = ROOT / "results"
TRIALS = 1000
SPREAD_LABELS = ["<=1", "(1,2]", "(2,3]", ">3"]


def spread_bucket(spread: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(spread, errors="coerce"), bins=[-np.inf, 1.0, 2.0, 3.0, np.inf], labels=SPREAD_LABELS).astype(str)


def load_candidates() -> list[dict[str, Any]]:
    return json.loads((RESULTS / "strict_selected_candidates.json").read_text(encoding="utf-8"))["candidates"]


def sample_sum_without_replacement(values: np.ndarray, k: int, rng: np.random.Generator) -> float:
    n = len(values)
    if k <= 0 or n <= 0:
        return 0.0
    if k >= n:
        return float(values.sum())
    if k > n // 2:
        exclude = rng.choice(n, size=n - k, replace=False)
        return float(values.sum() - values[exclude].sum())
    idx = rng.choice(n, size=k, replace=False)
    return float(values[idx].sum())


def exact_trials_for_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    policy = {p["policy_id"]: p for p in policy_catalog()}[candidate["policy_id"]]
    horizon = int(candidate["horizon"])
    split = str(candidate["split"])
    group_values: dict[tuple[str, str, str], np.ndarray] = {}
    group_required: dict[tuple[str, str, str], int] = {}

    for frame in iter_event_frames(split, horizons=[horizon]):
        decision = policy_decision(policy, frame)
        buckets = spread_bucket(frame["spread_ticks"]).to_numpy(dtype=object)
        sessions = frame["session_id"].astype(str).to_numpy(dtype=object)
        for side, decision_value, col in [("bid", 1, f"bid_pnl_H{horizon}"), ("ask", 2, f"ask_pnl_H{horizon}")]:
            values = np.nan_to_num(frame[col].to_numpy(dtype=float), nan=0.0)
            for session_id in np.unique(sessions):
                sess_mask = sessions == session_id
                for bucket in SPREAD_LABELS:
                    mask = sess_mask & (buckets == bucket)
                    if not mask.any():
                        continue
                    key = (side, str(session_id), bucket)
                    group_values[key] = np.concatenate([group_values.get(key, np.array([], dtype=float)), values[mask]])
                    group_required[key] = group_required.get(key, 0) + int((mask & (decision == decision_value)).sum())

    rng = np.random.default_rng(RANDOM_SEED + horizon)
    trial_pnls = np.zeros(TRIALS, dtype=float)
    for t in range(TRIALS):
        total = 0.0
        for key, values in group_values.items():
            total += sample_sum_without_replacement(values, int(group_required.get(key, 0)), rng)
        trial_pnls[t] = total

    model_pnl = float(candidate["total_pnl"])
    return {
        "split": split,
        "policy_id": str(candidate["policy_id"]),
        "horizon": horizon,
        "trials": TRIALS,
        "model_pnl": model_pnl,
        "selected_action_count": int(candidate["selected_action_count"]),
        "exact_random_mean": float(trial_pnls.mean()),
        "exact_random_std": float(trial_pnls.std(ddof=1)),
        "exact_random_90p": float(np.quantile(trial_pnls, 0.90)),
        "exact_random_95p": float(np.quantile(trial_pnls, 0.95)),
        "exact_random_99p": float(np.quantile(trial_pnls, 0.99)),
        "empirical_p_value_exact_random_ge_model": float(np.mean(trial_pnls >= model_pnl)),
        "beats_exact_random_mean": bool(model_pnl > float(trial_pnls.mean())),
        "beats_exact_random_95p": bool(model_pnl > float(np.quantile(trial_pnls, 0.95))),
        "matching_keys": "side + session_id + spread_bucket + exact selected count",
        "sampling_method": "actual_without_replacement_matched_trials",
    }


def update_report(exact: pd.DataFrame) -> None:
    report = ROOT / "STRICT_BASELINE_EXCESS_PNL_REPORT.md"
    text = report.read_text(encoding="utf-8")
    addition = "\n\n## Exact Random Check For Forwarded Candidates\n\n"
    addition += "The top candidates forwarded to Phase 02C were additionally checked with actual without-replacement matched random trials on validation. This check matches side, session, spread bucket, and exact selected count.\n\n"
    addition += exact.to_markdown(index=False)
    addition += "\n"
    report.write_text(text + addition, encoding="utf-8")


def main() -> None:
    candidates = load_candidates()
    rows = []
    for candidate in candidates:
        rows.append(exact_trials_for_candidate(candidate))
        print(f"exact random complete {candidate['policy_id']} H{candidate['horizon']}")
    exact = pd.DataFrame(rows)
    exact.to_csv(RESULTS / "random_summary_strict_exact_top_candidates.csv", index=False)
    update_report(exact)
    print(f"Wrote {RESULTS / 'random_summary_strict_exact_top_candidates.csv'}")


if __name__ == "__main__":
    main()
