from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common_phase02d import (
    FROZEN_OOS_SPLITS,
    RESULTS,
    SELECTED_HORIZON,
    SELECTED_POLICY_ID,
    SELECTED_RULE_DEFINITION,
    ensure_dirs,
    write_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE02_DIR = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
if str(PHASE02_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE02_DIR))

from common_phase02 import RANDOM_SEED, iter_event_frames, policy_catalog, policy_decision  # noqa: E402


TRIALS = 1000
SPREAD_LABELS = ["<=1", "(1,2]", "(2,3]", ">3"]
EXACT_RANDOM_ZERO_PVALUE_WORDING = (
    "In 1000 exact random trials, no random trial exceeded the selected policy. "
    "Therefore empirical p-value < 0.001."
)


def spread_bucket(spread: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(spread, errors="coerce"), bins=[-np.inf, 1.0, 2.0, 3.0, np.inf], labels=SPREAD_LABELS).astype(str)


def sample_sum(values: np.ndarray, k: int, rng: np.random.Generator) -> float:
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


def exact_oos_for_split(split: str) -> dict:
    policy = {p["policy_id"]: p for p in policy_catalog()}[SELECTED_POLICY_ID]
    group_values_lists: dict[tuple[str, str, str], list[np.ndarray]] = {}
    required: dict[tuple[str, str, str], int] = {}
    model_pnl = 0.0
    selected_total = 0

    for frame in iter_event_frames(split, horizons=[SELECTED_HORIZON]):
        decision = policy_decision(policy, frame)
        buckets = spread_bucket(frame["spread_ticks"]).to_numpy(dtype=object)
        sessions = frame["session_id"].astype(str).to_numpy(dtype=object)
        for side, decision_value, pnl_col in [("bid", 1, f"bid_pnl_H{SELECTED_HORIZON}"), ("ask", 2, f"ask_pnl_H{SELECTED_HORIZON}")]:
            values = np.nan_to_num(frame[pnl_col].to_numpy(dtype=float), nan=0.0)
            side_selected = decision == decision_value
            model_pnl += float(values[side_selected].sum())
            selected_total += int(side_selected.sum())
            for session_id in np.unique(sessions):
                session_mask = sessions == session_id
                for bucket in SPREAD_LABELS:
                    mask = session_mask & (buckets == bucket)
                    if not mask.any():
                        continue
                    key = (side, str(session_id), bucket)
                    group_values_lists.setdefault(key, []).append(values[mask])
                    required[key] = required.get(key, 0) + int((mask & side_selected).sum())

    group_values = {key: np.concatenate(chunks) for key, chunks in group_values_lists.items()}
    rng = np.random.default_rng(RANDOM_SEED + SELECTED_HORIZON + (1000 if split == "new4" else 0))
    trials = np.zeros(TRIALS, dtype=float)
    for trial in range(TRIALS):
        total = 0.0
        for key, values in group_values.items():
            total += sample_sum(values, required.get(key, 0), rng)
        trials[trial] = total

    return {
        "split": split,
        "policy_id": SELECTED_POLICY_ID,
        "horizon": SELECTED_HORIZON,
        "trials": TRIALS,
        "model_pnl": float(model_pnl),
        "selected_action_count": int(selected_total),
        "exact_random_mean": float(trials.mean()),
        "exact_random_std": float(trials.std(ddof=1)),
        "exact_random_90p": float(np.quantile(trials, 0.90)),
        "exact_random_95p": float(np.quantile(trials, 0.95)),
        "exact_random_99p": float(np.quantile(trials, 0.99)),
        "empirical_p_value_exact_random_ge_model": float(np.mean(trials >= model_pnl)),
        "beats_exact_random_mean": bool(model_pnl > float(trials.mean())),
        "beats_exact_random_95p": bool(model_pnl > float(np.quantile(trials, 0.95))),
        "matching_keys": "side + session_id + spread_bucket + exact selected count",
        "requested_match": "matched count + side ratio + session; spread bucket included as stricter control",
        "computational_status": "completed_exact_without_replacement",
    }


def main() -> None:
    ensure_dirs()
    rows = []
    failures = []
    for split in FROZEN_OOS_SPLITS:
        try:
            row = exact_oos_for_split(split)
            rows.append(row)
            print(f"Exact OOS random completed for {split}.")
        except Exception as exc:  # noqa: BLE001
            failures.append({"split": split, "reason": repr(exc)})
            print(f"Exact OOS random failed for {split}: {exc}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "exact_random_oos_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(RESULTS / "exact_random_oos_failures.csv", index=False)
    lines = [
        "# Exact Random OOS Report",
        "",
        f"Selected policy: `{SELECTED_POLICY_ID}`, horizon `H{SELECTED_HORIZON}`.",
        f"Rule: {SELECTED_RULE_DEFINITION}",
        "",
        "The OOS exact random check was attempted on `test` and `NEW4` with 1000 actual without-replacement trials. The requested match was selected count, side ratio, and session. The implementation also matched spread bucket, making the control stricter.",
        "",
    ]
    if out.empty:
        lines.append("Exact OOS random trials were not completed. Finite-population matched random from Phase 02B should be used as the fallback hurdle.")
    else:
        if (out["empirical_p_value_exact_random_ge_model"].astype(float) == 0.0).any():
            lines.extend([EXACT_RANDOM_ZERO_PVALUE_WORDING, ""])
        display = out.copy()
        display["empirical_p_value_exact_random_ge_model"] = display[
            "empirical_p_value_exact_random_ge_model"
        ].map(lambda value: "<0.001" if float(value) == 0.0 else f"{float(value):.6f}")
        lines.append(display.to_markdown(index=False))
    if failures:
        lines.extend(["", "## Failures", "", pd.DataFrame(failures).to_markdown(index=False)])
    write_markdown(RESULTS / "exact_random_oos_report.md", lines)


if __name__ == "__main__":
    main()
