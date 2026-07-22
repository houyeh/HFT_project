from __future__ import annotations

import numpy as np
import pandas as pd

from stress_robust_utils import (
    DIAGNOSTICS_DIR,
    STRESSES,
    baseline_policy_masks,
    decomp_to_row,
    ensure_dirs,
    load_frames_actions,
    simulate_decomp,
)


def _random_v3_decomp(frame: pd.DataFrame, actions: pd.DataFrame, stress_name: str, trials: int = 128) -> dict:
    stress = STRESSES[stress_name]
    ref_bid, ref_ask = baseline_policy_masks(frame, "016_v2")
    rng = np.random.default_rng(20260527)
    grouped = frame.groupby(["session_id", "spread_bucket"], observed=False, sort=True)
    rows = []
    for _ in range(int(trials)):
        pull_bid = np.zeros(len(frame), dtype=bool)
        pull_ask = np.zeros(len(frame), dtype=bool)
        for _, idx in grouped.groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            k_bid = int(np.sum(ref_bid[idx_arr]))
            k_ask = int(np.sum(ref_ask[idx_arr]))
            if k_bid:
                pull_bid[rng.choice(idx_arr, size=min(k_bid, len(idx_arr)), replace=False)] = True
            if k_ask:
                pull_ask[rng.choice(idx_arr, size=min(k_ask, len(idx_arr)), replace=False)] = True
        result = simulate_decomp(frame, actions, pull_bid, pull_ask, stress)
        rows.append(decomp_to_row("random_v3_mean", "test", stress_name, result))
    df = pd.DataFrame(rows)
    mean = df.drop(columns=["policy", "split", "stress_name"]).mean(numeric_only=True).to_dict()
    mean.update({"policy": "random_v3_mean", "split": "test", "stress_name": stress_name, "random_trials": int(trials)})
    return mean


def run() -> None:
    ensure_dirs()
    frame, actions = load_frames_actions("test")
    rows = []
    for policy in ["no_cancel", "015_common", "016_v2", "020_selected_policy"]:
        pull_bid, pull_ask = baseline_policy_masks(frame, policy)
        for stress_name, stress in STRESSES.items():
            result = simulate_decomp(frame, actions, pull_bid, pull_ask, stress)
            rows.append(decomp_to_row(policy, "test", stress_name, result))
    for stress_name in STRESSES:
        rows.append(_random_v3_decomp(frame, actions, stress_name))

    out = pd.DataFrame(rows)
    out.to_csv(DIAGNOSTICS_DIR / "stress_pnl_decomposition.csv", index=False)

    moderate = out[out["stress_name"] == "moderate"].copy()
    lines = [
        "# Stress PnL Decomposition",
        "",
        "Decomposition identity: `net_pnl = spread_capture - adverse_loss - fees`.",
        "`adverse_loss` is the reconciled residual containing adverse markout, stress toxic penalty, and terminal inventory mark-to-market effects.",
        "",
        "## Moderate Stress Summary",
        "",
    ]
    for _, row in moderate.sort_values("net_pnl", ascending=False).iterrows():
        lines.append(
            f"- {row['policy']}: net `{row['net_pnl']:.0f}`, spread_capture `{row['spread_capture']:.0f}`, "
            f"adverse_loss `{row['adverse_loss']:.0f}`, fees `{row['fees']:.0f}`, trades `{int(row['trade_count'])}`, "
            f"toxic fills `{int(row['toxic_fill_count'])}`, non-toxic fills `{int(row['non_toxic_fill_count'])}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The moderate-stress failure occurs when spread capture and fewer surviving fills cannot offset adverse residual loss plus fees. Policies can still be defensive if they reduce adverse loss relative to no-cancel/random, but absolute profitability requires positive net after this decomposition.",
        ]
    )
    (DIAGNOSTICS_DIR / "stress_pnl_decomposition.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {DIAGNOSTICS_DIR / 'stress_pnl_decomposition.csv'}")
