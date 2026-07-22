from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from queue_stress_core import (
    ROOT,
    StressConfig,
    base_expected_values,
    ensure_random_masks,
    load_frames,
    policy_masks,
    random_v3_reference_pnl,
    simulate_policy,
)


HEATMAP_HAIRCUTS = [1.0, 0.75, 0.5, 0.25]
HEATMAP_TOXIC_MULTIPLIERS = [1.0, 1.25, 1.5, 2.0]
DELAY_TICKS = [0, 1, 2, 3, 5, 10]
MODERATE_DELAY_TICKS = [0, 1, 2]


def _policy_result(
    policies: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    actions: pd.DataFrame,
    policy: str,
    stress: StressConfig,
) -> dict[str, float]:
    frame, pull_bid, pull_ask = policies[policy]
    return simulate_policy(frame, pull_bid, pull_ask, actions, stress)


def _plot_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(HEATMAP_TOXIC_MULTIPLIERS)))
    ax.set_xticklabels([str(v) for v in HEATMAP_TOXIC_MULTIPLIERS])
    ax.set_yticks(np.arange(len(HEATMAP_HAIRCUTS)))
    ax.set_yticklabels([str(v) for v in HEATMAP_HAIRCUTS])
    ax.set_xlabel("toxic_multiplier")
    ax.set_ylabel("neutral_fill_haircut")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Net PnL")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:,.0f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_heatmaps(
    policies: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    actions: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    matrices: dict[str, np.ndarray] = {}
    for policy in ["015_common", "016_v2_binary"]:
        matrix = np.zeros((len(HEATMAP_HAIRCUTS), len(HEATMAP_TOXIC_MULTIPLIERS)), dtype=float)
        for i, haircut in enumerate(HEATMAP_HAIRCUTS):
            for j, toxic in enumerate(HEATMAP_TOXIC_MULTIPLIERS):
                stress = StressConfig(neutral_fill_haircut=haircut, toxic_multiplier=toxic)
                result = _policy_result(policies, actions, policy, stress)
                matrix[i, j] = float(result["net_pnl"])
                rows.append(
                    {
                        "policy": policy,
                        "neutral_fill_haircut": haircut,
                        "toxic_multiplier": toxic,
                        "cancel_delay_ticks": 0,
                        "net_pnl": result["net_pnl"],
                        "trade_count": result["trade_count"],
                        "max_drawdown": result["max_drawdown"],
                    }
                )
        matrices[policy] = matrix

    _plot_heatmap(matrices["015_common"], "015 Common Queue Stress", out_dir / "queue_stress_heatmap_015.png")
    _plot_heatmap(matrices["016_v2_binary"], "016 v2 Binary Queue Stress", out_dir / "queue_stress_heatmap_016_v2.png")
    values = pd.DataFrame(rows)
    values.to_csv(out_dir / "queue_stress_heatmap_values.csv", index=False)
    return values


def run_delay_sweep(
    policies: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    actions: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for policy in ["015_common", "016_v2_binary"]:
        for delay in DELAY_TICKS:
            stress = StressConfig(cancel_delay_ticks=delay)
            result = _policy_result(policies, actions, policy, stress)
            rows.append(
                {
                    "policy": policy,
                    "cancel_delay_ticks": delay,
                    "neutral_fill_haircut": 1.0,
                    "good_fill_haircut": 1.0,
                    "toxic_multiplier": 1.0,
                    "net_pnl": result["net_pnl"],
                    "trade_count": result["trade_count"],
                    "max_drawdown": result["max_drawdown"],
                }
            )
    delay_df = pd.DataFrame(rows)
    delay_df.to_csv(out_dir / "delay_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for policy, group in delay_df.groupby("policy", sort=True):
        ax.plot(group["cancel_delay_ticks"], group["net_pnl"], marker="o", label=policy)
    ax.set_xlabel("cancel_delay_ticks")
    ax.set_ylabel("Net PnL")
    ax.set_title("Cancel Delay Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "delay_sensitivity.png", dpi=150)
    plt.close(fig)
    return delay_df


def _decision(base_pnl: float, moderate_pnl: float, no_cancel_pnl: float, random_v3_pnl: float) -> str:
    if moderate_pnl < no_cancel_pnl or moderate_pnl < random_v3_pnl:
        return "Fail"
    if base_pnl > 0 and moderate_pnl >= 0.5 * base_pnl:
        return "Strong Pass"
    return "Partial Pass"


def run_acceptance_summary(
    frame_016: pd.DataFrame,
    policies: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    actions: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    expected = base_expected_values()

    no_cancel_moderate = min(
        float(_policy_result(policies, actions, "no_cancel", StressConfig(neutral_fill_haircut=0.5, toxic_multiplier=1.5, cancel_delay_ticks=d))["net_pnl"])
        for d in MODERATE_DELAY_TICKS
    )
    random_v3_moderate = random_v3_reference_pnl(
        frame_016,
        actions,
        StressConfig(neutral_fill_haircut=0.5, toxic_multiplier=1.5, cancel_delay_ticks=max(MODERATE_DELAY_TICKS)),
    )

    rows = []
    for policy, label in [("015_common", "015_common_universe"), ("016_v2_binary", "016_v2_binary")]:
        moderate_by_delay = []
        for delay in MODERATE_DELAY_TICKS:
            stress = StressConfig(neutral_fill_haircut=0.5, toxic_multiplier=1.5, cancel_delay_ticks=delay)
            result = _policy_result(policies, actions, policy, stress)
            moderate_by_delay.append(float(result["net_pnl"]))
        moderate_pnl = float(min(moderate_by_delay))
        base_pnl = float(expected[policy])
        rows.append(
            {
                "candidate": label,
                "base_pnl": base_pnl,
                "moderate_stress_pnl": moderate_pnl,
                "moderate_stress_min_delay_range": "0_to_2_ticks",
                "moderate_no_cancel_pnl": no_cancel_moderate,
                "moderate_random_v3_mean_pnl": random_v3_moderate,
                "retention_vs_base": float(moderate_pnl / base_pnl) if base_pnl else np.nan,
                "decision": _decision(base_pnl, moderate_pnl, no_cancel_moderate, random_v3_moderate),
                "decision_rule": "Fail if moderate stress PnL is below no-cancel or random_v3; otherwise Strong Pass if it keeps at least 50% of base PnL.",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "queue_stress_acceptance_summary.csv", index=False)
    return summary


def run(out_dir: Path = ROOT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_016, frame_015 = load_frames()
    actions, _ = ensure_random_masks(frame_016, out_dir)
    policies = policy_masks(frame_016, frame_015)

    run_heatmaps(policies, actions, out_dir)
    run_delay_sweep(policies, actions, out_dir)
    run_acceptance_summary(frame_016, policies, actions, out_dir)


if __name__ == "__main__":
    run(ROOT)
    print(f"Stress plots and acceptance summary written to {ROOT}")
