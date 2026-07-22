from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE02_DIR = PROJECT_ROOT / "n2ew_era" / "02_short_horizon_sequential_proxy_baselines"
PHASE02B_DIR = PROJECT_ROOT / "n2ew_era" / "02b_strict_baseline_and_excess_pnl"
for path in [PHASE02_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common_phase02 import iter_event_frames, policy_catalog, policy_decision  # noqa: E402


ROOT = PROJECT_ROOT / "n2ew_era" / "02c_queue_aware_stress"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FILL_KEEP_RATES = [1.0, 0.75, 0.5, 0.25, 0.1]
ADVERSE_MULTIPLIERS = [1.0, 1.25, 1.5, 2.0]
ACTION_DELAYS = [0, 1, 2, 5, 10]
MODERATE_FILL_KEEP = 0.5
MODERATE_ADVERSE = 1.5


@dataclass
class StressState:
    state_count: int = 0
    selected_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0
    cum_pnl: float = 0.0
    peak: float = 0.0
    max_drawdown: float = 0.0


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_candidates() -> list[dict[str, Any]]:
    path = PHASE02B_DIR / "results" / "strict_selected_candidates.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("candidates", [])


def policy_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    lookup = {p["policy_id"]: p for p in policy_catalog()}
    policy = dict(lookup[str(candidate["policy_id"])])
    policy["selected_horizon"] = int(candidate["horizon"])
    policy["phase02b_selection_row"] = candidate
    return policy


def delayed_decision(decision: np.ndarray, delay: int) -> np.ndarray:
    if delay <= 0:
        return decision.copy()
    out = np.zeros_like(decision)
    if delay < len(decision):
        out[delay:] = decision[:-delay]
    return out


def selected_pnl(frame: pd.DataFrame, decision: np.ndarray, horizon: int) -> np.ndarray:
    bid = frame[f"bid_pnl_H{horizon}"].to_numpy(dtype=float)
    ask = frame[f"ask_pnl_H{horizon}"].to_numpy(dtype=float)
    out = np.zeros(len(frame), dtype=float)
    bid_mask = decision == 1
    ask_mask = decision == 2
    out[bid_mask] = np.nan_to_num(bid[bid_mask], nan=0.0)
    out[ask_mask] = np.nan_to_num(ask[ask_mask], nan=0.0)
    return out


def stress_pnl(base_pnl: np.ndarray, fill_keep_rate: float, adverse_multiplier: float) -> np.ndarray:
    return np.where(base_pnl > 0.0, base_pnl * float(fill_keep_rate), base_pnl * float(adverse_multiplier))


def update_state(state: StressState, pnl: np.ndarray, decision: np.ndarray) -> None:
    selected = decision != 0
    state.state_count += int(len(pnl))
    state.selected_count += int(selected.sum())
    selected_pnl = pnl[selected]
    state.win_count += int((selected_pnl > 0.0).sum())
    state.loss_count += int((selected_pnl < 0.0).sum())
    state.total_pnl += float(pnl.sum())
    if len(pnl):
        eq = state.cum_pnl + np.cumsum(pnl)
        running_peak = np.maximum.accumulate(np.maximum(eq, state.peak))
        dd = running_peak - eq
        state.max_drawdown = max(state.max_drawdown, float(dd.max()))
        state.peak = max(state.peak, float(running_peak[-1]))
        state.cum_pnl = float(eq[-1])


def evaluate_policy(policy: dict[str, Any], splits: list[str]) -> pd.DataFrame:
    horizon = int(policy["selected_horizon"])
    rows: list[dict[str, Any]] = []
    for split in splits:
        states = {
            (fill, adv, delay): StressState()
            for fill in FILL_KEEP_RATES
            for adv in ADVERSE_MULTIPLIERS
            for delay in ACTION_DELAYS
        }
        for frame in iter_event_frames(split, horizons=[horizon]):
            base_decision = policy_decision(policy, frame)
            for delay in ACTION_DELAYS:
                dec = delayed_decision(base_decision, delay)
                base = selected_pnl(frame, dec, horizon)
                for fill in FILL_KEEP_RATES:
                    for adv in ADVERSE_MULTIPLIERS:
                        update_state(states[(fill, adv, delay)], stress_pnl(base, fill, adv), dec)
        for (fill, adv, delay), state in states.items():
            pnl_count = max(state.selected_count, 1)
            rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "horizon": horizon,
                    "split": split,
                    "fill_keep_rate": float(fill),
                    "adverse_multiplier": float(adv),
                    "action_delay_events": int(delay),
                    "state_count": int(state.state_count),
                    "selected_action_count": int(state.selected_count),
                    "selection_rate": float(state.selected_count / state.state_count) if state.state_count else 0.0,
                    "stress_total_pnl": float(state.total_pnl),
                    "win_rate": float(state.win_count / pnl_count) if state.selected_count else 0.0,
                    "loss_rate": float(state.loss_count / pnl_count) if state.selected_count else 0.0,
                    "max_drawdown": float(state.max_drawdown),
                    "positive_under_stress": bool(state.total_pnl > 0.0),
                    "moderate_stress": bool(fill == MODERATE_FILL_KEEP and adv == MODERATE_ADVERSE and delay == 0),
                }
            )
        print(f"stress evaluated policy={policy['policy_id']} split={split} H{horizon}")
    return pd.DataFrame(rows)


def plot_outputs(grid: pd.DataFrame, main_policy_id: str, main_horizon: int) -> None:
    if grid.empty:
        return
    main = grid[
        (grid["policy_id"].astype(str) == main_policy_id)
        & (grid["horizon"].astype(int) == int(main_horizon))
        & (grid["action_delay_events"].astype(int) == 0)
    ].copy()
    splits = [s for s in ["test", "new4"] if s in set(main["split"].astype(str))]
    fig, axes = plt.subplots(1, max(1, len(splits)), figsize=(6.2 * max(1, len(splits)), 4.8), squeeze=False)
    for ax, split in zip(axes.ravel(), splits):
        work = main[main["split"].astype(str) == split]
        piv = work.pivot(index="adverse_multiplier", columns="fill_keep_rate", values="stress_total_pnl").sort_index(ascending=True)
        piv = piv.reindex(columns=sorted(piv.columns, reverse=True))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels([f"{r:g}" for r in piv.index])
        ax.set_xlabel("fill_keep_rate")
        ax.set_ylabel("adverse_multiplier")
        ax.set_title(f"{split} stress PnL")
        if MODERATE_FILL_KEEP in piv.columns and MODERATE_ADVERSE in piv.index:
            ax.scatter([list(piv.columns).index(MODERATE_FILL_KEEP)], [list(piv.index).index(MODERATE_ADVERSE)], color="black", marker="x", s=80)
        fig.colorbar(im, ax=ax, label="stress_total_pnl")
    fig.suptitle(f"Queue Stress Heatmap: {main_policy_id}, H{main_horizon}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURES / "queue_stress_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    delay = grid[
        (grid["policy_id"].astype(str) == main_policy_id)
        & (grid["horizon"].astype(int) == int(main_horizon))
        & (grid["fill_keep_rate"].astype(float) == MODERATE_FILL_KEEP)
        & (grid["adverse_multiplier"].astype(float) == MODERATE_ADVERSE)
    ].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for split, group in delay.groupby("split", sort=False):
        group = group.sort_values("action_delay_events")
        ax.plot(group["action_delay_events"], group["stress_total_pnl"], marker="o", label=str(split))
    ax.axhline(0, color="#20242b", linewidth=1.0)
    ax.set_title(f"Delay Sensitivity at Moderate Stress: {main_policy_id}, H{main_horizon}")
    ax.set_xlabel("Action delay events")
    ax.set_ylabel("stress_total_pnl")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "delay_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(candidates: list[dict[str, Any]], grid: pd.DataFrame) -> None:
    lines = [
        "# Queue-Aware Stress Report",
        "",
        "Phase 02C applies proxy queue/fill stress only to policies that passed Phase 02B. It does not implement true queue replay, aggressor-side replay, or order-ID level matching.",
        "",
    ]
    if not candidates:
        lines.extend(
            [
                "No candidate passed Phase 02B. Queue stress was therefore not run on any policy.",
                "",
                "This is intentional: Phase 02C is gated by strict excess-PnL and matched-random hurdles, so failed Phase 02B candidates are not promoted into stress testing.",
            ]
        )
        write_markdown(ROOT / "QUEUE_AWARE_STRESS_REPORT.md", lines)
        return

    main_policy_id = str(candidates[0]["policy_id"])
    main_horizon = int(candidates[0]["horizon"])
    moderate = grid[
        (grid["policy_id"].astype(str) == main_policy_id)
        & (grid["horizon"].astype(int) == main_horizon)
        & (grid["moderate_stress"].astype(bool))
    ].copy()
    break_even_rows = []
    for split in ["test", "new4"]:
        work = grid[
            (grid["policy_id"].astype(str) == main_policy_id)
            & (grid["horizon"].astype(int) == main_horizon)
            & (grid["split"].astype(str) == split)
            & (grid["action_delay_events"].astype(int) == 0)
        ]
        for adv, group in work.groupby("adverse_multiplier"):
            positive = group[group["stress_total_pnl"].astype(float) > 0.0]
            min_keep = float(positive["fill_keep_rate"].min()) if not positive.empty else np.nan
            break_even_rows.append({"split": split, "adverse_multiplier": adv, "min_fill_keep_rate_positive": min_keep})
    break_even = pd.DataFrame(break_even_rows)
    break_even.to_csv(RESULTS / "fill_keep_break_even.csv", index=False)

    lines.extend(
        [
            f"Main stressed candidate: `{main_policy_id}` at horizon `H{main_horizon}`.",
            "",
            "Stress formula:",
            "",
            "```text",
            "if base_pnl > 0: stressed_pnl = base_pnl * fill_keep_rate",
            "if base_pnl < 0: stressed_pnl = base_pnl * adverse_multiplier",
            "action_delay_events shifts the policy decision forward within the session; initial delayed rows are no_quote.",
            "```",
            "",
            "## Moderate Stress",
            "",
            "Moderate stress is `fill_keep_rate = 0.5`, `adverse_multiplier = 1.5`, `action_delay_events = 0`.",
            "",
            moderate.to_markdown(index=False),
            "",
            "## Fill Keep Break-Even",
            "",
            break_even.to_markdown(index=False),
            "",
            "## Outputs",
            "",
            "- `results/queue_stress_grid.csv`",
            "- `results/delay_sensitivity.csv`",
            "- `results/fill_keep_break_even.csv`",
            "- `figures/queue_stress_heatmap.png`",
            "- `figures/delay_sensitivity.png`",
        ]
    )
    write_markdown(ROOT / "QUEUE_AWARE_STRESS_REPORT.md", lines)


def main() -> None:
    ensure_dirs()
    candidates = read_candidates()
    if not candidates:
        pd.DataFrame().to_csv(RESULTS / "queue_stress_grid.csv", index=False)
        pd.DataFrame().to_csv(RESULTS / "delay_sensitivity.csv", index=False)
        write_report(candidates, pd.DataFrame())
        print("No Phase 02B candidates passed; wrote skip report.")
        return

    policies = [policy_from_candidate(c) for c in candidates]
    grid = pd.concat([evaluate_policy(p, ["val", "test", "new4"]) for p in policies], ignore_index=True)
    grid.to_csv(RESULTS / "queue_stress_grid.csv", index=False)
    delay = grid[
        (grid["fill_keep_rate"].astype(float) == MODERATE_FILL_KEEP)
        & (grid["adverse_multiplier"].astype(float) == MODERATE_ADVERSE)
    ].copy()
    delay.to_csv(RESULTS / "delay_sensitivity.csv", index=False)
    plot_outputs(grid, policies[0]["policy_id"], int(policies[0]["selected_horizon"]))
    write_report(candidates, grid)
    print(f"Phase 02C queue stress complete. Wrote {ROOT / 'QUEUE_AWARE_STRESS_REPORT.md'}")


if __name__ == "__main__":
    main()
