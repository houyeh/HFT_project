from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from queue_stress_core import (
    MAKER_FEE,
    MASK_SEED,
    MAX_INV,
    POINT_VALUE,
    ROOT,
    V16_DIR,
    StressConfig,
    base_expected_values,
    ensure_random_masks,
    load_frames,
    policy_masks,
    simulate_policy,
)


HAIRCUTS = [1.0, 0.75, 0.5, 0.25]
TOXIC_MULTIPLIERS = [1.0, 1.25, 1.5, 2.0]
MODERATE = StressConfig(neutral_fill_haircut=0.5, toxic_multiplier=1.5, cancel_delay_ticks=0)
RANDOM_V3_STRESS_TRIALS = 5000
RANDOM_V3_REPORTING_SEED = MASK_SEED + 77


def _load_strict_random_module():
    sys.path.insert(0, str(V16_DIR))
    path = V16_DIR / "15_strict_random_baseline.py"
    spec = importlib.util.spec_from_file_location("strict_random_for_018_reporting", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strict_random_for_018_reporting"] = module
    spec.loader.exec_module(module)
    return module


def _as_matrix(df: pd.DataFrame, value_col: str) -> np.ndarray:
    matrix = np.zeros((len(HAIRCUTS), len(TOXIC_MULTIPLIERS)), dtype=float)
    for i, haircut in enumerate(HAIRCUTS):
        for j, toxic in enumerate(TOXIC_MULTIPLIERS):
            row = df[(df["neutral_fill_haircut"] == haircut) & (df["toxic_multiplier"] == toxic)].iloc[0]
            matrix[i, j] = float(row[value_col])
    return matrix


def _stress_label(haircut: float, toxic: float) -> str:
    return f"haircut={haircut:g}, toxic={toxic:g}, delay=0"


def _candidate_grid(policies: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]], actions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in ["015_common", "016_v2_binary", "no_cancel"]:
        frame, pull_bid, pull_ask = policies[policy]
        for haircut in HAIRCUTS:
            for toxic in TOXIC_MULTIPLIERS:
                stress = StressConfig(neutral_fill_haircut=haircut, toxic_multiplier=toxic, cancel_delay_ticks=0)
                result = simulate_policy(frame, pull_bid, pull_ask, actions, stress)
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
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "queue_stress_policy_grid.csv", index=False)
    return out


def _random_v3_masks(frame: pd.DataFrame, actions: pd.DataFrame) -> np.ndarray:
    strict = _load_strict_random_module()
    strict_actions, _, _ = strict._build_actions(frame)
    if len(strict_actions) != len(actions):
        raise RuntimeError(f"Action count mismatch: strict={len(strict_actions)} stress={len(actions)}")
    if not np.array_equal(strict_actions["action_id"].to_numpy(dtype=int), actions["action_id"].to_numpy(dtype=int)):
        raise RuntimeError("Action id mismatch between strict random and stress engine.")

    specs, _ = strict._build_standard_specs(frame, strict_actions, method="random_v3", key_cols=["session_id", "spread_bucket"])
    rng = np.random.default_rng(RANDOM_V3_REPORTING_SEED)
    masks = np.zeros((RANDOM_V3_STRESS_TRIALS, len(actions)), dtype=bool)
    for trial in range(RANDOM_V3_STRESS_TRIALS):
        masks[trial] = strict._sample_cancels(specs, rng, len(actions))
    return masks


def _session_action_arrays(frame: pd.DataFrame, actions: pd.DataFrame) -> list[dict[str, Any]]:
    frame_by_index = frame[["mid_price"]].copy()
    future = actions["future_mid_return_H_ticks"].to_numpy(dtype=float)
    side_bid = (actions["side"].astype(str).to_numpy() == "bid")
    favorable = np.where(side_bid, future > 0, future < 0)
    adverse_ticks = np.where(side_bid, np.maximum(0.0, -future), np.maximum(0.0, future))

    action_aug = actions.copy()
    action_aug["side_bid"] = side_bid
    action_aug["favorable"] = favorable
    action_aug["adverse_ticks"] = adverse_ticks
    out: list[dict[str, Any]] = []
    for session_id, group in action_aug.groupby("session_id", sort=True):
        g = group.sort_values(["row_index", "side"]).reset_index(drop=True)
        session_frame = frame[frame["session_id"].astype(str) == str(session_id)].sort_values("local_row_id")
        out.append(
            {
                "session_id": str(session_id),
                "action_id": g["action_id"].to_numpy(dtype=np.int64),
                "side_bid": g["side_bid"].to_numpy(dtype=bool),
                "price": g["price"].to_numpy(dtype=float),
                "neutral_u": g["neutral_u"].to_numpy(dtype=float),
                "good_u": g["good_u"].to_numpy(dtype=float),
                "favorable": g["favorable"].to_numpy(dtype=bool),
                "adverse_ticks": g["adverse_ticks"].to_numpy(dtype=float),
                "final_mid": float(session_frame["mid_price"].iloc[-1]),
                "mid_by_action": frame_by_index.loc[g["row_index"].to_numpy(dtype=np.int64), "mid_price"].to_numpy(dtype=float),
            }
        )
    return out


def _simulate_action_mask(session_arrays: list[dict[str, Any]], canceled: np.ndarray, stress: StressConfig) -> float:
    total = 0.0
    neutral_cut = float(stress.neutral_fill_haircut)
    good_cut = float(stress.good_fill_haircut)
    extra_mult = max(float(stress.toxic_multiplier) - 1.0, 0.0)
    for sess in session_arrays:
        cash = 0.0
        fees = 0.0
        inventory = 0
        action_id = sess["action_id"]
        side_bid = sess["side_bid"]
        price = sess["price"]
        neutral_u = sess["neutral_u"]
        good_u = sess["good_u"]
        favorable = sess["favorable"]
        adverse_ticks = sess["adverse_ticks"]
        for i in range(len(action_id)):
            aid = int(action_id[i])
            if canceled[aid] or neutral_u[i] > neutral_cut:
                continue
            if favorable[i] and good_u[i] > good_cut:
                continue
            if side_bid[i]:
                if inventory >= MAX_INV:
                    continue
                cash -= price[i] * POINT_VALUE
                inventory += 1
            else:
                if inventory <= -MAX_INV:
                    continue
                cash += price[i] * POINT_VALUE
                inventory -= 1
            if extra_mult:
                cash -= extra_mult * adverse_ticks[i] * POINT_VALUE
            fees += MAKER_FEE
        total += cash - fees + inventory * float(sess["final_mid"]) * POINT_VALUE
    return float(total)


def _random_v3_stress_distribution(frame: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    cached = ROOT / "random_v3_stress_distribution.csv"
    if cached.exists():
        existing = pd.read_csv(cached)
        if set(existing["stress_label"]) >= {_stress_label(h, t) for h in HAIRCUTS for t in TOXIC_MULTIPLIERS}:
            return existing

    masks = _random_v3_masks(frame, actions)
    session_arrays = _session_action_arrays(frame, actions)
    strict_dist = pd.read_csv(V16_DIR / "results" / "strict_random_baseline_distribution.csv")
    strict_v3 = strict_dist[strict_dist["method"] == "random_v3"].iloc[0]

    raw_base = np.array([_simulate_action_mask(session_arrays, masks[i], StressConfig()) for i in range(RANDOM_V3_STRESS_TRIALS)], dtype=float)
    mean_offset = float(strict_v3["mean"]) - float(raw_base.mean())

    rows = []
    trial_rows = []
    for haircut in HAIRCUTS:
        for toxic in TOXIC_MULTIPLIERS:
            stress = StressConfig(neutral_fill_haircut=haircut, toxic_multiplier=toxic, cancel_delay_ticks=0)
            raw = np.array([_simulate_action_mask(session_arrays, masks[i], stress) for i in range(RANDOM_V3_STRESS_TRIALS)], dtype=float)
            calibrated = raw + mean_offset
            stress_label = _stress_label(haircut, toxic)
            if stress == StressConfig():
                mean = float(strict_v3["mean"])
                p95 = float(strict_v3["p95"])
            else:
                mean = float(calibrated.mean())
                p95 = float(np.percentile(calibrated, 95))
            rows.append(
                {
                    "policy": "random_v3_mean",
                    "stress_label": stress_label,
                    "neutral_fill_haircut": haircut,
                    "toxic_multiplier": toxic,
                    "cancel_delay_ticks": 0,
                    "trials": RANDOM_V3_STRESS_TRIALS,
                    "raw_mean": float(raw.mean()),
                    "calibration_offset_to_base_mean": mean_offset,
                    "mean": mean,
                    "p95": p95,
                    "p99": float(np.percentile(calibrated, 99)),
                }
            )
            if stress == MODERATE:
                for trial, value in enumerate(calibrated):
                    trial_rows.append({"trial": trial, "stress_label": stress_label, "net_pnl": float(value)})

    dist = pd.DataFrame(rows)
    dist.to_csv(cached, index=False)
    pd.DataFrame(trial_rows).to_csv(ROOT / "random_v3_moderate_trials.csv", index=False)
    return dist


def _update_acceptance(policy_grid: pd.DataFrame, random_dist: pd.DataFrame) -> pd.DataFrame:
    mod_no_cancel = float(
        policy_grid[
            (policy_grid["policy"] == "no_cancel")
            & (policy_grid["neutral_fill_haircut"] == MODERATE.neutral_fill_haircut)
            & (policy_grid["toxic_multiplier"] == MODERATE.toxic_multiplier)
        ].iloc[0]["net_pnl"]
    )
    mod_random = random_dist[
        (random_dist["neutral_fill_haircut"] == MODERATE.neutral_fill_haircut)
        & (random_dist["toxic_multiplier"] == MODERATE.toxic_multiplier)
    ].iloc[0]
    random_mean = float(mod_random["mean"])
    random_p95 = float(mod_random["p95"])

    rows = []
    for policy, label in [("015_common", "015_common_universe"), ("016_v2_binary", "016_v2_binary")]:
        base = float(policy_grid[(policy_grid["policy"] == policy) & (policy_grid["neutral_fill_haircut"] == 1.0) & (policy_grid["toxic_multiplier"] == 1.0)].iloc[0]["net_pnl"])
        moderate = float(
            policy_grid[
                (policy_grid["policy"] == policy)
                & (policy_grid["neutral_fill_haircut"] == MODERATE.neutral_fill_haircut)
                & (policy_grid["toxic_multiplier"] == MODERATE.toxic_multiplier)
            ].iloc[0]["net_pnl"]
        )
        gap_no_cancel = moderate - mod_no_cancel
        gap_random_mean = moderate - random_mean
        gap_random_p95 = moderate - random_p95
        beats_no_cancel = gap_no_cancel > 0
        beats_random_mean = gap_random_mean > 0
        beats_random_p95 = gap_random_p95 > 0
        if moderate <= 0 and beats_no_cancel and (beats_random_mean or beats_random_p95):
            status = "Partial Pass"
        elif moderate > 0 and beats_no_cancel and beats_random_mean:
            status = "Strong Pass"
        else:
            status = "Fail"
        rows.append(
            {
                "candidate": label,
                "moderate_stress_setting": "neutral_fill_haircut=0.5; toxic_multiplier=1.5; cancel_delay_ticks=0",
                "base_pnl": base,
                "moderate_stress_pnl": moderate,
                "moderate_no_cancel_pnl": mod_no_cancel,
                "moderate_random_v3_mean_pnl": random_mean,
                "moderate_random_v3_95p_pnl": random_p95,
                "gap_vs_no_cancel": gap_no_cancel,
                "gap_vs_random_v3_mean": gap_random_mean,
                "gap_vs_random_v3_95p": gap_random_p95,
                "beats_no_cancel": beats_no_cancel,
                "beats_random_v3_mean": beats_random_mean,
                "beats_random_v3_95p": beats_random_p95,
                "absolute_profitability_pass": bool(moderate > 0),
                "decision_status": status,
                "decision_logic": "Partial Pass: moderate_stress_pnl <= 0 but beats no_cancel and beats random_v3_mean or random_v3_95p.",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "queue_stress_acceptance_summary.csv", index=False)
    return summary


def _break_even_frontier(policy_grid: pd.DataFrame, random_dist: pd.DataFrame) -> pd.DataFrame:
    random_grid = random_dist.rename(columns={"mean": "net_pnl"})[
        ["policy", "neutral_fill_haircut", "toxic_multiplier", "cancel_delay_ticks", "net_pnl"]
    ]
    combined = pd.concat(
        [
            policy_grid[["policy", "neutral_fill_haircut", "toxic_multiplier", "cancel_delay_ticks", "net_pnl"]],
            random_grid,
        ],
        ignore_index=True,
    )
    rows = []
    for policy in ["015_common", "016_v2_binary", "no_cancel", "random_v3_mean"]:
        for toxic in TOXIC_MULTIPLIERS:
            sub = combined[(combined["policy"] == policy) & (combined["toxic_multiplier"] == toxic)].sort_values(
                "neutral_fill_haircut", ascending=False
            )
            positives = sub[sub["net_pnl"] > 0]
            negatives = sub[sub["net_pnl"] <= 0]
            if positives.empty:
                last_positive = "NONE"
                last_positive_pnl = np.nan
                min_haircut = np.nan
            else:
                last_pos = positives.sort_values("neutral_fill_haircut", ascending=True).iloc[0]
                last_positive = _stress_label(float(last_pos["neutral_fill_haircut"]), toxic)
                last_positive_pnl = float(last_pos["net_pnl"])
                min_haircut = float(last_pos["neutral_fill_haircut"])
            if negatives.empty:
                first_negative = "NONE"
                first_negative_pnl = np.nan
            else:
                neg = negatives.sort_values("neutral_fill_haircut", ascending=False).iloc[0]
                first_negative = _stress_label(float(neg["neutral_fill_haircut"]), toxic)
                first_negative_pnl = float(neg["net_pnl"])
            rows.append(
                {
                    "policy": policy,
                    "toxic_multiplier": toxic,
                    "minimum_neutral_fill_haircut_to_stay_positive": min_haircut,
                    "last_positive_region": last_positive,
                    "last_positive_pnl": last_positive_pnl,
                    "first_negative_region": first_negative,
                    "first_negative_pnl": first_negative_pnl,
                }
            )
    frontier = pd.DataFrame(rows)
    frontier.to_csv(ROOT / "break_even_frontier.csv", index=False)
    _write_break_even_md(frontier)
    return frontier


def _write_break_even_md(frontier: pd.DataFrame) -> None:
    lines = [
        "# Break-Even Frontier",
        "",
        "Grid definition: `neutral_fill_haircut in {1.0, 0.75, 0.5, 0.25}`, `toxic_multiplier in {1.0, 1.25, 1.5, 2.0}`, and `cancel_delay_ticks=0`.",
        "",
        "The frontier is descriptive. It is not a new model-selection pass and it does not tune thresholds on test data.",
        "",
        "## Summary",
        "",
    ]
    for policy in ["015_common", "016_v2_binary", "no_cancel", "random_v3_mean"]:
        lines.append(f"### {policy}")
        sub = frontier[frontier["policy"] == policy]
        for _, row in sub.iterrows():
            min_haircut = row["minimum_neutral_fill_haircut_to_stay_positive"]
            min_text = "NONE on this grid" if pd.isna(min_haircut) else f"{float(min_haircut):.2f}"
            lines.append(
                f"- toxic={float(row['toxic_multiplier']):.2f}: min positive haircut `{min_text}`; "
                f"last positive `{row['last_positive_region']}`; first negative `{row['first_negative_region']}`."
            )
        lines.append("")
    (ROOT / "break_even_frontier.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_matrix(matrix: np.ndarray, title: str, path: Path, *, label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = None
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(TOXIC_MULTIPLIERS)))
    ax.set_xticklabels([str(v) for v in TOXIC_MULTIPLIERS])
    ax.set_yticks(np.arange(len(HAIRCUTS)))
    ax.set_yticklabels([str(v) for v in HAIRCUTS])
    ax.set_xlabel("toxic_multiplier")
    ax.set_ylabel("neutral_fill_haircut")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            ax.text(x, y, f"{value:,.0f}", ha="center", va="center", fontsize=8, color="black")
    moderate_x = TOXIC_MULTIPLIERS.index(1.5)
    moderate_y = HAIRCUTS.index(0.5)
    ax.scatter([moderate_x], [moderate_y], s=240, marker="o", facecolors="none", edgecolors="black", linewidths=2.2)
    ax.text(moderate_x, moderate_y + 0.34, "moderate", ha="center", va="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _final_plots(policy_grid: pd.DataFrame, random_dist: pd.DataFrame) -> None:
    no_cancel = policy_grid[policy_grid["policy"] == "no_cancel"].copy()
    random_mean = random_dist.rename(columns={"mean": "random_v3_mean_pnl"}).copy()
    for policy, name, pnl_path, no_cancel_path, random_path in [
        ("015_common", "015_common", "queue_stress_heatmap_015_common.png", "queue_stress_gap_vs_no_cancel_015.png", "queue_stress_gap_vs_random_v3_015.png"),
        ("016_v2_binary", "016_v2_binary", "queue_stress_heatmap_016_v2_binary.png", "queue_stress_gap_vs_no_cancel_016.png", "queue_stress_gap_vs_random_v3_016.png"),
    ]:
        sub = policy_grid[policy_grid["policy"] == policy].copy()
        pnl_matrix = _as_matrix(sub, "net_pnl")
        _plot_matrix(pnl_matrix, f"{name}: Net PnL Stress Map", ROOT / pnl_path, label="Net PnL (red is negative)")

        gap_no = sub.merge(
            no_cancel[["neutral_fill_haircut", "toxic_multiplier", "net_pnl"]].rename(columns={"net_pnl": "no_cancel_pnl"}),
            on=["neutral_fill_haircut", "toxic_multiplier"],
            how="left",
        )
        gap_no["gap"] = gap_no["net_pnl"] - gap_no["no_cancel_pnl"]
        _plot_matrix(_as_matrix(gap_no, "gap"), f"{name}: Gap vs No-Cancel", ROOT / no_cancel_path, label="PnL gap vs no-cancel")

        gap_random = sub.merge(
            random_mean[["neutral_fill_haircut", "toxic_multiplier", "random_v3_mean_pnl"]],
            on=["neutral_fill_haircut", "toxic_multiplier"],
            how="left",
        )
        gap_random["gap"] = gap_random["net_pnl"] - gap_random["random_v3_mean_pnl"]
        _plot_matrix(_as_matrix(gap_random, "gap"), f"{name}: Gap vs Matched Random v3 Mean", ROOT / random_path, label="PnL gap vs random_v3_mean")


def _delay_note() -> None:
    delay = pd.read_csv(ROOT / "delay_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for policy, group in delay.groupby("policy", sort=True):
        ax.plot(group["cancel_delay_ticks"], group["net_pnl"], marker="o", label=policy)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("cancel_delay_ticks")
    ax.set_ylabel("Net PnL")
    ax.set_title("Delay Sensitivity")
    ax.text(
        0.5,
        -0.22,
        "Delay sensitivity is used only to motivate latency risk-control. Do not claim FPGA necessity is proven.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "delay_sensitivity.png", dpi=160)
    plt.close(fig)
    (ROOT / "delay_sensitivity.md").write_text(
        "# Delay Sensitivity Caveat\n\n"
        "Delay sensitivity is used only to motivate latency risk-control. Do not claim FPGA necessity is proven.\n",
        encoding="utf-8",
    )


def _universe_alignment_md() -> None:
    alignment = pd.read_csv(ROOT / "015_common_audit" / "universe_alignment_report.csv")
    common = alignment[alignment["track"] == "015_common_universe"].iloc[0]
    text = f"""# 015 Original vs 015 Common Universe

The 015 audit is intentionally reported as two tracks.

- `015_original_universe`: preserves the historical 015 audit result and its original event universe.
- `015_common_universe`: re-scores the frozen 015 model on the 016_v2 quote-eligible test universe for apples-to-apples 018 queue stress.

The PnL moves from `170,625` on `015_original_universe` to `90,845` on `015_common_universe`.
This drop is due to the evaluation universe changing for the apples-to-apples 018 comparison, not necessarily a degradation in the 015 model itself.

Common-universe coverage:

- original_event_count: `{int(common['original_event_count'])}`
- common_universe_event_count: `{int(common['common_universe_event_count'])}`
- overlap_count: `{int(common['overlap_count'])}`
- coverage_rate: `{float(common['coverage_rate']):.6f}`
- missing_prediction_rate: `{float(common['missing_prediction_rate']):.6f}`
- dropped_event_count: `{int(common['dropped_event_count'])}`
"""
    (ROOT / "universe_alignment_report.md").write_text(text, encoding="utf-8")


def _write_report(summary: pd.DataFrame, frontier: pd.DataFrame, policy_grid: pd.DataFrame, random_dist: pd.DataFrame) -> None:
    expected = base_expected_values()
    no_mod = float(summary.iloc[0]["moderate_no_cancel_pnl"])
    rand_mod_mean = float(summary.iloc[0]["moderate_random_v3_mean_pnl"])
    rand_mod_p95 = float(summary.iloc[0]["moderate_random_v3_95p_pnl"])
    row015 = summary[summary["candidate"] == "015_common_universe"].iloc[0]
    row016 = summary[summary["candidate"] == "016_v2_binary"].iloc[0]

    text = f"""# 018 Queue-Aware Stress Test Report

## 1. Purpose (Not a profitability test)

018 is a stress audit of frozen candidate cancel policies. It is not a production profitability claim and it does not add features, change thresholds, or select a new model on test outcomes. The purpose is to ask whether defensive cancel edge survives when queue/fill assumptions become less favorable.

## 2. Pre-018 Freeze

Both frozen candidates passed the 13-item freeze checklist in `pre_018_model_freeze_report.csv`. The common test sessions are `20260127Day` and `20260127Night`; the common quote-eligible universe has `1,166,321` rows. The stress engine also reproduces the frozen base PnL exactly in `stress_reproducibility_check.csv`.

## 3. Stress Definitions

The moderate setting used for the main reporting claim is `neutral_fill_haircut=0.5`, `toxic_multiplier=1.5`, and `cancel_delay_ticks=0`.

- `neutral_fill_haircut`: uniformly keeps only a fixed fraction of realized maker fills.
- `good_fill_haircut`: preferentially removes favorable fills and is reserved for pessimistic appendix analysis.
- `toxic_multiplier`: increases adverse fill damage after a fill.
- `cancel_delay_ticks`: delays cancel effectiveness by event ticks.

Fixed random masks are recorded in `random_mask_manifest.json`.

## 4. Base PnL

- no_cancel: `{expected['no_cancel']:,.0f}`
- matched random_v3 mean: `{expected['random_v3_mean']:,.3f}`
- 015_common_universe: `{expected['015_common']:,.0f}`
- 016_v2_binary: `{expected['016_v2_binary']:,.0f}`

The original historical 015 result remains `170,625`; the common-universe 018 result is `90,845` because the evaluation universe was changed for apples-to-apples comparison.

## 5. Moderate Stress Result

Moderate stress makes absolute profitability fail, but relative defense survives.

- no_cancel moderate PnL: `{no_mod:,.0f}`
- random_v3_mean moderate PnL: `{rand_mod_mean:,.3f}`
- random_v3_95p moderate PnL: `{rand_mod_p95:,.3f}`
- 015_common moderate PnL: `{float(row015['moderate_stress_pnl']):,.0f}`; gap vs no-cancel `{float(row015['gap_vs_no_cancel']):,.0f}`; gap vs random_v3_mean `{float(row015['gap_vs_random_v3_mean']):,.3f}`; gap vs random_v3_95p `{float(row015['gap_vs_random_v3_95p']):,.3f}`.
- 016_v2_binary moderate PnL: `{float(row016['moderate_stress_pnl']):,.0f}`; gap vs no-cancel `{float(row016['gap_vs_no_cancel']):,.0f}`; gap vs random_v3_mean `{float(row016['gap_vs_random_v3_mean']):,.3f}`; gap vs random_v3_95p `{float(row016['gap_vs_random_v3_95p']):,.3f}`.

The relative pass is against no-cancel and random_v3_mean. Neither candidate clears the random_v3_95p hurdle under moderate stress.

Decision status: both candidates are `Partial Pass`.

## 6. Heatmap / Break-even Frontier

Report-ready figures:

- `queue_stress_heatmap_015_common.png`
- `queue_stress_heatmap_016_v2_binary.png`
- `queue_stress_gap_vs_no_cancel_015.png`
- `queue_stress_gap_vs_no_cancel_016.png`
- `queue_stress_gap_vs_random_v3_015.png`
- `queue_stress_gap_vs_random_v3_016.png`

The break-even table is in `break_even_frontier.csv` and the narrative summary is in `break_even_frontier.md`. The main visual message is that the green absolute-PnL region is fragile under moderate queue/fill stress, while positive relative gaps can remain.

## 7. Delay Sensitivity (FPGA framing caveat)

Delay sensitivity is used only to motivate latency risk-control. Do not claim FPGA necessity is proven.

The delay sweep remains diagnostic: 016_v2_binary declines from `132,590` at delay 0 to `81,520` at delay 10, while 015_common is non-monotone but still latency-sensitive. This supports studying low-latency risk-control paths, not claiming a proven production FPGA requirement.

## 8. Final Interpretation

Under moderate queue/fill stress, both 015_common and 016_v2_binary fail the absolute profitability criterion but retain relative defensive value against no-cancel and matched random baselines. Therefore, 018 is a Partial Pass...

This result should be presented as a conservative robustness audit: the cancel policies are not proof of real-world HFT profitability, but they do preserve defensive signal value under a harsher proxy execution framework. The next step is queue-aware sensitivity and execution realism, not model expansion.
"""
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")


def _write_slide_update(summary: pd.DataFrame) -> None:
    row015 = summary[summary["candidate"] == "015_common_universe"].iloc[0]
    row016 = summary[summary["candidate"] == "016_v2_binary"].iloc[0]
    text = f"""# Slide Update 018

## Slide A: Why queue-aware stress?

We froze the 015 and 016_v2 candidates, then weakened the optimistic proxy execution assumptions. The goal is not to prove profitability; it is to test whether cancel signals still reduce downside versus no-cancel and matched random baselines when queue priority, adverse selection, and cancel timing are less favorable.

## Slide B: Moderate stress result

Conclusion: Partial Pass. Profitability does not survive moderate stress, but the defensive edge does.

- 015_common moderate PnL: `{float(row015['moderate_stress_pnl']):,.0f}`, gap vs no-cancel `{float(row015['gap_vs_no_cancel']):,.0f}`, gap vs random_v3_mean `{float(row015['gap_vs_random_v3_mean']):,.0f}`.
- 016_v2_binary moderate PnL: `{float(row016['moderate_stress_pnl']):,.0f}`, gap vs no-cancel `{float(row016['gap_vs_no_cancel']):,.0f}`, gap vs random_v3_mean `{float(row016['gap_vs_random_v3_mean']):,.0f}`.

Message: absolute PnL fails; relative defensive cancel value survives.

## Slide C: Conservative FPGA framing

Latency matters for risk-control, but FPGA necessity is not proven by this audit. The delay sensitivity result motivates low-latency cancel-risk controls as a research direction; it does not establish production PnL or a hardware requirement.
"""
    (ROOT / "slide_update_018.md").write_text(text, encoding="utf-8")


def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    frame_016, frame_015 = load_frames()
    actions, _ = ensure_random_masks(frame_016, ROOT)
    policies = policy_masks(frame_016, frame_015)

    policy_grid = _candidate_grid(policies, actions)
    random_dist = _random_v3_stress_distribution(frame_016, actions)
    summary = _update_acceptance(policy_grid, random_dist)
    frontier = _break_even_frontier(policy_grid, random_dist)
    _final_plots(policy_grid, random_dist)
    _delay_note()
    _universe_alignment_md()
    _write_report(summary, frontier, policy_grid, random_dist)
    _write_slide_update(summary)

    manifest = {
        "moderate_setting": {
            "neutral_fill_haircut": MODERATE.neutral_fill_haircut,
            "toxic_multiplier": MODERATE.toxic_multiplier,
            "cancel_delay_ticks": MODERATE.cancel_delay_ticks,
        },
        "random_v3_stress_trials": RANDOM_V3_STRESS_TRIALS,
        "random_v3_reporting_seed": RANDOM_V3_REPORTING_SEED,
        "note": "Final reporting only; no model or feature changes.",
    }
    (ROOT / "final_reporting_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Final 018 reporting package written to {ROOT}")
