from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PHASE05A = ROOT.parent / "05a_phc_lite_microstructure_sequence_feasibility"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402
import common_05c2 as c05c2  # noqa: E402
from models.phc_lite_stp import PHCLiteSTP  # noqa: E402


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_stats(split: str, group_type: str, labels: np.ndarray, values: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    labels = np.asarray(labels).astype(str)
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        v = values[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        rows.append(
            {
                "split": split,
                "group_type": group_type,
                "group_value": label,
                "num_samples": int(len(v)),
                "gain_mean": float(np.mean(v)),
                "gain_std": float(np.std(v)),
                "gain_min": float(np.min(v)),
                "gain_q01": float(np.quantile(v, 0.01)),
                "gain_q05": float(np.quantile(v, 0.05)),
                "gain_q25": float(np.quantile(v, 0.25)),
                "gain_q50": float(np.quantile(v, 0.50)),
                "gain_q75": float(np.quantile(v, 0.75)),
                "gain_q95": float(np.quantile(v, 0.95)),
                "gain_q99": float(np.quantile(v, 0.99)),
                "gain_max": float(np.max(v)),
            }
        )
    return rows


def qbucket(values: np.ndarray, prefix: str) -> np.ndarray:
    s = pd.Series(values)
    try:
        out = pd.qcut(s.rank(method="first"), q=5, labels=[f"{prefix}_q{i}" for i in range(1, 6)])
        return out.astype(str).to_numpy()
    except ValueError:
        return np.array([f"{prefix}_single"] * len(values), dtype=object)


def compute_gain_state_means(model: torch.nn.Module, payload: dict[str, Any], batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    x = payload["x"]
    means = []
    last_means = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(dtype=torch.float32)
            _, gains = model.forward_with_gains(xb)
            g = gains.detach().cpu().numpy().astype(np.float32)
            means.append(g.mean(axis=(1, 2)))
            last_means.append(g[:, -1, :].mean(axis=1))
    return np.concatenate(means), np.concatenate(last_means)


def gain_diagnostics() -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model, ckpt, selected = c05c2.selected_model()
    rows = []
    hist_data = {}
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decision, _ = c05c2.stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        state_gain_mean, last_gain_mean = compute_gain_state_means(model, payload)
        hist_data[split] = state_gain_mean
        raw = payload["target_raw"].numpy()
        selected_mask = decision != p05a.DECISION_NO_QUOTE
        selected_pnl = pnl[selected_mask]
        outcome = np.array(["non_selected"] * len(decision), dtype=object)
        outcome[selected_mask & (pnl > 0)] = "selected_positive"
        outcome[selected_mask & (pnl <= 0)] = "selected_negative"
        rows.extend(group_stats(split, "overall_state_mean_gain", np.array(["all"] * len(state_gain_mean)), state_gain_mean))
        rows.extend(group_stats(split, "overall_last_timestep_gain", np.array(["all"] * len(last_gain_mean)), last_gain_mean))
        rows.extend(group_stats(split, "selected_status", np.where(selected_mask, "selected", "non_selected"), state_gain_mean))
        rows.extend(group_stats(split, "selected_outcome", outcome, state_gain_mean))
        rows.extend(group_stats(split, "event_rate_bucket", qbucket(raw[:, p05a.TARGET_RAW_COLUMNS.index("event_rate_past_50")], "event_rate"), state_gain_mean))
        rows.extend(group_stats(split, "volatility_bucket", qbucket(raw[:, p05a.TARGET_RAW_COLUMNS.index("volatility_past_50")], "volatility"), state_gain_mean))
        rows.extend(group_stats(split, "imbalance_change_bucket", qbucket(raw[:, p05a.TARGET_RAW_COLUMNS.index("imbalance_change_past")], "imbalance_change"), state_gain_mean))
        if len(selected_pnl):
            rows.append(
                {
                    "split": split,
                    "group_type": "selected_outcome_pnl_context",
                    "group_value": "selected_actions",
                    "num_samples": int(len(selected_pnl)),
                    "gain_mean": float(state_gain_mean[selected_mask].mean()),
                    "gain_std": float(state_gain_mean[selected_mask].std()),
                    "gain_min": float(state_gain_mean[selected_mask].min()),
                    "gain_q01": float(np.quantile(state_gain_mean[selected_mask], 0.01)),
                    "gain_q05": float(np.quantile(state_gain_mean[selected_mask], 0.05)),
                    "gain_q25": float(np.quantile(state_gain_mean[selected_mask], 0.25)),
                    "gain_q50": float(np.quantile(state_gain_mean[selected_mask], 0.50)),
                    "gain_q75": float(np.quantile(state_gain_mean[selected_mask], 0.75)),
                    "gain_q95": float(np.quantile(state_gain_mean[selected_mask], 0.95)),
                    "gain_q99": float(np.quantile(state_gain_mean[selected_mask], 0.99)),
                    "gain_max": float(state_gain_mean[selected_mask].max()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_stp_gain_statistics.csv", index=False)
    plt.figure(figsize=(9, 5))
    for split, values in hist_data.items():
        plt.hist(values, bins=80, alpha=0.45, density=True, label=split)
    plt.title("STP-like Gain Distribution by Split")
    plt.xlabel("state-level mean gain_t")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "phc_stp_gain_distribution.png", dpi=160)
    plt.close()
    overall = out[out["group_type"].eq("overall_state_mean_gain")]
    selected = out[out["group_type"].eq("selected_status")]
    regime = out[out["group_type"].isin(["event_rate_bucket", "volatility_bucket", "imbalance_change_bucket"])]
    nontrivial = bool((overall["gain_std"] > 0.005).any())
    regime_range = regime.groupby(["split", "group_type"])["gain_mean"].agg(lambda s: float(s.max() - s.min())).reset_index(name="gain_mean_range")
    regime_responsive = bool((regime_range["gain_mean_range"] > 0.005).any())
    write_markdown(
        ROOT / "PHC_STP_GAIN_BEHAVIOR_REPORT.md",
        [
            "# PHC STP-like Gain Behavior Report",
            "",
            "Gain diagnostics use the validation-selected STP-like model and exclude NEW5.",
            "",
            f"Is gain_t non-trivial by state-level std > 0.005? `{nontrivial}`.",
            f"Does gain_t vary across regime buckets by mean range > 0.005? `{regime_responsive}`.",
            "",
            "## Overall Gain Statistics",
            "",
            overall.to_markdown(index=False),
            "",
            "## Selected vs Non-selected",
            "",
            selected.to_markdown(index=False),
            "",
            "## Regime Mean Ranges",
            "",
            regime_range.to_markdown(index=False),
            "",
            "Interpretation: gain_t is diagnostic only. STP is not adopted unless it matches or beats frozen PHC-lite v1 on OOS, causal-fill, and oracle capture.",
        ],
    )
    return out


def session_paired() -> pd.DataFrame:
    v1 = pd.read_csv(PHASE05A / "results" / "phc_lite_session_pnl.csv")
    stp = pd.read_csv(RESULTS / "phc_stp_session_pnl.csv")
    v1 = v1[v1["policy_id"].eq("phc_lite")].copy()
    stp = stp[stp["policy_id"].eq("phc_stp")].copy()
    stp["variant"] = "stp_50epoch"
    variants = [stp]

    extended_policy = RESULTS / "phc_stp_extended_selected_policy.json"
    if extended_policy.exists():
        import json

        selected = json.loads(extended_policy.read_text(encoding="utf-8"))
        ckpt = torch.load(ROOT / "models" / f"phc_stp_extended_seed{int(selected['seed'])}_best_val.pt", map_location="cpu", weights_only=False)
        model = PHCLiteSTP(int(ckpt["feature_dim"]), hidden_dim=int(ckpt["hidden_dim"]), mtl_steps=int(ckpt["mtl_steps"]), dropout=0.0)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        rows = []
        for split in ["val", "new3", "new4"]:
            payload = p05a.load_sequences(split)
            scores = c05c2.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
            decision = p05a.policy_from_scores(scores, float(selected["threshold"]))
            pnl = p05a.decision_pnl_from_payload(payload, decision)
            code = payload["session_code"].numpy()
            for session_code, session_id in enumerate(payload["session_labels"]):
                mask = code == session_code
                rows.append(
                    {
                        "split": split,
                        "policy_id": "phc_stp_extended80_selected",
                        "session_id": session_id,
                        "session_excess_pnl_H20": float(pnl[mask].sum()),
                        "selected_action_count": int((decision[mask] != p05a.DECISION_NO_QUOTE).sum()),
                        "state_count": int(mask.sum()),
                        "variant": "stp_extended80_selected",
                    }
                )
        variants.append(pd.DataFrame(rows))

    stp_all = pd.concat(variants, ignore_index=True)
    paired = stp_all.merge(v1, on=["split", "session_id"], suffixes=("_stp", "_v1"))
    paired["pnl_delta_stp_minus_v1"] = paired["session_excess_pnl_H20_stp"] - paired["session_excess_pnl_H20_v1"]
    paired["stp_wins_session"] = paired["pnl_delta_stp_minus_v1"] >= 0
    paired["selected_count_delta_stp_minus_v1"] = paired["selected_action_count_stp"] - paired["selected_action_count_v1"]
    paired.to_csv(RESULTS / "phc_stp_vs_v1_session_paired.csv", index=False)
    summary = paired.groupby(["variant", "split"]).agg(
        sessions=("session_id", "count"),
        stp_winning_sessions=("stp_wins_session", "sum"),
        mean_delta=("pnl_delta_stp_minus_v1", "mean"),
        median_delta=("pnl_delta_stp_minus_v1", "median"),
        total_delta=("pnl_delta_stp_minus_v1", "sum"),
        delta_std=("pnl_delta_stp_minus_v1", "std"),
    ).reset_index()
    write_markdown(
        ROOT / "PHC_STP_SESSION_PAIRED_COMPARISON.md",
        [
            "# PHC STP-like vs PHC-lite v1 Session Paired Comparison",
            "",
            "Session paired comparison uses val/NEW3/NEW4 only. NEW5 is not used.",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Session Rows",
            "",
            paired.to_markdown(index=False),
            "",
            "STP should only be kept as a future regime-specific option if it wins identifiable regimes without degrading OOS and causal-fill. Current adoption still requires matching or beating v1.",
        ],
    )
    return paired


def main() -> None:
    gain_diagnostics()
    session_paired()
    print("gain_and_session_diagnostics_complete")


if __name__ == "__main__":
    main()
