from __future__ import annotations

import gc
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07f_sparse_linear_policy"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE07E = PROJECT_ROOT / "n2ew_era" / "07e_async_ml_control_plane"
HELPER_PATH = ROOT / "00_run_sparse_linear_policy.py"

TARGET_TRAIN_HOLD_RATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
BASE_FEATURE_SET = "base"
BASE_C = 0.0003


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_module("hold_bias_helpers_07f", HELPER_PATH)
p05a = helpers.p05a


def selected_coefficients() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "sparse_action_coefficients.csv")
    return frame[
        frame["feature_set"].eq(BASE_FEATURE_SET)
        & np.isclose(frame["C"].astype(float), BASE_C)
    ].copy()


def logits_from_coefficients(payload: dict[str, Any], coefficients: pd.DataFrame) -> np.ndarray:
    x, names = helpers.base_features(payload)
    logits = np.zeros((len(x), 3), dtype=np.float64)
    for action in (0, 1, 2):
        rows = coefficients[coefficients["action"].eq(action)]
        intercept = rows[rows["term"].eq("<intercept>")]
        if len(intercept) != 1:
            raise RuntimeError(f"Expected one intercept for action {action}")
        logits[:, action] = float(intercept.iloc[0]["raw_coefficient"])
        for row in rows[~rows["term"].eq("<intercept>")].itertuples(index=False):
            logits[:, action] += float(row.raw_coefficient) * x[:, names.index(str(row.term))]
    return logits


def bias_grid(train_logits: np.ndarray) -> pd.DataFrame:
    margin = train_logits[:, 0] - np.max(train_logits[:, 1:3], axis=1)
    rows = [{"target_train_hold_rate": float(np.mean(margin >= 0)), "hold_bias_delta": 0.0, "grid_source": "original"}]
    for rate in TARGET_TRAIN_HOLD_RATES:
        cutoff = float(np.quantile(margin, 1.0 - rate))
        rows.append({"target_train_hold_rate": rate, "hold_bias_delta": -cutoff, "grid_source": "train_margin_quantile"})
    frame = pd.DataFrame(rows).sort_values("hold_bias_delta").drop_duplicates("hold_bias_delta").reset_index(drop=True)
    return frame


def decisions_from_bias(logits: np.ndarray, delta: float) -> np.ndarray:
    adjusted = logits.copy()
    adjusted[:, 0] += delta
    return np.argmax(adjusted, axis=1).astype(np.int8)


def main() -> None:
    coefficients = selected_coefficients()
    norms = p05a.compute_fill_norms_from_train_sequences()
    policy = json.loads((PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json").read_text(encoding="utf-8"))
    teacher_threshold = float(policy["threshold"])
    static_theta = float(json.loads((PHASE07E / "results" / "run_metadata.json").read_text(encoding="utf-8"))["selected_static_theta"])

    train_payload = helpers.load_payload("train")
    train_logits = logits_from_coefficients(train_payload, coefficients)
    grid = bias_grid(train_logits)
    grid.to_csv(RESULTS / "hold_bias_grid.csv", index=False)
    del train_payload, train_logits
    gc.collect()

    rows: list[dict[str, Any]] = []
    decision_cache: dict[tuple[str, float], np.ndarray] = {}
    for split in helpers.SPLITS:
        payload = helpers.load_payload(split)
        logits = logits_from_coefficients(payload, coefficients)
        teacher = helpers.teacher_scores(split, payload)
        teacher_decision = p05a.policy_from_scores(teacher, teacher_threshold)
        for candidate in grid.itertuples(index=False):
            delta = float(candidate.hold_bias_delta)
            decision = decisions_from_bias(logits, delta)
            policy_id = f"sparse_action_holdbias_{delta:+.9g}"
            row = helpers.causal_metrics(split, policy_id, payload, decision, norms)
            disagreement = decision != teacher_decision
            row.update({
                "hold_bias_delta": delta,
                "target_train_hold_rate": float(candidate.target_train_hold_rate),
                "grid_source": str(candidate.grid_source),
                "actual_hold_rate": float(np.mean(decision == p05a.DECISION_NO_QUOTE)),
                "action_agreement_rate": float(np.mean(~disagreement)),
                "decision_flip_count": int(disagreement.sum()),
                "fit_split": "train", "selection_split": "val",
                "new3_used_for_selection": False, "new4_used_for_selection": False,
                "new5_used": False,
            })
            rows.append(row)
            decision_cache[(split, delta)] = decision
        del payload, logits, teacher
        gc.collect()

    results = pd.DataFrame(rows)
    val = results[results["split"].eq("val")]
    champion = val.sort_values(
        ["sequential_excess_pnl", "actual_hold_rate"], ascending=[False, True]
    ).iloc[0]
    selected_delta = float(champion["hold_bias_delta"])
    selected = results[np.isclose(results["hold_bias_delta"], selected_delta)].copy()
    references, reference_decisions = helpers.reference_results(teacher_threshold, norms, static_theta)
    selected = selected.merge(
        references.pivot(index="split", columns="policy_id", values="sequential_excess_pnl").reset_index(),
        on="split",
    )
    selected["pnl_retention_vs_phc_pct"] = 100.0 * selected["sequential_excess_pnl"] / selected["phc_eventwise_reference"]
    selected["pnl_delta_vs_static"] = selected["sequential_excess_pnl"] - selected["queue_theta_val_tuned"]

    # Reuse the exact paired-session machinery with an adapter key.
    adapted = {(split, "hold_bias", 0): decision_cache[(split, selected_delta)] for split in helpers.SPLITS}
    champion_adapter = champion.copy()
    champion_adapter["variant"] = "hold_bias"
    champion_adapter["budget_per_output"] = 0
    bootstrap = helpers.paired_bootstrap(champion_adapter, adapted, reference_decisions, norms)

    results.to_csv(RESULTS / "hold_bias_sweep_metrics.csv", index=False)
    selected.to_csv(RESULTS / "hold_bias_selected_results.csv", index=False)
    bootstrap.to_csv(RESULTS / "hold_bias_paired_session_bootstrap.csv", index=False)
    champion_json = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in champion.to_dict().items()}
    (RESULTS / "hold_bias_selected.json").write_text(json.dumps(champion_json, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, split in zip(axes, helpers.SPLITS):
        subset = results[results["split"].eq(split)].sort_values("actual_hold_rate")
        axis.plot(100.0 * subset["actual_hold_rate"], subset["sequential_excess_pnl"], marker="o")
        static_pnl = float(references[(references["split"].eq(split)) & (references["policy_id"].eq("queue_theta_val_tuned"))].iloc[0]["sequential_excess_pnl"])
        phc_pnl = float(references[(references["split"].eq(split)) & (references["policy_id"].eq("phc_eventwise_reference"))].iloc[0]["sequential_excess_pnl"])
        axis.axhline(static_pnl, color="black", linestyle="--", label="static queue")
        axis.axhline(phc_pnl, color="gray", linestyle=":", label="PHC")
        selected_row = selected[selected["split"].eq(split)].iloc[0]
        axis.scatter([100.0 * selected_row["actual_hold_rate"]], [selected_row["sequential_excess_pnl"]], color="red", zorder=5, label="val-selected bias")
        axis.set_title(split)
        axis.set_xlabel("Hold / no-quote rate (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Causal-fill sequential excess PnL (proxy units)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Fixed sparse formula: hold-bias sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES / "hold_bias_pnl_curve.png", dpi=180)
    plt.close(fig)

    val_row = selected[selected["split"].eq("val")].iloc[0]
    oos_rows = selected[selected["split"].isin(["new3", "new4"])]
    all_above_static = bool(val_row["pnl_delta_vs_static"] > 0 and np.all(oos_rows["pnl_delta_vs_static"] > 0))
    status = "resolved_positive" if all_above_static else "mixed"
    lines = [
        "# Hold-Bias Sensitivity of the Sparse Direct-Action Policy\n",
        f"status: `{status}`\n",
        "## Fixed-formula test\n",
        "All five terms and six nonzero class coefficients from `sparse_action_base_C0.0003` are frozen. This experiment changes only the scalar bias added to the `no_quote` logit. Bias candidates correspond to predeclared train hold-rate targets of 10%, 15%, 20%, 25%, 30%, 35%, 40%, and 50%, plus the original zero-bias point. Val selects one bias; NEW3/NEW4 are post-selection diagnostics. NEW5 was not loaded.\n",
        "## Val-selected bias\n",
        f"Val selected hold-bias delta `{selected_delta:.9g}`, corresponding to train target hold rate `{float(champion['target_train_hold_rate']):.2%}`.\n",
        selected[["split", "hold_bias_delta", "target_train_hold_rate", "actual_hold_rate", "action_agreement_rate", "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown", "phc_eventwise_reference", "pnl_retention_vs_phc_pct", "queue_theta_val_tuned", "pnl_delta_vs_static"]].to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Full sweep\n",
        results[["split", "hold_bias_delta", "target_train_hold_rate", "actual_hold_rate", "action_agreement_rate", "sequential_excess_pnl", "fill_count", "avg_pnl_per_fill", "max_drawdown"]].to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Paired session bootstrap\n",
        "The canonical simulator carries inventory across sessions; the bootstrap resets inventory per session. Both estimands are retained.\n",
        bootstrap.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Conclusion\n",
        (
            "A single hold-bias correction is sufficient to keep the fixed sparse formula above the tuned static rule on val, NEW3, and NEW4. The remaining gap to PHC quantifies linear-boundary capacity rather than excessive holding.\n"
            if all_above_static
            else "Changing only the hold bias does not keep the sparse formula above the tuned static rule on all three splits. Excessive holding is therefore not the sole cause; the linear decision boundary itself loses task value.\n"
        ),
    ]
    (ROOT / "AUDIT_sparse_hold_bias.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] selected_delta={selected_delta:.9g} val_pnl={champion['sequential_excess_pnl']:.2f}", flush=True)
    print(f"[OK] report={ROOT / 'AUDIT_sparse_hold_bias.md'}", flush=True)


if __name__ == "__main__":
    main()
