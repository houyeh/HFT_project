from __future__ import annotations

import copy
import json
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common_034 import FIGURES, PROJECT_ROOT, RESULTS, ROOT, V28, V30, ensure_dirs, fmt_money, set_style, write_markdown


COMMON_030_DIR = PROJECT_ROOT / "new_era" / "30_ml_training_adequacy_audit"
if str(COMMON_030_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_030_DIR))

from common_030 import FEATURE_COLUMNS, MAX_VETO_RATE, evaluate_veto, load_all_splits, threshold_curve  # noqa: E402


MODEL_NAME = "small_mlp_64_32_epoch_rerun"
RANDOM_STATE = 20260734
MAX_EPOCHS = 250
PATIENCE = 25
LEARNING_RATE = 1e-3


def search_and_copy_existing_logs() -> list[Path]:
    recovered_dir = RESULTS / "recovered_logs"
    recovered_dir.mkdir(parents=True, exist_ok=True)
    patterns = ["*loss*.csv", "*history*.csv"]
    found: list[Path] = []
    for root in [V28, V30]:
        for pattern in patterns:
            for src in root.rglob(pattern):
                if src.is_file():
                    dst = recovered_dir / f"{root.name}__{src.name}"
                    shutil.copyfile(src, dst)
                    found.append(dst)
    return found


def matrix(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    if fit:
        if scaler is None:
            raise ValueError("scaler required when fit=True")
        x = scaler.fit_transform(x)
    elif scaler is not None:
        x = scaler.transform(x)
    y = df["negative_ev_moderate"].astype(int).to_numpy()
    return x.astype(np.float64), y


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan


def safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan


def train_epoch_mlp(frames: dict[str, pd.DataFrame]) -> tuple[MLPClassifier, StandardScaler, pd.DataFrame, dict]:
    scaler = StandardScaler()
    x_train, y_train = matrix(frames["train"], scaler, fit=True)
    x_val, y_val = matrix(frames["val"], scaler, fit=False)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=LEARNING_RATE,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    classes = np.asarray([0, 1], dtype=int)
    history: list[dict] = []
    best_model = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_left = PATIENCE
    for epoch in range(1, MAX_EPOCHS + 1):
        clf.partial_fit(x_train, y_train, classes=classes)
        train_score = clf.predict_proba(x_train)[:, 1]
        val_score = clf.predict_proba(x_val)[:, 1]
        train_loss = float(log_loss(y_train, train_score, labels=[0, 1]))
        val_loss = float(log_loss(y_val, val_score, labels=[0, 1]))
        val_pr = safe_pr_auc(y_val, val_score)
        val_roc = safe_roc_auc(y_val, val_score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc_negative_ev": val_pr,
                "val_roc_auc_negative_ev": val_roc,
                "learning_rate": LEARNING_RATE,
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "best_epoch_so_far": best_epoch if best_model is not None else epoch,
            }
        )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(clf)
            patience_left = PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    if best_model is None:
        best_model = copy.deepcopy(clf)
        best_epoch = len(history)
        best_val_loss = float(history[-1]["val_loss"])
    hist = pd.DataFrame(history)
    hist["selected_checkpoint"] = hist["epoch"].eq(best_epoch)
    meta = {
        "model_name": MODEL_NAME,
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "early_stopped": bool(len(hist) < MAX_EPOCHS),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "new4_used_for_selection": False,
    }
    return best_model, scaler, hist, meta


def add_scores(df: pd.DataFrame, model: MLPClassifier, scaler: StandardScaler) -> pd.DataFrame:
    x, _ = matrix(df, scaler, fit=False)
    out = df.copy()
    out["negative_ev_score"] = model.predict_proba(x)[:, 1]
    out["model_name"] = MODEL_NAME
    return out


def select_threshold_on_val_scored(val_scored: pd.DataFrame) -> dict:
    curve = threshold_curve(val_scored, "negative_ev_score", max_veto_rate=MAX_VETO_RATE)
    curve.to_csv(RESULTS / f"{MODEL_NAME}_validation_threshold_curve.csv", index=False)
    eligible = curve[curve["constraints_pass"].astype(bool)].copy()
    if eligible.empty:
        eligible = curve.copy()
    selected = eligible.sort_values(["after_veto_pnl", "veto_count"], ascending=[False, True]).iloc[0].to_dict()
    selected.update({"selection_split": "validation", "test_used_for_selection": False, "max_veto_rate": MAX_VETO_RATE})
    return selected


def frozen_incremental(frames: dict[str, pd.DataFrame], model: MLPClassifier, scaler: StandardScaler, threshold: float) -> pd.DataFrame:
    rows = []
    pred_parts = []
    for split in ["val", "test", "new4"]:
        scored = add_scores(frames[split], model, scaler)
        pred_parts.append(scored.assign(phase034_split=split))
        row = evaluate_veto(scored, "negative_ev_score", threshold)
        row.update(
            {
                "split": split,
                "model_name": MODEL_NAME,
                "threshold": float(threshold),
                "selection_split": "validation",
                "test_used_for_selection": False,
                "new4_used_for_selection": False,
                "accepted_useful_ml": bool(row["after_veto_pnl"] > row["baseline_pnl"]),
            }
        )
        rows.append(row)
    pd.concat(pred_parts, ignore_index=True).to_csv(RESULTS / f"{MODEL_NAME}_predictions.csv", index=False)
    out = pd.DataFrame(rows)
    out["beats_non_ml_baseline_proxy_pnl"] = out["after_veto_pnl"] > out["baseline_pnl"]
    out["accepted_useful_ml_both_oos"] = bool(
        out.loc[out["split"].isin(["test", "new4"]), "beats_non_ml_baseline_proxy_pnl"].all()
    )
    out.to_csv(RESULTS / "frozen_incremental_value_from_rerun.csv", index=False)
    return out


def plot_curves(history: pd.DataFrame, frozen: pd.DataFrame) -> None:
    set_style()
    best_epoch = int(history.loc[history["selected_checkpoint"], "epoch"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(history["epoch"], history["train_loss"], label="train loss", color="#276ef1")
    ax.plot(history["epoch"], history["val_loss"], label="val loss", color="#c43c35")
    ax.axvline(best_epoch, color="#111827", linestyle="--", lw=1.2, label=f"selected epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.set_title("Residual EV small_mlp_64_32 epoch loss curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_epoch_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(history["epoch"], history["val_pr_auc_negative_ev"], color="#586b8f", marker="o", markersize=2.4, label="val PR-AUC")
    ax.plot(history["epoch"], history["val_roc_auc_negative_ev"], color="#2f9e44", marker="s", markersize=2.4, label="val ROC-AUC")
    ax.axvline(best_epoch, color="#111827", linestyle="--", lw=1.2, label=f"selected epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation metric")
    ax.set_title("Validation PR-AUC and ROC-AUC curves for negative EV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_epoch_metric_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    gap = history["val_loss"] - history["train_loss"]
    axes[0].plot(history["epoch"], gap, color="#7a4cc2")
    axes[0].axhline(0, color="#111827", lw=1)
    axes[0].axvline(best_epoch, color="#111827", linestyle="--", lw=1.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("val_loss - train_loss")
    axes[0].set_title("Overfit gap")
    bars = axes[1].bar(frozen["split"], frozen["incremental_vs_baseline"], color=["#586b8f", "#c43c35", "#c43c35"])
    axes[1].axhline(0, color="#111827", lw=1)
    axes[1].set_ylabel("Incremental PnL vs baseline")
    axes[1].set_title("Frozen incremental value")
    for bar, value in zip(bars, frozen["incremental_vs_baseline"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, fmt_money(value), ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("ML overfit and OOS transfer diagnostic", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_overfit_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_reports(
    history: pd.DataFrame,
    meta: dict,
    threshold: dict,
    frozen: pd.DataFrame,
    recovered_logs: list[Path],
    model: MLPClassifier,
    scaler: StandardScaler,
) -> None:
    with (RESULTS / f"{MODEL_NAME}_model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "scaler": scaler, "metadata": meta}, handle)
    (RESULTS / f"{MODEL_NAME}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    oos = frozen[frozen["split"].isin(["test", "new4"])].copy()
    accepted = bool(oos["beats_non_ml_baseline_proxy_pnl"].all())
    lines = [
        "# ML Epoch Curve Recovery Report",
        "",
        "Step 1 searched Phase 28 and Phase 30 for existing `*loss*.csv` or `*history*.csv` files.",
        "",
        f"- Recovered existing logs: `{len(recovered_logs)}`.",
    ]
    if recovered_logs:
        lines.extend([f"- `{path}`" for path in recovered_logs])
    else:
        lines.append("- No usable epoch logs were found, so a diagnostic rerun was performed.")
    lines.extend(
        [
            "",
            "## Diagnostic Rerun",
            "",
            f"- Model: `{MODEL_NAME}`.",
            "- Universe: Phase 030 residual baseline universe, equivalent to the Phase 028 `bid_only_spread_ge_1` universe.",
            "- Features: causal residual EV features from `common_028.FEATURE_COLUMNS`.",
            "- Early stopping: validation log loss only.",
            f"- Selected checkpoint epoch: `{meta['best_epoch']}`.",
            f"- Best validation loss: `{meta['best_val_loss']:.6f}`.",
            f"- Learning rate: `{meta['learning_rate']}`.",
            f"- Train samples: `{meta['train_samples']}`.",
            f"- Validation samples: `{meta['val_samples']}`.",
            "- Test and NEW4 were not used for tuning, threshold selection, or checkpoint selection.",
            "",
            "## Validation Threshold",
            "",
            f"- Threshold selected on validation: `{float(threshold['threshold']):.8f}`.",
            f"- Validation veto count: `{int(threshold['veto_count'])}`.",
            f"- Validation incremental PnL: `{fmt_money(float(threshold['incremental_vs_baseline']))}`.",
            "",
            "## Frozen Incremental Value",
            "",
        ]
    )
    for row in frozen.itertuples(index=False):
        lines.append(
            f"- `{row.split}`: baseline `{fmt_money(row.baseline_pnl)}`, after veto `{fmt_money(row.after_veto_pnl)}`, incremental `{fmt_money(row.incremental_vs_baseline)}`, beats baseline `{bool(row.beats_non_ml_baseline_proxy_pnl)}`."
        )
    lines.extend(
        [
            "",
            f"Accepted useful ML on both OOS splits: `{accepted}`.",
            "",
            "Conclusion: the real epoch curves improve transparency of the rejected ML route. They do not revive ML unless frozen test and NEW4 both beat the non-ML baseline proxy PnL.",
            "",
            "Figures:",
            "",
            "- `figures/ml_epoch_loss_curves.png`",
            "- `figures/ml_epoch_metric_curves.png`",
            "- `figures/ml_overfit_diagnostic.png`",
        ]
    )
    write_markdown(ROOT / "ML_EPOCH_CURVE_RECOVERY_REPORT.md", lines)
    write_markdown(
        ROOT / "ml_model_results_explanation.md",
        [
            "# ML Model Results Explanation",
            "",
            "The residual ML route was rerun with epoch-level diagnostics after the first 034 report found that earlier phases did not save epoch logs.",
            "",
            "Rerun setup:",
            "",
            "- Model: `small_mlp_64_32_epoch_rerun`.",
            "- Universe: the same residual baseline universe used by Phase 028/030.",
            "- Features: causal tabular features only.",
            "- Checkpoint selection: validation log loss with early stopping.",
            "- Threshold selection: validation only.",
            "- Frozen evaluation: test and NEW4 exactly after validation selection.",
            "",
            "Observed outcome:",
            "",
            *[
                f"- `{row.split}`: baseline `{fmt_money(row.baseline_pnl)}`, after ML veto `{fmt_money(row.after_veto_pnl)}`, incremental `{fmt_money(row.incremental_vs_baseline)}`."
                for row in frozen.itertuples(index=False)
            ],
            "",
            f"Accepted useful ML on both OOS splits: `{accepted}`.",
            "",
            "Conservative interpretation: the real epoch curves show that the residual ML route was tested transparently. The route remains rejected unless frozen OOS incremental PnL is positive on both test and NEW4.",
            "",
            "Figures:",
            "",
            "- `figures/ml_epoch_loss_curves.png`",
            "- `figures/ml_epoch_metric_curves.png`",
            "- `figures/ml_overfit_diagnostic.png`",
        ],
    )
    write_markdown(
        RESULTS / "ml_training_curves_missing.md",
        [
            "# ML Training Curves Recovered",
            "",
            "Earlier Phase 28/30 epoch logs were not found, so Phase 034 performed a diagnostic rerun and produced real epoch-level curves.",
            "",
            "Use these figures instead of the old placeholder:",
            "",
            "- `figures/ml_epoch_loss_curves.png`",
            "- `figures/ml_epoch_metric_curves.png`",
            "- `figures/ml_overfit_diagnostic.png`",
        ],
    )


def main() -> None:
    ensure_dirs()
    (RESULTS / "recovered_logs").mkdir(parents=True, exist_ok=True)
    recovered_logs = search_and_copy_existing_logs()
    frames = load_all_splits()
    model, scaler, history, meta = train_epoch_mlp(frames)
    history.to_csv(RESULTS / f"{MODEL_NAME}_epoch_history.csv", index=False)
    val_scored = add_scores(frames["val"], model, scaler)
    selected = select_threshold_on_val_scored(val_scored)
    frozen = frozen_incremental(frames, model, scaler, float(selected["threshold"]))
    plot_curves(history, frozen)
    meta["selected_threshold"] = float(selected["threshold"])
    meta["selected_val_veto_count"] = int(selected["veto_count"])
    write_reports(history, meta, selected, frozen, recovered_logs, model, scaler)


if __name__ == "__main__":
    main()
    print(f"Wrote ML epoch curve recovery outputs to {ROOT}")
