from __future__ import annotations

import gc
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07g_tiny_mlp_policy"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE07E = PROJECT_ROOT / "n2ew_era" / "07e_async_ml_control_plane"
PHASE07F = PROJECT_ROOT / "n2ew_era" / "07f_sparse_linear_policy"

WIDTHS = (4, 8)
OBJECTIVES = ("direct_action", "score_regression")
SEEDS = (0, 1, 2)
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
MIN_DELTA = 1e-6


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_module("tiny_mlp_helpers_07g", PHASE07F / "00_run_sparse_linear_policy.py")
p05a = helpers.p05a


@dataclass(frozen=True)
class Candidate:
    objective: str
    width: int
    seed: int

    @property
    def candidate_id(self) -> str:
        return f"tiny_mlp_{self.objective}_h{self.width}_seed{self.seed}"


class TinyMLP(torch.nn.Module):
    def __init__(self, input_dim: int, width: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def weighted_loss(
    objective: str, prediction: torch.Tensor, target: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    if objective == "direct_action":
        values = torch.nn.functional.cross_entropy(prediction, target.long(), reduction="none")
    else:
        values = torch.mean((prediction - target) ** 2, dim=1)
    if weights is None:
        return values.mean()
    return torch.sum(values * weights) / torch.sum(weights)


def train_candidate(
    candidate: Candidate,
    x_train: np.ndarray,
    action_train: np.ndarray,
    score_train: np.ndarray,
    sample_weights: np.ndarray,
    x_val: np.ndarray,
    action_val: np.ndarray,
    score_val: np.ndarray,
) -> tuple[TinyMLP, list[dict[str, Any]], dict[str, Any]]:
    set_seed(candidate.seed)
    output_dim = 3 if candidate.objective == "direct_action" else 2
    model = TinyMLP(x_train.shape[1], candidate.width, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_x = torch.from_numpy(x_train.astype(np.float32, copy=False))
    train_action = torch.from_numpy(action_train.astype(np.int64, copy=False))
    train_score = torch.from_numpy(score_train.astype(np.float32, copy=False))
    train_weight = torch.from_numpy(sample_weights.astype(np.float32, copy=False))
    val_x = torch.from_numpy(x_val.astype(np.float32, copy=False))
    val_action = torch.from_numpy(action_val.astype(np.int64, copy=False))
    val_score = torch.from_numpy(score_val.astype(np.float32, copy=False))
    target_train = train_action if candidate.objective == "direct_action" else train_score
    target_val = val_action if candidate.objective == "direct_action" else val_score

    generator = torch.Generator().manual_seed(candidate.seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        weighted_sum = 0.0
        weight_total = 0.0
        for begin in range(0, len(train_x), BATCH_SIZE):
            index = permutation[begin : begin + BATCH_SIZE]
            batch_weight = train_weight[index]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[index])
            loss = weighted_loss(candidate.objective, prediction, target_train[index], batch_weight)
            loss.backward()
            optimizer.step()
            batch_weight_sum = float(batch_weight.sum())
            weighted_sum += float(loss.detach()) * batch_weight_sum
            weight_total += batch_weight_sum

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_loss = float(weighted_loss(candidate.objective, val_prediction, target_val, None))
        history.append({
            **asdict(candidate),
            "candidate_id": candidate.candidate_id,
            "epoch": epoch,
            "train_loss": weighted_sum / weight_total,
            "val_loss": val_loss,
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint captured for {candidate.candidate_id}")
    model.load_state_dict(best_state)
    metadata = {
        **asdict(candidate),
        "candidate_id": candidate.candidate_id,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_loss": best_loss,
        "early_stopped": len(history) < MAX_EPOCHS,
        "wall_seconds": time.perf_counter() - started,
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, MODELS / f"{candidate.candidate_id}.pt")
    return model, history, metadata


def predict(
    model: TinyMLP, objective: str, standardized_x: np.ndarray, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(standardized_x.astype(np.float32, copy=False))).numpy()
    if objective == "direct_action":
        decision = np.argmax(output, axis=1).astype(np.int8)
    else:
        decision = p05a.policy_from_scores(output, threshold).astype(np.int8)
    return output.astype(np.float64), decision


def main() -> None:
    for directory in (RESULTS, MODELS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, int(torch.get_num_threads()))))

    selected_policy = json.loads(
        (PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json").read_text(encoding="utf-8")
    )
    threshold = float(selected_policy["threshold"])
    static_theta = float(json.loads(
        (PHASE07E / "results" / "run_metadata.json").read_text(encoding="utf-8")
    )["selected_static_theta"])
    norms = p05a.compute_fill_norms_from_train_sequences()

    train_payload = helpers.load_payload("train")
    val_payload = helpers.load_payload("val")
    train_raw, feature_names = helpers.base_features(train_payload)
    val_raw, _ = helpers.base_features(val_payload)
    mean = train_raw.mean(axis=0)
    scale = train_raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_x = ((train_raw - mean) / scale).astype(np.float32)
    val_x = ((val_raw - mean) / scale).astype(np.float32)
    train_scores = helpers.teacher_scores("train", train_payload)
    val_scores = helpers.teacher_scores("val", val_payload)
    train_actions = p05a.policy_from_scores(train_scores, threshold)
    val_actions = p05a.policy_from_scores(val_scores, threshold)
    weights, boundary_scale = helpers.boundary_weights(train_scores, threshold)

    candidates: dict[str, tuple[Candidate, TinyMLP]] = {}
    curve_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        for width in WIDTHS:
            for seed in SEEDS:
                candidate = Candidate(objective, width, seed)
                model, history, metadata = train_candidate(
                    candidate, train_x, train_actions, train_scores, weights,
                    val_x, val_actions, val_scores,
                )
                candidates[candidate.candidate_id] = (candidate, model)
                training_rows.extend(history)
                curve_rows.append(metadata)
                print(
                    f"[fit] {candidate.candidate_id} params={metadata['parameter_count']} "
                    f"best_epoch={metadata['best_epoch']} val_loss={metadata['best_val_loss']:.8g}",
                    flush=True,
                )
    pd.DataFrame(training_rows).to_csv(RESULTS / "tiny_mlp_training_curves.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(RESULTS / "tiny_mlp_training_summary.csv", index=False)

    del train_raw, train_x, train_payload, train_scores, train_actions
    gc.collect()

    metric_rows: list[dict[str, Any]] = []
    decision_cache: dict[tuple[str, str], np.ndarray] = {}
    for split in helpers.SPLITS:
        payload = val_payload if split == "val" else helpers.load_payload(split)
        raw_x, _ = helpers.base_features(payload)
        standardized = ((raw_x - mean) / scale).astype(np.float32)
        teacher_scores = helpers.teacher_scores(split, payload)
        teacher_decision = p05a.policy_from_scores(teacher_scores, threshold)
        for candidate_id, (candidate, model) in candidates.items():
            output, decision = predict(model, candidate.objective, standardized, threshold)
            row = helpers.causal_metrics(split, candidate_id, payload, decision, norms)
            disagreement = decision != teacher_decision
            row.update({
                **asdict(candidate),
                "candidate_id": candidate_id,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "action_agreement_rate": float(np.mean(~disagreement)),
                "decision_flip_count": int(disagreement.sum()),
                "score_r2_uniform": (
                    float(r2_score(teacher_scores, output, multioutput="uniform_average"))
                    if candidate.objective == "score_regression" else np.nan
                ),
                "teacher_hold_rate": float(np.mean(teacher_decision == p05a.DECISION_NO_QUOTE)),
                "student_hold_rate": float(np.mean(decision == p05a.DECISION_NO_QUOTE)),
                "fit_split": "train",
                "selection_split": "val",
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used": False,
            })
            metric_rows.append(row)
            decision_cache[(split, candidate_id)] = decision
        if split != "val":
            del payload
        gc.collect()
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "tiny_mlp_metrics.csv", index=False)

    champion = metrics[metrics["split"] == "val"].sort_values(
        ["sequential_excess_pnl", "parameter_count", "candidate_id"],
        ascending=[False, True, True],
    ).iloc[0]
    champion_id = str(champion["candidate_id"])
    references, reference_decisions = helpers.reference_results(threshold, norms, static_theta)
    selected = metrics[metrics["candidate_id"] == champion_id].merge(
        references.pivot(index="split", columns="policy_id", values="sequential_excess_pnl").reset_index(),
        on="split",
    )
    selected["pnl_retention_vs_phc_pct"] = (
        100.0 * selected["sequential_excess_pnl"] / selected["phc_eventwise_reference"]
    )
    selected["pnl_delta_vs_static"] = (
        selected["sequential_excess_pnl"] - selected["queue_theta_val_tuned"]
    )
    selected.to_csv(RESULTS / "tiny_mlp_selected_results.csv", index=False)

    bootstrap_decisions = {
        (split, champion_id, 0): decision_cache[(split, champion_id)] for split in helpers.SPLITS
    }
    bootstrap_champion = pd.Series({
        "variant": champion_id,
        "budget_per_output": 0,
        "policy_id": champion_id,
    })
    bootstrap = helpers.paired_bootstrap(
        bootstrap_champion, bootstrap_decisions, reference_decisions, norms
    )
    bootstrap.to_csv(RESULTS / "tiny_mlp_paired_session_bootstrap.csv", index=False)

    champion_model = candidates[champion_id][1]
    champion_record = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in champion.to_dict().items()
    }
    champion_record.update({
        "threshold": threshold,
        "feature_names": feature_names,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "boundary_weight_scale": boundary_scale,
        "state_dict": {
            name: value.detach().numpy().tolist()
            for name, value in champion_model.state_dict().items()
        },
    })
    (RESULTS / "tiny_mlp_selected.json").write_text(
        json.dumps(champion_record, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for axis, split in zip(axes, helpers.SPLITS):
        subset = metrics[metrics["split"] == split]
        for objective, marker in (("direct_action", "o"), ("score_regression", "s")):
            objective_rows = subset[subset["objective"] == objective]
            grouped = objective_rows.groupby("width")["sequential_excess_pnl"].agg(["mean", "std"])
            axis.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker=marker, label=objective)
        static_value = float(references[
            (references["split"] == split) & (references["policy_id"] == "queue_theta_val_tuned")
        ]["sequential_excess_pnl"].iloc[0])
        phc_value = float(references[
            (references["split"] == split) & (references["policy_id"] == "phc_eventwise_reference")
        ]["sequential_excess_pnl"].iloc[0])
        axis.axhline(static_value, color="black", linestyle="--", label="static queue")
        axis.axhline(phc_value, color="gray", linestyle=":", label="PHC")
        axis.set_title(split)
        axis.set_xlabel("Hidden width")
        axis.set_xticks(WIDTHS)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Causal-fill sequential excess PnL (proxy units)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Tiny MLP distillation: mean +/- seed standard deviation")
    fig.tight_layout()
    fig.savefig(FIGURES / "tiny_mlp_pnl_comparison.png", dpi=180)
    plt.close(fig)

    val_selected = selected[selected["split"] == "val"].iloc[0]
    oos_selected = selected[selected["split"].isin(["new3", "new4"])]
    positive = bool(
        val_selected["pnl_retention_vs_phc_pct"] >= 80.0
        and val_selected["pnl_delta_vs_static"] > 0
        and np.all(oos_selected["pnl_delta_vs_static"] > 0)
    )
    mixed = bool(
        val_selected["pnl_delta_vs_static"] > 0
        or np.any(oos_selected["pnl_delta_vs_static"] > 0)
    )
    status = "resolved_positive" if positive else ("mixed" if mixed else "resolved_negative")
    aggregate = metrics.groupby(["split", "objective", "width"], as_index=False).agg(
        mean_pnl=("sequential_excess_pnl", "mean"),
        std_pnl=("sequential_excess_pnl", "std"),
        mean_action_agreement=("action_agreement_rate", "mean"),
        mean_hold_rate=("student_hold_rate", "mean"),
        parameter_count=("parameter_count", "first"),
    )
    aggregate.to_csv(RESULTS / "tiny_mlp_seed_summary.csv", index=False)
    lines = [
        "# Tiny MLP Policy Distillation Audit\n",
        f"status: `{status}`\n",
        "## Claim boundary\n",
        "This is a CPU-only feasibility experiment. The frozen 05C4 seed-2 PHC is not retrained. Train fits the tiny MLPs, val selects one exact objective/width/seed candidate, and NEW3/NEW4 are post-selection diagnostics. NEW5 was not loaded. No HLS, routed timing, power, or on-board claim is made.\n",
        "## Design\n",
        f"The fixed candidate grid is hidden width `{list(WIDTHS)}`, objective `{list(OBJECTIVES)}`, and seed `{list(SEEDS)}`. Every network has one ReLU hidden layer. `direct_action` predicts buy/sell/hold with boundary-weighted cross entropy; `score_regression` predicts the two frozen PHC scores with boundary-weighted MSE and applies the unchanged PHC threshold. Training uses Adam at `{LEARNING_RATE}`, batch size `{BATCH_SIZE}`, at most `{MAX_EPOCHS}` epochs, and val-loss early stopping with patience `{PATIENCE}`.\n",
        "## Val-selected champion\n",
        f"Val selected `{champion_id}` with `{int(champion['parameter_count'])}` trainable parameters.\n",
        selected[[
            "split", "candidate_id", "objective", "width", "seed", "parameter_count",
            "action_agreement_rate", "student_hold_rate", "sequential_excess_pnl", "fill_count",
            "avg_pnl_per_fill", "max_drawdown", "phc_eventwise_reference",
            "pnl_retention_vs_phc_pct", "queue_theta_val_tuned", "pnl_delta_vs_static",
        ]].to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Seed stability and capacity sweep\n",
        aggregate.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## All candidates\n",
        metrics.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Paired session bootstrap\n",
        "The canonical simulator carries inventory across sessions; the bootstrap resets inventory per session. Both estimands are retained.\n",
        bootstrap.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Decision\n",
        (
            "The tiny MLP clears the predeclared task-level gate: val retains at least 80% of PHC PnL, and the val-selected candidate beats the tuned static rule on val, NEW3, and NEW4. It is worth a separate fixed-point/HLS feasibility phase.\n"
            if positive
            else "The tiny MLP does not clear the full task-level gate. Its result still identifies whether a small nonlinear boundary improves on the sparse linear and shallow-rule baselines, but it is not yet a drop-in replacement for PHC.\n"
        ),
    ]
    (ROOT / "AUDIT_tiny_mlp_policy.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[OK] champion={champion_id} val_pnl={champion['sequential_excess_pnl']:.2f} "
        f"status={status}",
        flush=True,
    )
    print(f"[OK] report={ROOT / 'AUDIT_tiny_mlp_policy.md'}", flush=True)


if __name__ == "__main__":
    main()
