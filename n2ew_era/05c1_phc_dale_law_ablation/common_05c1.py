from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05c1_phc_dale_law_ablation"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C0 = PROJECT_ROOT / "n2ew_era" / "05c0_freeze_phc_lite_v1"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402
from models.phc_lite_dale import PHCLiteDale, count_parameters  # noqa: E402


SEEDS = [0, 1, 2]
EXCITATORY_RATIO = 0.8
HIDDEN_DIM = 16
MTL_STEPS = 1
MAX_EPOCHS = 50
BATCH_SIZE = 4096
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(p05a.json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def predict_scores(model: PHCLiteDale, payload: dict[str, Any], target_scale: float, batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        x = payload["x"]
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0) * float(target_scale)


def load_model(checkpoint_path: Path) -> tuple[PHCLiteDale, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PHCLiteDale(
        int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        mtl_steps=int(ckpt["mtl_steps"]),
        dropout=float(ckpt.get("dropout", 0.0)),
        excitatory_ratio=float(ckpt.get("excitatory_ratio", EXCITATORY_RATIO)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def train_seed(seed: int) -> pd.DataFrame:
    ensure_dirs()
    set_seed(seed)
    scaler = p05a.load_scaler()
    target_scale = float(scaler["target_scale_train_std_no_mean_shift"])
    train_payload = p05a.load_sequences("train")
    val_payload = p05a.load_sequences("val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PHCLiteDale(
        len(p05a.PHC_FEATURES),
        hidden_dim=HIDDEN_DIM,
        mtl_steps=MTL_STEPS,
        dropout=0.0,
        excitatory_ratio=EXCITATORY_RATIO,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    huber = torch.nn.HuberLoss(delta=1.0)
    train_loader = torch.utils.data.DataLoader(
        p05a.SequenceTensorDataset(train_payload, target_scale),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    best_metric = -float("inf")
    best_epoch = -1
    early_counter = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        grad_norms = []
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = huber(pred, yb)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            opt.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(grad_norm)
        train_scores = predict_scores(model, train_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
        val_scores = predict_scores(model, val_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
        train_y = np.column_stack([train_payload["y_bid"].numpy(), train_payload["y_ask"].numpy()])
        val_y = np.column_stack([val_payload["y_bid"].numpy(), val_payload["y_ask"].numpy()])
        val_decision = p05a.policy_from_scores(val_scores, 0.0)
        val_pnl = float(p05a.decision_pnl_from_payload(val_payload, val_decision).sum())
        val_selected = int((val_decision != p05a.DECISION_NO_QUOTE).sum())
        row = {
            "epoch": epoch,
            "seed": seed,
            "train_loss": float(np.mean(losses)),
            "val_loss": p05a.huber_np((val_scores - val_y) / target_scale),
            "train_huber_loss": p05a.huber_np((train_scores - train_y) / target_scale),
            "val_huber_loss": p05a.huber_np((val_scores - val_y) / target_scale),
            "train_spearman_bid": p05a.safe_spearman(train_scores[:, 0], train_y[:, 0]),
            "val_spearman_bid": p05a.safe_spearman(val_scores[:, 0], val_y[:, 0]),
            "train_spearman_ask": p05a.safe_spearman(train_scores[:, 1], train_y[:, 1]),
            "val_spearman_ask": p05a.safe_spearman(val_scores[:, 1], val_y[:, 1]),
            "val_policy_excess_pnl": val_pnl,
            "val_selected_count": val_selected,
            "val_selection_rate": val_selected / len(val_decision) if len(val_decision) else 0.0,
            "val_pr_auc_bid": p05a.safe_pr_auc(val_y[:, 0] > 0, val_scores[:, 0]),
            "val_pr_auc_ask": p05a.safe_pr_auc(val_y[:, 1] > 0, val_scores[:, 1]),
            "learning_rate": LR,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "early_stop_counter": early_counter,
            "max_epoch_hit": False,
            "parameter_count": count_parameters(model),
            "device": device,
            "hidden_dim": HIDDEN_DIM,
            "mtl_steps": MTL_STEPS,
            "excitatory_ratio": EXCITATORY_RATIO,
        }
        rows.append(row)
        if val_pnl > best_metric + 1e-3:
            best_metric = val_pnl
            best_epoch = epoch
            early_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "feature_dim": len(p05a.PHC_FEATURES),
                "hidden_dim": HIDDEN_DIM,
                "mtl_steps": MTL_STEPS,
                "dropout": 0.0,
                "excitatory_ratio": EXCITATORY_RATIO,
                "seed": seed,
                "best_epoch": best_epoch,
                "target_scale": target_scale,
                "selection_metric": "validation_policy_excess_pnl_at_threshold_0",
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used_for_selection": False,
            }
            torch.save(checkpoint, MODELS / f"phc_dale_seed{seed}_best_val.pt")
        else:
            early_counter += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_dim": len(p05a.PHC_FEATURES),
                "hidden_dim": HIDDEN_DIM,
                "mtl_steps": MTL_STEPS,
                "dropout": 0.0,
                "excitatory_ratio": EXCITATORY_RATIO,
                "seed": seed,
                "epoch": epoch,
                "target_scale": target_scale,
            },
            MODELS / f"phc_dale_seed{seed}_last.pt",
        )
        print(f"dale seed={seed} epoch={epoch} val_pnl={val_pnl:.2f} selected={val_selected} grad={row['grad_norm']:.4f}")
        if early_counter >= PATIENCE:
            break
    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve.loc[curve.index[-1], "max_epoch_hit"] = int(curve["epoch"].iloc[-1]) >= MAX_EPOCHS
    curve.to_csv(RESULTS / f"dale_training_curve_seed{seed}.csv", index=False)
    plot_curve(curve, seed)
    write_training_report(curve, seed)
    return curve


def plot_curve(curve: pd.DataFrame, seed: int) -> None:
    if curve.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["train_loss"], label="train")
    plt.plot(curve["epoch"], curve["val_loss"], label="val")
    plt.title(f"Dale PHC Loss Curve seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"dale_loss_curve_seed{seed}.png", dpi=160)
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["val_policy_excess_pnl"], marker="o")
    plt.title(f"Dale PHC Validation Policy PnL seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Val policy excess PnL")
    plt.tight_layout()
    plt.savefig(FIGURES / f"dale_val_policy_pnl_seed{seed}.png", dpi=160)
    plt.close()


def write_training_report(curve: pd.DataFrame, seed: int) -> None:
    best = curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
    max_epoch_hit = bool(int(curve["epoch"].iloc[-1]) >= MAX_EPOCHS)
    still_improving = bool(float(curve["val_policy_excess_pnl"].iloc[-1]) >= float(best["val_policy_excess_pnl"]) - 1e-6)
    write_markdown(
        ROOT / f"PHC_DALE_TRAINING_CONVERGENCE_REPORT_seed{seed}.md",
        [
            f"# PHC Dale Training Convergence Report seed {seed}",
            "",
            f"Selected epoch: `{int(best['epoch'])}` by validation policy excess PnL.",
            f"Best validation policy PnL: `{float(best['val_policy_excess_pnl']):,.2f}`.",
            f"Train loss decreased: `{bool(curve['train_loss'].iloc[-1] < curve['train_loss'].iloc[0])}`.",
            f"Validation loss decreased or plateaued: `{bool(curve['val_loss'].iloc[-1] <= curve['val_loss'].iloc[0])}`.",
            f"Max epoch hit: `{max_epoch_hit}`.",
            f"Validation PnL still at/near high-water mark on final epoch: `{still_improving}`.",
            f"Gradient explosion detected: `{bool(curve['grad_norm'].max() > 100.0)}`.",
            "",
            "NEW3/NEW4/NEW5 were not used for epoch selection or threshold selection.",
        ],
    )


def threshold_grid(scores: np.ndarray) -> list[float]:
    max_scores = np.max(scores, axis=1)
    finite = max_scores[np.isfinite(max_scores)]
    q = np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if len(finite) else np.array([0.0])
    return sorted(set([0.0, *q.tolist()]))


def select_seed_and_threshold() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val = p05a.load_sequences("val")
    for seed in SEEDS:
        ckpt_path = MODELS / f"phc_dale_seed{seed}_best_val.pt"
        if not ckpt_path.exists():
            train_seed(seed)
        model, ckpt = load_model(ckpt_path)
        scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
        for threshold in threshold_grid(scores):
            decision = p05a.policy_from_scores(scores, float(threshold))
            row = p05a.metric_from_decision("val", f"phc_dale_seed{seed}", val, decision)
            row.update(
                {
                    "seed": seed,
                    "threshold": float(threshold),
                    "selection_split": "val",
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "phc_dale_validation_threshold_search.csv", index=False)
    best = grid.sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).iloc[0].to_dict()
    selected = {
        "model": "phc_lite_dale",
        "variant": "PHC-lite + Dale's Law",
        "selection_split": "val",
        "seed": int(best["seed"]),
        "threshold": float(best["threshold"]),
        "target": "excess_pnl_H20",
        "seq_len": 128,
        "stride": 32,
        "hidden_dim": HIDDEN_DIM,
        "mtl_steps": MTL_STEPS,
        "excitatory_ratio": EXCITATORY_RATIO,
        "validation_metrics": p05a.json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    write_json(RESULTS / "phc_dale_selected_policy.json", selected)
    return grid, selected


def dale_weight_sign_audit(checkpoint_path: Path | None = None) -> pd.DataFrame:
    if checkpoint_path is None:
        selected = read_json(RESULTS / "phc_dale_selected_policy.json")
        checkpoint_path = MODELS / f"phc_dale_seed{int(selected['seed'])}_best_val.pt"
    model, ckpt = load_model(checkpoint_path)
    weight = model.synapse.effective_weight().detach().cpu().numpy()
    sign = model.synapse.dale_sign.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for j, s in enumerate(sign):
        col = weight[:, j]
        if s > 0:
            violations = int((col < -1e-12).sum())
        else:
            violations = int((col > 1e-12).sum())
        rows.append(
            {
                "hidden_dim": model.hidden_dim,
                "presynaptic_unit": j,
                "unit_type": "excitatory" if s > 0 else "inhibitory",
                "sign_value": float(s),
                "outgoing_weight_min": float(col.min()),
                "outgoing_weight_max": float(col.max()),
                "sign_violations": violations,
                "constraint_enforced_during_training": True,
            }
        )
    out = pd.DataFrame(rows)
    out["excitatory_units"] = int((sign > 0).sum())
    out["inhibitory_units"] = int((sign < 0).sum())
    out.to_csv(RESULTS / "dale_weight_sign_audit.csv", index=False)
    total_violations = int(out["sign_violations"].sum())
    write_markdown(
        ROOT / "DALE_WEIGHT_SIGN_AUDIT.md",
        [
            "# Dale Weight Sign Audit",
            "",
            "Dale's Law is enforced on columns of the synapse matrix because `message = z @ W_eff.T + bias`; column `j` is the outgoing fan-out from presynaptic hidden unit `j`.",
            "",
            f"Hidden dim: `{model.hidden_dim}`.",
            f"Excitatory units: `{int((sign > 0).sum())}`.",
            f"Inhibitory units: `{int((sign < 0).sum())}`.",
            f"Total sign violations: `{total_violations}`.",
            "",
            out.to_markdown(index=False),
        ],
    )
    if total_violations != 0:
        raise RuntimeError("Dale sign audit failed.")
    return out


def selected_model() -> tuple[PHCLiteDale, dict[str, Any], dict[str, Any]]:
    selected = read_json(RESULTS / "phc_dale_selected_policy.json")
    model, ckpt = load_model(MODELS / f"phc_dale_seed{int(selected['seed'])}_best_val.pt")
    return model, ckpt, selected


def dale_decision_for_payload(payload: dict[str, Any], model: PHCLiteDale, ckpt: dict[str, Any], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, threshold), scores


def evaluate_oos() -> tuple[pd.DataFrame, pd.DataFrame]:
    model, ckpt, selected = selected_model()
    threshold = float(selected["threshold"])
    rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decision, _ = dale_decision_for_payload(payload, model, ckpt, threshold)
        row = p05a.metric_from_decision(split, "phc_dale", payload, decision)
        row["seed_selected_on_val"] = int(selected["seed"])
        row["threshold_selected_on_val"] = threshold
        row["new3_used_for_selection"] = False
        row["new4_used_for_selection"] = False
        row["new5_used_for_selection"] = False
        rows.append(row)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        code = payload["session_code"].numpy()
        for session_code, session_id in enumerate(payload["session_labels"]):
            mask = code == session_code
            session_rows.append(
                {
                    "split": split,
                    "policy_id": "phc_dale",
                    "session_id": session_id,
                    "session_excess_pnl_H20": float(pnl[mask].sum()),
                    "selected_action_count": int((decision[mask] != p05a.DECISION_NO_QUOTE).sum()),
                    "state_count": int(mask.sum()),
                }
            )
    dale = pd.DataFrame(rows)
    base = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    compare = pd.concat(
        [
            base[base["policy_id"].isin(["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            dale,
        ],
        ignore_index=True,
    )
    dale.to_csv(RESULTS / "phc_dale_oos_eval.csv", index=False)
    compare.to_csv(RESULTS / "phc_dale_vs_v1_hgb_queue.csv", index=False)
    pd.DataFrame(session_rows).to_csv(RESULTS / "phc_dale_session_pnl.csv", index=False)
    write_oos_report(compare)
    return dale, compare


def write_oos_report(compare: pd.DataFrame) -> None:
    subset = compare[compare["policy_id"].isin(["phc_dale", "phc_lite", "ml_hgb_regressor_shallow_convergence_selected", "side_by_queue_imbalance_abs_0p1"])]
    write_markdown(
        ROOT / "PHC_DALE_OOS_EVALUATION_REPORT.md",
        [
            "# PHC Dale OOS Evaluation Report",
            "",
            "Dale variant is selected using validation only. NEW3/NEW4 are diagnostic OOS only. NEW5 is not used.",
            "",
            "Caveat: the HGB comparison is loaded as a frozen Phase 04B/05A artifact. The local runtime emits a scikit-learn version warning when deserializing that joblib; no HGB retraining or retuning is performed in this phase.",
            "",
            subset.to_markdown(index=False),
        ],
    )


def matched_random() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    rows = []
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decision, _ = dale_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
        _, summary = p05a.matched_random_for_split(split, decision, n_trials=p05a.RANDOM_TRIALS, seed=20260607)
        row = summary.iloc[0].to_dict()
        row["policy_id"] = "phc_dale"
        row["seed_selected_on_val"] = int(selected["seed"])
        row["threshold_selected_on_val"] = float(selected["threshold"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_dale_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "PHC_DALE_MATCHED_RANDOM_REPORT.md",
        [
            "# PHC Dale Matched Random Report",
            "",
            "Matched random matches selected count, session, side, and spread bucket where feasible. Empirical p-value is fraction of random trials whose PnL >= Dale PnL.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def causal_fill_eval() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows = []
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decision, _ = dale_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
        probs = p05a.fill_probabilities(payload, decision, norms)
        fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, "05c1_dale_conservative_l2_stress") < probs)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
        rows.append(
            {
                "split": split,
                "policy_id": "phc_dale",
                "horizon": p05a.HORIZON,
                "fill_model": "conservative_l2_stress",
                "inventory_cap": 1,
                "fee_multiplier": 1.0,
                "sequential_excess_pnl": float(total),
                "fill_count": int(fill_count),
                "forced_no_quote_count": int(forced),
                "max_abs_inventory": int(max_inv),
                "max_drawdown": float(max_dd),
                "win_rate": win / max(fill_count, 1) if fill_count else 0.0,
                "loss_rate": loss / max(fill_count, 1) if fill_count else 0.0,
                "avg_trade_excess_pnl": total / max(fill_count, 1) if fill_count else 0.0,
                "fill_uses_future_pnl": False,
                "new5_used_for_selection": False,
            }
        )
    dale = pd.DataFrame(rows)
    base = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    compare = pd.concat(
        [
            base[base["policy_id"].isin(["side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            dale,
        ],
        ignore_index=True,
    )
    dale.to_csv(RESULTS / "phc_dale_causal_fill_eval.csv", index=False)
    compare.to_csv(RESULTS / "phc_dale_causal_fill_vs_v1.csv", index=False)
    write_markdown(
        ROOT / "PHC_DALE_CAUSAL_FILL_REPORT.md",
        [
            "# PHC Dale Causal-fill Report",
            "",
            "Same proxy family as PHC-lite v1: H20 exit, conservative L2 fill, inventory cap 1. Not true queue replay. NEW5 is not used.",
            "",
            compare.to_markdown(index=False),
        ],
    )
    return compare


def seed_stability() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = pd.read_csv(PHASE05A / "results" / "phc_lite_seed_stability.csv")
    for seed in SEEDS:
        ckpt_path = MODELS / f"phc_dale_seed{seed}_best_val.pt"
        model, ckpt = load_model(ckpt_path)
        val = p05a.load_sequences("val")
        scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
        best_threshold = 0.0
        best_pnl = -float("inf")
        for threshold in threshold_grid(scores):
            decision = p05a.policy_from_scores(scores, float(threshold))
            pnl = float(p05a.decision_pnl_from_payload(val, decision).sum())
            if pnl > best_pnl:
                best_pnl = pnl
                best_threshold = float(threshold)
        for split in ["val", "new3", "new4"]:
            payload = p05a.load_sequences(split)
            decision = p05a.policy_from_scores(predict_scores(model, payload, target_scale=float(ckpt["target_scale"])), best_threshold)
            metric = p05a.metric_from_decision(split, f"phc_dale_seed{seed}", payload, decision)
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "best_epoch": int(ckpt.get("best_epoch", -1)),
                    "threshold": best_threshold,
                    "pnl": float(metric["total_excess_pnl_H20"]),
                    "selected_action_count": int(metric["selected_action_count"]),
                    "positive": bool(metric["total_excess_pnl_H20"] > 0),
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_dale_seed_stability.csv", index=False)
    return out


def oracle_capture() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    rows: list[dict[str, Any]] = []
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decisions = {
            "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
            "ml_hgb_regressor_shallow_convergence_selected": p05a.hgb_decision(payload)[0],
        }
        # Use frozen PHC-lite v1 metrics from 05A instead of recomputing threshold here.
        phc_policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
        phc_model, phc_ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
        decisions["phc_lite"] = p05a.policy_from_scores(p05a.predict_scores(phc_model, payload, target_scale=float(phc_ckpt["target_scale"])), float(phc_policy["threshold"]))
        decisions["phc_dale"] = dale_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))[0]
        bid = payload["y_bid"].numpy().astype(float)
        ask = payload["y_ask"].numpy().astype(float)
        best = np.maximum(bid, ask)
        side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK)
        for policy_id, decision in decisions.items():
            count = int((decision != p05a.DECISION_NO_QUOTE).sum())
            candidates = np.arange(len(best))
            order = candidates[np.argsort(best[candidates])[::-1]]
            chosen = order[:count]
            oracle_decision = np.zeros(len(best), dtype=np.int8)
            oracle_decision[chosen] = side[chosen].astype(np.int8)
            policy_pnl = float(p05a.decision_pnl_from_payload(payload, decision).sum())
            oracle_pnl = float(p05a.decision_pnl_from_payload(payload, oracle_decision).sum())
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id,
                    "policy_selected_count": count,
                    "policy_pnl": policy_pnl,
                    "matched_count_oracle_pnl": oracle_pnl,
                    "capture_ratio": policy_pnl / oracle_pnl if oracle_pnl else np.nan,
                    "opportunity_gap": oracle_pnl - policy_pnl,
                    "oracle_uses_future_labels": True,
                    "new5_used": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_dale_window_oracle_capture.csv", index=False)
    write_markdown(
        ROOT / "PHC_DALE_ORACLE_HEADROOM_REPORT.md",
        [
            "# PHC Dale Oracle Headroom Report",
            "",
            "Oracle capture is computed on val/NEW3/NEW4 sequence-window universes only. NEW5 is not used.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def final_report() -> str:
    sign = pd.read_csv(RESULTS / "dale_weight_sign_audit.csv")
    oos = pd.read_csv(RESULTS / "phc_dale_vs_v1_hgb_queue.csv")
    matched = pd.read_csv(RESULTS / "phc_dale_matched_random_summary.csv")
    causal = pd.read_csv(RESULTS / "phc_dale_causal_fill_vs_v1.csv")
    seed = pd.read_csv(RESULTS / "phc_dale_seed_stability.csv")
    oracle = pd.read_csv(RESULTS / "phc_dale_window_oracle_capture.csv")
    total_sign_violations = int(sign["sign_violations"].sum())
    dale_oos = oos[oos["policy_id"].eq("phc_dale")].set_index("split")
    v1_oos = oos[oos["policy_id"].eq("phc_lite")].set_index("split")
    dale_causal = causal[causal["policy_id"].eq("phc_dale")].set_index("split")
    v1_causal = causal[causal["policy_id"].eq("phc_lite")].set_index("split")
    all_splits = ["val", "new3", "new4"]
    sign_ok = total_sign_violations == 0
    matched_ok = bool(matched["beats_random_95p"].astype(bool).all())
    seeds_positive = bool(seed["positive"].astype(bool).all())
    val_improves = float(dale_oos.loc["val", "total_excess_pnl_H20"]) >= float(v1_oos.loc["val", "total_excess_pnl_H20"])
    oos_improves = all(float(dale_oos.loc[s, "total_excess_pnl_H20"]) >= float(v1_oos.loc[s, "total_excess_pnl_H20"]) for s in ["new3", "new4"])
    causal_improves = all(float(dale_causal.loc[s, "sequential_excess_pnl"]) >= float(v1_causal.loc[s, "sequential_excess_pnl"]) for s in all_splits)
    if not sign_ok:
        status = "dale_constraint_implementation_failed"
    elif not seeds_positive:
        status = "dale_training_unstable"
    elif val_improves and not oos_improves:
        status = "dale_overfits_validation"
    elif val_improves and oos_improves and causal_improves and matched_ok:
        status = "dale_improves_phc_lite"
    elif not causal_improves or not oos_improves:
        status = "dale_hurts_performance"
    else:
        status = "dale_neutral_regularizer"
    write_markdown(
        ROOT / "PHC_DALE_ABLATION_REPORT.md",
        [
            "# PHC Dale Ablation Report",
            "",
            f"Final status: `{status}`.",
            "",
            "## What is Dale's Law?",
            "",
            "Dale's Law constrains each presynaptic hidden unit to have outgoing synaptic weights of one fixed sign: excitatory or inhibitory.",
            "",
            "## Implementation",
            "",
            "`message = z @ W_eff.T + bias`, with `W_eff = softplus(raw_weight) * sign_vector`. The sign vector is applied to columns of `W_eff`, so each presynaptic unit's outgoing fan-out has one sign. Ratio is 80% excitatory / 20% inhibitory.",
            "",
            "This is a controlled Dale's Law ablation only. It is not full PHCSSM, not ALIF, not STP, not STDP, not true exchange replay, and not real-world profitability proof.",
            "",
            "Caveat: the HGB comparison is loaded as a frozen Phase 04B/05A artifact. The local runtime emits a scikit-learn version warning when deserializing that joblib; no HGB retraining or retuning is performed in this phase.",
            "",
            "## Sign Audit",
            "",
            sign.to_markdown(index=False),
            "",
            "## OOS Evaluation",
            "",
            oos[oos["policy_id"].isin(["phc_dale", "phc_lite", "ml_hgb_regressor_shallow_convergence_selected", "side_by_queue_imbalance_abs_0p1"])].to_markdown(index=False),
            "",
            "## Matched Random",
            "",
            matched.to_markdown(index=False),
            "",
            "## Causal-fill Sequential",
            "",
            causal.to_markdown(index=False),
            "",
            "## Seed Stability",
            "",
            seed.to_markdown(index=False),
            "",
            "## Oracle Capture",
            "",
            oracle.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            f"Dale sign audit passed: `{sign_ok}`.",
            f"Matched random passed all splits: `{matched_ok}`.",
            f"Seed stability positive all splits/seeds: `{seeds_positive}`.",
            f"Validation action-level matches/improves PHC-lite v1: `{val_improves}`.",
            f"NEW3/NEW4 action-level matches/improves PHC-lite v1: `{oos_improves}`.",
            f"Causal-fill sequential matches/improves PHC-lite v1 on val/NEW3/NEW4: `{causal_improves}`.",
            "",
            "Forbidden claims remain: Dale variant is not full PHCSSM, not a fresh NEW5 holdout result, not real-world profitability proof, not true exchange replay, and not FPGA PnL proof.",
        ],
    )
    return status


def run_all() -> None:
    ensure_dirs()
    for seed in SEEDS:
        if not (MODELS / f"phc_dale_seed{seed}_best_val.pt").exists():
            train_seed(seed)
    select_seed_and_threshold()
    dale_weight_sign_audit()
    evaluate_oos()
    matched_random()
    causal_fill_eval()
    seed_stability()
    oracle_capture()
    print(final_report())
    print(ROOT / "PHC_DALE_ABLATION_REPORT.md")
