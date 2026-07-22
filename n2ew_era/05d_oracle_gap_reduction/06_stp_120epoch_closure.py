from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import common_05d as c

PHASE05A = c.PHASE05A
PHASE05C2 = c.PHASE05C2
if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PHASE05C2) not in sys.path:
    sys.path.insert(0, str(PHASE05C2))

import common_05a as p05a  # noqa: E402
import common_05c2 as c05c2  # noqa: E402
from models.phc_lite_stp import PHCLiteSTP, count_parameters  # noqa: E402

SEEDS = [0, 1, 2]
MAX_EPOCHS = 120
PATIENCE = 15
START_EPOCH = 80
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 16
MTL_STEPS = 1


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def predict_scores(model: PHCLiteSTP, payload: dict[str, Any], target_scale: float, batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        x = payload["x"]
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0) * float(target_scale)


def train_seed(seed: int) -> pd.DataFrame:
    set_seed(seed)
    scaler = p05a.load_scaler()
    target_scale = float(scaler["target_scale_train_std_no_mean_shift"])
    train_payload = p05a.load_sequences("train")
    val_payload = p05a.load_sequences("val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PHCLiteSTP(len(p05a.PHC_FEATURES), hidden_dim=HIDDEN_DIM, mtl_steps=MTL_STEPS, dropout=0.0).to(device)
    previous_best = PHASE05C2 / "models" / f"phc_stp_extended_seed{seed}_best_val.pt"
    previous_last = PHASE05C2 / "models" / f"phc_stp_extended_seed{seed}_last.pt"
    previous_curve = PHASE05C2 / "results" / f"phc_stp_extended_training_curve_seed{seed}.csv"
    if not previous_best.exists() or not previous_last.exists() or not previous_curve.exists():
        raise FileNotFoundError(f"Missing audited 80-epoch STP artifact for seed={seed}")
    previous_ckpt = torch.load(previous_last, map_location="cpu", weights_only=False)
    model.load_state_dict(previous_ckpt["model_state_dict"])
    previous_best_ckpt = torch.load(previous_best, map_location="cpu", weights_only=False)
    previous_best_ckpt.update(
        {
            "max_epoch": MAX_EPOCHS,
            "patience": PATIENCE,
            "continued_from_phase05c2_epoch80": True,
            "optimizer_state_continued": False,
            "optimizer_restart_caveat": "05C2 did not store optimizer state; epochs 81-120 restart AdamW from the 80-epoch weights.",
        }
    )
    torch.save(previous_best_ckpt, c.MODELS / f"phc_stp_120epoch_seed{seed}_best_val.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    huber = torch.nn.HuberLoss(delta=1.0)
    loader = torch.utils.data.DataLoader(
        p05a.SequenceTensorDataset(train_payload, target_scale),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    history = pd.read_csv(previous_curve)
    history["max_epoch"] = MAX_EPOCHS
    history["patience"] = PATIENCE
    history["continued_from_phase05c2_epoch80"] = False
    history["optimizer_state_continued"] = False
    best_idx = history["val_policy_excess_pnl"].astype(float).idxmax()
    best_metric = float(history.loc[best_idx, "val_policy_excess_pnl"])
    best_epoch = int(history.loc[best_idx, "epoch"])
    early_counter = 0
    rows: list[dict[str, Any]] = history.to_dict("records")
    curve_path = c.RESULTS / f"phc_stp_120epoch_training_curve_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(curve_path, index=False)
    for epoch in range(START_EPOCH + 1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        grad_norms = []
        for xb, yb in loader:
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
        val_scores = predict_scores(model, val_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
        train_y = np.column_stack([train_payload["y_bid"].numpy(), train_payload["y_ask"].numpy()])
        val_y = np.column_stack([val_payload["y_bid"].numpy(), val_payload["y_ask"].numpy()])
        val_decision = p05a.policy_from_scores(val_scores, 0.0)
        val_pnl = float(p05a.decision_pnl_from_payload(val_payload, val_decision).sum())
        row = {
            "epoch": epoch,
            "seed": seed,
            "train_loss": float(np.mean(losses)),
            "val_loss": p05a.huber_np((val_scores - val_y) / target_scale),
            "train_huber_loss": float("nan"),
            "val_huber_loss": p05a.huber_np((val_scores - val_y) / target_scale),
            "train_spearman_bid": float("nan"),
            "val_spearman_bid": p05a.safe_spearman(val_scores[:, 0], val_y[:, 0]),
            "train_spearman_ask": float("nan"),
            "val_spearman_ask": p05a.safe_spearman(val_scores[:, 1], val_y[:, 1]),
            "val_policy_excess_pnl": val_pnl,
            "val_selected_count": int((val_decision != p05a.DECISION_NO_QUOTE).sum()),
            "val_pr_auc_bid": p05a.safe_pr_auc(val_y[:, 0] > 0, val_scores[:, 0]),
            "val_pr_auc_ask": p05a.safe_pr_auc(val_y[:, 1] > 0, val_scores[:, 1]),
            "learning_rate": LR,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "early_stop_counter": early_counter,
            "parameter_count": count_parameters(model),
            "max_epoch": MAX_EPOCHS,
            "patience": PATIENCE,
            "new3_used_for_selection": False,
            "new4_used_for_selection": False,
            "new5_used_for_selection": False,
            "continued_from_phase05c2_epoch80": True,
            "optimizer_state_continued": False,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(curve_path, index=False)
        if val_pnl > best_metric + 1e-3:
            best_metric = val_pnl
            best_epoch = epoch
            early_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": len(p05a.PHC_FEATURES),
                    "hidden_dim": HIDDEN_DIM,
                    "mtl_steps": MTL_STEPS,
                    "dropout": 0.0,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "target_scale": target_scale,
                    "selection_metric": "validation_policy_excess_pnl_at_threshold_0",
                    "max_epoch": MAX_EPOCHS,
                    "patience": PATIENCE,
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                    "stp_like_gain": True,
                    "continued_from_phase05c2_epoch80": True,
                    "optimizer_state_continued": False,
                    "optimizer_restart_caveat": "05C2 did not store optimizer state; epochs 81-120 restart AdamW from the 80-epoch weights.",
                },
                c.MODELS / f"phc_stp_120epoch_seed{seed}_best_val.pt",
            )
        else:
            early_counter += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_dim": len(p05a.PHC_FEATURES),
                "hidden_dim": HIDDEN_DIM,
                "mtl_steps": MTL_STEPS,
                "dropout": 0.0,
                "seed": seed,
                "epoch": epoch,
                "target_scale": target_scale,
                "max_epoch": MAX_EPOCHS,
                "patience": PATIENCE,
                "new5_used_for_selection": False,
                "stp_like_gain": True,
                "continued_from_phase05c2_epoch80": True,
                "optimizer_state_continued": False,
                "optimizer_restart_caveat": "05C2 did not store optimizer state; epochs 81-120 restart AdamW from the 80-epoch weights.",
            },
            c.MODELS / f"phc_stp_120epoch_seed{seed}_last.pt",
        )
        print(f"stp120 seed={seed} epoch={epoch} val_pnl={val_pnl:.2f} selected={row['val_selected_count']} grad={row['grad_norm']:.4f}", flush=True)
        if early_counter >= PATIENCE:
            break
    return pd.DataFrame(rows)


def load_model(seed: int) -> tuple[PHCLiteSTP, dict[str, Any]]:
    ckpt = torch.load(c.MODELS / f"phc_stp_120epoch_seed{seed}_best_val.pt", map_location="cpu", weights_only=False)
    model = PHCLiteSTP(int(ckpt["feature_dim"]), hidden_dim=int(ckpt["hidden_dim"]), mtl_steps=int(ckpt["mtl_steps"]), dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def threshold_grid(scores: np.ndarray) -> list[float]:
    max_scores = np.max(scores, axis=1)
    finite = max_scores[np.isfinite(max_scores)]
    q = np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if len(finite) else np.array([0.0])
    return sorted(set([0.0, *q.tolist()]))


def select_model() -> dict[str, Any]:
    val = p05a.load_sequences("val")
    rows = []
    for seed in SEEDS:
        model, ckpt = load_model(seed)
        scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
        for threshold in threshold_grid(scores):
            decision = p05a.policy_from_scores(scores, float(threshold))
            row = p05a.metric_from_decision("val", f"phc_stp_120epoch_seed{seed}", val, decision)
            row.update(
                {
                    "seed": seed,
                    "threshold": float(threshold),
                    "best_epoch": int(ckpt["best_epoch"]),
                    "selection_split": "val",
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(c.RESULTS / "phc_stp_120epoch_threshold_search.csv", index=False)
    best = grid.sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).iloc[0].to_dict()
    selected = {
        "model": "phc_stp_120epoch",
        "selection_split": "val",
        "seed": int(best["seed"]),
        "threshold": float(best["threshold"]),
        "target": "excess_pnl_H20",
        "seq_len": 128,
        "stride": 32,
        "max_epoch": MAX_EPOCHS,
        "patience": PATIENCE,
        "validation_metrics": p05a.json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    c.write_json(c.RESULTS / "phc_stp_120epoch_selected_policy.json", selected)
    return selected


def evaluate(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, ckpt = load_model(int(selected["seed"]))
    threshold = float(selected["threshold"])
    v1_action = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    v1_causal = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    action_rows = []
    causal_rows = []
    oracle_rows = []
    for split in c.SPLITS:
        payload = p05a.load_sequences(split)
        scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decision = p05a.policy_from_scores(scores, threshold)
        metric = p05a.metric_from_decision(split, "phc_stp_120epoch", payload, decision)
        action_rows.append(metric)
        causal_rows.append(c.causal_fill_for_decision(split, "phc_stp_120epoch", payload, decision, "05d_stp120"))
        cap = c.oracle_capture_for_decision(payload, decision)
        cap.update({"split": split, "policy_id": "phc_stp_120epoch"})
        oracle_rows.append(cap)
    action = pd.DataFrame(action_rows)
    causal = pd.DataFrame(causal_rows)
    oracle = pd.DataFrame(oracle_rows)
    action.to_csv(c.RESULTS / "phc_stp_120epoch_oos_eval.csv", index=False)
    causal.to_csv(c.RESULTS / "phc_stp_120epoch_causal_fill.csv", index=False)
    oracle.to_csv(c.RESULTS / "phc_stp_120epoch_oracle_capture.csv", index=False)
    curves = []
    for seed in SEEDS:
        curve = pd.read_csv(c.RESULTS / f"phc_stp_120epoch_training_curve_seed{seed}.csv")
        curves.append(curve)
    convergence = pd.concat(curves, ignore_index=True)
    convergence.to_csv(c.RESULTS / "phc_stp_120epoch_convergence.csv", index=False)
    return action, causal, oracle


def report(selected: dict[str, Any], action: pd.DataFrame, causal: pd.DataFrame, oracle: pd.DataFrame) -> str:
    v1_action = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    v1_causal = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    v1_oracle = pd.read_csv(PHASE05C2 / "results" / "phc_stp_window_oracle_capture.csv")
    v1_action = v1_action[v1_action["policy_id"].eq("phc_lite")][["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "v1_action_pnl"})
    v1_causal = v1_causal[v1_causal["policy_id"].eq("phc_lite")][["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "v1_causal_pnl"})
    v1_oracle = v1_oracle[v1_oracle["policy_id"].eq("phc_lite")][["split", "capture_ratio"]].rename(columns={"capture_ratio": "v1_capture_ratio"})
    compare = action[["split", "total_excess_pnl_H20", "selected_action_count", "max_drawdown"]].merge(v1_action, on="split")
    compare = compare.merge(causal[["split", "sequential_excess_pnl", "max_drawdown"]].rename(columns={"max_drawdown": "causal_max_drawdown"}), on="split")
    compare = compare.merge(v1_causal, on="split").merge(oracle[["split", "capture_ratio"]], on="split").merge(v1_oracle, on="split")
    compare["action_delta_vs_v1"] = compare["total_excess_pnl_H20"] - compare["v1_action_pnl"]
    compare["causal_delta_vs_v1"] = compare["sequential_excess_pnl"] - compare["v1_causal_pnl"]
    compare["oracle_capture_delta_vs_v1"] = compare["capture_ratio"] - compare["v1_capture_ratio"]
    compare.to_csv(c.RESULTS / "phc_stp_120epoch_vs_v1_summary.csv", index=False)
    convergence = pd.read_csv(c.RESULTS / "phc_stp_120epoch_convergence.csv")
    seed_summary = convergence.groupby("seed").agg(
        max_epoch_seen=("epoch", "max"),
        best_epoch=("val_policy_excess_pnl", lambda s: int(convergence.loc[s.idxmax(), "epoch"])),
        best_val_pnl=("val_policy_excess_pnl", "max"),
        final_val_pnl=("val_policy_excess_pnl", "last"),
    ).reset_index()
    plateau = bool((seed_summary["max_epoch_seen"] < MAX_EPOCHS).all() or ((seed_summary["best_val_pnl"] - seed_summary["final_val_pnl"]).abs() < 50_000).all())
    action_ok = bool((compare["action_delta_vs_v1"] >= 0).all())
    causal_ok = bool((compare["causal_delta_vs_v1"] >= 0).all())
    oracle_ok = bool((compare["oracle_capture_delta_vs_v1"] >= 0).all())
    if action_ok and causal_ok and oracle_ok and plateau:
        status = "stp_adopted_as_phc_v2"
    elif action_ok and oracle_ok and not causal_ok:
        status = "stp_not_adopted_causal_fill_regression"
    elif not plateau:
        status = "stp_not_adopted_convergence_incomplete"
    else:
        status = "stp_neutral_future_work"
    c.write_markdown(
        c.ROOT / "PHC_STP_120EPOCH_CLOSURE_REPORT.md",
        [
            "# PHC STP 120-Epoch Closure Report",
            "",
            f"Final status: `{status}`.",
            "",
            "Protocol: max_epoch=120, patience=15, seeds 0/1/2, validation-only seed/threshold selection. NEW3/NEW4 are diagnostic OOS only. NEW5 is not used.",
            "",
            "Implementation note: this closure extends the audited Phase 05C.2 80-epoch STP-like checkpoints to epochs 81-120. Because 05C.2 did not store optimizer state, AdamW is restarted from the 80-epoch weights; this is an epoch-extension sanity closure, not a fresh-from-scratch 120-epoch training claim.",
            "",
            "## Selected Policy",
            "",
            "```json",
            json.dumps(selected, indent=2),
            "```",
            "",
            "## Seed Convergence Summary",
            "",
            seed_summary.to_markdown(index=False),
            "",
            "## STP 120 vs PHC-lite v1",
            "",
            compare.to_markdown(index=False),
            "",
            "Adoption requires action-level, causal-fill, oracle capture, and seed stability gates. This is STP-like dynamic gain only, not full biological STP.",
        ],
    )
    return status


def main() -> None:
    c.ensure_dirs()
    for seed in SEEDS:
        train_seed(seed)
    selected = select_model()
    action, causal, oracle = evaluate(selected)
    status = report(selected, action, causal, oracle)
    print(status)


if __name__ == "__main__":
    main()
