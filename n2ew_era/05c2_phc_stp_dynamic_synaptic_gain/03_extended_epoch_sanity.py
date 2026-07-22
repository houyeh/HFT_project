from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PHASE05A = ROOT.parent / "05a_phc_lite_microstructure_sequence_feasibility"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402
from common_05c2 import HIDDEN_DIM, LR, MTL_STEPS, WEIGHT_DECAY, predict_scores, threshold_grid, write_json, write_markdown  # noqa: E402
from models.phc_lite_stp import PHCLiteSTP, count_parameters  # noqa: E402

SEEDS = [0, 1, 2]
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 4096


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def train_seed(seed: int) -> pd.DataFrame:
    set_seed(seed)
    scaler = p05a.load_scaler()
    target_scale = float(scaler["target_scale_train_std_no_mean_shift"])
    train_payload = p05a.load_sequences("train")
    val_payload = p05a.load_sequences("val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PHCLiteSTP(len(p05a.PHC_FEATURES), hidden_dim=HIDDEN_DIM, mtl_steps=MTL_STEPS, dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    huber = torch.nn.HuberLoss(delta=1.0)
    loader = torch.utils.data.DataLoader(
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
    curve_path = RESULTS / f"phc_stp_extended_training_curve_seed{seed}.csv"
    for epoch in range(1, MAX_EPOCHS + 1):
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
        train_scores = predict_scores(model, train_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
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
            "train_huber_loss": p05a.huber_np((train_scores - train_y) / target_scale),
            "val_huber_loss": p05a.huber_np((val_scores - val_y) / target_scale),
            "train_spearman_bid": p05a.safe_spearman(train_scores[:, 0], train_y[:, 0]),
            "val_spearman_bid": p05a.safe_spearman(val_scores[:, 0], val_y[:, 0]),
            "train_spearman_ask": p05a.safe_spearman(train_scores[:, 1], train_y[:, 1]),
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
                },
                MODELS / f"phc_stp_extended_seed{seed}_best_val.pt",
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
            },
            MODELS / f"phc_stp_extended_seed{seed}_last.pt",
        )
        print(f"extended stp seed={seed} epoch={epoch} val_pnl={val_pnl:.2f} selected={row['val_selected_count']} grad={row['grad_norm']:.4f}", flush=True)
        if early_counter >= PATIENCE:
            break
    return pd.DataFrame(rows)


def load_extended_model(seed: int) -> tuple[PHCLiteSTP, dict[str, Any]]:
    ckpt = torch.load(MODELS / f"phc_stp_extended_seed{seed}_best_val.pt", map_location="cpu", weights_only=False)
    model = PHCLiteSTP(int(ckpt["feature_dim"]), hidden_dim=int(ckpt["hidden_dim"]), mtl_steps=int(ckpt["mtl_steps"]), dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def select_and_eval() -> tuple[pd.DataFrame, dict[str, Any]]:
    val = p05a.load_sequences("val")
    grid_rows: list[dict[str, Any]] = []
    seed_summaries = []
    for seed in SEEDS:
        curve = pd.read_csv(RESULTS / f"phc_stp_extended_training_curve_seed{seed}.csv")
        model, ckpt = load_extended_model(seed)
        scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
        before50 = float(curve[curve["epoch"] <= 50]["val_policy_excess_pnl"].max())
        after50 = float(curve[curve["epoch"] > 50]["val_policy_excess_pnl"].max()) if (curve["epoch"] > 50).any() else np.nan
        seed_summaries.append(
            {
                "seed": seed,
                "best_epoch_t0": int(curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]["epoch"]),
                "best_val_pnl_at_threshold0": float(curve["val_policy_excess_pnl"].max()),
                "best_val_pnl_epoch_le_50": before50,
                "best_val_pnl_epoch_gt_50": after50,
                "improved_after_epoch50": bool(np.isfinite(after50) and after50 > before50),
                "max_epoch_hit": int(curve["epoch"].max()) >= MAX_EPOCHS,
            }
        )
        for threshold in threshold_grid(scores):
            decision = p05a.policy_from_scores(scores, float(threshold))
            row = p05a.metric_from_decision("val", f"phc_stp_extended_seed{seed}", val, decision)
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
            grid_rows.append(row)
    seed_summary = pd.DataFrame(seed_summaries)
    seed_summary.to_csv(RESULTS / "phc_stp_extended_seed_summary.csv", index=False)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(RESULTS / "phc_stp_extended_threshold_search.csv", index=False)
    best = grid.sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).iloc[0].to_dict()
    selected = {
        "model": "phc_lite_stp_like_dynamic_gain_extended_epoch",
        "selection_split": "val",
        "seed": int(best["seed"]),
        "threshold": float(best["threshold"]),
        "target": "excess_pnl_H20",
        "seq_len": 128,
        "stride": 32,
        "hidden_dim": HIDDEN_DIM,
        "mtl_steps": MTL_STEPS,
        "max_epoch": MAX_EPOCHS,
        "patience": PATIENCE,
        "validation_metrics": p05a.json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    write_json(RESULTS / "phc_stp_extended_selected_policy.json", selected)
    return seed_summary, selected


def evaluate_selected(selected: dict[str, Any]) -> pd.DataFrame:
    model, ckpt = load_extended_model(int(selected["seed"]))
    threshold = float(selected["threshold"])
    v1_action = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    v1_causal = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    phc_policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    phc_model, phc_ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows = []
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decision = p05a.policy_from_scores(scores, threshold)
        action_metric = p05a.metric_from_decision(split, "phc_stp_extended", payload, decision)
        probs = p05a.fill_probabilities(payload, decision, norms)
        fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, "05c2_extended_stp") < probs)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)

        bid = payload["y_bid"].numpy().astype(float)
        ask = payload["y_ask"].numpy().astype(float)
        best_state_value = np.maximum(bid, ask)
        side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK)
        order = np.argsort(best_state_value)[::-1]
        count = int((decision != p05a.DECISION_NO_QUOTE).sum())
        oracle_decision = np.zeros(len(best_state_value), dtype=np.int8)
        chosen = order[:count]
        oracle_decision[chosen] = side[chosen].astype(np.int8)
        policy_pnl = float(p05a.decision_pnl_from_payload(payload, decision).sum())
        oracle_pnl = float(p05a.decision_pnl_from_payload(payload, oracle_decision).sum())

        phc_decision = p05a.policy_from_scores(p05a.predict_scores(phc_model, payload, target_scale=float(phc_ckpt["target_scale"])), float(phc_policy["threshold"]))
        phc_count = int((phc_decision != p05a.DECISION_NO_QUOTE).sum())
        phc_oracle_decision = np.zeros(len(best_state_value), dtype=np.int8)
        phc_chosen = order[:phc_count]
        phc_oracle_decision[phc_chosen] = side[phc_chosen].astype(np.int8)
        phc_oracle_pnl = float(p05a.decision_pnl_from_payload(payload, phc_oracle_decision).sum())
        phc_policy_pnl = float(p05a.decision_pnl_from_payload(payload, phc_decision).sum())

        v1_action_row = v1_action[(v1_action["split"].eq(split)) & (v1_action["policy_id"].eq("phc_lite"))].iloc[0]
        v1_causal_row = v1_causal[(v1_causal["split"].eq(split)) & (v1_causal["policy_id"].eq("phc_lite"))].iloc[0]
        rows.append(
            {
                "split": split,
                "seed_selected_on_val": int(selected["seed"]),
                "threshold_selected_on_val": threshold,
                "best_epoch": int(ckpt["best_epoch"]),
                "action_pnl": float(action_metric["total_excess_pnl_H20"]),
                "v1_action_pnl": float(v1_action_row["total_excess_pnl_H20"]),
                "action_delta_vs_v1": float(action_metric["total_excess_pnl_H20"]) - float(v1_action_row["total_excess_pnl_H20"]),
                "selected_action_count": int(action_metric["selected_action_count"]),
                "causal_fill_pnl": float(total),
                "v1_causal_fill_pnl": float(v1_causal_row["sequential_excess_pnl"]),
                "causal_delta_vs_v1": float(total) - float(v1_causal_row["sequential_excess_pnl"]),
                "causal_fill_count": int(fill_count),
                "causal_max_drawdown": float(max_dd),
                "oracle_capture_ratio": policy_pnl / oracle_pnl if oracle_pnl else np.nan,
                "v1_oracle_capture_ratio": phc_policy_pnl / phc_oracle_pnl if phc_oracle_pnl else np.nan,
                "oracle_capture_delta_vs_v1": (policy_pnl / oracle_pnl if oracle_pnl else np.nan) - (phc_policy_pnl / phc_oracle_pnl if phc_oracle_pnl else np.nan),
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        if not (RESULTS / f"phc_stp_extended_training_curve_seed{seed}.csv").exists():
            train_seed(seed)
    seed_summary, selected = select_and_eval()
    eval_summary = evaluate_selected(selected)
    eval_summary.to_csv(RESULTS / "phc_stp_extended_epoch_sanity.csv", index=False)
    after50_any = bool(seed_summary["improved_after_epoch50"].any())
    exceeds_v1_val = bool((eval_summary[eval_summary["split"].eq("val")]["action_delta_vs_v1"] >= 0).all())
    exceeds_v1_oos = bool((eval_summary[eval_summary["split"].isin(["new3", "new4"])]["action_delta_vs_v1"] >= 0).all())
    causal_improves = bool((eval_summary["causal_delta_vs_v1"] >= 0).all())
    overfit = bool(exceeds_v1_val and not exceeds_v1_oos)
    write_markdown(
        ROOT / "PHC_STP_EXTENDED_EPOCH_SANITY_REPORT.md",
        [
            "# PHC STP-like Extended Epoch Sanity Report",
            "",
            "Extended sanity uses max_epoch=80, patience=10, seeds 0/1/2, same TXF-only data, same features, same target `excess_pnl_H20`, and validation-only seed/threshold selection. NEW3/NEW4/NEW5 are not used for selection.",
            "",
            "## Seed Summary",
            "",
            seed_summary.to_markdown(index=False),
            "",
            "## Selected Extended Model Evaluation",
            "",
            eval_summary.to_markdown(index=False),
            "",
            "## Answers",
            "",
            f"Does validation PnL continue improving after epoch 50? `{after50_any}`.",
            f"Does STP exceed PHC-lite v1 on validation? `{exceeds_v1_val}`.",
            f"Does OOS NEW3/NEW4 improve versus v1? `{exceeds_v1_oos}`.",
            f"Does causal-fill improve versus v1? `{causal_improves}`.",
            f"Evidence of overfitting by val-improves/OOS-fails rule: `{overfit}`.",
            "",
            "This is still an STP-like dynamic gain ablation, not full biological STP, and it does not adopt STP unless it matches or beats frozen PHC-lite v1.",
        ],
    )
    print(json.dumps({"after50_any": after50_any, "exceeds_v1_val": exceeds_v1_val, "exceeds_v1_oos": exceeds_v1_oos, "causal_improves": causal_improves, "overfit": overfit}, indent=2))


if __name__ == "__main__":
    main()
