from __future__ import annotations

import json
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
ROOT = PROJECT_ROOT / "n2ew_era" / "05c2_phc_stp_dynamic_synaptic_gain"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402
from models.phc_lite_stp import PHCLiteSTP, count_parameters  # noqa: E402


SEEDS = [0, 1, 2]
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


def predict_scores(model: PHCLiteSTP, payload: dict[str, Any], target_scale: float, batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    model.eval()
    outs: list[np.ndarray] = []
    x = payload["x"]
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0) * float(target_scale)


def load_model(checkpoint_path: Path) -> tuple[PHCLiteSTP, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PHCLiteSTP(
        int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        mtl_steps=int(ckpt["mtl_steps"]),
        dropout=float(ckpt.get("dropout", 0.0)),
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
    model = PHCLiteSTP(
        len(p05a.PHC_FEATURES),
        hidden_dim=HIDDEN_DIM,
        mtl_steps=MTL_STEPS,
        dropout=0.0,
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
            "stp_like_gain": True,
        }
        rows.append(row)
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
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                    "stp_like_gain": True,
                },
                MODELS / f"phc_stp_seed{seed}_best_val.pt",
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
                "new5_used_for_selection": False,
                "stp_like_gain": True,
            },
            MODELS / f"phc_stp_seed{seed}_last.pt",
        )
        print(f"stp seed={seed} epoch={epoch} val_pnl={val_pnl:.2f} selected={val_selected} grad={row['grad_norm']:.4f}")
        if early_counter >= PATIENCE:
            break
    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve.loc[curve.index[-1], "max_epoch_hit"] = int(curve["epoch"].iloc[-1]) >= MAX_EPOCHS
    curve.to_csv(RESULTS / f"stp_training_curve_seed{seed}.csv", index=False)
    plot_curve(curve, seed)
    write_training_report(curve, seed)
    return curve


def plot_curve(curve: pd.DataFrame, seed: int) -> None:
    if curve.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["train_loss"], label="train")
    plt.plot(curve["epoch"], curve["val_loss"], label="val")
    plt.title(f"STP-like PHC Loss Curve seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"stp_loss_curve_seed{seed}.png", dpi=160)
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["val_policy_excess_pnl"], marker="o")
    plt.title(f"STP-like PHC Validation Policy PnL seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Val policy excess PnL")
    plt.tight_layout()
    plt.savefig(FIGURES / f"stp_val_policy_pnl_seed{seed}.png", dpi=160)
    plt.close()


def write_training_report(curve: pd.DataFrame, seed: int) -> None:
    best = curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
    max_epoch_hit = bool(int(curve["epoch"].iloc[-1]) >= MAX_EPOCHS)
    still_improving = bool(float(curve["val_policy_excess_pnl"].iloc[-1]) >= float(best["val_policy_excess_pnl"]) - 1e-6)
    write_markdown(
        ROOT / f"PHC_STP_TRAINING_CONVERGENCE_REPORT_seed{seed}.md",
        [
            f"# PHC STP-like Training Convergence Report seed {seed}",
            "",
            "This is an STP-like dynamic synaptic gain ablation, not full biological STP.",
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
        ckpt_path = MODELS / f"phc_stp_seed{seed}_best_val.pt"
        if not ckpt_path.exists():
            train_seed(seed)
        model, ckpt = load_model(ckpt_path)
        scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
        for threshold in threshold_grid(scores):
            decision = p05a.policy_from_scores(scores, float(threshold))
            row = p05a.metric_from_decision("val", f"phc_stp_seed{seed}", val, decision)
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
    grid.to_csv(RESULTS / "phc_stp_validation_threshold_search.csv", index=False)
    best = grid.sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).iloc[0].to_dict()
    selected = {
        "model": "phc_lite_stp_like_dynamic_gain",
        "variant": "PHC-lite + STP-like dynamic synaptic gain",
        "selection_split": "val",
        "seed": int(best["seed"]),
        "threshold": float(best["threshold"]),
        "target": "excess_pnl_H20",
        "seq_len": 128,
        "stride": 32,
        "hidden_dim": HIDDEN_DIM,
        "mtl_steps": MTL_STEPS,
        "validation_metrics": p05a.json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    write_json(RESULTS / "phc_stp_selected_policy.json", selected)
    return grid, selected


def selected_model() -> tuple[PHCLiteSTP, dict[str, Any], dict[str, Any]]:
    selected = read_json(RESULTS / "phc_stp_selected_policy.json")
    model, ckpt = load_model(MODELS / f"phc_stp_seed{int(selected['seed'])}_best_val.pt")
    return model, ckpt, selected


def stp_decision_for_payload(payload: dict[str, Any], model: PHCLiteSTP, ckpt: dict[str, Any], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, threshold), scores


def evaluate_oos() -> tuple[pd.DataFrame, pd.DataFrame]:
    model, ckpt, selected = selected_model()
    threshold = float(selected["threshold"])
    rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decision, _ = stp_decision_for_payload(payload, model, ckpt, threshold)
        row = p05a.metric_from_decision(split, "phc_stp", payload, decision)
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
                    "policy_id": "phc_stp",
                    "session_id": session_id,
                    "session_excess_pnl_H20": float(pnl[mask].sum()),
                    "selected_action_count": int((decision[mask] != p05a.DECISION_NO_QUOTE).sum()),
                    "state_count": int(mask.sum()),
                }
            )
    stp = pd.DataFrame(rows)
    base = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    compare = pd.concat(
        [
            base[base["policy_id"].isin(["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            stp,
        ],
        ignore_index=True,
    )
    stp.to_csv(RESULTS / "phc_stp_oos_eval.csv", index=False)
    compare.to_csv(RESULTS / "phc_stp_vs_v1_hgb_queue.csv", index=False)
    pd.DataFrame(session_rows).to_csv(RESULTS / "phc_stp_session_pnl.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_OOS_EVALUATION_REPORT.md",
        [
            "# PHC STP-like OOS Evaluation Report",
            "",
            "STP-like variant is selected using validation only. NEW3/NEW4 are diagnostic OOS only. NEW5 is not used.",
            "",
            "Caveat: the HGB comparison is loaded as a frozen Phase 04B/05A artifact. No HGB retraining or retuning is performed in this phase.",
            "",
            compare[compare["policy_id"].isin(["phc_stp", "phc_lite", "ml_hgb_regressor_shallow_convergence_selected", "side_by_queue_imbalance_abs_0p1"])].to_markdown(index=False),
        ],
    )
    return stp, compare


def matched_random() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    rows = []
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decision, _ = stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
        _, summary = p05a.matched_random_for_split(split, decision, n_trials=p05a.RANDOM_TRIALS, seed=20260609)
        row = summary.iloc[0].to_dict()
        row["policy_id"] = "phc_stp"
        row["seed_selected_on_val"] = int(selected["seed"])
        row["threshold_selected_on_val"] = float(selected["threshold"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_stp_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_MATCHED_RANDOM_REPORT.md",
        [
            "# PHC STP-like Matched Random Report",
            "",
            "Matched random matches selected count, session, side, and spread bucket where feasible. Empirical p-value is fraction of random trials whose PnL >= STP-like PnL.",
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
        decision, _ = stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
        probs = p05a.fill_probabilities(payload, decision, norms)
        fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, "05c2_stp_conservative_l2_stress") < probs)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
        rows.append(
            {
                "split": split,
                "policy_id": "phc_stp",
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
    stp = pd.DataFrame(rows)
    base = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    compare = pd.concat(
        [
            base[base["policy_id"].isin(["side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            stp,
        ],
        ignore_index=True,
    )
    stp.to_csv(RESULTS / "phc_stp_causal_fill_eval.csv", index=False)
    compare.to_csv(RESULTS / "phc_stp_causal_fill_vs_v1.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_CAUSAL_FILL_REPORT.md",
        [
            "# PHC STP-like Causal-fill Report",
            "",
            "Same proxy family as PHC-lite v1: H20 exit, conservative L2 fill, inventory cap 1. Not true queue replay. NEW5 is not used.",
            "",
            compare.to_markdown(index=False),
        ],
    )
    return compare


def seed_stability() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        ckpt_path = MODELS / f"phc_stp_seed{seed}_best_val.pt"
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
            metric = p05a.metric_from_decision(split, f"phc_stp_seed{seed}", payload, decision)
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
    out.to_csv(RESULTS / "phc_stp_seed_stability.csv", index=False)
    return out


def oracle_capture() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    rows: list[dict[str, Any]] = []
    phc_policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    phc_model, phc_ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    for split in ["val", "new3", "new4"]:
        payload = p05a.load_sequences(split)
        decisions = {
            "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
            "ml_hgb_regressor_shallow_convergence_selected": p05a.hgb_decision(payload)[0],
            "phc_lite": p05a.policy_from_scores(p05a.predict_scores(phc_model, payload, target_scale=float(phc_ckpt["target_scale"])), float(phc_policy["threshold"])),
            "phc_stp": stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))[0],
        }
        bid = payload["y_bid"].numpy().astype(float)
        ask = payload["y_ask"].numpy().astype(float)
        best = np.maximum(bid, ask)
        side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK)
        for policy_id, decision in decisions.items():
            count = int((decision != p05a.DECISION_NO_QUOTE).sum())
            order = np.argsort(best)[::-1]
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
    out.to_csv(RESULTS / "phc_stp_window_oracle_capture.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_ORACLE_HEADROOM_REPORT.md",
        [
            "# PHC STP-like Oracle Headroom Report",
            "",
            "Oracle capture is computed on val/NEW3/NEW4 sequence-window universes only. NEW5 is not used.",
            "Oracle uses future labels and is diagnostic only.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def final_report() -> str:
    oos = pd.read_csv(RESULTS / "phc_stp_vs_v1_hgb_queue.csv")
    matched = pd.read_csv(RESULTS / "phc_stp_matched_random_summary.csv")
    causal = pd.read_csv(RESULTS / "phc_stp_causal_fill_vs_v1.csv")
    seed = pd.read_csv(RESULTS / "phc_stp_seed_stability.csv")
    oracle = pd.read_csv(RESULTS / "phc_stp_window_oracle_capture.csv")
    stp_oos = oos[oos["policy_id"].eq("phc_stp")].set_index("split")
    v1_oos = oos[oos["policy_id"].eq("phc_lite")].set_index("split")
    stp_causal = causal[causal["policy_id"].eq("phc_stp")].set_index("split")
    v1_causal = causal[causal["policy_id"].eq("phc_lite")].set_index("split")
    stp_oracle = oracle[oracle["policy_id"].eq("phc_stp")].set_index("split")
    v1_oracle = oracle[oracle["policy_id"].eq("phc_lite")].set_index("split")
    all_splits = ["val", "new3", "new4"]
    matched_ok = bool(matched["beats_random_95p"].astype(bool).all())
    seeds_positive = bool(seed["positive"].astype(bool).all())
    val_improves = float(stp_oos.loc["val", "total_excess_pnl_H20"]) >= float(v1_oos.loc["val", "total_excess_pnl_H20"])
    oos_improves = all(float(stp_oos.loc[s, "total_excess_pnl_H20"]) >= float(v1_oos.loc[s, "total_excess_pnl_H20"]) for s in ["new3", "new4"])
    causal_improves = all(float(stp_causal.loc[s, "sequential_excess_pnl"]) >= float(v1_causal.loc[s, "sequential_excess_pnl"]) for s in all_splits)
    oracle_improves = all(float(stp_oracle.loc[s, "capture_ratio"]) >= float(v1_oracle.loc[s, "capture_ratio"]) for s in all_splits)
    if not seeds_positive:
        status = "stp_training_unstable"
    elif val_improves and not oos_improves:
        status = "stp_overfits_validation"
    elif val_improves and oos_improves and causal_improves and oracle_improves and matched_ok:
        status = "stp_improves_phc_lite"
    elif val_improves and oos_improves and not causal_improves:
        status = "stp_action_signal_but_strategy_degradation"
    elif not causal_improves or not oos_improves or not oracle_improves:
        status = "stp_hurts_performance"
    else:
        status = "stp_neutral_regularizer"
    write_markdown(
        ROOT / "PHC_STP_DYNAMIC_GAIN_ABLATION_REPORT.md",
        [
            "# PHC STP-like Dynamic Synaptic Gain Ablation Report",
            "",
            f"Final status: `{status}`.",
            "",
            "## Definition",
            "",
            "`gain_t = sigmoid(W_g encoded_t + U_g state_{t-1})` is applied to the synaptic message inside the PHC-lite multi-transmission loop. This is STP-like dynamic synaptic gain, not full biological STP.",
            "",
            "This controlled ablation uses the exact TXF-only PHC-lite v1 protocol: target `excess_pnl_H20`, seq_len 128, stride 32, seeds 0/1/2, validation-only epoch/threshold/seed selection, NEW3/NEW4 diagnostic OOS only, and no NEW5.",
            "",
            "## OOS Evaluation",
            "",
            oos[oos["policy_id"].isin(["phc_stp", "phc_lite", "ml_hgb_regressor_shallow_convergence_selected", "side_by_queue_imbalance_abs_0p1"])].to_markdown(index=False),
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
            f"Matched random passed all splits: `{matched_ok}`.",
            f"Seed stability positive all splits/seeds: `{seeds_positive}`.",
            f"Validation action-level matches/improves PHC-lite v1: `{val_improves}`.",
            f"NEW3/NEW4 action-level matches/improves PHC-lite v1: `{oos_improves}`.",
            f"Causal-fill sequential matches/improves PHC-lite v1 on val/NEW3/NEW4: `{causal_improves}`.",
            f"Oracle capture matches/improves PHC-lite v1 on val/NEW3/NEW4: `{oracle_improves}`.",
            "",
            "Forbidden claims remain: STP-like variant is not full PHCSSM, not a fresh NEW5 holdout result, not real-world profitability proof, not true exchange replay, and not FPGA PnL proof.",
        ],
    )
    return status


def run_all() -> None:
    ensure_dirs()
    for seed in SEEDS:
        if not (MODELS / f"phc_stp_seed{seed}_best_val.pt").exists():
            train_seed(seed)
    select_seed_and_threshold()
    evaluate_oos()
    matched_random()
    causal_fill_eval()
    seed_stability()
    oracle_capture()
    print(final_report())
    print(ROOT / "PHC_STP_DYNAMIC_GAIN_ABLATION_REPORT.md")


if __name__ == "__main__":
    run_all()
