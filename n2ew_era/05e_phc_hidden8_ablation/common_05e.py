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
ROOT = PROJECT_ROOT / "n2ew_era" / "05e_phc_hidden8_ablation"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C2 = PROJECT_ROOT / "n2ew_era" / "05c2_phc_stp_dynamic_synaptic_gain"
PHASE05C3 = PROJECT_ROOT / "n2ew_era" / "05c3_phc_stp_clean_200epoch"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"

if str(PHASE05C2) not in sys.path:
    sys.path.insert(0, str(PHASE05C2))
if str(PHASE05A) not in sys.path:
    sys.path.insert(1, str(PHASE05A))

import common_05a as p05a  # noqa: E402
from models.phc_lite_stp import PHCLiteSTP, count_parameters  # noqa: E402

SEEDS = [0, 1, 2]
HIDDEN_DIM = 8
MTL_STEPS = 1
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 4096
LR = 1e-3
LR_MIN = 1e-5
LR_SCHEDULER = "CosineAnnealingLR"
WEIGHT_DECAY = 1e-4
PHASE = "05e"
POLICY_ID = "phc_stp_v2_hidden8"
SPLITS = ["val", "new3", "new4"]
# Retain the 05C3 fill mask salt so the causal-fill comparison is controlled.
FILL_RANDOM_SALT = "05c3_stp_v2_conservative_l2_stress"


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


def best_ckpt_path(seed: int) -> Path:
    return MODELS / f"phc_stp_v2_seed{seed}_best_val.pt"


def last_ckpt_path(seed: int) -> Path:
    return MODELS / f"phc_stp_v2_seed{seed}_last.pt"


def curve_path(seed: int) -> Path:
    return RESULTS / f"phc_stp_v2_training_curve_seed{seed}.csv"


def make_checkpoint(
    model: PHCLiteSTP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    seed: int,
    epoch_key: str,
    epoch_value: int,
    target_scale: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "feature_dim": len(p05a.PHC_FEATURES),
        "hidden_dim": HIDDEN_DIM,
        "mtl_steps": MTL_STEPS,
        "dropout": 0.0,
        "seed": seed,
        epoch_key: int(epoch_value),
        "target_scale": float(target_scale),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lr_initial": LR,
        "lr_min": LR_MIN,
        "lr_scheduler": LR_SCHEDULER,
        "phase": PHASE,
        "optimizer_restarted": False,
        "selection_metric": "validation_policy_excess_pnl_at_threshold_0",
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }


def predict_scores(
    model: PHCLiteSTP,
    payload: dict[str, Any],
    target_scale: float,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
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
    if ckpt.get("phase") != PHASE:
        raise RuntimeError(f"Refusing non-{PHASE} checkpoint: {checkpoint_path}")
    if ckpt.get("optimizer_restarted") is not False:
        raise RuntimeError(f"Checkpoint is not marked clean optimizer_restarted=False: {checkpoint_path}")
    if ckpt.get("lr_scheduler") != LR_SCHEDULER or "scheduler_state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint is missing the {PHASE} cosine scheduler state: {checkpoint_path}")
    model = PHCLiteSTP(
        int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        mtl_steps=int(ckpt["mtl_steps"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def train_seed(seed: int, overwrite: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if best_ckpt_path(seed).exists() and curve_path(seed).exists() and not overwrite:
        return pd.read_csv(curve_path(seed))
    set_seed(seed)
    scaler = p05a.load_scaler()
    target_scale = float(scaler["target_scale_train_std_no_mean_shift"])
    train_payload = p05a.load_sequences("train")
    val_payload = p05a.load_sequences("val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PHCLiteSTP(len(p05a.PHC_FEATURES), hidden_dim=HIDDEN_DIM, mtl_steps=MTL_STEPS, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
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
    for epoch in range(1, MAX_EPOCHS + 1):
        current_lr = float(scheduler.get_last_lr()[0])
        model.train()
        losses: list[float] = []
        grad_norms: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = huber(pred, yb)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(grad_norm)
        val_scores = predict_scores(model, val_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
        train_scores = predict_scores(model, train_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
        val_y = np.column_stack([val_payload["y_bid"].numpy(), val_payload["y_ask"].numpy()])
        train_y = np.column_stack([train_payload["y_bid"].numpy(), train_payload["y_ask"].numpy()])
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
            "current_lr": current_lr,
            "lr_scheduler": LR_SCHEDULER,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "early_stop_counter": early_counter,
            "optimizer_restarted": False,
            "max_epoch_hit": False,
            "parameter_count": count_parameters(model),
            "device": device,
            "hidden_dim": HIDDEN_DIM,
            "mtl_steps": MTL_STEPS,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "phase": PHASE,
            "new3_used_for_selection": False,
            "new4_used_for_selection": False,
            "new5_used_for_selection": False,
        }
        rows.append(row)
        is_best = False
        if val_pnl > best_metric + 1e-3:
            best_metric = val_pnl
            best_epoch = epoch
            early_counter = 0
            is_best = True
        else:
            early_counter += 1
        pd.DataFrame(rows).to_csv(curve_path(seed), index=False)
        print(
            f"05e seed={seed} epoch={epoch} lr={current_lr:.8f} val_pnl={val_pnl:.2f} "
            f"selected={val_selected} grad={row['grad_norm']:.4f}",
            flush=True,
        )
        should_stop = early_counter >= PATIENCE
        scheduler.step()
        if is_best:
            torch.save(make_checkpoint(model, optimizer, scheduler, seed, "best_epoch", best_epoch, target_scale), best_ckpt_path(seed))
        torch.save(make_checkpoint(model, optimizer, scheduler, seed, "epoch", epoch, target_scale), last_ckpt_path(seed))
        if should_stop:
            break
    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve.loc[curve.index[-1], "max_epoch_hit"] = int(curve["epoch"].iloc[-1]) >= MAX_EPOCHS
        curve.to_csv(curve_path(seed), index=False)
    plot_training_curve(curve, seed)
    write_training_report(curve, seed)
    return curve


def plot_training_curve(curve: pd.DataFrame, seed: int) -> None:
    if curve.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["train_loss"], label="train")
    plt.plot(curve["epoch"], curve["val_loss"], label="val")
    plt.title(f"05E Hidden8 STP Loss Curve seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"phc_stp_v2_loss_curve_seed{seed}.png", dpi=160)
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["val_policy_excess_pnl"], marker="o", markersize=2)
    plt.title(f"05E Hidden8 STP Validation Policy PnL seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Val policy excess PnL")
    plt.tight_layout()
    plt.savefig(FIGURES / f"phc_stp_v2_val_policy_pnl_seed{seed}.png", dpi=160)
    plt.close()


def convergence_summary() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        path = curve_path(seed)
        if not path.exists():
            continue
        curve = pd.read_csv(path)
        if curve.empty:
            continue
        best_row = curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
        final_row = curve.iloc[-1]
        final_epoch = int(curve["epoch"].iloc[-1])
        early_stop = final_epoch < MAX_EPOCHS
        max_epoch_hit = final_epoch >= MAX_EPOCHS
        best_epoch = int(best_row["epoch"])
        rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_pnl": float(best_row["val_policy_excess_pnl"]),
                "final_epoch": final_epoch,
                "early_stop": bool(early_stop),
                "max_epoch_hit": bool(max_epoch_hit),
                "convergence_closed": bool(early_stop and best_epoch < (MAX_EPOCHS - PATIENCE + 5)),
                "strict_convergence_rule": "early_stop=True and best_epoch<185",
                "optimizer_restarted": False,
                "lr_at_best_epoch": float(best_row.get("current_lr", np.nan)),
                "lr_at_final_epoch": float(final_row.get("current_lr", np.nan)),
                "grad_norm_at_best_epoch": float(best_row.get("grad_norm", np.nan)),
                "grad_norm_at_final_epoch": float(final_row.get("grad_norm", np.nan)),
                "lr_scheduler": LR_SCHEDULER,
                "phase": PHASE,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_stp_v2_convergence_summary.csv", index=False)
    return out


def write_training_report(curve: pd.DataFrame, seed: int) -> None:
    if curve.empty:
        return
    best = curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
    final = curve.iloc[-1]
    final_epoch = int(curve["epoch"].iloc[-1])
    early_stop = final_epoch < MAX_EPOCHS
    max_epoch_hit = final_epoch >= MAX_EPOCHS
    convergence_closed = bool(early_stop and int(best["epoch"]) < 185)
    write_markdown(
        ROOT / f"PHC_STP_V2_TRAINING_CONVERGENCE_REPORT_seed{seed}.md",
        [
            f"# PHC Hidden8 Ablation Training Convergence Report seed {seed}",
            "",
            "This is a from-scratch clean training run. No 05C2/05C3/05C4/05D weights or optimizer state are loaded.",
            "This is a controlled single-variable ablation versus 05C4: `HIDDEN_DIM=8` instead of `HIDDEN_DIM=16`. All other 05C4 training hyperparameters are retained.",
            f"- Best epoch: `{int(best['epoch'])}`.",
            f"- Best validation policy PnL: `{float(best['val_policy_excess_pnl']):,.2f}`.",
            f"- Early stop triggered: `{early_stop}`.",
            f"- Hit max epoch 200: `{max_epoch_hit}`.",
            f"- Convergence closed by strict rule (`early_stop=True and best_epoch<185`): `{convergence_closed}`.",
            f"- LR scheduler: `{LR_SCHEDULER}`.",
            f"- LR at best epoch: `{float(best.get('current_lr', np.nan)):.8f}`.",
            f"- LR at final epoch: `{float(final.get('current_lr', np.nan)):.8f}`.",
            f"- Grad norm at best epoch: `{float(best.get('grad_norm', np.nan)):.6f}`.",
            f"- Grad norm at final epoch: `{float(final.get('grad_norm', np.nan)):.6f}`.",
            f"- Optimizer restarted: `False`.",
            f"- Phase: `{PHASE}`.",
            "",
            "NEW3/NEW4/NEW5 were not used for epoch selection.",
        ],
    )


def train_all(overwrite: bool = False) -> pd.DataFrame:
    ensure_dirs()
    curves = []
    for seed in SEEDS:
        curves.append(train_seed(seed, overwrite=overwrite))
    all_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    all_curves.to_csv(RESULTS / "phc_stp_v2_training_curve_all_seeds.csv", index=False)
    convergence_summary()
    return all_curves


def threshold_grid(scores: np.ndarray) -> list[float]:
    max_scores = np.max(scores, axis=1)
    finite = max_scores[np.isfinite(max_scores)]
    q = np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if len(finite) else np.array([0.0])
    return sorted(set([0.0, *q.tolist()]))


def select_seed_and_threshold() -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    val = p05a.load_sequences("val")
    for seed in SEEDS:
        if not best_ckpt_path(seed).exists():
            raise FileNotFoundError(f"Missing seed checkpoint: {best_ckpt_path(seed)}")
        model, ckpt = load_model(best_ckpt_path(seed))
        scores = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
        for threshold in threshold_grid(scores):
            decision = p05a.policy_from_scores(scores, float(threshold))
            row = p05a.metric_from_decision("val", f"{POLICY_ID}_seed{seed}", val, decision)
            row.update(
                {
                    "seed": seed,
                    "threshold": float(threshold),
                    "best_epoch": int(ckpt["best_epoch"]),
                    "selection_split": "val",
                    "phase": PHASE,
                    "optimizer_restarted": False,
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "phc_stp_v2_validation_threshold_search.csv", index=False)
    best = grid.sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).iloc[0].to_dict()
    selected = {
        "model": POLICY_ID,
        "variant": "PHC-lite + STP-like dynamic synaptic gain clean 200epoch cosine LR",
        "phase": PHASE,
        "selection_split": "val",
        "seed": int(best["seed"]),
        "threshold": float(best["threshold"]),
        "target": "excess_pnl_H20",
        "seq_len": 128,
        "stride": 32,
        "hidden_dim": HIDDEN_DIM,
        "mtl_steps": MTL_STEPS,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lr_initial": LR,
        "lr_min": LR_MIN,
        "lr_scheduler": LR_SCHEDULER,
        "optimizer_restarted": False,
        "validation_metrics": p05a.json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    write_json(RESULTS / "phc_stp_v2_selected_policy.json", selected)
    return grid, selected


def selected_model() -> tuple[PHCLiteSTP, dict[str, Any], dict[str, Any]]:
    selected = read_json(RESULTS / "phc_stp_v2_selected_policy.json")
    model, ckpt = load_model(best_ckpt_path(int(selected["seed"])))
    return model, ckpt, selected


def stp_decision_for_payload(payload: dict[str, Any], model: PHCLiteSTP, ckpt: dict[str, Any], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, threshold), scores


def v1_action_table() -> pd.DataFrame:
    return pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")


def v1_causal_table() -> pd.DataFrame:
    return pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")


def add_v1_action_delta(stp: pd.DataFrame) -> pd.DataFrame:
    v1 = v1_action_table()
    v1 = v1[v1["policy_id"].eq("phc_lite")][["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "phc_lite_v1_action_pnl"})
    out = stp.merge(v1, on="split", how="left")
    out["delta_vs_v1_action_pnl"] = out["total_excess_pnl_H20"] - out["phc_lite_v1_action_pnl"]
    out["delta_pct_vs_v1_action_pnl"] = out["delta_vs_v1_action_pnl"] / out["phc_lite_v1_action_pnl"].replace(0, np.nan)
    return out


def evaluate_oos() -> tuple[pd.DataFrame, pd.DataFrame]:
    model, ckpt, selected = selected_model()
    threshold = float(selected["threshold"])
    rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        payload = p05a.load_sequences(split)
        decision, _ = stp_decision_for_payload(payload, model, ckpt, threshold)
        row = p05a.metric_from_decision(split, POLICY_ID, payload, decision)
        row.update(
            {
                "seed_selected_on_val": int(selected["seed"]),
                "threshold_selected_on_val": threshold,
                "best_epoch": int(ckpt["best_epoch"]),
                "phase": PHASE,
                "optimizer_restarted": False,
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used_for_selection": False,
            }
        )
        rows.append(row)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        code = payload["session_code"].numpy()
        for session_code, session_id in enumerate(payload["session_labels"]):
            mask = code == session_code
            session_rows.append(
                {
                    "split": split,
                    "policy_id": POLICY_ID,
                    "session_id": session_id,
                    "session_excess_pnl_H20": float(pnl[mask].sum()),
                    "selected_action_count": int((decision[mask] != p05a.DECISION_NO_QUOTE).sum()),
                    "state_count": int(mask.sum()),
                }
            )
    stp = add_v1_action_delta(pd.DataFrame(rows))
    base = v1_action_table()
    compare = pd.concat(
        [
            base[base["policy_id"].isin(["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            stp,
        ],
        ignore_index=True,
    )
    stp.to_csv(RESULTS / "phc_stp_v2_oos_eval.csv", index=False)
    compare.to_csv(RESULTS / "phc_stp_v2_vs_v1_hgb_queue.csv", index=False)
    pd.DataFrame(session_rows).to_csv(RESULTS / "phc_stp_v2_session_pnl.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_V2_OOS_EVALUATION_REPORT.md",
        [
            "# PHC STP v2 Clean OOS Evaluation Report",
            "",
            "Seed and threshold are selected on validation only. NEW3/NEW4 are diagnostic OOS only. NEW5 is not used.",
            "",
            stp.to_markdown(index=False),
        ],
    )
    return stp, compare


def matched_random() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    rows = []
    for split in SPLITS:
        payload = p05a.load_sequences(split)
        decision, _ = stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
        _, summary = p05a.matched_random_for_split(split, decision, n_trials=p05a.RANDOM_TRIALS, seed=20260611)
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "policy_id": POLICY_ID,
                "seed_selected_on_val": int(selected["seed"]),
                "threshold_selected_on_val": float(selected["threshold"]),
                "phase": PHASE,
                "optimizer_restarted": False,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_stp_v2_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_V2_MATCHED_RANDOM_REPORT.md",
        [
            "# PHC STP v2 Matched Random Report",
            "",
            "Empirical p-value is the fraction of matched random trials whose PnL >= model PnL.",
            "If the observed fraction is `0.0`, report it as: in 1000 matched random trials, no random trial exceeded the selected policy; therefore empirical p-value < 0.001.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def causal_fill_eval() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows = []
    for split in SPLITS:
        payload = p05a.load_sequences(split)
        decision, _ = stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
        probs = p05a.fill_probabilities(payload, decision, norms)
        fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, FILL_RANDOM_SALT) < probs)
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
        rows.append(
            {
                "split": split,
                "policy_id": POLICY_ID,
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
                "phase": PHASE,
                "new5_used_for_selection": False,
            }
        )
    stp = pd.DataFrame(rows)
    v1 = v1_causal_table()
    v1_short = v1[v1["policy_id"].eq("phc_lite")][["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "phc_lite_v1_causal_fill_pnl"})
    stp = stp.merge(v1_short, on="split", how="left")
    stp["delta_vs_v1_causal_fill_pnl"] = stp["sequential_excess_pnl"] - stp["phc_lite_v1_causal_fill_pnl"]
    stp["delta_pct_vs_v1_causal_fill_pnl"] = stp["delta_vs_v1_causal_fill_pnl"] / stp["phc_lite_v1_causal_fill_pnl"].replace(0, np.nan)
    compare = pd.concat(
        [
            v1[v1["policy_id"].isin(["side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            stp,
        ],
        ignore_index=True,
    )
    stp.to_csv(RESULTS / "phc_stp_v2_causal_fill_eval.csv", index=False)
    compare.to_csv(RESULTS / "phc_stp_v2_causal_fill_vs_v1.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_V2_CAUSAL_FILL_REPORT.md",
        [
            "# PHC STP v2 Causal-fill Report",
            "",
            "Same proxy family as PHC-lite v1: H20 exit, conservative L2 fill, inventory cap 1. Not true queue replay. NEW5 is not used.",
            "",
            stp.to_markdown(index=False),
        ],
    )
    return stp


def seed_stability() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, ckpt = load_model(best_ckpt_path(seed))
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
        for split in SPLITS:
            payload = p05a.load_sequences(split)
            decision = p05a.policy_from_scores(predict_scores(model, payload, target_scale=float(ckpt["target_scale"])), best_threshold)
            metric = p05a.metric_from_decision(split, f"{POLICY_ID}_seed{seed}", payload, decision)
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "best_epoch": int(ckpt.get("best_epoch", -1)),
                    "threshold_selected_on_val_for_seed": best_threshold,
                    "action_pnl": float(metric["total_excess_pnl_H20"]),
                    "selected_action_count": int(metric["selected_action_count"]),
                    "action_pnl_positive": bool(metric["total_excess_pnl_H20"] > 0),
                    "phase": PHASE,
                    "optimizer_restarted": False,
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "phc_stp_v2_seed_stability.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_V2_SEED_STABILITY_REPORT.md",
        [
            "# PHC STP v2 Seed Stability Report",
            "",
            "Each seed uses its own validation-selected threshold. NEW3/NEW4/NEW5 are not used for selection.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def oracle_capture() -> pd.DataFrame:
    model, ckpt, selected = selected_model()
    phc_policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    phc_model, phc_ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        payload = p05a.load_sequences(split)
        decisions = {
            "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
            "ml_hgb_regressor_shallow_convergence_selected": p05a.hgb_decision(payload)[0],
            "phc_lite": p05a.policy_from_scores(p05a.predict_scores(phc_model, payload, target_scale=float(phc_ckpt["target_scale"])), float(phc_policy["threshold"])),
            POLICY_ID: stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))[0],
        }
        bid = payload["y_bid"].numpy().astype(float)
        ask = payload["y_ask"].numpy().astype(float)
        best = np.maximum(bid, ask)
        side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK)
        for policy_id, decision in decisions.items():
            count = int((decision != p05a.DECISION_NO_QUOTE).sum())
            chosen = np.argsort(best)[::-1][:count]
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
    stp = out[out["policy_id"].eq(POLICY_ID)].copy()
    v1 = out[out["policy_id"].eq("phc_lite")][["split", "capture_ratio"]].rename(columns={"capture_ratio": "phc_lite_v1_capture_ratio"})
    stp = stp.merge(v1, on="split", how="left")
    stp["delta_vs_v1_oracle_capture"] = stp["capture_ratio"] - stp["phc_lite_v1_capture_ratio"]
    stp["delta_pct_vs_v1_oracle_capture"] = stp["delta_vs_v1_oracle_capture"] / stp["phc_lite_v1_capture_ratio"].replace(0, np.nan)
    out.to_csv(RESULTS / "phc_stp_v2_window_oracle_capture.csv", index=False)
    stp.to_csv(RESULTS / "phc_stp_v2_oracle_capture_vs_v1.csv", index=False)
    write_markdown(
        ROOT / "PHC_STP_V2_ORACLE_CAPTURE_REPORT.md",
        [
            "# PHC STP v2 Oracle Capture Report",
            "",
            "Oracle capture is matched-count same-universe sequence-window oracle. Oracle uses future labels and is diagnostic only.",
            "",
            stp.to_markdown(index=False),
        ],
    )
    return out


def selected_summary() -> dict[str, Any]:
    return read_json(RESULTS / "phc_stp_v2_selected_policy.json")


def parameter_count_comparison() -> pd.DataFrame:
    feature_dim = len(p05a.PHC_FEATURES)
    hidden8 = PHCLiteSTP(feature_dim, hidden_dim=HIDDEN_DIM, mtl_steps=MTL_STEPS, dropout=0.0)
    hidden16 = PHCLiteSTP(feature_dim, hidden_dim=16, mtl_steps=MTL_STEPS, dropout=0.0)
    count8 = count_parameters(hidden8)
    count16 = count_parameters(hidden16)
    out = pd.DataFrame(
        [
            {
                "model": "phc_stp_v2_hidden8",
                "hidden_dim": HIDDEN_DIM,
                "parameter_count": count8,
                "feature_dim": feature_dim,
                "mtl_steps": MTL_STEPS,
            },
            {
                "model": "phc_stp_v2_cosine_lr_hidden16",
                "hidden_dim": 16,
                "parameter_count": count16,
                "feature_dim": feature_dim,
                "mtl_steps": MTL_STEPS,
            },
        ]
    )
    out["delta_vs_hidden16"] = out["parameter_count"] - count16
    out["ratio_vs_hidden16"] = out["parameter_count"] / count16 if count16 else np.nan
    out.to_csv(RESULTS / "phc_hidden8_parameter_count_comparison.csv", index=False)
    return out


def load_hidden16_summary() -> pd.DataFrame:
    path = PHASE05C4 / "results" / "phc_stp_v2_vs_phc_lite_v1_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen 05C4 hidden16 summary: {path}")
    base = pd.read_csv(path)
    required = {"split", "action_pnl", "causal_fill_pnl", "oracle_capture"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"05C4 hidden16 summary missing columns: {missing}")
    return base[["split", "action_pnl", "causal_fill_pnl", "oracle_capture"]].copy()


def compare_vs_hidden16(current: pd.DataFrame) -> pd.DataFrame:
    hidden16 = load_hidden16_summary().rename(
        columns={
            "action_pnl": "action_pnl_hidden16",
            "causal_fill_pnl": "causal_fill_pnl_hidden16",
            "oracle_capture": "oracle_capture_hidden16",
        }
    )
    hidden8 = current[["split", "action_pnl", "causal_fill_pnl", "oracle_capture"]].rename(
        columns={
            "action_pnl": "action_pnl_hidden8",
            "causal_fill_pnl": "causal_fill_pnl_hidden8",
            "oracle_capture": "oracle_capture_hidden8",
        }
    )
    out = hidden8.merge(hidden16, on="split", how="outer")
    for metric in ["action_pnl", "causal_fill_pnl", "oracle_capture"]:
        out[f"{metric}_delta_hidden8_minus_hidden16"] = out[f"{metric}_hidden8"] - out[f"{metric}_hidden16"]
        out[f"{metric}_delta_pct_hidden8_minus_hidden16"] = (
            out[f"{metric}_delta_hidden8_minus_hidden16"] / out[f"{metric}_hidden16"].replace(0, np.nan)
        )
    out.to_csv(RESULTS / "phc_hidden8_vs_hidden16_summary.csv", index=False)
    return out


def compare_seed_stability_vs_hidden16() -> tuple[pd.DataFrame, pd.DataFrame]:
    current_path = RESULTS / "phc_stp_v2_seed_stability.csv"
    hidden16_path = PHASE05C4 / "results" / "phc_stp_v2_seed_stability.csv"
    if not current_path.exists():
        raise FileNotFoundError(f"Missing current seed stability file: {current_path}")
    if not hidden16_path.exists():
        raise FileNotFoundError(f"Missing frozen 05C4 seed stability file: {hidden16_path}")
    current = pd.read_csv(current_path)[["seed", "split", "best_epoch", "threshold_selected_on_val_for_seed", "action_pnl", "action_pnl_positive"]]
    hidden16 = pd.read_csv(hidden16_path)[["seed", "split", "best_epoch", "threshold_selected_on_val_for_seed", "action_pnl", "action_pnl_positive"]]
    current = current.rename(
        columns={
            "best_epoch": "best_epoch_hidden8",
            "threshold_selected_on_val_for_seed": "threshold_hidden8",
            "action_pnl": "action_pnl_hidden8",
            "action_pnl_positive": "action_pnl_positive_hidden8",
        }
    )
    hidden16 = hidden16.rename(
        columns={
            "best_epoch": "best_epoch_hidden16",
            "threshold_selected_on_val_for_seed": "threshold_hidden16",
            "action_pnl": "action_pnl_hidden16",
            "action_pnl_positive": "action_pnl_positive_hidden16",
        }
    )
    by_seed = current.merge(hidden16, on=["seed", "split"], how="outer")
    by_seed["action_pnl_delta_hidden8_minus_hidden16"] = by_seed["action_pnl_hidden8"] - by_seed["action_pnl_hidden16"]
    by_seed["action_pnl_delta_pct_hidden8_minus_hidden16"] = (
        by_seed["action_pnl_delta_hidden8_minus_hidden16"] / by_seed["action_pnl_hidden16"].replace(0, np.nan)
    )
    by_seed.to_csv(RESULTS / "phc_hidden8_vs_hidden16_seed_stability_by_seed.csv", index=False)
    rows = []
    for split, grp in by_seed.groupby("split", dropna=False):
        rows.append(
            {
                "split": split,
                "hidden8_seed_count": int(grp["action_pnl_hidden8"].notna().sum()),
                "hidden16_seed_count": int(grp["action_pnl_hidden16"].notna().sum()),
                "hidden8_min_action_pnl": float(grp["action_pnl_hidden8"].min()),
                "hidden16_min_action_pnl": float(grp["action_pnl_hidden16"].min()),
                "hidden8_mean_action_pnl": float(grp["action_pnl_hidden8"].mean()),
                "hidden16_mean_action_pnl": float(grp["action_pnl_hidden16"].mean()),
                "mean_action_pnl_delta_hidden8_minus_hidden16": float(
                    grp["action_pnl_delta_hidden8_minus_hidden16"].mean()
                ),
                "hidden8_all_seed_action_positive": bool(grp["action_pnl_positive_hidden8"].fillna(False).astype(bool).all()),
                "hidden16_all_seed_action_positive": bool(grp["action_pnl_positive_hidden16"].fillna(False).astype(bool).all()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "phc_hidden8_vs_hidden16_seed_stability_summary.csv", index=False)
    return by_seed, summary


def classify_hidden8_vs_hidden16(hidden16_diff: pd.DataFrame) -> tuple[str, str]:
    metric_delta_cols = [
        "action_pnl_delta_hidden8_minus_hidden16",
        "causal_fill_pnl_delta_hidden8_minus_hidden16",
        "oracle_capture_delta_hidden8_minus_hidden16",
    ]
    metric_pct_cols = [
        "action_pnl_delta_pct_hidden8_minus_hidden16",
        "causal_fill_pnl_delta_pct_hidden8_minus_hidden16",
        "oracle_capture_delta_pct_hidden8_minus_hidden16",
    ]
    deltas = hidden16_diff[metric_delta_cols].to_numpy(dtype=float)
    pct_deltas = hidden16_diff[metric_pct_cols].to_numpy(dtype=float)
    if np.all(deltas >= 0):
        return "improves", "All val/NEW3/NEW4 action PnL, causal-fill PnL, and oracle capture deltas versus hidden16 are non-negative."
    if np.all(pct_deltas >= -0.02):
        return "matches", "No val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus hidden16."
    return "regresses", "At least one val/NEW3/NEW4 action PnL, causal-fill PnL, or oracle capture metric regresses by more than 2% versus hidden16."


def final_report() -> str:
    conv = convergence_summary()
    oos = pd.read_csv(RESULTS / "phc_stp_v2_oos_eval.csv")
    causal = pd.read_csv(RESULTS / "phc_stp_v2_causal_fill_eval.csv")
    matched = pd.read_csv(RESULTS / "phc_stp_v2_matched_random_summary.csv")
    seed = pd.read_csv(RESULTS / "phc_stp_v2_seed_stability.csv")
    oracle = pd.read_csv(RESULTS / "phc_stp_v2_oracle_capture_vs_v1.csv")
    selected = selected_summary()
    v1_action = v1_action_table()
    v1_action = v1_action[v1_action["policy_id"].eq("phc_lite")][["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "phc_lite_v1_action_pnl"})
    v1_causal = v1_causal_table()
    v1_causal = v1_causal[v1_causal["policy_id"].eq("phc_lite")][["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "phc_lite_v1_causal_fill_pnl"})
    compare = oos[["split", "total_excess_pnl_H20", "phc_lite_v1_action_pnl", "delta_vs_v1_action_pnl", "delta_pct_vs_v1_action_pnl"]].merge(
        causal[["split", "sequential_excess_pnl", "phc_lite_v1_causal_fill_pnl", "delta_vs_v1_causal_fill_pnl", "delta_pct_vs_v1_causal_fill_pnl"]],
        on="split",
        how="left",
    ).merge(
        oracle[["split", "capture_ratio", "phc_lite_v1_capture_ratio", "delta_vs_v1_oracle_capture", "delta_pct_vs_v1_oracle_capture"]],
        on="split",
        how="left",
    )
    compare = compare.rename(
        columns={
            "total_excess_pnl_H20": "action_pnl",
            "sequential_excess_pnl": "causal_fill_pnl",
            "capture_ratio": "oracle_capture",
        }
    )
    compare.to_csv(RESULTS / "phc_stp_v2_vs_phc_lite_v1_summary.csv", index=False)
    params = parameter_count_comparison()
    hidden16_diff = compare_vs_hidden16(compare)
    seed_by_seed, seed_summary = compare_seed_stability_vs_hidden16()
    verdict, verdict_reason = classify_hidden8_vs_hidden16(hidden16_diff)
    gates = []
    gate1 = bool(conv["convergence_closed"].astype(bool).all()) if len(conv) == len(SEEDS) else False
    gates.append({"gate": "Convergence closed", "pass": gate1, "rule": "all seeds early_stop=True and best_epoch<185"})
    gate2 = bool(seed["action_pnl_positive"].astype(bool).all())
    gates.append({"gate": "Seed stability", "pass": gate2, "rule": "all seeds x splits action PnL > 0"})
    gate3 = bool(matched["beats_random_95p"].astype(bool).all())
    gates.append({"gate": "Matched random", "pass": gate3, "rule": "val/NEW3/NEW4 all beat random 95p"})
    gate4 = bool((compare[compare["split"].isin(["new3", "new4"])]["delta_vs_v1_action_pnl"] >= 0).all())
    gates.append({"gate": "OOS improvement", "pass": gate4, "rule": "NEW3/NEW4 action PnL >= PHC-lite v1"})
    gate5 = bool((compare["delta_vs_v1_causal_fill_pnl"] >= 0).all())
    gates.append({"gate": "Causal-fill improvement", "pass": gate5, "rule": "val/NEW3/NEW4 sequential PnL >= PHC-lite v1"})
    gate6 = bool((compare["delta_vs_v1_oracle_capture"] >= 0).all())
    gates.append({"gate": "Oracle capture improvement", "pass": gate6, "rule": "val/NEW3/NEW4 capture ratio >= PHC-lite v1"})
    gates_df = pd.DataFrame(gates)
    gates_df.to_csv(RESULTS / "phc_stp_v2_adoption_gates.csv", index=False)
    if gates_df["pass"].astype(bool).all():
        status = "stp_v2_adopted_as_best_sequence_model"
    elif not gate1:
        status = "stp_v2_convergence_still_incomplete"
    else:
        status = "stp_v2_convergence_closed_no_improvement"
    hidden8_status = f"hidden8_{verdict}_vs_hidden16_baseline"
    write_markdown(
        ROOT / "PHC_HIDDEN8_ABLATION_REPORT.md",
        [
            "# PHC Hidden8 Ablation Report",
            "",
            f"Final verdict: hidden8 `{verdict}` vs hidden16 baseline on val/NEW3/NEW4.",
            "",
            f"Verdict status: `{hidden8_status}`.",
            "",
            verdict_reason,
            "",
            "## Protocol",
            "",
            "- Phase: `05e`.",
            "- Model: `PHCLiteSTP` imported from Phase 05C2 source code; architecture unchanged.",
            "- Training: from-scratch random initialization for each seed. No 05C2/05C3/05C4/05D weights or optimizer state loaded.",
            "- Controlled single-variable change versus 05C4: `HIDDEN_DIM=8` instead of `HIDDEN_DIM=16`.",
            "- Unchanged from 05C4: seeds, max epochs, patience, batch size, LR, LR_MIN, CosineAnnealingLR, weight decay, feature list, target, and causal-fill salt.",
            "- Optimizer and scheduler states are saved in both best and last checkpoints.",
            "- Selection split: validation only.",
            "- NEW3/NEW4: diagnostic OOS only.",
            "- NEW5: not used.",
            "- No files under `n2ew_era/06f_phc_lite_manual_hls/` are modified or evaluated in this phase.",
            "- Material-match criterion in this report: no key metric regresses by more than 2% versus hidden16.",
            "",
            "## Parameter Count Comparison",
            "",
            params.to_markdown(index=False),
            "",
            "## Selected Policy",
            "",
            "```json",
            json.dumps(selected, indent=2),
            "```",
            "",
            "## 7a. Convergence Status",
            "",
            conv[["seed", "best_epoch", "early_stop", "max_epoch_hit", "convergence_closed"]].to_markdown(index=False),
            "",
            "Convergence is closed only if `early_stop=True` and `best_epoch<185` for every seed.",
            "",
            "## 7b. vs PHC-lite v1 Comparison",
            "",
            compare.to_markdown(index=False),
            "",
            "## 7c. Diagnostic Gates vs PHC-lite v1",
            "",
            gates_df.to_markdown(index=False),
            "",
            f"Diagnostic gate status versus PHC-lite v1: `{status}`. This is not the final hidden8-vs-hidden16 verdict.",
            "",
            "## 7d. vs Frozen 05C4 Hidden16",
            "",
            hidden16_diff.to_markdown(index=False),
            "",
            "The hidden16 baseline values are read mechanically from `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_vs_phc_lite_v1_summary.csv`.",
            "",
            "## 7e. Seed Stability vs Hidden16",
            "",
            seed_summary.to_markdown(index=False),
            "",
            "### Per-Seed Details",
            "",
            seed_by_seed.to_markdown(index=False),
            "",
            "## 7f. Matched Random Summary",
            "",
            matched.to_markdown(index=False),
            "",
            "## 7g. Explicit Verdict",
            "",
            f"`hidden8 {verdict} vs hidden16 baseline on val/NEW3/NEW4`.",
            "",
            verdict_reason,
            "",
            "## 7h. Forbidden Claims",
            "",
            "- Not full biological STP",
            "- Not fresh NEW5 holdout",
            "- Not true exchange replay",
            "- Not FPGA PnL improvement proof",
            "- Not real-world profitability proof",
            "- Does not estimate or claim new HLS latency",
            "- Does not retrain or replace the deployed 06F HLS weights",
            "- phase = \"05e\"",
            "- optimizer_restarted = False",
            f"- lr_scheduler = \"{LR_SCHEDULER}\"",
        ],
    )
    return hidden8_status


def run_all() -> None:
    ensure_dirs()
    train_all(overwrite=False)
    select_seed_and_threshold()
    evaluate_oos()
    matched_random()
    causal_fill_eval()
    seed_stability()
    oracle_capture()
    print(final_report())
    print(ROOT / "PHC_HIDDEN8_ABLATION_REPORT.md")
