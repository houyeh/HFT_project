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
ROOT = PROJECT_ROOT / "n2ew_era" / "07a_tcn_feedforward_phc_tradeoff"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C0 = PROJECT_ROOT / "n2ew_era" / "05c0_freeze_phc_lite_v1"
LATENCY_REF = PROJECT_ROOT / "n2ew_era" / "06o_phc_lite_latency_path_to_1us" / "PHC_LATENCY_PATH_TO_1US_REPORT.md"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import common_05a as p05a  # noqa: E402
from src.models.tcn_prepad import TCNPrePadRegressor  # noqa: E402

PHASE = "07a"
SEQ_LEN = 128
TOTAL_PAD = TCNPrePadRegressor.TOTAL_PAD
SEQ_LEN_IN = TCNPrePadRegressor.SEQ_LEN_IN
CHANNELS = 16
KERNEL_SIZE = 3
DROPOUT = 0.1
NUM_OUTPUTS = 2
MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
SEEDS = [0, 1, 2]
SPLITS = ["val", "new3", "new4"]
VARIANTS = ["scratch", "distilled"]
POLICY_IDS = {
    "scratch": "tcn_prepad_scratch_16feat",
    "distilled": "tcn_prepad_distilled_16feat",
}
DISTILL_TEACHER_WEIGHT = 0.5
DISTILL_LABEL_WEIGHT = 0.5
FILL_RANDOM_SALT = "07a_tcn_prepad_conservative_l2_stress"


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(p05a.json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(out.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def policy_id(variant: str) -> str:
    return POLICY_IDS[variant]


def best_ckpt_path(variant: str, seed: int) -> Path:
    return MODELS / f"{policy_id(variant)}_seed{seed}_best_val.pt"


def last_ckpt_path(variant: str, seed: int) -> Path:
    return MODELS / f"{policy_id(variant)}_seed{seed}_last.pt"


def curve_path(variant: str, seed: int) -> Path:
    return RESULTS / f"{policy_id(variant)}_training_curve_seed{seed}.csv"


def target_scale() -> float:
    return float(p05a.load_scaler()["target_scale_train_std_no_mean_shift"])


def build_model() -> TCNPrePadRegressor:
    model = TCNPrePadRegressor(
        num_inputs=len(p05a.PHC_FEATURES),
        num_outputs=NUM_OUTPUTS,
        channels=CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    )
    if model.receptive_field() != 125:
        raise RuntimeError(f"TCNPrePadRegressor RF sanity failed: {model.receptive_field()} != 125")
    return model


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class TCNWindowDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any], scale: float, teacher_scores: np.ndarray | None = None) -> None:
        self.x = payload["x"]
        self.y = torch.stack([payload["y_bid"], payload["y_ask"]], dim=1).float() / float(scale)
        self.teacher = None
        if teacher_scores is not None:
            self.teacher = torch.from_numpy(np.asarray(teacher_scores, dtype=np.float32)) / float(scale)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = self.x[idx].float().permute(1, 0).contiguous()
        pad = torch.zeros(x.shape[0], TOTAL_PAD, dtype=x.dtype)
        x = torch.cat([pad, x], dim=1).contiguous()
        if self.teacher is None:
            return x, self.y[idx]
        return x, self.y[idx], self.teacher[idx]


def teacher_scores_path(split: str) -> Path:
    return RESULTS / f"phc_lite_v1_teacher_scores_{split}.npy"


def load_or_build_teacher_scores(split: str) -> np.ndarray:
    ensure_dirs()
    path = teacher_scores_path(split)
    if path.exists():
        return np.load(path)
    model, ckpt = p05a.load_model()
    payload = p05a.load_sequences(split)
    scores = p05a.predict_scores(model, payload, batch_size=2048, target_scale=float(ckpt["target_scale"]))
    np.save(path, scores.astype(np.float32))
    return scores.astype(np.float32)


def make_checkpoint(
    variant: str,
    model: TCNPrePadRegressor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    seed: int,
    epoch_key: str,
    epoch_value: int,
    scale: float,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "variant": variant,
        "policy_id": policy_id(variant),
        "feature_names": list(p05a.PHC_FEATURES),
        "feature_dim": len(p05a.PHC_FEATURES),
        "channels": CHANNELS,
        "dilations": list(TCNPrePadRegressor.DILATIONS),
        "kernel_size": KERNEL_SIZE,
        "seq_length": SEQ_LEN,
        "seq_length_in": SEQ_LEN_IN,
        "total_prepad": TOTAL_PAD,
        "receptive_field": model.receptive_field(),
        "num_outputs": NUM_OUTPUTS,
        "dropout": DROPOUT,
        "seed": seed,
        epoch_key: int(epoch_value),
        "target_scale": float(scale),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lr_initial": LR,
        "lr_min": LR_MIN,
        "lr_scheduler": "CosineAnnealingLR",
        "distill_teacher": "phc_lite_v1" if variant == "distilled" else None,
        "distill_teacher_weight": DISTILL_TEACHER_WEIGHT if variant == "distilled" else 0.0,
        "distill_label_weight": DISTILL_LABEL_WEIGHT if variant == "distilled" else 1.0,
        "phase": PHASE,
        "optimizer_restarted": False,
        "selection_metric": "validation_policy_excess_pnl_at_threshold_0",
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }


def load_model(variant: str, checkpoint_path: Path) -> tuple[TCNPrePadRegressor, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("phase") != PHASE:
        raise RuntimeError(f"Refusing non-07A checkpoint: {checkpoint_path}")
    if ckpt.get("variant") != variant:
        raise RuntimeError(f"Variant mismatch in checkpoint {checkpoint_path}: {ckpt.get('variant')} != {variant}")
    model = build_model()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_scores(
    model: TCNPrePadRegressor,
    payload: dict[str, Any],
    scale: float,
    batch_size: int = 2048,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    x = payload["x"]
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device=device, dtype=torch.float32).permute(0, 2, 1).contiguous()
            zeros = torch.zeros(xb.size(0), xb.size(1), TOTAL_PAD, device=xb.device, dtype=xb.dtype)
            xb = torch.cat([zeros, xb], dim=2).contiguous()
            outs.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0) * float(scale)


def train_seed(variant: str, seed: int, overwrite: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if best_ckpt_path(variant, seed).exists() and curve_path(variant, seed).exists() and not overwrite:
        return pd.read_csv(curve_path(variant, seed))
    set_seed(seed)
    scale = target_scale()
    train_payload = p05a.load_sequences("train")
    val_payload = p05a.load_sequences("val")
    teacher = load_or_build_teacher_scores("train") if variant == "distilled" else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    huber = torch.nn.HuberLoss(delta=1.0)
    loader = torch.utils.data.DataLoader(
        TCNWindowDataset(train_payload, scale, teacher_scores=teacher),
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
        for batch in loader:
            if variant == "distilled":
                xb, yb, tb = batch
                tb = tb.to(device=device, dtype=torch.float32)
            else:
                xb, yb = batch
                tb = None
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss_label = huber(pred, yb)
            if tb is not None:
                loss_teacher = huber(pred, tb)
                loss = DISTILL_LABEL_WEIGHT * loss_label + DISTILL_TEACHER_WEIGHT * loss_teacher
            else:
                loss_teacher = torch.tensor(0.0, device=device)
                loss = loss_label
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(grad_norm)
        val_scores = predict_scores(model, val_payload, scale=scale, batch_size=BATCH_SIZE, device=device)
        train_scores = predict_scores(model, train_payload, scale=scale, batch_size=BATCH_SIZE, device=device)
        val_y = np.column_stack([val_payload["y_bid"].numpy(), val_payload["y_ask"].numpy()])
        train_y = np.column_stack([train_payload["y_bid"].numpy(), train_payload["y_ask"].numpy()])
        val_decision = p05a.policy_from_scores(val_scores, 0.0)
        val_pnl = float(p05a.decision_pnl_from_payload(val_payload, val_decision).sum())
        val_selected = int((val_decision != p05a.DECISION_NO_QUOTE).sum())
        row = {
            "epoch": epoch,
            "seed": seed,
            "variant": variant,
            "policy_id": policy_id(variant),
            "train_loss": float(np.mean(losses)),
            "val_loss": p05a.huber_np((val_scores - val_y) / scale),
            "train_huber_loss": p05a.huber_np((train_scores - train_y) / scale),
            "val_huber_loss": p05a.huber_np((val_scores - val_y) / scale),
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
            "lr_scheduler": "CosineAnnealingLR",
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "early_stop_counter": early_counter,
            "optimizer_restarted": False,
            "max_epoch_hit": False,
            "parameter_count": count_parameters(model),
            "device": device,
            "channels": CHANNELS,
            "kernel_size": KERNEL_SIZE,
            "dropout": DROPOUT,
            "seq_len": SEQ_LEN,
            "seq_len_in": SEQ_LEN_IN,
            "batch_size": BATCH_SIZE,
            "phase": PHASE,
            "distill_teacher_weight": DISTILL_TEACHER_WEIGHT if variant == "distilled" else 0.0,
            "new3_used_for_selection": False,
            "new4_used_for_selection": False,
            "new5_used_for_selection": False,
        }
        rows.append(row)
        if val_pnl > best_metric + 1e-3:
            best_metric = val_pnl
            best_epoch = epoch
            early_counter = 0
            torch.save(
                make_checkpoint(variant, model, optimizer, scheduler, seed, "best_epoch", best_epoch, scale),
                best_ckpt_path(variant, seed),
            )
        else:
            early_counter += 1
        torch.save(make_checkpoint(variant, model, optimizer, scheduler, seed, "epoch", epoch, scale), last_ckpt_path(variant, seed))
        curve = pd.DataFrame(rows)
        curve.loc[curve.index[-1], "early_stop_counter"] = early_counter
        curve.to_csv(curve_path(variant, seed), index=False)
        print(
            f"07a variant={variant} seed={seed} epoch={epoch} lr={current_lr:.8f} "
            f"val_pnl={val_pnl:.2f} selected={val_selected} grad={row['grad_norm']:.4f}",
            flush=True,
        )
        scheduler.step()
        if early_counter >= PATIENCE:
            break
    if rows:
        curve = pd.DataFrame(rows)
        curve.loc[curve.index[-1], "max_epoch_hit"] = bool(int(curve["epoch"].iloc[-1]) >= MAX_EPOCHS)
        curve.to_csv(curve_path(variant, seed), index=False)
        write_training_report(variant, seed, curve)
        plot_training_curve(variant, seed, curve)
    return pd.DataFrame(rows)


def train_all_seeds() -> pd.DataFrame:
    curves = []
    for variant in VARIANTS:
        for seed in SEEDS:
            curves.append(train_seed(variant, seed))
    out = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    out.to_csv(RESULTS / "tcn_prepad_training_curve_all_variants.csv", index=False)
    return out


def write_training_report(variant: str, seed: int, curve: pd.DataFrame) -> None:
    if curve.empty:
        return
    best = curve.sort_values(["val_policy_excess_pnl", "epoch"], ascending=[False, True]).iloc[0]
    write_markdown(
        ROOT / f"TCN_PREPAD_{variant.upper()}_TRAINING_CONVERGENCE_REPORT_seed{seed}.md",
        [
            f"# TCN PrePad {variant} Training Report seed {seed}",
            "",
            f"- Variant: `{policy_id(variant)}`.",
            f"- Best epoch: `{int(best['epoch'])}`.",
            f"- Best validation policy PnL at threshold 0: `{float(best['val_policy_excess_pnl']):,.2f}`.",
            f"- Val Spearman bid/ask: `{float(best['val_spearman_bid']):.6f}` / `{float(best['val_spearman_ask']):.6f}`.",
            f"- Val PR-AUC bid/ask: `{float(best['val_pr_auc_bid']):.6f}` / `{float(best['val_pr_auc_ask']):.6f}`.",
            f"- Early stop triggered: `{int(curve['epoch'].iloc[-1]) < MAX_EPOCHS}`.",
            f"- Max epoch hit: `{bool(curve['max_epoch_hit'].iloc[-1]) if 'max_epoch_hit' in curve else False}`.",
            "- NEW3/NEW4/NEW5 were not used for training, epoch selection, or threshold selection.",
        ],
    )


def plot_training_curve(variant: str, seed: int, curve: pd.DataFrame) -> None:
    ensure_dirs()
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["epoch"], curve["train_loss"], label="train")
    ax.plot(curve["epoch"], curve["val_loss"], label="val")
    ax.set_title(f"{policy_id(variant)} loss seed {seed}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{policy_id(variant)}_loss_seed{seed}.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["epoch"], curve["val_policy_excess_pnl"], marker="o", markersize=2)
    ax.set_title(f"{policy_id(variant)} validation policy PnL seed {seed}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("excess PnL H20")
    fig.tight_layout()
    fig.savefig(FIGURES / f"{policy_id(variant)}_val_pnl_seed{seed}.png", dpi=160)
    plt.close(fig)


def select_policy() -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    rows = []
    selected: dict[str, Any] = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            ckpt_path = best_ckpt_path(variant, seed)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            model, ckpt = load_model(variant, ckpt_path)
            val = p05a.load_sequences("val")
            scores = predict_scores(model, val, scale=float(ckpt["target_scale"]))
            max_scores = np.max(scores, axis=1)
            finite = max_scores[np.isfinite(max_scores)]
            thresholds = sorted(
                set([0.0, *np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).tolist()])
            )
            for threshold in thresholds:
                decision = p05a.policy_from_scores(scores, float(threshold))
                row = p05a.metric_from_decision("val", policy_id(variant), val, decision)
                row.update(
                    {
                        "variant": variant,
                        "seed": seed,
                        "threshold": float(threshold),
                        "selection_split": "val",
                        "new3_used_for_selection": False,
                        "new4_used_for_selection": False,
                        "new5_used_for_selection": False,
                    }
                )
                rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["variant", "total_excess_pnl_H20", "selected_action_count"], ascending=[True, False, True])
    grid.to_csv(RESULTS / "tcn_prepad_validation_threshold_search.csv", index=False)
    for variant in VARIANTS:
        best = grid[grid["variant"].eq(variant)].sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).iloc[0].to_dict()
        ckpt = torch.load(best_ckpt_path(variant, int(best["seed"])), map_location="cpu", weights_only=False)
        payload = {
            "model": policy_id(variant),
            "variant": variant,
            "selection_split": "val",
            "target": "excess_pnl_H20",
            "feature_count": len(p05a.PHC_FEATURES),
            "features": list(p05a.PHC_FEATURES),
            "seq_len": SEQ_LEN,
            "seq_len_in": SEQ_LEN_IN,
            "total_prepad": TOTAL_PAD,
            "channels": CHANNELS,
            "kernel_size": KERNEL_SIZE,
            "receptive_field": int(ckpt["receptive_field"]),
            "seed": int(best["seed"]),
            "checkpoint": str(best_ckpt_path(variant, int(best["seed"]))),
            "best_epoch": int(ckpt.get("best_epoch", -1)),
            "threshold": float(best["threshold"]),
            "target_scale": float(ckpt["target_scale"]),
            "validation_metrics": p05a.json_sanitize(best),
            "new3_used_for_selection": False,
            "new4_used_for_selection": False,
            "new5_used_for_selection": False,
        }
        selected[variant] = payload
        write_json(RESULTS / f"{policy_id(variant)}_selected_policy.json", payload)
    write_json(RESULTS / "tcn_prepad_selected_policies.json", selected)
    return grid, selected


def selected_model(variant: str) -> tuple[TCNPrePadRegressor, dict[str, Any], dict[str, Any]]:
    selected = read_json(RESULTS / f"{policy_id(variant)}_selected_policy.json")
    model, ckpt = load_model(variant, Path(selected["checkpoint"]))
    return model, ckpt, selected


def tcn_decision_for_payload(variant: str, payload: dict[str, Any], model: TCNPrePadRegressor, ckpt: dict[str, Any], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = predict_scores(model, payload, scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, threshold), scores


def phc_lite_v1_decision(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model, ckpt = p05a.load_model()
    policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    scores = p05a.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, float(policy["threshold"])), scores


def evaluate_oos() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    session_rows = []
    selected_payload = {variant: selected_model(variant) for variant in VARIANTS}
    for split in SPLITS:
        payload = p05a.load_sequences(split)
        decisions = {
            "no_quote": p05a.no_quote_decision(payload),
            "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(payload),
            "phc_lite_v1_teacher": phc_lite_v1_decision(payload)[0],
        }
        for variant, (model, ckpt, selected) in selected_payload.items():
            decision, _ = tcn_decision_for_payload(variant, payload, model, ckpt, float(selected["threshold"]))
            decisions[policy_id(variant)] = decision
        for pid, decision in decisions.items():
            metric = p05a.metric_from_decision(split, pid, payload, decision)
            if pid in POLICY_IDS.values():
                variant = next(v for v, p in POLICY_IDS.items() if p == pid)
                metric["variant"] = variant
                metric["seed"] = int(selected_payload[variant][2]["seed"])
            rows.append(metric)
            pnl = p05a.decision_pnl_from_payload(payload, decision)
            code = payload["session_code"].numpy()
            for session_code, session_id in enumerate(payload["session_labels"]):
                mask = code == session_code
                session_rows.append(
                    {
                        "split": split,
                        "policy_id": pid,
                        "session_id": session_id,
                        "session_excess_pnl_H20": float(pnl[mask].sum()),
                        "selected_action_count": int((decision[mask] != p05a.DECISION_NO_QUOTE).sum()),
                        "state_count": int(mask.sum()),
                    }
                )
    metrics = pd.DataFrame(rows)
    sessions = pd.DataFrame(session_rows)
    metrics.to_csv(RESULTS / "tcn_prepad_oos_eval.csv", index=False)
    sessions.to_csv(RESULTS / "tcn_prepad_session_pnl.csv", index=False)
    write_markdown(
        ROOT / "TCN_PREPAD_OOS_EVALUATION_REPORT.md",
        [
            "# TCN PrePad OOS Evaluation Report",
            "",
            "Evaluation uses the Phase 05A sequence-window universe and the same action-value decision rule as PHC-lite. NEW5 is not loaded.",
            "",
            md_table(metrics),
        ],
    )
    return metrics, sessions


def matched_random(n_trials: int = p05a.RANDOM_TRIALS) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        model, ckpt, selected = selected_model(variant)
        for split in SPLITS:
            payload = p05a.load_sequences(split)
            decision, _ = tcn_decision_for_payload(variant, payload, model, ckpt, float(selected["threshold"]))
            _, summary = p05a.matched_random_for_split(split, decision, n_trials=n_trials, seed=20260623 + 1000 * SEEDS.index(int(selected["seed"])))
            row = summary.iloc[0].to_dict()
            row.update({"policy_id": policy_id(variant), "variant": variant, "seed": int(selected["seed"])})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_prepad_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "TCN_PREPAD_MATCHED_RANDOM_REPORT.md",
        [
            "# TCN PrePad Matched Random Report",
            "",
            "Matched random uses the Phase 05A matched-count/session/side/spread-bucket method. Empirical p-value is the fraction of random trials with PnL >= model PnL.",
            "",
            md_table(out),
        ],
    )
    return out


def causal_fill_eval() -> pd.DataFrame:
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows = []
    for variant in VARIANTS:
        model, ckpt, selected = selected_model(variant)
        for split in SPLITS:
            payload = p05a.load_sequences(split)
            decision, _ = tcn_decision_for_payload(variant, payload, model, ckpt, float(selected["threshold"]))
            probs = p05a.fill_probabilities(payload, decision, norms)
            fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, FILL_RANDOM_SALT) < probs)
            pnl = p05a.decision_pnl_from_payload(payload, decision)
            total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id(variant),
                    "variant": variant,
                    "seed": int(selected["seed"]),
                    "horizon": p05a.HORIZON,
                    "fill_model": "conservative_l2_stress",
                    "inventory_cap": 1,
                    "sequential_excess_pnl": float(total),
                    "fill_count": int(fill_count),
                    "forced_no_quote_count": int(forced),
                    "max_abs_inventory": int(max_inv),
                    "max_drawdown": float(max_dd),
                    "win_rate": win / max(fill_count, 1) if fill_count else 0.0,
                    "loss_rate": loss / max(fill_count, 1) if fill_count else 0.0,
                    "avg_trade_excess_pnl": total / max(fill_count, 1) if fill_count else 0.0,
                    "fill_uses_future_pnl": False,
                    "new5_used": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_prepad_causal_fill_eval.csv", index=False)
    write_markdown(
        ROOT / "TCN_PREPAD_CAUSAL_FILL_REPORT.md",
        [
            "# TCN PrePad Causal-Fill Report",
            "",
            "Same proxy family as PHC-lite: H20 exit, conservative L2 fill, inventory cap 1. This is not true exchange replay. NEW5 is not used.",
            "",
            md_table(out),
        ],
    )
    return out


def seed_stability(n_trials: int = p05a.RANDOM_TRIALS) -> pd.DataFrame:
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            ckpt_path = best_ckpt_path(variant, seed)
            if not ckpt_path.exists():
                continue
            model, ckpt = load_model(variant, ckpt_path)
            val = p05a.load_sequences("val")
            val_scores = predict_scores(model, val, scale=float(ckpt["target_scale"]))
            finite = np.max(val_scores, axis=1)
            thresholds = sorted(set([0.0, *np.quantile(finite[np.isfinite(finite)], [0.5, 0.75, 0.9, 0.95]).tolist()]))
            best_thr = 0.0
            best_pnl = -float("inf")
            for thr in thresholds:
                decision = p05a.policy_from_scores(val_scores, float(thr))
                pnl = float(p05a.decision_pnl_from_payload(val, decision).sum())
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_thr = float(thr)
            for split in SPLITS:
                payload = p05a.load_sequences(split)
                scores = predict_scores(model, payload, scale=float(ckpt["target_scale"]))
                decision = p05a.policy_from_scores(scores, best_thr)
                metric = p05a.metric_from_decision(split, policy_id(variant), payload, decision)
                _, random_summary = p05a.matched_random_for_split(split, decision, n_trials=n_trials, seed=20260623 + seed * 10000)
                probs = p05a.fill_probabilities(payload, decision, norms)
                fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, f"{FILL_RANDOM_SALT}_seed{seed}") < probs)
                pnl = p05a.decision_pnl_from_payload(payload, decision)
                total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
                rows.append(
                    {
                        "variant": variant,
                        "policy_id": policy_id(variant),
                        "seed": seed,
                        "split": split,
                        "best_epoch": int(ckpt.get("best_epoch", -1)),
                        "threshold": best_thr,
                        "action_pnl": float(metric["total_excess_pnl_H20"]),
                        "selected_action_count": int(metric["selected_action_count"]),
                        "matched_random_95p": float(random_summary.iloc[0]["random_95p"]),
                        "matched_random_pass": bool(random_summary.iloc[0]["beats_random_95p"]),
                        "empirical_p_value_fraction_random_ge_model": float(random_summary.iloc[0]["empirical_p_value_fraction_random_ge_model"]),
                        "causal_fill_pnl": float(total),
                        "causal_fill_count": int(fill_count),
                        "causal_fill_max_drawdown": float(max_dd),
                        "new3_used_for_selection": False,
                        "new4_used_for_selection": False,
                        "new5_used_for_selection": False,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_prepad_seed_stability.csv", index=False)
    write_markdown(
        ROOT / "TCN_PREPAD_SEED_STABILITY_REPORT.md",
        [
            "# TCN PrePad Seed Stability Report",
            "",
            "Each seed selects its own threshold on validation only, then evaluates val/NEW3/NEW4. NEW5 is not loaded.",
            "",
            md_table(out) if not out.empty else "No seed stability rows available.",
        ],
    )
    return out


def estimate_tcn_latency() -> pd.DataFrame:
    model = build_model()
    mac_rows = []
    in_ch = len(p05a.PHC_FEATURES)
    length = SEQ_LEN_IN
    total_macs = 0
    for block_idx, dilation in enumerate(TCNPrePadRegressor.DILATIONS):
        conv1_len = length - (KERNEL_SIZE - 1) * dilation
        conv1_macs = in_ch * CHANNELS * KERNEL_SIZE * conv1_len
        conv2_len = conv1_len - (KERNEL_SIZE - 1) * dilation
        conv2_macs = CHANNELS * CHANNELS * KERNEL_SIZE * conv2_len
        downsample_macs = (in_ch * CHANNELS * conv2_len) if in_ch != CHANNELS else 0
        total_macs += conv1_macs + conv2_macs + downsample_macs
        mac_rows.extend(
            [
                {"block": block_idx, "op": "conv1", "dilation": dilation, "out_len": conv1_len, "macs": int(conv1_macs)},
                {"block": block_idx, "op": "conv2", "dilation": dilation, "out_len": conv2_len, "macs": int(conv2_macs)},
                {"block": block_idx, "op": "downsample", "dilation": dilation, "out_len": conv2_len, "macs": int(downsample_macs)},
            ]
        )
        length = conv2_len
        in_ch = CHANNELS
    fc_macs = CHANNELS * NUM_OUTPUTS
    total_macs += fc_macs
    mac_rows.append({"block": "readout", "op": "fc", "dilation": 1, "out_len": 1, "macs": int(fc_macs)})
    mac_df = pd.DataFrame(mac_rows)
    mac_df.to_csv(RESULTS / "tcn_prepad_mac_breakdown.csv", index=False)
    rows = []
    for rate_name, macs_per_second in [("fpga_effective_50gmac_per_s", 50.0e9), ("cpp_int8_reference_8gmac_per_s", 8.0e9)]:
        rows.append(
            {
                "model": "tcn_prepad_16feat",
                "estimate_type": "pre_synthesis_macs_only_not_hls_verified",
                "feature_dim": len(p05a.PHC_FEATURES),
                "seq_len_in": SEQ_LEN_IN,
                "seq_len_out": SEQ_LEN,
                "channels": CHANNELS,
                "blocks": len(TCNPrePadRegressor.DILATIONS),
                "kernel_size": KERNEL_SIZE,
                "receptive_field": model.receptive_field(),
                "parameter_count": count_parameters(model),
                "macs_per_sample": int(total_macs),
                "assumed_rate_name": rate_name,
                "assumed_macs_per_second": macs_per_second,
                "estimated_latency_us": float(total_macs / macs_per_second * 1e6),
                "vendor_synthesis_run": False,
                "csim_run": False,
                "csynth_run": False,
                "cosim_run": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_prepad_latency_estimate.csv", index=False)
    return out


def load_existing_phc_v1_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    action = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    causal = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    seed = pd.read_csv(PHASE05A / "results" / "phc_lite_seed_stability.csv")
    return action, causal, seed


def final_report() -> pd.DataFrame:
    ensure_dirs()
    action_path = RESULTS / "tcn_prepad_oos_eval.csv"
    causal_path = RESULTS / "tcn_prepad_causal_fill_eval.csv"
    random_path = RESULTS / "tcn_prepad_matched_random_summary.csv"
    seed_path = RESULTS / "tcn_prepad_seed_stability.csv"
    latency_path = RESULTS / "tcn_prepad_latency_estimate.csv"
    action = pd.read_csv(action_path) if action_path.exists() else pd.DataFrame()
    causal = pd.read_csv(causal_path) if causal_path.exists() else pd.DataFrame()
    random = pd.read_csv(random_path) if random_path.exists() else pd.DataFrame()
    seed = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    latency = pd.read_csv(latency_path) if latency_path.exists() else estimate_tcn_latency()
    phc_action, phc_causal, phc_seed = load_existing_phc_v1_tables()

    rows = []
    for split in SPLITS:
        phc_a = phc_action[(phc_action["split"].eq(split)) & (phc_action["policy_id"].eq("phc_lite"))]
        phc_c = phc_causal[(phc_causal["split"].eq(split)) & (phc_causal["policy_id"].eq("phc_lite"))]
        if not phc_a.empty:
            rows.append(
                {
                    "model": "PHC-lite v1 teacher",
                    "split": split,
                    "action_pnl": float(phc_a["total_excess_pnl_H20"].iloc[0]),
                    "causal_fill_pnl": float(phc_c["sequential_excess_pnl"].iloc[0]) if not phc_c.empty else np.nan,
                    "selected_action_count": int(phc_a["selected_action_count"].iloc[0]),
                    "win_rate": float(phc_a["win_rate"].iloc[0]),
                    "matched_random_pass": True,
                    "estimated_latency_us": 83.16,
                    "latency_source": "06F routed PHC step baseline at 100MHz",
                }
            )
        for variant in VARIANTS:
            pid = policy_id(variant)
            if action.empty or "split" not in action.columns or "policy_id" not in action.columns:
                continue
            a = action[(action["split"].eq(split)) & (action["policy_id"].eq(pid))]
            c = (
                causal[(causal["split"].eq(split)) & (causal["policy_id"].eq(pid))]
                if not causal.empty and {"split", "policy_id"}.issubset(causal.columns)
                else pd.DataFrame()
            )
            r = (
                random[(random["split"].eq(split)) & (random["policy_id"].eq(pid))]
                if not random.empty and {"split", "policy_id"}.issubset(random.columns)
                else pd.DataFrame()
            )
            if a.empty:
                continue
            rows.append(
                {
                    "model": pid,
                    "split": split,
                    "action_pnl": float(a["total_excess_pnl_H20"].iloc[0]),
                    "causal_fill_pnl": float(c["sequential_excess_pnl"].iloc[0]) if not c.empty else np.nan,
                    "selected_action_count": int(a["selected_action_count"].iloc[0]),
                    "win_rate": float(a["win_rate"].iloc[0]),
                    "matched_random_pass": bool(r["beats_random_95p"].iloc[0]) if not r.empty else np.nan,
                    "estimated_latency_us": float(latency[latency["assumed_rate_name"].eq("fpga_effective_50gmac_per_s")]["estimated_latency_us"].iloc[0]) if not latency.empty else np.nan,
                    "latency_source": "pre-synthesis MAC estimate only, not csynth/cosim",
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        phc_lookup = summary[summary["model"].eq("PHC-lite v1 teacher")].set_index("split")
        deltas = []
        for _, row in summary.iterrows():
            split = row["split"]
            if row["model"] == "PHC-lite v1 teacher" or split not in phc_lookup.index:
                deltas.append({"action_delta_vs_phc": 0.0, "causal_delta_vs_phc": 0.0, "latency_speedup_vs_phc": 1.0})
            else:
                phc_row = phc_lookup.loc[split]
                deltas.append(
                    {
                        "action_delta_vs_phc": float(row["action_pnl"] - phc_row["action_pnl"]),
                        "causal_delta_vs_phc": float(row["causal_fill_pnl"] - phc_row["causal_fill_pnl"]),
                        "latency_speedup_vs_phc": float(83.16 / row["estimated_latency_us"]) if row["estimated_latency_us"] else np.nan,
                    }
                )
        summary = pd.concat([summary.reset_index(drop=True), pd.DataFrame(deltas)], axis=1)
    summary.to_csv(RESULTS / "tcn_vs_phc_tradeoff_summary.csv", index=False)

    seed_summary = pd.DataFrame()
    if not seed.empty:
        seed_summary = seed.groupby(["variant", "split"]).agg(
            action_pnl_mean=("action_pnl", "mean"),
            action_pnl_std=("action_pnl", "std"),
            causal_fill_pnl_mean=("causal_fill_pnl", "mean"),
            causal_fill_pnl_std=("causal_fill_pnl", "std"),
            all_positive=("action_pnl", lambda s: bool((s > 0).all())),
        ).reset_index()
        seed_summary.to_csv(RESULTS / "tcn_prepad_seed_variance_summary.csv", index=False)

    status = "tcn_tradeoff_incomplete_missing_training_outputs"
    if not summary.empty and set(POLICY_IDS.values()).issubset(set(summary["model"])):
        beats_phc = []
        for pid in POLICY_IDS.values():
            sub = summary[summary["model"].eq(pid)]
            beats_phc.append(bool((sub["action_delta_vs_phc"] >= 0).all() and (sub["causal_delta_vs_phc"] >= 0).all()))
        status = "tcn_distillation_closes_gap" if beats_phc[1] else "tcn_feedforward_latency_tradeoff_but_accuracy_gap"
        if beats_phc[0] and beats_phc[1]:
            status = "tcn_feedforward_matches_phc"

    lines = [
        "# TCN Feedforward vs PHC-lite Trade-off Report",
        "",
        f"Final status: `{status}`.",
        "",
        "## Scope",
        "",
        "- This phase evaluates a feedforward `TCNPrePadRegressor` on the Phase 05A TXF-only sequence-window universe.",
        "- Variants: `tcn_prepad_scratch_16feat` and `tcn_prepad_distilled_16feat`.",
        "- Both variants use the same 16 PHC-lite causal features, `seq_len=128`, horizon `H20`, and validation-only threshold/seed selection.",
        "- Distillation uses frozen PHC-lite v1 bid/ask action-value scores as soft labels. NEW3, NEW4, and NEW5 are not used for training or selection.",
        "",
        "## Model / Latency Estimate",
        "",
        md_table(latency) if not latency.empty else "Latency estimate missing.",
        "",
        "The latency values above are pre-synthesis MAC-rate estimates. They are not csynth/cosim/P&R validated and must not be cited as FPGA-measured latency.",
        "",
        "## PHC-lite vs TCN Trade-off",
        "",
        md_table(summary) if not summary.empty else "Trade-off summary missing because model outputs have not been generated.",
        "",
        "## Seed Stability",
        "",
        md_table(seed_summary) if not seed_summary.empty else "Seed stability summary missing because seed evaluation has not been generated.",
        "",
        "## Existing Context",
        "",
        f"- PHC latency path report: `{LATENCY_REF}`.",
        "- Existing 06C/06D TCN reports are useful context, but this phase is the 16-feature scratch/distillation comparison requested here.",
        "",
        "## Claim Boundaries",
        "",
        "- No NEW5 usage.",
        "- No real-world profitability claim.",
        "- No true exchange replay claim.",
        "- No FPGA PnL improvement claim.",
        "- TCN latency here is pre-synthesis estimate only unless a later HLS phase runs csim/csynth/cosim/P&R.",
    ]
    write_markdown(ROOT / "TCN_FEEDFORWARD_PHC_TRADEOFF_REPORT.md", lines)
    return summary


def run_pipeline() -> None:
    train_all_seeds()
    select_policy()
    evaluate_oos()
    matched_random()
    causal_fill_eval()
    seed_stability()
    estimate_tcn_latency()
    final_report()
