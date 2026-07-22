from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "06d_tcn_5block_prepad_hls4ml"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
HLS4ML_PRJ = ROOT / "hls4ml_prj"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE06C = PROJECT_ROOT / "n2ew_era" / "06c_tcn_action_value_hls4ml"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import common_05a as p05a  # noqa: E402
from src.models.tcn_prepad import TCNPrePadRegressor  # noqa: E402

PHASE = "06d"
SEQ_LEN = 128
SEQ_LEN_IN = 252
TOTAL_PAD = 124
STRIDE = 32
CHANNEL_LIST = [16, 16, 16, 16, 16]
KERNEL_SIZE = 3
DROPOUT = 0.1
NUM_OUTPUTS = 2
MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3
LR_MIN = 1e-5
LR_SCHEDULER = "CosineAnnealingLR"
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
SEEDS = [0, 1, 2]
SPLITS = ["val", "new3", "new4"]
VARIANTS = ["18feat"]
POLICY_IDS = {
    "18feat": "tcn_5block_prepad_18feat",
}
FEAT_16 = list(p05a.PHC_FEATURES)
FEAT_18 = list(p05a.PHC_FEATURES) + ["hawkes_excite", "time_in_session_norm"]
# Same mask salt as 05C4 so causal-fill comparisons are controlled by policy only.
FILL_RANDOM_SALT = "05c3_stp_v2_conservative_l2_stress"


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, FIGURES, HLS4ML_PRJ]:
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


def feature_names(variant: str) -> list[str]:
    if variant == "18feat":
        return FEAT_18
    raise ValueError(f"Unknown variant: {variant}")


def num_inputs(variant: str) -> int:
    return len(feature_names(variant))


def policy_id(variant: str) -> str:
    return POLICY_IDS[variant]


def best_ckpt_path(variant: str, seed: int) -> Path:
    return MODELS / f"tcn_5block_prepad_{variant}_seed{seed}_best_val.pt"


def last_ckpt_path(variant: str, seed: int) -> Path:
    return MODELS / f"tcn_5block_prepad_{variant}_seed{seed}_last.pt"


def curve_path(variant: str, seed: int) -> Path:
    return RESULTS / f"tcn_5block_prepad_{variant}_training_curve_seed{seed}.csv"


def target_scale() -> float:
    return float(p05a.load_scaler()["target_scale_train_std_no_mean_shift"])


def fit_feat18_extra_scaler() -> dict[str, Any]:
    ensure_dirs()
    path = RESULTS / "feat18_extra_scaler.json"
    if path.exists():
        return read_json(path)
    values: list[np.ndarray] = []
    missing = 0
    rows = 0
    for part in p05a.part_paths("train"):
        frame = p05a.read_event_frame(part)
        hawkes = p05a.numeric_series(frame, "hawkes_excite")
        vals = np.log1p(np.maximum(hawkes, 0.0)).astype(np.float32)
        missing += int((~np.isfinite(vals)).sum())
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        values.append(vals)
        rows += len(vals)
    all_vals = np.concatenate(values) if values else np.zeros(1, dtype=np.float32)
    mean = float(np.mean(all_vals))
    std = float(np.std(all_vals))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    payload = {
        "hawkes_log1p_mean_train": mean,
        "hawkes_log1p_std_train": std,
        "train_rows": int(rows),
        "missing_after_coerce": int(missing),
        "normalization_fit_split": "train",
        "new3_used_for_scaler": False,
        "new4_used_for_scaler": False,
        "new5_used_for_scaler": False,
    }
    write_json(path, payload)
    return payload


def add_feat18_features(frame: pd.DataFrame, base_norm: np.ndarray, extra: dict[str, Any]) -> np.ndarray:
    hawkes = p05a.numeric_series(frame, "hawkes_excite")
    hawkes_log = np.log1p(np.maximum(hawkes, 0.0)).astype(np.float32)
    hawkes_z = (hawkes_log - float(extra["hawkes_log1p_mean_train"])) / max(float(extra["hawkes_log1p_std_train"]), 1e-6)
    event = pd.to_numeric(frame["event_index"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    session = frame["session_id"].astype(str)
    time_norm = np.zeros(len(frame), dtype=np.float32)
    for _, idx in frame.groupby(session, sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=np.int64)
        vals = event[loc]
        lo = float(np.min(vals)) if len(vals) else 0.0
        hi = float(np.max(vals)) if len(vals) else lo
        time_norm[loc] = ((vals - lo) / (hi - lo + 1e-6)).astype(np.float32)
    extra_arr = np.column_stack([hawkes_z.astype(np.float32), time_norm.astype(np.float32)])
    return np.concatenate([base_norm.astype(np.float32), extra_arr], axis=1).astype(np.float32)


def build_windows_for_frame_18(frame: pd.DataFrame, scaler: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any] | None:
    base_norm, raw = p05a.normalized_features(frame, scaler)
    x = add_feat18_features(frame, base_norm, extra)
    n = len(frame)
    if n < SEQ_LEN:
        return None
    starts = np.arange(0, n - SEQ_LEN + 1, STRIDE, dtype=np.int64)
    xw = np.empty((len(starts), SEQ_LEN, x.shape[1]), dtype=np.float16)
    for i, start in enumerate(starts):
        xw[i] = x[start : start + SEQ_LEN].astype(np.float16)
    target_idx = starts + SEQ_LEN - 1
    sessions = frame["session_id"].astype(str).to_numpy()
    unique_sessions, session_code = np.unique(sessions[target_idx], return_inverse=True)
    return {
        "x": xw,
        "y_bid": frame["bid_excess_pnl_H20"].to_numpy(dtype=np.float32)[target_idx],
        "y_ask": frame["ask_excess_pnl_H20"].to_numpy(dtype=np.float32)[target_idx],
        "event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)[target_idx],
        "timestamp_us": pd.to_numeric(frame["timestamp_us"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)[target_idx],
        "target_raw": raw[target_idx].astype(np.float32),
        "session_code": session_code.astype(np.int32),
        "session_labels": unique_sessions.tolist(),
        "start_event_index": pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)[starts],
    }


def build_enriched_sequences() -> pd.DataFrame:
    ensure_dirs()
    scaler = p05a.load_scaler()
    extra = fit_feat18_extra_scaler()
    manifest_rows: list[dict[str, Any]] = []
    for split in p05a.ALL_SPLITS:
        chunks: list[dict[str, Any]] = []
        source_events = 0
        for path in p05a.part_paths(split):
            frame = p05a.read_event_frame(path)
            source_events += len(frame)
            built = build_windows_for_frame_18(frame, scaler, extra)
            if built is None:
                manifest_rows.append(
                    {
                        "split": split,
                        "part": path.name,
                        "event_rows": len(frame),
                        "window_rows": 0,
                        "feature_dim": len(FEAT_18),
                        "skipped_reason": "fewer_rows_than_seq_len",
                    }
                )
                continue
            chunks.append(built)
            manifest_rows.append(
                {
                    "split": split,
                    "part": path.name,
                    "event_rows": len(frame),
                    "window_rows": int(len(built["y_bid"])),
                    "seq_len": SEQ_LEN,
                    "seq_len_in": SEQ_LEN_IN,
                    "total_prepad": TOTAL_PAD,
                    "stride": STRIDE,
                    "feature_dim": len(FEAT_18),
                    "bid_positive_rate": float((built["y_bid"] > 0).mean()) if len(built["y_bid"]) else 0.0,
                    "ask_positive_rate": float((built["y_ask"] > 0).mean()) if len(built["y_ask"]) else 0.0,
                    "target_mean_bid": float(np.mean(built["y_bid"])) if len(built["y_bid"]) else 0.0,
                    "target_mean_ask": float(np.mean(built["y_ask"])) if len(built["y_ask"]) else 0.0,
                    "skipped_reason": "",
                }
            )
            print(f"06d feat18 built split={split} part={path.name} windows={len(built['y_bid'])}", flush=True)
        if not chunks:
            continue
        session_code, session_labels = p05a.merge_session_codes(chunks)
        payload = {
            "x": torch.from_numpy(np.concatenate([c["x"] for c in chunks], axis=0)),
            "y_bid": torch.from_numpy(np.concatenate([c["y_bid"] for c in chunks]).astype(np.float32)),
            "y_ask": torch.from_numpy(np.concatenate([c["y_ask"] for c in chunks]).astype(np.float32)),
            "event_index": torch.from_numpy(np.concatenate([c["event_index"] for c in chunks]).astype(np.int64)),
            "timestamp_us": torch.from_numpy(np.concatenate([c["timestamp_us"] for c in chunks]).astype(np.int64)),
            "target_raw": torch.from_numpy(np.concatenate([c["target_raw"] for c in chunks], axis=0).astype(np.float32)),
            "session_code": torch.from_numpy(session_code.astype(np.int32)),
            "session_labels": session_labels,
            "feature_names": FEAT_18,
            "target_raw_columns": p05a.TARGET_RAW_COLUMNS,
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
            "target_horizon": p05a.HORIZON,
            "source_event_rows": int(source_events),
            "split": split,
            "new5_used": False,
        }
        torch.save(payload, RESULTS / f"feat18_sequences_{split}.pt")
        print(f"06d feat18 saved split={split} total_windows={len(payload['y_bid'])}", flush=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(RESULTS / "feat18_sequence_manifest.csv", index=False)
    write_markdown(
        ROOT / "TCN_FEAT18_SEQUENCE_DATASET_REPORT.md",
        [
            "# TCN 5-Block PrePad 18-Feature Sequence Dataset Report",
            "",
            "`tcn_5block_prepad_18feat` uses the same 18 causal features as 06C: PHC_FEATURES plus train-normalized `hawkes_excite` and causal `time_in_session_norm`.",
            "",
            "- Product universe: TXF-only inherited from Phase 04B.",
            "- Sequence setting: `seq_len=128`, `stride=32`, target is final timestep H20.",
            "- Model input setting: caller prepends 124 zero ticks, so `seq_len_in=252` and the valid-conv model emits `seq_len_out=128`.",
            "- Scalers: PHC feature scaler and hawkes scaler fit on train only.",
            "- NEW5 is not loaded.",
            "",
            manifest.groupby("split", as_index=False).agg(event_rows=("event_rows", "sum"), window_rows=("window_rows", "sum")).to_markdown(index=False),
        ],
    )
    return manifest


def prepare_enriched_sequences() -> pd.DataFrame:
    """Reuse 06C 18-feature sequence tensors when available; otherwise rebuild."""
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in p05a.ALL_SPLITS:
        dst = RESULTS / f"feat18_sequences_{split}.pt"
        src = PHASE06C / "results" / f"feat18_sequences_{split}.pt"
        if dst.exists():
            payload = torch.load(dst, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "split": split,
                    "source": "06d_existing",
                    "path": str(dst),
                    "window_rows": int(len(payload["y_bid"])),
                    "feature_dim": int(payload["x"].shape[-1]),
                    "new5_used": bool(payload.get("new5_used", False)),
                }
            )
        elif src.exists():
            shutil.copy2(src, dst)
            payload = torch.load(dst, map_location="cpu", weights_only=False)
            rows.append(
                {
                    "split": split,
                    "source": "06c_reused_copy",
                    "path": str(dst),
                    "source_path": str(src),
                    "window_rows": int(len(payload["y_bid"])),
                    "feature_dim": int(payload["x"].shape[-1]),
                    "new5_used": bool(payload.get("new5_used", False)),
                }
            )
        else:
            missing.append(split)
    if missing:
        manifest = build_enriched_sequences()
        manifest["source"] = "06d_rebuilt"
    else:
        manifest = pd.DataFrame(rows)
        manifest.to_csv(RESULTS / "feat18_sequence_manifest.csv", index=False)
        write_markdown(
            ROOT / "TCN_FEAT18_SEQUENCE_DATASET_REPORT.md",
            [
                "# TCN 5-Block PrePad 18-Feature Sequence Dataset Report",
                "",
                "06D reuses the 06C 18-feature TXF-only sequence tensors when available. The model pre-padding is applied by the 06D dataset/model caller, not by rebuilding labels.",
                "",
                "- NEW5 is not loaded.",
                "- Feature tensor shape remains `(windows, 128, 18)` before pre-padding.",
                "- 06D model input shape is `(batch, 18, 252)` after caller-side zero prepad.",
                "",
                manifest.to_markdown(index=False),
            ],
        )
    if not manifest.empty and manifest.get("new5_used", pd.Series([False])).astype(bool).any():
        raise RuntimeError("Phase 06D hard rule violation: sequence manifest indicates NEW5 usage")
    return manifest


def load_payload(variant: str, split: str) -> dict[str, Any]:
    if variant == "18feat":
        path = RESULTS / f"feat18_sequences_{split}.pt"
        source = "06d"
        if not path.exists():
            source_path = PHASE06C / "results" / f"feat18_sequences_{split}.pt"
            if source_path.exists():
                payload = torch.load(source_path, map_location="cpu", weights_only=False)
                source = "06c_reused"
            else:
                prepare_enriched_sequences()
                payload = torch.load(path, map_location="cpu", weights_only=False)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["source_sequence_artifact"] = source
    else:
        raise ValueError(variant)
    if payload.get("new5_used"):
        raise RuntimeError("Phase 06D hard rule violation: loaded payload marked new5_used=True")
    return payload


class TCNSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any], scale: float) -> None:
        self.x = payload["x"]
        self.y = torch.stack([payload["y_bid"], payload["y_ask"]], dim=1).float() / float(scale)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[idx].float().permute(1, 0).contiguous()
        pad = torch.zeros(x.shape[0], TOTAL_PAD, dtype=x.dtype)
        x = torch.cat([pad, x], dim=1).contiguous()
        return x, self.y[idx]


def build_model(variant: str) -> TCNPrePadRegressor:
    model = TCNPrePadRegressor(
        num_inputs=num_inputs(variant),
        num_outputs=NUM_OUTPUTS,
        channels=CHANNEL_LIST[0],
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    )
    if model.receptive_field() != 125:
        raise RuntimeError(f"TCN 06D RF sanity failed: {model.receptive_field()} != 125")
    return model


def _sanity_check() -> dict[str, Any]:
    model = build_model("18feat")
    rf = model.receptive_field()
    if rf != 125:
        raise AssertionError(f"RF={rf}, expected 125")
    x_dummy = torch.zeros(2, num_inputs("18feat"), SEQ_LEN_IN)
    with torch.no_grad():
        out = model(x_dummy)
    if tuple(out.shape) != (2, 2):
        raise AssertionError(f"output shape {tuple(out.shape)}, expected (2,2)")
    payload = {
        "rf": int(rf),
        "params": count_parameters(model),
        "output_shape": tuple(out.shape),
        "seq_len_in": SEQ_LEN_IN,
        "seq_len_out": SEQ_LEN,
        "total_prepad": TOTAL_PAD,
    }
    print(
        f"[06d sanity] RF={payload['rf']} | params={payload['params']} | "
        f"output_shape={payload['output_shape']}",
        flush=True,
    )
    return payload


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


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
        "model_variant": variant,
        "policy_id": policy_id(variant),
        "feature_names": feature_names(variant),
        "feature_dim": num_inputs(variant),
        "channel_list": CHANNEL_LIST,
        "dilations": TCNPrePadRegressor.DILATIONS,
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
        "lr_scheduler": LR_SCHEDULER,
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
        raise RuntimeError(f"Refusing non-06D checkpoint: {checkpoint_path}")
    if ckpt.get("model_variant") != variant:
        raise RuntimeError(f"Variant mismatch in checkpoint {checkpoint_path}: {ckpt.get('model_variant')} != {variant}")
    if ckpt.get("lr_scheduler") != LR_SCHEDULER or "scheduler_state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint is missing scheduler state: {checkpoint_path}")
    model = build_model(variant)
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
    if variant == "18feat":
        prepare_enriched_sequences()
    set_seed(seed)
    scale = target_scale()
    train_payload = load_payload(variant, "train")
    val_payload = load_payload(variant, "val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN)
    huber = torch.nn.HuberLoss(delta=1.0)
    loader = torch.utils.data.DataLoader(
        TCNSequenceDataset(train_payload, scale),
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
            "model_variant": variant,
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
            "lr_scheduler": LR_SCHEDULER,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "early_stop_counter": early_counter,
            "optimizer_restarted": False,
            "max_epoch_hit": False,
            "parameter_count": count_parameters(model),
            "device": device,
            "channel_list": "-".join(map(str, CHANNEL_LIST)),
            "kernel_size": KERNEL_SIZE,
            "dropout": DROPOUT,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
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
        pd.DataFrame(rows).to_csv(curve_path(variant, seed), index=False)
        print(
            f"06d variant={variant} seed={seed} epoch={epoch} lr={current_lr:.8f} "
            f"val_pnl={val_pnl:.2f} selected={val_selected} grad={row['grad_norm']:.4f}",
            flush=True,
        )
        should_stop = early_counter >= PATIENCE
        scheduler.step()
        if is_best:
            torch.save(make_checkpoint(variant, model, optimizer, scheduler, seed, "best_epoch", best_epoch, scale), best_ckpt_path(variant, seed))
        torch.save(make_checkpoint(variant, model, optimizer, scheduler, seed, "epoch", epoch, scale), last_ckpt_path(variant, seed))
        if should_stop:
            break
    curve = pd.DataFrame(rows)
    if not curve.empty:
        curve.loc[curve.index[-1], "max_epoch_hit"] = int(curve["epoch"].iloc[-1]) >= MAX_EPOCHS
        curve.to_csv(curve_path(variant, seed), index=False)
    plot_training_curve(curve, variant, seed)
    write_training_report(curve, variant, seed)
    return curve


def plot_training_curve(curve: pd.DataFrame, variant: str, seed: int) -> None:
    if curve.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["train_loss"], label="train")
    plt.plot(curve["epoch"], curve["val_loss"], label="val")
    plt.title(f"TCN action-value {variant} loss seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"tcn_5block_prepad_{variant}_loss_curve_seed{seed}.png", dpi=160)
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(curve["epoch"], curve["val_policy_excess_pnl"], marker="o", markersize=2)
    plt.title(f"TCN action-value {variant} validation PnL seed {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Val policy excess PnL")
    plt.tight_layout()
    plt.savefig(FIGURES / f"tcn_5block_prepad_{variant}_val_policy_pnl_seed{seed}.png", dpi=160)
    plt.close()


def write_training_report(curve: pd.DataFrame, variant: str, seed: int) -> None:
    if curve.empty:
        return
    best = curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
    final = curve.iloc[-1]
    early_stop = int(final["epoch"]) < MAX_EPOCHS
    convergence_closed = bool(early_stop and int(best["epoch"]) < 185)
    write_markdown(
        ROOT / f"TCN_AV_{variant.upper()}_TRAINING_CONVERGENCE_REPORT_seed{seed}.md",
        [
            f"# TCN 5-Block PrePad {variant} Training Report seed {seed}",
            "",
            "Training uses TXF-only 18-feature sequence tensors. No NEW5 is loaded. Epoch selection is validation-only.",
            f"- Best epoch: `{int(best['epoch'])}`.",
            f"- Best validation policy PnL: `{float(best['val_policy_excess_pnl']):,.2f}`.",
            f"- Early stop triggered: `{early_stop}`.",
            f"- Hit max epoch: `{bool(int(final['epoch']) >= MAX_EPOCHS)}`.",
            f"- Convergence closed (`early_stop=True and best_epoch<185`): `{convergence_closed}`.",
            f"- LR at best epoch: `{float(best.get('current_lr', np.nan)):.8f}`.",
            f"- LR at final epoch: `{float(final.get('current_lr', np.nan)):.8f}`.",
            f"- Grad norm at best epoch: `{float(best.get('grad_norm', np.nan)):.6f}`.",
            f"- Grad norm at final epoch: `{float(final.get('grad_norm', np.nan)):.6f}`.",
            f"- LR scheduler: `{LR_SCHEDULER}`.",
        ],
    )


def train_all(overwrite: bool = False) -> pd.DataFrame:
    ensure_dirs()
    _sanity_check()
    if not (RESULTS / "feat18_sequences_train.pt").exists():
        prepare_enriched_sequences()
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            rows.append(train_seed(variant, seed, overwrite=overwrite))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(RESULTS / "tcn_5block_prepad_training_curve_all_variants.csv", index=False)
    convergence_summary()
    return out


def convergence_summary() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = curve_path(variant, seed)
            if not path.exists():
                continue
            curve = pd.read_csv(path)
            if curve.empty:
                continue
            best = curve.sort_values("val_policy_excess_pnl", ascending=False).iloc[0]
            final = curve.iloc[-1]
            early_stop = int(final["epoch"]) < MAX_EPOCHS
            rows.append(
                {
                    "model_variant": variant,
                    "policy_id": policy_id(variant),
                    "seed": seed,
                    "best_epoch": int(best["epoch"]),
                    "best_val_pnl": float(best["val_policy_excess_pnl"]),
                    "final_epoch": int(final["epoch"]),
                    "early_stop": bool(early_stop),
                    "max_epoch_hit": bool(int(final["epoch"]) >= MAX_EPOCHS),
                    "convergence_closed": bool(early_stop and int(best["epoch"]) < 185),
                    "strict_convergence_rule": "early_stop=True and best_epoch<185",
                    "lr_at_best_epoch": float(best.get("current_lr", np.nan)),
                    "lr_at_final_epoch": float(final.get("current_lr", np.nan)),
                    "grad_norm_at_best_epoch": float(best.get("grad_norm", np.nan)),
                    "grad_norm_at_final_epoch": float(final.get("grad_norm", np.nan)),
                    "phase": PHASE,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_5block_prepad_convergence_summary.csv", index=False)
    return out


def threshold_grid(scores: np.ndarray) -> list[float]:
    max_scores = np.max(scores, axis=1)
    finite = max_scores[np.isfinite(max_scores)]
    q = np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if len(finite) else np.array([0.0])
    return sorted(set([0.0, *q.tolist()]))


def select_policy() -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    policies: dict[str, Any] = {}
    for variant in VARIANTS:
        val = load_payload(variant, "val")
        for seed in SEEDS:
            path = best_ckpt_path(variant, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            model, ckpt = load_model(variant, path)
            scores = predict_scores(model, val, scale=float(ckpt["target_scale"]))
            for threshold in threshold_grid(scores):
                decision = p05a.policy_from_scores(scores, float(threshold))
                row = p05a.metric_from_decision("val", f"{policy_id(variant)}_seed{seed}", val, decision)
                row.update(
                    {
                        "model_variant": variant,
                        "seed": seed,
                        "threshold": float(threshold),
                        "best_epoch": int(ckpt["best_epoch"]),
                        "selection_split": "val",
                        "phase": PHASE,
                        "new3_used_for_selection": False,
                        "new4_used_for_selection": False,
                        "new5_used_for_selection": False,
                    }
                )
                rows.append(row)
        variant_grid = pd.DataFrame([r for r in rows if r["model_variant"] == variant])
        best = variant_grid.sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).iloc[0].to_dict()
        selected = {
            "model": policy_id(variant),
            "model_variant": variant,
            "selection_split": "val",
            "seed": int(best["seed"]),
            "threshold": float(best["threshold"]),
            "target": "excess_pnl_H20",
            "seq_len": SEQ_LEN,
            "seq_len_in": SEQ_LEN_IN,
            "total_prepad": TOTAL_PAD,
            "receptive_field": 125,
            "stride": STRIDE,
            "feature_names": feature_names(variant),
            "num_inputs": num_inputs(variant),
            "channel_list": CHANNEL_LIST,
            "dilations": TCNPrePadRegressor.DILATIONS,
            "kernel_size": KERNEL_SIZE,
            "dropout": DROPOUT,
            "lr_scheduler": LR_SCHEDULER,
            "validation_metrics": p05a.json_sanitize(best),
            "new3_used_for_selection": False,
            "new4_used_for_selection": False,
            "new5_used_for_selection": False,
        }
        policies[variant] = selected
        write_json(RESULTS / f"tcn_5block_prepad_{variant}_selected_policy.json", selected)
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "tcn_5block_prepad_validation_threshold_search.csv", index=False)
    write_json(RESULTS / "tcn_5block_prepad_selected_policies.json", policies)
    return grid, policies


def selected_model(variant: str) -> tuple[TCNPrePadRegressor, dict[str, Any], dict[str, Any]]:
    selected = read_json(RESULTS / f"tcn_5block_prepad_{variant}_selected_policy.json")
    model, ckpt = load_model(variant, best_ckpt_path(variant, int(selected["seed"])))
    return model, ckpt, selected


def tcn_decision_for_payload(variant: str, payload: dict[str, Any], model: TCNPrePadRegressor, ckpt: dict[str, Any], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    scores = predict_scores(model, payload, scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, threshold), scores


def evaluate_oos() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    base = p05a.evaluate_or_load_metrics()
    for variant in VARIANTS:
        model, ckpt, selected = selected_model(variant)
        threshold = float(selected["threshold"])
        for split in SPLITS:
            payload = load_payload(variant, split)
            decision, _ = tcn_decision_for_payload(variant, payload, model, ckpt, threshold)
            row = p05a.metric_from_decision(split, policy_id(variant), payload, decision)
            row.update(
                {
                    "model_variant": variant,
                    "seed_selected_on_val": int(selected["seed"]),
                    "threshold_selected_on_val": threshold,
                    "best_epoch": int(ckpt["best_epoch"]),
                    "phase": PHASE,
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
                        "policy_id": policy_id(variant),
                        "model_variant": variant,
                        "session_id": session_id,
                        "session_excess_pnl_H20": float(pnl[mask].sum()),
                        "selected_action_count": int((decision[mask] != p05a.DECISION_NO_QUOTE).sum()),
                        "state_count": int(mask.sum()),
                    }
                )
    tcn = pd.DataFrame(rows)
    compare = pd.concat(
        [
            base[base["policy_id"].isin(["side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            pd.read_csv(PHASE05C4 / "results" / "phc_stp_v2_oos_eval.csv").assign(policy_id="phc_stp_v2_cosine"),
            tcn,
        ],
        ignore_index=True,
    )
    tcn.to_csv(RESULTS / "tcn_5block_prepad_oos_eval.csv", index=False)
    compare.to_csv(RESULTS / "tcn_5block_prepad_model_comparison_action.csv", index=False)
    pd.DataFrame(session_rows).to_csv(RESULTS / "tcn_5block_prepad_session_pnl.csv", index=False)
    write_markdown(
        ROOT / "TCN_ACTION_VALUE_OOS_EVALUATION_REPORT.md",
        [
            "# TCN 5-Block PrePad OOS Evaluation Report",
            "",
            "Seed and threshold selection are validation-only. NEW3/NEW4 are diagnostic OOS. NEW5 is not loaded.",
            "",
            tcn.to_markdown(index=False),
        ],
    )
    return tcn, compare


def matched_random() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        model, ckpt, selected = selected_model(variant)
        for split in SPLITS:
            payload = load_payload(variant, split)
            decision, _ = tcn_decision_for_payload(variant, payload, model, ckpt, float(selected["threshold"]))
            _, summary = p05a.matched_random_for_split(split, decision, n_trials=p05a.RANDOM_TRIALS, seed=20260612)
            row = summary.iloc[0].to_dict()
            row.update(
                {
                    "policy_id": policy_id(variant),
                    "model_variant": variant,
                    "seed_selected_on_val": int(selected["seed"]),
                    "threshold_selected_on_val": float(selected["threshold"]),
                    "phase": PHASE,
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_5block_prepad_matched_random_summary.csv", index=False)
    write_markdown(
        ROOT / "TCN_ACTION_VALUE_MATCHED_RANDOM_REPORT.md",
        [
            "# TCN 5-Block PrePad Matched Random Report",
            "",
            "Empirical p-value is the fraction of matched random trials whose PnL >= model PnL. If the observed fraction is `0.0`, report it as empirical p-value < 0.001.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def causal_fill_eval() -> pd.DataFrame:
    norms = p05a.compute_fill_norms_from_train_sequences()
    rows = []
    for variant in VARIANTS:
        model, ckpt, selected = selected_model(variant)
        for split in SPLITS:
            payload = load_payload(variant, split)
            decision, _ = tcn_decision_for_payload(variant, payload, model, ckpt, float(selected["threshold"]))
            probs = p05a.fill_probabilities(payload, decision, norms)
            fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, FILL_RANDOM_SALT) < probs)
            pnl = p05a.decision_pnl_from_payload(payload, decision)
            total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
            rows.append(
                {
                    "split": split,
                    "policy_id": policy_id(variant),
                    "model_variant": variant,
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
    out = pd.DataFrame(rows)
    c4 = pd.read_csv(PHASE05C4 / "results" / "phc_stp_v2_causal_fill_eval.csv").assign(policy_id="phc_stp_v2_cosine")
    base = p05a.v1_causal_table() if hasattr(p05a, "v1_causal_table") else pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    compare = pd.concat(
        [
            base[base["policy_id"].isin(["side_by_queue_imbalance_abs_0p1", "ml_hgb_regressor_shallow_convergence_selected", "phc_lite"])],
            c4,
            out,
        ],
        ignore_index=True,
    )
    out.to_csv(RESULTS / "tcn_5block_prepad_causal_fill_eval.csv", index=False)
    compare.to_csv(RESULTS / "tcn_5block_prepad_model_comparison_causal_fill.csv", index=False)
    write_markdown(
        ROOT / "TCN_ACTION_VALUE_CAUSAL_FILL_REPORT.md",
        [
            "# TCN 5-Block PrePad Causal-Fill Report",
            "",
            "Same proxy family as PHC-lite/05C4: H20 exit, conservative L2 fill, inventory cap 1. Not true queue replay. NEW5 is not used.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def seed_stability() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            model, ckpt = load_model(variant, best_ckpt_path(variant, seed))
            val = load_payload(variant, "val")
            scores = predict_scores(model, val, scale=float(ckpt["target_scale"]))
            best_threshold = 0.0
            best_pnl = -float("inf")
            for threshold in threshold_grid(scores):
                decision = p05a.policy_from_scores(scores, float(threshold))
                pnl = float(p05a.decision_pnl_from_payload(val, decision).sum())
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_threshold = float(threshold)
            for split in SPLITS:
                payload = load_payload(variant, split)
                decision = p05a.policy_from_scores(predict_scores(model, payload, scale=float(ckpt["target_scale"])), best_threshold)
                metric = p05a.metric_from_decision(split, f"{policy_id(variant)}_seed{seed}", payload, decision)
                rows.append(
                    {
                        "model_variant": variant,
                        "seed": seed,
                        "split": split,
                        "best_epoch": int(ckpt.get("best_epoch", -1)),
                        "threshold_selected_on_val_for_seed": best_threshold,
                        "action_pnl": float(metric["total_excess_pnl_H20"]),
                        "selected_action_count": int(metric["selected_action_count"]),
                        "action_pnl_positive": bool(metric["total_excess_pnl_H20"] > 0),
                        "phase": PHASE,
                        "new3_used_for_selection": False,
                        "new4_used_for_selection": False,
                        "new5_used_for_selection": False,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_5block_prepad_seed_stability.csv", index=False)
    write_markdown(
        ROOT / "TCN_ACTION_VALUE_SEED_STABILITY_REPORT.md",
        [
            "# TCN 5-Block PrePad Seed Stability Report",
            "",
            "Each seed uses its own validation-selected threshold. NEW3/NEW4/NEW5 are not used for selection.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def oracle_capture() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    phc_policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    phc_model, phc_ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    c4_model, c4_ckpt, c4_selected = load_05c4_model()
    for variant in VARIANTS:
        model, ckpt, selected = selected_model(variant)
        for split in SPLITS:
            payload = load_payload(variant, split)
            c4_payload = p05a.load_sequences(split)
            decisions = {
                "side_by_queue_imbalance_abs_0p1": p05a.queue_rule_decision(c4_payload),
                "ml_hgb_regressor_shallow_convergence_selected": p05a.hgb_decision(c4_payload)[0],
                "phc_lite_v1": p05a.policy_from_scores(p05a.predict_scores(phc_model, c4_payload, target_scale=float(phc_ckpt["target_scale"])), float(phc_policy["threshold"])),
                "phc_stp_v2_cosine": c4_decision_for_payload(c4_payload, c4_model, c4_ckpt, float(c4_selected["threshold"]))[0],
                policy_id(variant): tcn_decision_for_payload(variant, payload, model, ckpt, float(selected["threshold"]))[0],
            }
            y_payload = payload
            bid = y_payload["y_bid"].numpy().astype(float)
            ask = y_payload["y_ask"].numpy().astype(float)
            best = np.maximum(bid, ask)
            side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK)
            for pid, decision in decisions.items():
                if len(decision) != len(best):
                    continue
                count = int((decision != p05a.DECISION_NO_QUOTE).sum())
                chosen = np.argsort(best)[::-1][:count]
                oracle_decision = np.zeros(len(best), dtype=np.int8)
                oracle_decision[chosen] = side[chosen].astype(np.int8)
                policy_pnl = float(p05a.decision_pnl_from_payload(y_payload, decision).sum())
                oracle_pnl = float(p05a.decision_pnl_from_payload(y_payload, oracle_decision).sum())
                rows.append(
                    {
                        "split": split,
                        "comparison_feature_universe": variant,
                        "policy_id": pid,
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
    out.to_csv(RESULTS / "tcn_5block_prepad_window_oracle_capture.csv", index=False)
    write_markdown(
        ROOT / "TCN_ACTION_VALUE_ORACLE_CAPTURE_REPORT.md",
        [
            "# TCN 5-Block PrePad Oracle Capture Report",
            "",
            "Oracle capture is matched-count same-universe sequence-window oracle. Oracle uses future labels and is diagnostic only.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def load_05c4_model() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if str(PHASE05C4) not in sys.path:
        sys.path.insert(0, str(PHASE05C4))
    import common_05c4 as c4  # noqa: PLC0415

    return c4.selected_model()


def c4_decision_for_payload(payload: dict[str, Any], model: Any, ckpt: dict[str, Any], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if str(PHASE05C4) not in sys.path:
        sys.path.insert(0, str(PHASE05C4))
    import common_05c4 as c4  # noqa: PLC0415

    return c4.stp_decision_for_payload(payload, model, ckpt, threshold)


def hls4ml_conversion() -> pd.DataFrame:
    ensure_dirs()
    variant = "18feat"
    status = {
        "model_variant": variant,
        "policy_id": policy_id(variant),
        "hls4ml_attempted": True,
        "cpp_project_generated": False,
        "hls4ml_convertible": False,
        "hls4ml_compile_success": False,
        "vivado_synthesis_run": False,
        "input_shape": "1,18,252",
        "precision": "ap_fixed<16,6>",
        "reuse_factor": 1,
        "strategy": "Latency",
        "constantpad1d_layers_present": False,
        "parsed_layer_count": 0,
        "hls4ml_version": "",
        "error": "",
    }
    try:
        import hls4ml  # type: ignore

        model, ckpt, selected = selected_model(variant)
        model.eval()
        status["hls4ml_version"] = str(getattr(hls4ml, "__version__", "unknown"))
        status["constantpad1d_layers_present"] = bool(any(m.__class__.__name__ == "ConstantPad1d" for m in model.modules()))
        cfg = hls4ml.utils.config_from_pytorch_model(
            model,
            input_shape=(1, num_inputs(variant), SEQ_LEN_IN),
            default_precision="ap_fixed<16,6>",
            granularity="name",
        )
        for layer_cfg in cfg.get("LayerName", {}).values():
            layer_cfg["ReuseFactor"] = 1
        out_dir = HLS4ML_PRJ / "tcn_5block_prepad_18feat"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        hls_model = hls4ml.converters.convert_from_pytorch_model(
            model,
            hls_config=cfg,
            input_shape=(1, num_inputs(variant), SEQ_LEN_IN),
            output_dir=str(out_dir),
            project_name="tcn_5block_prepad_18feat",
            backend="Vivado",
        )
        layers = getattr(hls_model, "get_layers", lambda: [])()
        status["parsed_layer_count"] = int(len(list(layers)))
        hls_model.compile()
        hls_model.write()
        status.update(
            {
                "cpp_project_generated": bool(out_dir.exists()),
                "hls4ml_convertible": True,
                "hls4ml_compile_success": True,
                "selected_seed": int(selected["seed"]),
                "best_epoch": int(ckpt["best_epoch"]),
                "error": "",
            }
        )
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    out = pd.DataFrame([status])
    out.to_csv(RESULTS / "tcn_5block_prepad_18feat_hls4ml_conversion_status.csv", index=False)
    write_markdown(
        ROOT / "TCN_ACTION_VALUE_HLS4ML_CONVERSION_REPORT.md",
        [
            "# TCN 5-Block PrePad hls4ml Conversion Report",
            "",
            "`tcn_5block_prepad_18feat` is attempted with dummy input shape `(1, 18, 252)`. The model contains no `nn.ConstantPad1d` layers; the caller pre-pads input tensors. No Vivado synthesis is run.",
            "",
            out.to_markdown(index=False),
        ],
    )
    return out


def adoption_gates(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        pid = policy_id(variant)
        sub = summary[summary["policy_id"].eq(pid)]
        if sub.empty:
            continue
        conv = convergence_summary()
        conv_sub = conv[conv["model_variant"].eq(variant)]
        random_path = RESULTS / "tcn_5block_prepad_matched_random_summary.csv"
        random = pd.read_csv(random_path) if random_path.exists() else pd.DataFrame()
        seed_path = RESULTS / "tcn_5block_prepad_seed_stability.csv"
        seed = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
        causal_path = RESULTS / "tcn_5block_prepad_causal_fill_eval.csv"
        causal = pd.read_csv(causal_path) if causal_path.exists() else pd.DataFrame()
        oracle_path = RESULTS / "tcn_5block_prepad_window_oracle_capture.csv"
        oracle = pd.read_csv(oracle_path) if oracle_path.exists() else pd.DataFrame()
        rows.extend(
            [
                {
                    "model_variant": variant,
                    "policy_id": pid,
                    "gate": "Convergence closed",
                    "pass": bool(len(conv_sub) == len(SEEDS) and conv_sub["convergence_closed"].astype(bool).all()),
                    "rule": "all seeds early_stop=True and best_epoch<185",
                },
                {
                    "model_variant": variant,
                    "policy_id": pid,
                    "gate": "Seed stability",
                    "pass": bool(not seed.empty and seed[seed["model_variant"].eq(variant)]["action_pnl_positive"].astype(bool).all()),
                    "rule": "all seeds x splits action PnL > 0",
                },
                {
                    "model_variant": variant,
                    "policy_id": pid,
                    "gate": "Matched random",
                    "pass": bool(not random.empty and random[random["model_variant"].eq(variant)]["beats_random_95p"].astype(bool).all()),
                    "rule": "val/NEW3/NEW4 all beat random 95p",
                },
                {
                    "model_variant": variant,
                    "policy_id": pid,
                    "gate": "Action PnL improvement over PHC-lite v1",
                    "pass": bool((sub["action_delta_vs_phc_lite_v1"] >= 0).all()),
                    "rule": "val/NEW3/NEW4 action PnL >= PHC-lite v1",
                },
                {
                    "model_variant": variant,
                    "policy_id": pid,
                    "gate": "Causal-fill improvement over PHC-lite v1",
                    "pass": bool(not causal.empty and (summary[summary["policy_id"].eq(pid)]["causal_fill_delta_vs_phc_lite_v1"] >= 0).all()),
                    "rule": "val/NEW3/NEW4 sequential PnL >= PHC-lite v1",
                },
                {
                    "model_variant": variant,
                    "policy_id": pid,
                    "gate": "Oracle capture improvement over PHC-lite v1",
                    "pass": bool((sub["oracle_capture_delta_vs_phc_lite_v1"] >= 0).all()),
                    "rule": "val/NEW3/NEW4 capture ratio >= PHC-lite v1",
                },
            ]
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tcn_5block_prepad_adoption_gates.csv", index=False)
    return out


def final_report() -> str:
    action = pd.read_csv(RESULTS / "tcn_5block_prepad_oos_eval.csv") if (RESULTS / "tcn_5block_prepad_oos_eval.csv").exists() else pd.DataFrame()
    causal = pd.read_csv(RESULTS / "tcn_5block_prepad_causal_fill_eval.csv") if (RESULTS / "tcn_5block_prepad_causal_fill_eval.csv").exists() else pd.DataFrame()
    oracle = pd.read_csv(RESULTS / "tcn_5block_prepad_window_oracle_capture.csv") if (RESULTS / "tcn_5block_prepad_window_oracle_capture.csv").exists() else pd.DataFrame()
    c4_action = pd.read_csv(PHASE05C4 / "results" / "phc_stp_v2_oos_eval.csv").assign(policy_id="phc_stp_v2_cosine")
    c4_causal = pd.read_csv(PHASE05C4 / "results" / "phc_stp_v2_causal_fill_eval.csv").assign(policy_id="phc_stp_v2_cosine")
    c4_oracle = pd.read_csv(PHASE05C4 / "results" / "phc_stp_v2_oracle_capture_vs_v1.csv").assign(policy_id="phc_stp_v2_cosine")
    v1_action = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    v1_causal = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    all_pids = [
        "side_by_queue_imbalance_abs_0p1",
        "ml_hgb_regressor_shallow_convergence_selected",
        "phc_lite",
        "phc_stp_v2_cosine",
        "tcn_action_value_18feat",
        "tcn_5block_prepad_18feat",
    ]
    rows = []
    for split in SPLITS:
        phc_action = float(v1_action[(v1_action["split"].eq(split)) & (v1_action["policy_id"].eq("phc_lite"))]["total_excess_pnl_H20"].iloc[0])
        phc_causal = float(v1_causal[(v1_causal["split"].eq(split)) & (v1_causal["policy_id"].eq("phc_lite"))]["sequential_excess_pnl"].iloc[0])
        phc_oracle = float(c4_oracle[c4_oracle["split"].eq(split)]["phc_lite_v1_capture_ratio"].iloc[0])
        for pid in all_pids:
            if pid == "tcn_5block_prepad_18feat":
                a = action[(action["split"].eq(split)) & (action["policy_id"].eq(pid))]
                c = causal[(causal["split"].eq(split)) & (causal["policy_id"].eq(pid))]
                o = oracle[(oracle["split"].eq(split)) & (oracle["policy_id"].eq(pid))]
            elif pid == "tcn_action_value_18feat":
                a = pd.read_csv(PHASE06C / "results" / "tcn_av_oos_eval.csv")
                c = pd.read_csv(PHASE06C / "results" / "tcn_av_causal_fill_eval.csv")
                o = pd.read_csv(PHASE06C / "results" / "tcn_av_window_oracle_capture.csv")
                a = a[(a["split"].eq(split)) & (a["policy_id"].eq(pid))]
                c = c[(c["split"].eq(split)) & (c["policy_id"].eq(pid))]
                o = o[(o["split"].eq(split)) & (o["policy_id"].eq(pid))]
            elif pid == "phc_stp_v2_cosine":
                a = c4_action[c4_action["split"].eq(split)]
                c = c4_causal[c4_causal["split"].eq(split)]
                o = c4_oracle[c4_oracle["split"].eq(split)]
            else:
                a = v1_action[(v1_action["split"].eq(split)) & (v1_action["policy_id"].eq(pid if pid != "phc_lite" else "phc_lite"))]
                c = v1_causal[(v1_causal["split"].eq(split)) & (v1_causal["policy_id"].eq(pid if pid != "phc_lite" else "phc_lite"))]
                o = oracle[(oracle["split"].eq(split)) & (oracle["policy_id"].eq("phc_lite_v1" if pid == "phc_lite" else pid))]
            if a.empty:
                continue
            action_pnl = float(a["total_excess_pnl_H20"].iloc[0])
            causal_pnl = float(c["sequential_excess_pnl"].iloc[0]) if not c.empty else np.nan
            capture = float(o["capture_ratio"].iloc[0]) if not o.empty and "capture_ratio" in o.columns else (phc_oracle if pid == "phc_lite" else np.nan)
            rows.append(
                {
                    "split": split,
                    "policy_id": pid,
                    "action_pnl": action_pnl,
                    "causal_fill_pnl": causal_pnl,
                    "oracle_capture": capture,
                    "action_delta_vs_phc_lite_v1": action_pnl - phc_action,
                    "causal_fill_delta_vs_phc_lite_v1": causal_pnl - phc_causal if np.isfinite(causal_pnl) else np.nan,
                    "oracle_capture_delta_vs_phc_lite_v1": capture - phc_oracle if np.isfinite(capture) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "tcn_5block_prepad_model_comparison_summary.csv", index=False)
    gates = adoption_gates(summary)
    hls = pd.read_csv(RESULTS / "tcn_5block_prepad_18feat_hls4ml_conversion_status.csv") if (RESULTS / "tcn_5block_prepad_18feat_hls4ml_conversion_status.csv").exists() else pd.DataFrame()
    rf_table = receptive_field_comparison()
    status = "tcn_5block_prepad_not_adopted"
    for variant in VARIANTS:
        sub = gates[gates["model_variant"].eq(variant)]
        if not sub.empty and sub["pass"].astype(bool).all():
            status = "tcn_5block_prepad_adopted"
    write_markdown(
        ROOT / "TCN_5BLOCK_PREPAD_FINAL_REPORT.md",
        [
            "# TCN 5-Block PrePad Final Report",
            "",
            f"Final status: `{status}`.",
            "",
            "## 1. Architecture Summary",
            "",
            f"- Blocks: `5`.",
            f"- Dilations: `{TCNPrePadRegressor.DILATIONS}`.",
            f"- Kernel size: `{KERNEL_SIZE}`.",
            f"- Receptive field: `125` ticks.",
            f"- Output sequence length: `{SEQ_LEN}`.",
            f"- Pre-padded input length: `{SEQ_LEN_IN}`.",
            f"- Total left prepad: `{TOTAL_PAD}`.",
            f"- Channels: `{CHANNEL_LIST}`.",
            f"- Variant: `18feat` only.",
            "",
            "## 2. Receptive Field Comparison",
            "",
            rf_table.to_markdown(index=False),
            "",
            "## 3. Model Comparison",
            "",
            summary.to_markdown(index=False),
            "",
            "## 4. Architecture Comparison Notes",
            "",
            "`tcn_5block_prepad_18feat` is compared against the 06C `tcn_action_value_18feat`, PHC-lite v1, and 05C4 STP v2. The 06D change is the deeper valid-conv TCN with external pre-padding; it still uses the 18-feature sequence tensor.",
            "",
            "## 5. Adoption Gates",
            "",
            gates.to_markdown(index=False),
            "",
            "## 6. hls4ml Status",
            "",
            hls.to_markdown(index=False) if not hls.empty else "hls4ml conversion was not run.",
            "",
            "## 7. Conclusion",
            "",
            "Do not claim this TCN beats PHC-lite unless all six gates pass. If all gates fail to pass, 06D is a diagnostic A3D3-track contribution: it tests whether the 06C receptive-field and ConstantPad1d issues were the limiting factors.",
            "",
            "## 8. Forbidden Claims",
            "",
            "- No NEW5 usage.",
            "- No true exchange replay claim.",
            "- No real-world profitability claim.",
            "- No FPGA PnL improvement claim.",
            "- hls4ml conversion, if successful, is C++ project generation only; Vivado synthesis is not run.",
            "- Do not claim TCN beats PHC-lite unless all six adoption gates pass.",
        ],
    )
    return status


def receptive_field_comparison() -> pd.DataFrame:
    out = pd.DataFrame(
        [
            {
                "phase": "06c",
                "model": "3-block dilated TCN",
                "dilations": "[1, 2, 4]",
                "internal_padding": "ConstantPad1d",
                "seq_len_in": 128,
                "seq_len_out": 128,
                "receptive_field_ticks": 29,
                "hls4ml_issue": "ConstantPad1d parser failed",
            },
            {
                "phase": "06d",
                "model": "5-block prepad valid-conv TCN",
                "dilations": str(TCNPrePadRegressor.DILATIONS),
                "internal_padding": "none",
                "seq_len_in": SEQ_LEN_IN,
                "seq_len_out": SEQ_LEN,
                "receptive_field_ticks": 125,
                "hls4ml_issue": "prepad outside model; no ConstantPad1d",
            },
        ]
    )
    out.to_csv(RESULTS / "tcn_5block_prepad_rf_comparison.csv", index=False)
    return out


def run_all() -> None:
    ensure_dirs()
    build_enriched_sequences()
    train_all(overwrite=False)
    select_policy()
    evaluate_oos()
    matched_random()
    causal_fill_eval()
    seed_stability()
    oracle_capture()
    hls4ml_conversion()
    print(final_report())


if __name__ == "__main__":
    run_all()
