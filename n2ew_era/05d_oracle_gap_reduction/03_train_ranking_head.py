from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import common_05d as c

PHASE05A = c.PHASE05A
if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))

import common_05a as p05a  # noqa: E402

HIDDEN_DIM = 16
MTL_STEPS = 1
BATCH_SIZE = 4096
MAX_EPOCHS = 50
PATIENCE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
MARGIN = 0.10
LAMBDA_GRID = [0.1, 0.3, 1.0]
SEEDS = [0, 1, 2]
PILOT_VARIANTS = [
    {"variant": "phc_rank_head", "lambda_rank": 0.1, "lambda_gate": 0.0},
    {"variant": "phc_rank_head", "lambda_rank": 0.3, "lambda_gate": 0.0},
    {"variant": "phc_rank_head", "lambda_rank": 1.0, "lambda_gate": 0.0},
    {"variant": "phc_rank_head_plus_quote_gate", "lambda_rank": 0.3, "lambda_gate": 0.1},
]


class PHCRankingHead(nn.Module):
    """PHC-lite v1 core with ranking/quote-gate auxiliary outputs.

    This is not full PHCSSM. The architecture intentionally mirrors frozen
    PHC-lite v1 so the ablation isolates the auxiliary action-ranking loss.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = HIDDEN_DIM, mtl_steps: int = MTL_STEPS) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.mtl_steps = int(mtl_steps)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.candidate = nn.Linear(hidden_dim, hidden_dim)
        self.synapse = nn.Linear(hidden_dim, hidden_dim)
        self.loop_norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Linear(hidden_dim, 2)
        self.quote_gate = nn.Linear(hidden_dim, 1)
        self.decay_param = nn.Parameter(torch.logit(torch.tensor(0.90)).repeat(hidden_dim))
        self.alpha = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, feature_dim], got {tuple(x.shape)}")
        batch, seq_len, _ = x.shape
        state = x.new_zeros((batch, self.hidden_dim))
        decay = torch.sigmoid(self.decay_param).view(1, -1)
        flat = x.reshape(batch * seq_len, self.feature_dim)
        encoded = self.encoder(flat).reshape(batch, seq_len, self.hidden_dim)
        gates = torch.sigmoid(self.gate(encoded.reshape(batch * seq_len, self.hidden_dim))).reshape(batch, seq_len, self.hidden_dim)
        candidates = torch.tanh(self.candidate(encoded.reshape(batch * seq_len, self.hidden_dim))).reshape(batch, seq_len, self.hidden_dim)
        for t in range(seq_len):
            z = decay * state + gates[:, t, :] * candidates[:, t, :]
            for _ in range(self.mtl_steps):
                message = torch.nn.functional.silu(self.synapse(z))
                z = self.loop_norm(z + self.alpha * message)
            state = z
        return self.readout(state), self.quote_gate(state).squeeze(-1)


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any], target_scale: float) -> None:
        self.x = payload["x"]
        y = torch.stack([payload["y_bid"], payload["y_ask"]], dim=1).float()
        self.y = y / float(target_scale)
        no_quote = torch.zeros((len(y), 1), dtype=torch.float32)
        stacked = torch.cat([self.y, no_quote], dim=1)
        self.oracle_action = torch.argmax(stacked, dim=1).long()
        self.quote_should_act = (torch.maximum(y[:, 0], y[:, 1]) > 0).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx].float(), self.y[idx], self.oracle_action[idx], self.quote_should_act[idx]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def predict_scores(model: PHCRankingHead, payload: dict[str, Any], target_scale: float, batch_size: int = 4096, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    gates = []
    with torch.no_grad():
        for start in range(0, len(payload["x"]), batch_size):
            xb = payload["x"][start : start + batch_size].to(device=device, dtype=torch.float32)
            out, gate = model(xb)
            scores.append(out.detach().cpu().numpy().astype(np.float32))
            gates.append(gate.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0) * float(target_scale), np.concatenate(gates, axis=0)


def ranking_loss(scores: torch.Tensor, target_action: torch.Tensor, margin: float) -> torch.Tensor:
    no_quote = scores.new_zeros((scores.shape[0], 1))
    all_scores = torch.cat([scores, no_quote], dim=1)
    target = all_scores.gather(1, target_action.view(-1, 1))
    losses = torch.relu(float(margin) - (target - all_scores))
    mask = torch.ones_like(losses)
    mask = mask.scatter(1, target_action.view(-1, 1), 0.0)
    return (losses * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def train_one(seed: int, variant: str, lambda_rank: float, lambda_gate: float) -> pd.DataFrame:
    set_seed(seed)
    scaler = p05a.load_scaler()
    target_scale = float(scaler["target_scale_train_std_no_mean_shift"])
    train_payload = p05a.load_sequences("train")
    val_payload = p05a.load_sequences("val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PHCRankingHead(len(p05a.PHC_FEATURES)).to(device)
    init_path = PHASE05A / "models" / f"phc_lite_seed{seed}_best_val.pt"
    initialized_from = ""
    if init_path.exists():
        base_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        current = model.state_dict()
        copied = 0
        for key, value in base_ckpt["model_state_dict"].items():
            if key in current and current[key].shape == value.shape:
                current[key] = value
                copied += int(value.numel())
        model.load_state_dict(current)
        initialized_from = str(init_path)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    huber = torch.nn.HuberLoss(delta=1.0)
    bce = torch.nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(
        RankingDataset(train_payload, target_scale),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    best_metric = -float("inf")
    early_counter = 0
    rows = []
    run_id = f"{variant}_lam{str(lambda_rank).replace('.', 'p')}_seed{seed}"
    curve_path = c.RESULTS / f"phc_rank_training_curve_{run_id}.csv"
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        grad_norms = []
        for xb, yb, oracle_action, quote_should_act in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            oracle_action = oracle_action.to(device=device)
            quote_should_act = quote_should_act.to(device=device)
            opt.zero_grad(set_to_none=True)
            scores, quote_logit = model(xb)
            loss_huber = huber(scores, yb)
            loss_rank = ranking_loss(scores, oracle_action, MARGIN)
            loss_gate = bce(quote_logit, quote_should_act) if lambda_gate > 0 else scores.new_tensor(0.0)
            loss = loss_huber + float(lambda_rank) * loss_rank + float(lambda_gate) * loss_gate
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
            opt.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(grad_norm)
        val_scores, val_gate = predict_scores(model, val_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
        train_scores, _ = predict_scores(model, train_payload, target_scale=target_scale, batch_size=BATCH_SIZE, device=device)
        train_y = np.column_stack([train_payload["y_bid"].numpy(), train_payload["y_ask"].numpy()])
        val_y = np.column_stack([val_payload["y_bid"].numpy(), val_payload["y_ask"].numpy()])
        val_decision = p05a.policy_from_scores(val_scores, 0.0)
        val_pnl = float(p05a.decision_pnl_from_payload(val_payload, val_decision).sum())
        row = {
            "run_id": run_id,
            "variant": variant,
            "seed": seed,
            "lambda_rank": lambda_rank,
            "lambda_gate": lambda_gate,
            "margin": MARGIN,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_huber_loss": p05a.huber_np((val_scores - val_y) / target_scale),
            "train_huber_loss": p05a.huber_np((train_scores - train_y) / target_scale),
            "val_policy_excess_pnl": val_pnl,
            "val_selected_count": int((val_decision != p05a.DECISION_NO_QUOTE).sum()),
            "val_selection_rate": float((val_decision != p05a.DECISION_NO_QUOTE).mean()),
            "val_spearman_bid": p05a.safe_spearman(val_scores[:, 0], val_y[:, 0]),
            "val_spearman_ask": p05a.safe_spearman(val_scores[:, 1], val_y[:, 1]),
            "val_pr_auc_bid": p05a.safe_pr_auc(val_y[:, 0] > 0, val_scores[:, 0]),
            "val_pr_auc_ask": p05a.safe_pr_auc(val_y[:, 1] > 0, val_scores[:, 1]),
            "quote_gate_mean": float(np.mean(val_gate)),
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "early_stop_counter": early_counter,
            "parameter_count": parameter_count(model),
            "device": device,
            "initialized_from_phc_v1_checkpoint": initialized_from,
            "new3_used_for_selection": False,
            "new4_used_for_selection": False,
            "new5_used_for_selection": False,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(curve_path, index=False)
        if val_pnl > best_metric + 1e-3:
            best_metric = val_pnl
            early_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": len(p05a.PHC_FEATURES),
                    "hidden_dim": HIDDEN_DIM,
                    "mtl_steps": MTL_STEPS,
                    "seed": seed,
                    "variant": variant,
                    "lambda_rank": lambda_rank,
                    "lambda_gate": lambda_gate,
                    "margin": MARGIN,
                    "best_epoch": epoch,
                    "target_scale": target_scale,
                    "selection_metric": "validation_policy_excess_pnl_at_threshold_0",
                    "initialized_from_phc_v1_checkpoint": initialized_from,
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                },
                c.MODELS / f"phc_rank_{run_id}_best_val.pt",
            )
        else:
            early_counter += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_dim": len(p05a.PHC_FEATURES),
                "hidden_dim": HIDDEN_DIM,
                "mtl_steps": MTL_STEPS,
                "seed": seed,
                "variant": variant,
                "lambda_rank": lambda_rank,
                "lambda_gate": lambda_gate,
                "margin": MARGIN,
                "epoch": epoch,
                "target_scale": target_scale,
                "initialized_from_phc_v1_checkpoint": initialized_from,
            },
            c.MODELS / f"phc_rank_{run_id}_last.pt",
        )
        print(f"rank run={run_id} epoch={epoch} val_pnl={val_pnl:.2f} selected={row['val_selected_count']} grad={row['grad_norm']:.4f}", flush=True)
        if early_counter >= PATIENCE:
            break
    return pd.DataFrame(rows)


def load_rank_model(run_id: str) -> tuple[PHCRankingHead, dict[str, Any]]:
    ckpt = torch.load(c.MODELS / f"phc_rank_{run_id}_best_val.pt", map_location="cpu", weights_only=False)
    model = PHCRankingHead(int(ckpt["feature_dim"]), hidden_dim=int(ckpt["hidden_dim"]), mtl_steps=int(ckpt["mtl_steps"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def threshold_grid(scores: np.ndarray) -> list[float]:
    finite = np.max(scores, axis=1)
    finite = finite[np.isfinite(finite)]
    qs = np.quantile(finite, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]) if len(finite) else np.array([0.0])
    return sorted(set([0.0, *qs.tolist()]))


def evaluate_run_on_val(run_id: str) -> pd.DataFrame:
    model, ckpt = load_rank_model(run_id)
    val = p05a.load_sequences("val")
    scores, _ = predict_scores(model, val, target_scale=float(ckpt["target_scale"]))
    rows = []
    for threshold in threshold_grid(scores):
        decision = p05a.policy_from_scores(scores, threshold)
        metric = p05a.metric_from_decision("val", run_id, val, decision)
        oracle = c.oracle_capture_for_decision(val, decision)
        metric.update(
            {
                "run_id": run_id,
                "variant": ckpt["variant"],
                "seed": int(ckpt["seed"]),
                "lambda_rank": float(ckpt["lambda_rank"]),
                "lambda_gate": float(ckpt["lambda_gate"]),
                "best_epoch": int(ckpt["best_epoch"]),
                "threshold": float(threshold),
                "oracle_capture_ratio": oracle["capture_ratio"],
                "selection_split": "val",
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used_for_selection": False,
            }
        )
        rows.append(metric)
    return pd.DataFrame(rows)


def train_and_select() -> dict[str, Any]:
    all_curves = []
    pilot_run_ids = []
    for cfg in PILOT_VARIANTS:
        curve = train_one(seed=0, **cfg)
        all_curves.append(curve)
        pilot_run_ids.append(curve.iloc[0]["run_id"])
    pilot_search = pd.concat([evaluate_run_on_val(run_id) for run_id in pilot_run_ids], ignore_index=True)
    pilot_search.to_csv(c.RESULTS / "phc_rank_pilot_threshold_search.csv", index=False)
    best_pilot = pilot_search.sort_values(["total_excess_pnl_H20", "oracle_capture_ratio", "selected_action_count"], ascending=[False, False, True]).iloc[0].to_dict()
    selected_variant = str(best_pilot["variant"])
    selected_lambda_rank = float(best_pilot["lambda_rank"])
    selected_lambda_gate = float(best_pilot["lambda_gate"])
    seed_run_ids = [str(best_pilot["run_id"])]
    for seed in [1, 2]:
        curve = train_one(seed=seed, variant=selected_variant, lambda_rank=selected_lambda_rank, lambda_gate=selected_lambda_gate)
        all_curves.append(curve)
        seed_run_ids.append(curve.iloc[0]["run_id"])
    curves = pd.concat(all_curves, ignore_index=True)
    curves.to_csv(c.RESULTS / "phc_rank_training_curve.csv", index=False)
    threshold_search = pd.concat([evaluate_run_on_val(run_id) for run_id in seed_run_ids], ignore_index=True)
    threshold_search.to_csv(c.RESULTS / "phc_rank_threshold_search.csv", index=False)
    best = threshold_search.sort_values(["total_excess_pnl_H20", "oracle_capture_ratio", "selected_action_count"], ascending=[False, False, True]).iloc[0].to_dict()
    selected = {
        "model": str(best["variant"]),
        "run_id": str(best["run_id"]),
        "selection_split": "val",
        "target": "excess_pnl_H20",
        "seed": int(best["seed"]),
        "lambda_rank": float(best["lambda_rank"]),
        "lambda_gate": float(best["lambda_gate"]),
        "margin": MARGIN,
        "threshold": float(best["threshold"]),
        "best_epoch": int(best["best_epoch"]),
        "selection_notes": "variant/lambda/threshold selected on validation only; NEW3/NEW4/NEW5 not used for selection",
        "validation_metrics": p05a.json_sanitize(best),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
    }
    c.write_json(c.RESULTS / "phc_rank_selected_policy.json", selected)
    return selected


def evaluate_selected(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model, ckpt = load_rank_model(str(selected["run_id"]))
    threshold = float(selected["threshold"])
    action_rows = []
    causal_rows = []
    oracle_rows = []
    error_rows = []
    for split in c.SPLITS:
        payload = p05a.load_sequences(split)
        scores, gate = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decision = p05a.policy_from_scores(scores, threshold)
        metric = p05a.metric_from_decision(split, str(selected["model"]), payload, decision)
        metric.update(
            {
                "run_id": selected["run_id"],
                "lambda_rank": selected["lambda_rank"],
                "lambda_gate": selected["lambda_gate"],
                "threshold": threshold,
                "quote_gate_mean": float(np.mean(gate)),
            }
        )
        action_rows.append(metric)
        causal_rows.append(c.causal_fill_for_decision(split, str(selected["model"]), payload, decision, "05d_rank_selected"))
        cap = c.oracle_capture_for_decision(payload, decision)
        cap.update({"split": split, "policy_id": str(selected["model"]), "run_id": selected["run_id"]})
        oracle_rows.append(cap)
        oracle_action, _, _ = c.oracle_decision_and_pnl(payload)
        err = c.error_type(decision, oracle_action)
        for error_name, count in pd.Series(err).value_counts().items():
            error_rows.append({"split": split, "policy_id": str(selected["model"]), "error_type": error_name, "state_count": int(count), "error_rate": float(count / max(len(err), 1))})
    action = pd.DataFrame(action_rows)
    causal = pd.DataFrame(causal_rows)
    oracle = pd.DataFrame(oracle_rows)
    errors = pd.DataFrame(error_rows)
    action.to_csv(c.RESULTS / "phc_rank_oos_eval.csv", index=False)
    action.to_csv(c.RESULTS / "phc_rank_vs_v1_hgb_queue.csv", index=False)
    causal.to_csv(c.RESULTS / "phc_rank_causal_fill.csv", index=False)
    oracle.to_csv(c.RESULTS / "phc_rank_oracle_capture.csv", index=False)
    errors.to_csv(c.RESULTS / "phc_rank_oracle_error.csv", index=False)
    return action, causal, oracle, errors


def run_matched_random(selected: dict[str, Any]) -> pd.DataFrame:
    model, ckpt = load_rank_model(str(selected["run_id"]))
    rows = []
    for split in c.SPLITS:
        payload = p05a.load_sequences(split)
        scores, _ = predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
        decision = p05a.policy_from_scores(scores, float(selected["threshold"]))
        _, summary = p05a.matched_random_for_split(split, decision, n_trials=1000, seed=20260604)
        summary["policy_id"] = str(selected["model"])
        summary["run_id"] = selected["run_id"]
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(c.RESULTS / "phc_rank_matched_random_summary.csv", index=False)
    return out


def write_report(selected: dict[str, Any], action: pd.DataFrame, causal: pd.DataFrame, oracle: pd.DataFrame, random_summary: pd.DataFrame) -> None:
    v1 = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")
    v1 = v1[v1["policy_id"].eq("phc_lite")][["split", "total_excess_pnl_H20"]].rename(columns={"total_excess_pnl_H20": "v1_action_pnl"})
    v1_causal = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    v1_causal = v1_causal[v1_causal["policy_id"].eq("phc_lite")][["split", "sequential_excess_pnl"]].rename(columns={"sequential_excess_pnl": "v1_causal_pnl"})
    merged = action.merge(v1, on="split", how="left").merge(causal[["split", "sequential_excess_pnl"]], on="split", how="left").merge(v1_causal, on="split", how="left").merge(oracle[["split", "capture_ratio"]], on="split", how="left")
    merged["action_delta_vs_v1"] = merged["total_excess_pnl_H20"] - merged["v1_action_pnl"]
    merged["causal_delta_vs_v1"] = merged["sequential_excess_pnl"] - merged["v1_causal_pnl"]
    merged.to_csv(c.RESULTS / "phc_rank_vs_v1_summary.csv", index=False)
    accepted = bool((merged["action_delta_vs_v1"] >= 0).all() and (merged["causal_delta_vs_v1"] >= 0).all() and random_summary["beats_random_95p"].all())
    lines = [
        "# PHC Ranking Head Report",
        "",
        "This experiment adds a validation-selected ranking-aware auxiliary loss to PHC-lite v1. Oracle labels are used only inside train/validation labels for the ranking target; NEW3/NEW4/NEW5 are not used for selection.",
        "",
        "## Selected Configuration",
        f"- Variant: `{selected['model']}`.",
        f"- Run id: `{selected['run_id']}`.",
        f"- Lambda rank: `{selected['lambda_rank']}`.",
        f"- Lambda gate: `{selected['lambda_gate']}`.",
        f"- Threshold: `{selected['threshold']}`.",
        "",
        "## Comparison Against PHC-lite v1",
        merged[["split", "total_excess_pnl_H20", "v1_action_pnl", "action_delta_vs_v1", "sequential_excess_pnl", "v1_causal_pnl", "causal_delta_vs_v1", "capture_ratio"]].to_markdown(index=False),
        "",
        "## Matched Random",
        random_summary[["split", "model_total_excess_pnl_H20", "random_95p", "beats_random_95p", "empirical_p_value_wording"]].to_markdown(index=False),
        "",
        "## Decision",
        (
            "`ranking_head_candidate`: ranking head passes the required action, causal-fill, and matched-random gates versus PHC-lite v1."
            if accepted
            else "`ranking_head_not_adopted`: ranking head does not improve PHC-lite v1 across all required OOS gates."
        ),
        "",
        "This remains proxy excess markout research, not true exchange replay or real-world profitability proof.",
    ]
    c.write_markdown(c.ROOT / "PHC_RANKING_HEAD_REPORT.md", lines)


def main() -> None:
    c.ensure_dirs()
    selected = train_and_select()
    action, causal, oracle, _ = evaluate_selected(selected)
    random_summary = run_matched_random(selected)
    write_report(selected, action, causal, oracle, random_summary)
    print("wrote PHC ranking head outputs")


if __name__ == "__main__":
    main()
