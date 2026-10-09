from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import f_classif


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07l_ultrafast_policy_sweep"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PHASE07G = PROJECT_ROOT / "n2ew_era" / "07g_tiny_mlp_policy"
PHASE07G_MAIN = PHASE07G / "00_run_tiny_mlp_policy.py"
PHASE07G_CORRECTION = PHASE07G / "02_audit_terminal_settlement_correction.py"

SPLITS = ("val", "new3", "new4")
QAT_FORMATS = ((8, 4), (10, 5), (12, 6))
QAT_EPOCHS = 50
QAT_PATIENCE = 10
BATCH_SIZE = 2048
LEARNING_RATE = 2e-4


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase = load_module("phase07g_for_07l", PHASE07G_MAIN)
correction = load_module("phase07g_correction_for_07l", PHASE07G_CORRECTION)
helpers = phase.helpers
p05a = phase.p05a


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fake_quantize(value: torch.Tensor, total_bits: int, integer_bits: int) -> torch.Tensor:
    frac_bits = total_bits - integer_bits
    scale = float(1 << frac_bits)
    qmin = float(-(1 << (total_bits - 1)))
    qmax = float((1 << (total_bits - 1)) - 1)
    quantized = torch.clamp(torch.floor(value * scale), qmin, qmax) / scale
    return value + (quantized - value).detach()


class QATTinyMLP(torch.nn.Module):
    def __init__(self, total_bits: int, integer_bits: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 8)
        self.fc2 = torch.nn.Linear(8, 3)
        self.total_bits = total_bits
        self.integer_bits = integer_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        args = (self.total_bits, self.integer_bits)
        xq = fake_quantize(x, *args)
        w1 = fake_quantize(self.fc1.weight, *args)
        b1 = fake_quantize(self.fc1.bias, *args)
        hidden = torch.nn.functional.linear(xq, w1, b1).relu()
        hidden = fake_quantize(hidden, *args)
        w2 = fake_quantize(self.fc2.weight, *args)
        b2 = fake_quantize(self.fc2.bias, *args)
        return torch.nn.functional.linear(hidden, w2, b2)


def load_champion_state(model: QATTinyMLP) -> None:
    selected = json.loads((PHASE07G / "results" / "tiny_mlp_selected.json").read_text(encoding="utf-8"))
    state = selected["state_dict"]
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(state["net.0.weight"], dtype=torch.float32))
        model.fc1.bias.copy_(torch.tensor(state["net.0.bias"], dtype=torch.float32))
        model.fc2.weight.copy_(torch.tensor(state["net.2.weight"], dtype=torch.float32))
        model.fc2.bias.copy_(torch.tensor(state["net.2.bias"], dtype=torch.float32))


def quantize_int(values: np.ndarray, total_bits: int, integer_bits: int) -> tuple[np.ndarray, int]:
    frac = total_bits - integer_bits
    scaled = np.floor(np.asarray(values, dtype=np.float64) * float(1 << frac))
    qmin = -(1 << (total_bits - 1))
    qmax = (1 << (total_bits - 1)) - 1
    overflow = int(np.count_nonzero((scaled < qmin) | (scaled > qmax)))
    return np.clip(scaled, qmin, qmax).astype(np.int64), overflow


def qat_integer_predict(model: QATTinyMLP, x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    bits, integer = model.total_bits, model.integer_bits
    frac = bits - integer
    xq, ox = quantize_int(x, bits, integer)
    w1, ow1 = quantize_int(model.fc1.weight.detach().numpy(), bits, integer)
    b1, ob1 = quantize_int(model.fc1.bias.detach().numpy(), bits, integer)
    w2, ow2 = quantize_int(model.fc2.weight.detach().numpy(), bits, integer)
    b2, ob2 = quantize_int(model.fc2.bias.detach().numpy(), bits, integer)
    hidden_acc = xq @ w1.T + (b1 << frac)
    hidden_acc = np.maximum(hidden_acc, 0)
    hidden = hidden_acc >> frac
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    oh = int(np.count_nonzero((hidden < qmin) | (hidden > qmax)))
    hidden = np.clip(hidden, qmin, qmax)
    output_acc = hidden @ w2.T + (b2 << frac)
    return output_acc.astype(np.float64) / float(1 << (2 * frac)), {
        "input_saturation_count": ox,
        "w1_saturation_count": ow1,
        "b1_saturation_count": ob1,
        "hidden_saturation_count": oh,
        "w2_saturation_count": ow2,
        "b2_saturation_count": ob2,
    }


def train_qat(
    total_bits: int, integer_bits: int, train_x: np.ndarray, train_y: np.ndarray,
    train_weight: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
) -> tuple[QATTinyMLP, list[dict[str, Any]], dict[str, Any]]:
    set_seed(1)
    model = QATTinyMLP(total_bits, integer_bits)
    load_champion_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tx = torch.from_numpy(train_x.astype(np.float32, copy=False))
    ty = torch.from_numpy(train_y.astype(np.int64, copy=False))
    tw = torch.from_numpy(train_weight.astype(np.float32, copy=False))
    vx = torch.from_numpy(val_x.astype(np.float32, copy=False))
    vy = torch.from_numpy(val_y.astype(np.int64, copy=False))
    generator = torch.Generator().manual_seed(1)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, QAT_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(tx), generator=generator)
        for begin in range(0, len(tx), BATCH_SIZE):
            idx = permutation[begin: begin + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            losses = torch.nn.functional.cross_entropy(model(tx[idx]), ty[idx], reduction="none")
            loss = torch.sum(losses * tw[idx]) / torch.sum(tw[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.cross_entropy(model(vx), vy))
        history.append({
            "candidate_id": f"qat_h8_q{total_bits}_{integer_bits}",
            "epoch": epoch,
            "val_loss": val_loss,
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= QAT_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("QAT did not produce a checkpoint")
    model.load_state_dict(best_state)
    metadata = {
        "candidate_id": f"qat_h8_q{total_bits}_{integer_bits}",
        "total_bits": total_bits,
        "integer_bits": integer_bits,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_loss": best_loss,
        "wall_seconds": time.perf_counter() - started,
    }
    torch.save({"state_dict": best_state, "metadata": metadata}, MODELS / f"{metadata['candidate_id']}.pt")
    return model, history, metadata


@dataclass
class LUTPolicy:
    feature_indices: np.ndarray
    feature_names: list[str]
    edges: list[np.ndarray]
    shape: tuple[int, ...]
    values: np.ndarray
    fallback: np.ndarray
    target_kind: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        index = np.zeros(len(x), dtype=np.int64)
        stride = 1
        for source, edges, bins in zip(reversed(self.feature_indices), reversed(self.edges), reversed(self.shape)):
            digit = np.searchsorted(edges, x[:, source], side="right")
            digit = np.clip(digit, 0, bins - 1)
            index += stride * digit
            stride *= bins
        result = self.values[index]
        missing = ~np.isfinite(result).all(axis=1)
        if missing.any():
            result = result.copy()
            result[missing] = self.fallback
        return result


def fit_lut(
    x: np.ndarray, scores: np.ndarray, actions: np.ndarray, weights: np.ndarray,
    feature_indices: np.ndarray, bins: int, target_kind: str,
) -> LUTPolicy:
    edges = []
    digits = []
    for feature in feature_indices:
        edge = np.unique(np.quantile(x[:, feature], np.linspace(0, 1, bins + 1)[1:-1]))
        # Keep a rectangular hardware address even if tied quantiles collapse.
        if len(edge) < bins - 1:
            lo, hi = np.nanmin(x[:, feature]), np.nanmax(x[:, feature])
            edge = np.linspace(lo, hi, bins + 1)[1:-1]
        edges.append(edge)
        digits.append(np.clip(np.searchsorted(edge, x[:, feature], side="right"), 0, bins - 1))
    flat = np.ravel_multi_index(tuple(digits), (bins,) * len(feature_indices))
    cell_count = bins ** len(feature_indices)
    if target_kind == "score_mean":
        values = np.full((cell_count, 2), np.nan, dtype=np.float64)
        fallback = np.average(scores, axis=0, weights=weights)
        for cell in np.unique(flat):
            mask = flat == cell
            values[cell] = np.average(scores[mask], axis=0, weights=weights[mask])
    else:
        values = np.full((cell_count, 3), np.nan, dtype=np.float64)
        fallback = np.bincount(actions, weights=weights, minlength=3).astype(np.float64)
        for cell in np.unique(flat):
            mask = flat == cell
            values[cell] = np.bincount(actions[mask], weights=weights[mask], minlength=3)
    return LUTPolicy(feature_indices, [], edges, (bins,) * len(feature_indices), values, fallback, target_kind)


def lut_decision(model: LUTPolicy, x: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    output = model.predict(x)
    if model.target_kind == "score_mean":
        return output, p05a.policy_from_scores(output, threshold).astype(np.int8)
    return output, np.argmax(output, axis=1).astype(np.int8)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    selected = json.loads((PHASE07G / "results" / "tiny_mlp_selected.json").read_text(encoding="utf-8"))
    threshold = float(selected["threshold"])
    mean = np.asarray(selected["feature_mean"], dtype=np.float64)
    scale = np.asarray(selected["feature_scale"], dtype=np.float64)
    feature_names = list(selected["feature_names"])
    norms = p05a.compute_fill_norms_from_train_sequences()

    train_payload = helpers.load_payload("train")
    val_payload = helpers.load_payload("val")
    train_raw, _ = helpers.base_features(train_payload)
    val_raw, _ = helpers.base_features(val_payload)
    train_x = ((train_raw - mean) / scale).astype(np.float32)
    val_x = ((val_raw - mean) / scale).astype(np.float32)
    train_scores = helpers.teacher_scores("train", train_payload)
    val_scores = helpers.teacher_scores("val", val_payload)
    train_actions = p05a.policy_from_scores(train_scores, threshold).astype(np.int64)
    val_actions = p05a.policy_from_scores(val_scores, threshold).astype(np.int64)
    weights, boundary_scale = helpers.boundary_weights(train_scores, threshold)

    histories = []
    training_rows = []
    qat_models: dict[str, QATTinyMLP] = {}
    for total_bits, integer_bits in QAT_FORMATS:
        model, history, metadata = train_qat(
            total_bits, integer_bits, train_x, train_actions, weights, val_x, val_actions
        )
        qat_models[metadata["candidate_id"]] = model
        histories.extend(history)
        training_rows.append(metadata)
        print(f"[QAT] {metadata}", flush=True)
    pd.DataFrame(histories).to_csv(RESULTS / "qat_training_curves.csv", index=False)
    pd.DataFrame(training_rows).to_csv(RESULTS / "qat_training_summary.csv", index=False)

    finite = np.isfinite(train_x).all(axis=1)
    scores_f, _ = f_classif(train_x[finite], train_actions[finite])
    ranking = np.argsort(np.nan_to_num(scores_f, nan=-np.inf))[::-1]
    pd.DataFrame({
        "rank": np.arange(1, len(ranking) + 1),
        "feature_index": ranking,
        "feature_name": [feature_names[index] for index in ranking],
        "train_f_score": scores_f[ranking],
    }).to_csv(RESULTS / "lut_train_feature_ranking.csv", index=False)

    lut_models: dict[str, LUTPolicy] = {}
    for feature_count, bins_values in ((2, (8, 16)), (3, (4, 8)), (4, (4,))):
        indices = np.sort(ranking[:feature_count])
        for bins in bins_values:
            for target_kind in ("score_mean", "action_majority"):
                candidate_id = f"lut_{target_kind}_f{feature_count}_b{bins}"
                model = fit_lut(
                    train_x, train_scores, train_actions, weights, indices, bins, target_kind
                )
                model.feature_names = [feature_names[index] for index in indices]
                lut_models[candidate_id] = model

    rows = []
    decision_cache: dict[tuple[str, str], np.ndarray] = {}
    reference_decisions: dict[tuple[str, str], np.ndarray] = {}
    for split in SPLITS:
        payload = val_payload if split == "val" else helpers.load_payload(split)
        raw, _ = helpers.base_features(payload)
        x = ((raw - mean) / scale).astype(np.float32)
        teacher = helpers.teacher_scores(split, payload)
        teacher_decision = p05a.policy_from_scores(teacher, threshold).astype(np.int8)
        reference_decisions[(split, "phc_eventwise_reference")] = teacher_decision
        ref = correction.evaluate_policy(split, "phc_eventwise_reference", payload, teacher_decision, norms)
        ref.update({"family": "reference", "candidate_id": "phc_eventwise_reference"})
        rows.append(ref)
        for candidate_id, model in qat_models.items():
            output, audit = qat_integer_predict(model, x)
            decision = np.argmax(output, axis=1).astype(np.int8)
            row = correction.evaluate_policy(split, candidate_id, payload, decision, norms)
            row.update({
                "family": "qat_mlp", "candidate_id": candidate_id,
                "total_bits": model.total_bits, "integer_bits": model.integer_bits,
                "cell_count": np.nan, "feature_count": 16,
                "feature_names": "|".join(feature_names),
                "action_agreement_vs_phc": float(np.mean(decision == teacher_decision)),
                "action_flip_count_vs_phc": int(np.count_nonzero(decision != teacher_decision)),
                "saturation_count": int(sum(audit.values())),
            })
            rows.append(row)
            decision_cache[(split, candidate_id)] = decision
        for candidate_id, model in lut_models.items():
            output, decision = lut_decision(model, x, threshold)
            row = correction.evaluate_policy(split, candidate_id, payload, decision, norms)
            row.update({
                "family": "lut_policy", "candidate_id": candidate_id,
                "total_bits": np.nan, "integer_bits": np.nan,
                "cell_count": int(np.prod(model.shape)),
                "feature_count": len(model.feature_indices),
                "feature_names": "|".join(model.feature_names),
                "action_agreement_vs_phc": float(np.mean(decision == teacher_decision)),
                "action_flip_count_vs_phc": int(np.count_nonzero(decision != teacher_decision)),
                "saturation_count": 0,
            })
            rows.append(row)
            decision_cache[(split, candidate_id)] = decision

    metrics = pd.DataFrame(rows)
    phc = metrics[metrics["family"] == "reference"][["split", "corrected_sequential_excess_pnl"]].rename(
        columns={"corrected_sequential_excess_pnl": "phc_corrected_pnl"}
    )
    metrics = metrics.merge(phc, on="split", how="left")
    metrics["pnl_retention_vs_phc_pct"] = 100.0 * metrics["corrected_sequential_excess_pnl"] / metrics["phc_corrected_pnl"]
    metrics.to_csv(RESULTS / "ultrafast_policy_sweep.csv", index=False)

    candidate_metrics = metrics[metrics["family"] != "reference"]
    val = candidate_metrics[candidate_metrics["split"] == "val"].sort_values(
        ["corrected_sequential_excess_pnl", "family", "candidate_id"], ascending=[False, True, True]
    )
    champions = val.groupby("family", as_index=False).head(1)
    champion_ids = set(champions["candidate_id"])
    selected_rows = candidate_metrics[candidate_metrics["candidate_id"].isin(champion_ids)].copy()
    selected_rows["selection_split"] = "val"
    selected_rows["new3_used_for_selection"] = False
    selected_rows["new4_used_for_selection"] = False
    selected_rows["new5_used"] = False
    selected_rows.to_csv(RESULTS / "selected_family_champions.csv", index=False)

    manifest = {
        "status": "complete",
        "threshold": threshold,
        "boundary_weight_scale": boundary_scale,
        "feature_selection": "ANOVA F score against frozen PHC actions on train only",
        "candidate_selection": "maximum corrected sequential excess PnL on val within each family",
        "champions": champions[["family", "candidate_id", "corrected_sequential_excess_pnl"]].to_dict("records"),
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used": False,
    }
    (RESULTS / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
