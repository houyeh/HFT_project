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
ROOT = PROJECT_ROOT / "n2ew_era" / "05d_oracle_gap_reduction"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"
PHASE05C2 = PROJECT_ROOT / "n2ew_era" / "05c2_phc_stp_dynamic_synaptic_gain"

if str(PHASE05A) not in sys.path:
    sys.path.insert(0, str(PHASE05A))
if str(PHASE05C2) not in sys.path:
    sys.path.insert(0, str(PHASE05C2))

import common_05a as p05a  # noqa: E402
import common_05c2 as c05c2  # noqa: E402
from models.phc_lite_stp import PHCLiteSTP  # noqa: E402


SPLITS = ["val", "new3", "new4"]
MODEL_ORDER = [
    "side_by_queue_imbalance_abs_0p1",
    "ml_hgb_regressor_shallow_convergence_selected",
    "phc_lite_v1",
    "phc_stp_extended80",
]


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(p05a.json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def session_ids(payload: dict[str, Any]) -> np.ndarray:
    labels = np.asarray(payload["session_labels"], dtype=object)
    codes = payload["session_code"].numpy().astype(int)
    return labels[codes]


def oracle_decision_and_pnl(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bid = payload["y_bid"].numpy().astype(np.float64)
    ask = payload["y_ask"].numpy().astype(np.float64)
    bid = np.nan_to_num(bid, nan=0.0, posinf=0.0, neginf=0.0)
    ask = np.nan_to_num(ask, nan=0.0, posinf=0.0, neginf=0.0)
    decision = np.zeros(len(bid), dtype=np.int8)
    bid_best = (bid >= ask) & (bid > 0)
    ask_best = (ask > bid) & (ask > 0)
    decision[bid_best] = p05a.DECISION_BID
    decision[ask_best] = p05a.DECISION_ASK
    pnl = np.zeros(len(bid), dtype=np.float64)
    pnl[bid_best] = bid[bid_best]
    pnl[ask_best] = ask[ask_best]
    oracle_value = np.maximum(np.maximum(bid, ask), 0.0)
    return decision, pnl, oracle_value


def phc_v1_decision_scores(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    model, ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    scores = p05a.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, float(policy["threshold"])), scores


def stp_extended80_decision_scores(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    policy_path = PHASE05C2 / "results" / "phc_stp_extended_selected_policy.json"
    if not policy_path.exists():
        return np.zeros(len(payload["y_bid"]), dtype=np.int8), np.zeros((len(payload["y_bid"]), 2), dtype=np.float32)
    policy = read_json(policy_path)
    ckpt_path = PHASE05C2 / "models" / f"phc_stp_extended_seed{int(policy['seed'])}_best_val.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PHCLiteSTP(
        int(ckpt["feature_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        mtl_steps=int(ckpt["mtl_steps"]),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scores = c05c2.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    return p05a.policy_from_scores(scores, float(policy["threshold"])), scores


def model_decisions(payload: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    q = p05a.queue_rule_decision(payload)
    hgb_decision, hgb_scores = p05a.hgb_decision(payload)
    phc_decision, phc_scores = phc_v1_decision_scores(payload)
    stp_decision, stp_scores = stp_extended80_decision_scores(payload)
    return {
        "side_by_queue_imbalance_abs_0p1": (q, np.full((len(q), 2), np.nan, dtype=np.float32)),
        "ml_hgb_regressor_shallow_convergence_selected": (hgb_decision, hgb_scores),
        "phc_lite_v1": (phc_decision, phc_scores),
        "phc_stp_extended80": (stp_decision, stp_scores),
    }


def error_type(model_action: np.ndarray, oracle_action: np.ndarray) -> np.ndarray:
    out = np.empty(len(model_action), dtype=object)
    model_selected = model_action != p05a.DECISION_NO_QUOTE
    oracle_selected = oracle_action != p05a.DECISION_NO_QUOTE
    out[(model_action == oracle_action) & oracle_selected] = "correct_selected"
    out[(model_action == p05a.DECISION_NO_QUOTE) & oracle_selected] = "missed_opportunity"
    out[model_selected & (oracle_action == p05a.DECISION_NO_QUOTE)] = "false_positive"
    out[((model_action == p05a.DECISION_BID) & (oracle_action == p05a.DECISION_ASK)) | ((model_action == p05a.DECISION_ASK) & (oracle_action == p05a.DECISION_BID))] = "side_error"
    out[(model_action == p05a.DECISION_NO_QUOTE) & (oracle_action == p05a.DECISION_NO_QUOTE)] = "correct_no_quote"
    return out


def selected_scores(scores: np.ndarray, decision: np.ndarray) -> np.ndarray:
    out = np.full(len(decision), np.nan, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        return out
    bid = decision == p05a.DECISION_BID
    ask = decision == p05a.DECISION_ASK
    out[bid] = scores[bid, 0]
    out[ask] = scores[ask, 1]
    return out


def model_pnl(payload: dict[str, Any], decision: np.ndarray) -> np.ndarray:
    return p05a.decision_pnl_from_payload(payload, decision)


def action_name(values: np.ndarray) -> np.ndarray:
    mapping = {
        p05a.DECISION_NO_QUOTE: "no_quote",
        p05a.DECISION_BID: "bid",
        p05a.DECISION_ASK: "ask",
    }
    return np.array([mapping.get(int(v), "unknown") for v in values], dtype=object)


def raw_feature(payload: dict[str, Any], name: str) -> np.ndarray:
    raw = payload["target_raw"].numpy()
    idx = p05a.TARGET_RAW_COLUMNS.index(name)
    return raw[:, idx].astype(np.float64)


def quantile_bucket(values: np.ndarray, prefix: str, q: int = 5) -> np.ndarray:
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    try:
        out = pd.qcut(s.rank(method="first"), q=q, labels=[f"{prefix}_q{i}" for i in range(1, q + 1)])
        return out.astype(str).to_numpy()
    except ValueError:
        return np.array([f"{prefix}_single"] * len(values), dtype=object)


def spread_bucket(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bins = [-np.inf, 0, 1, 2, 3, 5, np.inf]
    labels = ["spread_le0", "spread_1", "spread_2", "spread_3", "spread_4_5", "spread_gt5"]
    return pd.cut(s, bins=bins, labels=labels).astype(str).to_numpy()


def time_in_session_bucket(payload: dict[str, Any], q: int = 5) -> np.ndarray:
    event = payload["event_index"].numpy().astype(np.float64)
    sessions = payload["session_code"].numpy()
    out = np.empty(len(event), dtype=object)
    for code in np.unique(sessions):
        mask = sessions == code
        local = event[mask]
        if len(local) <= 1:
            out[mask] = "time_single"
            continue
        denom = max(float(local.max() - local.min()), 1.0)
        pos = (local - local.min()) / denom
        out[mask] = quantile_bucket(pos, "time", q=q)
    return out


def session_pnl(payload: dict[str, Any], decision: np.ndarray) -> pd.DataFrame:
    pnl = model_pnl(payload, decision)
    sessions = session_ids(payload)
    return pd.DataFrame({"session_id": sessions, "pnl": pnl, "selected": decision != p05a.DECISION_NO_QUOTE}).groupby("session_id").agg(
        session_pnl=("pnl", "sum"),
        selected_count=("selected", "sum"),
        state_count=("selected", "size"),
    ).reset_index()


def causal_fill_for_decision(split: str, policy_id: str, payload: dict[str, Any], decision: np.ndarray, salt: str) -> dict[str, Any]:
    norms = p05a.compute_fill_norms_from_train_sequences()
    probs = p05a.fill_probabilities(payload, decision, norms)
    fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, salt) < probs)
    pnl = model_pnl(payload, decision)
    total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(payload, decision, fills, pnl, inventory_cap=1)
    return {
        "split": split,
        "policy_id": policy_id,
        "sequential_excess_pnl": float(total),
        "fill_count": int(fill_count),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_inv),
        "max_drawdown": float(max_dd),
        "win_rate": win / max(fill_count, 1) if fill_count else 0.0,
        "loss_rate": loss / max(fill_count, 1) if fill_count else 0.0,
        "fill_uses_future_pnl": False,
        "new5_used": False,
    }


def oracle_capture_for_decision(payload: dict[str, Any], decision: np.ndarray) -> dict[str, float]:
    oracle_action, _, _ = oracle_decision_and_pnl(payload)
    bid = payload["y_bid"].numpy().astype(np.float64)
    ask = payload["y_ask"].numpy().astype(np.float64)
    best = np.maximum(bid, ask)
    side = np.where(bid >= ask, p05a.DECISION_BID, p05a.DECISION_ASK)
    count = int((decision != p05a.DECISION_NO_QUOTE).sum())
    order = np.argsort(best)[::-1]
    chosen = order[:count]
    oracle_decision = np.zeros(len(best), dtype=np.int8)
    oracle_decision[chosen] = side[chosen].astype(np.int8)
    policy_pnl = float(model_pnl(payload, decision).sum())
    oracle_pnl = float(model_pnl(payload, oracle_decision).sum())
    return {
        "policy_selected_count": count,
        "policy_pnl": policy_pnl,
        "matched_count_oracle_pnl": oracle_pnl,
        "capture_ratio": policy_pnl / oracle_pnl if oracle_pnl else np.nan,
        "opportunity_gap": oracle_pnl - policy_pnl,
        "oracle_uses_future_labels": True,
    }
