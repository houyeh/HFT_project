from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "05d_oracle_gap_reduction"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

PHASE04B1 = PROJECT_ROOT / "n2ew_era" / "04b1_ml_completion_patch"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"

for path in [PHASE05A]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common_05a as p05a  # noqa: E402

VAL_SPLIT = "val"
HGB_POLICY_ID = "ml_hgb_regressor_shallow_convergence_selected"
PHC_POLICY_ID = "phc_lite_v1"
QUEUE_POLICY_ID = "side_by_queue_imbalance_abs_0p1"
ALLOWED_EVAL_SPLITS = {VAL_SPLIT}


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, FIGURES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def assert_val_split(split: str) -> None:
    if split not in ALLOWED_EVAL_SPLITS:
        raise RuntimeError(f"Matched-fill diagnostics are val-only; refused split={split!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(p05a.json_sanitize(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_val_payload() -> dict[str, Any]:
    return p05a.load_sequences(VAL_SPLIT)


def finite_targets(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bid = payload["y_bid"].numpy().astype(np.float64)
    ask = payload["y_ask"].numpy().astype(np.float64)
    bid = np.nan_to_num(bid, nan=0.0, posinf=0.0, neginf=0.0)
    ask = np.nan_to_num(ask, nan=0.0, posinf=0.0, neginf=0.0)
    return bid, ask


def greedy_count_side_oracle(payload: dict[str, Any], bid_count: int, ask_count: int, positive_only: bool = False) -> np.ndarray:
    bid, ask = finite_targets(payload)
    candidates: list[tuple[float, int, int]] = []
    for idx, value in enumerate(bid):
        if (not positive_only) or value > 0.0:
            candidates.append((float(value), idx, p05a.DECISION_BID))
    for idx, value in enumerate(ask):
        if (not positive_only) or value > 0.0:
            candidates.append((float(value), idx, p05a.DECISION_ASK))
    candidates.sort(reverse=True, key=lambda item: item[0])
    remaining = {p05a.DECISION_BID: int(bid_count), p05a.DECISION_ASK: int(ask_count)}
    used = np.zeros(len(bid), dtype=bool)
    decision = np.zeros(len(bid), dtype=np.int8)
    for _, idx, side in candidates:
        if used[idx] or remaining[side] <= 0:
            continue
        decision[idx] = side
        used[idx] = True
        remaining[side] -= 1
        if remaining[p05a.DECISION_BID] <= 0 and remaining[p05a.DECISION_ASK] <= 0:
            break
    return decision


def action_names(decision: np.ndarray) -> np.ndarray:
    out = np.full(len(decision), "no_quote", dtype=object)
    out[decision == p05a.DECISION_BID] = "bid"
    out[decision == p05a.DECISION_ASK] = "ask"
    return out


def accepted_fill_trace(
    payload: dict[str, Any],
    decision: np.ndarray,
    fills: np.ndarray,
    pnl: np.ndarray,
    inventory_cap: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(decision)
    accepted = np.zeros(n, dtype=bool)
    close_pnl = np.zeros(n, dtype=np.float64)
    close_side = np.zeros(n, dtype=np.int64)
    equity = np.zeros(n, dtype=np.float64)
    session_code = payload["session_code"].numpy()
    event_index = payload["event_index"].numpy()
    session_positions = {code: np.where(session_code == code)[0] for code in np.unique(session_code)}
    close_lookup: dict[tuple[int, int], int] = {}
    for code, positions in session_positions.items():
        events = event_index[positions]
        for pos in positions:
            target_event = event_index[pos] + p05a.HORIZON
            j = np.searchsorted(events, target_event, side="left")
            close_lookup[(int(code), int(pos))] = int(positions[min(j, len(positions) - 1)])

    inventory = 0
    total_pnl = 0.0
    max_abs_inventory = 0
    forced = 0
    fill_count = 0
    win = 0
    loss = 0
    peak = 0.0
    max_dd = 0.0
    for t in range(n):
        if close_pnl[t] != 0.0 or close_side[t] != 0:
            inventory -= int(close_side[t])
            total_pnl += float(close_pnl[t])
        if fills[t]:
            side_sign = 1 if decision[t] == p05a.DECISION_BID else -1
            if abs(inventory + side_sign) > inventory_cap:
                forced += 1
            else:
                trade_pnl = float(pnl[t])
                accepted[t] = True
                inventory += side_sign
                max_abs_inventory = max(max_abs_inventory, abs(inventory))
                close_idx = close_lookup[(int(session_code[t]), int(t))]
                close_side[close_idx] += side_sign
                close_pnl[close_idx] += trade_pnl
                fill_count += 1
                win += int(trade_pnl > 0.0)
                loss += int(trade_pnl < 0.0)
        equity[t] = total_pnl
        peak = max(peak, total_pnl)
        max_dd = max(max_dd, peak - total_pnl)

    summary = {
        "horizon": p05a.HORIZON,
        "fill_model": "conservative_l2_stress",
        "inventory_cap": int(inventory_cap),
        "sequential_excess_pnl": float(total_pnl),
        "fill_count": int(fill_count),
        "fill_attempt_count": int(np.asarray(fills, dtype=bool).sum()),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_abs_inventory),
        "max_drawdown": float(max_dd),
        "win_rate": float(win / max(fill_count, 1)) if fill_count else 0.0,
        "loss_rate": float(loss / max(fill_count, 1)) if fill_count else 0.0,
        "avg_trade_excess_pnl": float(total_pnl / max(fill_count, 1)) if fill_count else 0.0,
        "fill_uses_future_pnl": False,
    }
    return accepted, summary


def causal_fill_for_decision(payload: dict[str, Any], decision: np.ndarray, salt: str) -> tuple[np.ndarray, dict[str, Any]]:
    norms = p05a.compute_fill_norms_from_train_sequences()
    probs = p05a.fill_probabilities(payload, decision, norms)
    fills = (decision != p05a.DECISION_NO_QUOTE) & (p05a.stable_uniform(payload, decision, salt) < probs)
    pnl = p05a.decision_pnl_from_payload(payload, decision)
    return accepted_fill_trace(payload, decision.astype(np.int8), fills.astype(bool), pnl.astype(np.float64), inventory_cap=1)


def phc_decision_scores(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    policy = p05a.read_json(PHASE05A / "results" / "phc_lite_selected_policy.json")
    model, ckpt = p05a.load_model(PHASE05A / "models" / "phc_lite_best_val.pt")
    scores = p05a.predict_scores(model, payload, target_scale=float(ckpt["target_scale"]))
    decision = p05a.policy_from_scores(scores, float(policy["threshold"]))
    return decision, scores


def hgb_decision_scores(payload: dict[str, Any], model: Any | None = None, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    if model is None:
        model = joblib.load(PHASE04B1 / "models" / "hgb_regressor_iter_120.joblib")
    if threshold is None:
        closure = read_json(PHASE04B1 / "results" / "txf_hgb_convergence_closure.json")
        threshold = float(closure["selected_threshold_on_validation"])
    bid = model.predict(p05a.hgb_feature_matrix(payload, "bid").to_numpy(dtype=np.float32)).astype(np.float32)
    ask = model.predict(p05a.hgb_feature_matrix(payload, "ask").to_numpy(dtype=np.float32)).astype(np.float32)
    scores = np.column_stack([bid, ask])
    return p05a.policy_from_scores(scores, float(threshold)), scores


def policy_decision_scores(policy_id: str, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if policy_id == HGB_POLICY_ID:
        return hgb_decision_scores(payload)
    if policy_id == PHC_POLICY_ID:
        return phc_decision_scores(payload)
    raise ValueError(policy_id)


def matched_count_side_oracle_for_policy(payload: dict[str, Any], policy_decision: np.ndarray) -> np.ndarray:
    return greedy_count_side_oracle(
        payload,
        bid_count=int((policy_decision == p05a.DECISION_BID).sum()),
        ask_count=int((policy_decision == p05a.DECISION_ASK).sum()),
        positive_only=False,
    )


def hgb_visible_state_features(payload: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    bid = p05a.hgb_feature_matrix(payload, "bid").to_numpy(dtype=np.float32)
    ask = p05a.hgb_feature_matrix(payload, "ask").to_numpy(dtype=np.float32)
    names = [f"bid__{name}" for name in p05a.HGB_ML_FEATURES] + [f"ask__{name}" for name in p05a.HGB_ML_FEATURES]
    return np.concatenate([bid, ask], axis=1).astype(np.float32), names


def phc_visible_state_features(payload: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    x = payload["x"].numpy().astype(np.float32)
    feature_names = list(payload.get("feature_names", p05a.PHC_FEATURES))
    names = [f"t_minus_{x.shape[1] - 1 - t:03d}__{name}" for t in range(x.shape[1]) for name in feature_names]
    return x.reshape(x.shape[0], -1).astype(np.float32), names


def visible_features_for_policy(policy_id: str, payload: dict[str, Any]) -> tuple[np.ndarray, list[str], str]:
    if policy_id == HGB_POLICY_ID:
        x, names = hgb_visible_state_features(payload)
        return x, names, "hgb_bid_and_ask_action_feature_vectors_concatenated"
    if policy_id == PHC_POLICY_ID:
        x, names = phc_visible_state_features(payload)
        return x, names, "phc_sequence_window_x_flattened"
    raise ValueError(policy_id)


def threshold_grid_from_scores(scores: np.ndarray) -> list[float]:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return [0.0]
    values = [-50.0, -15.0, 0.0, 15.0]
    values.extend(np.quantile(finite, [0.5, 0.75, 0.90, 0.95]).tolist())
    return sorted({float(v) for v in values})


def load_hgb_capacity_model(max_iter: int) -> Any:
    path = PHASE04B1 / "models" / f"hgb_regressor_iter_{int(max_iter)}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def load_04b1_validation_thresholds() -> dict[int, dict[str, Any]]:
    path = PHASE04B1 / "results" / "txf_hgb_iteration_sweep.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_csv(path)
    rows = rows[(rows["row_type"].eq("validation_sweep")) & (rows["split"].eq(VAL_SPLIT))].copy()
    out: dict[int, dict[str, Any]] = {}
    for _, row in rows.iterrows():
        max_iter = int(row["max_iter"])
        out[max_iter] = {
            "threshold_selected_on_val": float(row["selected_threshold_on_val"]),
            "validation_policy_pnl_04b1": float(row["validation_policy_pnl"]),
            "validation_selected_action_count_04b1": int(float(row["validation_selected_action_count"])),
            "hgb_n_iter_04b1": int(float(row["hgb_n_iter"])),
            "threshold_source_path": str(path),
            "threshold_source_row_type": "validation_sweep",
            "threshold_source_split": VAL_SPLIT,
        }
    return out


def select_threshold_on_sequence_val(payload: dict[str, Any], scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in threshold_grid_from_scores(scores):
        decision = p05a.policy_from_scores(scores, float(threshold))
        metric = p05a.metric_from_decision(VAL_SPLIT, "hgb_capacity_scan", payload, decision)
        metric["threshold"] = float(threshold)
        rows.append(metric)
    grid = pd.DataFrame(rows).sort_values(["total_excess_pnl_H20", "selected_action_count"], ascending=[False, True]).reset_index(drop=True)
    return float(grid.iloc[0]["threshold"]), grid


def hgb_structural_work(model: Any) -> dict[str, Any]:
    tree_count = 0
    leaf_depths: list[float] = []
    split_node_count = 0
    leaf_node_count = 0
    for predictors_at_iter in getattr(model, "_predictors", []):
        for predictor in predictors_at_iter:
            nodes = getattr(predictor, "nodes", None)
            if nodes is None or "is_leaf" not in nodes.dtype.names:
                continue
            tree_count += 1
            is_leaf = nodes["is_leaf"].astype(bool)
            leaf_node_count += int(is_leaf.sum())
            split_node_count += int((~is_leaf).sum())
            if "depth" in nodes.dtype.names:
                leaf_depths.extend(nodes["depth"][is_leaf].astype(float).tolist())
    avg_leaf_depth = float(np.mean(leaf_depths)) if leaf_depths else float("nan")
    max_leaf_depth = float(np.max(leaf_depths)) if leaf_depths else float("nan")
    # Structural-work proxy: one unit per split comparison on the traversed path
    # plus one final leaf-value lookup, multiplied by the number of boosted trees.
    structural_work_units = float(tree_count * (avg_leaf_depth + 1.0)) if np.isfinite(avg_leaf_depth) else float("nan")
    return {
        "tree_count": int(tree_count),
        "n_iter_actual": int(getattr(model, "n_iter_", tree_count)),
        "max_iter_requested": int(getattr(model, "max_iter", tree_count)),
        "max_leaf_nodes": int(getattr(model, "max_leaf_nodes", -1)),
        "split_node_count": int(split_node_count),
        "leaf_node_count": int(leaf_node_count),
        "avg_leaf_depth": avg_leaf_depth,
        "max_leaf_depth": max_leaf_depth,
        "structural_work_units": structural_work_units,
        "structural_work_definition": "tree_count * (mean_leaf_depth + 1 leaf lookup)",
    }


def append_artifact_paths(lines: list[str]) -> list[str]:
    artifact_rows = []
    for label, path in [
        ("diagnostic_a_summary", RESULTS / "matched_fill_separability_summary.csv"),
        ("diagnostic_a_curves", RESULTS / "matched_fill_separability_curves.csv"),
        ("diagnostic_a_figure", FIGURES / "matched_fill_separability_roc_pr.png"),
        ("diagnostic_b_summary", RESULTS / "hgb_capacity_matched_fill_capture.csv"),
        ("diagnostic_b_thresholds", RESULTS / "hgb_capacity_threshold_grids.csv"),
        ("diagnostic_b_figure", FIGURES / "hgb_capacity_vs_capture_ratio.png"),
    ]:
        artifact_rows.append({"artifact": label, "path": str(path), "exists": path.exists()})
    lines.extend(["", "## Artifact Paths", "", pd.DataFrame(artifact_rows).to_markdown(index=False)])
    return lines


def write_combined_report() -> None:
    lines = [
        "# Matched-Fill Separability and HGB Capacity Diagnostics",
        "",
        "Scope: `val` split only. No NEW3, NEW4, or NEW5 data are loaded by these diagnostics.",
        "Oracle rows use future H20 labels and are diagnostic only. Fill proxy is `conservative_l2_stress`, H20 close, `inventory_cap=1`.",
        "Execution note: HGB joblib loads emitted sklearn `InconsistentVersionWarning` because the stored estimators were created with sklearn 1.7.2 and this run used sklearn 1.5.2.",
        "",
    ]
    a_path = RESULTS / "matched_fill_separability_summary.csv"
    b_path = RESULTS / "hgb_capacity_matched_fill_capture.csv"
    if a_path.exists():
        a = pd.read_csv(a_path)
        a = a.fillna("")
        lines.extend(["## Diagnostic A Numbers", "", a.to_markdown(index=False), ""])
    else:
        lines.extend(["## Diagnostic A Numbers", "", "`results/matched_fill_separability_summary.csv` not present.", ""])
    if b_path.exists():
        b = pd.read_csv(b_path)
        display_cols = [
            "split",
            "capacity_id",
            "max_iter",
            "tree_count",
            "avg_leaf_depth",
            "structural_work_units",
            "threshold_selected_on_val",
            "policy_selected_count",
            "policy_sequential_excess_pnl",
            "oracle_sequential_excess_pnl",
            "matched_fill_capture_ratio",
            "policy_fill_count",
            "oracle_fill_count",
        ]
        keep = [col for col in display_cols if col in b.columns]
        lines.extend(["## Diagnostic B Numbers", "", b[keep].to_markdown(index=False), ""])
    else:
        lines.extend(["## Diagnostic B Numbers", "", "`results/hgb_capacity_matched_fill_capture.csv` not present.", ""])
    append_artifact_paths(lines)
    write_markdown(ROOT / "MATCHED_FILL_SEPARABILITY_CAPACITY_DIAGNOSTICS_REPORT.md", lines)
