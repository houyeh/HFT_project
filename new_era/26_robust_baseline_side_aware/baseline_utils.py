from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "new_era" / "26_robust_baseline_side_aware"
RESULTS_DIR = ROOT / "results"

V16_DIR = PROJECT_ROOT / "new_era" / "16_binary_label_v2"
V18_DIR = PROJECT_ROOT / "new_era" / "18_queue_aware"
V23_DIR = PROJECT_ROOT / "new_era" / "23_root_cause_stress_profitability"
for path in [V16_DIR, V18_DIR, V23_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from queue_stress_core import MAKER_FEE, POINT_VALUE, build_action_table  # noqa: E402
from round2_audit_utils import add_market_buckets  # noqa: E402
from root_cause_utils import alignment_metrics, eval_quote_selection, load_context, write_markdown  # noqa: E402


SPREAD_BUCKETS = ["<=1", "(1,2]", "(2,3]", ">3"]
SPREAD_THRESHOLDS = [1.0, 2.0, 3.0]


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_label_frame() -> pd.DataFrame:
    labels = pd.read_parquet(V16_DIR / "outputs" / "binary_labels_v2.parquet").reset_index(drop=True)
    labels["event_id"] = np.arange(len(labels), dtype=np.int64)
    return add_market_buckets(labels.sort_values(["session_id", "local_row_id"]).reset_index(drop=True))


def stable_u(event_id: int, side: str, salt: str) -> float:
    digest = hashlib.blake2b(f"026|{salt}|{int(event_id)}|{side}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64 - 1)


def actions_from_frame(frame: pd.DataFrame, *, salt: str) -> pd.DataFrame:
    frame = add_market_buckets(frame.sort_values(["session_id", "local_row_id"]).reset_index(drop=True))
    actions = build_action_table(frame)
    if actions.empty:
        return actions
    idx = actions["row_index"].to_numpy(dtype=np.int64)
    side = actions["side"].astype(str).to_numpy()
    actions["event_id"] = frame.loc[idx, "event_id"].to_numpy(dtype=np.int64)
    actions["spread_ticks"] = frame.loc[idx, "spread_ticks"].to_numpy(dtype=float)
    actions["spread_bucket"] = frame.loc[idx, "spread_bucket"].astype(str).to_numpy()
    actions["side_toxic"] = np.where(side == "bid", frame.loc[idx, "bid_toxic"].to_numpy(dtype=bool), frame.loc[idx, "ask_toxic"].to_numpy(dtype=bool))
    if "bid_prob_toxic" in frame.columns and "ask_prob_toxic" in frame.columns:
        actions["side_score_016"] = np.where(side == "bid", frame.loc[idx, "bid_prob_toxic"].to_numpy(dtype=float), frame.loc[idx, "ask_prob_toxic"].to_numpy(dtype=float))
    final_mid = frame.groupby("session_id", sort=True)["mid_price"].last().astype(float).to_dict()
    actions["final_mid"] = actions["session_id"].map(final_mid).astype(float)
    actions["neutral_u"] = [stable_u(eid, s, f"{salt}_neutral") for eid, s in zip(actions["event_id"], side)]
    fut = actions["future_mid_return_H_ticks"].astype(float).to_numpy()
    side_markout = np.where(side == "bid", fut, -fut)
    terminal_ticks = np.where(
        side == "bid",
        actions["final_mid"].astype(float).to_numpy() - actions["price"].astype(float).to_numpy(),
        actions["price"].astype(float).to_numpy() - actions["final_mid"].astype(float).to_numpy(),
    )
    adverse = np.maximum(0.0, -side_markout)
    eligible = actions["neutral_u"].astype(float).to_numpy() <= 0.5
    value = terminal_ticks * POINT_VALUE - MAKER_FEE - 0.5 * adverse * POINT_VALUE
    actions["quote_value_moderate"] = np.where(eligible, value, 0.0)
    actions["oracle_positive_quote"] = actions["quote_value_moderate"].astype(float) > 0.0
    return actions


def val_test_actions(split: str) -> pd.DataFrame:
    _, actions = load_context(split)
    return actions.copy()


def candidate_policies() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for thr in SPREAD_THRESHOLDS:
        policies.append({"policy_id": f"spread_only_ge_{thr:g}", "policy_type": "spread_only", "min_spread_ticks": thr})
        policies.append({"policy_id": f"bid_only_spread_ge_{thr:g}", "policy_type": "bid_only_spread", "min_spread_ticks": thr})
    for bid_thr in SPREAD_THRESHOLDS:
        for ask_thr in SPREAD_THRESHOLDS:
            policies.append(
                {
                    "policy_id": f"side_specific_spread_bid{bid_thr:g}_ask{ask_thr:g}",
                    "policy_type": "side_specific_spread_threshold",
                    "bid_min_spread_ticks": bid_thr,
                    "ask_min_spread_ticks": ask_thr,
                }
            )
    for mask in range(1, 2 ** (2 * len(SPREAD_BUCKETS))):
        lut: dict[str, bool] = {}
        bits = f"{mask:0{2 * len(SPREAD_BUCKETS)}b}"
        bit_idx = 0
        for side in ["bid", "ask"]:
            for bucket in SPREAD_BUCKETS:
                lut[f"{side}|{bucket}"] = bool(int(bits[bit_idx]))
                bit_idx += 1
        policies.append({"policy_id": f"side_spread_lut_{mask:03d}", "policy_type": "side_spread_bucket_lut", "quote_lut": lut})
    return policies


def apply_non_ml_policy(actions: pd.DataFrame, policy: dict[str, Any]) -> np.ndarray:
    side = actions["side"].astype(str).to_numpy()
    spread = actions["spread_ticks"].astype(float).to_numpy()
    kind = policy["policy_type"]
    if kind == "spread_only":
        return spread >= float(policy["min_spread_ticks"])
    if kind == "bid_only_spread":
        return (side == "bid") & (spread >= float(policy["min_spread_ticks"]))
    if kind == "side_specific_spread_threshold":
        return np.where(
            side == "bid",
            spread >= float(policy["bid_min_spread_ticks"]),
            spread >= float(policy["ask_min_spread_ticks"]),
        )
    if kind == "side_spread_bucket_lut":
        buckets = actions["spread_bucket"].astype(str).to_numpy()
        lut = policy["quote_lut"]
        return np.asarray([bool(lut.get(f"{s}|{b}", False)) for s, b in zip(side, buckets)], dtype=bool)
    raise ValueError(f"Unknown non-ML policy: {kind}")


def evaluate_non_ml(actions: pd.DataFrame, policy: dict[str, Any], *, split: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = apply_non_ml_policy(actions, policy)
    row = eval_quote_selection(actions, selected, policy["policy_id"])
    out = {
        **policy,
        "split": split,
        "selection_split": "validation",
        "test_used_for_selection": False,
        **row,
    }
    if extra:
        out.update(extra)
    return out


def select_non_ml_policy(grid: pd.DataFrame) -> pd.Series:
    work = grid.copy()
    work["constraints_pass"] = (
        (work["selected_quote_count"].astype(int) >= 10)
        & (work["selection_rate"].astype(float) >= 0.03)
        & (work["selection_rate"].astype(float) <= 0.85)
    )
    eligible = work[work["constraints_pass"]].copy()
    if eligible.empty:
        eligible = work.copy()
    eligible["selection_objective"] = eligible["selected_quote_pnl"].astype(float) - 0.02 * eligible["false_positive_loss"].astype(float)
    return eligible.sort_values(["selection_objective", "selected_quote_pnl", "oracle_precision"], ascending=[False, False, False]).iloc[0]


def policy_from_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    kind = str(data["policy_type"])
    policy = {"policy_id": str(data["policy_id"]), "policy_type": kind}
    if kind in {"spread_only", "bid_only_spread"}:
        policy["min_spread_ticks"] = float(data["min_spread_ticks"])
    elif kind == "side_specific_spread_threshold":
        policy["bid_min_spread_ticks"] = float(data["bid_min_spread_ticks"])
        policy["ask_min_spread_ticks"] = float(data["ask_min_spread_ticks"])
    elif kind == "side_spread_bucket_lut":
        lut = data["quote_lut"]
        if isinstance(lut, str):
            lut = json.loads(lut.replace("'", '"'))
        policy["quote_lut"] = {str(k): bool(v) for k, v in lut.items()}
    else:
        raise ValueError(kind)
    return policy


def load_selected_non_ml_policy() -> dict[str, Any]:
    return read_json(RESULTS_DIR / "non_ml_selected_policy.json")["selected_policy"]


def matched_random_veto_pnls(
    actions: pd.DataFrame,
    baseline_selected: np.ndarray,
    veto_mask: np.ndarray,
    *,
    trials: int = 1000,
    seed: int = 20260525,
) -> np.ndarray:
    work = actions.reset_index(drop=True).copy()
    baseline_selected = np.asarray(baseline_selected, dtype=bool)
    veto_mask = np.asarray(veto_mask, dtype=bool)
    values = work["quote_value_moderate"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    groups = work.groupby(["side", "session_id", "spread_bucket"], observed=False, sort=True).groups
    pnls = []
    for _ in range(int(trials)):
        random_veto = np.zeros(len(work), dtype=bool)
        for _, idx in groups.items():
            idx_arr = np.asarray(list(idx), dtype=np.int64)
            candidates = idx_arr[baseline_selected[idx_arr]]
            k = int(veto_mask[idx_arr].sum())
            if k:
                random_veto[rng.choice(candidates, size=min(k, len(candidates)), replace=False)] = True
        pnls.append(float(values[baseline_selected & ~random_veto].sum()))
    return np.asarray(pnls, dtype=float)

