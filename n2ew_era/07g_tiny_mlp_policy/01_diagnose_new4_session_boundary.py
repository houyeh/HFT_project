from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07g_tiny_mlp_policy"
RESULTS = ROOT / "results"
MAIN = ROOT / "00_run_tiny_mlp_policy.py"
FILL_SALT = "05c3_stp_v2_conservative_l2_stress"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase = load_module("tiny_mlp_phase_07g_diagnostic", MAIN)
helpers = phase.helpers
p05a = phase.p05a


def trace_simulator(
    payload: dict[str, Any], decision: np.ndarray, fills: np.ndarray, pnl: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n = len(decision)
    session_code = payload["session_code"].numpy()
    event_index = payload["event_index"].numpy()
    positions_by_session = {
        int(code): np.where(session_code == code)[0] for code in np.unique(session_code)
    }
    close_lookup: dict[tuple[int, int], int] = {}
    for code, positions in positions_by_session.items():
        events = event_index[positions]
        for pos in positions:
            target = event_index[pos] + p05a.HORIZON
            local_close = np.searchsorted(events, target, side="left")
            close_lookup[(code, int(pos))] = int(positions[min(local_close, len(positions) - 1)])

    close_pnl = np.zeros(n, dtype=np.float64)
    close_side = np.zeros(n, dtype=np.int64)
    inventory = 0
    total = 0.0
    forced = 0
    rows: list[dict[str, Any]] = []
    first_by_session = {code: int(positions[0]) for code, positions in positions_by_session.items()}
    last_by_session = {code: int(positions[-1]) for code, positions in positions_by_session.items()}
    state: dict[int, dict[str, Any]] = {}
    for code in positions_by_session:
        state[code] = {
            "session_code": code,
            "session_id": str(payload["session_labels"][code]),
            "start_inventory": 0,
            "end_inventory": 0,
            "fill_count": 0,
            "forced_count": 0,
            "terminal_event_fill": 0,
            "same_index_close_scheduled": 0,
        }

    for t in range(n):
        code = int(session_code[t])
        if t == first_by_session[code]:
            state[code]["start_inventory"] = inventory
        if close_pnl[t] != 0.0 or close_side[t] != 0:
            inventory -= int(close_side[t])
            total += float(close_pnl[t])
        if fills[t]:
            side = 1 if decision[t] == p05a.DECISION_BID else -1
            if abs(inventory + side) > 1:
                forced += 1
                state[code]["forced_count"] += 1
            else:
                inventory += side
                close_index = close_lookup[(code, t)]
                close_side[close_index] += side
                close_pnl[close_index] += float(pnl[t])
                state[code]["fill_count"] += 1
                if close_index == t:
                    state[code]["same_index_close_scheduled"] += 1
                if t == last_by_session[code]:
                    state[code]["terminal_event_fill"] += 1
        if t == last_by_session[code]:
            state[code]["end_inventory"] = inventory
            rows.append(state[code])
    return pd.DataFrame(rows), {
        "full_split_total": total,
        "full_split_forced_count": forced,
        "final_inventory": inventory,
        "same_index_close_scheduled": int(sum(row["same_index_close_scheduled"] for row in rows)),
        "sessions_starting_nonzero": int(sum(row["start_inventory"] != 0 for row in rows)),
        "sessions_ending_nonzero": int(sum(row["end_inventory"] != 0 for row in rows)),
    }


def main() -> None:
    selected = json.loads((RESULTS / "tiny_mlp_selected.json").read_text(encoding="utf-8"))
    candidate_id = str(selected["candidate_id"])
    model = phase.TinyMLP(16, int(selected["width"]), 3)
    checkpoint = torch.load(ROOT / "models" / f"{candidate_id}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    payload = helpers.load_payload("new4")
    raw, _ = helpers.base_features(payload)
    mean = np.asarray(selected["feature_mean"], dtype=np.float64)
    scale = np.asarray(selected["feature_scale"], dtype=np.float64)
    standardized = ((raw - mean) / scale).astype(np.float32)
    _, tiny_decision = phase.predict(model, "direct_action", standardized, float(selected["threshold"]))

    static_theta = float(json.loads(
        (phase.PHASE07E / "results" / "run_metadata.json").read_text(encoding="utf-8")
    )["selected_static_theta"])
    queue = payload["target_raw"].numpy()[:, p05a.TARGET_RAW_COLUMNS.index("queue_imbalance")]
    static_decision = np.zeros(len(queue), dtype=np.int8)
    static_decision[np.isfinite(queue) & (queue > static_theta)] = p05a.DECISION_BID
    static_decision[np.isfinite(queue) & (queue < -static_theta)] = p05a.DECISION_ASK
    teacher_scores = helpers.teacher_scores("new4", payload)
    teacher_decision = p05a.policy_from_scores(teacher_scores, float(selected["threshold"]))

    norms = p05a.compute_fill_norms_from_train_sequences()
    summary_rows = []
    for policy_id, decision in (
        (candidate_id, tiny_decision),
        ("queue_theta_val_tuned", static_decision),
        ("phc_eventwise_reference", teacher_decision),
    ):
        probabilities = p05a.fill_probabilities(payload, decision, norms)
        fills = (decision != p05a.DECISION_NO_QUOTE) & (
            p05a.stable_uniform(payload, decision, FILL_SALT) < probabilities
        )
        pnl = p05a.decision_pnl_from_payload(payload, decision)
        sessions, summary = trace_simulator(payload, decision, fills, pnl)
        sessions.insert(0, "policy_id", policy_id)
        sessions.to_csv(RESULTS / f"new4_session_boundary_{policy_id}.csv", index=False)
        summary_rows.append({"policy_id": policy_id, **summary})
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(RESULTS / "new4_session_boundary_summary.csv", index=False)
    print(summary_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
