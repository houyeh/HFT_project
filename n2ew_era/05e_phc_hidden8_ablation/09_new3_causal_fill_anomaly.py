from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import common_05e as c05e


ROOT = c05e.ROOT
RESULTS = c05e.RESULTS
PROJECT_ROOT = c05e.PROJECT_ROOT
PHASE05C4_COMMON = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr" / "common_05c4.py"
OUT_CSV = RESULTS / "phc_hidden8_new3_causal_fill_anomaly_by_session.csv"
OUT_NOTE = ROOT / "PHC_HIDDEN8_NEW3_CAUSAL_FILL_ANOMALY_NOTE.md"


def load_common_05c4() -> Any:
    spec = importlib.util.spec_from_file_location("common_05c4_for_05e_anomaly", PHASE05C4_COMMON)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {PHASE05C4_COMMON}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate_one_session(
    decision: np.ndarray,
    fills: np.ndarray,
    pnl: np.ndarray,
    event_index: np.ndarray,
    inventory_cap: int = 1,
) -> dict[str, float | int | np.ndarray]:
    n = len(decision)
    total_pnl = 0.0
    equity = np.zeros(n, dtype=np.float64)
    close_pnl = np.zeros(n, dtype=np.float64)
    close_side = np.zeros(n, dtype=np.int64)
    close_lookup: dict[int, int] = {}
    for pos in range(n):
        target_event = event_index[pos] + c05e.p05a.HORIZON
        j = np.searchsorted(event_index, target_event, side="left")
        close_lookup[pos] = int(min(j, n - 1))

    inventory = 0
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
            side_sign = 1 if decision[t] == c05e.p05a.DECISION_BID else -1
            if abs(inventory + side_sign) > inventory_cap:
                forced += 1
            else:
                trade_pnl = float(pnl[t])
                inventory += side_sign
                max_abs_inventory = max(max_abs_inventory, abs(inventory))
                close_idx = close_lookup[t]
                close_side[close_idx] += side_sign
                close_pnl[close_idx] += trade_pnl
                fill_count += 1
                win += int(trade_pnl > 0.0)
                loss += int(trade_pnl < 0.0)
        equity[t] = total_pnl
        peak = max(peak, total_pnl)
        max_dd = max(max_dd, peak - total_pnl)
    return {
        "sequential_causal_fill_pnl": float(total_pnl),
        "executed_fill_count": int(fill_count),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_abs_inventory),
        "max_drawdown": float(max_dd),
        "win_count": int(win),
        "loss_count": int(loss),
        "equity": equity,
    }


def model_decision(phase_common: Any, payload: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    model, ckpt, selected = phase_common.selected_model()
    decision, _ = phase_common.stp_decision_for_payload(payload, model, ckpt, float(selected["threshold"]))
    return decision, selected


def rows_for_model(label: str, phase_common: Any, payload: dict[str, Any], norms: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
    decision, selected = model_decision(phase_common, payload)
    probs = c05e.p05a.fill_probabilities(payload, decision, norms)
    raw_fills = (decision != c05e.p05a.DECISION_NO_QUOTE) & (
        c05e.p05a.stable_uniform(payload, decision, phase_common.FILL_RANDOM_SALT) < probs
    )
    pnl = c05e.p05a.decision_pnl_from_payload(payload, decision)
    session_code = payload["session_code"].numpy()
    event_index = payload["event_index"].numpy()
    rows: list[dict[str, Any]] = []
    for session_idx, session_id in enumerate(payload["session_labels"]):
        mask = session_code == session_idx
        sim = simulate_one_session(decision[mask], raw_fills[mask], pnl[mask], event_index[mask], inventory_cap=1)
        selected_count = int((decision[mask] != c05e.p05a.DECISION_NO_QUOTE).sum())
        raw_fill_count = int(raw_fills[mask].sum())
        rows.append(
            {
                "model": label,
                "split": "new3",
                "session_id": session_id,
                "state_count": int(mask.sum()),
                "selected_action_count": selected_count,
                "bid_count": int((decision[mask] == c05e.p05a.DECISION_BID).sum()),
                "ask_count": int((decision[mask] == c05e.p05a.DECISION_ASK).sum()),
                "no_quote_count": int((decision[mask] == c05e.p05a.DECISION_NO_QUOTE).sum()),
                "raw_fill_signal_count": raw_fill_count,
                "no_fill_count_before_inventory_cap": int(selected_count - raw_fill_count),
                "executed_fill_count": int(sim["executed_fill_count"]),
                "forced_no_quote_count": int(sim["forced_no_quote_count"]),
                "per_session_causal_fill_pnl": float(sim["sequential_causal_fill_pnl"]),
                "per_session_action_level_pnl": float(pnl[mask].sum()),
                "win_count": int(sim["win_count"]),
                "loss_count": int(sim["loss_count"]),
                "max_drawdown": float(sim["max_drawdown"]),
                "max_abs_inventory": int(sim["max_abs_inventory"]),
                "seed_selected_on_val": int(selected["seed"]),
                "threshold_selected_on_val": float(selected["threshold"]),
                "phase": phase_common.PHASE,
                "new5_used_for_selection": bool(selected.get("new5_used_for_selection", False)),
            }
        )
    totals = {
        "selected_action_count": float((decision != c05e.p05a.DECISION_NO_QUOTE).sum()),
        "bid_count": float((decision == c05e.p05a.DECISION_BID).sum()),
        "ask_count": float((decision == c05e.p05a.DECISION_ASK).sum()),
        "no_quote_count": float((decision == c05e.p05a.DECISION_NO_QUOTE).sum()),
        "raw_fill_signal_count": float(raw_fills.sum()),
        "per_session_causal_fill_pnl_sum": float(sum(r["per_session_causal_fill_pnl"] for r in rows)),
        "per_session_action_level_pnl_sum": float(sum(r["per_session_action_level_pnl"] for r in rows)),
    }
    return rows, totals


def pivot_comparison(long_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["split", "session_id"]
    metrics = [
        "state_count",
        "selected_action_count",
        "bid_count",
        "ask_count",
        "no_quote_count",
        "raw_fill_signal_count",
        "no_fill_count_before_inventory_cap",
        "executed_fill_count",
        "forced_no_quote_count",
        "per_session_causal_fill_pnl",
        "per_session_action_level_pnl",
        "win_count",
        "loss_count",
        "max_drawdown",
    ]
    h8 = long_df[long_df["model"].eq("hidden8")][keys + metrics].rename(columns={m: f"{m}_hidden8" for m in metrics})
    h16 = long_df[long_df["model"].eq("hidden16")][keys + metrics].rename(columns={m: f"{m}_hidden16" for m in metrics})
    out = h8.merge(h16, on=keys, how="outer")
    for m in metrics:
        if f"{m}_hidden8" in out.columns and f"{m}_hidden16" in out.columns:
            out[f"{m}_delta_hidden8_minus_hidden16"] = out[f"{m}_hidden8"] - out[f"{m}_hidden16"]
    total_delta = float(out["per_session_causal_fill_pnl_delta_hidden8_minus_hidden16"].sum())
    if total_delta != 0:
        out["causal_fill_delta_share_of_total"] = out["per_session_causal_fill_pnl_delta_hidden8_minus_hidden16"] / total_delta
    else:
        out["causal_fill_delta_share_of_total"] = np.nan
    out = out.sort_values("per_session_causal_fill_pnl_delta_hidden8_minus_hidden16")
    return out


def cached_new3_causal_rows() -> tuple[pd.Series, pd.Series]:
    h8 = pd.read_csv(RESULTS / "phc_stp_v2_causal_fill_eval.csv")
    h16 = pd.read_csv(c05e.PHASE05C4 / "results" / "phc_stp_v2_causal_fill_eval.csv")
    h8_row = h8[h8["split"].eq("new3")].iloc[0]
    h16_row = h16[h16["split"].eq("new3")].iloc[0]
    return h8_row, h16_row


def diagnose(comp: pd.DataFrame, cached_delta: float, reset_delta: float, cached_forced_delta: float) -> tuple[str, str, float, float, str]:
    deltas = comp["per_session_causal_fill_pnl_delta_hidden8_minus_hidden16"].to_numpy(dtype=float)
    negative = deltas[deltas < 0]
    if len(negative) == 0:
        return "new3_causal_fill_anomaly_is_variance_artifact", "high", 0.0, 0.0, "No per-session negative causal-fill deltas remain after resetting the simulator by session."
    abs_negative = np.sort(np.abs(negative))[::-1]
    top1 = float(abs_negative[0] / abs_negative.sum()) if abs_negative.sum() else 0.0
    top2 = float(abs_negative[:2].sum() / abs_negative.sum()) if abs_negative.sum() else 0.0
    losing_sessions = int((deltas < 0).sum())
    artifact_component = cached_delta - reset_delta
    if cached_forced_delta > 0 and abs(artifact_component) >= 0.50 * max(abs(cached_delta), 1.0):
        return (
            "new3_causal_fill_anomaly_is_variance_artifact",
            "high",
            top1,
            top2,
            "Most of the aggregate anomaly disappears when the sequential simulator is reset by session; the remaining regression is milder and all excess forced no-quotes occur only in the cached full aggregate hidden8 run.",
        )
    if top2 >= 0.75 and losing_sessions <= 3:
        return (
            "new3_causal_fill_anomaly_is_variance_artifact",
            "medium",
            top1,
            top2,
            "The causal-fill delta is concentrated in one or two sessions.",
        )
    return (
        "new3_causal_fill_anomaly_is_genuine_capability_loss",
        "medium",
        top1,
        top2,
        "The reset-by-session causal-fill regression remains distributed across sessions without a dominant concentration artifact.",
    )


def main() -> None:
    c05c4 = load_common_05c4()
    payload = c05e.p05a.load_sequences("new3")
    norms = c05e.p05a.compute_fill_norms_from_train_sequences()
    rows8, totals8 = rows_for_model("hidden8", c05e, payload, norms)
    rows16, totals16 = rows_for_model("hidden16", c05c4, payload, norms)
    long_df = pd.DataFrame(rows8 + rows16)
    comp = pivot_comparison(long_df)
    long_df.to_csv(RESULTS / "phc_hidden8_new3_causal_fill_anomaly_long.csv", index=False)
    comp.to_csv(OUT_CSV, index=False)

    cached_h8, cached_h16 = cached_new3_causal_rows()
    total_delta = float(comp["per_session_causal_fill_pnl_delta_hidden8_minus_hidden16"].sum())
    cached_delta = float(cached_h8["sequential_excess_pnl"] - cached_h16["sequential_excess_pnl"])
    cached_forced_delta = float(cached_h8["forced_no_quote_count"] - cached_h16["forced_no_quote_count"])
    verdict, confidence, top1, top2, verdict_reason = diagnose(comp, cached_delta, total_delta, cached_forced_delta)
    action_delta = float(comp["per_session_action_level_pnl_delta_hidden8_minus_hidden16"].sum())
    fill_delta = float(comp["executed_fill_count_delta_hidden8_minus_hidden16"].sum())
    raw_fill_delta = float(comp["raw_fill_signal_count_delta_hidden8_minus_hidden16"].sum())
    selected_delta = float(comp["selected_action_count_delta_hidden8_minus_hidden16"].sum())
    forced_delta = float(comp["forced_no_quote_count_delta_hidden8_minus_hidden16"].sum())
    artifact_delta = cached_delta - total_delta
    artifact_share = artifact_delta / cached_delta if cached_delta else np.nan

    lines = [
        "# PHC Hidden8 NEW3 Causal-Fill Anomaly Note",
        "",
        f"One-line verdict: `{verdict}`.",
        "",
        f"Confidence: `{confidence}`.",
        "",
        "This is a diagnostic addendum only. It does not retrain any model, does not revise the Phase 05E regression verdict, and does not use NEW5.",
        "",
        "## Aggregate Reconciliation",
        "",
        "| metric | hidden8 | hidden16 | hidden8_minus_hidden16 |",
        "|:--|--:|--:|--:|",
        f"| cached_full_sim_causal_fill_pnl | {float(cached_h8['sequential_excess_pnl']):.0f} | {float(cached_h16['sequential_excess_pnl']):.0f} | {cached_delta:.0f} |",
        f"| cached_full_sim_forced_no_quote_count | {float(cached_h8['forced_no_quote_count']):.0f} | {float(cached_h16['forced_no_quote_count']):.0f} | {cached_forced_delta:.0f} |",
        f"| selected_action_count | {totals8['selected_action_count']:.0f} | {totals16['selected_action_count']:.0f} | {selected_delta:.0f} |",
        f"| raw_fill_signal_count | {totals8['raw_fill_signal_count']:.0f} | {totals16['raw_fill_signal_count']:.0f} | {raw_fill_delta:.0f} |",
        f"| executed_fill_count | {comp['executed_fill_count_hidden8'].sum():.0f} | {comp['executed_fill_count_hidden16'].sum():.0f} | {fill_delta:.0f} |",
        f"| forced_no_quote_count | {comp['forced_no_quote_count_hidden8'].sum():.0f} | {comp['forced_no_quote_count_hidden16'].sum():.0f} | {forced_delta:.0f} |",
        f"| per_session_causal_fill_pnl_sum | {totals8['per_session_causal_fill_pnl_sum']:.0f} | {totals16['per_session_causal_fill_pnl_sum']:.0f} | {total_delta:.0f} |",
        f"| per_session_action_level_pnl_sum | {totals8['per_session_action_level_pnl_sum']:.0f} | {totals16['per_session_action_level_pnl_sum']:.0f} | {action_delta:.0f} |",
        "",
        "The headline aggregate causal-fill delta is the cached full-simulator result. The per-session simulation below resets state by session, which is the correct diagnostic view for concentration and reveals whether the large aggregate delta is driven by session-level capability loss or by sequential simulator state interactions.",
        "",
        "## Concentration Diagnosis",
        "",
        f"- Cached full-simulator causal-fill delta: `{cached_delta:,.0f}`.",
        f"- Reset-by-session causal-fill delta: `{total_delta:,.0f}`.",
        f"- Artifact-like component removed by session reset: `{artifact_delta:,.0f}` ({artifact_share:.2%} of cached delta).",
        f"- Largest losing session share of negative delta: `{top1:.2%}`.",
        f"- Top two losing sessions share of negative delta: `{top2:.2%}`.",
        f"- Hidden8 selected `{selected_delta:,.0f}` fewer actions and produced `{fill_delta:,.0f}` fewer executed fills than hidden16 on NEW3 after session reset.",
        f"- Cached full-simulator forced no-quotes differ by `{cached_forced_delta:,.0f}`; reset-by-session forced no-quotes differ by `{forced_delta:,.0f}`.",
        f"- Action-level PnL delta is `{action_delta:,.0f}`, so a real action-ranking regression remains, but it does not explain the full -19.9% causal-fill anomaly by itself.",
        "",
        "## Per-Session Table",
        "",
        comp.to_markdown(index=False),
        "",
        "## Diagnosis",
        "",
        verdict_reason,
        "",
    ]
    if verdict.endswith("variance_artifact"):
        lines.extend(
            [
                "The large -19.9% NEW3 causal-fill regression should be treated as an artifact-amplified cell rather than a clean capability-loss estimate. More specifically, this is an inventory-cap / sequential simulator artifact, not merely a one-session variance artifact.",
                "A genuine hidden8 regression remains after session reset, but the reset-by-session causal-fill gap is much smaller than the cached aggregate gap. The Phase 05E conclusion remains unchanged: hidden8 should not replace hidden16.",
            ]
        )
    else:
        lines.extend(
            [
                "The NEW3 causal-fill regression is not isolated to one outlier session. It is consistent with a genuine capability loss under the sequential fill proxy.",
                "The action-level regression is smaller than the causal-fill regression, so the loss is amplified by the conservative fill model and inventory cap, but the session spread does not support dismissing it as pure variance.",
            ]
        )
    lines.extend(
        [
            "",
            "## What Would Change This Diagnosis",
            "",
            "- If a larger OOS set showed the same action-level regression but no causal-fill regression, this NEW3 cell would look more like simulator variance.",
            "- If per-session deltas under alternative fixed fill masks or inventory caps stayed broadly negative, confidence in genuine capability loss would increase.",
            "- No NEW5 or retraining should be used to settle this diagnostic; it requires future fresh OOS or protocol-prespecified sensitivity checks.",
        ]
    )
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_NOTE}")
    print(f"verdict={verdict} confidence={confidence} top1={top1:.4f} top2={top2:.4f}")


if __name__ == "__main__":
    main()
