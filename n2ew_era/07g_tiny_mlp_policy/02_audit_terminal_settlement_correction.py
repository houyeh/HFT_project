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
BOOTSTRAP_SEED = 20261009


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


phase = load_module("tiny_mlp_phase_07g_terminal_fix", MAIN)
helpers = phase.helpers
p05a = phase.p05a


def corrected_simulation(
    payload: dict[str, Any], decision: np.ndarray, fills: np.ndarray, pnl: np.ndarray,
    inventory_cap: int = 1,
) -> dict[str, Any]:
    n = len(decision)
    total_pnl = 0.0
    equity = np.zeros(n, dtype=np.float64)
    close_pnl = np.zeros(n, dtype=np.float64)
    close_side = np.zeros(n, dtype=np.int64)
    inventory = 0
    max_abs_inventory = 0
    forced = 0
    fill_count = 0
    win = 0
    loss = 0
    immediate_settlement_count = 0
    session_code = payload["session_code"].numpy()
    event_index = payload["event_index"].numpy()
    session_positions = {
        int(code): np.where(session_code == code)[0] for code in np.unique(session_code)
    }
    close_lookup: dict[tuple[int, int], int] = {}
    for code, positions in session_positions.items():
        events = event_index[positions]
        for pos in positions:
            target_event = event_index[pos] + p05a.HORIZON
            local_index = np.searchsorted(events, target_event, side="left")
            close_lookup[(code, int(pos))] = int(positions[min(local_index, len(positions) - 1)])

    peak = 0.0
    max_drawdown = 0.0
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
                inventory += side_sign
                max_abs_inventory = max(max_abs_inventory, abs(inventory))
                close_index = close_lookup[(int(session_code[t]), int(t))]
                if close_index == t:
                    # The legacy loop has already consumed close arrays for t. Settle now
                    # instead of scheduling an event that can never be visited again.
                    inventory -= side_sign
                    total_pnl += trade_pnl
                    immediate_settlement_count += 1
                else:
                    close_side[close_index] += side_sign
                    close_pnl[close_index] += trade_pnl
                fill_count += 1
                win += int(trade_pnl > 0.0)
                loss += int(trade_pnl < 0.0)
        equity[t] = total_pnl
        peak = max(peak, total_pnl)
        max_drawdown = max(max_drawdown, peak - total_pnl)
    return {
        "sequential_excess_pnl": float(total_pnl),
        "fill_count": int(fill_count),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_abs_inventory),
        "max_drawdown": float(max_drawdown),
        "fill_win_rate": float(win / fill_count) if fill_count else 0.0,
        "fill_loss_rate": float(loss / fill_count) if fill_count else 0.0,
        "avg_pnl_per_fill": float(total_pnl / fill_count) if fill_count else 0.0,
        "terminal_immediate_settlement_count": int(immediate_settlement_count),
        "final_inventory": int(inventory),
    }


def evaluate_policy(
    split: str, policy_id: str, payload: dict[str, Any], decision: np.ndarray, norms: Any,
) -> dict[str, Any]:
    probabilities = p05a.fill_probabilities(payload, decision, norms)
    fills = (decision != p05a.DECISION_NO_QUOTE) & (
        p05a.stable_uniform(payload, decision, FILL_SALT) < probabilities
    )
    pnl = p05a.decision_pnl_from_payload(payload, decision)
    legacy = helpers.causal_metrics(split, policy_id, payload, decision, norms)
    corrected = corrected_simulation(payload, decision, fills, pnl)
    return {
        "split": split,
        "policy_id": policy_id,
        **{f"legacy_{key}": value for key, value in legacy.items() if key not in ("split", "policy_id")},
        **{f"corrected_{key}": value for key, value in corrected.items()},
        "pnl_correction": corrected["sequential_excess_pnl"] - legacy["sequential_excess_pnl"],
        "forced_count_correction": corrected["forced_no_quote_count"] - legacy["forced_no_quote_count"],
    }


def subset_payload(payload: dict[str, Any], mask: np.ndarray, session_code: int) -> dict[str, Any]:
    n = len(mask)
    output: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == n:
            output[key] = value[mask]
        else:
            output[key] = value
    # Preserve the original code because stable_uniform hashes session_code.
    # Re-numbering every isolated session to zero changes the deterministic fills.
    output["session_code"] = torch.full(
        (int(mask.sum()),), int(session_code), dtype=torch.int32
    )
    output["session_labels"] = list(payload["session_labels"])
    return output


def session_corrected_values(
    payload: dict[str, Any], decision: np.ndarray, norms: Any,
) -> dict[str, float]:
    values: dict[str, float] = {}
    codes = payload["session_code"].numpy()
    for code, label in enumerate(payload["session_labels"]):
        mask = codes == code
        sub = subset_payload(payload, mask, code)
        row = evaluate_policy("session", "session", sub, decision[mask], norms)
        values[str(label)] = float(row["corrected_sequential_excess_pnl"])
    return values


def paired_bootstrap(
    selected_id: str, selected_decisions: dict[tuple[str, str], np.ndarray],
    reference_decisions: dict[tuple[str, str], np.ndarray], norms: Any,
    n_trials: int = 10000,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for split in helpers.SPLITS:
        payload = helpers.load_payload(split)
        student_values = session_corrected_values(payload, selected_decisions[(split, selected_id)], norms)
        for comparator in ("queue_theta_val_tuned", "phc_eventwise_reference"):
            comparator_values = session_corrected_values(
                payload, reference_decisions[(split, comparator)], norms
            )
            labels = sorted(student_values)
            differences = np.asarray(
                [student_values[label] - comparator_values[label] for label in labels],
                dtype=np.float64,
            )
            samples = rng.choice(differences, size=(n_trials, len(differences)), replace=True).mean(axis=1)
            rows.append({
                "split": split,
                "student_policy_id": selected_id,
                "comparator": comparator,
                "session_count": len(labels),
                "corrected_total_difference": float(differences.sum()),
                "mean_session_difference": float(differences.mean()),
                "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_difference_le_zero": float((samples <= 0).mean()),
                "n_bootstrap": n_trials,
            })
    return pd.DataFrame(rows)


def main() -> None:
    selected_metadata = json.loads((RESULTS / "tiny_mlp_selected.json").read_text(encoding="utf-8"))
    mean = np.asarray(selected_metadata["feature_mean"], dtype=np.float64)
    scale = np.asarray(selected_metadata["feature_scale"], dtype=np.float64)
    threshold = float(selected_metadata["threshold"])
    static_theta = float(json.loads(
        (phase.PHASE07E / "results" / "run_metadata.json").read_text(encoding="utf-8")
    )["selected_static_theta"])
    norms = p05a.compute_fill_norms_from_train_sequences()
    training = pd.read_csv(RESULTS / "tiny_mlp_training_summary.csv")

    models: dict[str, tuple[Any, torch.nn.Module]] = {}
    for row in training.itertuples(index=False):
        output_dim = 3 if row.objective == "direct_action" else 2
        model = phase.TinyMLP(16, int(row.width), output_dim)
        checkpoint = torch.load(
            ROOT / "models" / f"{row.candidate_id}.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models[str(row.candidate_id)] = (row, model)

    rows: list[dict[str, Any]] = []
    decisions: dict[tuple[str, str], np.ndarray] = {}
    reference_decisions: dict[tuple[str, str], np.ndarray] = {}
    for split in helpers.SPLITS:
        payload = helpers.load_payload(split)
        raw, _ = helpers.base_features(payload)
        standardized = ((raw - mean) / scale).astype(np.float32)
        queue = payload["target_raw"].numpy()[:, p05a.TARGET_RAW_COLUMNS.index("queue_imbalance")]
        static = np.zeros(len(queue), dtype=np.int8)
        static[np.isfinite(queue) & (queue > static_theta)] = p05a.DECISION_BID
        static[np.isfinite(queue) & (queue < -static_theta)] = p05a.DECISION_ASK
        teacher = p05a.policy_from_scores(helpers.teacher_scores(split, payload), threshold)
        for policy_id, decision in (
            ("queue_theta_val_tuned", static),
            ("phc_eventwise_reference", teacher),
        ):
            row = evaluate_policy(split, policy_id, payload, decision, norms)
            row.update({"objective": "reference", "width": np.nan, "seed": np.nan, "parameter_count": np.nan})
            rows.append(row)
            reference_decisions[(split, policy_id)] = decision
        for candidate_id, (metadata, model) in models.items():
            _, decision = phase.predict(model, str(metadata.objective), standardized, threshold)
            row = evaluate_policy(split, candidate_id, payload, decision, norms)
            row.update({
                "objective": str(metadata.objective),
                "width": int(metadata.width),
                "seed": int(metadata.seed),
                "parameter_count": int(metadata.parameter_count),
            })
            rows.append(row)
            decisions[(split, candidate_id)] = decision

    all_results = pd.DataFrame(rows)
    all_results.to_csv(RESULTS / "terminal_settlement_all_policies.csv", index=False)

    no_terminal = all_results["corrected_terminal_immediate_settlement_count"] == 0
    regression_failures = all_results[
        no_terminal
        & (
            (np.abs(all_results["pnl_correction"]) > 1e-12)
            | (all_results["forced_count_correction"] != 0)
        )
    ]
    if not regression_failures.empty:
        raise AssertionError(
            "Corrected simulator changed policies without same-index terminal fills:\n"
            + regression_failures[["split", "policy_id", "pnl_correction", "forced_count_correction"]].to_string(index=False)
        )
    if not np.all(all_results["corrected_final_inventory"] == 0):
        failures = all_results[all_results["corrected_final_inventory"] != 0]
        raise AssertionError("Corrected simulator left nonzero inventory:\n" + failures.to_string(index=False))

    candidates = all_results[all_results["objective"].isin(phase.OBJECTIVES)].copy()
    champion = candidates[candidates["split"] == "val"].sort_values(
        ["corrected_sequential_excess_pnl", "parameter_count", "policy_id"],
        ascending=[False, True, True],
    ).iloc[0]
    champion_id = str(champion["policy_id"])
    selected = all_results[
        all_results["policy_id"].isin([champion_id, "queue_theta_val_tuned", "phc_eventwise_reference"])
    ].copy()
    selected.to_csv(RESULTS / "terminal_settlement_selected_comparison.csv", index=False)
    bootstrap = paired_bootstrap(champion_id, decisions, reference_decisions, norms)
    bootstrap.to_csv(RESULTS / "terminal_settlement_paired_bootstrap.csv", index=False)

    pivot = selected.pivot(index="split", columns="policy_id", values="corrected_sequential_excess_pnl").reset_index()
    champion_rows = selected[selected["policy_id"] == champion_id][[
        "split", "policy_id", "objective", "width", "seed", "parameter_count",
        "legacy_sequential_excess_pnl", "corrected_sequential_excess_pnl", "pnl_correction",
        "legacy_forced_no_quote_count", "corrected_forced_no_quote_count",
        "corrected_terminal_immediate_settlement_count", "corrected_final_inventory",
    ]].merge(pivot, on="split")
    champion_rows["retention_vs_phc_pct"] = (
        100.0 * champion_rows["corrected_sequential_excess_pnl"]
        / champion_rows["phc_eventwise_reference"]
    )
    champion_rows["delta_vs_static"] = (
        champion_rows["corrected_sequential_excess_pnl"]
        - champion_rows["queue_theta_val_tuned"]
    )
    champion_rows.to_csv(RESULTS / "terminal_settlement_champion_summary.csv", index=False)

    val = champion_rows[champion_rows["split"] == "val"].iloc[0]
    oos = champion_rows[champion_rows["split"].isin(["new3", "new4"])]
    positive = bool(
        val["retention_vs_phc_pct"] >= 80.0
        and val["delta_vs_static"] > 0
        and np.all(oos["delta_vs_static"] > 0)
    )
    status = "resolved_positive" if positive else "mixed"
    impacted = all_results[all_results["corrected_terminal_immediate_settlement_count"] > 0][[
        "split", "policy_id", "legacy_sequential_excess_pnl", "corrected_sequential_excess_pnl",
        "pnl_correction", "legacy_forced_no_quote_count", "corrected_forced_no_quote_count",
        "corrected_terminal_immediate_settlement_count",
    ]]
    lines = [
        "# Terminal Same-Index Settlement Audit\n",
        f"status: `{status}`\n",
        "## Scope\n",
        "This audit does not modify `common_05a.py` or overwrite any prior metric. It locally corrects one scheduling defect: when a fill is opened at the event chosen as its own clipped close index, the legacy loop schedules the close after that index has already been consumed. The corrected diagnostic settles that fill immediately. Train/val/NEW3/NEW4 data and deterministic fill draws are otherwise unchanged; NEW5 was not loaded.\n",
        "## Regression gate\n",
        f"Policies without a same-index terminal fill changed in neither PnL nor forced-count: `{regression_failures.empty}`. Every corrected run ended with zero inventory: `{bool(np.all(all_results['corrected_final_inventory'] == 0))}`.\n",
        "## Affected policy/split rows\n",
        impacted.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Val-selected tiny MLP under corrected semantics\n",
        f"Val selected `{champion_id}`. NEW3/NEW4 were not used for selection.\n",
        champion_rows.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Paired session bootstrap\n",
        bootstrap.to_markdown(index=False, floatfmt=".6f") + "\n",
        "## Decision\n",
        (
            "Under the locally corrected settlement semantics, the val-selected tiny MLP clears the predeclared task-level gate and beats the tuned static rule on val, NEW3, and NEW4. This is still CPU-only diagnostic evidence. Before treating it as the official result, the shared simulator correction must be reviewed and all frozen policy baselines must be regenerated under one versioned protocol.\n"
            if positive
            else "The local settlement correction removes the immortal-inventory artifact but the tiny MLP still does not clear the full gate. Further model work is required before HLS.\n"
        ),
    ]
    (ROOT / "AUDIT_terminal_settlement_correction.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] champion={champion_id} status={status}", flush=True)
    print(champion_rows.to_string(index=False), flush=True)
    print(f"[OK] report={ROOT / 'AUDIT_terminal_settlement_correction.md'}", flush=True)


if __name__ == "__main__":
    main()
