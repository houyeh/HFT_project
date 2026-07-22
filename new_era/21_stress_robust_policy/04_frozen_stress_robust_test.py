from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from stress_robust_utils import (
    EVALUATION_DIR,
    SEARCH_DIR,
    STRESSES,
    V18_DIR,
    apply_policy,
    baseline_policy_masks,
    decomp_to_row,
    ensure_dirs,
    load_frames_actions,
    simulate_decomp,
)


def _random_reference(stress_name: str) -> tuple[float, float]:
    dist = pd.read_csv(V18_DIR / "random_v3_stress_distribution.csv")
    stress = STRESSES[stress_name]
    row = dist[
        (dist["neutral_fill_haircut"] == stress.neutral_fill_haircut)
        & (dist["toxic_multiplier"] == stress.toxic_multiplier)
        & (dist["cancel_delay_ticks"] == stress.cancel_delay_ticks)
    ].iloc[0]
    return float(row["mean"]), float(row["p95"])


def _decision(rows: pd.DataFrame) -> str:
    mod = rows[(rows["policy"] == "021_selected") & (rows["stress_name"] == "moderate")].iloc[0]
    no = rows[(rows["policy"] == "no_cancel") & (rows["stress_name"] == "moderate")].iloc[0]
    random = rows[(rows["policy"] == "random_v3_95p") & (rows["stress_name"] == "moderate")].iloc[0]
    random_mean = rows[(rows["policy"] == "random_v3_mean") & (rows["stress_name"] == "moderate")].iloc[0]
    if mod["net_pnl"] > 0 and mod["net_pnl"] > random["net_pnl"] and mod["net_pnl"] > no["net_pnl"]:
        return "Ideal Pass"
    if mod["net_pnl"] > 0 and mod["net_pnl"] > random_mean["net_pnl"] and mod["net_pnl"] > no["net_pnl"]:
        return "Strong Defensive"
    if mod["net_pnl"] > no["net_pnl"] and mod["net_pnl"] > random_mean["net_pnl"]:
        return "Partial"
    return "Fail"


def run(force: bool = False) -> None:
    ensure_dirs()
    out_path = EVALUATION_DIR / "frozen_stress_robust_policy_test.csv"
    if out_path.exists() and not force:
        print(f"Frozen test output already exists: {out_path}")
        return

    selected = json.loads((SEARCH_DIR / "selected_stress_robust_policy.json").read_text(encoding="utf-8"))
    selected_policy = selected["selected_policy"]
    frame, actions = load_frames_actions("test")

    rows = []
    for policy_name in ["no_cancel", "015_common", "016_v2", "020_selected_policy"]:
        pb, pa = baseline_policy_masks(frame, policy_name)
        for stress_name, stress in STRESSES.items():
            rows.append(decomp_to_row(policy_name, "test", stress_name, simulate_decomp(frame, actions, pb, pa, stress)))

    pb, pa = apply_policy(frame, selected_policy)
    for stress_name, stress in STRESSES.items():
        rows.append(decomp_to_row("021_selected", "test", stress_name, simulate_decomp(frame, actions, pb, pa, stress)))
        mean, p95 = _random_reference(stress_name)
        rows.append(
            {
                "policy": "random_v3_mean",
                "split": "test",
                "stress_name": stress_name,
                "net_pnl": mean,
                "spread_capture": np.nan,
                "adverse_loss": np.nan,
                "fees": np.nan,
                "reconstructed_pnl": np.nan,
                "reconstruction_error": np.nan,
                "trade_count": np.nan,
                "toxic_fill_count": np.nan,
                "non_toxic_fill_count": np.nan,
                "max_drawdown": np.nan,
                "bid_cancel_count": np.nan,
                "ask_cancel_count": np.nan,
                "any_cancel_rate": np.nan,
            }
        )
        rows.append(
            {
                "policy": "random_v3_95p",
                "split": "test",
                "stress_name": stress_name,
                "net_pnl": p95,
                "spread_capture": np.nan,
                "adverse_loss": np.nan,
                "fees": np.nan,
                "reconstructed_pnl": np.nan,
                "reconstruction_error": np.nan,
                "trade_count": np.nan,
                "toxic_fill_count": np.nan,
                "non_toxic_fill_count": np.nan,
                "max_drawdown": np.nan,
                "bid_cancel_count": np.nan,
                "ask_cancel_count": np.nan,
                "any_cancel_rate": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    decision = _decision(out)
    out["decision_status_for_021"] = decision
    out["selection_split"] = "validation"
    out["test_used_for_selection"] = False
    out.to_csv(out_path, index=False)

    mod = out[out["stress_name"] == "moderate"].copy()
    sel = mod[mod["policy"] == "021_selected"].iloc[0]
    no = mod[mod["policy"] == "no_cancel"].iloc[0]
    r95 = mod[mod["policy"] == "random_v3_95p"].iloc[0]
    lines = [
        "# Frozen Stress-Robust Policy Test Report",
        "",
        "The selected 021 policy was frozen from validation before this test evaluation.",
        "",
        f"- Decision Status: `{decision}`",
        f"- 021 moderate test PnL: `{sel['net_pnl']:.0f}`",
        f"- Gap vs no-cancel moderate: `{sel['net_pnl'] - no['net_pnl']:.0f}`",
        f"- Gap vs random_v3_95p moderate: `{sel['net_pnl'] - r95['net_pnl']:.0f}`",
        "",
        "Decision definitions: Ideal Pass means positive under moderate and above no-cancel/random_v3_95p; Strong Defensive means positive and above no-cancel/random_v3_mean; Partial means negative or not p95-clearing but above no-cancel/random_v3_mean; Fail otherwise.",
    ]
    (EVALUATION_DIR / "stress_robust_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "selection_file": str(SEARCH_DIR / "selected_stress_robust_policy.json"),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "stress_locked": {k: v.__dict__ for k, v in STRESSES.items()},
        "decision_status": decision,
    }
    (EVALUATION_DIR / "frozen_test_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen 021 selected stress-robust policy on test once.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(force=args.force)
    print(f"Wrote {EVALUATION_DIR / 'frozen_stress_robust_policy_test.csv'}")
