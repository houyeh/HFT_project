from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _load_strict_random_module():
    sys.path.insert(0, str(ROOT))
    path = ROOT / "15_strict_random_baseline.py"
    spec = importlib.util.spec_from_file_location("strict_random_baseline_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strict_random_baseline_v2"] = module
    spec.loader.exec_module(module)
    return module


def _debug_random_v1() -> dict[str, object]:
    mod = _load_strict_random_module()
    frame = mod.load_test_frame()
    action_df, by_session, final_mid = mod._build_actions(frame)
    methods, _ = mod._method_specs(frame, action_df)
    specs = methods["random_v1"]
    rng = np.random.default_rng(20260523)

    trial_rows = []
    for trial in range(25):
        canceled = mod._sample_cancels(specs, rng, len(action_df))
        agg, _ = mod._simulate_actions(frame, by_session, final_mid, canceled)
        trial_rows.append(
            {
                "trial": trial,
                "canceled_fill_actions": int(canceled.sum()),
                "net_pnl": float(agg["net_pnl"]),
                "trade_count": int(agg["trade_count"]),
                "max_drawdown": float(agg["max_drawdown"]),
            }
        )

    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(RESULTS / "random_v1_debug_trials.csv", index=False)

    return {
        "rows": int(len(frame)),
        "actions_total": int(len(action_df)),
        "actions_by_side": action_df["side"].value_counts().to_dict(),
        "spec_rows": [
            {
                "side": spec.side,
                "required_model_cancels": spec.required_model_cancels,
                "candidate_pool_size": spec.candidate_pool_size,
                "action_ids_available_after_fix": int(len(spec.action_ids)),
            }
            for spec in specs
        ],
        "debug_trials": trial_rows,
        "debug_net_pnl_std": float(trial_df["net_pnl"].std(ddof=0)),
        "debug_canceled_action_mean": float(trial_df["canceled_fill_actions"].mean()),
    }


def run(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    debug = _debug_random_v1()

    dist_path = results_dir / "strict_random_baseline_distribution.csv"
    prior = pd.read_csv(dist_path) if dist_path.exists() else pd.DataFrame()
    prior_v1 = prior[prior["method"] == "random_v1"].iloc[0].to_dict() if not prior.empty else {}

    md = [
        "# random_v1 Debug Report",
        "",
        "## Symptom",
        "",
        "The Round 2 `strict_random_baseline_distribution.csv` reported `random_v1` with zero PnL variance.",
        "",
        "## Root Cause",
        "",
        "The `random_v1` grouping has no conditioning keys. In the prior implementation, the no-key bucket was represented as an empty tuple in one helper and as a nested tuple in another helper. That mismatch produced empty `action_ids` for `random_v1`, so no fill actions were ever cancelled in v1 trials. The result collapsed to the no-cancel PnL.",
        "",
        "## Fix",
        "",
        "`15_strict_random_baseline.py` now uses an explicit `ALL` bucket for no-key `random_v1` grouping, so candidate rows and fill action ids share the same key.",
        "",
        "## Prior v1 Row",
        "",
        f"- prior_mean: `{prior_v1.get('mean', 'NA')}`",
        f"- prior_std: `{prior_v1.get('std', 'NA')}`",
        f"- prior_p95: `{prior_v1.get('p95', 'NA')}`",
        "",
        "## Post-Fix Smoke Test",
        "",
        f"- common rows: `{debug['rows']}`",
        f"- fill actions total: `{debug['actions_total']}`",
        f"- fill actions by side: `{debug['actions_by_side']}`",
        f"- debug 25-trial net PnL std: `{debug['debug_net_pnl_std']:.4f}`",
        f"- mean cancelled fill actions in smoke test: `{debug['debug_canceled_action_mean']:.2f}`",
        "",
        "Spec rows:",
        "",
    ]
    for row in debug["spec_rows"]:
        md.append(
            f"- side={row['side']}, required_model_cancels={row['required_model_cancels']}, "
            f"candidate_pool_size={row['candidate_pool_size']}, action_ids_available_after_fix={row['action_ids_available_after_fix']}"
        )

    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "`random_v1` should be treated as the loosest sanity baseline. The main statistical hurdles remain the stricter `random_v2` and `random_v3` rows because they match session and spread structure.",
        ]
    )
    (results_dir / "random_v1_debug_report.md").write_text("\n".join(md), encoding="utf-8")

    (results_dir / "cancel_signal_definition.md").write_text(
        "# Cancel Signal Definition\n\n"
        "For `016_v2_binary`, a cancel signal is a side-specific binary quote action derived from the validation-frozen toxic-score threshold.\n\n"
        "- `pull_bid[t] = bid_prob_toxic[t] >= bid_threshold`\n"
        "- `pull_ask[t] = ask_prob_toxic[t] >= ask_threshold`\n"
        "- Frozen thresholds: bid `0.015858395025134087`, ask `0.02090054750442505`\n"
        "- Threshold source: validation only, target 10% cancel rate per side\n"
        "- Execution meaning: when `pull_bid` is true, the passive bid quote is cancelled for that event; when `pull_ask` is true, the passive ask quote is cancelled for that event.\n\n"
        "These are quote-risk proxy signals, not exchange-confirmed cancels or queue-position-aware fill outcomes.\n",
        encoding="utf-8",
    )

    (results_dir / "baseline_interpretation.md").write_text(
        "# Baseline Interpretation\n\n"
        "`016_v2_binary` does not replace `015_smoothed`. They are competing baseline candidates under different operating assumptions.\n\n"
        "- `015_smoothed` is a legacy smoothed mid-move TCN strategy with skew/gating behavior and a historically audited universe.\n"
        "- `016_v2_binary` is a side-specific quote-risk proxy cancel model with validation-frozen thresholds and stronger matched-random audit evidence.\n"
        "- The Pre-018 comparison treats both as candidates for queue-aware stress testing, not as a winner-takes-all replacement.\n\n"
        "The purpose of `018_queue_aware` is to stress both candidates under common execution assumptions and determine which remains robust when fills, adverse selection, and cancel latency are made less favorable.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 016_v2 report issue fixes.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.results_dir)
    print(f"016_v2 report issue fixes written to {args.results_dir}")
