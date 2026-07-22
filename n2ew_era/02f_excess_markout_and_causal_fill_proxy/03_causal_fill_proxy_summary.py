from __future__ import annotations

import numpy as np
import pandas as pd

from common_phase02f import (
    EVAL_SPLITS,
    FILL_MODELS,
    RESULTS,
    SELECTED_POLICY_ID,
    compute_train_fill_norms,
    deterministic_fill_mask,
    ensure_dirs,
    fill_probabilities,
    iter_excess_event_frames,
    policy_decision,
    policy_set,
    write_json,
)


def main() -> None:
    ensure_dirs()
    norms = compute_train_fill_norms()
    write_json(RESULTS / "causal_fill_proxy_norms.json", norms.__dict__)
    policies = policy_set()
    rows = []
    for split in EVAL_SPLITS:
        for policy_id in ["old_bid_only_spread_ge_1", SELECTED_POLICY_ID]:
            for model in FILL_MODELS:
                state_count = 0
                selected_count = 0
                prob_sum = 0.0
                fill_count = 0
                bid_fill_count = 0
                ask_fill_count = 0
                for frame in iter_excess_event_frames(split, horizons=[20]):
                    decision = policy_decision(policies[policy_id], frame)
                    probs = fill_probabilities(frame, decision, model, norms)
                    fills, _ = deterministic_fill_mask(frame, decision, model, norms)
                    selected = decision != 0
                    state_count += int(len(frame))
                    selected_count += int(selected.sum())
                    prob_sum += float(probs[selected].sum())
                    fill_count += int(fills.sum())
                    bid_fill_count += int((fills & (decision == 1)).sum())
                    ask_fill_count += int((fills & (decision == 2)).sum())
                rows.append(
                    {
                        "split": split,
                        "policy_id": policy_id,
                        "fill_model": model,
                        "state_count": int(state_count),
                        "selected_action_count": int(selected_count),
                        "mean_fill_probability_selected": float(prob_sum / max(selected_count, 1)) if selected_count else 0.0,
                        "deterministic_fill_count": int(fill_count),
                        "deterministic_fill_rate_selected": float(fill_count / max(selected_count, 1)) if selected_count else 0.0,
                        "bid_fill_count": int(bid_fill_count),
                        "ask_fill_count": int(ask_fill_count),
                        "uses_future_pnl_for_fill": False,
                        "normalization_source": "train_split_current_l2_medians",
                    }
                )
            print(f"fill proxy summarized split={split} policy={policy_id}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "causal_fill_proxy_summary.csv", index=False)
    print("Causal fill proxy summary complete.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

