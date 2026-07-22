from __future__ import annotations

import pandas as pd

from common_phase02 import RESULTS_DIR, ensure_dirs, selected_policy_entries, session_pnls


LOSO_SPLITS = ["val", "test", "new4"]


def main() -> None:
    ensure_dirs()
    rows: list[dict] = []
    for policy in selected_policy_entries():
        horizon = int(policy["selected_horizon"])
        for split in LOSO_SPLITS:
            sess = session_pnls(policy, split, horizon)
            if sess.empty:
                continue
            full_pnl = float(sess["session_pnl"].sum())
            for _, r in sess.iterrows():
                loso_pnl = full_pnl - float(r["session_pnl"])
                rows.append(
                    {
                        "role": policy.get("role", ""),
                        "policy_id": policy["policy_id"],
                        "horizon": horizon,
                        "split": split,
                        "dropped_session_id": str(r["session_id"]),
                        "full_sample_pnl": full_pnl,
                        "dropped_session_pnl_contribution": float(r["session_pnl"]),
                        "loso_pnl": loso_pnl,
                        "loso_positive": bool(loso_pnl > 0.0),
                        "full_selected_action_count": int(sess["selected_action_count"].sum()),
                        "dropped_selected_action_count": int(r["selected_action_count"]),
                        "remaining_selected_action_count": int(sess["selected_action_count"].sum() - int(r["selected_action_count"])),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "loso_robustness_by_policy.csv", index=False)
    print("LOSO audit complete:")
    if not out.empty:
        print(out.groupby(["split", "policy_id"])["loso_pnl"].agg(["min", "max", "count"]).to_string())
    print(f"Wrote {RESULTS_DIR / 'loso_robustness_by_policy.csv'}")


if __name__ == "__main__":
    main()
