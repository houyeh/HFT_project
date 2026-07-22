from __future__ import annotations

import pandas as pd

from ev_utils import RESULTS_DIR, ensure_dirs, make_quote_ev_labels


def run() -> None:
    ensure_dirs()
    labels = make_quote_ev_labels()
    pretest = labels[labels["split"] != "test"].copy()
    rows = []
    sessions = sorted(pretest["session_id"].astype(str).unique().tolist())
    for i in range(3, len(sessions) - 1):
        train_sessions = sessions[:i]
        val_session = sessions[i]
        test_session = sessions[i + 1]
        tr = pretest[pretest["session_id"].astype(str).isin(train_sessions)]
        va = pretest[pretest["session_id"].astype(str) == val_session]
        te = pretest[pretest["session_id"].astype(str) == test_session]
        rows.append(
            {
                "fold": len(rows),
                "train_sessions": "|".join(train_sessions),
                "val_session": val_session,
                "test_session": test_session,
                "train_samples": int(len(tr)),
                "val_samples": int(len(va)),
                "test_samples": int(len(te)),
                "train_positive_ev_rate": float(tr["quote_positive_ev_moderate"].mean()),
                "val_positive_ev_rate": float(va["quote_positive_ev_moderate"].mean()),
                "test_positive_ev_rate": float(te["quote_positive_ev_moderate"].mean()),
                "feasible": bool(len(tr) >= 100_000 and len(va) >= 20_000 and len(te) >= 20_000),
                "uses_current_burned_test": False,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "walk_forward_feasibility_report.csv", index=False)


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / 'walk_forward_feasibility_report.csv'}")
