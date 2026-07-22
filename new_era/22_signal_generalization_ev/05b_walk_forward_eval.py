from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ev_utils import RESULTS_DIR, ensure_dirs, make_quote_ev_labels, select_threshold_for_val_pnl


FEATURES = ["spread_ticks", "queue_imbalance_z", "trade_intensity_short_z", "cancel_pressure_short_z", "normalized_spread_z"]


def _fit(train: pd.DataFrame):
    if len(train) > 350_000:
        train = train.sample(n=350_000, random_state=20260525)
    X = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = train["quote_positive_ev_moderate"].to_numpy(dtype=int)
    model = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, tol=1e-3, random_state=20260525))
    model.fit(X, y)
    return model


def _scores(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float))[:, 1]


def run() -> None:
    ensure_dirs()
    feasibility = pd.read_csv(RESULTS_DIR / "walk_forward_feasibility_report.csv")
    labels = make_quote_ev_labels()
    rows = []
    for _, fold in feasibility[feasibility["feasible"]].iterrows():
        train_sessions = str(fold["train_sessions"]).split("|")
        val_session = str(fold["val_session"])
        test_session = str(fold["test_session"])
        train = labels[labels["session_id"].astype(str).isin(train_sessions)].copy()
        val = labels[labels["session_id"].astype(str) == val_session].copy()
        test = labels[labels["session_id"].astype(str) == test_session].copy()
        model = _fit(train)
        val_scores = _scores(model, val)
        threshold, val_pnl, val_quote_rate = select_threshold_for_val_pnl(val_scores, val["realized_quote_pnl_moderate"].to_numpy(dtype=float), quote_high_scores=True)
        for split_name, df, scores in [("val", val, val_scores), ("test", test, _scores(model, test))]:
            quote = scores >= threshold
            rows.append(
                {
                    "fold": int(fold["fold"]),
                    "split": split_name,
                    "train_sessions": fold["train_sessions"],
                    "val_session": val_session,
                    "test_session": test_session,
                    "selected_threshold_on_val": threshold,
                    "quote_rate": float(np.mean(quote)),
                    "policy_pnl": float(df.loc[quote, "realized_quote_pnl_moderate"].sum()),
                    "avg_quote_pnl": float(df.loc[quote, "realized_quote_pnl_moderate"].mean()) if quote.any() else np.nan,
                    "base_all_quote_pnl": float(df["realized_quote_pnl_moderate"].sum()),
                    "positive_ev_rate_quoted": float(df.loc[quote, "quote_positive_ev_moderate"].mean()) if quote.any() else np.nan,
                    "uses_current_burned_test": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "walk_forward_policy_results.csv", index=False)
    test = out[out["split"] == "test"]
    lines = [
        "# Walk-Forward Summary",
        "",
        "All folds exclude the current burned 20260127 test sessions. Thresholds are selected on each fold's validation session and evaluated on the next frozen session.",
        "",
        f"- feasible folds run: `{test['fold'].nunique() if not test.empty else 0}`",
        f"- mean frozen-test policy PnL: `{test['policy_pnl'].mean() if not test.empty else float('nan'):.3f}`",
        f"- positive frozen-test folds: `{int((test['policy_pnl'] > 0).sum()) if not test.empty else 0}` / `{len(test)}`",
    ]
    (RESULTS_DIR / "walk_forward_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
    print(f"Wrote {RESULTS_DIR / 'walk_forward_policy_results.csv'}")
