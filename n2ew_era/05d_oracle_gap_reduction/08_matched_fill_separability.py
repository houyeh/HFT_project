from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

import common_matched_fill_diagnostics as d


RANDOM_SEED = 20260728
TEST_SIZE = 0.30
CLASSIFIER_CONFIG = {
    "max_iter": 80,
    "max_leaf_nodes": 15,
    "learning_rate": 0.05,
    "l2_regularization": 0.1,
    "random_state": RANDOM_SEED,
}
POLICIES = [d.HGB_POLICY_ID, d.PHC_POLICY_ID]


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    n = len(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    weights = np.ones(n, dtype=np.float64)
    if pos and neg:
        weights[y == 1] = n / (2.0 * pos)
        weights[y == 0] = n / (2.0 * neg)
    return weights


def run_policy(payload: dict, policy_id: str) -> tuple[dict, list[dict]]:
    policy_decision, _ = d.policy_decision_scores(policy_id, payload)
    oracle_decision = d.matched_count_side_oracle_for_policy(payload, policy_decision)
    oracle_accepted, oracle_summary = d.causal_fill_for_decision(
        payload,
        oracle_decision,
        salt=f"05d_val_matched_fill_oracle_{policy_id}",
    )

    universe = oracle_accepted & (oracle_decision != d.p05a.DECISION_NO_QUOTE)
    positive_mask = universe & (policy_decision != oracle_decision)
    negative_mask = universe & (policy_decision == oracle_decision)
    sample_mask = positive_mask | negative_mask
    y = positive_mask[sample_mask].astype(np.int8)
    x_all, feature_names, feature_source = d.visible_features_for_policy(policy_id, payload)
    x = x_all[sample_mask]

    row = {
        "split": d.VAL_SPLIT,
        "policy_id": policy_id,
        "feature_source": feature_source,
        "feature_count": int(x.shape[1]) if x.ndim == 2 else 0,
        "state_count_val": int(len(policy_decision)),
        "policy_selected_count": int((policy_decision != d.p05a.DECISION_NO_QUOTE).sum()),
        "policy_bid_count": int((policy_decision == d.p05a.DECISION_BID).sum()),
        "policy_ask_count": int((policy_decision == d.p05a.DECISION_ASK).sum()),
        "oracle_bid_count": int((oracle_decision == d.p05a.DECISION_BID).sum()),
        "oracle_ask_count": int((oracle_decision == d.p05a.DECISION_ASK).sum()),
        "oracle_fill_count": int(oracle_summary["fill_count"]),
        "oracle_forced_no_quote_count": int(oracle_summary["forced_no_quote_count"]),
        "binary_sample_count": int(len(y)),
        "positive_missed_oracle_fill_count": int(y.sum()),
        "negative_policy_matches_oracle_fill_count": int((y == 0).sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "classifier_family": "HistGradientBoostingClassifier",
        **{f"classifier_{key}": value for key, value in CLASSIFIER_CONFIG.items()},
        "internal_eval": "stratified_shuffle_split_within_val",
        "test_size": TEST_SIZE,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "new3_used": False,
        "new4_used": False,
        "new5_used": False,
    }
    curve_rows: list[dict] = []

    if len(np.unique(y)) < 2 or min(int(y.sum()), int((y == 0).sum())) < 3:
        row["skip_reason"] = "fewer_than_two_classes_or_too_few_minority_samples"
        return row, curve_rows

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(x, y))
    clf = HistGradientBoostingClassifier(**CLASSIFIER_CONFIG)
    clf.fit(x[train_idx], y[train_idx], sample_weight=balanced_sample_weight(y[train_idx]))
    score = clf.predict_proba(x[test_idx])[:, 1]
    y_test = y[test_idx]
    row["train_sample_count"] = int(len(train_idx))
    row["test_sample_count"] = int(len(test_idx))
    row["test_positive_count"] = int(y_test.sum())
    row["test_negative_count"] = int((y_test == 0).sum())
    row["roc_auc"] = float(roc_auc_score(y_test, score))
    row["pr_auc"] = float(average_precision_score(y_test, score))
    row["skip_reason"] = ""

    fpr, tpr, _ = roc_curve(y_test, score)
    precision, recall, _ = precision_recall_curve(y_test, score)
    curve_rows.extend(
        {"split": d.VAL_SPLIT, "policy_id": policy_id, "curve": "roc", "x": float(xv), "y": float(yv)}
        for xv, yv in zip(fpr, tpr)
    )
    curve_rows.extend(
        {"split": d.VAL_SPLIT, "policy_id": policy_id, "curve": "pr", "x": float(xv), "y": float(yv)}
        for xv, yv in zip(recall, precision)
    )
    d.write_json(
        d.RESULTS / f"matched_fill_separability_feature_manifest_{policy_id}.json",
        {
            "split": d.VAL_SPLIT,
            "policy_id": policy_id,
            "feature_source": feature_source,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "forbidden_future_columns_included": False,
        },
    )
    return row, curve_rows


def plot_curves(curves: pd.DataFrame) -> None:
    if curves.empty:
        return
    fig, axes = d.plt.subplots(1, 2, figsize=(11, 4.5))
    for policy_id in POLICIES:
        roc = curves[(curves["policy_id"].eq(policy_id)) & (curves["curve"].eq("roc"))]
        pr = curves[(curves["policy_id"].eq(policy_id)) & (curves["curve"].eq("pr"))]
        if not roc.empty:
            axes[0].plot(roc["x"], roc["y"], label=policy_id)
        if not pr.empty:
            axes[1].plot(pr["x"], pr["y"], label=policy_id)
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d.FIGURES / "matched_fill_separability_roc_pr.png", dpi=180)
    d.plt.close(fig)


def main() -> None:
    d.ensure_dirs()
    payload = d.load_val_payload()
    rows = []
    curves = []
    for policy_id in POLICIES:
        row, curve_rows = run_policy(payload, policy_id)
        rows.append(row)
        curves.extend(curve_rows)
        print(
            f"diagnostic_a policy={policy_id} samples={row['binary_sample_count']} "
            f"pos={row['positive_missed_oracle_fill_count']} pr_auc={row['pr_auc']}"
        )
    summary = pd.DataFrame(rows)
    curve_df = pd.DataFrame(curves)
    summary.to_csv(d.RESULTS / "matched_fill_separability_summary.csv", index=False)
    curve_df.to_csv(d.RESULTS / "matched_fill_separability_curves.csv", index=False)
    plot_curves(curve_df)
    d.write_combined_report()
    print(d.ROOT / "MATCHED_FILL_SEPARABILITY_CAPACITY_DIAGNOSTICS_REPORT.md")


if __name__ == "__main__":
    main()
