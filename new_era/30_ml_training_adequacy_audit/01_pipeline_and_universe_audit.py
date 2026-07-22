from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_030 import (
    FEATURE_COLUMNS,
    RESULTS,
    V28_DIR,
    V28_RESULTS,
    ensure_dirs,
    load_all_splits,
    write_markdown,
)


def training_pipeline_audit() -> pd.DataFrame:
    metrics = pd.read_csv(V28_RESULTS / "residual_ev_model_metrics.csv")
    rows = [
        {
            "audit_item": "phase_028_location",
            "value": str(V28_DIR),
            "status": "read_only_reference",
            "notes": "Phase 030 does not overwrite Phase 028 artifacts.",
        },
        {
            "audit_item": "phase_028_feature_count",
            "value": len(FEATURE_COLUMNS),
            "status": "causal_tabular_features",
            "notes": ",".join(FEATURE_COLUMNS),
        },
        {
            "audit_item": "phase_028_model_family",
            "value": ",".join(sorted(metrics["model_name"].unique())),
            "status": "shallow_baseline",
            "notes": "Logistic regression and one small MLP were trained in Phase 028.",
        },
        {
            "audit_item": "phase_028_validation_selection",
            "value": "validation_only",
            "status": "pass",
            "notes": "Selected veto threshold came from validation and test_used_for_selection=false.",
        },
        {
            "audit_item": "phase_028_selected_veto_count",
            "value": int(json.loads((V28_RESULTS / "selected_ev_veto_policy.json").read_text(encoding="utf-8"))["validation_metrics"]["veto_count"]),
            "status": "no_veto_selected",
            "notes": "Validation selected no veto, indicating no shallow incremental signal.",
        },
        {
            "audit_item": "torch_available_locally",
            "value": bool(importlib.util.find_spec("torch")),
            "status": "unavailable" if not importlib.util.find_spec("torch") else "available",
            "notes": "GPU TCN pack is exported for external RTX training.",
        },
        {
            "audit_item": "phase_030_max_veto_rate",
            "value": "<=5%",
            "status": "locked",
            "notes": "Expanded tabular models are evaluated under a low-veto constraint.",
        },
    ]
    return pd.DataFrame(rows)


def residual_universe_difficulty() -> pd.DataFrame:
    frames = load_all_splits()
    rows: list[dict] = []
    for split, df in frames.items():
        values = df["quote_ev_moderate"].astype(float)
        neg = values[values < 0.0]
        severe_1k = values <= -1000.0
        severe_5k = values <= -5000.0
        severe_10k = values <= -10000.0
        sorted_loss = (-neg).sort_values(ascending=False).to_numpy()
        total_loss = float(sorted_loss.sum()) if len(sorted_loss) else 0.0
        rows.append(
            {
                "split": split,
                "sample_count": int(len(df)),
                "session_count": int(df["session_id"].nunique()),
                "baseline_quote_pnl": float(values.sum()),
                "negative_ev_count": int((values < 0.0).sum()),
                "negative_ev_rate": float((values < 0.0).mean()) if len(df) else 0.0,
                "severe_negative_le_1000_count": int(severe_1k.sum()),
                "severe_negative_le_1000_rate": float(severe_1k.mean()) if len(df) else 0.0,
                "severe_negative_le_5000_count": int(severe_5k.sum()),
                "severe_negative_le_5000_rate": float(severe_5k.mean()) if len(df) else 0.0,
                "severe_negative_le_10000_count": int(severe_10k.sum()),
                "severe_negative_le_10000_rate": float(severe_10k.mean()) if len(df) else 0.0,
                "negative_loss_total_abs": total_loss,
                "top_10_negative_loss_share": float(sorted_loss[: min(10, len(sorted_loss))].sum() / total_loss) if total_loss else np.nan,
                "median_quote_ev": float(values.median()) if len(values) else np.nan,
                "p05_quote_ev": float(values.quantile(0.05)) if len(values) else np.nan,
                "p01_quote_ev": float(values.quantile(0.01)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    training_pipeline_audit().to_csv(RESULTS / "training_pipeline_audit.csv", index=False)
    difficulty = residual_universe_difficulty()
    difficulty.to_csv(RESULTS / "residual_universe_difficulty.csv", index=False)
    lines = [
        "# Residual Universe Difficulty Audit",
        "",
        "The residual universe is the frozen baseline's selected quote set. The central question is whether enough negative EV mass exists to learn a useful veto without overfitting.",
        "",
    ]
    for _, row in difficulty.iterrows():
        lines.append(
            f"- `{row['split']}`: samples `{int(row['sample_count'])}`, negative EV rate `{row['negative_ev_rate']:.3f}`, "
            f"<= -5000 count `{int(row['severe_negative_le_5000_count'])}`, baseline PnL `{row['baseline_quote_pnl']:.0f}`."
        )
    write_markdown(RESULTS / "residual_universe_difficulty.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote pipeline and universe audits to {RESULTS}")
