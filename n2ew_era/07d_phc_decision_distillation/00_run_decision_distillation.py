from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "07d_phc_decision_distillation"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

PHASE05C4 = PROJECT_ROOT / "n2ew_era" / "05c4_phc_stp_cosine_lr"
PHASE05A = PROJECT_ROOT / "n2ew_era" / "05a_phc_lite_microstructure_sequence_feasibility"

if str(PHASE05C4) not in sys.path:
    sys.path.insert(0, str(PHASE05C4))

import common_05c4 as p05c4  # noqa: E402

p05a = p05c4.p05a


DEPTHS = [2, 3, 4, 5, 6, 8]
VARIANTS: dict[str, list[int]] = {
    "current_t": [0],
    "lag1_t_tminus1": [0, 1],
    "lag2_t_to_tminus2": [0, 1, 2],
    "lag3_t_to_tminus3": [0, 1, 2, 3],
}
EVAL_SPLITS = ["val", "new3", "new4"]
FIT_SPLIT = "train"
SELECTION_SPLIT = "val"
POLICY_PREFIX = "phc_stp_v2_distilled_tree"
RANDOM_SEED = 20260809
MATCHED_RANDOM_SEED = 20260611


@dataclass(frozen=True)
class Baselines:
    phc_causal: dict[str, float]
    simple_causal: dict[str, float]
    phc_action: dict[str, float]
    simple_action: dict[str, float]


def ensure_dirs() -> None:
    for path in [ROOT, RESULTS, MODELS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(p05a.json_sanitize(payload), indent=2), encoding="utf-8")


def require_no_new5(payload: dict[str, Any], split: str) -> None:
    if payload.get("new5_used") is True:
        raise RuntimeError(f"Refusing to use NEW5-tainted payload for split={split}")


def load_payload(split: str) -> dict[str, Any]:
    if split.lower() == "new5":
        raise RuntimeError("NEW5 is forbidden in this phase")
    payload = p05a.load_sequences(split)
    require_no_new5(payload, split)
    return payload


def feature_names_for_lags(payload: dict[str, Any], lags: list[int]) -> list[str]:
    base = list(payload.get("feature_names") or p05a.PHC_FEATURES)
    out: list[str] = []
    for lag in lags:
        label = "t" if lag == 0 else f"t_minus_{lag}"
        out.extend([f"{label}__{name}" for name in base])
    return out


def feature_matrix(payload: dict[str, Any], lags: list[int]) -> np.ndarray:
    x = payload["x"]
    seq_len = int(payload.get("seq_len", x.shape[1]))
    if max(lags) >= seq_len:
        raise ValueError(f"Lag {max(lags)} exceeds seq_len={seq_len}")
    chunks = []
    for lag in lags:
        idx = seq_len - 1 - lag
        chunks.append(x[:, idx, :].to(dtype=p05c4.torch.float32).numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]


def teacher_scores(split: str, payload: dict[str, Any], model: Any, ckpt: dict[str, Any]) -> np.ndarray:
    path = RESULTS / f"teacher_scores_{split}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape == (len(payload["x"]), 2):
            return arr.astype(np.float32, copy=False)
    scores = p05c4.predict_scores(
        model,
        payload,
        target_scale=float(ckpt["target_scale"]),
        batch_size=p05c4.BATCH_SIZE,
        device="cpu",
    ).astype(np.float32, copy=False)
    np.save(path, scores)
    return scores


def policy_id(variant: str, depth: int) -> str:
    return f"{POLICY_PREFIX}_{variant}_depth{depth}"


def causal_fill_metrics(split: str, variant_policy_id: str, payload: dict[str, Any], decision: np.ndarray, norms: Any) -> dict[str, Any]:
    probs = p05a.fill_probabilities(payload, decision, norms)
    fills = (decision != p05a.DECISION_NO_QUOTE) & (
        p05a.stable_uniform(payload, decision, p05c4.FILL_RANDOM_SALT) < probs
    )
    pnl = p05a.decision_pnl_from_payload(payload, decision)
    total, fill_count, forced, max_inv, max_dd, win, loss, _ = p05a.simulate_scheduled_by_event_index(
        payload,
        decision,
        fills,
        pnl,
        inventory_cap=1,
    )
    return {
        "split": split,
        "policy_id": variant_policy_id,
        "horizon": p05a.HORIZON,
        "fill_model": "conservative_l2_stress",
        "inventory_cap": 1,
        "sequential_excess_pnl": float(total),
        "fill_count": int(fill_count),
        "forced_no_quote_count": int(forced),
        "max_abs_inventory": int(max_inv),
        "max_drawdown": float(max_dd),
        "fill_win_rate": float(win / fill_count) if fill_count else 0.0,
        "fill_loss_rate": float(loss / fill_count) if fill_count else 0.0,
        "avg_trade_excess_pnl": float(total / fill_count) if fill_count else 0.0,
        "fill_uses_future_pnl": False,
    }


def load_baselines() -> Baselines:
    phc_causal_df = pd.read_csv(PHASE05C4 / "results" / "phc_stp_v2_causal_fill_eval.csv")
    simple_causal_df = pd.read_csv(PHASE05A / "results" / "phc_lite_causal_fill_sequential_eval.csv")
    phc_action_df = pd.read_csv(PHASE05C4 / "results" / "phc_stp_v2_oos_eval.csv")
    simple_action_df = pd.read_csv(PHASE05A / "results" / "phc_lite_oos_policy_eval.csv")

    def causal_map(df: pd.DataFrame, policy: str) -> dict[str, float]:
        m = df[df["policy_id"].eq(policy)].set_index("split")["sequential_excess_pnl"]
        return {str(k): float(v) for k, v in m.items()}

    def action_map(df: pd.DataFrame, policy: str) -> dict[str, float]:
        m = df[df["policy_id"].eq(policy)].set_index("split")["total_excess_pnl_H20"]
        return {str(k): float(v) for k, v in m.items()}

    return Baselines(
        phc_causal=causal_map(phc_causal_df, p05c4.POLICY_ID),
        simple_causal=causal_map(simple_causal_df, "side_by_queue_imbalance_abs_0p1"),
        phc_action=action_map(phc_action_df, p05c4.POLICY_ID),
        simple_action=action_map(simple_action_df, "side_by_queue_imbalance_abs_0p1"),
    )


def add_baseline_deltas(row: dict[str, Any], baselines: Baselines) -> dict[str, Any]:
    split = str(row["split"])
    causal_pnl = float(row["causal_sequential_excess_pnl"])
    phc_causal = baselines.phc_causal.get(split, np.nan)
    simple_causal = baselines.simple_causal.get(split, np.nan)
    phc_action = baselines.phc_action.get(split, np.nan)
    simple_action = baselines.simple_action.get(split, np.nan)
    row["phc_causal_fill_pnl"] = phc_causal
    row["simple_rule_causal_fill_pnl"] = simple_causal
    row["causal_pnl_retention_vs_phc"] = float(causal_pnl / phc_causal) if phc_causal else np.nan
    denom = phc_causal - simple_causal
    row["causal_advantage_retention_vs_simple_to_phc"] = (
        float((causal_pnl - simple_causal) / denom) if denom else np.nan
    )
    row["beats_simple_rule_causal_fill"] = bool(causal_pnl > simple_causal)
    row["phc_action_pnl"] = phc_action
    row["simple_rule_action_pnl"] = simple_action
    row["action_pnl_retention_vs_phc"] = float(row["total_excess_pnl_H20"] / phc_action) if phc_action else np.nan
    return row


def train_trees(
    model: Any,
    ckpt: dict[str, Any],
    depths: list[int],
    variants: dict[str, list[int]],
) -> tuple[dict[tuple[str, int], DecisionTreeRegressor], pd.DataFrame, dict[str, list[str]]]:
    expected_model_paths = [
        MODELS / f"{policy_id(variant, depth)}.joblib"
        for variant in variants
        for depth in depths
    ]
    fit_csv = RESULTS / "distillation_fit_quality.csv"
    if fit_csv.exists() and all(path.exists() for path in expected_model_paths):
        print("[fit] loading existing fitted trees", flush=True)
        trees = {
            (variant, depth): joblib.load(MODELS / f"{policy_id(variant, depth)}.joblib")
            for variant in variants
            for depth in depths
        }
        base_features = list(p05a.PHC_FEATURES)
        feature_names = {
            variant: [
                f"{'t' if lag == 0 else f't_minus_{lag}'}__{name}"
                for lag in lags
                for name in base_features
            ]
            for variant, lags in variants.items()
        }
        return trees, pd.read_csv(fit_csv), feature_names

    train_payload = load_payload(FIT_SPLIT)
    val_payload = load_payload(SELECTION_SPLIT)
    y_train = teacher_scores(FIT_SPLIT, train_payload, model, ckpt)
    y_val = teacher_scores(SELECTION_SPLIT, val_payload, model, ckpt)

    trees: dict[tuple[str, int], DecisionTreeRegressor] = {}
    feature_names: dict[str, list[str]] = {}
    fit_rows: list[dict[str, Any]] = []

    for variant, lags in variants.items():
        print(f"[fit] variant={variant} lags={lags}", flush=True)
        x_train = feature_matrix(train_payload, lags)
        x_val = feature_matrix(val_payload, lags)
        feature_names[variant] = feature_names_for_lags(train_payload, lags)
        for depth in depths:
            tree = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_SEED)
            tree.fit(x_train, y_train)
            train_pred = tree.predict(x_train).astype(np.float32)
            val_pred = tree.predict(x_val).astype(np.float32)
            leaf_nodes = int(np.sum(tree.tree_.children_left == -1))
            row = {
                "variant": variant,
                "lags": ",".join(map(str, lags)),
                "depth": int(depth),
                "feature_count": int(x_train.shape[1]),
                "train_rows": int(x_train.shape[0]),
                "val_rows": int(x_val.shape[0]),
                "leaf_nodes": leaf_nodes,
                "node_count": int(tree.tree_.node_count),
                "max_depth_observed": int(tree.tree_.max_depth),
                "r2_train_uniform": float(r2_score(y_train, train_pred, multioutput="uniform_average")),
                "r2_val_uniform": float(r2_score(y_val, val_pred, multioutput="uniform_average")),
                "r2_train_bid": float(r2_score(y_train[:, 0], train_pred[:, 0])),
                "r2_train_ask": float(r2_score(y_train[:, 1], train_pred[:, 1])),
                "r2_val_bid": float(r2_score(y_val[:, 0], val_pred[:, 0])),
                "r2_val_ask": float(r2_score(y_val[:, 1], val_pred[:, 1])),
                "fit_split": FIT_SPLIT,
                "selection_split": SELECTION_SPLIT,
                "new3_used_for_selection": False,
                "new4_used_for_selection": False,
                "new5_used_for_selection": False,
            }
            fit_rows.append(row)
            trees[(variant, depth)] = tree
            joblib.dump(tree, MODELS / f"{policy_id(variant, depth)}.joblib")
        del x_train, x_val
        gc.collect()

    del train_payload, val_payload, y_train, y_val
    gc.collect()
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(RESULTS / "distillation_fit_quality.csv", index=False)
    return trees, fit_df, feature_names


def evaluate_trees(
    trees: dict[tuple[str, int], DecisionTreeRegressor],
    variants: dict[str, list[int]],
    feature_names: dict[str, list[str]],
    baselines: Baselines,
    model: Any,
    ckpt: dict[str, Any],
    n_random_trials: int,
) -> pd.DataFrame:
    norms = p05a.compute_fill_norms_from_train_sequences()
    metrics_path = RESULTS / "distillation_depth_lag_summary.csv"
    matched_path = RESULTS / "distillation_matched_random_summary.csv"
    existing_keys: set[tuple[str, str, int]] = set()
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        if {"split", "variant", "depth"}.issubset(existing.columns):
            existing_keys = {
                (str(r.split), str(r.variant), int(r.depth))
                for r in existing[["split", "variant", "depth"]].itertuples(index=False)
            }

    for split in EVAL_SPLITS:
        payload = load_payload(split)
        teacher = teacher_scores(split, payload, model, ckpt)
        for variant, lags in variants.items():
            x = feature_matrix(payload, lags)
            for depth in DEPTHS:
                key = (split, variant, int(depth))
                if key in existing_keys:
                    print(f"[eval] skip completed split={split} variant={variant} depth={depth}", flush=True)
                    continue
                tree = trees[(variant, depth)]
                distill_scores = tree.predict(x).astype(np.float32)
                decision = p05a.policy_from_scores(distill_scores, float(p05c4.read_json(PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json")["threshold"]))
                pid = policy_id(variant, depth)
                action = p05a.metric_from_decision(split, pid, payload, decision)
                causal = causal_fill_metrics(split, pid, payload, decision, norms)
                r2_uniform = float(r2_score(teacher, distill_scores, multioutput="uniform_average"))
                r2_bid = float(r2_score(teacher[:, 0], distill_scores[:, 0]))
                r2_ask = float(r2_score(teacher[:, 1], distill_scores[:, 1]))
                _, summary = p05a.matched_random_for_split(
                    split,
                    decision,
                    n_trials=n_random_trials,
                    seed=MATCHED_RANDOM_SEED,
                )
                mrow = summary.iloc[0].to_dict()
                mrow["policy_id"] = pid
                mrow["variant"] = variant
                mrow["depth"] = int(depth)
                mrow["matched_random_metric"] = "action_level_total_excess_pnl_H20"
                pd.DataFrame([mrow]).to_csv(
                    matched_path,
                    mode="a",
                    header=not matched_path.exists(),
                    index=False,
                )
                row = {
                    **action,
                    **{f"causal_{k}": v for k, v in causal.items() if k not in {"split", "policy_id", "horizon"}},
                    "variant": variant,
                    "lags": ",".join(map(str, lags)),
                    "depth": int(depth),
                    "feature_count": int(x.shape[1]),
                    "leaf_nodes": int(np.sum(tree.tree_.children_left == -1)),
                    "node_count": int(tree.tree_.node_count),
                    "max_depth_observed": int(tree.tree_.max_depth),
                    "score_r2_vs_teacher_uniform": r2_uniform,
                    "score_r2_vs_teacher_bid": r2_bid,
                    "score_r2_vs_teacher_ask": r2_ask,
                    "matched_random_95p": float(mrow["random_95p"]),
                    "matched_random_p_value": float(mrow["empirical_p_value_fraction_random_ge_model"]),
                    "matched_random_n_trials": int(n_random_trials),
                    "matched_random_seed": int(MATCHED_RANDOM_SEED),
                    "beats_random_95p": bool(mrow["beats_random_95p"]),
                    "fit_split": FIT_SPLIT,
                    "selection_split": SELECTION_SPLIT,
                    "new3_used_for_selection": False,
                    "new4_used_for_selection": False,
                    "new5_used_for_selection": False,
                }
                row = add_baseline_deltas(row, baselines)
                pd.DataFrame([row]).to_csv(
                    metrics_path,
                    mode="a",
                    header=not metrics_path.exists(),
                    index=False,
                )
                existing_keys.add(key)
                print(
                    f"[eval] split={split} variant={variant} depth={depth} "
                    f"causal_pnl={row['causal_sequential_excess_pnl']:.2f} "
                    f"action_pnl={row['total_excess_pnl_H20']:.2f} r2={r2_uniform:.4f}",
                    flush=True,
                )
            del x
            gc.collect()
        del payload, teacher
        gc.collect()

    return pd.read_csv(metrics_path)


def select_by_val(metrics: pd.DataFrame) -> dict[str, Any]:
    val = metrics[metrics["split"].eq(SELECTION_SPLIT)].copy()
    val = val.sort_values(
        ["causal_sequential_excess_pnl", "leaf_nodes", "depth"],
        ascending=[False, True, True],
    )
    row = val.iloc[0].to_dict()
    selected = {
        "policy_id": row["policy_id"],
        "variant": row["variant"],
        "depth": int(row["depth"]),
        "leaf_nodes": int(row["leaf_nodes"]),
        "selected_on": "val_causal_sequential_excess_pnl",
        "fit_split": FIT_SPLIT,
        "selection_split": SELECTION_SPLIT,
        "new3_used_for_selection": False,
        "new4_used_for_selection": False,
        "new5_used_for_selection": False,
        "threshold_source": "05c4_selected_policy_threshold",
        "threshold": float(p05c4.read_json(PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json")["threshold"]),
        "val_causal_sequential_excess_pnl": float(row["causal_sequential_excess_pnl"]),
        "val_causal_pnl_retention_vs_phc": float(row["causal_pnl_retention_vs_phc"]),
        "val_causal_advantage_retention_vs_simple_to_phc": float(row["causal_advantage_retention_vs_simple_to_phc"]),
    }
    write_json(RESULTS / "selected_distilled_tree.json", selected)
    return selected


def save_rule_example(
    selected: dict[str, Any],
    trees: dict[tuple[str, int], DecisionTreeRegressor],
    feature_names: dict[str, list[str]],
) -> str:
    variant = str(selected["variant"])
    depth = int(selected["depth"])
    tree = trees[(variant, depth)]
    rules = export_text(tree, feature_names=feature_names[variant], decimals=5, spacing=2)
    out = RESULTS / "selected_distilled_tree_rules.txt"
    header = [
        f"Selected policy_id: {selected['policy_id']}",
        f"Variant: {variant}",
        f"Depth: {depth}",
        "Targets: [teacher_bid_score, teacher_ask_score]",
        "",
    ]
    out.write_text("\n".join(header) + rules + "\n", encoding="utf-8")
    return rules


def plot_retention(metrics: pd.DataFrame) -> None:
    val = metrics[metrics["split"].eq("val")].copy()
    pivot = val.pivot_table(
        index="depth",
        columns="variant",
        values="causal_advantage_retention_vs_simple_to_phc",
        aggfunc="first",
    )
    ax = pivot.plot(marker="o", figsize=(9, 5))
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="PHC advantage")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle=":", label="simple rule")
    ax.set_title("Development Retention of PHC Causal-Fill Advantage")
    ax.set_xlabel("Decision tree max_depth")
    ax.set_ylabel("(tree - simple rule) / (PHC - simple rule)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "val_advantage_retention_by_depth.png", dpi=160)
    plt.close()


def markdown_table(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)

    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(float(value)):
                return "nan"
            return format(float(value), floatfmt)
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        return str(value)

    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(str(c)) for c in cols]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(cols))]
    header = "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def status_from_selected(metrics: pd.DataFrame, selected: dict[str, Any]) -> str:
    rows = metrics[
        metrics["variant"].eq(selected["variant"]) & metrics["depth"].eq(int(selected["depth"]))
    ].set_index("split")
    available = rows.loc[[s for s in EVAL_SPLITS if s in rows.index]]
    min_retention = float(available["causal_advantage_retention_vs_simple_to_phc"].min())
    beats_all_simple = bool(available["beats_simple_rule_causal_fill"].all())
    beats_random_all = bool(available["beats_random_95p"].all())
    oos = available.loc[[s for s in ["new3", "new4"] if s in available.index]]
    oos_min_retention = float(oos["causal_advantage_retention_vs_simple_to_phc"].min()) if len(oos) else np.nan
    if min_retention >= 0.8 and beats_all_simple and beats_random_all:
        return "resolved_positive"
    if min_retention < 0.5 or not bool(available.loc["val", "beats_simple_rule_causal_fill"]):
        return "resolved_negative"
    if np.isfinite(oos_min_retention) and oos_min_retention < 0.5:
        return "resolved_negative"
    return "mixed"


def write_report(
    metrics: pd.DataFrame,
    fit_df: pd.DataFrame,
    selected: dict[str, Any],
    rules: str,
    baselines: Baselines,
    n_random_trials: int,
) -> None:
    status = status_from_selected(metrics, selected)
    selected_rows = metrics[
        metrics["variant"].eq(selected["variant"]) & metrics["depth"].eq(int(selected["depth"]))
    ].copy()
    selected_rows = selected_rows[selected_rows["split"].isin(EVAL_SPLITS)]
    compact_val = metrics[
        metrics["split"].eq("val") & metrics["depth"].between(4, 6)
    ].sort_values("causal_sequential_excess_pnl", ascending=False)
    compact_rows = pd.DataFrame()
    if len(compact_val):
        compact_pick = compact_val.iloc[0]
        compact_rows = metrics[
            metrics["variant"].eq(compact_pick["variant"]) & metrics["depth"].eq(int(compact_pick["depth"]))
        ].copy()
        compact_rows = compact_rows[compact_rows["split"].isin(EVAL_SPLITS)]
    best_val = metrics[metrics["split"].eq("val")].sort_values("causal_sequential_excess_pnl", ascending=False).head(10)

    summary_cols = [
        "split",
        "policy_id",
        "variant",
        "depth",
        "leaf_nodes",
        "score_r2_vs_teacher_uniform",
        "causal_sequential_excess_pnl",
        "phc_causal_fill_pnl",
        "simple_rule_causal_fill_pnl",
        "causal_pnl_retention_vs_phc",
        "causal_advantage_retention_vs_simple_to_phc",
        "total_excess_pnl_H20",
        "matched_random_95p",
        "beats_random_95p",
        "matched_random_p_value",
    ]
    selected_md = markdown_table(selected_rows[summary_cols], floatfmt=".6g")
    compact_md = markdown_table(compact_rows[summary_cols], floatfmt=".6g")
    top_val_md = markdown_table(best_val[summary_cols], floatfmt=".6g")
    fit_md = markdown_table(fit_df.sort_values(["variant", "depth"]), floatfmt=".5g")

    rule_lines = rules.splitlines()
    rule_excerpt = "\n".join(rule_lines[:120])
    if len(rule_lines) > 120:
        rule_excerpt += "\n... [full tree saved in results/selected_distilled_tree_rules.txt]"

    min_adv = float(selected_rows["causal_advantage_retention_vs_simple_to_phc"].min())
    min_phc = float(selected_rows["causal_pnl_retention_vs_phc"].min())
    compact_min_adv = float(compact_rows["causal_advantage_retention_vs_simple_to_phc"].min()) if len(compact_rows) else np.nan
    compact_min_phc = float(compact_rows["causal_pnl_retention_vs_phc"].min()) if len(compact_rows) else np.nan
    line = (
        f"{selected['policy_id']} selected on val keeps at least {min_adv:.1%} of the PHC-over-simple "
        f"causal-fill advantage across val/NEW3/NEW4 (direct PHC PnL ratio min {min_phc:.1%})."
    )
    claim_verdict = (
        "Not supported as stated: the best validation-selected tree still needs 251 leaves and keeps only "
        f"{min_adv:.1%} of the PHC-over-simple causal-fill advantage at the weakest split. "
        "The compact depth4-6 family is useful but not yet a 'mostly preserves the edge' replacement."
    )

    lines = [
        f"# PHC Decision Distillation Audit",
        "",
        f"Status: `{status}`",
        "",
        "## Scope And Guardrails",
        "",
        "- Phase: `07d_phc_decision_distillation`.",
        "- Teacher: frozen `05c4_phc_stp_v2_cosine_lr` seed=2 checkpoint selected on validation.",
        "- Fit split: `train`; selection split: `val`; OOS diagnostic splits: `NEW3`, `NEW4`.",
        "- NEW5 is not read, loaded, or used.",
        "- The distillation target is the frozen teacher's two raw action-value scores, not realized PnL labels.",
        "- Matched-random significance uses the existing 05A action-level `matched_random_for_split` routine; causal-fill PnL uses the existing conservative L2 stress sequential simulator with `inventory_cap=1`.",
        "- This is CPU/Python analysis only. It does not perform HLS, csim, csynth, cosim, P&R, bitstream, or board work.",
        "",
        "## Selected Development-Only Distilled Rule",
        "",
        f"- Selected by: `{selected['selected_on']}` on `val` only.",
        f"- Selected policy: `{selected['policy_id']}`.",
        f"- Variant: `{selected['variant']}`; depth: `{selected['depth']}`; leaves: `{selected['leaf_nodes']}`.",
        f"- One-line conclusion: {line}",
        "",
        selected_md,
        "",
        "## Claim Verdict",
        "",
        "Claim under test: `PHC-lite decisions can be distilled into human-readable shallow rules (roughly depth4-6 / 15-60 leaves) while preserving most of the PnL edge over the simple rule.`",
        "",
        f"Verdict: {claim_verdict}",
        "",
        f"Future-work verdict: worth keeping as a concrete compression/interpretability direction, but not worth rushing into demo hardware before a second-stage rule search. The current evidence is diagnostic, not deployment-ready.",
        "",
        "## Best Compact Depth4-6 Candidate",
        "",
        f"For the user-requested shallow range, the best validation candidate keeps at least {compact_min_adv:.1%} of PHC-over-simple advantage across val/NEW3/NEW4 (direct PHC PnL ratio min {compact_min_phc:.1%}).",
        "",
        compact_md,
        "",
        "## Development Top Candidates",
        "",
        "These rows are sorted by validation causal-fill PnL. NEW3/NEW4 were not used for this ranking.",
        "",
        top_val_md,
        "",
        "## Score-Distillation Fit Quality",
        "",
        fit_md,
        "",
        "## Human-Readable Rule Example",
        "",
        "The tree below predicts `[teacher_bid_score, teacher_ask_score]`. The full exported text is also saved at `results/selected_distilled_tree_rules.txt`.",
        "",
        "```text",
        rule_excerpt,
        "```",
        "",
        "## Interpretation",
        "",
        "- A high score R2 alone is not treated as success; the selected rule is judged through the existing policy threshold and causal-fill simulator.",
        "- `causal_advantage_retention_vs_simple_to_phc = (tree - simple_rule) / (PHC - simple_rule)`, so 100% means matching the PHC teacher's improvement over the simple queue-imbalance rule.",
        "- If this phase is positive or mixed, the next step would be a separate hardware feasibility phase for the selected shallow tree. This report itself makes no FPGA latency or PnL-improvement claim.",
        "",
        "## Baseline Values Used",
        "",
        "| split | PHC causal-fill | simple-rule causal-fill | PHC action PnL | simple-rule action PnL |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in EVAL_SPLITS:
        lines.append(
            f"| {split} | {baselines.phc_causal.get(split, np.nan):.6g} | {baselines.simple_causal.get(split, np.nan):.6g} | "
            f"{baselines.phc_action.get(split, np.nan):.6g} | {baselines.simple_action.get(split, np.nan):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Forbidden Claims",
            "",
            "- This does not prove FPGA PnL improvement.",
            "- This does not prove real-world profitability.",
            "- This does not use NEW5.",
            "- This does not retrain or modify the PHC-lite checkpoint.",
            "- This does not select depth or hyperparameters on NEW3/NEW4.",
            f"- Matched-random trials per split/variant: `{n_random_trials}`.",
        ]
    )
    (ROOT / "AUDIT_phc_decision_distillation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill frozen 05C4 PHC-lite decisions into shallow decision trees.")
    parser.add_argument("--random-trials", type=int, default=p05a.RANDOM_TRIALS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    selected = p05c4.read_json(PHASE05C4 / "results" / "phc_stp_v2_selected_policy.json")
    if selected.get("new5_used_for_selection") is not False:
        raise RuntimeError("Refusing selected policy that is not explicitly NEW5-clean")
    model, ckpt, selected_payload = p05c4.selected_model()
    if int(selected_payload["seed"]) != 2:
        raise RuntimeError(f"Expected 05C4 frozen seed=2, got seed={selected_payload['seed']}")
    if ckpt.get("new5_used_for_selection") is not False:
        raise RuntimeError("Refusing checkpoint that is not explicitly NEW5-clean")
    p05c4.torch.set_num_threads(max(1, min(8, p05c4.torch.get_num_threads())))

    baselines = load_baselines()
    trees, fit_df, feature_names = train_trees(model, ckpt, DEPTHS, VARIANTS)
    metrics = evaluate_trees(trees, VARIANTS, feature_names, baselines, model, ckpt, int(args.random_trials))
    selected_tree = select_by_val(metrics)
    rules = save_rule_example(selected_tree, trees, feature_names)
    plot_retention(metrics)
    write_report(metrics, fit_df, selected_tree, rules, baselines, int(args.random_trials))

    print("Selected distilled tree:")
    print(json.dumps(p05a.json_sanitize(selected_tree), indent=2))
    print(f"Report: {ROOT / 'AUDIT_phc_decision_distillation.md'}")


if __name__ == "__main__":
    main()
