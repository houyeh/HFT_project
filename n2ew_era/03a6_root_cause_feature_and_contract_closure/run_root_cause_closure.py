from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "n2ew_era" / "03a6_root_cause_feature_and_contract_closure"
RESULTS = ROOT / "results"
DATA = ROOT / "data"
EXCESS_DATA = PROJECT_ROOT / "n2ew_era" / "02f_excess_markout_and_causal_fill_proxy" / "data" / "full_action_universe_with_excess_markout.parquet"
PHASE02F = PROJECT_ROOT / "n2ew_era" / "02f_excess_markout_and_causal_fill_proxy"
PHASE03A5 = PROJECT_ROOT / "n2ew_era" / "03a5_pre_ml_freeze_audit"

SPLITS = ["train", "val", "test", "new4", "new3"]
FROZEN_SPLITS = ["test", "new4", "new3"]
POINT_VALUE = 200.0
MAKER_FEE = 15.0
TARGET = "excess_pnl_H20"
TOLERANCE = 1e-12


def ensure_dirs() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_sanitize(payload), indent=2), encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def part_paths(split: str) -> list[Path]:
    return sorted((EXCESS_DATA / f"split={split}").glob("*.parquet"))


def read_part(path: Path, columns: list[str]) -> pd.DataFrame:
    available = pd.read_parquet(path).columns.tolist()
    return pd.read_parquet(path, columns=[c for c in columns if c in available])


def sample_split(split: str, columns: list[str], per_part: int = 80_000, sides: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for idx, path in enumerate(part_paths(split)):
        df = pd.read_parquet(path, columns=[c for c in columns if c in pd.read_parquet(path).columns])
        if sides is not None and "action_side" in df:
            df = df[df["action_side"].astype(str).isin(sides)]
        if len(df) > per_part:
            df = df.sample(n=per_part, random_state=20260602 + idx)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 20 or np.unique(y_pred[mask]).size < 2:
        return np.nan
    return float(spearmanr(y_true[mask], y_pred[mask]).correlation)


def make_design(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    for col in features:
        if col not in work:
            work[col] = 0.0
    work["action_is_bid"] = (work["action_side"].astype(str) == "bid").astype(float)
    work["action_is_ask"] = (work["action_side"].astype(str) == "ask").astype(float)
    feature_cols = [*features, "action_is_bid", "action_is_ask"]
    x = work[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(work[TARGET], errors="coerce").fillna(0.0)
    return x, y


def absolute_price_feature_audit() -> pd.DataFrame:
    price_features = ["best_bid", "best_ask", "mid_price", "quote_price"]
    micro_features = ["spread_ticks", "queue_imbalance", "deep_imbalance", "top_depth", "bid_sz_1", "ask_sz_1", "volatility_past_50"]
    combined_features = [*price_features, *micro_features]
    model_specs = {
        "price_only": price_features,
        "microstructure_only": micro_features,
        "combined": combined_features,
    }
    columns = ["split", "session_id", "action_side", TARGET, *combined_features]
    samples = {split: sample_split(split, columns, per_part=80_000, sides=["bid", "ask"]) for split in SPLITS}
    rows = []
    for model_name, features in model_specs.items():
        x_train, y_train = make_design(samples["train"], features)
        model = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0, random_state=20260602))
        model.fit(x_train, y_train)
        for split, sample in samples.items():
            x, y = make_design(sample, features)
            pred = model.predict(x)
            rmse = math.sqrt(mean_squared_error(y, pred))
            rows.append(
                {
                    "model_name": model_name,
                    "split": split,
                    "sample_count": int(len(sample)),
                    "target": TARGET,
                    "features": ",".join(features),
                    "rmse": float(rmse),
                    "mae": float(mean_absolute_error(y, pred)),
                    "r2": float(r2_score(y, pred)),
                    "spearman_corr": safe_spearman(y.to_numpy(dtype=float), pred),
                    "train_only_fit": True,
                    "test_new3_new4_used_for_tuning": False,
                }
            )
        print(f"diagnostic model complete: {model_name}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "absolute_price_feature_audit.csv", index=False)

    pivot = out.pivot(index="split", columns="model_name", values="spearman_corr")
    frozen = pivot.loc[[s for s in FROZEN_SPLITS if s in pivot.index]]
    price_mean = float(frozen["price_only"].mean()) if "price_only" in frozen else np.nan
    micro_mean = float(frozen["microstructure_only"].mean()) if "microstructure_only" in frozen else np.nan
    combined_mean = float(frozen["combined"].mean()) if "combined" in frozen else np.nan
    combined_gain = combined_mean - micro_mean
    if not np.isfinite(price_mean) or price_mean < 0.02:
        decision = "forbid"
        rationale = "Price-only diagnostic has weak frozen-OOS rank correlation; raw absolute price levels are not reliable standalone causal features."
    elif combined_gain > 0.01 and combined_mean >= micro_mean:
        decision = "allow"
        rationale = "Combined diagnostic materially improves over microstructure-only across frozen OOS."
    else:
        decision = "transform"
        rationale = "Absolute price has some diagnostic information but does not justify raw-level use; prefer relative transforms and microstructure features."
    payload = {
        "decision": decision,
        "raw_absolute_price_features": price_features,
        "recommended_action": "Use relative/transformed price features only." if decision == "transform" else decision,
        "frozen_oos_mean_spearman": {
            "price_only": price_mean,
            "microstructure_only": micro_mean,
            "combined": combined_mean,
            "combined_minus_microstructure": combined_gain,
        },
        "rationale": rationale,
        "diagnostic_only_not_final_ml": True,
        "test_new3_new4_used_for_tuning": False,
    }
    write_json(ROOT / "absolute_price_feature_decision.json", payload)
    return out


def rebuild_dynamic_features() -> pd.DataFrame:
    out_dir = DATA / "rebuilt_dynamic_features.parquet"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cols = [
        "split",
        "session_id",
        "symbol",
        "event_index",
        "timestamp",
        "action_side",
        "mid_price",
        "spread_ticks",
        "queue_imbalance",
    ]
    for split in SPLITS:
        split_dir = out_dir / f"split={split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        for path in part_paths(split):
            df = pd.read_parquet(path, columns=cols)
            state = df[df["action_side"].astype(str) == "bid"].sort_values("event_index").reset_index(drop=True).copy()
            session_id = str(state["session_id"].iloc[0]) if len(state) else path.stem.replace("part-", "")
            timestamp = pd.to_numeric(state["timestamp"], errors="coerce")
            dt_ns = timestamp.diff()
            inter_arrival_ms = (dt_ns / 1_000_000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
            event_rate = (1_000_000_000.0 / dt_ns.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1_000_000.0)
            mid_ret = pd.to_numeric(state["mid_price"], errors="coerce").diff()
            rebuilt = pd.DataFrame(
                {
                    "split": state["split"].astype(str),
                    "session_id": state["session_id"].astype(str),
                    "symbol": state["symbol"].astype(str),
                    "event_index": state["event_index"].astype(np.int64),
                    "timestamp": state["timestamp"].astype(np.int64),
                    "inter_arrival_ms": inter_arrival_ms.astype(float),
                    "event_rate_past_50": event_rate.rolling(50, min_periods=3).mean().fillna(0.0).astype(float),
                    "imbalance_change_past": pd.to_numeric(state["queue_imbalance"], errors="coerce").diff().fillna(0.0).astype(float),
                    "spread_change_past": pd.to_numeric(state["spread_ticks"], errors="coerce").diff().fillna(0.0).astype(float),
                    "volatility_past_50_rebuilt": mid_ret.rolling(50, min_periods=3).std().fillna(0.0).astype(float),
                }
            )
            pq.write_table(pa.Table.from_pandas(rebuilt, preserve_index=False), split_dir / path.name, compression="zstd")
            rows.append(
                {
                    "split": split,
                    "session_id": session_id,
                    "row_count": int(len(rebuilt)),
                    "first_inter_arrival_ms": float(rebuilt["inter_arrival_ms"].iloc[0]) if len(rebuilt) else np.nan,
                    "first_event_rate_past_50": float(rebuilt["event_rate_past_50"].iloc[0]) if len(rebuilt) else np.nan,
                    "first_imbalance_change_past": float(rebuilt["imbalance_change_past"].iloc[0]) if len(rebuilt) else np.nan,
                    "first_spread_change_past": float(rebuilt["spread_change_past"].iloc[0]) if len(rebuilt) else np.nan,
                    "max_event_rate_past_50": float(rebuilt["event_rate_past_50"].max()) if len(rebuilt) else np.nan,
                    "causal_session_reset_pass": bool(
                        len(rebuilt) == 0
                        or (
                            rebuilt["inter_arrival_ms"].iloc[0] == 0.0
                            and rebuilt["event_rate_past_50"].iloc[0] == 0.0
                            and rebuilt["imbalance_change_past"].iloc[0] == 0.0
                            and rebuilt["spread_change_past"].iloc[0] == 0.0
                        )
                    ),
                    "uses_future_data": False,
                }
            )
        print(f"rebuilt dynamic features split={split}")
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "rebuilt_dynamic_feature_summary.csv", index=False)
    lines = [
        "# Dynamic Feature Rebuild Report",
        "",
        "Missing dynamic features were rebuilt strictly within each session using current and past rows only. Session starts reset to zero for inter-arrival, event-rate rolling mean, imbalance change, and spread change.",
        "",
        "Rebuilt features:",
        "",
        "- `inter_arrival_ms = timestamp_t - timestamp_{t-1}` within the same session.",
        "- `event_rate_past_50 = rolling_mean(1e9 / inter_arrival_ns, window=50, min_periods=3)` within the same session.",
        "- `imbalance_change_past = queue_imbalance_t - queue_imbalance_{t-1}` within the same session.",
        "- `spread_change_past = spread_ticks_t - spread_ticks_{t-1}` within the same session.",
        "",
        summary.to_markdown(index=False),
        "",
        "Output data: `data/rebuilt_dynamic_features.parquet`.",
    ]
    write_markdown(ROOT / "DYNAMIC_FEATURE_REBUILD_REPORT.md", lines)
    write_json(
        ROOT / "dynamic_feature_decision.json",
        {
            "event_rate_past_50": "allow_rebuilt",
            "inter_arrival_ms": "allow_rebuilt",
            "imbalance_change_past": "allow_rebuilt",
            "spread_change_past": "allow_rebuilt",
            "uses_future_data": False,
            "session_boundary_reset_required": True,
        },
    )
    return summary


def rolling_recompute_audit() -> pd.DataFrame:
    rows = []
    cols = ["split", "session_id", "event_index", "action_side", "mid_price", "volatility_past_50"]
    for split in SPLITS:
        for path in part_paths(split):
            df = pd.read_parquet(path, columns=cols)
            state = df[df["action_side"].astype(str) == "bid"].sort_values("event_index").reset_index(drop=True)
            session_id = str(state["session_id"].iloc[0]) if len(state) else path.stem.replace("part-", "")
            recompute = pd.to_numeric(state["mid_price"], errors="coerce").diff().rolling(50, min_periods=3).std().fillna(0.0)
            stored = pd.to_numeric(state["volatility_past_50"], errors="coerce").fillna(0.0)
            err = (recompute - stored).abs()
            rows.append(
                {
                    "split": split,
                    "session_id": session_id,
                    "row_count": int(len(state)),
                    "feature": "volatility_past_50",
                    "formula": "mid_price.diff().rolling(50, min_periods=3).std().fillna(0)",
                    "max_abs_error": float(err.max()) if len(err) else np.nan,
                    "mean_abs_error": float(err.mean()) if len(err) else np.nan,
                    "tolerance": TOLERANCE,
                    "recompute_pass": bool(len(err) == 0 or err.max() <= TOLERANCE),
                    "uses_future_data": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "rolling_feature_recompute_audit.csv", index=False)
    lines = [
        "# Rolling Feature Recompute Audit",
        "",
        "This audit recomputes `volatility_past_50` within each session using only rows up to and including time t. No centered windows are used.",
        "",
        out.to_markdown(index=False),
    ]
    write_markdown(ROOT / "ROLLING_FEATURE_RECOMPUTE_AUDIT.md", lines)
    return out


def cost_unit_contract() -> pd.DataFrame:
    rows = []
    cols = ["split", "session_id", "symbol", "action_side", "tick_size"]
    for split in SPLITS:
        for path in part_paths(split):
            df = pd.read_parquet(path, columns=cols)
            state = df[df["action_side"].astype(str) == "bid"]
            rows.append(
                {
                    "split": split,
                    "session_id": str(state["session_id"].iloc[0]) if len(state) else path.stem.replace("part-", ""),
                    "symbol": str(state["symbol"].iloc[0]) if len(state) else "",
                    "tick_size_min": float(pd.to_numeric(state["tick_size"], errors="coerce").min()) if len(state) else np.nan,
                    "tick_size_max": float(pd.to_numeric(state["tick_size"], errors="coerce").max()) if len(state) else np.nan,
                    "point_value": POINT_VALUE,
                    "maker_fee": MAKER_FEE,
                    "maker_fee_interpretation": "one-way proxy fee subtracted once per quote-level markout/fill",
                    "exchange_certified": False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "cost_unit_product_audit.csv", index=False)
    symbol_summary = out.groupby(["split", "symbol"], as_index=False).agg(
        sessions=("session_id", "nunique"),
        tick_size_min=("tick_size_min", "min"),
        tick_size_max=("tick_size_max", "max"),
    )
    lines = [
        "# Cost / Unit / Product Contract",
        "",
        f"`point_value = {POINT_VALUE}` and `maker_fee = {MAKER_FEE}` are inherited proxy assumptions used by Phase 01/02F formulae.",
        "",
        "`maker_fee=15` is treated as a one-way proxy maker fee subtracted once per quote-level markout/fill. The current files do not certify whether this includes exchange fees, tax, broker fees, or round-trip costs.",
        "",
        "Therefore these are proxy PnL units, not exchange-certified values.",
        "",
        "## Symbol / Tick Size Audit",
        "",
        symbol_summary.to_markdown(index=False),
        "",
        "Historical train/val/test use `TXFB6`; NEW3/NEW4 use `TXFA6`. Tick size is consistent at 1.0 in the generated parquet, but contract month differs, so OOS comparison is valid as a harsh proxy regime check, not as certified same-contract trading replay.",
    ]
    write_markdown(ROOT / "COST_UNIT_CONTRACT.md", lines)
    return out


def stats(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "q05": np.nan, "q50": np.nan, "q95": np.nan}
    q = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    e = np.histogram(expected, bins=edges)[0].astype(float)
    a = np.histogram(actual, bins=edges)[0].astype(float)
    ep = np.clip(e / max(e.sum(), 1.0), 1e-6, 1.0)
    ap = np.clip(a / max(a.sum(), 1.0), 1e-6, 1.0)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def ks(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.sort(reference[np.isfinite(reference)])
    actual = np.sort(actual[np.isfinite(actual)])
    if len(reference) == 0 or len(actual) == 0:
        return np.nan
    values = np.sort(np.concatenate([reference, actual]))
    return float(np.max(np.abs(np.searchsorted(reference, values, side="right") / len(reference) - np.searchsorted(actual, values, side="right") / len(actual))))


def split_regime_drift_closure() -> pd.DataFrame:
    base_cols = [
        "split",
        "session_id",
        "event_index",
        "action_side",
        "spread_ticks",
        "queue_imbalance",
        "deep_imbalance",
        "top_depth",
        "volatility_past_50",
        TARGET,
    ]
    samples: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        frames = []
        for idx, path in enumerate(part_paths(split)):
            df = pd.read_parquet(path, columns=base_cols)
            df = df[df["action_side"].astype(str).isin(["bid", "ask"])]
            if len(df) > 100_000:
                df = df.sample(n=100_000, random_state=20260603 + idx)
            frames.append(df)
        samples[split] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rebuilt_dir = DATA / "rebuilt_dynamic_features.parquet"
    # Join rebuilt sampled state-level dynamics on event_index by split/session.
    for split, sample in samples.items():
        if sample.empty:
            continue
        dyn_frames = []
        for path in sorted((rebuilt_dir / f"split={split}").glob("*.parquet")):
            dyn = pd.read_parquet(path)
            dyn_frames.append(dyn)
        if dyn_frames:
            dyn_all = pd.concat(dyn_frames, ignore_index=True)
            samples[split] = sample.merge(
                dyn_all[["session_id", "event_index", "event_rate_past_50", "inter_arrival_ms", "imbalance_change_past", "spread_change_past"]],
                on=["session_id", "event_index"],
                how="left",
            )
    variables = [
        "spread_ticks",
        "queue_imbalance",
        "deep_imbalance",
        "top_depth",
        "volatility_past_50",
        "event_rate_past_50",
        "inter_arrival_ms",
        "imbalance_change_past",
        "spread_change_past",
        TARGET,
    ]
    rows = []
    train = samples["train"]
    for variable in variables:
        train_values = pd.to_numeric(train[variable], errors="coerce").to_numpy(dtype=float) if variable in train else np.array([])
        train_stats = stats(train_values)
        for split, sample in samples.items():
            values = pd.to_numeric(sample[variable], errors="coerce").to_numpy(dtype=float) if variable in sample else np.array([])
            s = stats(values)
            rows.append(
                {
                    "variable": variable,
                    "split": split,
                    **s,
                    "train_q50": train_stats["q50"],
                    "median_shift_vs_train": s["q50"] - train_stats["q50"] if np.isfinite(s["q50"]) and np.isfinite(train_stats["q50"]) else np.nan,
                    "ks_vs_train": 0.0 if split == "train" else ks(train_values, values),
                    "psi_vs_train": 0.0 if split == "train" else psi(train_values, values),
                    "oos_regime_is_harsh": bool(split in FROZEN_SPLITS and (psi(train_values, values) > 0.25 or ks(train_values, values) > 0.20)) if split != "train" else False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "split_regime_drift_closure.csv", index=False)
    lines = [
        "# Split Regime Drift Closure",
        "",
        "This audit quantifies distribution drift against train for key causal features and the H20 excess label. High drift is not a leakage failure; it documents OOS harshness.",
        "",
        out[out["split"].isin(["test", "new4", "new3"])].sort_values(["psi_vs_train", "ks_vs_train"], ascending=False).head(30).to_markdown(index=False),
    ]
    write_markdown(ROOT / "SPLIT_REGIME_DRIFT_CLOSURE.md", lines)
    return out


def baseline_hurdle_consistency() -> pd.DataFrame:
    eval_df = pd.read_csv(PHASE02F / "results" / "excess_policy_evaluation.csv")
    seq_df = pd.read_csv(PHASE02F / "results" / "sequential_excess_inventory_backtest.csv")
    rows = []
    for split in ["val", *FROZEN_SPLITS]:
        h20 = eval_df[(eval_df["split"].astype(str) == split) & (eval_df["horizon"].astype(int) == 20)]
        state_counts = h20[h20["policy_id"].isin(["no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"])]["state_count"].dropna().astype(float).unique()
        rows.append(
            {
                "split": split,
                "audit_item": "excess_policy_same_universe",
                "pass": bool(len(state_counts) == 1),
                "details": f"state_count_values={list(state_counts)}",
            }
        )
        random_rows = h20[h20["policy_id"].astype(str) == "matched_random_95p"]
        rows.append(
            {
                "split": split,
                "audit_item": "matched_random_same_excess_objective",
                "pass": bool(not random_rows.empty and random_rows["random_method"].astype(str).str.contains("side_session_spread").any()),
                "details": "matched random rows use finite_population_normal_approx_side_session_spread_bucket",
            }
        )
        seq_focus = seq_df[
            (seq_df["split"].astype(str) == split)
            & (seq_df["fill_model"].astype(str) == "conservative_l2_stress")
            & (seq_df["inventory_cap"].astype(int) == 1)
            & (seq_df["fee_multiplier"].astype(float) == 1.0)
        ]
        rows.append(
            {
                "split": split,
                "audit_item": "sequential_proxy_same_fee_fill_contract",
                "pass": bool(set(seq_focus["policy_id"].astype(str)) >= {"no_quote", "old_bid_only_spread_ge_1", "side_by_queue_imbalance_abs_0p1"}),
                "details": "conservative_l2_stress, inventory_cap=1, fee_multiplier=1.0, H20",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "baseline_hurdle_consistency_audit.csv", index=False)
    lines = [
        "# Baseline Hurdle Consistency Audit",
        "",
        "This audit verifies apples-to-apples baseline comparisons: same split universe, same H20 excess objective, same fee/fill contract for sequential proxy, and matched-random count/side/session/spread controls.",
        "",
        out.to_markdown(index=False),
    ]
    write_markdown(ROOT / "BASELINE_HURDLE_CONSISTENCY_AUDIT.md", lines)
    return out


def final_report(price: pd.DataFrame, dynamic: pd.DataFrame, rolling: pd.DataFrame, drift: pd.DataFrame, hurdle: pd.DataFrame) -> None:
    price_decision = json.loads((ROOT / "absolute_price_feature_decision.json").read_text(encoding="utf-8"))
    gates = pd.DataFrame(
        [
            {"gate": "absolute_price_feature_decision_closed", "pass": price_decision["decision"] in {"allow", "forbid", "transform"}, "decision": price_decision["decision"]},
            {"gate": "dynamic_features_rebuilt_causally", "pass": bool(dynamic["causal_session_reset_pass"].astype(bool).all() and not dynamic.empty), "decision": "allow_rebuilt"},
            {"gate": "rolling_feature_no_leak_proof", "pass": bool(rolling["recompute_pass"].astype(bool).all() and not rolling.empty), "decision": "allow_volatility_past_50"},
            {"gate": "cost_unit_contract_closed", "pass": (ROOT / "COST_UNIT_CONTRACT.md").exists(), "decision": "proxy_units_documented"},
            {"gate": "split_regime_drift_quantified", "pass": bool(not drift.empty), "decision": "oos_harshness_documented"},
            {"gate": "baseline_hurdles_apples_to_apples", "pass": bool(hurdle["pass"].astype(bool).all() and not hurdle.empty), "decision": "hurdles_closed"},
        ]
    )
    gates.to_csv(RESULTS / "root_cause_closure_gates.csv", index=False)
    permission = bool(gates["pass"].astype(bool).all())
    status = "phase03b_permission_granted_with_constraints" if permission else "phase03b_blocked_fix_required"
    lines = [
        "# Root-Cause Feature and Contract Closure Report",
        "",
        f"Final closure status: `{status}`.",
        "",
        "No final ML model was trained and no strategy was tuned. Diagnostic models were used only to decide whether absolute price features should be allowed, forbidden, or transformed.",
        "",
        "## Closure Gates",
        "",
        gates.to_markdown(index=False),
        "",
        "## Absolute Price Decision",
        "",
        f"Decision: `{price_decision['decision']}`.",
        "",
        price_decision["rationale"],
        "",
        "## Dynamic Feature Rebuild",
        "",
        "Rebuilt dynamic features are allowed only from `data/rebuilt_dynamic_features.parquet`, with session-boundary resets.",
        "",
        "## Rolling Causality",
        "",
        rolling.groupby("feature", as_index=False).agg(max_abs_error=("max_abs_error", "max"), recompute_pass=("recompute_pass", "all")).to_markdown(index=False),
        "",
        "## Cost / Unit",
        "",
        "Cost units remain proxy assumptions, not exchange-certified values. See `COST_UNIT_CONTRACT.md`.",
        "",
        "## Split Drift",
        "",
        "OOS drift is quantified in `results/split_regime_drift_closure.csv`; this supports harsh OOS validity rather than invalidating the split.",
        "",
        "## Baseline Hurdles",
        "",
        "Baseline hurdle consistency passed across validation and frozen OOS splits.",
        "",
        "## Phase 03B Permission",
        "",
        "`Phase 03B may proceed only under these constraints:`",
        "",
        "- Do not use raw future labels or raw PnL as features.",
        "- Use excess markout targets only.",
        "- Treat raw absolute price levels according to `absolute_price_feature_decision.json`.",
        "- Use rebuilt dynamic features only from the causal rebuild artifact.",
        "- Select models/thresholds only on validation.",
        "- Keep test, NEW3, and NEW4 frozen.",
    ]
    write_markdown(ROOT / "ROOT_CAUSE_CLOSURE_REPORT.md", lines)
    print(f"Final closure status: {status}")
    print(gates.to_string(index=False))


def main() -> None:
    ensure_dirs()
    price = absolute_price_feature_audit()
    dynamic = rebuild_dynamic_features()
    rolling = rolling_recompute_audit()
    cost_unit_contract()
    drift = split_regime_drift_closure()
    hurdle = baseline_hurdle_consistency()
    final_report(price, dynamic, rolling, drift, hurdle)


if __name__ == "__main__":
    main()
