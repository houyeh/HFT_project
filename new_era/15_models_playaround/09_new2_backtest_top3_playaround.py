from __future__ import annotations

"""
09 - NEW2 backtest: top-3 playaround models (fixed config)
=========================================================

用途：
- 讀取 015 playaround 的 `experiment_summary.csv`，挑出 pnl_improvement_flatten 前三名的 experiments
- 對 NEW2 資料做「out-of-sample」回測：
  - baseline（不看模型，best bid/ask maker-only）
  - model（使用該 experiment 的 model weights + 當時在 NEW/val 選到的策略參數；不在 NEW2 上重新 sweep）

輸出位置（依你指定的 NEW2_result）：
`new_era/backtest_outputs/NEW2_result/top3_playaround/`

注意：
- 這支需要 torch（會做模型推論）。請用你原本跑 015 的 python 環境執行。
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from _common import _ensure_project_root_on_path

_ensure_project_root_on_path()

from new_era.models_playaround_15.config import (  # noqa: E402
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    FeatureSet,
    LabelConfig,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
    WindowConfig,
)
from new_era.models_playaround_15.data_loading import load_new_tw_data  # noqa: E402
from new_era.models_playaround_15.paths import get_project_root  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _maybe_load_cached_row(out_exp: Path) -> dict[str, object] | None:
    cached = out_exp / "summary_new2.json"
    if not cached.exists():
        return None
    try:
        payload = _read_json(cached)
    except Exception:
        return None
    if isinstance(payload, dict) and payload.get("experiment_id"):
        return payload  # type: ignore[return-value]
    return None


def _load_cfg_from_experiment_dir(exp_dir: Path) -> ExperimentConfig:
    cfg_dict = _read_json(exp_dir / "experiment_config.json")

    data = cfg_dict.get("data") or {}
    window = cfg_dict.get("window") or {}
    split = cfg_dict.get("split") or {}
    features = cfg_dict.get("features") or {}
    labels = cfg_dict.get("labels") or {}
    model = cfg_dict.get("model") or {}
    train = cfg_dict.get("train") or {}
    backtest = cfg_dict.get("backtest") or {}

    return ExperimentConfig(
        data=DataConfig(
            data_dir=Path(str(data.get("data_dir") or "")),
            symbols=tuple(data.get("symbols") or ("TXFB6",)),
            zscore_window=int(data.get("zscore_window") or 2000),
        ),
        window=WindowConfig(
            window_size=int(window.get("window_size") or 100),
            horizon=int(window.get("horizon") or 10),
            threshold_ticks=float(window.get("threshold_ticks") or 1.0),
            sample_stride_train=int(window.get("sample_stride_train") or 5),
        ),
        split=SplitConfig(
            train_session_frac=float(split.get("train_session_frac") or 0.70),
            val_session_frac=float(split.get("val_session_frac") or 0.15),
        ),
        features=FeatureSet(feature_count=int(features.get("feature_count") or 8)),
        labels=LabelConfig(
            label_type=str(labels.get("label_type") or "mid_move_3class"),  # type: ignore[arg-type]
            adverse_threshold_ticks=float(labels.get("adverse_threshold_ticks") or 1.0),
            taker_fee_twd=float(labels.get("taker_fee_twd") or 50.0),
            point_value_twd_per_point=float(labels.get("point_value_twd_per_point") or 200.0),
            exit_cost_ticks=float(labels.get("exit_cost_ticks") or 0.0),
            margin_ticks=float(labels.get("margin_ticks") or 0.0),
        ),
        model=ModelConfig(
            channel_list=tuple(int(x) for x in (model.get("channel_list") or (16, 16, 16))),
            kernel_size=int(model.get("kernel_size") or 3),
            dropout=float(model.get("dropout") or 0.1),
        ),
        train=TrainingConfig(
            epochs=int(train.get("epochs") or 30),
            batch_size=int(train.get("batch_size") or 1024),
            lr=float(train.get("lr") or 1e-3),
            seed=int(train.get("seed") or 42),
            device=str(train.get("device") or "auto"),
        ),
        backtest=BacktestConfig(
            symbol=str(backtest.get("symbol") or "TXFB6"),
            point_value_twd_per_point=float(backtest.get("point_value_twd_per_point") or 200.0),
            maker_fee_twd=float(backtest.get("maker_fee_twd") or 15.0),
            taker_fee_twd=float(backtest.get("taker_fee_twd") or 50.0),
            max_inventory=int(backtest.get("max_inventory") or 3),
            sample_stride=int(backtest.get("sample_stride") or 1),
            allow_active_alpha=bool(backtest.get("allow_active_alpha") or False),
        ),
    )


def _load_selected_params(exp_dir: Path) -> dict:
    path = exp_dir / "backtest_outputs" / "selected_strategy_config.json"
    if not path.exists():
        raise FileNotFoundError(f"selected_strategy_config.json not found: {path}")
    return _read_json(path)


def _run_one_top_model_on_new2(
    *,
    exp_dir: Path,
    cfg_template: ExperimentConfig,
    new2_lob: pd.DataFrame,
    new2_trades: pd.DataFrame,
    out_dir: Path,
) -> dict[str, object]:
    out_exp = out_dir / exp_dir.name
    cached = _maybe_load_cached_row(out_exp)
    if cached is not None:
        print(f"[NEW2 top3] cache hit: {exp_dir.name}")
        return cached

    # Torch / model imports only when needed (allows the rest of the script to be imported without torch).
    import torch

    from src.data.tw_replay import infer_tick_size_from_lob
    from src.models.tcn import HLSFriendlyTCNClassifier
    from src.training.datasets import WindowedLOBSequenceDataset
    from new_era.tcn_pipeline_utils import apply_rolling_zscore
    from src.utils.tw_train_helpers import build_tw_feature_frame

    from new_era.models_playaround_15.backtest_val_sweep import (
        StrategyParams,
        _collect_predictions,
        _save_inventory_curve_plot,
        _save_pnl_curve_plot,
        run_mm_simulation,
    )

    tick_size = float(infer_tick_size_from_lob(new2_lob))

    cfg = cfg_template
    cfg = replace(cfg, data=replace(cfg.data, data_dir=get_project_root() / "data" / "NEW2"))

    selected = _load_selected_params(exp_dir)
    params = StrategyParams(
        strategy=str(selected.get("strategy") or "tcn_gated"),
        prob_threshold=float(selected.get("prob_threshold") or 0.7),
        edge_threshold=float(selected.get("edge_threshold") or 0.0),
        max_spread_ticks=float(selected.get("max_spread_ticks") or 3.0),
        skew_ticks=int(selected.get("skew_ticks") or 1),
        allow_inside_skew=bool(selected.get("allow_inside_skew") if "allow_inside_skew" in selected else True),
    )

    # Build session datasets (NEW2 has multiple sessions)
    session_ids = list(new2_lob["session_id"].drop_duplicates())
    datasets: list[tuple[str, WindowedLOBSequenceDataset]] = []
    for sid in session_ids:
        lob_df = new2_lob[new2_lob["session_id"] == sid]
        trd_df = new2_trades[new2_trades["session_id"] == sid] if not new2_trades.empty else new2_trades
        if len(lob_df) < cfg.window.window_size + cfg.window.horizon:
            continue
        feature_frame = build_tw_feature_frame(
            lob_df.reset_index(drop=True),
            trd_df.reset_index(drop=True),
            cfg.backtest.symbol,
            tick_size,
        ).dropna()
        zscore_frame = apply_rolling_zscore(feature_frame, cfg.features.selected_features, window=cfg.data.zscore_window)
        ds = WindowedLOBSequenceDataset(
            feature_df=zscore_frame,
            symbol=cfg.backtest.symbol,
            tick_size=tick_size,
            window_size=cfg.window.window_size,
            horizon=cfg.window.horizon,
            threshold_ticks=cfg.window.threshold_ticks,
            sample_stride=cfg.backtest.sample_stride,
            feature_cols=cfg.features.z_features,
            label_type=cfg.labels.label_type,
            adverse_threshold_ticks=cfg.labels.adverse_threshold_ticks,
            taker_fee_twd=cfg.labels.taker_fee_twd,
            point_value_twd_per_point=cfg.labels.point_value_twd_per_point,
            exit_cost_ticks=cfg.labels.exit_cost_ticks,
            margin_ticks=cfg.labels.margin_ticks,
        )
        datasets.append((sid, ds))

    if not datasets:
        raise RuntimeError(f"No usable session datasets in NEW2 for cfg window/horizon: {exp_dir.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs,
        num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        seq_length=cfg.window.window_size,
    ).to(device)
    weights_path = exp_dir / "checkpoints" / "tcn_weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    probs, _, ask_px, bid_px = _collect_predictions(model, datasets, device=device)

    baseline_params = StrategyParams(strategy="baseline", prob_threshold=float("nan"), edge_threshold=float("nan"))
    baseline, baseline_paths = run_mm_simulation(
        cfg=cfg,
        probs=probs,
        ask_px=ask_px,
        bid_px=bid_px,
        tick_size=tick_size,
        params=baseline_params,
        return_paths=True,
    )
    model_res, model_paths = run_mm_simulation(
        cfg=cfg,
        probs=probs,
        ask_px=ask_px,
        bid_px=bid_px,
        tick_size=tick_size,
        params=params,
        return_paths=True,
    )

    out_exp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([baseline]).to_csv(out_exp / "baseline_new2.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([model_res]).to_csv(out_exp / "model_new2.csv", index=False, encoding="utf-8-sig")
    _write_json(out_exp / "selected_strategy_config.json", selected)

    # Plots (optional but useful)
    if baseline_paths and model_paths:
        _save_pnl_curve_plot(
            pnl_base=baseline_paths["pnl_path"],
            pnl_model=model_paths["pnl_path"],
            title=f"NEW2 cumulative PnL: baseline vs {params.strategy}",
            out_path=out_exp / "pnl_curve_new2.png",
        )
        _save_inventory_curve_plot(
            inv=model_paths["inventory_path"],
            title=f"NEW2 inventory curve ({params.strategy})",
            out_path=out_exp / "inventory_curve_new2.png",
        )

    row = {
        "experiment_id": exp_dir.name,
        "label_type": cfg.labels.label_type,
        "feature_count": cfg.features.feature_count,
        "selected_strategy": params.strategy,
        "selected_prob_threshold": params.prob_threshold,
        "selected_edge_threshold": params.edge_threshold,
        "baseline_pnl_after_fees_flatten": baseline.get("pnl_after_fees_flatten"),
        "model_pnl_after_fees_flatten": model_res.get("pnl_after_fees_flatten"),
        "improvement_flatten": float(model_res["pnl_after_fees_flatten"]) - float(baseline["pnl_after_fees_flatten"]),
        "baseline_pnl_after_fees": baseline.get("pnl_after_fees"),
        "model_pnl_after_fees": model_res.get("pnl_after_fees"),
        "improvement": float(model_res["pnl_after_fees"]) - float(baseline["pnl_after_fees"]),
        "baseline_trades": baseline.get("trades"),
        "model_trades": model_res.get("trades"),
        "baseline_max_drawdown": baseline.get("max_drawdown"),
        "model_max_drawdown": model_res.get("max_drawdown"),
        "tick_size": tick_size,
        "device": str(device),
    }
    _write_json(out_exp / "summary_new2.json", row)
    return row


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="TXFB6")
    parser.add_argument("--k", type=int, default=3, help="top-k experiments by pnl_improvement_flatten")
    parser.add_argument("--results-root", type=str, default="new_era/15_models_playaround/015_models_playaround_result")
    parser.add_argument("--out-dir", type=str, default="new_era/backtest_outputs/NEW2_result/top3_playaround")
    args = parser.parse_args()

    project_root = get_project_root()
    results_root = (project_root / args.results_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = results_root / "experiment_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"experiment_summary.csv not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    for c in ("pnl_improvement_flatten", "pnl_improvement"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["pnl_improvement_flatten", "pnl_improvement"], ascending=[False, False])
    top = df.head(int(args.k)).copy()
    if top.empty:
        raise RuntimeError("No experiments found in experiment_summary.csv")

    symbol = str(args.symbol)
    data_dir = project_root / "data" / "NEW2"
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    lob = lob_frames.get(symbol)
    trades = trade_frames.get(symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]))
    if lob is None or lob.empty:
        raise RuntimeError(f"No LOB rows found for symbol={symbol} in {data_dir}")

    rows: list[dict[str, object]] = []
    for exp_id in top["experiment_id"].tolist():
        exp_dir = results_root / str(exp_id)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")
        cfg_template = _load_cfg_from_experiment_dir(exp_dir)
        print(f"[NEW2 top3] running: {exp_dir.name}")
        rows.append(
            _run_one_top_model_on_new2(
                exp_dir=exp_dir,
                cfg_template=cfg_template,
                new2_lob=lob,
                new2_trades=trades,
                out_dir=out_dir,
            )
        )

    out_csv = out_dir / "top3_playaround_on_new2_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    _write_json(out_dir / "top3_selected_experiments.json", {"top_k": int(args.k), "experiments": top.to_dict(orient="records")})
    print(f"[NEW2 top3] wrote -> {out_csv}")


if __name__ == "__main__":
    main()
