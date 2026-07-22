from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from new_era.models_playaround_15.config import DataConfig, ExperimentConfig
from new_era.models_playaround_15.data_loading import load_new_tw_data
from new_era.models_playaround_15.metadata import save_experiment_config, write_json
from new_era.models_playaround_15.paths import ExperimentPaths, build_experiment_paths, get_project_root
from new_era.tcn_pipeline_utils import apply_rolling_zscore, split_ordered_session_ids
from src.data.tw_replay import infer_tick_size_from_lob
from src.models.tcn import HLSFriendlyTCNClassifier
from src.training.datasets import WindowedLOBSequenceDataset
from src.utils.tw_train_helpers import build_tw_feature_frame


def _ensure_project_root_on_path(project_root: Path) -> None:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _thresholds_prob() -> list[float]:
    return [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]


def _thresholds_edge() -> list[float]:
    return [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def _build_session_dataset(
    cfg: ExperimentConfig,
    lob_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    symbol: str,
    tick_size: float,
    *,
    sample_stride: int,
) -> WindowedLOBSequenceDataset:
    feature_frame = build_tw_feature_frame(
        lob_df.reset_index(drop=True),
        trade_df.reset_index(drop=True),
        symbol,
        tick_size,
    ).dropna()
    zscore_frame = apply_rolling_zscore(feature_frame, cfg.features.selected_features, window=cfg.data.zscore_window)
    return WindowedLOBSequenceDataset(
        feature_df=zscore_frame,
        symbol=symbol,
        tick_size=tick_size,
        window_size=cfg.window.window_size,
        horizon=cfg.window.horizon,
        threshold_ticks=cfg.window.threshold_ticks,
        sample_stride=sample_stride,
        feature_cols=cfg.features.z_features,
        label_type=cfg.labels.label_type,
        adverse_threshold_ticks=cfg.labels.adverse_threshold_ticks,
        taker_fee_twd=cfg.labels.taker_fee_twd,
        point_value_twd_per_point=cfg.labels.point_value_twd_per_point,
        exit_cost_ticks=cfg.labels.exit_cost_ticks,
        margin_ticks=cfg.labels.margin_ticks,
    )


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    datasets: list[tuple[str, WindowedLOBSequenceDataset]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_probs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_ask: list[np.ndarray] = []
    all_bid: list[np.ndarray] = []

    for session_id, dataset in datasets:
        loader = DataLoader(dataset, batch_size=2048, shuffle=False)
        for batch in loader:
            batch_x = batch["x"].transpose(1, 2).to(device)
            probs = torch.softmax(model(batch_x), dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(batch["y"].numpy())
            all_ask.append(batch["ask_px_1"].numpy())
            all_bid.append(batch["bid_px_1"].numpy())

    if not all_probs:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_ask, axis=0),
        np.concatenate(all_bid, axis=0),
    )


@dataclass(frozen=True)
class StrategyParams:
    strategy: str
    prob_threshold: float
    edge_threshold: float
    max_spread_ticks: float = 3.0
    skew_ticks: int = 1
    allow_inside_skew: bool = True


def _choose_quotes_maker_only(
    *,
    cfg: ExperimentConfig,
    prob: np.ndarray,
    bid: float,
    ask: float,
    tick_size: float,
    inventory: int,
    params: StrategyParams,
) -> tuple[float | None, float | None]:
    quote_bid: float | None = bid
    quote_ask: float | None = ask

    if params.strategy == "baseline":
        pass

    elif cfg.num_classes == 3:
        p_down, p_up = float(prob[0]), float(prob[2])
        edge = p_up - p_down
        spread = ask - bid

        if params.strategy == "tcn_gated":
            if p_down >= params.prob_threshold and edge <= -params.edge_threshold:
                quote_bid = None
            if p_up >= params.prob_threshold and edge >= params.edge_threshold:
                quote_ask = None

        elif params.strategy == "tcn_skew":
            if spread / tick_size > params.max_spread_ticks:
                quote_bid = None
                quote_ask = None
            elif p_up >= params.prob_threshold and edge >= params.edge_threshold:
                quote_ask = None
                if params.allow_inside_skew and spread >= 2 * tick_size:
                    quote_bid = min(bid + params.skew_ticks * tick_size, ask - tick_size)
            elif p_down >= params.prob_threshold and edge <= -params.edge_threshold:
                quote_bid = None
                if params.allow_inside_skew and spread >= 2 * tick_size:
                    quote_ask = max(ask - params.skew_ticks * tick_size, bid + tick_size)
        else:
            raise ValueError(f"Unknown strategy: {params.strategy}")

    elif cfg.num_classes == 2:
        p_pos = float(prob[1]) if len(prob) > 1 else float(prob[0])

        if params.strategy == "tcn_gated":
            if cfg.labels.label_type in ("toxic_bid_binary", "adverse_bid_fill_binary"):
                if p_pos >= params.prob_threshold:
                    quote_bid = None
            elif cfg.labels.label_type in ("toxic_ask_binary", "adverse_ask_fill_binary"):
                if p_pos >= params.prob_threshold:
                    quote_ask = None
            else:
                if p_pos >= params.prob_threshold:
                    quote_bid = None
                    quote_ask = None

        elif params.strategy == "tcn_skew":
            if cfg.labels.label_type in ("toxic_bid_binary", "adverse_bid_fill_binary"):
                if p_pos >= params.prob_threshold:
                    quote_bid = None
            elif cfg.labels.label_type in ("toxic_ask_binary", "adverse_ask_fill_binary"):
                if p_pos >= params.prob_threshold:
                    quote_ask = None
            else:
                if p_pos >= params.prob_threshold:
                    quote_bid = None
                    quote_ask = None
        else:
            raise ValueError(f"Unknown strategy: {params.strategy}")

    else:
        raise ValueError(f"Unsupported num_classes: {cfg.num_classes}")

    if inventory >= cfg.backtest.max_inventory:
        quote_bid = None
    if inventory <= -cfg.backtest.max_inventory:
        quote_ask = None

    if quote_bid is not None and quote_ask is not None and quote_ask <= quote_bid:
        quote_bid = None
        quote_ask = None

    return quote_bid, quote_ask


def _max_drawdown(pnl_path: np.ndarray) -> float:
    pnl_path = np.asarray(pnl_path, dtype=np.float64)
    if pnl_path.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(pnl_path)
    drawdowns = peaks - pnl_path
    return float(np.max(drawdowns))


def _future_min_max(mid: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    mid = np.asarray(mid, dtype=np.float64)
    if len(mid) <= horizon:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(mid[1:], window_shape=horizon)
    return view.min(axis=1), view.max(axis=1)


def run_mm_simulation(
    *,
    cfg: ExperimentConfig,
    probs: np.ndarray,
    ask_px: np.ndarray,
    bid_px: np.ndarray,
    tick_size: float,
    params: StrategyParams,
    return_paths: bool = False,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray] | None]:
    probs = np.asarray(probs, dtype=np.float64)
    ask_px = np.asarray(ask_px, dtype=np.float64)
    bid_px = np.asarray(bid_px, dtype=np.float64)

    point_value = float(cfg.backtest.point_value_twd_per_point)
    maker_fee = float(cfg.backtest.maker_fee_twd)
    taker_fee = float(cfg.backtest.taker_fee_twd)

    inventory = 0
    gross_cash = 0.0
    fees_paid = 0.0
    maker_trades = 0
    quote_sides = 0
    spread_capture_ticks: list[float] = []
    adverse_selection_ticks: list[float] = []

    mid = (ask_px + bid_px) / 2.0
    fut_min, fut_max = _future_min_max(mid, horizon=cfg.window.horizon)

    pnl_path: list[float] = []
    inv_path: list[int] = []

    for i in range(len(probs) - 1):
        bid = float(bid_px[i])
        ask = float(ask_px[i])
        next_bid = float(bid_px[i + 1])
        next_ask = float(ask_px[i + 1])

        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(next_bid) and np.isfinite(next_ask)):
            pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * point_value)
            inv_path.append(inventory)
            continue
        if ask <= bid:
            pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * point_value)
            inv_path.append(inventory)
            continue

        quote_bid, quote_ask = _choose_quotes_maker_only(
            cfg=cfg,
            prob=probs[i],
            bid=bid,
            ask=ask,
            tick_size=tick_size,
            inventory=inventory,
            params=params,
        )
        if quote_bid is not None:
            quote_sides += 1
        if quote_ask is not None:
            quote_sides += 1

        spread_capture_ticks_now = (ask - bid) / (2.0 * tick_size) if tick_size > 0 else 0.0

        if quote_bid is not None and next_ask <= quote_bid and inventory < cfg.backtest.max_inventory:
            gross_cash -= quote_bid * point_value
            fees_paid += maker_fee
            inventory += 1
            maker_trades += 1
            spread_capture_ticks.append(float(spread_capture_ticks_now))
            if i < len(fut_min):
                adverse = max(0.0, (quote_bid - float(fut_min[i])) / tick_size)
                adverse_selection_ticks.append(float(adverse))

        if quote_ask is not None and next_bid >= quote_ask and inventory > -cfg.backtest.max_inventory:
            gross_cash += quote_ask * point_value
            fees_paid += maker_fee
            inventory -= 1
            maker_trades += 1
            spread_capture_ticks.append(float(spread_capture_ticks_now))
            if i < len(fut_max):
                adverse = max(0.0, (float(fut_max[i]) - quote_ask) / tick_size)
                adverse_selection_ticks.append(float(adverse))

        pnl_path.append(gross_cash - fees_paid + inventory * float(mid[i]) * point_value)
        inv_path.append(inventory)

    final_mid = float(mid[-1]) if len(mid) else 0.0
    final_bid = float(bid_px[-1]) if len(bid_px) else final_mid
    final_ask = float(ask_px[-1]) if len(ask_px) else final_mid
    gross_pnl = gross_cash + inventory * final_mid * point_value
    net_pnl = gross_pnl - fees_paid

    # Realized variant: force-flatten inventory at the end using a conservative taker liquidation.
    # This makes it easier to compare strategies without "ending inventory drift" artifacts.
    flatten_cash = float(gross_cash)
    flatten_fees = float(fees_paid)
    forced_liq_trades = 0
    if inventory > 0:
        forced_liq_trades = int(inventory)
        flatten_cash += float(inventory) * final_bid * point_value
        flatten_fees += float(abs(inventory)) * taker_fee
    elif inventory < 0:
        forced_liq_trades = int(abs(inventory))
        flatten_cash -= float(abs(inventory)) * final_ask * point_value
        flatten_fees += float(abs(inventory)) * taker_fee

    pnl_after_fees_flatten = float(flatten_cash - flatten_fees)
    pnl_arr = np.asarray(pnl_path, dtype=np.float64) if pnl_path else np.zeros((0,), dtype=np.float64)
    inv_arr = np.asarray(inv_path, dtype=np.int64) if inv_path else np.zeros((0,), dtype=np.int64)

    result: dict[str, float | int | str] = {
        "strategy": params.strategy,
        "prob_threshold": float(params.prob_threshold),
        "edge_threshold": float(params.edge_threshold),
        "pnl": float(gross_pnl),
        "pnl_after_fees": float(net_pnl),
        "pnl_after_fees_flatten": float(pnl_after_fees_flatten),
        "fees_paid": float(fees_paid),
        "fees_paid_flatten": float(flatten_fees),
        "trades": int(maker_trades),
        "maker_trades": int(maker_trades),
        "taker_trades": int(0),
        "ending_inventory": int(inventory),
        "ending_inventory_flatten": int(0),
        "forced_liquidation_trades": int(forced_liq_trades),
        "max_abs_inventory": int(np.max(np.abs(inv_arr))) if inv_arr.size else 0,
        "fill_rate": float(maker_trades / max(quote_sides, 1)),
        "avg_spread_capture": float(np.mean(spread_capture_ticks)) if spread_capture_ticks else 0.0,
        "avg_adverse_selection": float(np.mean(adverse_selection_ticks)) if adverse_selection_ticks else 0.0,
        "max_drawdown": float(_max_drawdown(pnl_arr)),
    }

    paths_out = {"pnl_path": pnl_arr, "inventory_path": inv_arr} if return_paths else None
    return result, paths_out


def _save_pnl_curve_plot(
    pnl_base: np.ndarray,
    pnl_model: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pnl_base, label="baseline", linewidth=1.2)
    ax.plot(pnl_model, label="model", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative PnL (TWD)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_inventory_curve_plot(inv: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(inv, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("inventory")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_pnl_bar(baseline_pnl: float, model_pnl: float, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline", "model"], [baseline_pnl, model_pnl], color=["#4C78A8", "#F58518"])
    ax.set_ylabel("PnL (TWD)")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_sweep_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(
        index="edge_threshold",
        columns="prob_threshold",
        values="improvement_vs_baseline",
        aggfunc="mean",
    ).sort_index(ascending=True)
    mat = pivot.to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("prob_threshold")
    ax.set_ylabel("edge_threshold")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def backtest_val_sweep_then_test(cfg: ExperimentConfig, paths: ExperimentPaths) -> dict[str, object]:
    project_root = get_project_root()
    _ensure_project_root_on_path(project_root)
    paths.ensure_dirs()
    save_experiment_config(paths, cfg)

    if not paths.model_weights_pth.exists():
        raise FileNotFoundError(f"Model weights not found: {paths.model_weights_pth}")

    data_dir = cfg.data.data_dir
    symbol = cfg.backtest.symbol
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"No LOB data found for {symbol} in {data_dir}")

    symbol_lob = lob_frames[symbol]
    symbol_trades = trade_frames.get(
        symbol, pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"])
    )
    session_ids = list(symbol_lob["session_id"].drop_duplicates())
    train_sessions, val_sessions, test_sessions = split_ordered_session_ids(
        session_ids,
        train_frac=cfg.split.train_session_frac,
        val_frac=cfg.split.val_session_frac,
    )

    if not val_sessions or not test_sessions:
        raise RuntimeError("Need at least one validation session and one test session.")

    tick_size = float(infer_tick_size_from_lob(symbol_lob))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HLSFriendlyTCNClassifier(
        num_inputs=cfg.num_inputs,
        num_classes=cfg.num_classes,
        channel_list=list(cfg.model.channel_list),
        kernel_size=cfg.model.kernel_size,
        dropout=cfg.model.dropout,
        seq_length=cfg.window.window_size,
    ).to(device)
    model.load_state_dict(torch.load(paths.model_weights_pth, map_location=device, weights_only=True))
    model.eval()

    def _build_session_list(session_ids: list[str]) -> list[tuple[str, WindowedLOBSequenceDataset]]:
        out: list[tuple[str, WindowedLOBSequenceDataset]] = []
        for sid in session_ids:
            lob_df = symbol_lob[symbol_lob["session_id"] == sid]
            trd_df = symbol_trades[symbol_trades["session_id"] == sid]
            if len(lob_df) < cfg.window.window_size + cfg.window.horizon:
                continue
            out.append(
                (
                    sid,
                    _build_session_dataset(
                        cfg,
                        lob_df=lob_df,
                        trade_df=trd_df,
                        symbol=symbol,
                        tick_size=tick_size,
                        sample_stride=cfg.backtest.sample_stride,
                    ),
                )
            )
        return out

    val_datasets = _build_session_list(val_sessions)
    test_datasets = _build_session_list(test_sessions)
    if not val_datasets or not test_datasets:
        raise RuntimeError("Validation/test sessions did not contain enough rows for the configured window/horizon.")

    val_probs, _, val_ask, val_bid = _collect_predictions(model, val_datasets, device=device)
    test_probs, _, test_ask, test_bid = _collect_predictions(model, test_datasets, device=device)

    baseline_params = StrategyParams(strategy="baseline", prob_threshold=float("nan"), edge_threshold=float("nan"))
    baseline_val, baseline_val_paths = run_mm_simulation(
        cfg=cfg,
        probs=val_probs,
        ask_px=val_ask,
        bid_px=val_bid,
        tick_size=tick_size,
        params=baseline_params,
        return_paths=True,
    )

    rows: list[dict[str, float | int | str]] = []
    rows.append({**baseline_val})

    strategies = ["tcn_gated", "tcn_skew"]
    prob_thresholds = _thresholds_prob()
    edge_thresholds = _thresholds_edge() if cfg.num_classes == 3 else [0.0]

    for strategy in strategies:
        for pthr in prob_thresholds:
            for ethr in edge_thresholds:
                params = StrategyParams(strategy=strategy, prob_threshold=float(pthr), edge_threshold=float(ethr))
                res, _ = run_mm_simulation(
                    cfg=cfg,
                    probs=val_probs,
                    ask_px=val_ask,
                    bid_px=val_bid,
                    tick_size=tick_size,
                    params=params,
                    return_paths=False,
                )
                rows.append(res)

    sweep_df = pd.DataFrame(rows)
    sweep_df["improvement_vs_baseline"] = sweep_df["pnl_after_fees"] - float(baseline_val["pnl_after_fees"])
    sweep_df["improvement_vs_baseline_flatten"] = sweep_df["pnl_after_fees_flatten"] - float(
        baseline_val["pnl_after_fees_flatten"]
    )
    sweep_df = sweep_df.sort_values(["pnl_after_fees", "strategy"], ascending=[False, True]).reset_index(drop=True)

    val_sweep_csv = paths.backtest_dir / "val_sweep_results.csv"
    sweep_df.to_csv(val_sweep_csv, index=False, encoding="utf-8-sig")

    candidates = sweep_df[sweep_df["strategy"].isin(["tcn_gated", "tcn_skew"])].copy()
    best = candidates.iloc[0].to_dict() if not candidates.empty else {}
    selected = {
        "strategy": best.get("strategy", "baseline"),
        "prob_threshold": float(best.get("prob_threshold", float("nan"))),
        "edge_threshold": float(best.get("edge_threshold", float("nan"))),
        "max_spread_ticks": 3.0,
        "skew_ticks": 1,
        "allow_inside_skew": True,
        "selection_metric": "pnl_after_fees_on_val",
        "val_baseline_pnl_after_fees": float(baseline_val["pnl_after_fees"]),
        "val_selected_pnl_after_fees": float(best.get("pnl_after_fees", float("nan"))),
        "val_baseline_pnl_after_fees_flatten": float(baseline_val["pnl_after_fees_flatten"]),
        "val_selected_pnl_after_fees_flatten": float(best.get("pnl_after_fees_flatten", float("nan"))),
    }

    write_json(paths.backtest_dir / "selected_strategy_config.json", selected)

    model_params = StrategyParams(
        strategy=str(selected["strategy"]),
        prob_threshold=float(selected["prob_threshold"]),
        edge_threshold=float(selected["edge_threshold"]),
        max_spread_ticks=float(selected["max_spread_ticks"]),
        skew_ticks=int(selected["skew_ticks"]),
        allow_inside_skew=bool(selected["allow_inside_skew"]),
    )

    model_val, model_val_paths = run_mm_simulation(
        cfg=cfg,
        probs=val_probs,
        ask_px=val_ask,
        bid_px=val_bid,
        tick_size=tick_size,
        params=model_params,
        return_paths=True,
    )

    baseline_test, baseline_test_paths = run_mm_simulation(
        cfg=cfg,
        probs=test_probs,
        ask_px=test_ask,
        bid_px=test_bid,
        tick_size=tick_size,
        params=baseline_params,
        return_paths=True,
    )
    model_test, model_test_paths = run_mm_simulation(
        cfg=cfg,
        probs=test_probs,
        ask_px=test_ask,
        bid_px=test_bid,
        tick_size=tick_size,
        params=model_params,
        return_paths=True,
    )

    test_row = {
        "baseline_pnl": float(baseline_test["pnl_after_fees"]),
        "model_pnl": float(model_test["pnl_after_fees"]),
        "improvement_vs_baseline": float(model_test["pnl_after_fees"] - baseline_test["pnl_after_fees"]),
        "baseline_pnl_flatten": float(baseline_test["pnl_after_fees_flatten"]),
        "model_pnl_flatten": float(model_test["pnl_after_fees_flatten"]),
        "improvement_vs_baseline_flatten": float(
            model_test["pnl_after_fees_flatten"] - baseline_test["pnl_after_fees_flatten"]
        ),
        "trades": int(model_test["trades"]),
        "maker_trades": int(model_test["maker_trades"]),
        "taker_trades": int(model_test["taker_trades"]),
        "ending_inventory": int(model_test["ending_inventory"]),
        "ending_inventory_flatten": int(model_test["ending_inventory_flatten"]),
        "forced_liquidation_trades": int(model_test["forced_liquidation_trades"]),
        "max_abs_inventory": int(model_test["max_abs_inventory"]),
        "max_drawdown": float(model_test["max_drawdown"]),
    }
    pd.DataFrame([test_row]).to_csv(paths.backtest_dir / "test_final_results.csv", index=False, encoding="utf-8-sig")

    if baseline_val_paths and model_val_paths:
        _save_pnl_curve_plot(
            pnl_base=baseline_val_paths["pnl_path"],
            pnl_model=model_val_paths["pnl_path"],
            title="Validation cumulative PnL (selected config)",
            out_path=paths.backtest_dir / "pnl_curve_val_selected.png",
        )
    if baseline_test_paths and model_test_paths:
        _save_pnl_curve_plot(
            pnl_base=baseline_test_paths["pnl_path"],
            pnl_model=model_test_paths["pnl_path"],
            title="Test cumulative PnL (fixed from val selection)",
            out_path=paths.backtest_dir / "pnl_curve_test_final.png",
        )
        _save_inventory_curve_plot(
            inv=model_test_paths["inventory_path"],
            title="Test inventory curve (model)",
            out_path=paths.backtest_dir / "inventory_curve_test_final.png",
        )

    _save_pnl_bar(
        baseline_pnl=float(baseline_test["pnl_after_fees"]),
        model_pnl=float(model_test["pnl_after_fees"]),
        title="Baseline vs Model (test)",
        out_path=paths.backtest_dir / "baseline_vs_model_pnl_bar.png",
    )
    _save_pnl_bar(
        baseline_pnl=float(baseline_test["pnl_after_fees_flatten"]),
        model_pnl=float(model_test["pnl_after_fees_flatten"]),
        title="Baseline vs Model (test, forced flatten)",
        out_path=paths.backtest_dir / "baseline_vs_model_pnl_bar_flatten.png",
    )

    if cfg.num_classes == 3:
        heat_df = sweep_df[sweep_df["strategy"].isin(["tcn_gated", "tcn_skew"])].copy()
        heat_df = heat_df[np.isfinite(heat_df["prob_threshold"]) & np.isfinite(heat_df["edge_threshold"])]
        if not heat_df.empty:
            _save_sweep_heatmap(
                heat_df,
                title="Val sweep heatmap (PnL improvement vs baseline)",
                out_path=paths.backtest_dir / "threshold_sweep_heatmap_val.png",
            )
            if "improvement_vs_baseline_flatten" in heat_df.columns:
                heat_df_flat = heat_df.copy()
                heat_df_flat["improvement_vs_baseline"] = heat_df_flat["improvement_vs_baseline_flatten"]
                _save_sweep_heatmap(
                    heat_df_flat,
                    title="Val sweep heatmap (PnL improvement vs baseline, forced flatten)",
                    out_path=paths.backtest_dir / "threshold_sweep_heatmap_val_flatten.png",
                )

    return {
        "device": str(device),
        "symbol": symbol,
        "sessions": {"train": train_sessions, "val": val_sessions, "test": test_sessions},
        "val_sweep_csv": str(val_sweep_csv),
        "selected": selected,
        "test_final": test_row,
    }


def build_default_config() -> ExperimentConfig:
    project_root = get_project_root()
    return ExperimentConfig(
        data=DataConfig(
            data_dir=project_root / "data" / "NEW",
            symbols=("TXFB6",),
            zscore_window=2000,
        )
    )


def main() -> None:
    cfg = build_default_config()
    paths = build_experiment_paths(cfg)
    print(f"[backtest] dataset: {paths.dataset_npz}")
    print(f"[backtest] weights: {paths.model_weights_pth}")
    out = backtest_val_sweep_then_test(cfg, paths)
    print(f"[backtest] val_sweep_csv={out['val_sweep_csv']}")
    print(
        f"[backtest] test baseline={out['test_final']['baseline_pnl']:.2f} model={out['test_final']['model_pnl']:.2f} (+{out['test_final']['improvement_vs_baseline']:.2f})"
    )

if __name__ == "__main__":
    main()
