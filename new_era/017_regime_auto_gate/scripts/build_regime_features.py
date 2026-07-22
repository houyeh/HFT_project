from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import (
    FEATURES_DIR,
    MODEL_DIR,
    NEW3_DIR,
    NEW3_SYMBOL,
    NEW_ALL_DIR,
    NEW_SYMBOL,
    REPORTS_DIR,
    DatasetBundle,
    build_event_frame,
    build_regime_features,
    build_session_datasets,
    build_split_map,
    collect_predictions_extended,
    detect_symbol_from_data_dir,
    ensure_output_dirs,
    get_logger,
    infer_tick_size_from_lob,
    load_experiment_cfg,
    load_new_tw_data,
    load_trained_model,
    ordered_feature_columns,
    resolve_device,
)


def _build_bundle(
    *,
    dataset: str,
    data_dir: Path,
    preferred_symbol: str,
    cfg,
    model,
    device,
) -> DatasetBundle:
    symbol = detect_symbol_from_data_dir(data_dir, preferred_symbol)
    lob_frames, trade_frames = load_new_tw_data(data_dir, [symbol])
    if symbol not in lob_frames:
        raise RuntimeError(f"No LOB data loaded for {dataset}/{symbol} from {data_dir}")

    symbol_lob = lob_frames[symbol]
    symbol_trades = trade_frames.get(
        symbol,
        pd.DataFrame(columns=["timestamp", "session_id", "trade_price", "trade_volume"]),
    )
    tick_size = float(infer_tick_size_from_lob(symbol_lob))
    session_ids = list(symbol_lob["session_id"].drop_duplicates())

    if dataset == "NEW":
        split_map = build_split_map(session_ids, cfg)
    else:
        split_map = {sid: "oos_full" for sid in session_ids}

    datasets = build_session_datasets(
        cfg=cfg,
        symbol_lob=symbol_lob,
        symbol_trades=symbol_trades,
        symbol=symbol,
        tick_size=tick_size,
        session_ids=session_ids,
    )
    pred_pack = collect_predictions_extended(model, datasets, device, cfg.num_classes)
    event_df = build_event_frame(pred_pack, dataset=dataset, tick_size=tick_size, split_map=split_map)
    feat_df = build_regime_features(event_df, tick_size=tick_size)
    feat_df = feat_df[ordered_feature_columns(feat_df)]
    return DatasetBundle(
        dataset=dataset,
        symbol=symbol,
        tick_size=tick_size,
        session_ids=session_ids,
        split_map=split_map,
        event_df=feat_df,
    )


def _write_feature_report(bundles: list[DatasetBundle]) -> None:
    key_fields = [
        "timestamp",
        "imbalance",
        "rolling_imbalance_mean_100",
        "microprice_minus_mid_ticks",
    ]
    lines: list[str] = []
    lines.append("# Regime Feature Report")
    lines.append("")
    lines.append("## Causality / Leakage Check")
    lines.append("- All regime features are computed with backward-looking rolling or EWMA operators only.")
    lines.append("- No full-session future return, future trend, or cumulative future information is used.")
    lines.append("- Event-level masks for downstream backtests are based on current/past event information only.")
    lines.append("")
    lines.append("## Dataset Coverage")
    lines.append("| dataset | symbol | sessions | rows | split_summary |")
    lines.append("|---|---:|---:|---:|---|")
    for bundle in bundles:
        split_counts = bundle.event_df["split"].value_counts(dropna=False).to_dict()
        split_summary = ", ".join(f"{k}:{v}" for k, v in sorted(split_counts.items()))
        lines.append(
            f"| {bundle.dataset} | {bundle.symbol} | {len(bundle.session_ids)} | {len(bundle.event_df)} | {split_summary} |"
        )
    lines.append("")
    lines.append("## Field Availability")
    lines.append("| dataset | field | available | missing_ratio |")
    lines.append("|---|---|---|---:|")
    for bundle in bundles:
        for field in key_fields:
            if field not in bundle.event_df.columns:
                lines.append(f"| {bundle.dataset} | {field} | no | 1.0000 |")
                continue
            series = bundle.event_df[field]
            missing_ratio = float(series.isna().mean()) if len(series) else 1.0
            available = "yes" if missing_ratio < 1.0 else "no"
            lines.append(f"| {bundle.dataset} | {field} | {available} | {missing_ratio:.4f} |")
    lines.append("")

    new_bundle = next((b for b in bundles if b.dataset == "NEW"), None)
    if new_bundle is not None:
        val_sessions = sorted({sid for sid, split in new_bundle.split_map.items() if split == "val"})
        lines.append("## Validation Caveat")
        lines.append(
            f"- NEW validation session count = {len(val_sessions)} ({', '.join(val_sessions) if val_sessions else 'none'})."
        )
        lines.append(
            "- If this count is 1, validation-driven policy selection has high variance and should be treated cautiously."
        )
        lines.append("")

    (REPORTS_DIR / "regime_feature_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    logger = get_logger("build_regime_features")
    logger.info("Loading config and model from %s", MODEL_DIR)
    cfg = load_experiment_cfg(MODEL_DIR)
    device = resolve_device(cfg)
    model = load_trained_model(cfg, device, MODEL_DIR)

    bundles: list[DatasetBundle] = []
    for dataset, data_dir, symbol in [
        ("NEW", NEW_ALL_DIR, NEW_SYMBOL),
        ("NEW3", NEW3_DIR, NEW3_SYMBOL),
    ]:
        logger.info("Building causal regime features for %s from %s", dataset, data_dir)
        bundle = _build_bundle(
            dataset=dataset,
            data_dir=data_dir,
            preferred_symbol=symbol,
            cfg=cfg,
            model=model,
            device=device,
        )
        bundles.append(bundle)
        out_csv = FEATURES_DIR / f"regime_features_{dataset}.csv"
        bundle.event_df.to_csv(out_csv, index=False)
        logger.info("Saved %s rows to %s", len(bundle.event_df), out_csv)

    _write_feature_report(bundles)
    meta = {
        "model_dir": str(MODEL_DIR),
        "datasets": [
            {
                "dataset": b.dataset,
                "symbol": b.symbol,
                "tick_size": b.tick_size,
                "rows": int(len(b.event_df)),
                "sessions": b.session_ids,
            }
            for b in bundles
        ],
    }
    (FEATURES_DIR / "feature_build_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Feature build complete.")


if __name__ == "__main__":
    main()

