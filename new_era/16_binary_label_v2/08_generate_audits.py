from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_repro_manifest(results_dir: Path) -> None:
    thresholds = _load_json(OUTPUTS / "selected_thresholds.json")
    label_meta = _load_json(OUTPUTS / "binary_labels_v2_metadata.json")
    train_latest = _load_json(OUTPUTS / "latest_training_run.json")
    bid_cfg = _load_json(OUTPUTS / "model_weighted_bce" / "bid_model" / "model_config.json")
    ask_cfg = _load_json(OUTPUTS / "model_weighted_bce" / "ask_model" / "model_config.json")
    test = pd.read_csv(OUTPUTS / "test_backtest_summary.csv").iloc[0]

    lines = [
        "# Reproducibility Manifest",
        "",
        "## Exact Pipeline",
        "",
        "- Experiment root: `new_era/16_binary_label_v2`",
        "- Data source: `data/NEW_ALL`",
        "- Symbol: `TXFB6`",
        f"- Label fill window: `{label_meta['fill_proxy_window_ticks']}` tick",
        f"- Label horizon: `{label_meta['label_horizon_ticks']}` ticks",
        f"- Adverse threshold: `{label_meta['adverse_threshold_ticks']}` ticks",
        f"- Training run: `{train_latest['model_dir']}`",
        "- Model: `HLSFriendlyTCNClassifier`, channels `[16, 16, 16]`, kernel `3`, dropout `0.1`",
        f"- Sampling mode: `{bid_cfg['sampling_mode']}`",
        f"- Epochs: `{bid_cfg['epochs']}`",
        f"- Batch size: `{bid_cfg['batch_size']}`",
        f"- Learning rate: `{bid_cfg['lr']}`",
        f"- Seed: `{bid_cfg['seed']}`",
        f"- Bid pos_weight: `{bid_cfg['pos_weight']:.6f}`",
        f"- Ask pos_weight: `{ask_cfg['pos_weight']:.6f}`",
        f"- Threshold selection split: `{thresholds['selection_split']}`",
        f"- Test used during threshold selection: `{thresholds['test_used']}`",
        f"- Frozen bid threshold: `{thresholds['selected_thresholds']['bid']}`",
        f"- Frozen ask threshold: `{thresholds['selected_thresholds']['ask']}`",
        f"- Final test net PnL: `{float(test['net_pnl']):.2f}`",
        "",
        "## One-Block Reproduction Command",
        "",
        "```bash",
        "python new_era/16_binary_label_v2/01_reproduce_016.py",
        "python new_era/16_binary_label_v2/02_preprocess_quote_eligible.py",
        "python new_era/16_binary_label_v2/03_make_binary_labels.py",
        "python new_era/16_binary_label_v2/04_train_binary_model.py --sampling_mode weighted_bce",
        "python new_era/16_binary_label_v2/05_select_thresholds.py",
        "python new_era/16_binary_label_v2/06_backtest_binary_cancel.py",
        "python new_era/16_binary_label_v2/08_generate_audits.py",
        "python new_era/16_binary_label_v2/09_filter_waterfall.py",
        "python new_era/16_binary_label_v2/10_robust_random_baseline.py --trials 5000",
        "python new_era/16_binary_label_v2/11_model_metrics.py",
        "python new_era/16_binary_label_v2/12_pnl_decomposition.py",
        "python new_era/16_binary_label_v2/13_feature_ablation.py",
        "```",
    ]
    (results_dir / "reproducibility_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def _write_leakage_audit(results_dir: Path) -> None:
    answers = [
        ("1. Are z-scores causal?", "Yes. Past-only z-score uses `shift(1)` before rolling in `02_preprocess_quote_eligible.py:78`."),
        ("2. Are centered windows used?", "No. The only normalization window is `shift(1).rolling(...)`, documented in `02_preprocess_quote_eligible.py:351`."),
        ("3. Are labels allowed to use future data?", "Yes, labels use future data only as targets. `03_make_binary_labels.py` defines `future_mid_return_H_ticks` and fill proxies, not features."),
        ("4. Is train balancing isolated from val/test?", "Yes. Split construction is explicit in `04_train_binary_model.py:104`, and downsampling only branches inside train data at `04_train_binary_model.py:387`."),
        ("5. Was threshold selection validation-only?", "Yes. `05_select_thresholds.py:274-275` writes `selection_split='val'` and `test_used=False`."),
        ("6. Does threshold selection read test predictions?", "No. `05_select_thresholds.py` reads only `val_predictions_bid.parquet` and `val_predictions_ask.parquet` from the model directory."),
        ("7. Does the backtest use frozen thresholds?", "Yes. `06_backtest_binary_cancel.py:293` reads `selected_thresholds`, and line `295` rejects non-validation threshold files."),
        ("8. Is test evaluated once?", "Yes. `06_backtest_binary_cancel.py:325` stamps `test_evaluation_count=1` into the summary."),
        ("9. Are raw data or 015 baselines overwritten?", "No. All 016-v2 artifacts write under `new_era/16_binary_label_v2/outputs` or `results`."),
        ("10. Is aggressor-side ground truth used?", "No. The pipeline explicitly uses a proxy because true `aggressor_side` and queue position are unavailable."),
    ]
    lines = ["# Leakage Audit", "", "Advisor 10-question checklist:", ""]
    for q, a in answers:
        lines.extend([f"## {q}", a, ""])
    (results_dir / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _write_label_definition(results_dir: Path) -> None:
    lines = [
        "# Formal Binary Label Definition",
        "",
        "Let `t` denote a quote-eligible event index within a single session.",
        "",
        "- Current best bid: `b_t`.",
        "- Current best ask: `a_t`.",
        "- Current mid: `m_t = (a_t + b_t) / 2`.",
        "- Tick size: `q`.",
        "- Fill horizon: `tau_fill = 1` raw quote tick.",
        "- Adverse horizon: `H = 10` raw quote ticks.",
        "- Adverse threshold: `theta = 0.5` ticks.",
        "",
        "Bid fill proxy:",
        "",
        "`F_bid(t) = 1{min_trade_price[t+1, t+tau_fill] <= b_t OR min_ask[t+1, t+tau_fill] <= b_t}`",
        "",
        "Ask fill proxy:",
        "",
        "`F_ask(t) = 1{max_trade_price[t+1, t+tau_fill] >= a_t OR max_bid[t+1, t+tau_fill] >= a_t}`",
        "",
        "Future mid return in ticks:",
        "",
        "`r_H(t) = (m_{t+H} - m_t) / q`",
        "",
        "Labels:",
        "",
        "`bid_toxic(t) = 1{F_bid(t)=1 and r_H(t) <= -theta}`",
        "",
        "`ask_toxic(t) = 1{F_ask(t)=1 and r_H(t) >= theta}`",
        "",
        "Interpretation: this is a side-specific maker quote-cancel risk target, not an active alpha label and not true fill truth.",
    ]
    (results_dir / "binary_label_definition.md").write_text("\n".join(lines), encoding="utf-8")


def _write_threshold_report(results_dir: Path) -> None:
    payload = _load_json(OUTPUTS / "selected_thresholds.json")
    test = pd.read_csv(OUTPUTS / "test_backtest_summary.csv").iloc[0]
    lines = [
        "# Threshold Selection Report",
        "",
        "Thresholds were frozen before test evaluation.",
        "",
        f"- Selection split: `{payload['selection_split']}`",
        f"- `test_used`: `{payload['test_used']}`",
        f"- Selection mode: `{payload['selection_mode']}`",
        f"- Target validation cancel rate per side: `{payload['target_cancel_rate']}`",
        f"- Bid threshold: `{payload['selected_thresholds']['bid']}`",
        f"- Ask threshold: `{payload['selected_thresholds']['ask']}`",
        "",
        "Validation evidence:",
        "",
        f"- Bid val cancel rate: `{payload['selected_val_metrics']['bid']['cancel_rate']:.6f}`",
        f"- Ask val cancel rate: `{payload['selected_val_metrics']['ask']['cancel_rate']:.6f}`",
        f"- Combined val proxy cancel value: `{payload['selected_val_metrics']['combined']['proxy_cancel_value_ticks']:.2f}` ticks",
        "",
        "Test evidence, evaluated after freezing:",
        "",
        f"- Test evaluation count: `{int(test['test_evaluation_count'])}`",
        f"- Test net PnL: `{float(test['net_pnl']):.2f}`",
        "",
        "Code references:",
        "",
        "- `05_select_thresholds.py:274-275` writes validation-only metadata.",
        "- `06_backtest_binary_cancel.py:295` rejects threshold files that are not validation-only.",
        "- `06_backtest_binary_cancel.py:383` documents blind application to test.",
    ]
    (results_dir / "threshold_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_repro_manifest(results_dir)
    _write_leakage_audit(results_dir)
    _write_label_definition(results_dir)
    _write_threshold_report(results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static advisor audit markdowns.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS)
    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_args()
    generate(args.results_dir)
    print(f"Audit markdowns written to {args.results_dir}")
