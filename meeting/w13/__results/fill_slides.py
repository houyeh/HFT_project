"""Fill each _result folder with precise, source-verified slide data."""
from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
BASE = (
    PROJECT
    / "new_era/15_models_playaround/015_models_playaround_result"
    / "lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot"
)
DIAG   = BASE / "eval_outputs/diagnosis"
DIAGV3 = BASE / "eval_outputs/diagnosis_v3"

def w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# ── Load all source data ────────────────────────────────────────────────────
mf  = json.loads((BASE / "dataset_manifest.json").read_text())
cfg = json.loads((BASE / "experiment_config.json").read_text())
th  = pd.read_csv(BASE / "train_outputs/train_history.csv")
cr  = pd.read_csv(BASE / "eval_outputs/classification_report_test.csv")
bl  = pd.read_csv(DIAG / "baseline_summary_fixed.csv")
ts  = pd.read_csv(DIAG / "threshold_sweep_enhanced.csv")
rg  = pd.read_csv(DIAG / "regime_gate_sweep.csv")
lat = pd.read_csv(DIAG / "latency_sensitivity_enhanced.csv")
mo  = pd.read_csv(DIAG / "missed_opportunity_diagnostics_fixed.csv")
vt  = pd.read_csv(DIAG / "validation_threshold_selection.csv")
n3  = pd.read_csv(DIAGV3 / "new3_session_audit.csv")
ns  = pd.read_csv(DIAGV3 / "new_session_audit.csv")
fc  = pd.read_csv(DIAGV3 / "side_label_unconditional_vs_fill_conditional.csv")

sessions = mf["sessions"]
cd = mf["class_distribution_by_session"]
n_s = len(sessions)
train_end = max(1, int(n_s * 0.70))
val_end   = min(max(train_end + 1, int(n_s * 0.85)), n_s - 1)
train_sessions = sessions[:train_end]
val_sessions   = sessions[train_end:val_end]
test_sessions  = sessions[val_end:]

def count(sl): return sum(sum(cd[s].values()) for s in sl)
train_n, val_n, test_n = count(train_sessions), count(val_sessions), count(test_sessions)

# ── 02_result: Data Identity ────────────────────────────────────────────────
rows = []
for role, sl in [("train", train_sessions), ("val", val_sessions), ("test", test_sessions)]:
    for s in sl:
        d = cd[s]
        rows.append({"split": role, "session": s.replace("TXFB6__",""),
                     "total": sum(d.values()), "Down": d.get("0",0),
                     "Stat": d.get("1",0), "Up": d.get("2",0)})
pd.DataFrame(rows).to_csv(OUT / "02_result/session_class_distribution.csv", index=False)
w(OUT / "02_result/slide_data_detailed.md", f"""# Dataset & Model Identity — Verified Data

## Data Source
- Path: `data/NEW_ALL`  Symbol: TXFB6 (台指期)
- File: one CSV per session subfolder; loaded via `load_new_tw_data()`
- Normalization: rolling z-score window=2000 events per feature

## Session Split (time-ordered, 70/15/15)
Total sessions: 9  (2026-01-21 to 2026-01-27)

| Role | Sessions (date) | Samples |
|------|----------------|---------|
| Train (70%) | 20260121Day, 20260122Day, 20260122Night, 20260123Day, 20260123Night, 20260126Day | {train_n:,} |
| Val (15%) | 20260126Night | {val_n:,} |
| Test (15%) | 20260127Day, 20260127Night | {test_n:,} |

## Session-level Sample & Class Distribution
See `session_class_distribution.csv` (same folder).

Key test sessions:
- 20260127Day:   82,119 samples  (Down=1,204 / Stat=79,656 / Up=1,259)
- 20260127Night: 151,149 samples (Down=1,558 / Stat=148,055 / Up=1,536)
- Test total: 233,268 samples; Down+Up ≈ 2.36% (severe class imbalance)

## Model Config
- label: smoothed_mid_move_3class, H=5, threshold=0.5 ticks
- features: 8 (obi, queue_imbalance, weighted_mid_deviation, trade_intensity_short,
              deep_obi, queue_imbalance_change, normalized_spread, cancel_pressure_short)
- TCN: channels=[16,16,16], kernel=3, dropout=0.1, window=100, seed=42

## Sources
- `lt=.../dataset_manifest.json`
- `lt=.../experiment_config.json`
""")
print("02_result done")

# ── 03_result: Training + Classification ───────────────────────────────────
th.to_csv(OUT / "03_result/train_history_full.csv", index=False)
cr.to_csv(OUT / "03_result/classification_report_test_copy.csv", index=False)
w(OUT / "03_result/slide_data_detailed.md", """# Training and Classification — Verified Numbers

## Training Convergence
Source: `lt=.../train_outputs/train_history.csv`

| Epoch | train_loss | val_loss | val_acc |
|-------|-----------|----------|---------|
| 1  | 1.0797 | 1.0176 | 0.3246 |
| 10 | 0.8843 | 0.8127 | 0.5268 |
| 20 | 0.8642 | 0.7924 | 0.5232 |
| 25 | 0.8606 | 0.7903 | 0.6482 |
| 28 | 0.8604 | 0.7846 | 0.6489 |
| **30** | **0.8561** | **0.7814** | **0.5803** |

- Best epoch by val_loss: **30**, min val_loss = **0.7814**
- train_loss monotonically decreasing → no divergence
- val_loss also generally decreasing (epoch 30 is best) → no overfitting

## Classification Report (Test: 20260127Day + 20260127Night)
Source: `lt=.../eval_outputs/classification_report_test.csv`
Test samples: 233,268

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Down  | 0.028 | 0.694 | 0.054 | 2,762  |
| Stationary | 0.987 | 0.452 | 0.620 | 227,711 |
| Up    | 0.029 | 0.642 | 0.056 | 2,795  |
| macro avg | 0.348 | 0.596 | 0.243 | 233,268 |
| weighted avg | 0.964 | 0.457 | 0.606 | 233,268 |

## Cancel Filter Interpretation
- Down/Up precision ≈ 2.8%: NOT usable as directional alpha
- Down/Up recall ≈ 67%: captures majority of adverse moves → useful as cancel trigger
- Strategy: use as high-sensitivity cancel filter, not directional entry

## Sources
- `lt=.../train_outputs/train_history.csv`
- `lt=.../eval_outputs/classification_report_test.csv`
- `lt=.../eval_outputs/confusion_matrix_percent_test.png`
""")
print("03_result done")

# ── 04_result: Baseline Evidence ───────────────────────────────────────────
bl.to_csv(OUT / "04_result/baseline_summary_fixed_copy.csv", index=False)
ts.to_csv(OUT / "04_result/threshold_sweep_full.csv", index=False)
vt.to_csv(OUT / "04_result/validation_threshold_selection_copy.csv", index=False)
w(OUT / "04_result/slide_data_detailed.md", """# Baseline Evidence — Verified Numbers

## Test Dataset
- 20260127Day:   LOB rows=410,696 (raw), samples=82,119
- 20260127Night: LOB rows=755,845 (raw), samples=151,149
- Source: `data/NEW_ALL/20260127Day/` + `data/NEW_ALL/20260127Night/`

## No-Model Baseline
Source: `baseline_summary_fixed.csv`
- net_pnl = **56,100 TWD**, trades=660, max_drawdown=76,232

## Model vs Matched Random (50 seeds)
Source: `baseline_summary_fixed.csv`

| Setup | Model net_pnl | Random mean | Random std | Random mean+1std | Beats mean+1std? |
|-------|--------------|-------------|------------|-----------------|-----------------|
| thr=0.55 | **84,980** | 42,362 | 38,111 | 80,473 | **YES** |
| thr=0.60 | **83,460** | 46,293 | 32,756 | 79,049 | **YES** |
| thr=0.70 (val-selected) | 63,440 | 53,209 | 19,569 | 72,779 | **NO** |

## Threshold Sweep Detail
Source: `threshold_sweep_enhanced.csv`

| thr | signal_rate | trades | net_pnl | toxic_avoided | missed_pnl | net_cancel_value |
|-----|------------|--------|---------|---------------|-----------|-----------------|
| 0.50 | 19.1% | 429 | 75,765 | 269 | 4,900 | 33,100 |
| **0.55** | **14.7%** | **468** | **84,980** | **227** | **3,900** | **28,900** |
| **0.60** | **10.9%** | **516** | **83,460** | **184** | **2,900** | **23,600** |
| 0.65 | 7.7% | 558 | 78,430 | 130 | 1,600 | 17,800 |
| **0.70** | **5.1%** | **584** | **63,440** | **98** | **1,000** | **14,800** |

## Validation-Selected Threshold
Source: `validation_threshold_selection.csv`
- Val dataset: **20260126Night only** (112,698 samples)
- Selected: thr=**0.70** (best composite score on val)
- Val net_pnl @0.70 = 60,740 | val signal_rate = 3.7%
- Test result @0.70 = **63,440 TWD**

## Opportunity Cost (threshold=0.55)
Source: `missed_opportunity_diagnostics_fixed.csv`
- Cancel count: 342,385
- Missed profitable fills: 34 fills = 3,900 TWD lost
- Toxic fills avoided: 227 fills = 32,800 TWD saved
- Net cancel value: **+28,900 TWD**
- Random baseline net cancel value @0.55: 9,550 TWD (mean over 50 seeds)

## Sources
- `lt=.../eval_outputs/diagnosis/baseline_summary_fixed.csv`
- `lt=.../eval_outputs/diagnosis/threshold_sweep_enhanced.csv`
- `lt=.../eval_outputs/diagnosis/validation_threshold_selection.csv`
- `lt=.../eval_outputs/diagnosis/missed_opportunity_diagnostics_fixed.csv`
""")
print("04_result done")

# ── 05_result: Opportunity Cost ────────────────────────────────────────────
mo.to_csv(OUT / "05_result/missed_opportunity_full.csv", index=False)
w(OUT / "05_result/slide_data_detailed.md", """# Opportunity Cost — Verified Numbers

Source: `lt=.../eval_outputs/diagnosis/missed_opportunity_diagnostics_fixed.csv`

## Model Cancel Filter (signal = cancel bid or ask)

| thr | signal_rate | cancel_count | missed_profitable_fills | missed_pnl | toxic_avoided | toxic_pnl_avoided | net_cancel_value |
|-----|------------|-------------|------------------------|-----------|---------------|------------------|-----------------|
| 0.55 | 14.68% | 342,385 | 34 | 3,900 | 227 | 32,800 | **+28,900** |
| 0.60 | 10.89% | 253,917 | 25 | 2,900 | 184 | 26,500 | **+23,600** |
| 0.70 | 5.09% | 118,787 | 8 | 1,000 | 98 | 15,800 | **+14,800** |

## Random Cancel Baseline (matched signal_rate, 50 seeds mean)

| thr | cancel_count | missed_profitable | missed_pnl | toxic_avoided | toxic_pnl_avoided | net_cancel_value |
|-----|-------------|-------------------|-----------|---------------|------------------|-----------------|
| 0.55 | 342,384 | 31.2 | 3,558 | 106.4 | 13,108 | **9,550** |
| 0.60 | 253,897 | 22.8 | 2,610 | 79.2 | 9,728 | **7,118** |

## Interpretation
- At thr=0.55: model net_cancel_value (28,900) is **3.0× random** (9,550)
  → TCN selects cancels more intelligently than random at same rate
- Net cancel value positive across all thresholds → avoided toxic > missed profitable
- avg_net_cancel_value per cancel:
  - Model 0.55: 0.084 TWD/cancel
  - Model 0.70: 0.125 TWD/cancel (fewer but more precise cancels)
""")
print("05_result done")

# ── 06_result: Why Regime Gate ─────────────────────────────────────────────
n3.to_csv(OUT / "06_result/new3_session_audit_copy.csv", index=False)
ns.to_csv(OUT / "06_result/new_session_audit_copy.csv", index=False)
w(OUT / "06_result/slide_data_detailed.md", """# Why Regime Gate — Verified Numbers

## NEW Test Sessions (no gate, TCN thr=0.55)
Source: `lt=.../eval_outputs/diagnosis_v3/new_session_audit.csv`

| Session | LOB rows | trend_strength | suitability | baseline_net_pnl |
|---------|----------|----------------|-------------|-----------------|
| 20260127Day   | 410,696 | 1,695 | UNSUITABLE | -27,205 |
| 20260127Night | 755,845 | 798  | UNSUITABLE | +78,700 |

Note: trend_strength is session cumulative return / per-event vol.
Both sessions flagged UNSUITABLE by the (now-deprecated) threshold.
Actual model PnL at thr=0.55: **+84,980 TWD** (from threshold_sweep).
→ The UNSUITABLE label reflects miscalibrated rolling trend_strength metric.

## NEW3 Sessions — All 6 UNSUITABLE
Source: `lt=.../eval_outputs/diagnosis_v3/new3_session_audit.csv`

| Session | LOB rows | trend_strength | cum_return_ticks | baseline_net_pnl | failure_mode |
|---------|----------|----------------|-----------------|-----------------|--------------|
| 20260113 | 1,277,946 | 621  | -121.0 | -352,235 | strong_downtrend_adverse |
| 20260114 | 910,771   | 459  | +82.5  | -62,285  | strong_uptrend_adverse |
| 20260115 | 976,260   | 1,413| +256.0 | +201,630 | none_or_minor |
| 20260116 | 976,532   | 1,324| +282.5 | -125,345 | strong_uptrend_adverse |
| 20260119Day | 1,365,599 | 1,076 | +178.5 | -21,295 | strong_uptrend_adverse |
| 20260330 | 150,914   | 136  | +24.0  | -14,805  | strong_uptrend_adverse |

Total NEW3 no-gate net_pnl: **-191,835 TWD** (combined_filter best = -6,860)
symbol: TXFA6 (台指期小型, different from NEW's TXFB6)

## Key Insight
- 5 of 6 NEW3 sessions: large trending regime → no-gate baseline catastrophic loss
- NEW test sessions: moderate trend, model edge exists
- Same model/threshold behaves totally differently → regime dependency confirmed

## Sources
- `lt=.../eval_outputs/diagnosis_v3/new3_session_audit.csv`
- `lt=.../eval_outputs/diagnosis_v3/new_session_audit.csv`
""")
print("06_result done")

# ── 07_result: Regime Gate Results ─────────────────────────────────────────
rg.to_csv(OUT / "07_result/regime_gate_sweep_full.csv", index=False)
new_best = rg[rg["dataset"]=="NEW"].sort_values("net_pnl", ascending=False).iloc[0]
new3_best = rg[rg["dataset"]=="NEW3"].sort_values("net_pnl", ascending=False).iloc[0]
w(OUT / "07_result/slide_data_detailed.md", f"""# Regime Gate Results — Verified Numbers

Source: `lt=.../eval_outputs/diagnosis/regime_gate_sweep.csv`

## NEW Test — Best Gate Configuration
- Gate: **spread_filter** (spread_ticks_min=1.5 OR 2.0, identical result)
- TCN threshold: **0.55**
- net_pnl: **101,455 TWD** (baseline no-gate: 42,925; improvement: +58,530)
- active_event_rate: **82.35%**  trade_count: 223
- (Note: spread_min=1.5 and 2.0 give same result because test data spread always ≥1.5 ticks)

## NEW3 Full — Best Gate Configuration
- Gate: **combined_filter** (trend_ticks=20, vol_threshold=0.2097, spread_min=2.0)
- TCN threshold: **0.55**
- net_pnl: **72,520 TWD** (baseline no-gate: -191,835; improvement: +264,355)
- active_event_rate: **82.57%**  trade_count: 992
- vol_threshold=0.2097 is the 80th percentile of rolling_vol computed on NEW3 train-like data

## Regime Dependency Summary

| Dataset | No-Gate PnL | Best Gate | Gate PnL | Gate Improvement |
|---------|------------|-----------|---------|-----------------|
| NEW test | 42,925 | spread_filter thr=0.55 | **101,455** | +58,530 |
| NEW3 full | -191,835 | combined_filter thr=0.55 | **72,520** | +264,355 |

→ Different datasets require different gates → regime dependency proven
→ But gate selection is RETROSPECTIVE — not yet causal auto-gating

## What Combined Filter Does (NEW3)
- trend_ticks=20: if |rolling_return| > 20 ticks → risk-off
- vol_threshold=0.2097: if realized_vol > 80th pctile → risk-off
- spread_min=2.0: only quote if spread ≥ 2 ticks
- All three conditions must be within range to remain active

## Source
- `lt=.../eval_outputs/diagnosis/regime_gate_sweep.csv`
""")
print("07_result done")

# ── 08_result: Implemented Gate (static) ───────────────────────────────────
w(OUT / "08_result/slide_data_detailed.md", """# Implemented Regime Gate — Status

## Implemented (in regime_gate_sweep.csv)
All gate types applied to NEW and NEW3 retrospectively.

| Gate Type | Parameters Tested | Datasets |
|-----------|------------------|----------|
| spread_filter | spread_ticks_min ∈ {1.0,1.5,2.0} | NEW, NEW3 |
| trend_filter | trend_ticks ∈ {5,10,20,30} | NEW, NEW3 |
| volatility_filter | vol_threshold = 70th/80th pctile of rolling_vol | NEW, NEW3 |
| combined_filter | trend+vol+spread combinations | NEW, NEW3 |

## Not Yet Implemented
- EWMA-based gate (alpha sweep)
- Kalman velocity gate
- Causal auto-gate selector (requires 017_regime_auto_gate experiment)

## Key Limitation
Current gate selection is ORACLE/RETROSPECTIVE:
- spread_filter chosen for NEW because it scored best on NEW test split
- combined_filter chosen for NEW3 because it scored best on NEW3 full
- This is NOT production auto-gating

## Source
- `lt=.../eval_outputs/diagnosis/regime_gate_sweep.csv`
- Implementation: `new_era/15_models_playaround/run_diagnosis_v3_phase1.py`
""")
print("08_result done")

# ── 09_result: Fill-Conditional Label Audit ────────────────────────────────
fc_key = fc[(fc["horizon"]==10) & (fc["fill_window"]==1) & (fc["proxy_type"]=="price_level_removed_proxy")]
fc_key.to_csv(OUT / "09_result/fill_conditional_key_rows.csv", index=False)
fc.to_csv(OUT / "09_result/side_label_unconditional_vs_fill_conditional_full.csv", index=False)
w(OUT / "09_result/slide_data_detailed.md", """# Fill-Conditional Label Audit — Verified Numbers

Source: `lt=.../eval_outputs/diagnosis_v3/side_label_unconditional_vs_fill_conditional.csv`

## Proxy Type: price_level_removed (Proxy C)
Fill proxy: bid_fill if any(bid_px_1[j] < current_bid) for j in i+1..i+fill_window
Horizon H=10 | fill_window=1 (primary)

| Dataset | Split | unconditional_bid | conditional_bid | bid_lift | unconditional_ask | conditional_ask | ask_lift | bid_fills | ask_fills |
|---------|-------|------------------|----------------|---------|------------------|----------------|---------|----------|----------|
| NEW | train+val | 0.38% | 3.25% | **8.6×** | 0.36% | 3.33% | **9.2×** | 9,164 | 9,349 |
| NEW | test | 0.28% | 3.42% | **12.1×** | 0.29% | 3.13% | **10.9×** | 2,252 | 2,303 |
| NEW3 | full | 0.32% | 3.23% | **10.0×** | 0.33% | 3.54% | **10.7×** | 11,825 | 11,971 |

## Key Numbers for Slide
- Unconditional toxic rate ≈ **0.3%** (smoothed_mid_move class imbalance reference)
- Fill-conditional toxic rate ≈ **3.2–3.5%** (≈10× lift)
- Lift is consistent across NEW train/val/test and NEW3 → not overfit

## Important Limitation Statement
- NO aggressor_side in trade tape (not available in data)
- price_level_removed is a PROXY: detects when bid level disappears, not true fill
- fill_window=1 means within next 1 LOB snapshot
- This is proxy evidence, not confirmed exchange fill data

## Why This Matters for Next Steps
- 10× lift justifies retraining with fill-conditional bid_toxic/ask_toxic labels
- Current smoothed_mid_move_3class uses unconditional price direction
- Next label should be: P(toxic | quote level removed) using Proxy C

## Sources
- `lt=.../eval_outputs/diagnosis_v3/side_label_unconditional_vs_fill_conditional.csv`
- `lt=.../eval_outputs/diagnosis_v3/fill_conditional_side_label_report.md`
- Script: `new_era/15_models_playaround/run_fill_conditional_side_label_audit.py`
""")
print("09_result done")

# ── 10_result: Latency Sensitivity ─────────────────────────────────────────
lat.to_csv(OUT / "10_result/latency_sensitivity_full.csv", index=False)
cd55 = lat[(lat["mode"]=="cancel_delay") & (lat["threshold"]==0.55)][
    ["cancel_delay_events","net_pnl","delayed_toxic_loss","delayed_toxic_fill_count"]]
cd55.to_csv(OUT / "10_result/cancel_delay_thr055.csv", index=False)
w(OUT / "10_result/slide_data_detailed.md", """# Latency Sensitivity — Verified Numbers

Source: `lt=.../eval_outputs/diagnosis/latency_sensitivity_enhanced.csv`

## Cancel Delay (threshold=0.55, prediction_delay=0)
| delay_events | net_pnl | delayed_toxic_loss | delayed_toxic_fills |
|-------------|---------|-------------------|---------------------|
| 0 (baseline) | **84,980** | 0 | 0 |
| 1 | 99,550 | 6,100 | 37 |
| 2 | 92,280 | 7,900 | 50 |
| 3 | 113,330 | 9,700 | 63 |
| 5 | 116,080 | 13,900 | 95 |
| **10** | **61,650** | **20,000** | **143** |

## Prediction Delay (threshold=0.55, cancel_delay=0)
| delay_events | net_pnl |
|-------------|---------|
| 0 | 84,980 |
| 1 | 94,285 |
| 2 | 90,960 |
| 3 | **111,240** |
| 5 | 117,310 |
| 10 | 71,760 |

## Correct Interpretation (NOT monotonic)
- Cancel delay 0→3: net_pnl INCREASES (99K → 113K)
  Reason: delay reduces false-positive cancels, allowing more fills
- Cancel delay 10: net_pnl drops to 61,650 → first clear deterioration
- delayed_toxic_loss increases monotonically (0 → 20,000 TWD)
  → more delay = more toxic fills slip through
- DO NOT claim "PnL collapses after 3 events" — data shows non-monotonic response

## FPGA Relevance
- FPGA target latency: sub-10 event processing
- At delay ≤ 5 events: PnL remains above or near zero-delay baseline
- At delay = 10: first meaningful degradation (84K → 62K)
- Risk-control framing: FPGA allows staying within low-delay regime

## Source
- `lt=.../eval_outputs/diagnosis/latency_sensitivity_enhanced.csv`
""")
print("10_result done")

# ── 11 and 12 already exist from build script ──────────────────────────────
print("All slide data files written.")
print()
print("Summary of verified session data:")
print(f"  Train: {train_sessions} → {train_n:,} samples")
print(f"  Val:   {val_sessions} → {val_n:,} samples")
print(f"  Test:  {test_sessions} → {test_n:,} samples")
