# Fill-Conditional Label Audit — Verified Numbers

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
