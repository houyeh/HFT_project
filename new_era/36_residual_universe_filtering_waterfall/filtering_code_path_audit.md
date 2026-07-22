# Filtering Code Path Audit

| Stage | Code/data path | Function or artifact | Filtering role |
| --- | --- | --- | --- |
| Binary label frame | `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\16_binary_label_v2\outputs\binary_labels_v2.parquet` | artifact | Already quote-eligible historical rows; loaded by `load_label_split`. |
| Split loader | `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\28_residual_quote_ev_veto\common_028.py:118` | `load_label_split` | Reads `binary_labels_v2.parquet`, filters by split, sorts by session/local row, adds `event_id`. |
| Historical train action table | `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\26_robust_baseline_side_aware\baseline_utils.py:75` | `actions_from_frame` | Builds sparse passive-fill candidate actions and computes moderate proxy EV for train. |
| Val/test action table | `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py:200` | `load_actions_for_split` | Reuses 018/root-cause action context for val/test and computes stress EV columns if missing. |
| NEW4 action table | `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py:128` and `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\27_non_ml_final_baseline_audit\common_027.py:154` | `build_new4_frame`, `build_new4_actions` | Loads external OOS valid-book frame and builds candidate actions; no 016 train/val label parquet exists for NEW4. |
| Baseline policy | `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\26_robust_baseline_side_aware\baseline_utils.py:138` | `apply_non_ml_policy` | `bid_only_spread_ge_1` keeps bid-side actions with `spread_ticks >= 1`. |
| Residual ML universe | `C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\28_residual_quote_ev_veto\common_028.py:194` | `baseline_universe` | Materializes only baseline-selected actions and labels `negative_ev_moderate = quote_ev_moderate < 0`. |
