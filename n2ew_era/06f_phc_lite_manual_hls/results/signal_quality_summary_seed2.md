# Signal Quality Summary For Deployed Seed 2

Lineage basis: Task 3 confirms the HLS-exported checkpoint is Phase 05C4 `PHCLiteSTP` / `phc_stp_v2_cosine_lr`, seed 2. This table therefore uses Phase 05C4 seed-2 outputs only, matching the actual checkpoint exported to HLS.

| split   | total_excess_pnl_H20   |   win_rate | empirical_p_value(matched random)   | beats_queue_rule   | beats_hgb   |
|:--------|:-----------------------|-----------:|:------------------------------------|:-------------------|:------------|
| val     | 2,037,925              |   0.473656 | < 0.001                             | True               | True        |
| new3    | 5,237,410              |   0.505113 | < 0.001                             | True               | True        |
| new4    | 1,871,685              |   0.495543 | < 0.001                             | True               | True        |

Source files:
- `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_oos_eval.csv`
- `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_matched_random_summary.csv`
- `n2ew_era/05c4_phc_stp_cosine_lr/results/phc_stp_v2_vs_v1_hgb_queue.csv`

「此驗證基於量化前float32模型，尚未驗證ap_fixed<24,12>定點化後是否保留
同等訊號品質；causal-fill proxy為模擬撮合，非真實exchange replay；
不可據此聲稱FPGA本身改善了PnL。」
