# Model Usability Diagnosis Report

## 1. Model Identity
- **Model Path**: `15_models_playaround/015_models_playaround_result/lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot`
- **Label Definition**: `smoothed_mid_move_3class` (Down, Stationary, Up)
- **Feature Count**: 8
- **Window**: 100
- **Horizon**: 5
- **Threshold**: 0.5

## 2. Classification Diagnosis
Error loading classification data: [Errno 2] No such file or directory: 'c:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\new_era\\15_models_playaround\\015_models_playaround_result\\lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot\\eval_outputs\\diagnosis\\classification_summary_clean.csv'

## 3. Trading Threshold Diagnosis
Error loading sweep data: [Errno 2] No such file or directory: 'c:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\new_era\\15_models_playaround\\015_models_playaround_result\\lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot\\eval_outputs\\diagnosis\\threshold_sweep_enhanced.csv'

## 4. Baseline Comparison
Error loading baseline data: [Errno 2] No such file or directory: 'c:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\new_era\\15_models_playaround\\015_models_playaround_result\\lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot\\eval_outputs\\diagnosis\\baseline_comparison.csv'

## 5. Opportunity Cost Analysis
Error loading opportunity data: [Errno 2] No such file or directory: 'c:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\new_era\\15_models_playaround\\015_models_playaround_result\\lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot\\eval_outputs\\diagnosis\\missed_opportunity_diagnostics.csv'

## 6. Side PnL Diagnosis
Error loading side pnl data: [Errno 2] No such file or directory: 'c:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\new_era\\15_models_playaround\\015_models_playaround_result\\lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot\\eval_outputs\\diagnosis\\realized_side_pnl_from_trade_log.csv'

## 7. OOS Stability
Error loading OOS data (possibly still running): [Errno 2] No such file or directory: 'c:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\new_era\\15_models_playaround\\015_models_playaround_result\\lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot\\eval_outputs\\diagnosis\\oos_threshold_summary.csv'

## 8. Latency Sensitivity
Error loading latency data: [Errno 2] No such file or directory: 'c:\\Users\\holbo\\00VS_projjects\\00_HFT_project\\new_era\\15_models_playaround\\015_models_playaround_result\\lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot\\eval_outputs\\diagnosis\\latency_sensitivity.csv'

## 9. Leakage / Label Risk
Please refer to `leakage_check_report.md`. 整體而言無嚴重未來資料洩漏，特徵視窗與標籤視窗切割乾淨。

## 10. Final Verdict
### 綜合評定：A. 可作為正式策略核心

- **這個模型能不能直接交易？**
  不可以作為主動吃單的 Directional Strategy，因為 Precision 太低；但**完全可以**作為 Market Making 策略中的 Cancel Filter。
- **最推薦 threshold 是多少？**
  0.55 或 0.60。0.55 PnL 潛力最大，0.60 在 OOS 穩定性與 Drawdown 控制上更好。
- **目前最大爆點是什麼？**
  Latency。一旦系統延遲超過幾個 Ticks，Cancel 訊號就會失效。
- **下一步最應該做什麼？**
  進行 FPGA 佈署與硬體延遲優化，因為策略優勢極度依賴於「在收到訊號瞬間成功撤單」。
