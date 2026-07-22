# 015 模型試驗系統（playaround）

這個資料夾提供「**01_xxx → 02_xxx → 03_xxx ...**」的**可執行順序腳本**（好追蹤、好重跑）。

> 注意：因為 Python 的 `import` 規則限制（模組/套件名稱不能以數字開頭），核心可重用程式碼放在 `new_era/models_playaround_15/`（可被正常 import）。
> 你平常只需要跑這個資料夾的 `01_*.py`～`07_*.py` 就好。

## 執行順序（建議）

0. `00_run_all_one_click.py`：一鍵跑完一批 ablation（label_type × feature_count），最後自動產 summary/report + `_charts/` 統整圖表
1. `01_dataset_generation.py`：產生 sequence dataset（含 label）
2. `02_train_tcn.py`：訓練 TCN + checkpoint + history
3. `03_evaluate_tcn.py`：輸出完整評估圖表與 CSV（confusion matrix / PR / ROC / calibration / threshold diagnostics…）
4. `04_backtest_val_sweep_then_test.py`：validation sweep 選 threshold → test 固定一次
5. `05_run_matrix_batch1.py`：跑第一批 label types（mid_move/smoothed/toxic_bid/toxic_ask）× feature_count=6/7/8/9
6. `06_run_matrix_batch2.py`：跑第二批 label types（adverse_fill / active_profitable）× feature_count=6/7/8/9
7. `07_generate_summary_report.py`：彙整產出 `experiment_summary.csv` + `experiment_report.md`

### 常用參數（00_run_all_one_click.py）

- 只跑 batch1：`python new_era/15_models_playaround/00_run_all_one_click.py --batches 1`
- 跑 batch1+batch2：`python new_era/15_models_playaround/00_run_all_one_click.py --batches 1,2`
- 只重跑 backtest（不重訓）用來更新 backtest 輸出：`python new_era/15_models_playaround/00_run_all_one_click.py --batches 1,2 --only-backtest`

## 你通常要改的地方（最重要）

請直接改 `_common.py` 裡的常用參數區塊：

- `LABEL_TYPE`
- `FEATURE_COUNT`
- `WINDOW_SIZE`, `HORIZON`, `THRESHOLD_TICKS`
- `ADVERSE_THRESHOLD_TICKS`（toxic/adverse_fill 類 label 會用）
- training/backtest 的 fee、inventory 等

## 輸出位置

每個 experiment 會寫到：

`new_era/15_models_playaround/015_models_playaround_result/<experiment_id>/`

## Windows 路徑長度注意

為了避免 Windows `MAX_PATH` 造成 `matplotlib.savefig(...)` 噴 `FileNotFoundError`，`<experiment_id>` 會使用短格式：

- 例：`lt=toxic_bid_binary__fc=6__w=100__H=10__thr=1__adv=1`

舊的長格式資料夾（`label_type=...`）在執行 pipeline 時會自動嘗試搬移/合併到新的短格式資料夾。

## 環境快照

每次執行 `00_run_all_one_click.py` 會在 results root 產生：

- `environment_snapshot.json`
- `pip_freeze.txt`

用來把同一套環境搬到另一台機器或 Colab。
