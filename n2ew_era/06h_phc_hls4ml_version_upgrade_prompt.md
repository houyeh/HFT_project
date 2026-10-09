# Prompt：PHC hls4ml 蒸餾學生模型 — 版本升級排查 clang-tidy 卡點

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。
> 這是診斷/環境排查任務，不是模型或架構修改任務。

---

## 背景與現況

`06h_phc_hls4ml_distilled_student/` 已經把 PHC-STP teacher 蒸餾成一個 hls4ml 友善的「學生」模型（拿掉兩個 LayerNorm，只用 Dense/Activation/Multiply/Add/Concatenate），**hls4ml 解析成功**（`convertible=True`，這是比 06G 直接轉 PHC-lite 本人更進一步的結果），但所有 build 嘗試都在 Vitis csynth 階段失敗。

環境：

- hls4ml `1.3.0`（`.venv\Lib\site-packages\hls4ml`）
- TensorFlow `2.21.0` / Keras `3.14–3.15`
- Vitis `2025.2`（`C:\AMDDesignTools\2025.2\Vitis`，SW Build 6295257，2025-11-13）

## 根本原因（這次診斷後的修正版，不要沿用舊結論）

之前以為卡點是：

```
ERROR: [HLS 200-101] config_array_partition: Unknown option '-maximum_size'.
```

**這個判斷是錯的，已被排除。** 這行錯誤被 `build_prj.tcl` 包在 `catch{config_array_partition -maximum_size $maximum_size}` 裡，Vitis 印出 ERROR 後照樣繼續往下跑 `set_part`、`csynth_design`，不是真正造成 build 失敗的原因（見 `results/06h_build_full_log_clamped32_nocsim_settings64fix.txt` 第 30–58 行，錯誤之後 csynth_design 仍被呼叫）。

**真正的終止點**是 Vitis 對 hls4ml 自帶的 `firmware/nnet_utils/*.h` runtime library 做 clang-tidy 語法檢查時失敗：

```
WARNING: [HLS 207-7036] unexpected pragma argument 'softmax', expects function/operation (firmware/nnet_utils/nnet_activation.h:402:36)
WARNING: [HLS 207-6996] 'Resource pragma' is deprecated, use 'bind_op/bind_storage pragma' instead (firmware/nnet_utils/nnet_dense_resource.h:33:13)
...
Syntax check failed in clang-tidy
```

這些警告/失敗全部發生在 hls4ml 自己產生的 library 檔案，不是使用者模型代碼，且橫跨 unclamped 與 clamped32 兩種精度設定都出現同樣的模式。特徵符合「hls4ml 版本落後於 Vitis 版本」——hls4ml 1.3.0 的 nnet_utils 樣板碼很可能沒有針對 Vitis 2025.2（C-Synthesis 用 clang-16 前端）的較新語法檢查規則驗證過。

## 這次要做什麼

**不要再手動修改 hls4ml 產生出來的 firmware/*.h 檔案**（那是產生物，會被下次轉換覆蓋，改了也不持久）。改成在**獨立環境**測試「升級 hls4ml 版本」能不能繞過這個問題。

### 步驟 1：查版本相容性線索（不需要寫程式，先查資料）

- 查 hls4ml 的 PyPI release history（`pip index versions hls4ml` 或直接看 PyPI 頁面）目前最新穩定版是多少，是否高於 1.3.0。
- 查 hls4ml GitHub repo（`fastmachinelearning/hls4ml`）的 issue/PR 或 CHANGELOG，搜尋關鍵字 `clang-tidy`、`Vitis 2025`、`Vitis HLS 2025.2`、`nnet_utils syntax`，確認是否有其他使用者回報同樣問題、以及是否已在較新版本修掉。
- 如果找到明確對應的 hls4ml 版本（例如 changelog 寫「fix nnet_utils pragma syntax for Vitis 2024.x/2025.x」），優先鎖定那個版本，不用盲試最新版。

### 步驟 2：在獨立 venv 測試升級（不要動主專案 .venv）

主專案 `.venv` 被 05/06/07/08 各階段依賴（版本鎖定過），**不要直接在裡面升級 hls4ml**，以免弄壞其他已凍結的階段。改成：

```bash
cd n2ew_era/06h_phc_hls4ml_distilled_student
python -m venv .venv_hls4ml_upgrade_test
# Windows:
.venv_hls4ml_upgrade_test\Scripts\activate
pip install tensorflow==2.21.0 keras hls4ml==<步驟1找到的版本或最新版>
```

若新版 hls4ml 對 keras/tensorflow 版本有不同要求，以新版 hls4ml 官方相容表為準，不用死守 2.21.0——但要在報告裡記錄實際安裝的版本組合。

### 步驟 3：用新環境重跑同一個 build（沿用既有模型匯出結果，不用重新蒸餾）

`06h_phc_hls4ml_distilled_student/models/` 底下應該已經有蒸餾好的學生模型權重/架構定義，**不要重新訓練、不要重新蒸餾**，只需要在新環境下重新跑 hls4ml 的 `convert_from_keras_model` → `build`（重用 `01_build_clamped32_nocsim.py` 或 `02_build_clamped32_plain_nocsim.py` 的邏輯，改成用新 venv 的 hls4ml/vitis-run 呼叫）。先跑 `clamped32_plain`（最單純的設定）這一個組合就好，不用一次跑所有變體。

## 絕對不動的部分

- `05c4_phc_stp_cosine_lr` teacher 權重、任何訓練資料、NEW5——完全不碰。
- 學生模型架構本身（Dense/Activation/Multiply/Add/Concatenate 的排列）——不要為了配合工具而改架構，這次只測工具版本。
- 主專案 `.venv`——只在新建的 `.venv_hls4ml_upgrade_test` 裡操作。
- `06f_phc_lite_manual_hls`、`06g_phc_hls4ml_a3d3_retry` 的既有凍結產物——不要覆寫。

## 驗收標準

1. 新環境下的 build log **不再出現** `Syntax check failed in clang-tidy`。
2. `csynth_report_count > 0`（實際產生出 csynth 報告，不只是 log 跑完）。
3. 若成功，回報：DSP/LUT/FF/BRAM 利用率、latency/II cycles，並比照 06H 既有的 `06F Reference Endpoint` 表格格式跟 8,316 cycles/83.16µs 的手刻 HLS 版本並列比較。
4. 若仍失敗：記錄新的失敗訊息（可能是全新的錯誤，不是 clang-tidy 那個），跟這次的 hls4ml/Vitis 版本組合一併回報，不要重複嘗試同一組合超過一次。

## 回報格式

- 實際使用的 hls4ml / TensorFlow / Keras 版本號。
- 是否找到官方 changelog/issue 佐證這是已知版本相容性問題（附連結）。
- 新環境下 build 的完整終止訊息（成功的話是 csynth 報告摘要；失敗的話是完整錯誤堆疊，不要只截前幾行）。
- 明確結論句：「升版 hls4ml 到 X.Y.Z（有/沒有）解決 clang-tidy 卡點」，附上判斷依據。
- 如果成功，明確標註這仍然是「蒸餾學生模型，非 PHC-lite 本人」，並附上 06H 既有的 joint_sign_agreement=0.950（val split）等蒸餾品質數字，避免後續引用時跟 06F 的 bit-exact 驗證混為一談。
