# Colab 執行說明：空間前端規模掃描

## 為什麼改用 Colab

本機實測：**C=32 的單 epoch 需 733 秒**（C=8 為 58 秒）。成本隨 C² 增長，推算 C=64 約 2,900 秒/epoch。

| C | 每 epoch | 3 seeds 總計 |
|---:|---:|---:|
| 8 | 58 s | 2.4 h ✓（已完成） |
| 32 | 733 s | **27 h** |
| 64 | ~2,900 s | **108 h** |

CPU 不可行。空間卷積是 GPU 的強項，且 Colab 不消耗 Delta 的 SU 額度。

**額外好處**：Colab 的檔案系統與專案 repo 完全分離，**不可能污染既有的 C=8 結果**。

---

## 要上傳的檔案（共 62 MB）

```
14_train_spatial_frontend_probe.py
02_train_phc_lite_stage1.py
fi2010_phc_lite.py
fi2010_tcn.py
spatial_conv_front_end.py
colab_spatial_scaling.py
data/cache/train_cf7.npz
data/cache/test_cf7.npz
data/cache/test_cf8.npz
data/cache/test_cf9.npz
```

**建議先打包成 zip 上傳到 Google Drive**，比逐檔上傳可靠得多（Colab 直傳大檔容易中斷）。

---

## Colab 儲存格

### 儲存格 1：掛載 Drive 並解壓

```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/work
!unzip -o -q "/content/drive/MyDrive/fi2010_spatial.zip" -d /content/work
%cd /content/work
!ls -la
!ls -la data/cache
```

### 儲存格 2：確認是 GPU 執行階段

```python
!nvidia-smi
```

沒有輸出就去：**執行階段 → 變更執行階段類型 → T4 GPU**。

### 儲存格 3：設定決定性環境（**必須在任何 torch import 之前**）

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print(os.environ["CUBLAS_WORKSPACE_CONFIG"])
```

> ⚠️ 這一行若設在 `import torch` 之後，`use_deterministic_algorithms` 會在第一次 cuBLAS 呼叫時直接拋錯。
> 若此儲存格之前已經 import 過 torch，**請先「執行階段 → 重新啟動執行階段」再從頭跑**。

### 儲存格 4：🔴 移植一致性閘（先做這個，不通過就停）

```python
!python colab_spatial_scaling.py --stage check
```

它會跑 **C=8 seed0 共 3 個 epoch**，與本機 CPU 的前三個 epoch 逐項比對。

| 判定 | 條件（epoch 1 `train_loss` 相對差） | 下一步 |
|---|---|---|
| `PASS` | < 1e-3 | 進入儲存格 5 |
| `INVESTIGATE` | 1e-3 ～ 1e-1 | **停**，把輸出貼出來 |
| `FAIL` | > 1e-1 | **停**，移植有結構性問題 |

本機 CPU 參考值：**epoch 1 `train_loss` = 1.0702502296431846**，epoch 1 耗時 106.3 s。

> 這一步也順便告訴你 GPU 快多少：看輸出裡的 `epoch1 = ? s` 對照 106.3 s。

### 儲存格 5：C=32（3 seeds）

```python
!python colab_spatial_scaling.py --stage c32
```

### 儲存格 6：C=64（3 seeds）—— 參數配對點

```python
!python colab_spatial_scaling.py --stage c64
```

### 儲存格 7：產生報告區塊

```python
!python colab_spatial_scaling.py --stage report
```

輸出會寫到 `/content/drive/MyDrive/fi2010_spatial_scaling/REPORT_BLOCK_spatial.md`，貼給我即可。

---

## 防斷線

Colab 免費版會在閒置約 90 分鐘或總時長約 12 小時後斷線。

**本腳本的保護**：每個 seed 跑完**立刻**把 metrics / curves / log 複製到 Drive。斷線最多只損失當前那一個 run，已完成的不會消失。

**建議**：
- 儲存格 5 和 6 **分開跑**，不要一次排隊。C=32 預估 20–60 分鐘，C=64 預估 1–3 小時。
- 跑的時候別關瀏覽器分頁。
- 若 C=64 中途斷線，重跑儲存格 6 即可——已完成的 seed 會被 `14_` 的 `metric_exists` 檢查跳過（除非加 `--force-rerun`）。

---

## 固定配方（不可更動）

與既有 C=8 探針完全相同：

```
model-kind phc / horizon 50 / hidden-dim 16 / kernel-size 3 / dropout 0.1
loss raw / max-epochs 60 / patience 10 / min-delta 1e-5
batch 512 / eval-batch 1024 / lr 1e-3 / lr-min 1e-5
weight-decay 0.0 / grad-clip 5.0
```

**唯一變因是 `--front-channels`。**

三個與環境有關、不影響數值的調整：
- `--device cuda`
- `--skip-preflight`（Colab 無 OneDrive/Anaconda，該守門無意義）
- `--max-expected-epoch-seconds 3600`（GPU 上不會觸發，調高避免誤殺；本機正是被預設的 240 s 殺掉的）

---

## 這個實驗要回答什麼

| 設計 | 參數量 | 把參數花在哪 | 結果 |
|---|---:|---|---|
| hidden64（容量軸） | **23,876** | 加寬隱藏層 | macro 63.88 ± 0.55，**−0.09 pp** |
| **spatial C=64（表徵軸）** | **~23,252** | 加大空間前端 | **待測** |

**兩者參數量只差 2.6%。** 同樣的預算、兩種花法，直接回答「資源多了該怎麼花」。

C=32（~8,276 參數）則落在 TCN scratch（8,547 參數、weighted-F1 67.60）旁邊，多一個現成對照。
