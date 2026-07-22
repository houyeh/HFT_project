# HFT 下一步計畫

> 生成時間：2026-05-18｜基於全專案掃描 + 所有 backtest 結果

---

## 一、現況診斷（讀完資料後的結論）

### 資料版圖

| 資料集 | 日期區間 | sessions | LOB 行數 | 用途 |
|--------|----------|----------|---------|------|
| `data/NEW` (訓練時用) | 2026-01 初 (推測) | ~15 sessions | ~575萬 | 15_playaround 訓練/val/test |
| `data/NEW1` | 20260123–20260127 | 6 sessions (Day+Night) | 未量但OOS結果存在 | OOS hybrid backtest |
| `data/NEW2` | 20260121–20260122 (3 sessions) | 3 sessions | 247 萬 | 完全OOS baseline + hybrid |
| `data/NEW3` | 20260113–20260516 (7 folders) | 7+ sessions | 未量 | **完全沒用過的新鮮資料** |

> **你說的資料量夠不夠見證市場様式？**
> 目前訓練資料只有 1–2 月的台指期（TXFB6 合約）。七種日內模式（開盤/午盤/收盤/夜盤波動/橫盤）重複出現過，但**跨月換約、重大事件、流動性枯竭**等極端態樣你幾乎沒碰到。以「部署到FPGA前要先確認穩健性」這個標準，**不夠**——NEW3 包含 3 月、5 月資料正好是你最缺的跨月驗證。

### 15_models_playaround 結果摘要

| 排名 | label_type | fc | test PnL(flatten) | 改善 | max DD |
|-----|------------|----|------------------|------|--------|
| 🥇 | smoothed_mid_move_3class | 8 | 169,875 TWD | +79,385 | 25,420 |
| 🥈 | mid_move_3class | 8 | 144,925 TWD | +54,435 | 31,550 |
| 🥉 | mid_move_3class | 9 | 120,925 TWD | +30,435 | 25,385 |
| 4 | active_buy_profitable_binary | 7 | 102,520 TWD | +12,030 | 37,960 |
| — | toxic/adverse 全系列 | all | 0 (等同 baseline) | 0 | — |

**Baseline（無腦造市）= 90,490 TWD**（在相同訓練期 test set 上）

> [!WARNING]
> 但 NEW2（完全OOS）上的 baseline 是 **−113,370 TWD**（虧損）；OOS hybrid backtest 上的 model 仍然是虧損（只是虧得少一點）。這代表目前模型「在訓練期 test 上看起來有效，但真正上線就爛」的過擬合跡象非常明顯。

### 根本問題診斷

1. **Train/Val/Test 都用同一批 NEW 資料**（同合約、同月份）→ test set 本質上還是 in-distribution。
2. **OOS 結果（NEW1/NEW2）** 顯示 baseline 本身就虧錢，表示那幾天市場單向走勢強，造市策略天然不適合。
3. **Macro F1 只有 0.30**，代表 Down/Up 類別的預測能力極弱（因為極度不平衡：up/down 各約 0.5%）。
4. **adverse/toxic 系列全部失敗**（F1≈0.5 = random），因為這些事件在資料中出現次數極少（訓練集 9 筆、測試集 2 筆）。

---

## 二、核心問題回答

### Q：我現在應該「判斷市場方向→選模型」還是「始終用同一種模型」？

**答：兩者之間有第三條路，而且是最正確的路。**

**階段性答案（你現在的開發進度）：先固定一個模型架構，但做 regime 條件門控。**

- 你現在的資料量和模型複雜度不足以支撐「多模型動態切換」（太容易 overfit 到切換條件本身）。
- 正確做法：**用 1 個 TCN 模型（smoothed_mid_move_3class，fc=8）做核心**，但在它的**輸出端加入 regime filter（波動率門控）**：
  - 高波動（趨勢行情）→ 讓模型方向性訊號影響報價偏斜
  - 低波動（橫盤）→ 回退到純 baseline 造市
- FPGA 部署端也只需要支援一個模型，不需要多模型並行，符合資源限制。

---

## 三、NEW3 資料該怎麼用？

> [!IMPORTANT]
> **NEW3 是你最寶貴的資產，絕對不能用來訓練。**

| 用途 | 做法 | 原因 |
|------|------|------|
| ✅ **純 OOS 驗證** | 只用來跑 backtest，絕對不用來 train/val | 保持無汙染，才能真正驗證泛化能力 |
| ✅ **跨月穩健性驗證** | 3月(20260330) + 5月(20260515) 分開跑 | 驗證換月/不同月份市場特性 |
| ❌ 不要用來訓練 | — | 用了之後你就沒有真正的OOS資料了 |

**立即可做的動作**：把 NEW3 的 7 個 sessions 用 `10_oos_backtest_top3_hybrid.py` 跑一遍（改 `--new2-dir` 指向 NEW3），看 best model 在 1月中旬、3月、5月各自的表現是否一致。

---

## 四、最擬真的回測方式

你目前的回測鏈已有 queue-aware hybrid simulator（`src/backtest/hybrid_market_maker`），這是正確方向。但要達到「最擬真」還差以下幾點：

### 現有問題

1. **訓練 test set 就是 OOS 的假象**：現在 playaround 的 baseline_test_pnl=90,490 是訓練期 data 的最後 15%，不是真正未來資料。
2. **跨 session 庫存沒重置**：真實日內交易每個 session 都要重置倉位。
3. **成交模型過於簡化**：「next_bid ≤ quote_bid」直接成交，沒有考慮 queue position。

### 改善建議（從高到低優先）

```
優先度 ★★★  → 立即要做
優先度 ★★   → 第二步
優先度 ★    → 有餘力再做
```

| 優先度 | 改善項目 | 說明 |
|--------|----------|------|
| ★★★ | **以 NEW3 為 ground truth OOS** | 完全分離，不能被模型見過 |
| ★★★ | **Walk-forward validation 跨月** | 用 NEW(1月) 訓練 → 在 NEW3(3月、5月) 測試，各月獨立報告 |
| ★★★ | **每個 session 獨立清倉** | 日末強制平倉，符合真實操作 |
| ★★ | **加入 regime filter** | rolling_volatility 門控，低信心不下單 |
| ★★ | **滑點加強**：queue position | 簡單版：假設在 queue 中間，fill rate = trade_vol / queue_ahead |
| ★ | **多 seed ensemble** | 3個 seed 的模型平均，降低訓練隨機性 |

---

## 五、下一步執行計畫（按優先順序）

### Step 1：NEW3 完整 OOS 驗證（1–2 天）

**目的**：搞清楚 best model（smoothed_mid_move_3class, fc=8）在完全沒見過的資料上到底賺不賺錢。

```bash
# 在專案根目錄執行（把 NEW3 當 OOS 跑 hybrid backtest）
python new_era/15_models_playaround/10_oos_backtest_top3_hybrid.py \
  --new2-dir data/NEW3 \
  --out-dir new_era/backtest_outputs/NEW3_oos_hybrid
```

**預期結果**：
- 如果 NEW3 的 improvement > 0 → 模型有跨月泛化能力，可以繼續
- 如果 NEW3 improvement ≤ 0 → 確認模型在 in-sample 月份的過擬合，需要回去改 feature/label

---

### Step 2：Walk-Forward 驗證（3–5 天）

**目的**：模擬真實部署流程，確認 re-train 策略。

```
Week t:   用 NEW(前4/5 sessions) train → 在 NEW(最後1/5) 測 → 記錄 OOS PnL
Week t+1: 滾動加入 NEW1 → 在 NEW2 測
Week t+2: 滾動加入 NEW2 → 在 NEW3 測
```

這是目前最能反映真實部署的方式。你的 `10_walk_forward_mm_backtest.py` 已有骨架，需要擴展到包含 NEW1/NEW2/NEW3。

**關鍵配置改動**（改 `_common.py`）：
```python
# 資料目錄改成包含所有批次
DATA_DIR = "data/NEW"  # 或改為 data/NEW_ALL（把所有session合併）
# 切分策略改為 walk-forward
SPLIT_MODE = "walk_forward"  # 而非靜態 70/15/15
```

---

### Step 3：加入 Regime Filter（1–2 天）

**目的**：解決「趨勢行情下 baseline 虧錢，模型也虧錢」的問題。

在 backtest 邏輯中加入：

```python
# 如果 rolling_volatility > 閾值（高波動趨勢行情）
# → 收緊庫存上限（max_inventory = 1）
# → 不報雙邊，只報模型指定方向
if rolling_vol > VOL_THRESHOLD:
    effective_max_inv = 1
    allow_skew = True
else:  # 低波動，純造市
    effective_max_inv = 3
    allow_skew = False
```

用 NEW3 驗證這個 filter 是否能讓虧損的 session 轉正。

---

### Step 4：確認 FPGA 路徑（視前面結果而定）

只有在 Step 1 確認模型有真實 OOS 獲利能力後，才值得繼續：

- 選定模型：`smoothed_mid_move_3class，fc=8，w=100`
- 繼續原本的 QAT → hls4ml → Vitis HLS 流程
- 目標：在 PYNQ-Z2 上跑 inference，latency < 1ms

> [!NOTE]
> 如果 Step 1 結果不好，FPGA 的事先暫停。先把 NEW3 加入訓練資料（re-train），或嘗試更短的 horizon（H=5 而非 H=10），讓 label 更均衡。

---

## 六、決策樹（快速查閱）

```
你現在在哪？
  └── 還沒跑 NEW3 OOS → [執行 Step 1]
        ├── 結果：improvement > 0 → [繼續 Step 2、3，然後 FPGA]
        └── 結果：improvement ≤ 0
              ├── 問題一：Feature drift 回來了 → 檢查 rolling_zscore 是否足夠
              └── 問題二：Label 太稀疏 → 嘗試 H=5，threshold=0.5 tick
```

---

## 七、關於「是否使用固定模型」的最終答案

**用固定模型架構 + 動態 regime filter，而不是多模型切換。**

理由：
1. FPGA 資源有限，只能承載 1 個推論引擎
2. 多模型切換的元學習需要更多歷史資料
3. 你的最佳模型（smoothed_mid_move_3class）的 loss-of-edge 主要來自趨勢行情，regime filter 可以解決這個問題而不需要換模型
