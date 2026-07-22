# A6：廠商問題清單

> 對應 Slide 11「下一步」的廠商接洽部分。
> 來源：week8.md §6.2，並加入本週 csynth 數字後的更新。

---

## 技術現況說明（給廠商看）

### 我們目前的設計
- **模型**：TCNClassifier（3 blocks，channels=16，kernel=3，dilation=1/2/4，classes=3）
- **輸入格式**：固定形狀 `int8_t[12 × 100]`（12 特徵，100 個事件的序列）
- **權重大小**：4,771 bytes（INT8 量化後）
- **計算量**：460,848 MACs/sample

### 目前 HLS 狀態（xc7z020clg400-1）

| 項目 | 數值 |
|---|---|
| Estimated clock | 6.978 ns（target: 5 ns）|
| Latency | ~133,612 cycles @ 6.978 ns ≈ 0.932 ms |
| BRAM | 9%（合理）|
| DSP48 | 194%（超限）|
| FF | 105%（超限）|
| LUT | 410%（超限）|

**現象**：csim 通過，csynth 可運行，但資源嚴重超限（特別是 LUT 超 4 倍），時脈也未達標。
**根因推斷**：目前 C++ 是 fully-parallel 寫法，所有 MAC 同時實例化，導致 LUT/DSP 炸開。

---

## 希望廠商給建議的問題清單

### Q1：目標器件選擇
> 在 `xc7z020`（200K LUT, 220 DSP）的規模上，TCN（460K MACs）從根本上能不能做 full-pipeline 推論？還是這個規模的 FPGA 就是不適合？
> 
> 如果不適合，換成哪個器件（例如 `xc7z045`、`xczu3eg`、`xcku5p`）是比較合理的起點？

---

### Q2：微架構策略

對於 `conv1d causal`（含 dilation）+ 全連接的混合架構，你會建議哪種微架構：

**(A) 時間復用（Time-multiplexing / Reuse Factor）**
- 限制同時實例化的 MAC 數量（例如只展開 input channel，不展開 output channel）
- 換取 latency 增大，但可以讓設計 fit
- pragma 大致如下：
  ```cpp
  #pragma HLS ALLOCATION instances=mul limit=36 operation
  ```

**(B) Systolic Array 結構**
- 用小型 MAC array tile（例如 8×8）以網格形式排列計算
- 對規則矩陣乘法（全連接層）特別有效
- 對 conv1d 較複雜（因為 causal padding 需要特殊 memory access pattern）

**(C) Shift Register / Line Buffer**
- 維護一個 shift register 存最近 K 個 event 的特徵向量
- 每來一個新 event 只更新最新的 slice，其餘不動（O(1) update）
- 最適合最終的「每個 event 跑一次推論」線上場景

> 你推薦從哪種方式入手？還是組合使用？

---

### Q3：記憶體佈局

> 我們的權重只有 4.8 KB，遠小於 xc7z020 的 BRAM 容量（約 4.9 MB）。
> 
> 但 csynth 顯示 BRAM 使用率只有 9%，而 LUT 卻超限。由此推斷，目前工具把很多東西放在 register/LUT 而不是 BRAM。
>
> **(a)** 對於 4.8 KB 的小型模型，你建議是把權重全部放 BRAM，還是放 LUTRAM？什麼情況下選哪個？
>
> **(b)** 對於中間激活值（16 個 channel × 100 個 time step = 1,600 bytes），要如何做 banking/partition 才能讓多個 port 同時存取？

---

### Q4：時脈（Timing）問題

> 目前 csynth 顯示 estimated clock = 6.978 ns，未達到 5 ns (200 MHz) 的目標。
>
> **(a)** 你推薦優先降低組合邏輯長度（切更多 pipeline stage），還是先降低資源（減少 congestion）？
>
> **(b)** 是否有經驗法則：在 `xc7z020` 上做 INT8×INT8 乘加，通常 critical path 在哪裡？

---

### Q5：展示策略

> 這是大學研究專題（不是實際部署），主要目的是「展示 HLS 可以做低延遲 AI 推論」。
>
> 你會建議：
>
> **(A)** 先換更大的器件（讓設計 fit），然後在這台上量測完整延遲，再回頭嘗試縮資源
>
> **(B)** 先在 xc7z020 上把資源壓下去（接受 latency 大一些），作為工程能力展示
>
> **(C)** 把模型結構改小（減少 channels 或 blocks），讓它能 fit 進 xc7z020

---

## 我們能提供給廠商的資料

1. **INT8 權重檔**：`artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts`
2. **Reference input/output**：
   - `reference_input.txt`（int8 量化後的測試輸入）
   - `reference_output.txt`（FP32 logits 與 softmax 輸出）
3. **Golden logits（int32）**：`kGoldenLogitsInt32 = {-3947, 220, -4732}`
4. **C++ 原始碼**：`cpp_hls_prep/` 完整目錄
5. **Manifest（模型 schema）**：
   - `artifacts/04_hardware_handoff/int8_export/tcn_manifest.json`
   - 包含：layer 名稱、形狀、scale、量化規則

---

## 廠商評估的期待目標

| 指標 | 期望 |
|---|---|
| 最終 FPGA latency | < 50 µs（理想）、< 100 µs（可接受）|
| DSP 利用率 | ≤ 80%（有餘裕）|
| LUT 利用率 | ≤ 80%（有餘裕）|
| Clock | ≥ 200 MHz（5 ns）|
| Correctness | 與 Python INT8 simulator logits 逐 bit 相等 |
