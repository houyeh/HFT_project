# 19. 從 TW TCN 到手刻 C++ / HLS 主線

## 這一章在講什麼
這一章的重點不是「再做一個 Python 模型」，而是：

> 如何把目前在 TW replay 上已經成熟的 TCN，開始改寫成適合 HLS 看的 C++ 版本。

---

## 1. 為什麼這一步要先做

目前專案有兩條線：

1. `CME`
   - 已做 adapter
   - 已做 direct validation
   - 但還沒形成穩定可交易結果

2. `TW replay`
   - 已有完整資料處理
   - 已有 feature pipeline
   - 已有訓練、回測、INT8 匯出
   - 最適合先做硬體化

所以如果你要在有限時間內把專案推向 `HLS competition ready`，最合理的主線就是：

`TW replay -> TCN -> INT8 -> C++ -> HLS`

而副線才是：

`CME -> adapter -> direct validation -> 分商品訓練`

---

## 2. 目前模型長什麼樣

這顆 TCN 的核心設定是：
- 輸入：`12 features x 100 events`
- backbone：`3-layer TCN`
- channels：`16, 16, 16`
- kernel size：`3`
- dilation：`1, 2, 4`
- 輸出：`3 classes`

資源規模：
- 參數量：`4,771`
- MACs / sample：`460,848`
- INT8 權重大小：約 `4.8 KB`

這個規模很重要，因為它夠小，才有把它推向 FPGA / HLS 的價值。

---

## 3. 以前的 C++ 版本哪裡還不夠

之前的 C++ 版本已經能：
- 讀 `.wts`
- 做完整 INT8 inference
- 跑 `conv1d / gemm` benchmark

但它還有很多典型的「軟體寫法」：
- `unordered_map`
- `vector`
- runtime tensor lookup
- runtime shape 推導

這些對一般 C++ 沒問題，但對 HLS 來說不夠友善。

---

## 4. 什麼叫做 HLS-friendly

對這個專案而言，`HLS-friendly` 的意思不是只要「可以編譯」，而是：

1. 輸入 shape 固定
2. 權重欄位固定
3. 每一層 loop 結構明確
4. 減少動態配置
5. 資料流與硬體管線更容易對應

因此，這一步的本質其實是把模型從：
- 動態軟體 inference

改寫成：
- 固定資料流的硬體前置版本

---

## 5. 新增的 fixed-shape C++ core

對應檔案：
- [fixed_tcn.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn.hpp)
- [fixed_tcn.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn.cpp)

這版的特點：
- 固定輸入 `12 x 100`
- 固定 block 結構
- 權重直接對應到固定欄位
- 主要推論核心使用固定大小陣列

這樣做的意義是：
- 更容易和 HLS 的 array / loop pipeline 概念對上
- 後面要拆成 testbench 或 kernel 會比較自然

---

## 6. 如何驗證改寫沒有出錯

build：

```powershell
cmake -S cpp_hls_prep -B cpp_hls_prep/build
cmake --build cpp_hls_prep/build --config Release
```

比對：

```powershell
.\cpp_hls_prep\build\tcn_hls_prep.exe `
  --compare-fixed `
  --weights artifacts\hardware_handoff\int8_export\tcn_int8_weights.wts `
  --ref-input artifacts\hardware_handoff\int8_export\reference_input.txt
```

目前結果：
- fixed core logits 與 dynamic core 一致
- `max probability diff = 0`

這表示：
- 改寫後的結構沒有改壞模型
- 至少在 reference input 上，功能是對得上的

---

## 7. 這一步和 hls4ml 的關係

`hls4ml` 目前已經試過，但卡在：
- `permute`
- `causal pad`

所以現在的策略不是死等工具自動轉，而是先把模型手刻成比較硬體友善的版本。

之後有兩種可能：

1. 直接沿著 manual C++ / HLS 路線走下去
2. 等模型圖更乾淨後，再回頭試一次 `hls4ml`

---

## 8. 這一步的真正價值

這一步最大的價值不是多快，而是讓專案從：
- 只有 Python 實驗結果

變成：
- 已經有可對應硬體資料流的固定 shape 核心

這對於：
- 跟老師討論
- 跟廠商溝通
- 規劃 HLS 組比賽

都非常重要。

---

## 9. 小結
如果用一句話總結這一章：

> 我們已經不是只有一顆 Python TCN，而是開始把這顆 TCN 改寫成真正可以往 HLS 交接的固定 shape C++ 版本。
