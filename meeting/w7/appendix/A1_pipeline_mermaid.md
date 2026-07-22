# A1：完整 Pipeline 流程圖

> 適合放在 Slide 2「研究 Pipeline」使用。
> 這個 Mermaid 圖可以直接貼入支援 Mermaid 的工具（如 Obsidian、GitHub、mermaid.live）。

---

## 完整資料與硬體交付流程

```mermaid
flowchart LR
  subgraph TW["🇹🇼 TW Replay 主線"]
    direction TB
    A["Raw replay CSV\nreplay1_TW.csv\n(~1.3M rows)"]
    B["LOB 重建\ntw_replay.py\n→ (bid/ask px×vol) L1~L5"]
    C["特徵工程\nexecution_features.py\nbuild_feature_dataframe()\n→ 12 維 execution-aware"]
    D["Sliding Window\n(100 events, stride=10)\n→ (N, 100, 12) tensor"]
    E["TCN Classifier 訓練\n3 blocks, ch=16, k=3\ndilation=1/2/4\nAdam + WeightedCE + EarlyStopping"]
    F["Frozen Model\nfrozen_tcn_best_model.pt"]
    A-->B-->C-->D-->E-->F
  end

  subgraph CME["🌎 CME 次線"]
    M["CME Logic20 CSV\n雙商品扁平快照\nCLK6 / ESM6"]
    N["CME Adapter\ncme_logic20.py\n→ 標準 LOB schema\n(L3-L5 用 tick_size 補齊、量補 0)"]
    O["Direct Validation\nevaluate_cme_frozen_model.py\ngated accuracy=0.82\nmacro-F1=0.39\nnPnL=-36,316 ticks ❌"]
    M-->N-->C
    F-->O
  end

  subgraph HW["⚙️ 硬體主線"]
    P["INT8 量化匯出\nexport_tcn_int8_artifacts.py\n→ .wts (4.8KB)\n→ reference_input.txt (int8)\n→ fixed_tcn_*.hpp (headers)"]
    Q["C++ Dynamic Inference\nint8_kernels.cpp\n讀 .wts 動態推論"]
    R["C++ Fixed-Shape Core\nfixed_tcn.cpp / _modules.cpp\n固定 shape=12×100\n→ HLS-friendly"]
    S["HLS Testbench\nhls_testbench.cpp\ngolden logits 比對\nPASS ✅"]
    T["Vitis HLS\ncsim: PASS ✅\ncsynth: 超限 ⚠️\nDSP=194% LUT=410%\nLatency≈0.932ms"]
    F-->P
    P-->Q
    P-->R
    R-->S-->T
    Q-.->|bit-accurate|R
  end

  subgraph VAL["🔬 Bit-Accurate Gate（Week 8）"]
    V1["Python INT8 Simulator\nsimulate_int8_inference.py\n逐層 numpy 整數運算"]
    V2["C++ 推論\ntcn_hls_prep.exe"]
    V3{{"logits_int32\n逐 bit 相等？"}}
    PASS["✅ PASS\n可進入 HLS 優化"]
    V1-->V3
    V2-->V3
    V3--"YES"-->PASS
    P-->V1
    P-->V2
  end
```

---

## 說明口白（可對應投影片講稿）

> 整個系統有三條線並行進行：
>
> **主線（左）**：從 TW replay 事件流，到 TCN 訓練，產出凍結模型。
>
> **次線（上方）**：CME Logic20 資料透過 adapter 接上相同的特徵工程流程，直接用 TW 訓練好的模型驗證——結論是 pipeline 通了，但交易 edge 沒有成立。
>
> **硬體主線（右下）**：凍結模型被量化成 INT8、匯出成 C++ header，再手刻成 HLS-friendly 固定 shape 版本，經過 csim 驗證後送進 Vitis HLS 合成。
>
> **中央的 Gate**：Python 整數模擬器和 C++ 的 `logits_int32` 必須逐 bit 完全相等，這是從 research 走到 hardware delivery 的最後一道驗證防線。

---

## 時間複雜度標注

| 模組 | 估計執行時間 |
|---|---|
| TW replay feature extraction | ~2 min / session |
| TCN training (6 epochs) | ~5 min (CPU) |
| INT8 export | < 30 sec |
| C++ build | < 1 min |
| Vitis HLS csim | ~5 min |
| Vitis HLS csynth | ~15-30 min |
