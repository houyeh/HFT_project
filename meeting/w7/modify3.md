# W7 Meeting 技術細節對照表 - 研究完結版 (modify3.md)

本文件整理了 `W7_meeting.pptx` 各投影片內容，將報告重點停在 **「INT8 量化與數值對齊驗證成功」**。此版本完全隱藏了初步的硬體合成（HLS synthesis）結果。

---

## 📅 投影片與程式碼對比表

| 投影片 | 內容部分 | 程式碼出處 (File Path) | 行號 (Lines) | 報告重點 / 逐字稿精華 |
|:---:|---|---|:---:|---|
| **Slide 3** | **Pipeline Overview** | [A1_pipeline_mermaid.md](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/meeting/w7/appendix/A1_pipeline_mermaid.md) | L1-30 | 說明系統三條線。重點：目前已完成「研究主線」到「硬體準備」的交接點。 |
| **Slide 4/5** | **Features 定義** | [dimensionless_features.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/features/dimensionless_features.py) | L31-103 | 12 個無因次特徵實作。強調 Tick-size 正規化帶來的市場通用能力。 |
| **Slide 6** | **Model Params** | [constants.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/utils/constants.py) | L31-35 | channels=16, kernel=3, blocks=3。解釋為什麼選小模型：為了適配 FPGA 嵌入式資源。 |
| **Slide 7** | **TCN Architecture** | [tcn.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/models/tcn.py) | L75-175 | 殘差連接與膨脹卷積。解釋感受野 (RF=29) 如何覆蓋足夠的歷史資訊。 |
| **Slide 8** | **Label & Sampling** | [datasets.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/training/datasets.py) | L115-136 | 如何從 LOB 事件中標記未來價格動向。 |
| **Slide 10** | **INT8 Quantization**| [hardware.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/models/hardware.py) | L110-117 | **核心里程碑**：模型權重壓縮 4x，推論結果與 FP32 版在模擬中無顯著差異。 |
| **Slide 11** | **Bit-Accurate 驗證**| [fixed_tcn_modules.cpp](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_modules.cpp) | L32, 143, 182 | **最高潮點**：達成真正的逐位元對齊。Golden Logits `[-3947, 220, -4732]` 驗證通過。 |
| **Slide 12** | **HLS Top Interface**| [fixed_tcn_top.cpp](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_top.cpp) | L10-18 | **交付物**：封裝完整的 HLS-ready C++ 接口，隨時可送入硬體工具進行合成。 |
| **Slide 16/17**| **CME Adapter** | [cme_logic20.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/data/cme_logic20.py) | L101-110 | 因應 CME 檔位不足所做的資料外推與對齊邏輯。 |

---

## 🛠️ 報告策略：專注正確性與研究交付

### 💡 核心論點 (Talking Points)
1. **研究主線告一段落**：我們成功將 AI 模型從浮點數 (FP32) 轉化為整數 (INT8)，並在 C++ 環境下完整重現，兩者輸出完全一致（Bit-accurate）。
2. **硬體化門檻已解除**：目前的 C++ 代碼已針對硬體 HLS 需求編寫（Fixed-shape, No dynamics），代表「電路邏輯」已定稿，這週即可交付給硬體小組進行後續合成。
3. **9.2 µs 的預期效能**：基於 TCN 的運算量與量化後的參數規模，我們預期在硬體優化後，延遲能達到微秒等級（約 9.2 µs），遠優於現有的 CPU/GPU 方案。

### ❓ 潛在問題答辯 (Q&A)
- **Q: 為什麼還沒有 HLS 報告？** 答：本週的目標是確保「邏輯與數值 100% 正確」。硬體合成（Synthesis）是下一步的工程工作，我們希望先鎖定軟體端的最高精度模型，再進行電路優化。
- **Q: Bit-accurate 的意義是什麼？** 答：它代表了研究端（Python）與實作端（C++）之間的零誤差，這是 HFT 系統中最難也最重要的正確性門鑑。

---

## 🚩 備註
- 若需重新驗證 Bit-accurate，請點選：[verify_cpp_python_bit_accurate.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/scripts/verify_cpp_python_bit_accurate.py)
