# W7 Meeting 技術細節對照表 (modify2.md)

本文件整理了 `W7_meeting.pptx` 各投影片內容對應的程式碼檔案、行號，以及報告時應說明的核心技術邏輯。

---

## 📅 投影片與程式碼對比表

| 投影片 | 內容部分 | 程式碼出處 (File Path) | 行號 (Lines) | 報告重點 / 逐字稿精華 |
|:---:|---|---|:---:|---|
| **Slide 3** | **Pipeline Overview** | [A1_pipeline_mermaid.md](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/meeting/w7/appendix/A1_pipeline_mermaid.md) | L1-30 | 說明系統三條線（TW 主線、硬體主線、CME 次線）如何在 Frozen Model 處交會。 |
| **Slide 4/5** | **Features 定義** | [dimensionless_features.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/features/dimensionless_features.py) | L31-103 | 12 個無因次特徵實作。重點在於 Tick-size 正規化，確保跨市場通用性。 |
| **Slide 6** | **Model Params** | [constants.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/utils/constants.py) | L31-35 | channels=16, kernel=3, blocks=3。解釋為什麼選小模型：適合 xc7z020 BRAM。 |
| **Slide 7** | **TCN Architecture** | [tcn.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/models/tcn.py) | L75-175 | 殘差連接 (Residual) 防止退化、膨脹卷積 (Dilated) 擴大感受野（RF=29）。 |
| **Slide 8** | **Label & Sampling** | [datasets.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/training/datasets.py) | L115-136 | 如何從 100 個 quote events 觀察未來 10 步的 mid-price 決定 3-class 標籤。 |
| **Slide 9** | **Gating 機制** | [signals.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/backtest/signals.py) | L89-95 | `take_trade` 邏輯：機率閾值 + Margin > 0 + Regime Guard（波動率與成交流向過濾）。 |
| **Slide 10** | **INT8 Quantization**| [hardware.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/models/hardware.py) | L110-117 | 使用 PyTorch FX PTQ。重點：Weight 壓縮 4x，Logit Diff 在驗證後為 0。 |
| **Slide 11** | **Bit-Accurate 驗證** | [fixed_tcn_test_vectors.hpp](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp) | L87-89 | Golden Logits `[-3947, 220, -4732]`。強調 bit-by-bit 相等是硬體交付的底線。 |
| **Slide 11** | **數值對齊驗證** | [fixed_tcn_modules.cpp](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_modules.cpp) | L32, 143, 182 | **核心重點**：透過 `std::round` 與 ReLU 實作，達成 Python=C++ 的 Bit-accurate 驗證。 |
| **Slide 12** | **HLS Top Interface** | [fixed_tcn_top.cpp](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_top.cpp) | L10-18 | C++ 接口已 HLS-ready，代表邏輯已具備硬體化條件。 |
| **Slide 14/15**| **初步合成結果** | (HLS Synthesis Report) | — | 誠實報告資源與時脈，但強調這只是「未優化的起點」，並非設計極限。 |
| **Slide 16/17**| **CME Adapter** | [cme_logic20.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/src/data/cme_logic20.py) | L101-110 | 因應 CME 檔位不足所做的資料擴展邏輯。 |

---

## 🛠️ 報告策略：專注量化與正確性

### 💡 核心論點 (Talking Points)
1. **INT8 量化成功**：模型從 19KB 壓縮到 4.8KB，且推論精度在軟體模擬中完全無損。
2. **Bit-accurate 達成**：這是本次報告最重要的里程碑。我們手刻的 C++ 在整數運算下，產出的 Logits 與 Python 模擬器 **完全一致**（Logit: `[-3947, 220, -4732]`）。
3. **硬體驗證 (HLS)**：目前已跑通 `csim`（功能正確）與初步 `csynth`（電路可生成）。雖然目前資源較多，但這屬於下一階段「工程優化」的範疇，不影響目前模型邏輯的正確性。

---

## 🚩 備註
- 若需更新圖表，請執行：`python meeting/w7/appendix/scripts/gen_all_figures.py`
- 若需重新驗證 Bit-accurate，請點選：[verify_cpp_python_bit_accurate.py](file:///c:/Users/holbo/00VS_projjects/00_HFT_project/scripts/verify_cpp_python_bit_accurate.py)
