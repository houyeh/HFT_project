# 下週行動計畫 (Next Steps)

1. **標籤重建**：建立專屬的 `bid_toxic` 與 `ask_toxic` 二元標籤。
2. **模型重訓**：使用新的 Side-specific 毒性標籤重新訓練輕量級模型。
3. **閾值選擇**：在有 Regime Gate 防護的前提下，透過 Validation 集嚴謹選擇 Threshold。
4. **效能比對**：與隨機撤單及基於規則 (Rule-based) 的基準進行嚴格比對。
5. **回測升級**：在宣稱硬體價值前，導入 Queue-aware (類似 HftBacktest) 的真實搓合驗證。
6. **FPGA MVP**：規劃簡單的 Rolling/EWMA Gate + 輕量級 TCN 撤單邏輯進入硬體。
