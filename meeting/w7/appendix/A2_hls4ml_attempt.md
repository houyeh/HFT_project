# A2：hls4ml 轉換嘗試的失敗紀錄

> 對應 Slide 7「為什麼選擇手刻 HLS」。
> 來源腳本：`scripts/05_hardware_handoff/attempt_hls4ml_conversion.py`

---

## 為什麼要嘗試 hls4ml？

hls4ml 是由 CERN 開發、Fermilab 維護的自動化工具，可以把 Keras/PyTorch 模型直接轉成 Vivado HLS C++ 專案。理論上，這條路可以省去手刻 C++ 的工作量。

我們嘗試了三種 wrapper 策略，結果如下：

---

## 三種嘗試方案

### 方案 1：`original`（直接轉，不包 wrapper）
```python
# 輸入 shape：(100, 12)，即 [seq_len, features]（channel-last 格式）
config = hls4ml.utils.config_from_pytorch_model(
    base, input_shape=(100, 12), granularity="name", backend="Vivado"
)
```
**失敗原因**：
- hls4ml 對 PyTorch 的 `Conv1d` 期望 channel-first 格式 `(batch, channels, seq_len)`
- 我們的 TCN 在 PyTorch 是把 features 當 channels（`(batch, 12, 100)`），但進模型前的 window tensor 是 `(batch, 100, 12)`
- 需要一個 `permute()` 操作，而 hls4ml 不直接支援這個 reshape 作為獨立算子

**錯誤訊息類型**：`RuntimeError: Shape mismatch / Unsupported layer type: Permute`

---

### 方案 2：`channel_first_mean`（包 channel-first + GlobalAvgPool）
```python
class ChannelFirstMeanWrapper(nn.Module):
    def forward(self, x):
        x = self.tcn(x)          # 輸入已是 (batch, 12, 100)
        x = x.mean(dim=2)        # (batch, 12) → GlobalAvgPool
        x = self.fc(x)           # (batch, 3)
        return x
```
**失敗原因**：
- `TCN` 內部的 `CausalConv1d` Layer 使用了 **左側 zero-padding**（`F.pad(x, (self.padding, 0))`），這個動態 padding 操作 hls4ml 無法解析成固定 HLS 算子
- TensorFlow/Keras 有 `ZeroPadding1D` 對應，但 PyTorch 的 functional pad 沒有直接對應的 hls4ml 算子

**錯誤訊息類型**：`KeyError: unsupported PyTorch operation: aten::pad`

---

### 方案 3：`channel_first_pool`（包 AdaptiveAvgPool1d）
```python
class ChannelFirstPoolWrapper(nn.Module):
    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)  # AdaptiveAvgPool1d(1)
        x = self.fc(x)
        return x
```
**失敗原因**：與方案 2 相同，問題根源在於 `CausalConv1d` 內部的 `F.pad`，不在 pooling 層。

**錯誤訊息類型**：`KeyError: unsupported PyTorch operation: aten::pad`

---

## 為什麼改走手刻 HLS？

| 比較項目 | hls4ml 自動轉換 | 手刻 HLS-friendly C++ |
|---|---|---|
| 對 causal padding 的支援 | ❌ 不支援動態 F.pad | ✅ 手動在 C++ 內左側補 0 |
| 對 permute/reshape 的支援 | ❌ 受限於算子庫 | ✅ 資料佈局由我們完全控制 |
| HLS pragma 精確控制 | ❌ 自動生成，難以細調 | ✅ 逐層加 pipeline/unroll/partition |
| Bit-accurate 驗證難度 | ⚠️ 黑盒輸出，難以逐層比對 | ✅ 可與 Python INT8 simulator 逐層比對 |
| 開發時間成本 | 低（若支援算子） | 高（需手刻 ~600 行 C++） |
| 長期可維護性 | 工具版本依賴 | 完全自主 |

**結論**：鑑於 TCN 的 causal padding 是核心操作，且硬體工程組的廠商需要可以直接閱讀和修改的 C++ source，手刻 HLS-friendly C++ 是更適合這個專案的技術選擇。

---

## 備著：若未來想繼續嘗試 hls4ml

可先把 `CausalConv1d` 替換成等價的「靜態 padding + standard Conv1d」，讓 pad 變成一個 static zero-pad buffer，再嘗試轉換：

```python
# 替代實作（靜態 padding 版）
class StaticPadCausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad_size = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
    
    def forward(self, x):
        # 改用 nn.ConstantPad1d 而非 F.pad
        x = nn.functional.pad(x, (self.pad_size, 0))
        return self.conv(x)
```

這樣 hls4ml 有機會把 padding 識別為 `ZeroPadding1D`，但並不保證完全成功。
