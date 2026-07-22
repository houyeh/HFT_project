# 03. 滑動視窗與張量 (Sliding Window and Tensors)

要在 PyTorch 中訓練時間序列模型（如 TCN 或 LSTM），我們不能只是單純地把一列一列的 CSV 資料餵進去。模型需要知道過去一段時間的「歷史脈絡」。這就需要用到**滑動視窗 (Sliding Window)** 將 2D 表格轉換為 **3D 張量 (Tensors)**。

## 1. 什麼是 3D Tensor？
在表格資料 (Pandas DataFrame) 中，資料是 2D 的：`[Row 數量, 特徵數量]`。
但在 PyTorch 的時間處理模型中，張量的標準形狀是 3D：
**`(Batch_Size, Sequence_Length, Num_Features)`** 簡稱 `(B, S, F)` 或 `(B, L, C)`。
*   **Batch_Size (B)**: 一次丟幾筆樣本進去訓練（例如 64）。
*   **Sequence_Length (S 或 L)**: 歷史回溯長度，也就是神經網路每次能看到過去幾步的數據（例如過去 100 個 Ticks）。
*   **Num_Features (F 或 C)**: 每個時間點有幾個特徵（例如 OBI, Spread, Return 等 5 個特徵）。

## 2. 滑動視窗 (Sliding Window) 怎麼運作？
假設我們有時間點 `t=1` 到 `t=1000` 的資料，且 Sequence Length 設定為 100。
*   第一個樣本：取 `t=1` 到 `t=100` 的資料作為特徵 (X)，用來預測 `t=101` 的目標 (Y)。
*   視窗往前滑動一格 (Stride=1)。
*   第二個樣本：取 `t=2` 到 `t=101` 的資料作為特徵 (X)，用來預測 `t=102` 的目標 (Y)。

## 3. 在 Python 中的實作技巧
如果我們用 `for` 迴圈來製作 Sliding Window，速度會極度緩慢。因此在 `preprocessing` 階段，我們通常會使用 `numpy.lib.stride_tricks.sliding_window_view` 這個內建函數，它透過直接操作記憶體位置 (Memory Pointers) 來瞬間完成 2D 到 3D 的轉換，完全不消耗額外記憶體與時間。
