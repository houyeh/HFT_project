# 04. Python 向量化運算 (Python Vectorization)

在處理高頻交易這種動輒數千萬行 (Millions of rows) 的 LOB 資料時，初學者最常犯的錯誤就是「使用 `for` 迴圈列舉每一行」。

## 1. For 迴圈的致命傷
Python 是一種直譯式語言 (Interpreted language)，每次迴圈執行時都要進行變數型態檢查、物件配置等操作。如果你寫了這樣的程式碼：
```python
for i in range(len(df)):
    df.loc[i, 'spread'] = df.loc[i, 'ask'] - df.loc[i, 'bid']
```
這在處理 1,000 萬行資料時，可能要跑好幾個小時甚至一天。這在量化研究中是不可接受的。

## 2. 什麼是向量化 (Vectorization)？
向量化是指利用底層使用 C 語言或 Fortran 編寫的函式庫 (如 NumPy 和 Pandas)，**同時對整組陣列 (Array) 進行數學運算**。它會跳過 Python 的直譯成本，直接利用 CPU 的 SIMD (單一指令多重數據) 指令集來加速。

向量化的寫法非常簡單俐落：
```python
df['spread'] = df['ask'] - df['bid']
```
這行程式碼的底層會將所有的 `ask` 和 `bid` 一次性送到 C 語言層級進行相減，速度比 `for` 迴圈**快上數千倍到數萬倍**。

## 3. 常見的向量化技巧
1.  **條件計算 (`np.where`)**：不要用 `if-else`。使用 `np.where(condition, value_if_true, value_if_false)`。
2.  **移動平均/差分 (`df.diff()`, `df.rolling().mean()`)**：要計算過去 10 分鐘的平均，絕對不要做雙層迴圈，直接使用 Pandas 的 `rolling()` 函數，其底層是高度最佳化的 C++ 實作。
3.  **前向填充 (`df.ffill()`)**：處理缺失值時，直接呼叫內建函數，瞬間填滿所有的 NaN。

學會 Vectorization 是成為合格計量工程師 (Quant Engineer) 的必經之路！
