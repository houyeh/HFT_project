# 一步步沿著這個專案走的故事線

自己教自己，講一遍給自己聽

## 1. 模型是怎麼做出來的、怎麼驗證？

### - 這個模型是由這邊的檔案運作的

`C:\Users\holbo\00VS_projjects\00_HFT_project\src\models\tcn.py`

用來訓練的資料集有：
而依據80/20的法則拆分後，用來回測的則有：

訓練的過程中，
用了 96,720筆資料
train: 67704
val:   14508
test:  14508

這次：
744603筆資料

加了?層網路
padding: ??
dropout: ?
optimizer: ??
learning_rate: ??

最後得到結果：
PnL: ??
Sharpe Ratio: ??
Max Drawdown: ??
Win Rate: ??

### - 最後，也用這套模式專門for CME 的資料去train

#### 。CL 資料訓練得到

#### 。ES 資料訓練得到

### - 接著 手癢也去把它變成了幣安的模型，方在虛擬倉回測後得到

## 2. 要怎麼把這個模型搬到FPGA上？

### - 把模型從FP32 量化成INT8

檔案：`C:\Users\holbo\00VS_projjects\00_HFT_project\src\models\tcn_int8.py`

可行性：

- latency算法：

--> 合理，且有機會在FPGA上得到比純軟體更低的 latency。

- MAC 數量估計的方法：

## 3. 把模型改成hls4ml-friendly的架構

### HLS跟一般軟體工程的C++ code差的地方

### 用MLP做的baseline

### 用Keras重構TCN，使其可以順利用hls4ml轉換

### 兩者的比較

## 4. 真的hls4ml合成出來： by Vivado

## 5. 把模型放上FPGA

### 模型： PYNQ-Z2

### 結果

-
