# C++ HLS Prep

這個資料夾現在是 `TW replay TCN` 的 C++ / HLS 前置主線，不再只是零散的 kernel 測試。

目前分成兩層：

1. `dynamic INT8 inference`
   - 可讀 `.wts`
   - 可做完整 INT8 推論
   - 方便驗證權重和輸出

2. `fixed-shape HLS-friendly core`
   - 固定輸入 `12 x 100`
   - 固定 block 結構
   - 用固定大小陣列取代 runtime tensor lookup
   - 更接近之後給 HLS 看的寫法

---

## 目錄說明

- [include/int8_kernels.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/int8_kernels.hpp)
  - 動態版 INT8 inference 宣告
- [include/fixed_tcn.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn.hpp)
  - 固定 shape 的 HLS-friendly TCN 宣告
- [src/int8_kernels.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/int8_kernels.cpp)
  - 動態版 INT8 推論與 kernel
- [src/fixed_tcn.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn.cpp)
  - 固定 shape、固定布局的推論核心
- [src/main.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/main.cpp)
  - build 後可直接跑的驗證程式

---

## Build

```powershell
cmake -S cpp_hls_prep -B cpp_hls_prep/build
cmake --build cpp_hls_prep/build --config Release
```

---

## 執行方式

### 1. 一般 INT8 inference

```powershell
.\cpp_hls_prep\build\tcn_hls_prep.exe `
  --weights artifacts\hardware_handoff\int8_export\tcn_int8_weights.wts `
  --ref-input artifacts\hardware_handoff\int8_export\reference_input.txt
```

### 2. 比對 fixed-shape HLS-friendly core

```powershell
.\cpp_hls_prep\build\tcn_hls_prep.exe `
  --weights artifacts\hardware_handoff\int8_export\tcn_int8_weights.wts `
  --ref-input artifacts\hardware_handoff\int8_export\reference_input.txt `
  --compare-fixed
```

這個模式會額外輸出：
- fixed core logits
- fixed core probabilities
- 和 dynamic core 的最大機率差異
- fixed core latency

### 3. 跑固定 header 的 HLS testbench

```powershell
& "C:\Users\holbo\00VS_projjects\00_HFT_project\cpp_hls_prep\build\tcn_hls_testbench.exe"
```

這個 testbench 直接使用：
- [fixed_tcn_weights.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated/fixed_tcn_weights.hpp)
- [fixed_tcn_test_vectors.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp)

也就是：
- 不再從 `.wts` 動態載入權重
- 直接使用固定陣列/標頭檔
- 直接比對 golden logits / probabilities

---

## 目前驗證到的事情

在目前 reference input 上：
- fixed core 和 dynamic core 輸出一致
- `max probability diff = 0`
- HLS testbench 可直接通過

這代表：
- 我們已經成功把同一顆 TCN
- 從一般 C++ inference
- 改寫成更固定、較接近 HLS 的資料流形式

---

## 為什麼要做 fixed-shape core

因為真正的 HLS 不喜歡：
- `unordered_map`
- `vector` 動態配置
- runtime shape 判斷
- 太多不固定的 tensor 路徑

所以這一步的重點不是先追 CPU latency，而是先把模型寫成：
- shape 固定
- loop 明確
- 權重欄位固定
- 容易被 HLS 理解

---

## 下一步

1. 將目前 header 形式的權重，再往 HLS 更偏好的 ROM / array partition 寫法推進
2. 在 Vitis HLS 內正式跑 `csim / csynth / export_design`
3. 觀察 `LUT / FF / BRAM / DSP / latency / II`

---

## Vitis HLS 腳本

這個資料夾已提供：
- [run_hls.tcl](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/run_hls.tcl)
- [run_vitis_hls_2025_2.ps1](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/run_vitis_hls_2025_2.ps1)

可直接用：

```powershell
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage csim -Part xc7z020clg400-1
```

或完整流程：

```powershell
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage full -Part xc7z020clg400-1
```

支援的 `Stage`：
- `csim`
- `csynth`
- `full`
