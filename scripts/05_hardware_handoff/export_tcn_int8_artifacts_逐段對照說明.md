# `export_tcn_int8_artifacts.py` 逐段對照說明

本文件依 `scripts/05_hardware_handoff/export_tcn_int8_artifacts.py` 的實際程式順序，逐段拆解：
- 每個區塊做什麼
- 為什麼這樣設計
- 前後依賴關係

---

## A. 檔頭說明與匯入

### A1. 模組 docstring
- 一開始就列出輸出物清單（`.wts`、`tensor_manifest.json`、reference input/output、兩個 HLS header）。
- 這是「訓練模型 -> 硬體交付」橋接腳本的契約說明。

### A2. import 與路徑注入
- 標準庫：`json`, `os`, `sys`, `Path`
- 第三方：`numpy`, `torch`
- `sys.path.insert(...)`：把專案根目錄塞進 import path，才能載入 `src.models.tcn`
- 專案模組：`build_default_model`

前後依賴：`main()` 會用 `build_default_model` 建模，再讀 checkpoint。

---

## B. 固定架構常數

- `FIXED_NUM_INPUTS = 12`
- `FIXED_NUM_CLASSES = 3`
- `FIXED_CHANNELS = 16`
- `FIXED_KERNEL_SIZE = 3`
- `FIXED_SEQ_LEN = 100`
- `FIXED_ACT_SCALE = 1/64`

用途：定義固定形狀的 INT8 推論規格，供 C++/HLS 一致使用。  
前後依賴：後續 reshape、卷積運算、header 生成都綁定這些常數。

---

## C. 量化與數值工具函式

### C1. `quantize_tensor_symmetric(tensor)`
- 將 FP32 tensor 做對稱量化（範圍 `[-127, 127]`）
- 回傳 `(int8_array, scale)`

依賴：`main()` 量化每個參數時使用。

### C2. `clamp_int8_array(array)`
- 把值裁切到 `[-127, 127]` 並轉 `int8`

依賴：requantize、殘差合併、pooling 後量化都會用。

### C3. `bias_to_accumulator(...)`
- 把 bias 從量化空間轉成 accumulator（`int32`）尺度
- 核心比值：`bias_scale / (input_scale * weight_scale)`

依賴：所有 conv/fc 的 accumulator 初始化都要先算這個。

### C4. `requantize_to_fixed_scale(acc, acc_scale)`
- 把 `int32 accumulator` 依比例縮放回固定 activation scale（`FIXED_ACT_SCALE`）
- 再做 int8 clamp

依賴：每層卷積輸出都要回到固定啟動尺度，才能串接下一層。

---

## D. 整數卷積運算核心

### D1. `causal_conv_accumulate(...)`
- 實作 causal + dilation 的 1D 卷積累加（`int32`）
- 時間索引用 `src_t = t - ((k-1-offset) * dilation)`，小於 0 就跳過

依賴：residual block 中兩個時間卷積都用它。

### D2. `pointwise_conv_accumulate(...)`
- 實作 1x1 卷積累加（等價於每個時間點做線性投影）

依賴：residual block 的 downsample 分支使用。

---

## E. TCN 子結構與分類頭

### E1. `residual_block(...)`
步驟：
1. `conv1`：bias 對齊 -> causal conv -> ReLU -> requantize  
2. `conv2`：bias 對齊 -> causal conv -> ReLU -> requantize  
3. residual：
   - 有 downsample 參數：走 pointwise conv + requantize
   - 否則直接 identity
4. 主分支 + residual 相加，經 ReLU，再 clamp 到 int8

依賴：`run_fixed_reference_inference` 連續呼叫 3 次形成完整 backbone。

### E2. `global_average_pool(hidden_q)`
- 對時間軸平均（整數除法）
- 回傳：
  - `pooled_int32`（原始整數池化值）
  - `pooled_q`（clamp 後 int8）

依賴：`fully_connected` 用 `pooled_q` 做分類。

### E3. `fully_connected(...)`
- 把 FC bias 轉到 accumulator 尺度
- 計算 `logits_int32`
- 轉回浮點得到 `logits_float`
- 用 `softmax` 算 `probabilities`

依賴：提供硬體路徑對比用的最終 golden 值。

---

## F. 固定參考推論總控

### F1. `run_fixed_reference_inference(quantized_tensors, input_q_flat)`
- 把 flat input reshape 成 `(12, 100)`
- 依序跑三個 residual block：
  - block0 dilation=1（含 downsample）
  - block1 dilation=2
  - block2 dilation=4
- 接 global average pooling + fully connected
- 回傳 pooled/logits/probabilities（整數與浮點版本）

前後依賴：
- 依賴 `quantized_tensors` 中所有指定 key（名稱必須與 state_dict 對齊）
- 結果給 `write_fixed_test_vectors_header` 與 manifest summary 使用。

---

## G. C++ 輸出格式工具

### G1. `format_cpp_scalar(value, kind)`
- 把單一值格式化成 C++ 字面值（`int8/int32/float`）

### G2. `format_cpp_array(array, kind, per_line=16)`
- 把陣列展平並排版成 C++ initializer list

依賴：兩個 header writer 都仰賴這組函式。

---

## H. Header 產生器

### H1. `write_fixed_weights_header(output_path, quantized_tensors)`
- 產生 `fixed_tcn_weights.hpp`
- 透過 `mapping` 將每個 tensor 名稱映射到 `FixedTCNWeights` 欄位
- 透過 `scale_mapping` 補齊對應 scale 欄位
- 輸出為可直接 include 的固定常數結構

關鍵依賴：`quantized_tensors` key 要完整且命名一致。

### H2. `write_fixed_test_vectors_header(output_path, input_q, fixed_reference)`
- 產生 `fixed_tcn_test_vectors.hpp`
- 內含：
  - `kReferenceInput`
  - `kGoldenLogitsInt32`
  - `kGoldenLogitsFloat`
  - `kGoldenProbabilities`
  - `kProbabilityTolerance`

用途：HLS testbench 的固定測試向量與 golden 比對標準。

---

## I. `main()` 主流程逐段

### I1. 建立輸出資料夾
- `artifacts/04_hardware_handoff/int8_export`
- `cpp_hls_prep/include/generated`

依賴：後續所有檔案輸出都寫到這兩處。

### I2. 建立模型並載入已凍結權重
- `build_default_model(num_inputs=12)`
- 載入 `frozen_tcn_best_model.pt`
- `model.eval()`

依賴：下一步量化從這個 `state_dict` 取得參數。

### I3. 量化全部權重並輸出 `.wts`
- 遍歷 `model.state_dict()`
- 每個 tensor：
  - 對稱量化成 int8 + scale
  - 記錄到 `manifest`
  - 存入 `quantized_tensors`
  - 寫入 `tcn_int8_weights.wts`

輸出格式包含：tensor 名稱、元素數、scale、shape、所有 int8 值。

### I4. 產生固定測試輸入
- `torch.manual_seed(99)` 保證可重現
- 生成 `sample_input_btf`（1,100,12）
- 轉成 `(12,100)` 再 flatten
- 量化成 `input_q`（int8）
- 寫出 `reference_input.txt`

依賴：`input_q` 同時餵給固定推論與 header 生成。

### I5. 產生 FP32 參考輸出
- 用原始 PyTorch 模型推論 `logits` 與 `softmax probabilities`
- 寫出 `reference_output.txt`

用途：可檢查「INT8 固定路徑」與「FP32 原模型」落差。

### I6. 跑固定整數參考推論
- `fixed_reference = run_fixed_reference_inference(quantized_tensors, input_q)`

輸出：`logits_int32/logits_float/probabilities` 等 golden 值。

### I7. 生成 HLS 需要的兩個 header
- `fixed_tcn_weights.hpp`
- `fixed_tcn_test_vectors.hpp`

目的：讓 HLS/測試不需要再讀文字檔，直接 include 常數。

### I8. 彙整 manifest 與 summary
- 組 `summary`（來源模型、檔案位置、參數量、scale、FP32 與 fixed 結果）
- 寫入 `tensor_manifest.json`（含 `summary` + `tensors`）
- 印出 summary 到終端

---

## J. 進入點

- `if __name__ == "__main__": main()`

用途：直接執行檔案時啟動整個匯出流程。

---

## K. 一句話總結整體依賴鏈

`frozen_tcn_best_model.pt`  
-> 量化成 int8 + scale（`quantized_tensors`）  
-> 產生 `.wts` 與 manifest  
-> 用固定整數算子跑 golden inference  
-> 生成 HLS 可直接使用的 `fixed_tcn_weights.hpp` 與 `fixed_tcn_test_vectors.hpp`。
