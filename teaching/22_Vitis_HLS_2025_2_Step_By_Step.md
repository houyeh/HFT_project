# 22. Vitis HLS 2025.2 一步一步實作教學

## 這份教學適合誰
如果你現在已經有：
- 一個可以在 C++ 跑的 fixed-shape TCN
- 一組固定權重 header
- 一組 fixed input / golden output

而你想要：
- 真的把它丟進 `Vitis HLS`
- 跑 `csim`
- 跑 `csynth`
- 看 synthesis report

那這份文件就是直接照著做的版本。

---

## 0. 你目前專案裡已經準備好的東西

### HLS top
- [fixed_tcn_top.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn_top.hpp)
- [fixed_tcn_top.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_top.cpp)

### HLS modules
- [fixed_tcn_modules.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/fixed_tcn_modules.hpp)
- [fixed_tcn_modules.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/fixed_tcn_modules.cpp)

### 固定權重與 golden vector
- [fixed_tcn_weights.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated/fixed_tcn_weights.hpp)
- [fixed_tcn_test_vectors.hpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated/fixed_tcn_test_vectors.hpp)

### HLS testbench
- [hls_testbench.cpp](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/src/hls_testbench.cpp)

### Vitis HLS 腳本
- [run_hls.tcl](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/run_hls.tcl)
- [run_vitis_hls_2025_2.ps1](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/run_vitis_hls_2025_2.ps1)

---

## 1. 本機環境確認

我目前在這台機器上確認到的安裝位置是：
- `C:\AMDDesignTools\2025.2\Vitis`
- `C:\AMDDesignTools\2025.2\Vivado`

AMD 官方對 Windows 啟動方式的說明可參考：
- [Setting Up the Environment to Run the Vitis Software Platform](https://docs.amd.com/r/2025.1-English/ug1742-vitis-release-notes/Setting-Up-the-Environment-to-Run-the-Vitis-Software-Platform)

其中 Windows 的核心概念就是：
1. 先跑 `settings64.bat`
2. 再啟動 `vitis` 或 `vitis-run`

---

## 2. 第一步：先重新產生權重與 golden header

在專案根目錄執行：

```powershell
python scripts\05_hardware_handoff\export_tcn_int8_artifacts.py
```

你應該會看到：
- `generated_weights_header`
- `generated_test_vectors_header`

對應輸出會寫到：
- [artifacts/04_hardware_handoff/int8_export](/C:/Users/holbo/00VS_projjects/00_HFT_project/artifacts/04_hardware_handoff/int8_export)
- [cpp_hls_prep/include/generated](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/include/generated)

---

## 3. 第二步：先在本機 C++ 驗證一次 testbench

```powershell
cmake -S cpp_hls_prep -B cpp_hls_prep/build
cmake --build cpp_hls_prep/build --config Release
& "C:\Users\holbo\00VS_projjects\00_HFT_project\cpp_hls_prep\build\tcn_hls_testbench.exe"
```

正確結果應該包含：

```text
[PASS] HLS testbench matched fixed golden logits/probabilities.
```

這一步很重要，因為如果連本機 C++ testbench 都沒過，就不要急著進 Vitis。

---

## 4. 第三步：先找出這台機器真的有安裝的 part

我實際在這台機器上查過，Vivado 能看到很多 `xc7z` 系列裝置。  
如果你想自己再查一次，可以在 PowerShell 執行：

```powershell
$tmp = Join-Path $env:TEMP 'list_parts.tcl'
@'
set parts [get_parts *]
puts "PART_COUNT=[llength $parts]"
set shown 0
foreach p $parts {
  if {$shown >= 20} {break}
  puts $p
  incr shown
}
exit
'@ | Set-Content -Path $tmp -Encoding ascii
cmd /c "\"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat\" -mode tcl -nojournal -nolog -notrace -source \"$tmp\""
```

我這次實際成功用來跑 HLS 的 part 是：
- `xc7z020clg400-1`

如果未來比賽指定了別的板子，你再把 `-Part` 換掉就好。

---

## 5. 第四步：先跑 csim

這是最推薦的第一步，因為快，而且可以先確認：
- `top function`
- `testbench`
- `golden vector`

指令：

```powershell
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage csim -Part xc7z020clg400-1
```

我已經在這台機器上實際跑過，結果是：
- `csim_design` 成功
- testbench 顯示 `[PASS]`

你看到類似下面這段就代表成功：

```text
[PASS] HLS testbench matched fixed golden logits/probabilities.
INFO: [SIM 211-1] CSim done with 0 errors.
```

---

## 6. 第五步：跑 csynth

當 `csim` 過了，再跑：

```powershell
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage csynth -Part xc7z020clg400-1
```

這一步會花比較久，因為 Vitis 要做真正的高階綜合。

### 我這邊實際觀察到的狀況
- `csynth_design` 可以開始跑
- 已經通過前面的語法與設計分析
- 會進到 loop pipeline、array partition 等 HLS 優化階段
- 但時間比 `csim` 長很多

也就是說，現在的專案已經不是卡在最前面，而是到了「正式做 synthesis」的階段。

---

## 7. 第六步：跑完整流程

如果你想一次跑：
- `csim`
- `csynth`
- `export_design`

就用：

```powershell
.\cpp_hls_prep\hls\run_vitis_hls_2025_2.ps1 -Stage full -Part xc7z020clg400-1
```

這會呼叫：
- [run_hls.tcl](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/run_hls.tcl)

而這支 Tcl 會自動做：
1. `open_project`
2. `set_top tcn_fixed_top`
3. `add_files`
4. `open_solution`
5. `set_part`
6. `create_clock`
7. `csim_design`
8. `csynth_design`
9. `export_design -format ip_catalog`

---

## 8. synthesis 結果要去哪裡看

如果 `csynth` 成功，主要結果會在：

- [cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report](/C:/Users/holbo/00VS_projjects/00_HFT_project/cpp_hls_prep/hls/project/fixed_tcn_hls/solution1/syn/report)

你之後最重要要看的幾個指標是：
- Estimated Clock / Target Clock
- Latency
- Initiation Interval
- LUT
- FF
- BRAM
- DSP

---

## 9. 如果工具卡住，最常見怎麼排

### 問題 1：part not installed
症狀：
- `Part 'xxx' is not installed`

解法：
- 換成這台機器實際有安裝的 part
- 我目前實測可用的是 `xc7z020clg400-1`

### 問題 2：host-side C++ 混進 synthesis
這次我已經做過一次這種修正：
- 把 `.wts` 載入與 `try/catch` 切到 host-only 檔案
- synthesis 路徑只保留 fixed core

所以你現在應該用的是：
- `fixed_tcn.cpp`
- `fixed_tcn_modules.cpp`
- `fixed_tcn_top.cpp`

而不是 host loader。

### 問題 3：std::copy / 動態容器引發 HLS 問題
這類問題很常見。  
解法通常是：
- 不要用 `std::copy`
- 改成明確的 `for` loop
- 不要用 `vector`
- 不要用 `unordered_map`

這也是為什麼我們要把模型改成 fixed-shape 版本。

---

## 10. 如果你想用 GUI 做

AMD 官方文件可以參考：
- [High-Level Synthesis with the Vitis](https://docs.amd.com/api/khub/documents/Id0U3PmdkCIHlbL~BUQABg/content)

但對你現在這個專案，我更建議先用 CLI / Tcl，因為：
- 可重現
- 比較不會漏設定
- 之後寫報告也比較清楚

如果你真的想開 GUI，也可以：

```powershell
cmd /c "\"C:\AMDDesignTools\2025.2\Vitis\settings64.bat\" && vitis -w C:\Users\holbo\00VS_projjects\00_HFT_project\cpp_hls_prep\hls\workspace"
```

之後再在 GUI 裡建立 HLS component，並使用同一組 source / top / testbench。

---

## 11. 建議你實際操作的順序

如果你今天就要自己跟著跑，我建議照這個順序：

1. `python scripts\05_hardware_handoff\export_tcn_int8_artifacts.py`
2. `cmake -S cpp_hls_prep -B cpp_hls_prep/build`
3. `cmake --build cpp_hls_prep/build --config Release`
4. 跑 `tcn_hls_testbench.exe`
5. 跑 `run_vitis_hls_2025_2.ps1 -Stage csim -Part xc7z020clg400-1`
6. 確認 `csim` PASS
7. 再跑 `-Stage csynth`
8. 最後再跑 `-Stage full`

---

## 12. 小結

如果用一句話總結這份教學：

> 你現在手上的專案，已經不是只有「準備要進 HLS」，而是已經有固定 top、固定權重、固定 testbench，以及可直接在本機 `AMDDesignTools 2025.2` 上執行的 `Vitis HLS` 腳本。
