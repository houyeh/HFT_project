# FPGA Demo Setup Guide: PYNQ-Z2 PHC Replay Wrapper

status: `v3diag_on_board_verified_positive_and_negative`

## V3 Diagonal On-Board Update (2026-10-04)

The fastest V3 diagonal design is now programmed and functionally verified on
the physical PYNQ-Z2. The timing-closed positive VIO bitstream completed all
4,096 NEW4 replay rows with zero exact readout drift and zero decision flips.
A separate timing-closed negative-control bitstream injected one 16-LSB fault
at configured index 17 and the on-board comparator asserted `fail=1` with
exactly one mismatch.

Use this positive diagnostic artifact for reproducible VIO readback:

```text
results_v3_diag\pnr_100mhz_readback_vio_postroute_physopt\phc_v3diag_demo_100mhz_readback_vio_postroute_physopt.bit
results_v3_diag\pnr_100mhz_readback_vio_postroute_physopt\phc_v3diag_demo_100mhz_readback_vio_postroute_physopt.ltx
```

The V3 latency headline is `530 cycles / 100 MHz = 5.300 us`. This is derived
from routed timing closure, not measured by an on-chip cycle counter or ILA.
See `V3DIAG_STAGE_E_REPORT.md` for the complete claim boundary and raw-log
paths. The older 25.733-us instructions below are retained as provenance for
the prior dense demo rather than as the current fastest artifact.

## Demo Narrative Baseline Update

- Replay burst stays on `NEW4_20260112_TXFA6`.
- Window selection rule: highest 1-second rolling event count among eligible `val/new3/new4` sessions. This is an objective reproducible rule, not manual cherry-picking for visual drama.
- CPU-side comparison baseline: N=4 concurrent LibTorch/TorchScript benchmark, not N=1.
- Fixed CPU mouthpiece for the demo: `n2ew_era\06f_phc_lite_manual_hls\results\cpu_concurrent_benchmark_N4.json`, round 1.
- N=4 round 1 numbers: mean `72.198630 us`, p50 `65.600000 us`, p95 `126.300000 us`, p99 `262.600000 us`, deadline `83.160000 us`, deadline miss rate `15.1250%`.
- N=4 round 2 also exists with deadline miss rate `15.1375%`; use round 1 consistently in slides/talk to keep one clean narrative.
- FPGA hardware line for this wrapper remains `25.733 us` from the complete 06S seed2 / 06L unroll16 90 MHz routed timing-met bitstream.

## 0. 先講結論

你現在沒有插板，沒關係。今晚已經能先完成到「可拿去插板測」的狀態：

- wrapper 正向 XSim 已通過：4096 筆 replay，`POSITIVE_TEST_PASS`
- wrapper 負向 XSim 已通過：故意竄改 index 17，成功抓到 mismatch
- PYNQ-Z2 board-level P&R 已通過：90 MHz core clock，WNS `+0.110 ns`
- bitstream 已產生：`C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper.bit`

但還不能宣稱「已上板驗證」。因為板子未接上，所以 JTAG programming、實機 LED、ILA/VIO readback 都尚未跑。

## 1. 這版 bitstream 是哪一個硬體點

這次實際可用的 wrapper 選用：

```text
06S seed2 / 06L unroll16 90 MHz complete RTL
headline latency = 2316 cycles / 25.733 us @ 90 MHz
```

原本比較漂亮的 06U V1 `22.36 us` 版本沒有用在這次 bitstream，原因是 repo 裡的 06U V1 RTL 匯出不完整，standalone wrapper XSim 缺 HLS 自動產生的子模組。

Demo staleness scoreboard 也已經改成使用 `25.733 us` 這個選定硬體延遲值。

## 2. 已完成檔案

Repo root:

```text
C:\Users\holbo\00VS_projjects\00_HFT_project
```

Phase folder:

```text
C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup
```

| 類別 | 檔案 |
|---|---|
| Bitstream | `results\phc_demo_wrapper.bit` |
| P&R timing report | `results\wrapper_timing_summary_routed.rpt` |
| P&R utilization report | `results\wrapper_utilization_routed.rpt` |
| P&R summary | `results\wrapper_pnr_summary.csv` |
| P&R full log | `logs\wrapper_pnr_vivado.log` |
| Positive XSim log | `logs\xsim_direct_positive_xsim.log` |
| Negative XSim log | `logs\xsim_direct_negative_xsim.log` |
| Board top RTL | `rtl\pynq_z2_phc_demo_board_top.sv` |
| Wrapper RTL | `rtl\pynq_z2_phc_demo_wrapper.sv` |
| Clock MMCM RTL | `rtl\pynq_z2_clock_90mhz.sv` |
| Board XDC | `xdc\pynq_z2_demo_wrapper_board.xdc` |
| Hardware detect Tcl | `tcl\check_pynq_z2_hw.tcl` |
| Program Tcl | `tcl\program_pynq_z2_demo.tcl` |

## 3. 目前 P&R 數字

| 指標 | 數字 |
|---|---:|
| Input clock | 125 MHz, PYNQ-Z2 `sysclk` H16 |
| Core clock | 90 MHz, MMCM output |
| Core latency | 2316 cycles / 25.733 us |
| Setup WNS | +0.110 ns |
| Setup failing endpoints | 0 |
| Hold WHS | +0.008 ns |
| Hold failing endpoints | 0 |
| LUT | 15,694 / 53,200 = 29.50% |
| FF | 20,294 / 106,400 = 19.07% |
| DSP | 131 / 220 = 59.55% |
| Block RAM Tile | 28.5 / 140 = 20.36% |
| MMCM | 1 / 4 = 25.00% |

## 4. 這版 bitstream 怎麼看結果

這是 no-ILA LED sanity bitstream。目前沒有 `.ltx` probes 檔。

LED 定義：

| LED | 意義 | 正常狀態 |
|---|---|---|
| LED0 | replay done | replay 結束後亮 |
| LED1 | pass | replay 結束且 comparator pass 後亮 |
| LED2 | fail | 正常應保持暗；亮代表 mismatch |
| LED3 | heartbeat | MMCM lock 且 clock 正常時閃爍 |

按鍵：

| 按鍵 | 功能 |
|---|---|
| BTN0 | reset / rerun |
| BTN1 | 手動 start；但 bitstream 也會 reset 後自動 start 一次 |

正常 demo 畫面應該是：

```text
LED3 持續閃爍
replay 跑完後 LED0 = 1
replay 跑完後 LED1 = 1
LED2 = 0
```

## 5. 明天插板後 GUI 操作

1. 接 PYNQ-Z2 電源。
2. 接 USB/JTAG 到電腦。
3. 確認板子 jumper/boot 設定允許 JTAG programming。
4. 開 Vivado 2025.2。
5. 點 `Open Hardware Manager`。
6. 點 `Open Target` -> `Auto Connect`。
7. 如果看到 `xc7z020` 裝置，右鍵 `Program Device`。
8. Bitstream 選：

```text
C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup\results\phc_demo_wrapper.bit
```

9. 這版沒有 `.ltx`，不要選 probes 檔。
10. Program 完成後看 LED：
    - LED3 是否閃爍
    - LED0/LED1 是否在 replay 完成後亮起
    - LED2 是否保持暗

## 6. 明天插板後 CLI 操作

### 6.1 如果你在 CMD / Command Prompt

CMD 裡不要打 PowerShell 的 `& '...'` 語法。直接跑我準備好的 `.cmd` helper：

```cmd
cd /d C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup
scripts\check_pynq_z2_hw.cmd
```

也可以不用 helper，CMD 原生寫法是：

```cmd
cd /d C:\Users\holbo\00VS_projjects\00_HFT_project
"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat" -mode batch -source "n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\check_pynq_z2_hw.tcl"
```

### 6.2 如果你在 PowerShell

PowerShell 才需要 `&` call operator：

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project
& 'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat' -mode batch -source n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\check_pynq_z2_hw.tcl
```

看到類似下面結果才代表有偵測到硬體：

```text
HW_DEVICES_BEGIN
xc7z020_...
HW_DEVICES_END
PYNQ_Z2_DETECTION_STATUS: hw_device_present
```

偵測到後再燒錄。CMD 寫法：

```cmd
cd /d C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06s2_pynq_z2_wrapper_bringup
scripts\program_pynq_z2_demo.cmd
```

PowerShell 寫法：

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project
& 'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat' -mode batch -source n2ew_era\06s2_pynq_z2_wrapper_bringup\tcl\program_pynq_z2_demo.tcl
```

成功時應看到：

```text
PYNQ_Z2_PROGRAM_STATUS: programmed
```

## 7. 常見失敗排查

### 偵測不到板子

現象：

```text
PYNQ_Z2_DETECTION_STATUS: no_hw_device_detected
```

先查：

- USB/JTAG 線有沒有接好
- 板子電源有沒有開
- Windows 裝置管理員有沒有看到 USB/JTAG 裝置
- Vivado Hardware Server 是否正常
- GUI 裡 `Open Target -> Auto Connect` 是否能看到裝置

### LED3 不閃

LED3 是 heartbeat。若不閃，通常代表 clock/reset 沒起來。

這版使用 PYNQ-Z2 的 H16 `sysclk` 125 MHz，再用 MMCM 產生 90 MHz core clock。如果 LED3 完全不動，優先檢查：

- 板子是否真的有 125 MHz sysclk
- XDC 是否套到正確板型
- MMCM 是否 lock
- BTN0 reset 是否被卡住

如果 H16 clock 在你的實板環境不可用，下一個 fallback 是改成 PS FCLK0 供 clock；那會需要另一版 block design，不是這版 bitstream 的行為。

### LED2 亮

LED2 是 fail。若亮，代表 replay comparator 抓到 mismatch。

不要把它硬說成 demo pass。請回頭看：

- `logs\xsim_direct_positive_xsim.log`
- `results\bram_expected_from_hls_csim_manifest.json`
- bitstream 是否就是 `results\phc_demo_wrapper.bit`

### 找不到 ILA probes

這是預期內的。現在 bitstream 是 no-ILA 版本，沒有 `.ltx`。這版先用 LED 做 sanity demo。

若真的需要 ILA，需要另外在 Vivado GUI 裡用 `MARK_DEBUG` nets 插入 debug core，或修好 direct-flow ILA IP 產生流程後重跑 P&R。

## 8. 現場口徑

建議用語：

```text
這個 bitstream 已經完成 wrapper-level XSim positive/negative 測試，並在 PYNQ-Z2 目標上完成 routed P&R 與 bitgen。因為昨晚板子沒有接上，JTAG program 與實板 LED sanity check 是今天插板後要做的最後一步。
```

不要說：

```text
已完成實體板上驗證。
已完成 ILA capture。
FPGA 改善了 PnL。
```

## 9. Pinout 依據

本版 XDC 使用 PYNQ-Z2 公開 pinout / master XDC 常見定義：

- `sysclk` 125 MHz: H16
- LED0/1/2/3: R14 / P14 / N16 / M14
- BTN0/BTN1: D19 / D20
- I/O standard: LVCMOS33

Reference:

- [PYNQ-Z2 Pinout](https://wiki.dcae.pub.ro/index.php/Pynq-Z2_-_Pinout_%282023%29)
- [PYNQ-Z2 Master Constraints snippet](https://gitlab.com/-/snippets/2440950)
- [PYNQ-Z2 pin assignment summary](https://deepwiki.com/Xilinx/xup_embedded_system_design_flow/3.1.2-pynq-z2-pin-assignments)
