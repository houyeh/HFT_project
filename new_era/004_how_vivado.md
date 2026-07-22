# 為什麼終端機會說找不到 `vivado_hls`？

你剛剛在 VS Code 終端機遇到 `vivado_hls is not recognized`，這是非常正常的現象！
**Vivado 不是可以用 `pip install` 安裝的 Python 套件**，它是 AMD/Xilinx 官方推出的超大型 FPGA 開發環境（動輒 50GB 以上）。

你必須獨立下載並安裝這個軟體，才能進行硬體合成。以下是從零到有的完整實戰指南：

---

## 🛠️ Step 1: 下載與安裝 Vivado

1. **取得軟體**：
   - 到 Xilinx 官網下載 **Vivado/Vitis 統一軟體平台 (推薦 2025.2 或更新版本)**。
   - *(注意：從 2020 年版本開始，Xilinx 將負責硬體合成的 `vivado_hls` 全面升級並改名為 `vitis_hls`，請確保安裝了 Vitis HLS 元件。)*
2. **安裝選項**：
   - 在安裝過程中，請務必勾選支援 **"Zynq-7000"** 晶片系列 (PYNQ-Z2 的核心)。
   - 務必確認勾選安裝 **"Vitis HLS"** (取代了舊版的 Vivado HLS)。

---

## 🏗️ Step 2: 執行 C Synthesis (將 C++ 轉為硬體電路 IP)

現在你已經有了 `004_hls4ml_prj` 資料夾（裡面裝滿了 C++ 程式碼），我們要把這些程式碼「燒」成矽智財 (IP Core)：

你有兩種超簡單的方法可以完成這件事（任選其一）：

### 方法 A：使用 Vitis HLS 2025.2 IDE 介面 (強烈推薦新手)
1. 從 Windows 開始選單搜尋並打開 **"Vitis HLS 2025.2"** 應用程式。
2. 點擊 **"Open Project"**，並瀏覽選擇我們產生的專案資料夾（例如 `007_hls4ml_tcn_prj`）。
3. 因為 Vitis 2025.2 已經全面升級為「Unified IDE」(介面長得很像 VS Code)，舊版的綠色播放鍵已經取消了。請點擊上方選單列的 **Terminal -> New Terminal**。
4. 在下方彈出的終端機中，直接輸入最新的編譯指令 (注意 2025 版已經改用 `vitis-run` 指令)：
   ```bash
   vitis-run --mode hls --tcl build_prj.tcl
   ```
5. 等待終端機跑完（C Synthesis），你就可以在左側資料夾 `myproject_prj/solution1/syn/report` 裡找到 `.rpt` 報表，查看神經網路的 **Latency (時鐘週期)** 與 **DSP/BRAM 使用量**！

### 方法 B：使用 Command Line (終端機指令)
1. 從 Windows 開始選單搜尋並打開 **"Vitis command line tool 2025.2"** 或 **"Vivado 2025.2 Tcl Shell"**。
   *(注意：不要用一般的 VS Code 終端機或 PowerShell，因為它們沒有綁定 FPGA 編譯器的環境變數！)*
2. 使用 `cd` 切換到你的專案目錄：
   ```bash
   cd C:\Users\holbo\00VS_projjects\00_HFT_project\new_era\007_hls4ml_tcn_prj
   ```
3. 執行自動合成腳本 (注意：舊版叫 `vitis_hls`，2025 新版請務必用 `vitis-run`)：
   ```bash
   vitis-run --mode hls --tcl build_prj.tcl
   ```
4. **等待與收成**：跑完後，它會在專案的 `solution1/impl/ip` 資料夾裡生出一個 `.zip` 檔 (這就是神經網路硬體 IP Core)。

---

## 🧩 Step 3: 在 Vivado 中繪製 Block Design (接線)

我們有了「神經網路大腦 (IP Core)」，現在要把它接上 PYNQ-Z2 的「中樞神經 (Zynq ARM 處理器)」。

1. **建立新專案**：
   - 打開有著精美 UI 的 **Vivado 2019.2** (不是 HLS)。
   - 建立一個 RTL Project，並在 Boards 分頁中選擇 **PYNQ-Z2**。
2. **匯入神經網路 IP**：
   - 左側選單點擊 `IP Catalog` -> 右鍵選 `Add Repository`，選擇剛剛產生的 `004_hls4ml_prj/myproject_prj/solution1/impl/ip` 資料夾。
3. **建立 Block Design (BD)**：
   - 點擊左側 `Create Block Design`。
   - 點擊 `+` 號，加入 `ZYNQ7 Processing System`。
   - 點擊 `+` 號，加入你的神經網路 IP (通常叫 `myproject_axi` 或類似名稱)。
   - 點擊畫面上方的 **"Run Block Automation"** 與 **"Run Connection Automation"**。Vivado 會像魔法一樣自動幫你把神經網路與 ARM 處理器之間的 AXI DMA 匯流排接好！
4. **包裝與生成**：
   - 在 Sources 面板的 `.bd` 檔上點擊右鍵，選 `Create HDL Wrapper`。
   - 最後，按下左下角的 **"Generate Bitstream"**！

---

## 🚀 Step 4: 燒錄進 PYNQ-Z2 實戰預測！

Vivado 跑完後，你會在專案的 `runs/impl_1` 資料夾中找到兩個無價之寶：
1. **`.bit` 檔** (硬體電路的靈魂，就是 FPGA 的燒錄檔)
2. **`.hwh` 檔** (硬體規格說明書，讓 Python 知道怎麼控制這塊硬體)

將這兩個檔案改名為同名 (例如 `hft_model.bit` 與 `hft_model.hwh`)，然後上傳到 PYNQ-Z2 的 Jupyter Notebook 裡。

在 PYNQ 裡跑起來非常簡單：
```python
from pynq import Overlay
import pynq.lib.dma

# 1. 載入神經網路硬體加速器
overlay = Overlay("hft_model.bit")
dma = overlay.axi_dma_0

# 2. 將特徵放進記憶體
input_buffer = pynq.allocate(shape=(800,), dtype='float32') # (你的 8 個特徵 x 100 ticks)
output_buffer = pynq.allocate(shape=(3,), dtype='float32')  # (Down, Stationary, Up)

# 寫入資料
input_buffer[:] = my_real_time_market_data

# 3. 光速觸發 FPGA (不佔用 CPU 運算)
dma.sendchannel.transfer(input_buffer)
dma.recvchannel.transfer(output_buffer)
dma.sendchannel.wait()
dma.recvchannel.wait()

print(f"FPGA 預測機率: {output_buffer}")
```
**恭喜你，你的神經網路現在是用光速在晶片裡面跑了！**
