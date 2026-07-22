// generate_report.js
// Usage: node generate_report.js
// Output: PHC_lite_HLS_Report.docx
// Requires: npm install docx  (run once in this folder)

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, TableOfContents,
  PageBreak
} = require('docx');
const fs = require('fs');

// ─── Helpers ──────────────────────────────────────────────────────────────────

const BLUE   = "2E5FA3";
const LGRAY  = "F2F2F2";
const DGRAY  = "404040";
const WHITE  = "FFFFFF";
const ACCENT = "1F7AC1";

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 120 },
    children: [new TextRun({ text, bold: true, color: BLUE, size: 36, font: "Arial" })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text, bold: true, color: DGRAY, size: 28, font: "Arial" })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 60 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial" })]
  });
}
function para(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 80, after: 100 },
    children: [new TextRun({ text, size: 22, font: "Arial", ...opts })]
  });
}
function paraRuns(runs) {
  return new Paragraph({
    spacing: { before: 80, after: 100 },
    children: runs.map(r =>
      typeof r === 'string'
        ? new TextRun({ text: r, size: 22, font: "Arial" })
        : new TextRun({ size: 22, font: "Arial", ...r })
    )
  });
}
function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, size: 22, font: "Arial" })]
  });
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}
function spacer(lines = 1) {
  return new Paragraph({ spacing: { before: 0, after: lines * 160 }, children: [new TextRun("")] });
}

// ─── Table builder ────────────────────────────────────────────────────────────

const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const PAGE_WIDTH = 9360; // US Letter 1" margins

function makeTable(headers, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  function cell(text, isHeader = false) {
    return new TableCell({
      borders: BORDERS,
      width: { size: colWidths[0], type: WidthType.DXA },
      shading: isHeader
        ? { fill: BLUE,  type: ShadingType.CLEAR }
        : { fill: WHITE, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 140, right: 140 },
      children: [new Paragraph({
        children: [new TextRun({
          text,
          size: 20,
          font: "Arial",
          bold: isHeader,
          color: isHeader ? WHITE : "000000"
        })]
      })]
    });
  }
  function row(cells, isHeader = false) {
    return new TableRow({
      tableHeader: isHeader,
      children: cells.map((c, i) => new TableCell({
        borders: BORDERS,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: isHeader
          ? { fill: BLUE, type: ShadingType.CLEAR }
          : { fill: i % 2 === 0 ? WHITE : LGRAY, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 140, right: 140 },
        children: [new Paragraph({
          children: [new TextRun({
            text: c,
            size: 20,
            font: "Arial",
            bold: isHeader,
            color: isHeader ? WHITE : "000000"
          })]
        })]
      }))
    });
  }
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      row(headers, true),
      ...rows.map(r => row(r, false))
    ]
  });
}

// ─── Document sections ────────────────────────────────────────────────────────

// TITLE PAGE
const titlePage = [
  spacer(3),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 160 },
    children: [new TextRun({ text: "PHC-lite Neural Network Inference on FPGA", bold: true, size: 56, font: "Arial", color: BLUE })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "High-Frequency Trading FPGA 競賽專案", size: 32, font: "Arial", color: DGRAY })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "完整技術報告 | 2026-06-15", size: 24, font: "Arial", color: DGRAY })]
  }),
  spacer(2),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 1 } },
    children: [new TextRun("")]
  }),
  spacer(1),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "平台：PYNQ-Z2（Xilinx xc7z020clg400-1）", size: 22, font: "Arial" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "工具：Vitis HLS 2025.2 + Vivado 2025.2", size: 22, font: "Arial" })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 80 },
    children: [new TextRun({ text: "狀態：csim ✓  cosim ✓  P&R Timing Met ✓", size: 22, font: "Arial", color: "2E7D32" })]
  }),
  pageBreak()
];

// SECTION 1: 背景與動機
const sec1 = [
  h1("1. 背景與動機：為什麼要這樣做？"),

  h2("1.1 什麼是高頻交易（HFT）？"),
  para("高頻交易（High-Frequency Trading, HFT）是一種在極短時間內（微秒到毫秒級別）自動下單買賣金融商品的策略。在傳統交易中，人類交易員觀察行情、分析訊號、決定下單，整個過程可能需要幾秒甚至幾分鐘。HFT 讓機器在人類還沒來得及反應之前就完成了這整個流程。"),
  para("速度為何如此關鍵？金融市場是一個競爭性環境：如果你比對手晚 1 微秒接收到行情變動的資訊，你的策略就可能因為「舊資料」而做出錯誤決策，甚至在訂單到達交易所之前，好的價格已經被別人搶走了。在某些策略中，延遲降低 10 µs 就可以直接影響獲利能力。"),

  h2("1.2 為什麼使用 FPGA 而不是 CPU 或 GPU？"),
  paraRuns([
    "現代 CPU 跑一個 Python 推論需要約 ",
    { text: "84 µs（中位數）到 779 µs（p99）", bold: true },
    "，而且這個時間是不確定的（jitter）——作業系統隨時可能把 CPU 讓給其他程序，造成突然的延遲飆升。GPU 雖然有高度並行能力，但它的 batch 特性讓單次推論的延遲反而更高，更不適合 HFT。"
  ]),
  para("FPGA（Field-Programmable Gate Array，現場可程式化閘陣列）是一種可以被重新「佈線」成任何數位電路的晶片。它的核心優勢："),
  bullet("確定性延遲（Deterministic Latency）：不受作業系統、記憶體垃圾回收、快取命中率影響，每次執行時間完全相同"),
  bullet("無軟體層（Bare Metal）：沒有 Linux、沒有 Python、沒有 PyTorch，訊號直接進晶片、算完直接出"),
  bullet("可客製化管線（Custom Pipeline）：可以根據模型架構設計專屬的計算電路，不像 CPU 需要共用通用指令集"),
  spacer(),

  h2("1.3 本專案的目標"),
  para("在 xc7z020 這顆學術/入門級 FPGA 上，實作一個 PHC-lite 神經網路的推論核心，並完成以下驗證："),
  bullet("C simulation（csim）：在電腦上確認邏輯正確性"),
  bullet("RTL Co-simulation（cosim）：在模擬的 RTL（硬體描述語言）層面確認 bit-exact 一致性"),
  bullet("Place & Route（P&R）：讓 Vivado 真正把電路佈在這顆晶片上，確認資源和時序都符合規格"),

  pageBreak()
];

// SECTION 2: 模型架構
const sec2 = [
  h1("2. PHC-lite 模型架構：我們在 FPGA 上跑的是什麼？"),

  h2("2.1 PHC-lite 是什麼？"),
  para("PHC-lite 是 PHCSSM（Predictive Hierarchical Continuous State-Space Model）的輕量版本，專為 FPGA 可行性展示而設計的 16 維隱藏狀態版本。它的核心概念是一個「記憶體型」神經網路：每一步的輸出不只依賴當前輸入，還依賴上一個時步的狀態（state），這讓它可以「記住」市場的歷史走勢。"),
  para("PHC-lite 的功能：接收當前市場特徵向量（16 個數字），結合上一步的隱藏狀態（16 個數字），輸出："),
  bullet("readout[2]：2 維的預測訊號（正負號代表買/賣方向）"),
  bullet("state_next[16]：更新後的隱藏狀態，傳入下一個時步"),

  h2("2.2 為什麼要用 Recurrent（遞迴）結構？"),
  para("市場是有時序性的——現在的價格和過去幾秒的走勢有關。一個只看「當前快照」的 feed-forward 網路會錯失這些歷史資訊。PHC-lite 透過 state 傳遞機制讓模型有記憶，每一步都在融合當前市場和過去趨勢的資訊。"),
  paraRuns([
    { text: "重要限制：", bold: true },
    "由於 t+1 步需要用到 t 步的 state_next 作為輸入，這兩步之間存在資料相依性，因此不能把多個時步的計算疊加 pipeline。每次計算必須等前一步完成。這不是實作上的缺陷，而是 recurrent 架構的根本約束。"
  ]),

  h2("2.3 計算流程逐步說明"),
  para("PHC-lite 的一個時步共分 9 個計算步驟："),

  makeTable(
    ["步驟", "計算內容", "維度變化", "白話說明"],
    [
      ["Step 1", "ENC_LINEAR → LayerNorm → SiLU", "16→16→16→16", "把輸入 x_t 線性投影，正規化，再過激活函數，得到 encoded 特徵"],
      ["Step 2", "GATE_LINEAR(encoded) → Sigmoid", "16→16→16", "計算「遺忘閘」：決定哪些過去記憶要保留（Sigmoid 輸出 0~1）"],
      ["Step 3", "CANDIDATE_LINEAR(encoded) → Tanh", "16→16→16", "計算「候選記憶」：這個時步想要寫入的新資訊（Tanh 輸出 -1~+1）"],
      ["Step 4", "state_temporal = DECAY⊙state_prev + gate⊙candidate", "16+16→16", "更新記憶：衰減舊狀態 + 加入新資訊（⊙ = element-wise 乘法）"],
      ["Step 5", "Gain 計算 → Sigmoid", "16+16→16", "計算增益：決定訊息傳遞的強度"],
      ["Step 6", "SYNAPSE_LINEAR(state) → SiLU", "16→16→16", "計算突觸訊息：模仿神經元之間的訊號傳遞"],
      ["Step 7", "loop_input = state + α⊙gain⊙message", "16→16", "整合訊息：把突觸訊息加回狀態（自迴路更新）"],
      ["Step 8", "z = LayerNorm(loop_input)", "16→16", "最終正規化：讓輸出值分布穩定"],
      ["Step 9", "readout = READOUT_LINEAR(z)", "16→2", "輸出預測訊號：2 維的買/賣方向強度"]
    ],
    [700, 2700, 1800, 4160]
  ),
  spacer(),

  h2("2.4 模型中的數學函數"),
  paraRuns([
    { text: "LayerNorm（層正規化）：", bold: true },
    "把一個向量中的 16 個數字調整為平均值 0、標準差 1 的分布。這能讓訓練更穩定，避免某個維度的數值過大或過小。公式：z = (x - mean) / sqrt(var + ε)，其中 mean 和 var 是這 16 個數字的平均值和變異數。"
  ]),
  paraRuns([
    { text: "Sigmoid(x)：", bold: true },
    "把任意實數壓縮到 (0, 1)，作為「閘門」使用。值越接近 1，代表「打開」（保留記憶）；越接近 0，代表「關閉」（遺忘）。"
  ]),
  paraRuns([
    { text: "Tanh(x)：", bold: true },
    "把任意實數壓縮到 (-1, +1)，用於表達「候選記憶」的方向性（正/負）。"
  ]),
  paraRuns([
    { text: "SiLU(x) = x × Sigmoid(x)：", bold: true },
    "一種平滑的激活函數，結合了線性特性和非線性特性，在深度學習中常用來取代 ReLU。"
  ]),

  pageBreak()
];

// SECTION 3: 工具鏈與設計策略
const sec3 = [
  h1("3. 工具鏈選擇：為什麼這樣設計？"),

  h2("3.1 什麼是 HLS（High-Level Synthesis）？"),
  para("傳統的 FPGA 設計需要工程師手寫 Verilog 或 VHDL——一種描述「電路連接方式」的語言，極度繁瑣且難以除錯。HLS（高層次合成）讓工程師可以用 C++ 寫設計，由工具自動翻譯成硬體電路（RTL）。"),
  para("類比：HLS 對 FPGA 的關係，就像編譯器對 CPU 的關係——你寫高階語言，工具幫你轉成機器語言（或電路）。"),

  h2("3.2 為什麼選手動 HLS 而不是 hls4ml？"),
  paraRuns([
    "hls4ml 是 CERN 開發的工具，可以直接把 PyTorch/Keras 模型自動轉成 HLS 程式碼，大幅降低入門門檻。但它有一個關鍵問題：",
    { text: "預設策略是盡量展開（unroll）所有迴圈，追求最低延遲，代價是資源用量極高。", bold: true }
  ]),
  para("在 xc7z020 這顆只有 220 個 DSP48E1 的入門級 FPGA 上，hls4ml 的預設設定很可能直接讓資源使用量超過 100%，導致無法 P&R（place and route，佈局佈線）。手動 HLS 讓我們能夠精確控制每個 pragma（指令），在延遲和資源之間做出明確的 tradeoff 選擇。"),

  h2("3.3 定點數（Fixed-Point）：為什麼不用浮點？"),
  para("浮點數（float32）是電腦軟體的標配，但在 FPGA 硬體上，一個浮點數乘法器需要大量 LUT 和 DSP。更嚴重的是，Vitis HLS 如果看到 C 標準庫的 tanh() 函數，會呼叫 IP 核心（IP core），一個核心就需要 2000+ 個 DSP，直接超過整個晶片的上限。"),
  para("定點數是一種「位置固定的小數」：ap_fixed<24,12> 代表用 24 個 bit 表示一個數字，其中 12 bit 是整數部分，12 bit 是小數部分。定點數的乘法直接對應一個 DSP48E1 硬體乘法器，非常高效。"),

  makeTable(
    ["型別", "C++ 別名", "整數位", "小數位", "範圍", "用途"],
    [
      ["ap_fixed<24,12>", "phc_fixed_t", "12 bit", "12 bit", "±2048，精度 1/4096", "I/O 資料、權重"],
      ["ap_fixed<40,16>", "phc_acc_t",  "16 bit", "24 bit", "±32768，精度極高", "乘法累加（避免溢位）"],
      ["ap_fixed<48,24>", "phc_var_t",  "24 bit", "24 bit", "±8388608", "LayerNorm 變異數計算"]
    ],
    [1800, 1500, 900, 900, 2400, 1860]
  ),
  spacer(),
  paraRuns([
    { text: "為什麼累加器要更寬？", bold: true },
    "把 16 個 24-bit 數字相乘再相加，結果的範圍比單個乘積大得多。若用 24-bit 存累加結果，很容易「overflow（溢位）」。用 40-bit 的累加器留有足夠的精度空間。"
  ]),

  h2("3.4 Tanh / Sigmoid 近似：LUT 查表法"),
  para("即使不用浮點，Tanh 也是一個超越函數（transcendental function）——不能直接用四則運算算出來，需要 Taylor 展開或其他近似方法，每種方法都需要額外硬體。"),
  para("我們的解決方案：預先在 Python 中計算 65 個均勻分布的 Tanh 值（範圍 [-4, +4]，步長 0.125），存在一個常數陣列（TANH_LUT[65]）裡。查表時，根據輸入值找到最近的兩個點，做線性插值。"),
  bullet("Sigmoid 從 Tanh 推導：sigmoid(x) = (1 + tanh(x/2)) / 2，共用同一個 LUT，不增加額外硬體"),
  bullet("SiLU 計算：silu(x) = x × sigmoid(x)，同樣共用 LUT"),
  bullet("精度：csim 實測 max_abs_diff = 0.00725，遠小於 tolerance = 1.0"),
  bullet("LUT 陣列加 ARRAY_PARTITION complete pragma，讓 HLS 把它放進 register（FF），每個時脈週期可以同時讀取任意索引，不像 BRAM 有讀取延遲"),

  pageBreak()
];

// SECTION 4: 優化歷程
const sec4 = [
  h1("4. 優化歷程：八輪 pragma 調整"),

  h2("4.0 背景知識：FPGA 資源"),
  para("在閱讀優化歷程之前，需要先了解 xc7z020 的四種主要資源："),
  makeTable(
    ["資源", "全名", "功能", "可用量", "類比"],
    [
      ["LUT", "Look-Up Table", "組合邏輯運算（AND/OR/加法等）", "53,200", "CPU 的 ALU"],
      ["FF",  "Flip-Flop（正反器）", "儲存一個 bit 的暫存器", "106,400", "CPU 的暫存器"],
      ["DSP48E1", "Digital Signal Processor", "高速乘法-累加（MAC）硬體", "220", "GPU 的矩陣乘法核心"],
      ["BRAM_18K", "Block RAM", "18Kb 的雙埠記憶體", "280 (= 140個36K)", "L1 Cache"]
    ],
    [1000, 2200, 3200, 900, 2060]
  ),
  spacer(),
  paraRuns([
    { text: "關鍵指標：", bold: true },
    "DSP 利用率是最主要的瓶頸指標。一個 16×16 的定點矩陣乘法，如果完全展開（每個乘法都有獨立硬體），需要 16×16 = 256 個乘法器，超過整個晶片的 DSP 數量（220）。所以我們只能序列化計算。"
  ]),

  h2("4.1 v1：初始版本（DSP 用量 2636%）"),
  paraRuns([
    { text: "問題：", bold: true },
    "C 標準函式庫的 tanh() 函數在 HLS 工具內部被轉換為雙精度浮點（double）的 IP 核心，一個核心就需要 2048 個 DSP，加上 sqrt 核心、exp 核心，合計超過 5800 個 DSP，是可用量的 26 倍。"
  ]),
  paraRuns([
    { text: "根因：", bold: true },
    "HLS 工具看到 tanh()，直接呼叫 Xilinx 內建的 generic_tanh_double_s IP，這是一個完整的 IEEE 754 double precision 浮點運算單元。"
  ]),
  paraRuns([{ text: "修復：", bold: true }, "改用 65 點查表的定點 Tanh（TANH_LUT），完全消除浮點 IP 核心的呼叫。"]),

  h2("4.2 v2：LUT Tanh 後（DSP 仍 1603%）"),
  paraRuns([{ text: "問題：", bold: true }, "每個 linear 函式有 #pragma HLS PIPELINE II=1，這個 pragma 讓 HLS 強制展開所有 inner loop，每個 16×16 矩陣乘法展開為 256 個並行乘法器，6 個 layer × 256 = 1536 個 DSP 同時存在。"]),
  paraRuns([{ text: "修復方向：", bold: true }, "需要讓 HLS 序列化計算，而不是全部並行。"]),

  h2("4.3 v3～v5：嘗試限制 UNROLL（全部無效）"),
  para("嘗試了以下三種方法，全部沒有降低 DSP："),
  bullet("v3：移除 inner loop 的 #pragma HLS UNROLL → 無效"),
  bullet("v4：把陣列 partition 從 complete 改成 cyclic → 無效"),
  bullet("v5：把 PIPELINE II 從 1 改成 297 → 無效"),
  paraRuns([
    { text: "根本原因：", bold: true },
    "任何形式的 #pragma HLS PIPELINE（不論 II 值是多少）都會觸發 Vitis HLS 的自動行為：WARNING [HLS 214-275] ——它會強制把所有 partial UNROLL 升格為 complete UNROLL。這是工具的設計決定，不是參數問題。只要 PIPELINE pragma 存在，這個行為就無法避免。"
  ]),

  h2("4.4 v6：刪除 PIPELINE pragma（DSP 降至 247%）"),
  paraRuns([{ text: "修復：", bold: true }, "完整刪除所有 #pragma HLS PIPELINE 行。結果 DSP 從 1603% 降至 247%，LUT 從 839% 降至 113%，有突破性進展。"]),
  paraRuns([{ text: "剩餘問題：", bold: true }, "HLS 把各 loop 自動 outline 成子模組，但 BRAM 使用量升至 34%（超出但還可接受），DSP 仍 247%（仍超）。"]),

  h2("4.5 v7：移除 helper ARRAY_PARTITION（DSP 仍 121%）"),
  para("移除了所有 helper 函式的 ARRAY_PARTITION pragma 後，編譯 log 出現新的三條連鎖反應："),
  bullet("[HLS 214-376]：HLS 的 pipeline_loops 全域設定，發現 16-iteration 迴圈，自動加上 PIPELINE"),
  bullet("[HLS 214-291]：上面的自動 pipeline 觸發了 inner loop 的完全展開"),
  bullet("[HLS 214-449]：inner loop 展開後，weight 陣列的索引變成常數，自動 partition weight matrix"),
  paraRuns([
    { text: "問題本質：", bold: true },
    "移除手動 pragma 沒有用——工具會自動補回來。pipeline_loops 是一個全域的自動優化設定，它主動尋找可以 pipeline 的迴圈並自動加上 pragma，然後觸發一連串的級聯展開。"
  ]),

  h2("4.6 v8：#pragma HLS PIPELINE off（最終版，DSP 21%）"),
  paraRuns([
    { text: "解決方案：", bold: true },
    "#pragma HLS PIPELINE off 是一個明確的「否定」指令，優先級高於 pipeline_loops 全域設定，告訴工具不對這個迴圈施加任何 pipeline 優化。"
  ]),
  para("套用範圍：所有 linear 函式的 outer 和 inner loop、所有 layernorm、activation 函式、主體的所有 element-wise loop。"),

  makeTable(
    ["版本", "關鍵動作", "DSP %", "LUT %", "FF %", "BRAM %", "csim"],
    [
      ["v1", "初始（double tanh）", "2636%", "2073%", "441%", "—", "未測"],
      ["v2", "65點LUT tanh", "1603%", "839%", "441%", "—", "PASS"],
      ["v3~v5", "調整 UNROLL/II（均無效）", "1603%", "839%", "441%", "—", "—"],
      ["v6", "刪除 PIPELINE pragma", "247%", "113%", "86%", "34%", "PASS"],
      ["v7", "移除 ARRAY_PARTITION", "121%", "77%", "52%", "0%", "—"],
      ["v8 ✓", "#pragma HLS PIPELINE off", "21%", "47%", "14%", "2%", "PASS"]
    ],
    [700, 2800, 900, 900, 900, 1000, 1160]
  ),

  pageBreak()
];

// SECTION 5: 驗證結果
const sec5 = [
  h1("5. 驗證結果：怎麼確認這個設計是正確的？"),

  h2("5.1 驗證鏈說明"),
  para("FPGA 設計有三個驗證層次，由軟體到硬體逐步逼近："),
  makeTable(
    ["層次", "工具", "模擬的是什麼", "通過代表什麼"],
    [
      ["C Simulation (csim)", "Vitis HLS (clang)", "純 C++ 邏輯，無時序概念", "演算法邏輯正確"],
      ["RTL Co-Simulation (cosim)", "ModelSim/Xsim", "時脈精確的 RTL 閘級電路", "RTL 與 C 行為 bit-exact 一致"],
      ["Place & Route (P&R)", "Vivado Implement", "實際佈局在矽晶片上的電路", "資源在預算內、時序能在 100MHz 收斂"]
    ],
    [2300, 1800, 3000, 2260]
  ),
  spacer(),

  h2("5.2 測試向量設計"),
  para("測試向量由 Python（PyTorch float32）生成，作為「金標準」參考輸出："),
  bullet("200 個驗證集 sequence × 128 個 timestep = 25,600 個 recurrent step"),
  bullet("每步驟：x_t[16] + state_prev[16] → readout[2] + state_next[16]"),
  bullet("state 遞迴傳遞：testbench 維護一個狀態機，把上一步的 state_next 傳入下一步的 state_prev"),
  bullet("tolerance（PHC_ABS_TOL）= 1.0：即 max_abs_diff < 1.0 才算 PASS"),
  paraRuns([
    { text: "注意 NEW5 的守門：", bold: true },
    "所有腳本都有明確的 new5_used=False 守門條件，確保測試向量完全來自 validation set，不使用保留測試集（new5），符合機器學習的標準實驗規範。"
  ]),

  h2("5.3 C Simulation（csim）結果"),
  makeTable(
    ["指標", "數值", "說明"],
    [
      ["測試步數", "25,600", "200 seq × 128 timestep"],
      ["Mismatches", "0", "全部通過"],
      ["max_abs_diff", "0.00725275", "定點近似的最大誤差"],
      ["mean_abs_diff", "0.000372553", "平均誤差（遠小於最大值）"],
      ["Tolerance", "1.0", "允許的最大誤差上限"],
      ["工具", "clang-16 (Debug mode)", "由 Vitis HLS csim 流程調用"]
    ],
    [2500, 2500, 4360]
  ),
  spacer(),

  h2("5.4 RTL Co-Simulation（cosim）結果"),
  makeTable(
    ["指標", "數值", "說明"],
    [
      ["測試步數", "25,600", "與 csim 相同的測試向量"],
      ["Mismatches", "0", "全部通過"],
      ["max_abs_diff", "0.00725275", "與 csim 完全相同"],
      ["mean_abs_diff", "0.000372553", "與 csim 完全相同"],
      ["結論", "bit-exact 一致", "RTL 電路與 C model 在每個 bit 上完全一致，無額外捨入"]
    ],
    [2500, 2500, 4360]
  ),
  spacer(),
  paraRuns([
    { text: "為什麼 csim 和 cosim 的 max_abs_diff 完全相同？", bold: true }
  ]),
  para("因為定點數的精度完全由 ap_fixed<24,12> 的位元寬決定，不會因為 RTL 合成過程產生額外誤差。HLS 工具保證在合成過程中不改變定點數運算的語義——這是 HLS 的核心承諾之一，也是我們使用定點數而非浮點數的額外好處（浮點 RTL 合成有時會因為運算順序改變而引入微小誤差）。"),

  h2("5.5 Sign Agreement Rate（方向一致性）"),
  para("max_abs_diff 只說明數值誤差大小，但作為交易訊號，更重要的是：定點輸出的方向（正/負）是否與 float32 reference 一致？"),
  para("符號規則：負值 → −1（賣訊號方向）；零/正值 → +1（買訊號方向）"),
  makeTable(
    ["維度", "總步數", "同號", "異號", "Sign Agreement"],
    [
      ["readout[0]", "25,600", "25,595", "5", "99.9805%"],
      ["readout[1]", "25,600", "25,589", "11", "99.9570%"],
      ["兩維聯合", "25,600", "25,584", "16", "99.9375%"]
    ],
    [2000, 1500, 1500, 1500, 2860]
  ),
  spacer(),
  paraRuns([
    { text: "關鍵解讀：", bold: true },
    "全部 16 個符號不一致的步驟，其 |ref| 均小於 0.00054——代表 float32 模型本身在這些時步的輸出幾乎為零（模型無方向性意見）。如果下游交易策略設有 confidence threshold（只在 |readout| 夠大時才進場），這 16 個時步本身就會被自動過濾掉，完全不影響實際決策。"
  ]),

  pageBreak()
];

// SECTION 6: 硬體結果
const sec6 = [
  h1("6. 硬體結果：真正佈在晶片上的樣子"),

  h2("6.1 HLS Csynth 估算（排程層面，v8）"),
  para("Csynth 是 HLS 合成的第一步：把 C++ 程式碼轉成 RTL（暫時不考慮真實佈線延遲），估算資源和時序。這是一個「上界估算」，Vivado 後續的邏輯優化通常能大幅降低實際 LUT 數量。"),
  makeTable(
    ["時脈", "目標週期", "HLS 估算週期", "不確定性（裕量）"],
    [["ap_clk @ 100MHz", "10.00 ns", "7.147 ns", "±2.70 ns"]],
    [2340, 2340, 2340, 2340]
  ),
  spacer(),
  makeTable(
    ["資源", "HLS 估算", "可用量", "利用率"],
    [
      ["DSP48E1", "46", "220", "20.91%"],
      ["LUT（Slice）", "24,836", "53,200", "46.68%"],
      ["FF（Register）", "14,518", "106,400", "13.65%"],
      ["BRAM_18K", "6", "280", "2.14%"]
    ],
    [2340, 2340, 2340, 2340]
  ),
  spacer(),
  paraRuns([
    { text: "Latency：", bold: true },
    "min/max = 8,316 / 8,316 cycles = 83.16 µs @ 100MHz。因為沒有 pipeline（序列計算），每個時步的延遲是固定的。Interval = 8,317 cycles（下一個輸入可以在第 8,317 個週期進入）。"
  ]),

  h2("6.2 Vivado P&R 實際結果（最終數字）"),
  para("P&R 是把 RTL 翻譯成真實電路、佈局在晶片上、完成佈線、驗證時序的完整流程。這是設計的最終確認。"),

  h3("6.2.1 時序（Timing）"),
  makeTable(
    ["階段", "Clock", "最差路徑（Achieved）", "Slack（WNS）", "狀態"],
    [
      ["Post-Synthesis", "10 ns", "9.729 ns", "+0.271 ns", "PASS"],
      ["Post-Implementation（Routed）", "10 ns", "9.980 ns", "+0.020 ns", "PASS ✓"]
    ],
    [2800, 1400, 2400, 2000, 760]
  ),
  spacer(),
  paraRuns([
    { text: "如何讀懂 WNS：", bold: true },
    "WNS（Worst Negative Slack）代表「最緊繃的 timing path 距違規還剩多少時間」。WNS = +0.020 ns 代表在 100MHz 時脈下，最慢的那條訊號路徑需要 9.980 ns 才能傳到，距 10 ns 的時脈邊界還剩 0.020 ns。只要 WNS ≥ 0，設計就是 timing met。"
  ]),
  paraRuns([
    { text: "風險評估：", bold: true },
    "+0.020 ns 的 margin 屬於極薄。Vivado 使用 Slow Process Corner 進行 STA（最保守條件），這已是最壞情況的估算，但仍建議生產部署時降頻至 90MHz（WNS 估計 +1.13 ns，latency 增至 92.4 µs）。"
  ]),
  paraRuns([
    { text: "22,941 個 timing endpoints 全部通過，TNS = 0.000 ns（沒有任何違規路徑）。", bold: true }
  ]),

  h3("6.2.2 資源利用率（Utilization Routed）"),
  makeTable(
    ["資源", "csynth 估算", "P&R 實際", "可用量", "實際 %", "備注"],
    [
      ["Slice LUT", "24,836", "7,555", "53,200", "14.20%", "LUT合併優化 -70%"],
      ["LUT as Logic", "—", "6,891", "53,200", "12.95%", "組合邏輯"],
      ["LUT as Memory", "—", "664", "17,400", "3.82%", "分散式 RAM"],
      ["Slice FF", "14,518", "8,217", "106,400", "7.72%", "正反器"],
      ["DSP48E1", "46", "47", "220", "21.36%", "乘法器（含 LayerNorm sqrt）"],
      ["BRAM_18K", "6", "6", "280", "2.14%", "與估算相同"]
    ],
    [1800, 1800, 1800, 1400, 1400, 1160]
  ),
  spacer(),
  paraRuns([
    { text: "為什麼 LUT 從 24,836 降到 7,555（降幅 70%）？", bold: true }
  ]),
  para("HLS Csynth 只做排程（scheduling）：把 operation 分配到時脈週期，每個 operation 計為一份 LUT 資源。它不做邏輯優化。Vivado synthesis 則執行完整的邏輯合併（logic merging）、常數傳播（constant propagation）、死碼消除（dead code elimination）。序列執行的設計（PIPELINE off）讓大量邏輯可以被時間複用（time-multiplexed），不需要多份複製，Vivado 的優化空間因此特別大。這個 70% 的降幅是設計序列化執行的直接結果，完全符合預期。"),

  h2("6.3 功耗分析"),
  makeTable(
    ["項目", "數值", "說明"],
    [
      ["Total On-Chip Power", "0.357 W", "Dynamic + Static"],
      ["Dynamic Power", "0.251 W", "電路切換消耗"],
      ["Device Static Power", "0.107 W", "漏電流（不可避免）"],
      ["Junction Temperature", "29.1 °C", "25°C 環境溫度下"],
      ["Max Ambient Temp", "80.9 °C", "設計的熱安全上限"],
      ["Confidence Level", "Medium", "內部節點活動率使用預設值（0.5 toggle rate）"]
    ],
    [2500, 2500, 4360]
  ),
  spacer(),
  paraRuns([
    { text: "解讀：", bold: true },
    "0.357W 對 xc7z020 的 TDP（典型 1.5–2W）而言相當保守，Junction Temperature 僅 29.1°C，熱餘量充足。Confidence Level 為 Medium 是因為 Vivado 沒有 VCD/SAIF 活動檔案，使用預設 toggle rate 0.5 估算內部節點動態功耗；若要精確功耗數字，需提供波形仿真活動檔案。"
  ]),

  pageBreak()
];

// SECTION 7: FPGA vs CPU
const sec7 = [
  h1("7. 效能比較：FPGA vs CPU"),

  h2("7.1 CPU Baseline 測量方法"),
  makeTable(
    ["項目", "設定"],
    [
      ["平台", "Windows 11，Intel CPU"],
      ["模型", "PHC-lite（phc_stp_step_hls_seed2.pt），PyTorch float32"],
      ["Batch size", "1（每次一個 step）"],
      ["計時方式", "time.perf_counter()，warm-up 100次後計時 10,000 次"],
      ["模式", "torch.no_grad()（關閉梯度計算）"],
      ["NEW5", "不使用（new5_used=False）"]
    ],
    [2500, 6860]
  ),
  spacer(),

  h2("7.2 延遲比較"),
  makeTable(
    ["指標", "FPGA（v8）", "CPU（no_grad）", "說明"],
    [
      ["Latency（固定）", "83.16 µs", "—", "FPGA 完全確定性"],
      ["Latency（mean）", "83.16 µs", "117.5 µs", "FPGA 快 1.4×"],
      ["Latency（median）", "83.16 µs", "84.2 µs", "幾乎相同"],
      ["Latency（min）", "83.16 µs", "61.2 µs", "CPU min 較低（JIT cache hit）"],
      ["Latency（p99）", "83.16 µs", "778.8 µs", "FPGA 優勢 9.4×"],
      ["Jitter", "0 µs（完全確定）", "高（OS, GC, cache）", "FPGA 核心優勢"],
      ["Determinism", "完全確定", "不確定", "HFT 最重要的特性"]
    ],
    [1800, 1800, 1800, 3960]
  ),
  spacer(),

  h2("7.3 正確的比較視角"),
  paraRuns([
    { text: "FPGA 的優勢不在 mean latency，而在 tail latency 和確定性。", bold: true }
  ]),
  para("CPU mean = 117 µs，FPGA = 83 µs，看似 FPGA 快 1.4 倍。但這個數字反映的是平均情況。在 HFT 場景中，重要的是「最壞情況會多糟」——CPU 的 p99 = 778 µs，代表每 100 次執行就有一次超過 778 µs，那一次的延遲等同於錯失交易機會。FPGA 不論何時執行都是精確的 83.16 µs，無任何 jitter。"),
  paraRuns([
    { text: "CPU min = 61 µs 比 FPGA 還快怎麼解釋？", bold: true }
  ]),
  para("CPU 的最佳情況（61 µs）是 JIT 編譯快取命中、CPU 完全佔用、無其他程序搶佔的「完美情況」，在真實系統中幾乎不可能持續發生。FPGA 的 83 µs 是在任何情況下都能保證的數字。"),

  h2("7.4 誠實聲明（必須在報告中說明）"),
  para("以下聲明在任何討論中不可超越："),
  bullet("83 µs 不滿足 HFT L1 場景的 < 10 µs 需求。本專案展示的是 FPGA 推論可行性，以及對 HLS 資源/延遲 tradeoff 的系統性理解。"),
  bullet("CPU baseline 的 p99 = 778.8 µs 包含 Python runtime 和 OS scheduling overhead，是真實 Python 管線下的延遲分布，不是最優化的 C++ CPU 推論時間。"),
  bullet("xc7z020 不是生產 HFT 硬體，是學術展示板（PYNQ-Z2）。"),
  bullet("PHC-lite 是 PHCSSM 的 16-dim 簡化版，特定 seed2 checkpoint，不代表完整模型效能。"),
  bullet("本報告不聲稱 FPGA 改善了 PnL，亦無交易回測結果。"),

  pageBreak()
];

// SECTION 8: 改進路徑 & 結論
const sec8 = [
  h1("8. 改進路徑與結論"),

  h2("8.1 若要縮短 Latency 的改進路徑"),
  para("以下是在現有基礎上，縮短推論延遲的三條路徑（均為初步估算，未經 synthesis 驗證）："),
  makeTable(
    ["方法", "預期 Latency", "代價", "適用場景"],
    [
      ["INT16 量化（ap_fixed<16,8>）", "~25 µs", "精度降低，需重新驗證 sign agreement", "同板 xc7z020"],
      ["Hidden dim 縮至 8", "~8 µs", "模型表達能力降低", "同板 xc7z020"],
      ["換用 Alveo U50（5,952 DSP）", "< 5 µs", "成本大幅上升，需重新 P&R", "生產 HFT 環境"]
    ],
    [2800, 1800, 3000, 1760]
  ),
  spacer(),

  h2("8.2 結論"),
  para("本專案在 xc7z020 FPGA 上完成了 PHC-lite 神經網路推論核心的完整實作與驗證："),
  bullet("8 輪系統性 HLS pragma 優化，從資源超標 26 倍到全部在預算內"),
  bullet("確認了 HLS PIPELINE pragma 觸發 cascade unroll 的根本機制，以及 pipeline_loops 自動接管的三連鎖反應"),
  bullet("csim + cosim 25,600 步全部通過，max_abs_diff = 0.00725，sign agreement 99.9375%"),
  bullet("P&R timing met（WNS +0.020 ns），所有資源在預算內（DSP 21%，LUT 14%，FF 8%，BRAM 2%）"),
  bullet("功耗 0.357W，Junction Temp 29.1°C，熱餘量充足"),
  bullet("與 CPU baseline 相比，FPGA 實現 jitter-free 確定性延遲，p99 改善 9.4 倍"),

  pageBreak()
];

// ─── Build document ────────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: DGRAY },
        paragraph: { spacing: { before: 240, after: 80 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 180, after: 60 }, outlineLevel: 2 }
      }
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      }
    ]
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 1 } },
            children: [
              new TextRun({ text: "PHC-lite HLS FPGA 技術報告", size: 18, font: "Arial", color: DGRAY }),
              new TextRun({ text: "\t2026-06-15", size: 18, font: "Arial", color: DGRAY }),
            ],
            tabStops: [{ type: "right", position: 9360 }]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "第 ", size: 18, font: "Arial", color: DGRAY }),
              new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Arial", color: DGRAY }),
              new TextRun({ text: " 頁", size: 18, font: "Arial", color: DGRAY })
            ]
          })]
        })
      },
      children: [
        ...titlePage,
        new TableOfContents("目 錄", { hyperlink: true, headingStyleRange: "1-3" }),
        pageBreak(),
        ...sec1, ...sec2, ...sec3, ...sec4, ...sec5, ...sec6, ...sec7, ...sec8
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("PHC_lite_HLS_Report.docx", buffer);
  console.log("✓ PHC_lite_HLS_Report.docx created successfully");
});
