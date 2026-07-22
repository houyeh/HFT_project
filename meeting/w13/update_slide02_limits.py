import pandas as pd
from pathlib import Path

ROOT = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project\meeting\w13")
DERRIK = ROOT / "___Derrik_results"
OUT = ROOT / "slides"

CSS = """
<style>
* {box-sizing:border-box;margin:0;padding:0}
body {font-family:"Noto Sans TC",sans-serif;background:#0d1117;color:#e6edf3}
.slide {width:1280px;min-height:720px;margin:30px auto;background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);border:1px solid #30363d;border-radius:16px;padding:48px 56px;display:flex;flex-direction:column;gap:20px; overflow-y:auto;}
h1 {font-size:2rem;font-weight:700;color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:14px; flex-shrink:0;}
h2 {font-size:1.1rem;font-weight:600;color:#79c0ff;margin-bottom:10px}
.tag {display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600}
.tag-blue {background:#1f4e79;color:#79c0ff}
.tag-red {background:#5c1a1a;color:#ff7b7b}
.tag-green {background:#0d3d21;color:#56d364}
.tag-orange {background:#794a15;color:#e3b341}
.grid-2 {display:grid;grid-template-columns:1fr 1fr;gap:24px}
.card {background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}
table {width:100%;border-collapse:collapse;font-size:.85rem}
th {background:#21262d;color:#8b949e;padding:8px 10px;text-align:left;font-weight:600}
td {padding:8px 10px;border-bottom:1px solid #21262d}
tr:last-child td {border-bottom:none}
.hl {color:#f0c27f;font-weight:700}
.red {color:#ff7b7b} .green {color:#56d364}
.note {font-size:.78rem;color:#8b949e;margin-top:8px;padding:8px 12px;background:#161b22;border-left:3px solid #388bfd;border-radius:4px}
.badge-row {display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}
ul {margin-left:20px;line-height:1.6}
li {margin-bottom:8px}
.flowchart {display:flex; align-items:center; justify-content:center; gap:10px; background:#0d1117; padding:15px; border-radius:8px; border:1px solid #30363d;}
.flow-box {background:#21262d; border:1px solid #58a6ff; padding:10px 15px; border-radius:6px; font-weight:bold; color:#c9d1d9; text-align:center;}
.flow-arrow {color:#8b949e; font-weight:bold; font-size:1.2rem;}
</style>
"""

df_04 = pd.read_csv(DERRIK / "04_backtest_baseline_thresholds" / "baseline_evidence.csv")
tr_04 = ""
for _, r in df_04.iterrows():
    pnl = f"{r['net_pnl']:,.0f}"
    rm = f"{r['random_mean']:,.0f}"
    rp = f"{r['random_mean_plus_1std']:,.0f}"
    beats = "<span class='tag tag-green'>✔ 擊敗</span>" if r['beats_random_mean_plus_1std'] else "<span class='tag tag-red'>❌ 未擊敗</span>"
    interp = r['interpretation']
    tr_04 += f"<tr><td>{r['experiment']}</td><td class='hl'>{pnl}</td><td>{rm}</td><td>{rp}</td><td>{beats}</td><td>{interp}</td></tr>"

html_02 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 02</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 02 — 回測策略與 Threshold 驗證結果</h1>
  
  <div class="card" style="margin-bottom:0px; padding:15px 20px;">
    <h2>回測策略流程 (Maker-only Cancel Filter)</h2>
    <div class="flowchart" style="margin-bottom:5px;">
      <div class="flow-box">1. 維持正常掛單<br><span style="font-size:0.75rem;font-weight:normal">做市商在最佳 Bid/Ask 持續掛單<br>提供流動性以賺取價差</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#e3b341">2. TCN 即時預測<br><span style="font-size:0.75rem;font-weight:normal">每個 tick 更新 P(Down), P(Up)</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#ff7b7b">3. 檢查閾值 (Threshold)<br><span style="font-size:0.75rem;font-weight:normal">P(Down/Up) > Threshold?</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#56d364">4. 執行防禦<br><span style="font-size:0.75rem;font-weight:normal">瞬間撤銷被狙擊側的掛單<br>避免被毒性流穿透</span></div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card">
      <h2>各閾值 (Thresholds) 的設定涵義</h2>
      <ul style="font-size:0.85rem;">
        <li><span class="hl">0.55 / 0.60 (較敏感)</span>：只要模型稍微覺得危險就撤單。撤單次數多，可能過濾掉一些安全利潤，但能最大程度擋下毒性單。</li>
        <li><span class="hl">0.70 (較保守)</span>：模型非常確定才撤單。撤單次數少，保留較多做市機會，但可能漏掉一些輕微毒性單。</li>
      </ul>
    </div>
    <div class="card">
      <h2>Validation Sweep：如何選出最佳參數？</h2>
      <p style="font-size:0.85rem; line-height:1.6;">我們在 <strong>Validation Set</strong> 執行參數掃描，讓閾值從 0.50 測試到 0.90。<br>結果發現 <strong>0.70</strong> 給出了理論最佳的淨 PnL。<br>然而套用到 <strong>Test Set (樣本外)</strong> 時，卻發現 0.55/0.60 表現更好。這顯示了最佳閾值具有強烈的「狀態依賴性」。</p>
    </div>
  </div>

  <div class="card" style="padding:15px 20px;">
    <h2>NEW 測試集 (Test Split) — PnL 實測證據</h2>
    <table>
      <tr><th>實驗設定</th><th>模型 Net PnL</th><th>隨機撤單平均值</th><th>隨機平均 + 1 Std</th><th>擊敗 Random+1Std</th><th>解讀</th></tr>
      {tr_04}
    </table>
    
    <!-- 新增：回測環境假設與學術限制聲明 -->
    <div style="margin-top:15px; padding:12px 16px; border-radius:6px; background:#1f242b; border:1px solid #30363d;">
      <h3 style="color:#e3b341; font-size:0.95rem; margin-bottom:8px;">⚠️ 回測環境假設與學術限制聲明 (Simulation Assumptions)</h3>
      <ul style="font-size:0.8rem; color:#c9d1d9; margin-left:15px; margin-bottom:0px;">
        <li><strong style="color:#56d364;">有考慮手續費 (Fees Included)：</strong>所有的 Net PnL 已經嚴格扣除了每口 <strong>15 TWD</strong> 的 Maker 手續費。</li>
        <li><strong style="color:#ff7b7b;">無考慮排隊順序 (Not Queue-aware)：</strong>目前採用寬鬆的觸價成交 (Touch-fill proxy) 假設。若未來價格碰到我們掛單的價位即視為成交，並未計算我們在 Limit Order Book (LOB) 中的真實排隊位置。</li>
        <li><strong style="color:#ff7b7b;">零延遲撤單假設 (Zero-Latency Cancel)：</strong>本表格數據建立在理想的零延遲假設上。這也是我們在 Slide 05 分析 Latency 敏感度，以及未來必須推動 FPGA 硬體落地的根本原因。</li>
      </ul>
    </div>

  </div>
</div></body></html>
"""
(OUT / "02_slides.html").write_text(html_02, encoding="utf-8")
print("Slide 02 updated with simulation limits.")
