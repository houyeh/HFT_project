import pandas as pd
from pathlib import Path

ROOT = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project\meeting\w13")
DERRIK = ROOT / "___Derrik_results"
OUT = ROOT / "slides"

CSS = """
<style>
* {box-sizing:border-box;margin:0;padding:0}
body {font-family:"Noto Sans TC",sans-serif;background:#0d1117;color:#e6edf3}
.slide {width:1280px;min-height:720px;margin:30px auto;background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);border:1px solid #30363d;border-radius:16px;padding:48px 56px;display:flex;flex-direction:column;gap:28px}
h1 {font-size:2rem;font-weight:700;color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:14px}
h2 {font-size:1.1rem;font-weight:600;color:#79c0ff;margin-bottom:10px}
.tag {display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600}
.tag-blue {background:#1f4e79;color:#79c0ff}
.tag-red {background:#5c1a1a;color:#ff7b7b}
.tag-green {background:#0d3d21;color:#56d364}
.grid-2 {display:grid;grid-template-columns:1fr 1fr;gap:24px}
.card {background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}
.img-wrap {text-align:center}
.img-wrap img {max-width:100%;border-radius:8px;border:1px solid #30363d}
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
</style>
"""

df_07 = pd.read_csv(DERRIK / "07_regime_gate_sweep" / "regime_gate_best_by_dataset.csv")
tr_07 = ""
for _, r in df_07.iterrows():
    pnl = f"{r['net_pnl']:,.0f}"
    base = f"{r['baseline_net_pnl']:,.0f}"
    imp = f"{r['improvement']:,.0f}"
    act = f"{r['active_event_rate']*100:.1f}%"
    color = "green" if r['improvement'] > 0 else "red"
    tr_07 += f"<tr><td>{r['dataset']}</td><td>{r['best_gate_type']}</td><td>{r['best_gate_params']}</td><td>{act}</td><td>{base}</td><td class='hl'>{pnl}</td><td class='{color}'>+{imp}</td></tr>"

html_04 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 04</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 04 — Regime Gate 的加入與成效</h1>
  
  <div class="grid-2" style="margin-bottom: -4px;">
    <div class="card">
      <h2 style="color:#56d364">Rolling Trend (滾動趨勢濾網)</h2>
      <p style="margin-bottom:8px; line-height:1.5"><strong>原理：</strong>偵測市場是否處於強烈的單向走勢中。當市場出現極端趨勢時，逆勢做市（Maker）容易被單向毒性單連續穿透，風險極高，必須暫停或採取防禦。</p>
      <p style="line-height:1.5"><strong>實際作法：</strong>計算過去一段時間內（例如 5 分鐘）的絕對價格位移 (Absolute Price Move)。當 <code>Trend Strength > trend_ticks</code> (例如超過 20 點) 時，我們視為進入強趨勢 Regime，立刻啟動防禦機制 (暫停掛單)。</p>
    </div>
    <div class="card">
      <h2 style="color:#56d364">Spread Filter (價差濾網)</h2>
      <p style="margin-bottom:8px; line-height:1.5"><strong>原理：</strong>透過最佳買賣報價的距離 (Spread) 來衡量市場即時流動性。當 Spread 擴大時，通常代表流動性枯竭、造市商撤退，且即將發生劇烈波動。</p>
      <p style="line-height:1.5"><strong>實際作法：</strong>即時監控當下的 Bid-Ask Spread。當發現 <code>Spread > spread_ticks_min</code> (例如超過 2.0 點，而平常台指期都是 1 點) 時，我們視為流動性枯竭的危險狀態，立刻啟動防禦機制。</p>
    </div>
  </div>
  
  <div class="card">
    <h2>加入巨觀濾網後的防禦力提升 (Regime Dependency)</h2>
    <p style="margin-bottom:12px">加入上述 Regime Gate 之後的績效表現對比：</p>
    <table>
      <tr><th>資料集</th><th>最佳 Gate 類型</th><th>參數設定</th><th>模型活躍率</th><th>基準 PnL (無 Gate)</th><th>Gate+模型 PnL</th><th>提升 (Improvement)</th></tr>
      {tr_07}
    </table>
    <p class="note" style="margin-top:16px">
      <strong>關鍵成效：</strong><br>
      1. <strong>大幅止損 NEW3：</strong> 透過 Combined Filter (趨勢+波動率+價差)，NEW3 的虧損從 -55萬 大幅拉回到接近平盤，提升了 <strong>26 萬</strong>。<br>
      2. <strong>並非停止交易：</strong> 模型活躍率 (Active Event Rate) 仍維持在 <strong>82% 左右</strong>，證明不是因為完全不交易而避開虧損，而是精準避開了毒性時段。<br>
      3. <strong>狀態依賴性 (Regime Dependency)：</strong> NEW 和 NEW3 的最佳 Gate 參數完全不同，這成為我們下一步開發 Auto-Gating 最直接的鐵證。
    </p>
  </div>
</div></body></html>
"""
(OUT / "04_slides.html").write_text(html_04, encoding="utf-8")
print("Updated Slide 04 successfully.")
