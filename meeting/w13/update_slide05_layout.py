import pandas as pd
from pathlib import Path
import base64

ROOT = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project\meeting\w13")
DERRIK = ROOT / "___Derrik_results"
OUT = ROOT / "slides"

def b64(p):
    if not p.exists(): return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

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
.grid-3 {display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}
.card {background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px; display:flex; flex-direction:column;}
.img-wrap {text-align:center; flex-grow:1; display:flex; align-items:center; justify-content:center; background:#0d1117; border-radius:8px; padding:10px;}
.img-wrap img {max-width:100%; max-height:260px; border-radius:4px;}
table {width:100%;border-collapse:collapse;font-size:.85rem}
th {background:#21262d;color:#8b949e;padding:8px 10px;text-align:left;font-weight:600}
td {padding:8px 10px;border-bottom:1px solid #21262d}
tr:last-child td {border-bottom:none}
.hl {color:#f0c27f;font-weight:700}
.red {color:#ff7b7b} .green {color:#56d364}
.note {font-size:.78rem;color:#8b949e;margin-top:8px;padding:8px 12px;background:#161b22;border-left:3px solid #388bfd;border-radius:4px; line-height:1.5;}
.badge-row {display:flex;gap:10px;flex-wrap:wrap;margin-top:6px; margin-bottom:10px;}
ul {margin-left:20px;line-height:1.6}
li {margin-bottom:8px}
.flowchart {display:flex; align-items:center; justify-content:center; gap:10px; background:#0d1117; padding:15px; border-radius:8px; border:1px solid #30363d;}
.flow-box {background:#21262d; border:1px solid #58a6ff; padding:10px 15px; border-radius:6px; font-weight:bold; color:#c9d1d9; text-align:center;}
.flow-arrow {color:#8b949e; font-weight:bold; font-size:1.2rem;}
.caveat-box {margin-top:12px; padding:10px; border-radius:6px; background:#2d1a1a; border:1px solid #5c1a1a;}
.caveat-box h3 {color:#ff7b7b; font-size:0.9rem; margin-bottom:4px;}
.formula-box {background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px; font-family:monospace; font-size:0.75rem;}
</style>
"""

df_09 = pd.read_csv(DERRIK / "09_fill_conditional_labels" / "fill_conditional_label_summary.csv")
tr_09 = ""
for _, r in df_09.iterrows():
    tr_09 += f"<tr><td>{r['dataset']}</td><td>{r['bid_toxic_rate_conditional']*100:.2f}%</td><td class='hl green'>{r['bid_toxic_rate_lift']:.2f}x</td><td>{r['ask_toxic_rate_conditional']*100:.2f}%</td><td class='hl green'>{r['ask_toxic_rate_lift']:.2f}x</td></tr>"

lat_img = b64(DERRIK / "10_latency_sensitivity" / "cancel_delay_net_pnl.png")

html_05 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 05</title>{CSS}</head><body>
<div class="slide" style="overflow-y:auto">
  <h1 style="font-size:1.6rem">Slide 05 — Cancel Delay Increases Risk, But PnL Is Non-monotonic (撤單延遲增加風險，但 PnL 呈非單調變化)</h1>

  <div class="grid-2">
    <!-- Left Card: Fill-Conditional Findings -->
    <div class="card" style="padding:18px;">
      <h2>發現一：條件成交 (Fill-Conditional) 的毒性飆升</h2>
      <p style="font-size:0.85rem; margin-bottom:12px; line-height:1.5;">
        不是直接看所有事件，而是先用 <strong>Proxy C : price level removed</strong> 判斷 quote 是否「可能被成交／價位被移除」，再計算後續 markout 是否 toxic。
      </p>
      
      <table style="font-size:0.75rem; margin-bottom:12px;">
        <tr><th>資料集</th><th>Bid 條件毒性</th><th>Bid 提升</th><th>Ask 條件毒性</th><th>Ask 提升</th></tr>
        {tr_09}
      </table>
      
      <div class="grid-2" style="gap:12px; margin-bottom:12px;">
        <div class="formula-box">
          <strong style="color:#e3b341;">Bid toxic</strong><br>
          若 bid 價位被移除，且<br>
          (future_mid - bid_px) / tick < -0.5<br>
          <span style="color:#8b949e">➔ 接到下跌刀</span>
        </div>
        <div class="formula-box">
          <strong style="color:#e3b341;">Ask toxic</strong><br>
          若 ask 價位被移除，且<br>
          (ask_px - future_mid) / tick < -0.5<br>
          <span style="color:#8b949e">➔ 賣在上漲前</span>
        </div>
      </div>
      
      <p style="font-size:0.75rem; margin-bottom:12px; line-height:1.5;">
        <span style="color:#56d364">Lift</span> = 條件毒性率 / 無條件毒性率。例如 3.25% / 8.55 ≈ 0.38%，代表 toxic risk 主要集中在 quote likely-filled events。
      </p>
      
      <div class="badge-row" style="margin-top:0; margin-bottom:12px;">
        <span class="tag tag-blue">指引：值得實作 side-specific quote-risk label</span>
        <span class="tag tag-orange">不是保證會贏 baseline</span>
      </div>
      
      <div style="background:#21262d; border-left:3px solid #e3b341; padding:8px 12px; border-radius:4px; font-size:0.75rem; color:#8b949e;">
        <strong style="color:#e3b341">⚠️</strong> 僅能支持「值得驗證 bid_toxic / ask_toxic controlled ablation」；基於 Proxy C，非交易所真實成交 Absolute Truth。
      </div>
    </div>
    
    <!-- Right Card: Latency Sensitivity -->
    <div class="card" style="padding:15px;">
      <h2>發現二：撤單延遲 (Cancel Delay) 帶來的非單調風險</h2>
      <div class="img-wrap" style="padding:10px;"><img src="{lat_img}" alt="Latency vs PnL" style="max-height:180px;"></div>
      <p class="note" style="margin-top:12px; font-size:0.8rem; padding:12px;">
        延遲增加的初期，PnL 呈現非單調性 (Non-monotonic) 變化，甚至可能因為過濾假陽性而稍微回升。但當延遲過高，防禦力開始顯著惡化。<br><br>
        <span class="hl">FPGA 是用來降低最差情況下推論與撤單路徑延遲的候選方案；此處尚未證明端到端 (End-to-end) 的交易所撤單延遲效益。</span>
      </p>
    </div>
  </div>
</div></body></html>
"""
(OUT / "05_slides.html").write_text(html_05, encoding="utf-8")

print("Slide 05 fully updated to match requested layout.")
