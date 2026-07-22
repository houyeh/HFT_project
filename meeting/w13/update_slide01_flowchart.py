import base64
from pathlib import Path

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
h2 {font-size:1.1rem;font-weight:600;color:#79c0ff;margin-bottom:12px}
.tag {display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600}
.tag-blue {background:#1f4e79;color:#79c0ff}
.tag-red {background:#5c1a1a;color:#ff7b7b}
.tag-green {background:#0d3d21;color:#56d364}
.tag-orange {background:#794a15;color:#e3b341}
.grid-2 {display:grid;grid-template-columns:1fr 1fr;gap:24px}
.card {background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px; display:flex; flex-direction:column;}
.img-wrap {text-align:center; flex-grow:1; display:flex; align-items:center; justify-content:center; background:#0d1117; border-radius:8px; padding:10px;}
.img-wrap img {max-width:100%; max-height:260px; border-radius:4px;}
table {width:100%;border-collapse:collapse;font-size:.85rem}
th {background:#21262d;color:#8b949e;padding:8px 10px;text-align:left;font-weight:600}
td {padding:8px 10px;border-bottom:1px solid #21262d}
tr:last-child td {border-bottom:none}
.hl {color:#f0c27f;font-weight:700}
.red {color:#ff7b7b} .green {color:#56d364}
.note {font-size:.78rem;color:#8b949e;margin-top:8px;padding:8px 12px;background:#161b22;border-left:3px solid #388bfd;border-radius:4px; line-height:1.5;}
.badge-row {display:flex;gap:10px;flex-wrap:wrap;margin-top:8px; margin-bottom:12px;}
.flowchart {display:flex; align-items:center; justify-content:center; gap:10px; background:#0d1117; padding:15px; border-radius:8px; border:1px solid #30363d;}
.flow-box {background:#21262d; border:1px solid #58a6ff; padding:10px 15px; border-radius:6px; font-weight:bold; color:#c9d1d9; text-align:center;}
.flow-arrow {color:#8b949e; font-weight:bold; font-size:1.2rem;}
</style>
"""

loss_img = b64(DERRIK / "03_model_classification" / "01_training_loss_plot_annotated.png")
cm_img   = b64(DERRIK / "03_model_classification" / "02_confusion_matrix_percent_test.png")

html_01 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 01</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 01 — 最終模型：從方向性預測到毒性撤單過濾器</h1>

  <!-- 在最上方加入模型 Flowchart -->
  <div class="card" style="padding:15px; margin-bottom:-4px;">
    <h2 style="margin-bottom:8px;">模型架構 (Temporal Convolutional Network, TCN)</h2>
    <div class="flowchart" style="font-size:0.85rem; padding:10px;">
      <div class="flow-box" style="border-color:#8b949e">Input<br><span style="font-size:0.75rem;font-weight:normal">8 Features<br>100 Ticks</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box">TCN Block 1<br><span style="font-size:0.75rem;font-weight:normal">Channels: 16</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box">TCN Block 2<br><span style="font-size:0.75rem;font-weight:normal">Channels: 16</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box">TCN Block 3<br><span style="font-size:0.75rem;font-weight:normal">Channels: 16</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#56d364">Softmax<br><span style="font-size:0.75rem;font-weight:normal">3 Classes</span></div>
    </div>
  </div>

  <!-- 2x2 Grid Layout -->
  <div class="grid-2" style="margin-bottom:-4px;">
    <!-- Top Left: Model Summary -->
    <div class="card" style="padding:20px;">
      <h2 style="margin-bottom:8px;">模型與訓練摘要</h2>
      <table>
        <tr><th style="width:25%;">項目</th><th>值</th></tr>
        <tr><td>模型類型</td><td>TCN (Temporal Convolutional Network)</td></tr>
        <tr><td>輸入特徵</td><td><span class="hl">8</span> 個微觀結構特徵 / <span class="hl">100</span> ticks</td></tr>
        <tr><td>目標標籤</td><td><span class="tag tag-blue">smoothed_mid_move_3class</span></td></tr>
        <tr><td>資料來源</td><td>NEW_ALL (商品: TXFB6)</td></tr>
        <tr><td>訓練資料</td><td>718,562 筆（6 sessions）</td></tr>
        <tr><td>驗證資料</td><td>112,698 筆（1 session）</td></tr>
        <tr><td>測試資料</td><td>233,268 筆（2 sessions）</td></tr>
        <tr><td>最佳模型</td><td><span class="hl">Epoch 30</span> (Val Loss: 0.7814)</td></tr>
      </table>
      <p class="note" style="margin-top:10px; font-family:monospace; font-size:0.7rem;">
        快照版本: expC_snapshot (zscore_window=2000)
      </p>
    </div>

    <!-- Top Right: Strategy Positioning & Classification -->
    <div class="card" style="padding:20px;">
      <h2 style="margin-bottom:8px;">策略定位與分類報告 (Test Split)</h2>
      <table style="margin-bottom:6px;">
        <tr><th>輸出訊號</th><th>動作</th><th>用途</th></tr>
        <tr><td>P(Down) 高</td><td>撤銷買單 (Cancel Bid)</td><td class="green">✔ 防禦毒性下跌</td></tr>
        <tr><td>P(Up) 高</td><td>撤銷賣單 (Cancel Ask)</td><td class="green">✔ 防禦毒性上漲</td></tr>
      </table>
      
      <div class="badge-row" style="margin-top:4px; margin-bottom:8px;">
        <span class="tag tag-red">❌ 非主動交易</span>
        <span class="tag tag-red">❌ 非 Production-ready</span>
        <span class="tag tag-green">✔ 防禦撤單過濾器</span>
      </div>

      <table style="margin-top:auto;">
        <tr><th>類別</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
        <tr><td>Down</td><td class="red">0.028</td><td class="green">0.694</td><td>0.054</td><td>2,762</td></tr>
        <tr><td>Stationary</td><td class="green">0.987</td><td>0.452</td><td>0.620</td><td>227,711</td></tr>
        <tr><td>Up</td><td class="red">0.029</td><td class="green">0.642</td><td>0.056</td><td>2,795</td></tr>
        <tr style="background:#1f242b;"><td><strong>Macro Avg</strong></td><td>0.348</td><td>0.596</td><td>0.243</td><td>233,268</td></tr>
        <tr style="background:#1f242b;"><td><strong>Weighted Avg</strong></td><td>0.964</td><td>0.457</td><td>0.606</td><td>233,268</td></tr>
      </table>
    </div>
  </div>

  <div class="grid-2" style="flex-grow:1;">
    <!-- Bottom Left: Loss Curve -->
    <div class="card" style="padding:15px;">
      <h2 style="margin-bottom:8px;">訓練收斂曲線 (Loss Curve)</h2>
      <div class="img-wrap" style="padding:5px;">
        <img src="{loss_img}" alt="Loss Curve">
      </div>
    </div>

    <!-- Bottom Right: Confusion Matrix -->
    <div class="card" style="padding:15px;">
      <h2 style="margin-bottom:8px;">混淆矩陣 (Confusion Matrix) — 測試集</h2>
      <div class="img-wrap" style="padding:5px;">
        <img src="{cm_img}" alt="Confusion Matrix">
      </div>
    </div>
  </div>

</div></body></html>
"""
(OUT / "01_slides.html").write_text(html_01, encoding="utf-8")
print("Slide 01 updated with flowchart on top.")
