import pandas as pd
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
</style>
"""

# ======================= SLIDE 01 =======================
loss_img = b64(DERRIK / "03_model_classification" / "01_training_loss_plot_annotated.png")
cm_img   = b64(DERRIK / "03_model_classification" / "02_confusion_matrix_percent_test.png")

html_01 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 01</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 01 — 最終模型：從方向性預測到毒性撤單過濾器</h1>

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

  <div class="grid-2" style="margin-bottom:-4px;">
    <div class="card" style="padding:20px;">
      <h2 style="margin-bottom:8px;">模型與訓練摘要</h2>
      <table>
        <tr><th style="width:25%;">項目</th><th>值</th></tr>
        <tr><td>模型類型</td><td>TCN</td></tr>
        <tr><td>輸入特徵</td><td><span class="hl">8</span> 微觀結構特徵 / <span class="hl">100</span> ticks</td></tr>
        <tr><td>目標標籤</td><td><span class="tag tag-blue">smoothed_mid_move_3class</span></td></tr>
        <tr><td>資料來源</td><td>NEW_ALL (商品: TXFB6)</td></tr>
        <tr><td>訓練/驗證/測試</td><td>718k (6天) / 112k (1天) / 233k (2天)</td></tr>
        <tr><td>最佳模型</td><td><span class="hl">Epoch 30</span> (Val Loss: 0.7814)</td></tr>
      </table>
      <div class="caveat-box">
        <h3 style="color:#ff7b7b;font-size:0.8rem;">⚠️ 學術限制 (Caveats)</h3>
        <ul style="font-size:0.75rem; color:#c9d1d9; margin-left:15px; margin-bottom:0;">
          <li>目前的回測並未完全具備排隊感知 (Queue-aware)；僅用於一階訊號比較。</li>
          <li>0.55/0.60 僅為測試集掃描候選值；0.70 才是驗證集選出的。這不代表 0.55 為可上線的標準。</li>
        </ul>
      </div>
    </div>

    <div class="card" style="padding:20px;">
      <h2 style="margin-bottom:8px;">策略定位與分類報告 (Test Split)</h2>
      <table style="margin-bottom:4px;">
        <tr><th>輸出訊號</th><th>動作</th><th>用途</th></tr>
        <tr><td>P(Down) 高</td><td>撤銷買單 (Cancel Bid)</td><td class="green">✔ 防禦毒性下跌</td></tr>
      </table>
      <table style="margin-top:auto;">
        <tr><th>類別</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
        <tr><td>Down</td><td class="red">0.028</td><td class="green">0.694</td><td>0.054</td><td>2,762</td></tr>
        <tr><td>Stationary</td><td class="green">0.987</td><td>0.452</td><td>0.620</td><td>227,711</td></tr>
        <tr><td>Up</td><td class="red">0.029</td><td class="green">0.642</td><td>0.056</td><td>2,795</td></tr>
        <tr style="background:#1f242b;"><td><strong>Macro Avg</strong></td><td>0.348</td><td>0.596</td><td>0.243</td><td>233,268</td></tr>
      </table>
      <p class="note" style="margin-top:8px; font-size:0.75rem;">
        低精確率代表這不適合做為主動方向性交易；高召回率則支持將其作為高敏感度的撤單風險警示訊號。
      </p>
    </div>
  </div>

  <div class="grid-2" style="flex-grow:1;">
    <div class="card" style="padding:15px;">
      <h2 style="margin-bottom:8px;">訓練收斂曲線 (Loss Curve)</h2>
      <div class="img-wrap" style="padding:5px;"><img src="{loss_img}" alt="Loss Curve"></div>
    </div>
    <div class="card" style="padding:15px;">
      <h2 style="margin-bottom:8px;">混淆矩陣 (Confusion Matrix) — 測試集</h2>
      <div class="img-wrap" style="padding:5px;"><img src="{cm_img}" alt="Confusion Matrix"></div>
    </div>
  </div>
</div></body></html>
"""
(OUT / "01_slides.html").write_text(html_01, encoding="utf-8")


# ======================= SLIDE 02 =======================
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
      <div class="flow-box">1. 維持掛單<br><span style="font-size:0.75rem;font-weight:normal">做市商維持雙邊掛單</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#e3b341">2. 即時預測<br><span style="font-size:0.75rem;font-weight:normal">每 tick 更新 P(Down/Up)</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#ff7b7b">3. 檢查閾值<br><span style="font-size:0.75rem;font-weight:normal">P > Threshold?</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#56d364">4. 執行防禦<br><span style="font-size:0.75rem;font-weight:normal">撤銷被狙擊側的掛單</span></div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card">
      <h2>各閾值 (Thresholds) 的設定涵義</h2>
      <ul style="font-size:0.85rem;">
        <li><span class="hl">0.55 / 0.60 (較敏感)</span>：提高毒性流警示 (toxic-flow warning) 的敏感度，但也增加假陽性撤單 (false-positive cancel)，可能過濾掉安全利潤。</li>
        <li><span class="hl">0.70 (較保守)</span>：確認極度危險才撤單，保留較多做市機會，但防禦不足。</li>
      </ul>
    </div>
    <div class="card">
      <h2>Validation Sweep：最佳參數迷思</h2>
      <p style="font-size:0.85rem; line-height:1.6;">在 <strong>Validation Set</strong> 從 0.50 掃描到 0.90，理論最佳為 <strong>0.70</strong>。<br>但套用於 Test Set 時 0.55/0.60 反而表現較好。<br><br><span style="color:#ff7b7b;font-weight:bold;">重點提醒：</span>0.55 絕對不是可上線的閾值，它僅是測試集掃描的候選值。最佳閾值具有強烈的「狀態依賴性」。</p>
    </div>
  </div>

  <div class="card" style="padding:15px 20px;">
    <h2>NEW 測試集 (Test Split) — PnL 實測證據</h2>
    <table>
      <tr><th>實驗設定</th><th>模型 Net PnL</th><th>隨機撤單平均值</th><th>隨機平均 + 1 Std</th><th>擊敗 Random+1Std</th></tr>
      {tr_04}
    </table>
    
    <div class="caveat-box">
      <h3>⚠️ 學術限制聲明 (Simulation Assumptions)</h3>
      <ul style="font-size:0.8rem; color:#c9d1d9; margin-left:15px; margin-bottom:0px;">
        <li><strong style="color:#ff7b7b;">非真實排隊 (Not Queue-aware)：</strong>目前的純造市 (Maker-only) 回測並未完全具備排隊感知；PnL 僅供訊號比較，而非實際上線的獲利預估。</li>
        <li><strong style="color:#56d364;">有考慮手續費：</strong>所有的 Net PnL 已經嚴格扣除了每口 15 TWD 的 Maker 手續費。</li>
      </ul>
    </div>
  </div>
</div></body></html>
"""
(OUT / "02_slides.html").write_text(html_02, encoding="utf-8")


# ======================= SLIDE 03 =======================
df_06 = pd.read_csv(DERRIK / "06_regime_stress_test" / "new3_regime_summary.csv")
tr_06 = ""
for _, r in df_06.iterrows():
    pnl = f"{r['baseline_net_pnl']:,.0f}"
    color = "red" if r['baseline_net_pnl'] < 0 else "green"
    tr_06 += f"<tr><td>{r['session_id']}</td><td>{r['trend_strength']:.2f}</td><td>{r['suitability_label']}</td><td class='{color}'>{pnl}</td></tr>"

html_03 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 03</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 03 — 為什麼需要狀態濾網 (Why Regime Gate)</h1>
  
  <div class="card" style="margin-bottom:0px;">
    <h2>如何得知 NEW3 處於強烈趨勢環境？</h2>
    <div class="grid-2">
      <div>
        <p style="font-size:0.9rem; line-height:1.6;"><strong>Trend Strength 評估方式：</strong><br>
        Session 層級的趨勢評估，使用的是相對於已實現波動率 (Realized Volatility) 的累積中價位移 (Cumulative Mid-price Displacement)。</p>
        <p style="font-size:0.9rem; line-height:1.6; margin-top:10px;">這顯示 NEW3 具備高強度的持續性單邊推升或下跌，不適合傳統靜態做市。</p>
      </div>
      <div class="caveat-box" style="margin-top:0;">
        <h3>⚠️ TCN 的微觀盲點</h3>
        <p style="color:#c9d1d9; font-size:0.85rem; line-height:1.5;">
          微觀的撤單濾網無法處理宏觀的方向性風險與庫存狀態風險 (Inventory regime risk)。<br>
          在強趨勢中，100 ticks 的視窗不足以判斷宏觀方向，導致錯誤維持掛單而遭受連續單向虧損。
        </p>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>壓力測試：模型在 NEW3 的真實表現 (OOS)</h2>
    <table>
      <tr><th>Session ID</th><th>趨勢強度 (Trend)</th><th>適合度標籤</th><th>無 Gate 基準 PnL (Baseline)</th></tr>
      {tr_06}
      <tr style="background:#21262d"><td colspan="3"><strong>NEW3 總虧損對比</strong></td><td class="red hl" style="font-size:1rem">Baseline: -557,165 | Model: -551,705</td></tr>
    </table>
    <p class="note" style="margin-top:16px">
      <strong>核心數據：</strong> 在 NEW3 中，Baseline 虧損 -557,165，Model (0.70) 虧損 -551,705，**改善幅度僅 +5,460**。<br>
      這明確證明單憑 TCN 模型無法對抗強勢的 Macro Regime。
    </p>
  </div>
</div></body></html>
"""
(OUT / "03_slides.html").write_text(html_03, encoding="utf-8")


# ======================= SLIDE 04 =======================
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
  
  <div class="grid-3" style="margin-bottom:0px;">
    <div class="card" style="padding:15px;">
      <h2 style="color:#56d364; font-size:1rem;">Rolling Trend</h2>
      <p style="font-size:0.8rem;">偵測極端單向走勢。<code>Trend > limit</code> 啟動防禦。</p>
    </div>
    <div class="card" style="padding:15px;">
      <h2 style="color:#56d364; font-size:1rem;">Volatility Filter</h2>
      <p style="font-size:0.8rem;">計算視窗內標準差，<code>Vol > limit</code> 暫停雙向掛單。</p>
    </div>
    <div class="card" style="padding:15px;">
      <h2 style="color:#56d364; font-size:1rem;">Spread Filter</h2>
      <p style="font-size:0.8rem;">衡量流動性枯竭。<code>Spread > limit</code> 啟動防禦。</p>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card" style="border-color:#e3b341;">
      <h2 style="color:#e3b341">Combined Filter 綜合機制</h2>
      <p style="font-size:0.85rem; line-height:1.5;">綜合濾網 (Combined Filter) 結合了趨勢風險規避、波動率風險規避，以及價差/流動性條件。任一異常即停機。</p>
      <div class="caveat-box" style="margin-top:8px;">
        <h3>⚠️ 狀態校準現狀</h3>
        <p style="font-size:0.8rem; color:#ff7b7b">017 動態閾值 (Auto-gating) 實驗失敗，因此狀態校準仍是未解問題。<br>此處展示的僅是靜態基於規則的濾網對比，而非可用於上線的自動狀態選擇器。</p>
      </div>
    </div>
    <div class="card">
      <h2>對照組 <code>_no_gate_reference</code> 意義</h2>
      <p style="font-size:0.85rem; line-height:1.5;">台指期 Spread 大多時間 ≥ 1，我們設定 <code>spread_ticks_min: 1.0</code> 作為基準對照組 (控制變因，代表幾乎永遠不防禦)。<br>
      只有套用更嚴格的參數 (如 NEW3 的綜合濾網) 進行對比，才能實證「防禦濾網能有效減損」。</p>
    </div>
  </div>
  
  <div class="card">
    <h2>防禦力提升證據 (Rule-based Static Check)</h2>
    <table>
      <tr><th>資料集</th><th>最佳 Gate 類型</th><th>嚴格防禦參數</th><th>模型活躍率</th><th>基準 PnL (對照組)</th><th>嚴格防禦後 PnL</th><th>提升 (Improvement)</th></tr>
      {tr_07}
    </table>
    <p class="note" style="margin-top:8px">
      目前的回測並未完全具備排隊感知 (Queue-aware)；僅用於一階訊號比較。
    </p>
  </div>
</div></body></html>
"""
(OUT / "04_slides.html").write_text(html_04, encoding="utf-8")


# ======================= SLIDE 05 =======================
df_09 = pd.read_csv(DERRIK / "09_fill_conditional_labels" / "fill_conditional_label_summary.csv")
tr_09 = ""
for _, r in df_09.iterrows():
    tr_09 += f"<tr><td>{r['dataset']}</td><td>{r['bid_toxic_rate_conditional']*100:.2f}%</td><td class='hl green'>{r['bid_toxic_rate_lift']:.2f}x</td><td>{r['ask_toxic_rate_conditional']*100:.2f}%</td><td class='hl green'>{r['ask_toxic_rate_lift']:.2f}x</td></tr>"

lat_img = b64(DERRIK / "10_latency_sensitivity" / "cancel_delay_net_pnl.png")

html_05 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 05</title>{CSS}</head><body>
<div class="slide" style="overflow-y:auto">
  <h1 style="font-size:1.6rem">Slide 05 — Cancel Delay Increases Risk, But PnL Is Non-monotonic (撤單延遲增加風險，但 PnL 呈非單調變化)</h1>

  <div class="grid-2">
    <div class="card" style="padding:15px;">
      <h2>發現一：條件成交 (Fill-Conditional) 毒性飆升</h2>
      <div style="background:#21262d; padding:10px; border-radius:6px; margin-bottom:10px; font-family:monospace; font-size:0.75rem;">
        Bid toxic = P(future_mid − current_bid < −0.5 tick | bid price level removed)<br>
        Ask toxic = P(current_ask − future_mid < −0.5 tick | ask price level removed)<br>
        Lift = conditional toxic rate / unconditional toxic rate
      </div>
      <table style="font-size:0.75rem;">
        <tr><th>資料集</th><th>Bid 條件毒性</th><th>Bid 提升</th><th>Ask 條件毒性</th><th>Ask 提升</th></tr>
        {tr_09}
      </table>
      <p class="note" style="font-size:0.7rem;">值得實作並驗證單邊特定的條件成交報價風險標籤 (side-specific fill-conditional quote-risk labels)；目前的 016 實驗仍屬控制性消融 (controlled ablation)，尚無法保證優於平滑基準模型。<br><br>
      Proxy C 僅是基於價格檔位被移除的假設，並非交易所真實的成交真相。</p>
    </div>
    
    <div class="card" style="padding:15px;">
      <h2>發現二：撤單延遲 (Cancel Delay) 帶來的非單調風險</h2>
      <div class="img-wrap"><img src="{lat_img}" alt="Latency vs PnL" style="max-height:160px;"></div>
      <p class="note" style="margin-top:8px; font-size:0.75rem;">
        延遲增加的初期，PnL 呈現非單調性 (Non-monotonic) 變化，甚至可能因為過濾假陽性而稍微回升。但當延遲過高，防禦力開始顯著惡化。<br><br>
        <span class="hl">FPGA 是用來降低最差情況下推論與撤單路徑延遲的候選方案；此處尚未證明端到端 (End-to-end) 的交易所撤單延遲效益。</span>
      </p>
    </div>
  </div>
</div></body></html>
"""
(OUT / "05_slides.html").write_text(html_05, encoding="utf-8")


# ======================= SLIDE 06 =======================
html_06 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 06</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 06 — 目前狀態與下週計畫</h1>
  
  <div class="grid-2">
    <div class="card">
      <h2 style="color:#56d364">目前已達成 (Achieved Milestones)</h2>
      <ul style="font-size:0.9rem;">
        <li>TCN 在 NEW 測試集具備局部優勢：0.55/0.60 成功擊敗 Random+1Std。</li>
        <li>靜態狀態濾網成功拯救 NEW3：Combined filter 使 NEW3 虧損大幅收斂並轉正。</li>
        <li>016 二元標籤流程已作為概念驗證 (PoC) 證實可行，但模型尚未就緒。</li>
        <li>017 簡易動態閾值 (Auto-gate) 實驗失敗；這顯示了更深層的狀態依賴性 (Regime dependency)。</li>
        <li>已量化延遲風險，但 PnL 隨延遲呈現非單調變化。</li>
      </ul>
    </div>
    
    <div class="card">
      <h2 style="color:#ff7b7b">目前尚未達成 (Not Achieved)</h2>
      <ul style="font-size:0.9rem;">
        <li>驗證集選出的閾值 (0.70) 未能擊敗 Random+1Std。</li>
        <li>Binary 模型未能擊敗 Random+1Std，且僅觸發 77 筆交易。</li>
        <li>尚未達成穩健的動態閾值選擇器 (Auto-gate selector)。</li>
        <li>尚未完成具備排隊感知 (Queue-aware) 的真實回測。</li>
        <li>FPGA 對於 PnL 的絕對必要性尚未被完全證實。</li>
      </ul>
    </div>
  </div>

  <div class="card" style="margin-top:0px;">
    <h2 style="color:#58a6ff">下週核心行動計畫 (Next Steps)</h2>
    <ol style="margin-left:20px; line-height:1.8; font-size:0.95rem;">
      <li><strong>Queue-aware 驗證：</strong>實作成交懲罰 (Fill haircut)、優質成交懲罰 (Good-fill haircut) 與排隊逼近演算法。</li>
      <li><strong>狀態研究路徑 (Regime path)：</strong>短期維持靜態 Combined filter；長期需引入更多 Session 來訓練狀態分類器。</li>
      <li><strong>二元標籤路徑 (Binary path)：</strong>保留 016 作為標籤 PoC；嚴禁在測試集上進行閾值擬合。</li>
    </ol>
  </div>
</div></body></html>
"""
(OUT / "06_slides.html").write_text(html_06, encoding="utf-8")

print("Slides completely translated to Chinese and written to files.")
