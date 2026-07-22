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

.flowchart {display:flex; align-items:center; justify-content:center; gap:10px; background:#0d1117; padding:15px; border-radius:8px; border:1px solid #30363d;}
.flow-box {background:#21262d; border:1px solid #58a6ff; padding:10px 15px; border-radius:6px; font-weight:bold; color:#c9d1d9; text-align:center;}
.flow-arrow {color:#8b949e; font-weight:bold; font-size:1.2rem;}
</style>
"""

# ======================= SLIDE 01 =======================
loss_img = b64(DERRIK / "03_model_classification" / "01_training_loss_plot_annotated.png")
cm_img   = b64(DERRIK / "03_model_classification" / "02_confusion_matrix_percent_test.png")

html_01 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 01</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 01 — 最終模型：從方向性預測到毒性撤單過濾器</h1>

  <!-- 新增：模型架構與 Latency 解釋 -->
  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card">
      <h2>模型架構 (Architecture)</h2>
      <div class="flowchart" style="font-size:0.8rem; padding:10px;">
        <div class="flow-box" style="border-color:#8b949e">Input<br><span style="font-size:0.7rem;font-weight:normal">8 Features<br>100 Ticks</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-box">TCN Block 1<br><span style="font-size:0.7rem;font-weight:normal">Dilated Conv</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-box">TCN Block 2<br><span style="font-size:0.7rem;font-weight:normal">Dilated Conv</span></div>
        <div class="flow-arrow">➔</div>
        <div class="flow-box" style="border-color:#56d364">Softmax<br><span style="font-size:0.7rem;font-weight:normal">3 Classes</span></div>
      </div>
      <p class="note" style="margin-top:10px;">輕量級 TCN 架構，專為低延遲推論設計，無遞迴(RNN)負擔。</p>
    </div>
    <div class="card">
      <h2>硬體速度需求：Latency < 5 ticks 代表多快？</h2>
      <p style="font-size:0.9rem; line-height:1.5;">一個 "tick" 是指 Order book 的一次事件更新。在快市 (Volatile bursts) 中，<strong>5 ticks 的發生間隔可能遠小於 1 毫秒 (ms)</strong>。</p>
      <div class="badge-row" style="margin-top:10px">
        <span class="tag tag-red">Python/軟體推論：常有 GC 或 OS 抖動，容易錯失黃金防禦期。</span>
        <span class="tag tag-green">FPGA 硬體加速：確保 Deterministic 的奈秒/微秒級撤單。</span>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>模型摘要與策略定位</h2>
      <table>
        <tr><th>輸入</th><td><span class="hl">8</span> 個特徵 / <span class="hl">100</span> ticks</td></tr>
        <tr><th>資料</th><td>Train 718k / Val 112k / Test 233k</td></tr>
        <tr><th>定位</th><td><span class="tag tag-green">防禦撤單</span> <span class="tag tag-red" style="margin-left:5px">非方向性下單</span></td></tr>
      </table>
      <p class="note" style="margin-top:8px">Down/Up Precision≈<strong>2.8%</strong> (盤整多)，但 Recall≈<strong>67%</strong> (捕捉多數不利移動)，寧可錯殺不願放過。</p>
    </div>
    <div class="card">
      <h2>分類報告（測試集）</h2>
      <table>
        <tr><th>類別</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
        <tr><td>Down</td><td class="red">0.028</td><td class="green">0.694</td><td>0.054</td><td>2,762</td></tr>
        <tr><td>Stationary</td><td class="green">0.987</td><td>0.452</td><td>0.620</td><td>227,711</td></tr>
        <tr><td>Up</td><td class="red">0.029</td><td class="green">0.642</td><td>0.056</td><td>2,795</td></tr>
      </table>
    </div>
  </div>

  <!-- Loss Curve + Confusion Matrix -->
  <div class="grid-2">
    <div class="card">
      <h2>訓練收斂曲線 (Loss Curve)</h2>
      <div class="img-wrap"><img src="{loss_img}" alt="Loss Curve" style="max-height:220px;"></div>
    </div>
    <div class="card">
      <h2>混淆矩陣 (Confusion Matrix)</h2>
      <div class="img-wrap"><img src="{cm_img}" alt="Confusion Matrix" style="max-height:220px;"></div>
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
  
  <!-- 新增：回測策略流程圖 -->
  <div class="card" style="margin-bottom:0px;">
    <h2>回測策略流程 (Maker-only Cancel Filter)</h2>
    <div class="flowchart">
      <div class="flow-box">做市商維持掛單<br><span style="font-size:0.75rem;font-weight:normal">Bid & Ask in market</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#e3b341">TCN 即時預測<br><span style="font-size:0.75rem;font-weight:normal">P(Down), P(Up)</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#ff7b7b">檢查閾值<br><span style="font-size:0.75rem;font-weight:normal">P > Threshold?</span></div>
      <div class="flow-arrow">➔</div>
      <div class="flow-box" style="border-color:#56d364">執行防禦<br><span style="font-size:0.75rem;font-weight:normal">撤銷被狙擊方向之掛單</span></div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card">
      <h2>各閾值 (Thresholds) 的設定涵義</h2>
      <ul style="font-size:0.9rem;">
        <li><span class="hl">0.55 / 0.60 (較敏感)</span>：只要模型稍微覺得危險就撤單。撤單次數多，可能過濾掉一些安全利潤，但能最大程度擋下毒性單。</li>
        <li><span class="hl">0.70 (較保守)</span>：模型非常確定才撤單。撤單次數少，保留較多做市機會，但可能漏掉一些輕微毒性單。</li>
      </ul>
    </div>
    <div class="card">
      <h2>Validation Loss 的選取過程</h2>
      <p style="font-size:0.9rem; line-height:1.6;">我們在 <strong>Validation Set</strong> 進行了 Threshold Sweep，發現 <strong>0.70</strong> 給出了理論上最佳的 Simulated Metrics。<br><br>然而，將這些閾值套用到 <strong>Test Set (OOS)</strong> 時，卻發現 0.55/0.60 的表現更好。這顯示了最佳閾值具有強烈的「狀態依賴性」，也是我們後續必須做 Auto-gating 的原因之一。</p>
    </div>
  </div>

  <div class="card">
    <h2>NEW 測試集 (Test Split) — PnL 實測證據</h2>
    <table>
      <tr><th>實驗設定</th><th>模型 Net PnL</th><th>隨機撤單平均值</th><th>隨機平均 + 1 Std</th><th>擊敗 Random+1Std</th><th>解讀</th></tr>
      {tr_04}
    </table>
    <p class="note" style="margin-top:16px">
      <strong>重點結論：</strong> 0.55 / 0.60 展現出優異的局部優勢 (Local Edge)，成功擊敗 Random+1Std。0.70 表現不如預期，因此 0.55 暫定為目前的「測試候選 (Test-sweep candidate)」。
    </p>
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
    tr_06 += f"<tr><td>{r['session_id']}</td><td>{r['trend_strength']:.2f}</td><td>{r['suitability_label']}</td><td class='{color}'>{pnl}</td><td>{r['main_failure_mode']}</td></tr>"

html_03 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 03</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 03 — 為什麼需要狀態濾網 (Why Regime Gate)</h1>
  
  <!-- 新增：怎麼知道 NEW3 充滿強趨勢 -->
  <div class="card" style="margin-bottom:0px;">
    <h2>我們如何得知 NEW3 充滿強烈趨勢？</h2>
    <div class="grid-2">
      <div>
        <p style="font-size:0.9rem; line-height:1.6;">我們透過計算每個 Session 的 <strong>Trend Strength (趨勢強度)</strong> 來量化市場狀態：<br>
        <code>Trend Strength = 絕對價格位移 (Absolute Price Displacement) / 時間長度</code></p>
        <p style="font-size:0.9rem; line-height:1.6; margin-top:10px;">數據顯示，NEW3 的各個 Session 具有異常高的趨勢強度指標（如表格所示，數值顯著放大），這代表市場在短時間內出現了巨大的單向推進（例如百點以上的急拉或急殺）。</p>
      </div>
      <div style="background:#0d1117; padding:15px; border-radius:8px; border:1px solid #30363d; display:flex; align-items:center;">
        <p style="color:#e6edf3; font-size:0.85rem; line-height:1.5;">
          <span class="tag tag-red">災難發生原因：</span><br>
          TCN 的 Window 只有 100 ticks。在緩步攀升或急劇推升的強趨勢中，100 ticks 內的特徵可能看起來像「震盪」，導致模型判斷安全而維持掛單，結果被方向性的毒性巨量單 (Toxic block trades) 瞬間穿透。
        </p>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>壓力測試：微觀模型在 NEW3 下的崩潰</h2>
    <table>
      <tr><th>Session ID</th><th>趨勢強度 (Trend)</th><th>適合度標籤</th><th>無 Gate 基準 PnL</th><th>主要失敗模式</th></tr>
      {tr_06}
      <tr style="background:#21262d"><td colspan="3"><strong>NEW3 總虧損</strong></td><td colspan="2" class="red hl" style="font-size:1.1rem">-557,165 </td></tr>
    </table>
    <p class="note" style="margin-top:16px">
      <strong>核心洞見：</strong> 微觀的過濾器救不了宏觀的崩盤！這證明 <strong>NEW3 不是無效數據，而是必須透過事前 Causal Gate 防禦的極端市場狀態。</strong>
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
  
  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card">
      <h2 style="color:#56d364">Rolling Trend (滾動趨勢濾網)</h2>
      <p style="font-size:0.85rem; margin-bottom:8px;"><strong>原理：</strong>偵測極端單向走勢以防 Maker 被連續穿透。<br><strong>作法：</strong>計算過去一段時間內絕對價格位移，當 <code>Trend > trend_ticks</code> 時啟動防禦 (暫停掛單)。</p>
    </div>
    <div class="card">
      <h2 style="color:#56d364">Spread Filter (價差濾網)</h2>
      <p style="font-size:0.85rem; margin-bottom:8px;"><strong>原理：</strong>衡量即時流動性枯竭狀況。<br><strong>作法：</strong>即時監控 Bid-Ask Spread。當 <code>Spread > spread_ticks_min</code> (如 >2.0) 時啟動防禦。</p>
    </div>
  </div>

  <!-- 新增：Combined Filter 與 no_gate_reference 解釋 -->
  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card" style="border-color:#e3b341;">
      <h2 style="color:#e3b341">什麼是 Combined Filter？</h2>
      <p style="font-size:0.85rem; line-height:1.5;">這是<strong>綜合防禦機制</strong>。邏輯為 <code>Trend Filter OR Volatility Filter OR Spread Filter</code>。只要任何一個巨觀指標發出危險訊號，模型就全線暫停做市。這在異常極端的 NEW3 中展現了最強的止損效果。</p>
    </div>
    <div class="card">
      <h2>關於 <code>_no_gate_reference</code> 資料集</h2>
      <p style="font-size:0.85rem; line-height:1.5;">為了取得基準對比，我們使用 <code>spread_ticks_min: 1.0</code> 作為對照組。因為台指期 (TXF) 的 Spread 絕大多數時間都大於等於 1，這表示 Gate <strong>幾乎永遠不會觸發</strong>，等同於「完全無濾網 (No Gate)」的裸跑狀態。</p>
    </div>
  </div>
  
  <div class="card">
    <h2>加入巨觀濾網後的防禦力提升 (Regime Dependency)</h2>
    <table>
      <tr><th>資料集</th><th>最佳 Gate 類型</th><th>參數設定</th><th>模型活躍率</th><th>基準 PnL (無 Gate)</th><th>Gate+模型 PnL</th><th>提升 (Improvement)</th></tr>
      {tr_07}
    </table>
    <p class="note" style="margin-top:16px">
      <strong>成效：</strong> 透過 Combined Filter，NEW3 虧損從 -55萬 大幅拉回，提升了 <strong>26 萬</strong>。活躍率仍有 82%，證明不是完全停止交易。
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
  <h1>Slide 05 — 深度診斷方法與新發現</h1>
  
  <!-- 新增：診斷方法與實作方式 -->
  <div class="card" style="margin-bottom:0px; padding:15px 20px;">
    <h2>診斷方法與實作方式 (Diagnostic Framework)</h2>
    <p style="font-size:0.9rem; line-height:1.6;">我們並非只看總 PnL，而是實作了 <code>simulate_with_metrics</code> 診斷框架。該框架會將模型的每一次預測 (Prediction) 與未來的真實 Order book 事件 (Future LOB events) 進行交叉比對 (Cross-reference)。<br>
    透過這個微觀標記，我們能精確計算出：
    <span class="hl">機會成本 (Opportunity Cost)</span> = 成功避開的毒性虧損 (Avoided Toxic Loss) <strong>對比</strong> 錯失的獲利機會 (Missed Profitable PnL)。這讓我們得以用數據證明模型撤單是「利大於弊」。</p>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>發現一：條件成交 (Fill-Conditional) 的毒性飆升</h2>
      <p style="font-size:0.85rem; margin-bottom:8px">在「有成交條件下」的毒性風險，相比「無條件」有顯著倍數提升：</p>
      <table style="font-size:0.75rem;">
        <tr><th>資料集</th><th>Bid 條件毒性</th><th>Bid 提升</th><th>Ask 條件毒性</th><th>Ask 提升</th></tr>
        {tr_09}
      </table>
      <div class="badge-row" style="margin-top:10px">
        <span class="tag tag-blue">指引：必須拆分 bid_toxic 與 ask_toxic 重訓</span>
      </div>
      <p class="note" style="font-size:0.7rem;">⚠️ 基於 Proxy C，並非交易所真實成交的 Absolute Truth。</p>
    </div>
    
    <div class="card">
      <h2>發現二：撤單延遲 (Cancel Latency) 造成的防禦崩潰</h2>
      <div class="img-wrap"><img src="{lat_img}" alt="Latency vs PnL" style="max-height:160px;"></div>
      <p class="note" style="margin-top:8px; font-size:0.75rem;">
        當 Delay 達到 <strong>10 ticks</strong> 時防禦力崩盤。<br>
        <span class="red">👉 這給了我們投入 FPGA 硬體加速最強的學術理由：極致的 Risk-Control。</span>
      </p>
    </div>
  </div>
</div></body></html>
"""
(OUT / "05_slides.html").write_text(html_05, encoding="utf-8")

print("Slides 01 to 05 updated successfully.")
