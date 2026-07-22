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
  <h1>Slide 01 — 最終模型：架構與成效指標</h1>

  <div class="card" style="margin-bottom:0px; padding:15px;">
    <h2>模型架構 (Temporal Convolutional Network, TCN)</h2>
    <div class="flowchart" style="font-size:0.85rem; padding:12px;">
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

  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card">
      <h2>模型摘要與策略定位</h2>
      <table>
        <tr><th>目標標籤</th><td><code>smoothed_mid_move_3class</code></td></tr>
        <tr><th>資料來源</th><td>NEW_ALL (商品: TXFB6，台指期)</td></tr>
        <tr><th>資料切分</th><td>Train (6天) / Val (1天) / Test (2天)</td></tr>
        <tr><th>策略定位</th><td><span class="tag tag-green">Maker 防禦撤單過濾器</span><br><span style="font-size:0.75rem; color:#8b949e;">負責在危險來臨時撤單，絕不負責主動進場</span></td></tr>
      </table>
    </div>
    <div class="card">
      <h2>分類報告（Test Split，233k 筆資料）</h2>
      <table>
        <tr><th>類別</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
        <tr><td>Down</td><td class="red">0.028</td><td class="green">0.694</td><td>0.054</td><td>2,762</td></tr>
        <tr><td>Stationary</td><td class="green">0.987</td><td>0.452</td><td>0.620</td><td>227,711</td></tr>
        <tr><td>Up</td><td class="red">0.029</td><td class="green">0.642</td><td>0.056</td><td>2,795</td></tr>
        <tr style="background:#21262d"><td>Macro Avg</td><td>0.348</td><td>0.596</td><td>0.243</td><td>233,268</td></tr>
        <tr style="background:#21262d"><td>Weighted Avg</td><td>0.964</td><td>0.457</td><td>0.606</td><td>233,268</td></tr>
      </table>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>訓練收斂曲線 (Loss Curve)</h2>
      <div class="img-wrap"><img src="{loss_img}" alt="Loss Curve" style="height:250px; object-fit:contain;"></div>
    </div>
    <div class="card">
      <h2>混淆矩陣 (Confusion Matrix)</h2>
      <div class="img-wrap"><img src="{cm_img}" alt="Confusion Matrix" style="height:250px; object-fit:contain;"></div>
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
  
  <div class="card" style="margin-bottom:0px;">
    <h2>回測策略流程 (Maker-only Cancel Filter)</h2>
    <div class="flowchart" style="margin-bottom:10px;">
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
      <ul style="font-size:0.9rem;">
        <li><span class="hl">0.55 / 0.60 (較敏感)</span>：只要模型稍微覺得危險就撤單。撤單次數多，可能過濾掉一些安全利潤，但能最大程度擋下毒性單。</li>
        <li><span class="hl">0.70 (較保守)</span>：模型非常確定才撤單。撤單次數少，保留較多做市機會，但可能漏掉一些輕微毒性單。</li>
      </ul>
    </div>
    <div class="card">
      <h2>Validation Sweep：如何選出最佳參數？</h2>
      <p style="font-size:0.9rem; line-height:1.6;">我們在 <strong>Validation Set</strong> 執行參數掃描 (Sweeping)，讓閾值從 0.50 一路測試到 0.90。<br>結果發現 <strong>0.70</strong> 在驗證集上給出了理論最佳的淨 PnL。<br><br>然而，將這些閾值套用到 <strong>Test Set (樣本外)</strong> 時，卻發現 0.55/0.60 的表現更好。這顯示了最佳閾值具有強烈的「狀態依賴性」。</p>
    </div>
  </div>

  <div class="card">
    <h2>NEW 測試集 (Test Split) — PnL 實測證據</h2>
    <table>
      <tr><th>實驗設定</th><th>模型 Net PnL</th><th>隨機撤單平均值</th><th>隨機平均 + 1 Std</th><th>擊敗 Random+1Std</th><th>解讀</th></tr>
      {tr_04}
    </table>
    <p class="note" style="margin-top:8px">
      <strong>重點結論：</strong> 0.55 / 0.60 成功擊敗 Random+1Std，具備局部優勢 (Local Edge)。依 Val 選出的 0.70 卻未能穩健擊敗，故 0.55 暫定為目前的測試候選。
    </p>
  </div>
</div></body></html>
"""
(OUT / "02_slides.html").write_text(html_02, encoding="utf-8")


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
      <h2 style="color:#56d364; font-size:1rem;">Rolling Trend (趨勢濾網)</h2>
      <p style="font-size:0.8rem; margin-bottom:8px;"><strong>原理：</strong>偵測極端單向走勢以防被穿透。<br><strong>作法：</strong>計算過去一段時間內絕對價格位移，當 <code>Trend > trend_ticks</code> 時啟動防禦。</p>
    </div>
    <div class="card" style="padding:15px;">
      <h2 style="color:#56d364; font-size:1rem;">Volatility Filter (波動率濾網)</h2>
      <p style="font-size:0.8rem; margin-bottom:8px;"><strong>原理：</strong>監控市場真實波動，波動過大代表風險飆升。<br><strong>作法：</strong>計算一段視窗內的標準差，當 <code>Vol > vol_threshold</code> 時，暫停 Maker 雙向掛單。</p>
    </div>
    <div class="card" style="padding:15px;">
      <h2 style="color:#56d364; font-size:1rem;">Spread Filter (價差濾網)</h2>
      <p style="font-size:0.8rem; margin-bottom:8px;"><strong>原理：</strong>衡量即時流動性枯竭狀況。<br><strong>作法：</strong>當 <code>Spread > spread_ticks_min</code> 時啟動防禦。</p>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card" style="border-color:#e3b341;">
      <h2 style="color:#e3b341">1. Combined Filter 與 活躍率算法</h2>
      <p style="font-size:0.85rem; line-height:1.5;"><strong>Combined Filter (綜合防禦機制)：</strong>邏輯為 <code>Trend OR Volatility OR Spread</code>。任一異常即停機。<br>
      <strong>活躍率 (Active Event Rate)：</strong><br>
      <code>(總評估次數 - 因 Gate 被暫停的次數) / 總評估次數</code>。<br>
      代表模型真正在市場運作的時間比例。82% 表示我們只放棄了 18% 最危險的時間。</p>
    </div>
    <div class="card">
      <h2>2. 對照組：什麼是 <code>_no_gate_reference</code>？</h2>
      <p style="font-size:0.85rem; line-height:1.5;">台指期的 Spread 大多時間 ≥ 1。我們設定 <code>spread_ticks_min: 1.0</code> 作為**對照組 (Control Group)**，表示 Gate 永遠不觸發 (裸跑狀態)。<br>
      <strong>濾網的意義：</strong>當我們在 NEW 實驗加上 <code>spread_ticks_min: 2.0</code> 時，代表條件變**嚴格**了。只有這種對比，才能證明「開啟防禦濾網」真的能擋下虧損！</p>
    </div>
  </div>
  
  <div class="card">
    <h2>加入巨觀濾網後的防禦力提升 (Regime Dependency)</h2>
    <table>
      <tr><th>資料集</th><th>最佳 Gate 類型</th><th>嚴格防禦參數</th><th>模型活躍率</th><th>基準 PnL (對照組裸跑)</th><th>嚴格防禦後 PnL</th><th>防禦提升 (Improvement)</th></tr>
      {tr_07}
    </table>
    <p class="note" style="margin-top:10px">
      <strong>成效：</strong> NEW3 虧損從 -55萬 大幅拉回，防禦提升 <strong>26 萬</strong>。活躍率仍有 82%，證明不是完全停止交易，而是精準避險。
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
  
  <div class="grid-2" style="margin-bottom:0px;">
    <div class="card">
      <h2>微觀診斷方法：如何算出機會成本？</h2>
      <p style="font-size:0.85rem; line-height:1.5;">我們實作了 <code>simulate_with_metrics</code> 診斷框架，將預測與 LOB 事件精確交叉比對 (Cross-reference)：</p>
      <ul style="font-size:0.8rem; margin-top:5px;">
        <li><strong>步驟 1：</strong>當模型發出 Cancel 訊號，往未來觀察 <code>fill_window</code> (10 ticks)。</li>
        <li><strong>步驟 2：</strong>若該掛單原本會被成交，再往未來觀察 <code>horizon</code> (10 ticks)。</li>
        <li><strong>步驟 3：</strong>若價格最終往不利方向移動，代表這是一筆 <strong>毒性單 (Toxic fill)</strong>。</li>
      </ul>
      <div style="background:#21262d; padding:10px; border-radius:6px; margin-top:10px; font-size:0.85rem;">
        <span class="hl">機會成本 (Opportunity Cost)</span> =<br>
        <span class="green">成功避開的毒性虧損</span> <strong>減去</strong> <span class="red">錯失的安全獲利機會</span>
      </div>
    </div>

    <div class="card">
      <h2>硬體速度需求：Latency < 5 ticks 代表多快？</h2>
      <p style="font-size:0.9rem; line-height:1.5; margin-bottom:10px;">一個 "tick" 是指 Order book 的一次事件更新。在快市 (Volatile bursts) 中，<strong>5 ticks 的發生間隔可能遠小於 1 毫秒 (ms)</strong>。</p>
      <div class="badge-row">
        <span class="tag tag-red">Python/軟體推論：常有 GC 抖動，容易錯失黃金防禦期</span>
        <span class="tag tag-green">FPGA 硬體加速：確保 Deterministic 奈秒級撤單</span>
      </div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>發現一：條件成交 (Fill-Conditional) 的毒性飆升</h2>
      <p style="font-size:0.85rem; margin-bottom:8px">在「有成交條件下」的毒性風險，相比「無條件」有顯著提升：</p>
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
      <div class="img-wrap"><img src="{lat_img}" alt="Latency vs PnL" style="max-height:150px;"></div>
      <p class="note" style="margin-top:8px; font-size:0.75rem;">
        當 Delay 達到 <strong>10 ticks</strong> 時防禦力崩盤。<br>
        <span class="red">👉 這印證了上述 FPGA 硬體加速的強烈學術理由：極致風險控管。</span>
      </p>
    </div>
  </div>
</div></body></html>
"""
(OUT / "05_slides.html").write_text(html_05, encoding="utf-8")

print("Slides 01, 02, 04, 05 heavily refined according to user instructions.")
