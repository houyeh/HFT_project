import pandas as pd
import base64
from pathlib import Path

ROOT = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project\meeting\w13")
DERRIK = ROOT / "___Derrik_results"
OUT = ROOT / "slides"
OUT.mkdir(exist_ok=True)

def b64(p):
    if not p.exists(): return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

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

# ======================= SLIDE 00 =======================
html_00 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 00</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 00 — 模型與標籤選擇過程</h1>
  <div class="grid-2">
    <div class="card">
      <h2>為什麼放棄舊的 Toxic Labels?</h2>
      <ul>
        <li><span class="red">類別極度不平衡</span>：舊版的 <code>adverse_bid_fill</code> 要求「價格跌破 + 立即成交」，這在真實 Order book 極少發生，導致正樣本近乎為 0。</li>
        <li><span class="red">Macro F1 接近 0.5 (隨機)</span>：模型無法單憑 Order book 區分出嚴格條件下的毒性，變成盲目預測多數類。</li>
        <li><span class="red">嚴格利潤條件導致大量假陽性</span>：<code>active_buy_profitable</code> 要求利潤克服 Spread + 50 TWD 手續費，門檻過高，引發大量 False Positives，導致 PnL 平均虧損一萬元以上。</li>
      </ul>
      <div class="badge-row"><span class="tag tag-red">結論：全數拒絕舊版二元標籤</span></div>
    </div>
    <div class="card">
      <h2>Label 海選結果 (8 種 Label 評估)</h2>
      <table>
        <tr><th>Label 類型</th><th>平均 PnL 改善</th><th>決策</th></tr>
        <tr><td>smoothed_mid_move_3class</td><td class="green">正提升 (最高)</td><td><span class="tag tag-green">最終選取</span></td></tr>
        <tr><td>mid_move_3class</td><td class="green">正提升</td><td><span class="tag tag-blue">候補</span></td></tr>
        <tr><td>toxic_bid_binary</td><td>0.00</td><td><span class="tag tag-red">拒絕</span></td></tr>
        <tr><td>toxic_ask_binary</td><td class="red">-685.00</td><td><span class="tag tag-red">拒絕</span></td></tr>
        <tr><td>adverse_fill_binary</td><td>0.00</td><td><span class="tag tag-red">拒絕</span></td></tr>
        <tr><td>active_profitable</td><td class="red">-11,796.00</td><td><span class="tag tag-red">拒絕</span></td></tr>
      </table>
      <p class="note">最終選定的 <code>smoothed_mid_move_3class</code> 雖非完美，但目前能提供最平衡、最穩定的風險評分 (Risk-score)。</p>
    </div>
  </div>
</div></body></html>
"""
(OUT / "00_slides.html").write_text(html_00, encoding="utf-8")


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
  <div class="card">
    <h2>NEW 測試集 (Test Split) — 策略：Maker-only 撤單過濾器</h2>
    <p style="margin-bottom:12px">邏輯：當模型預測危險機率 > Threshold 時，立刻撤銷掛單。</p>
    <table>
      <tr><th>實驗設定</th><th>模型 Net PnL</th><th>隨機撤單平均值</th><th>隨機平均 + 1 Std</th><th>是否擊敗 Random+1Std</th><th>解讀</th></tr>
      {tr_04}
    </table>
    <p class="note" style="margin-top:16px">
      <strong>重點結論：</strong><br>
      1. <strong>Threshold 0.55 / 0.60 展現出局部優勢 (Local Edge)</strong>，成功且穩健地擊敗了 Random + 1 Std 的嚴格基準。<br>
      2. <strong>誠實聲明：</strong>依據 Validation Loss 選出的 0.70 並未能在測試集穩健擊敗 Random+1Std，因此目前的 0.55 僅為「測試候選 (Test-sweep candidate)」，策略尚未達到 Production-ready。
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
  <div class="card">
    <h2>壓力測試：微觀模型在強趨勢下的崩潰 (NEW3 Dataset)</h2>
    <p style="margin-bottom:12px">我們將只看過去 100 ticks 的微觀模型，放入充滿強烈趨勢的 <code>NEW3</code> 資料集測試，結果如下：</p>
    <table>
      <tr><th>Session ID</th><th>趨勢強度 (Trend)</th><th>適合度標籤</th><th>無 Gate 基準 PnL</th><th>主要失敗模式</th></tr>
      {tr_06}
      <tr style="background:#21262d"><td colspan="3"><strong>NEW3 總虧損</strong></td><td colspan="2" class="red hl" style="font-size:1.1rem">-557,165 </td></tr>
    </table>
    <p class="note" style="margin-top:16px">
      <strong>核心洞見：</strong><br>
      微觀的過濾器救不了宏觀的崩盤！TCN 看不到 100 ticks 外的大趨勢，導致在強趨勢的 Regime 遭受毀滅性虧損。<br>
      這證明 <strong>NEW3 不是無效數據，而是必須透過事前 Causal Gate (因果邏輯閘) 防禦的極端市場狀態。</strong>
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
  <div class="card">
    <h2>加入巨觀濾網後的防禦力提升 (Regime Dependency)</h2>
    <p style="margin-bottom:12px">我們加入了 Rolling Trend 與 Spread Filter，這兩者構成了我們的 Regime Gate：</p>
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


# ======================= SLIDE 05 =======================
df_09 = pd.read_csv(DERRIK / "09_fill_conditional_labels" / "fill_conditional_label_summary.csv")
tr_09 = ""
for _, r in df_09.iterrows():
    tr_09 += f"<tr><td>{r['dataset']}</td><td>{r['bid_toxic_rate_conditional']*100:.2f}%</td><td class='hl green'>{r['bid_toxic_rate_lift']:.2f}x</td><td>{r['ask_toxic_rate_conditional']*100:.2f}%</td><td class='hl green'>{r['ask_toxic_rate_lift']:.2f}x</td></tr>"

lat_img = b64(DERRIK / "10_latency_sensitivity" / "cancel_delay_net_pnl.png")

html_05 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 05</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 05 — 深度診斷新發現：Label 與 Latency</h1>
  
  <div class="grid-2">
    <div class="card">
      <h2>發現一：條件成交 (Fill-Conditional) 的毒性飆升</h2>
      <p style="margin-bottom:8px">在「有成交條件下」的毒性風險，相比「無條件」有顯著倍數提升：</p>
      <table>
        <tr><th>資料集</th><th>Bid 條件毒性率</th><th>Bid 提升倍數</th><th>Ask 條件毒性率</th><th>Ask 提升倍數</th></tr>
        {tr_09}
      </table>
      <div class="badge-row" style="margin-top:10px">
        <span class="tag tag-blue">指引：必須拆分 bid_toxic 與 ask_toxic 重新訓練</span>
      </div>
      <p class="note">⚠️ 基於替代指標 (Proxy C)，並非交易所真實成交的 Absolute Truth。</p>
    </div>
    
    <div class="card">
      <h2>發現二：撤單延遲 (Cancel Latency) 造成的防禦崩潰</h2>
      <div class="img-wrap"><img src="{lat_img}" alt="Latency vs PnL"></div>
      <p class="note" style="margin-top:8px">
        1. 在 Delay 3~5 時，PnL 可能非單調上升（過濾假陽性）。<br>
        2. 當 Delay 達到 <strong>10 ticks</strong> 時，防禦力崩盤，表現大幅惡化。<br>
        <span class="red">👉 這給了我們投入 FPGA 硬體加速最強的學術理由：不是為了絕對獲利，而是極致的 Risk-Control。</span>
      </p>
    </div>
  </div>
</div></body></html>
"""
(OUT / "05_slides.html").write_text(html_05, encoding="utf-8")


# ======================= SLIDE 06 =======================
html_06 = f"""<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8"><title>Slide 06</title>{CSS}</head><body>
<div class="slide">
  <h1>Slide 06 — 目前狀態與下週行動事項</h1>
  
  <div class="grid-2">
    <div class="card">
      <h2 style="color:#56d364">目前已達成的里程碑 (C+ / B- Frontier)</h2>
      <ul>
        <li>證明 TCN 在 NEW 測試集上有 Local edge。</li>
        <li>模型 (Threshold 0.55 / 0.60) 成功擊敗 Random+1std。</li>
        <li>Regime Gate 成功在 NEW3 壓力測試中發揮救援作用。</li>
        <li>Fill-conditional 審查明確了毒性率的提升方向。</li>
        <li>具體量化了 Cancel Delay 帶來的虧損崩潰點。</li>
      </ul>
      <h2 style="color:#ff7b7b; margin-top:20px">目前尚未達成</h2>
      <ul>
        <li>Validation 嚴謹選出的 Threshold 尚未能在測試集穩健擊敗 Random+1std。</li>
        <li>尚未完成真實的 Queue-aware (HftBacktest-style) 回測。</li>
        <li>尚未開發出動態適應的 Auto-Gating 機制。</li>
      </ul>
    </div>
    
    <div class="card">
      <h2 style="color:#58a6ff">下週核心行動計畫 (Next Steps)</h2>
      <ol style="margin-left:20px; line-height:1.8; margin-top:10px">
        <li><strong>標籤重建與重訓：</strong> 建立專屬的 <code>bid_toxic</code> 與 <code>ask_toxic</code> 二元標籤，並使用這兩個 Side-specific 的新標籤重新訓練輕量級模型。</li>
        <li><strong>嚴格的閾值驗證：</strong> 在有 Regime Gate 防護的前提下，透過 Validation 集嚴謹選擇 Threshold，並與 Rule-based 基準進行對比。</li>
        <li><strong>Auto-Gating 實作：</strong> 使用 Causal Feature (如 Spread, Volatility) 建立自動辨識市場狀態的機制，<span class="red">絕不使用未來函數或偷看 NEW3 調參</span>。</li>
        <li><strong>FPGA MVP 規劃：</strong> 在進行硬體獲利宣稱前，規劃簡單的 Rolling/EWMA Gate + 輕量級 TCN 撤單邏輯的硬體落地路徑，專注解決 Latency > 10 的死穴。</li>
      </ol>
    </div>
  </div>
</div></body></html>
"""
(OUT / "06_slides.html").write_text(html_06, encoding="utf-8")

print("All slides 00, 02, 03, 04, 05, 06 generated successfully!")
