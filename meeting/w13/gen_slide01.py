
import base64, pathlib
ROOT = pathlib.Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project\meeting\w13")
DERRIK = ROOT / "___Derrik_results"
OUT = ROOT / "slides"
OUT.mkdir(exist_ok=True)

def b64(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

loss_img = b64(DERRIK / "03_model_classification" / "01_training_loss_plot_annotated.png")
cm_img   = b64(DERRIK / "03_model_classification" / "02_confusion_matrix_percent_test.png")

html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>W13 Slide 01 — 最終模型</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Noto Sans TC",sans-serif;background:#0d1117;color:#e6edf3}}
.slide{{width:1280px;min-height:720px;margin:30px auto;background:linear-gradient(135deg,#161b22 0%,#0d1117 100%);border:1px solid #30363d;border-radius:16px;padding:48px 56px;display:flex;flex-direction:column;gap:28px}}
h1{{font-size:2rem;font-weight:700;color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:14px}}
h2{{font-size:1.1rem;font-weight:600;color:#79c0ff;margin-bottom:10px}}
.tag{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600}}
.tag-blue{{background:#1f4e79;color:#79c0ff}}
.tag-red{{background:#5c1a1a;color:#ff7b7b}}
.tag-green{{background:#0d3d21;color:#56d364}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px}}
.img-wrap{{text-align:center}}
.img-wrap img{{max-width:100%;border-radius:8px;border:1px solid #30363d}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:#21262d;color:#8b949e;padding:8px 10px;text-align:left;font-weight:600}}
td{{padding:8px 10px;border-bottom:1px solid #21262d}}
tr:last-child td{{border-bottom:none}}
.hl{{color:#f0c27f;font-weight:700}}
.red{{color:#ff7b7b}}.green{{color:#56d364}}
.note{{font-size:.78rem;color:#8b949e;margin-top:8px;padding:8px 12px;background:#161b22;border-left:3px solid #388bfd;border-radius:4px}}
.badge-row{{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}}
</style>
</head>
<body>
<div class="slide">
  <h1>Slide 01 — 最終模型：從方向性預測到毒性撤單過濾器</h1>

  <div class="grid-2">
    <!-- 左：模型架構摘要 -->
    <div class="card">
      <h2>模型架構與訓練摘要</h2>
      <table>
        <tr><th>項目</th><th>值</th></tr>
        <tr><td>模型類型</td><td>TCN (Temporal Convolutional Network)</td></tr>
        <tr><td>輸入特徵數</td><td><span class="hl">8</span> 個微觀結構特徵</td></tr>
        <tr><td>觀察視窗</td><td><span class="hl">100 ticks</span></td></tr>
        <tr><td>目標標籤</td><td><span class="tag tag-blue">smoothed_mid_move_3class</span></td></tr>
        <tr><td>訓練資料</td><td>718,562 筆（6 sessions）</td></tr>
        <tr><td>驗證資料</td><td>112,698 筆（1 session）</td></tr>
        <tr><td>測試資料</td><td>233,268 筆（2 sessions）</td></tr>
        <tr><td>最佳 Epoch</td><td><span class="hl">Epoch 30</span></td></tr>
        <tr><td>最低 Val Loss</td><td><span class="hl">0.7814</span></td></tr>
        <tr><td>快照版本</td><td>expC_snapshot (zscore_window=2000)</td></tr>
      </table>
      <p class="note">來源：<code>TXFB6__2026012x.csv</code> × 9 sessions（data/NEW_ALL）<br>
      模型目錄：<code>lt=smoothed_mid_move_3class__fc=8__w=100__H=5__thr=0p5__expC_snapshot</code></p>
    </div>

    <!-- 右：策略定位 -->
    <div class="card">
      <h2>策略定位：不是主動交易，是防禦撤單</h2>
      <table>
        <tr><th>輸出訊號</th><th>動作</th><th>用途</th></tr>
        <tr><td>P(Down) 高</td><td>撤銷買單 (Cancel Bid)</td><td class="green">✔ 防禦毒性下跌</td></tr>
        <tr><td>P(Up) 高</td><td>撤銷賣單 (Cancel Ask)</td><td class="green">✔ 防禦毒性上漲</td></tr>
        <tr><td>P(Stationary) 高</td><td>維持掛單</td><td>繼續做市</td></tr>
      </table>
      <div class="badge-row" style="margin-top:16px">
        <span class="tag tag-red">❌ 非主動方向性交易</span>
        <span class="tag tag-red">❌ 非 Production-ready</span>
        <span class="tag tag-green">✔ 高敏感度撤單觸發器</span>
      </div>
      <p class="note" style="margin-top:14px">模型的 Down/Up Recall ≈ <strong>67%</strong>（捕捉多數不利移動），但 Precision ≈ <strong>2.8%</strong>（市場多數時間在盤整），因此只適合作為防禦性撤單，而非方向性進場。</p>
    </div>
  </div>

  <!-- Loss Curve + Confusion Matrix -->
  <div class="grid-2">
    <div class="card">
      <h2>訓練收斂曲線 (Loss Curve)</h2>
      <div class="img-wrap"><img src="{loss_img}" alt="Loss Curve"></div>
      <p class="note">Epoch 30 為最佳（val_loss=<span class="hl">0.7814</span>），train_loss 單調遞減，無 Overfitting。</p>
    </div>
    <div class="card">
      <h2>混淆矩陣 (Confusion Matrix) — 測試集</h2>
      <div class="img-wrap"><img src="{cm_img}" alt="Confusion Matrix"></div>
      <p class="note">Up/Down Precision≈<span class="red">2.8%</span>（低），Recall≈<span class="green">67%</span>（高）→ 適合撤單觸發，不適合主動交易。</p>
    </div>
  </div>

  <!-- Classification Table -->
  <div class="card">
    <h2>分類報告（測試集：TXFB6__20260127Day + Night，共 233,268 筆）</h2>
    <table>
      <tr><th>類別</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th><th>解讀</th></tr>
      <tr><td>Down</td><td class="red">0.028</td><td class="green">0.694</td><td>0.054</td><td>2,762</td><td>高召回 → 適合撤單觸發，精準度低為代價</td></tr>
      <tr><td>Stationary</td><td class="green">0.987</td><td>0.452</td><td>0.620</td><td>227,711</td><td>主導類別，高精準度</td></tr>
      <tr><td>Up</td><td class="red">0.029</td><td class="green">0.642</td><td>0.056</td><td>2,795</td><td>高召回 → 適合撤單觸發，精準度低為代價</td></tr>
      <tr><td><strong>Macro Avg</strong></td><td>0.348</td><td>0.596</td><td>0.243</td><td>233,268</td><td>—</td></tr>
    </table>
  </div>

</div>
</body>
</html>"""

(OUT / "01_slides.html").write_text(html, encoding="utf-8")
print("Done: slides/01_slides.html")
