# n2ew_era 專題進度簡報——給新加入競賽組員

> 本文字可直接複製貼到 Word。每張圖表的數據解讀與軸/單位說明，都直接寫在該圖的插入標記**下方**（不再集中放在獨立速查表），報告者照著圖往下讀即可，不用跳到別的章節對照。所有數字均直接取自對應的 `.md` 報告或 `.csv` 結果檔，未經本人二次加工估算；少數需要再次確認來源檔的地方會明確標註「待覆核」。

---

## 0. 一句話定位

n2ew_era 是「台指期（TXF）L2 order book 高頻造市報價控制」研究線，目標是用因果（causal，只看當下與過去資訊）特徵去決定要不要報價、報哪一邊，並用「短horizon可執行的序列代理損益（sequential proxy PnL）」而不是「終端報價 EV」來評估好壞。

---

## 0.5 給組員的直接答案：這個方法現在到底能不能賺錢？

這節是專門寫給第一次看這個專案的組員的，目的是用最少的專業術語把「能不能賺錢」「怎麼測的」「證據是什麼」「哪裡還不確定」一次講清楚。後面所有章節都是這節每一句話的證據來源，可以對照查證。

**先講結論：在目前測過的條件下，答案是「能，但有但書」。** 有但書的意思不是在迴避問題，而是這個領域（高頻造市）本來就沒有「測一次就永遠成立」這種事，下面把「能」跟「但書」分開講。

**「能賺錢」的證據是什麼：**

1. **唯一一次真正算數的最終測試結果是正的。** 整個專案把資料切成五段，其中一段（NEW5）被鎖起來，規則是「只能在最有信心的時候，對一個候選模型，測試一次，測完就封存，不能因為結果不滿意就重測」。這就像考試只能交一次卷，不能寫完看分數不滿意再擦掉重寫。目前唯一交出去的這一卷，是 HGB（一種梯度提升樹模型）的造市規則，結果是賺錢的（序列代理損益 = 893,555，正值，詳見 §3.5）。這是目前唯一一個「沒有被回頭微調過、貨真價實的out-of-sample結果」。
2. **就算把「同樣的規則拿去不同時間段重測」也是賺錢的。** 在另外兩段平時看得到但不能拿來調參的資料（NEW3、NEW4）上，無論是簡單規則、ML模型、還是PHC序列模型，序列代理損益全部是正值，而且ML都贏過簡單規則、PHC都贏過ML——這個排名在我們測過的9種不同執行假設組合下都沒有變過（詳見§2.4新增段落、§1理由二）。
3. **賺錢的方式合理，不是統計巧合。** §2.4會用人話把PnL怎麼算的整個機制講一遍——核心是「猜會不會成交」跟「猜划不划算」是兩條完全分開、互不污染的判定線，這保證了PnL數字不是「先知道未來再回頭挑」這種作弊算法生出來的。

**「但書」是什麼，必須老實講：**

1. **PHC（目前看起來表現最好的模型）還沒有交那張只能交一次的考卷。** 也就是說PHC目前的"贏"，都還是在「平時看得到」的資料上測出來的，還沒經過最嚴格的那一關。為什麼還不給它交卷，§1有三個具體理由，全部跟「會不會賺錢」這個問題本身有關，不是單純保守。
2. **赚多少、赚得有多接近「理論上限」，還沒有一個誠實、嚴格的單一數字。** 詳見§2.6——上限這個東西本身要拆成三層才講得清楚，最嚴格的那一層（同時限制出手次數、又要真的穿過成交模擬器）目前還在補（見下方Task C的最新進度，仍未完全收斂）。
3. **手續費一旦假設提高到5倍，簡單規則會由賺轉賠**（§2.4結尾）——這代表目前的"賺錢"結論對手續費假設是敏感的，不是在任何成本水準下都穩。
4. **硬體（HLS/FPGA）那邊，目前驗證到「邏輯設計層面正確、時序通過」，但還沒有真的插上實體板子跑過。**（§3.10）

**軟體驗證現況一句話總結：** 模型/PnL計算/回測這條鏈，每一步都有自己的稽核報告可以逐行對照原始csv（§2.7），目前已知的方法論缺口都列在§4，沒有任何缺口是被刻意隱藏或淡化的。

**HLS現況一句話總結：** 唯一完整跑過csim→csynth→cosim→Vivado P&R全流程、且時序通過（timing met）的，是PHC-lite的seed2版本；其他種子（seed0/seed1）的硬體驗證目前還沒有真正執行過（這是本輪審查新發現的落差，詳見§3.10的更新段落，跟舊版"已知缺口尚待補"的措辭不同的是：這次確認了連嘗試補的程式碼都還不存在）。

---

## 1. 現在該給 PHC 開鎖 NEW5 嗎？——Reviewer 建議：**還不該**

這是被問到一定要能完整作答的問題，答案不是單純「等久一點」，而是有三個具體、可驗證的理由：

**理由一：03B6→04B1 的 ML 輸贏逆轉，到現在沒有任何報告解釋根本機制（詳見 §3.4）。** PHC 系列模型是在「TXF-only」資料上開發的，跟 04B1 的 ML 模型站在同一個資料基礎上。如果我們連「篩掉非TXF商品為什麼會讓 ML 從輸轉贏」都還沒拆解清楚，就無法排除 PHC 目前在 NEW3/NEW4 上贏 HGB／queue rule，是不是踩在同一個還沒搞懂的機制上——換句話說，現在的信心程度撐不起「打開只能用一次的鎖」這個決定的風險。

**理由二：oracle 約束（inventory cap、fill model）敏感度——本輪已補測，結果好消息壞消息各一個（詳見 §2.4、§2.6）。**

好消息：Codex 已完成 9 組合（`inventory_cap∈{1,2,3}` × `fill_model∈{depth_haircut, queue_side_haircut, conservative_l2_stress}`）的敏感度掃描（`AUDIT_oracle_assumption_sensitivity_sweep.md`）。在全部 27 個（split×假設組合）OOS 觀測點上，排名 `PHC > HGB > queue rule` 百分之百一致，沒有一個組合反過來。這比「只測過一組假設」是實質的進步。

壞消息（必須老實講，否則答辯時會被抓到誇大）：這個掃描的「3 個 inventory_cap」實際上只有 2 種真正不同的狀態——`cap=2` 跟 `cap=3` 在所有組合下數字完全相同（連小數點都不差），代表這份資料在目前的成交頻率下從未真正用滿到 cap=2 以上，所以嚴格來說這次掃描測的是「cap=1 vs cap≥2」，不是真正獨立的三組假設。對外溝通時要把「9組合敏感度掃描」改講成「3種成交機率模型 × (cap=1 / cap≥2) 兩種庫存狀態，排名在所有組合下一致」，這樣才不會被問到「cap=2跟cap=3數字一樣，你是不是隨便填的」時答不出來。

另外，承接這個掃描的下一步任務（Task C：「同時限制出手次數、又要真的穿過成交模擬器」的最嚴格oracle）目前 status 是 **mixed**，不是完全收斂：在 development（train）切分上，PHC 的目標成交數量始終對不齊（誤差 21.4%），這代表「同時滿足兩個最嚴格條件」這個方法本身在高選股密度的情況下有已知的失效模式，OOS（val/NEW3/NEW4）九個cell倒是全部精確匹配。這個方法論缺口還沒完全補起來，是繼續暫緩開NEW5鎖的合理理由之一。

**理由三：記憶消融結果本身留了一個尚待解釋的疑點。** `fixed_decay_0.9`（固定衰減）幾乎打平了「學習到的 decay」（§3.7 表格：1,927,640 vs 1,927,170，差距僅 470），這代表 PHC-lite 目前驗證到的優勢，可能主要來自「序列順序 + 閘控狀態」這個結構本身，而不是「每個 channel 各自學到的衰減值」。在還沒搞清楚「PHC 贏的具體來源是什麼」之前去開 NEW5，風險是：萬一 NEW5 結果不如預期，會很難回答審查者「所以到底是什麼東西在起作用」。

**不構成阻擋 NEW5 的因素（誠實列出，避免過度保守）：** HLS 跨種子穩定性未測、PYNQ-Z2 未上板——這兩項屬於**硬體驗證**範疇，跟「軟體層面 PHC 策略是否該動用 NEW5」是兩個獨立決策，不應該混為一談去拖延或加速 NEW5 時程。`phc_stp_v2_cosine_lr`（05C4 seed2）已通過六項採用門檻（adoption gates）這件事，回答的是「該不該把它當成 PHC 系列的代表候選」，不是「該不該開 NEW5 鎖」——這是兩個不同層級的問題，常被混著問，要分開回答。

**建議的具體順序（本輪更新）：** §3.4 的 reversal root cause 診斷跟 oracle 約束敏感度掃描兩項都已經做完一輪，但結論都還沒到「乾淨關閉」：前者官方自評是「diagnosed but inconclusive」（診斷出真實的ML決策邊界差異，但不能完全排除其他解釋）；後者排名結論穩健、但掃描廣度本身有名實不符的問題（見上方理由二）。在開NEW5鎖之前，建議再補兩件事：(1) 把「matched-count + simulator oracle」（Task C）在 development split 上的 21.4% 誤差問題解掉，讓最嚴格版本的oracle在所有split上都收斂；(2) 把理由二裡「實際只有兩種庫存狀態」這件事的措辭對外修正清楚，避免造成不實的嚴謹度印象。這兩項都不會碰NEW5，做完才是開鎖的合理時間點。

---

## 2. 目前打算做到的東西（專案目標）

### 2.1 為什麼要重置（new_era → n2ew_era）

`PROJECT_RESET.md` 列出三個重置理由，三者都已用原始檔案核對過（非事後合理化）：

1. **評估目標用錯**：舊代用「終端報價 EV」當主要指標，這只能算「如果這筆報價一路抱到收盤會怎樣」，不能反映造市策略真正逐筆執行的損益。新代改用「sequential proxy PnL（causal-fill excess PnL）」。
2. **資料切分有洩漏風險**：舊代的切分方式可能讓未來資訊滲入訓練。新代重新定義並**凍結**了 NEW1-NEW5 五段資料，角色固定不可逆。
3. **比較基準不完整**：新代補齊完整階層：No quote（不報價，恆為0）→ Old baseline（舊規則）→ Simple L2 rule（簡單規則）→ Best Val non-ML rule（驗證集選出的最佳非ML規則）→ ML 全動作模型。

### 2.2 凍結資料切分角色（這是不可逆規則，被問到一定要答對）

| 角色 | 資料 | 用途 | 規則 |
|---|---|---|---|
| development | NEW1 + NEW2（對應檔案內常標記為 train/val） | 訓練 + 驗證 | 可反覆迭代調參 |
| used_oos_diagnostic | NEW3 + NEW4 | 開發期診斷用 out-of-sample | 可以看結果，但**不能拿來選模型或調參** |
| final_holdout_locked | NEW5 | 最終單次測試 | 用過一次就鎖定，**終身只能用一次** |

目前**只有 04C 階段對 NEW5 開過一次鎖**（HGB 候選的最終測試）。PHC 系列模型尚未對 NEW5 開鎖——詳見 §1 的完整理由。

### 2.3 只考慮哪一段時間？——資料範圍白話說明

整條線的原始資料來源是台指期（TXF）的 L2 order book 逐筆事件，總計 **14,256,951 筆原始事件**（`01_full_l2_action_universe_builder` 階段的輸入規模，依 session 切成 train 6 個交易日、val 1 個交易日、test 2 個交易日、new4 4 個交易日；NEW5 另外凍結、本報告所有「已揭露」數字都不包含 NEW5，因為它只能在最終一次性測試時打開）。換句話說，回測**不是**對「全部歷史」做一次平均，而是按交易日（session）切成完全分開的時間段，development 階段只能看 train+val 那幾天，NEW3/NEW4 那幾天平時看得到但不能拿來調參，NEW5 那幾天平時完全看不到。

### 2.4 現在的 PnL 是什麼？代理 PnL 怎麼算？怎樣算賺/賠——白話版回測機制

這一段把舊版「2.1」用人話重講一次。

**第一步：「狀態」是什麼。** 每一個可報價的時間點（quote-eligible event）是一個狀態，狀態描述了當下的 L2 order book（價差、佇列不平衡、深度、過去事件率/波動率等，全部只用「現在與過去」的資訊，絕不偷看未來）。

**第二步：「動作」是什麼。** 在每個狀態，策略要選一個動作：報 bid、報 ask、或不報價（no_quote）。

**第三步：PnL（損益）怎麼算——這是核心公式：**

```
若報 bid： excess_bid_pnl_H = (未來第H個事件的中價 − 現在的中價) × 點值 − 手續費
若報 ask： excess_ask_pnl_H = (現在的中價 − 未來第H個事件的中價) × 點值 − 手續費
若不報價： excess_pnl_H = 0
```

這裡的「未來中價」**只當成事後算分的標籤**，從來不會被當成特徵餵給模型或規則去做決策——這就是「causal（因果）」的意思：決策永遠只看當下，算分才看未來。叫它「代理 PnL（proxy PnL）」是因為三個地方跟真實成交不一樣：(1) 用一個機率模型猜「這筆報價會不會被成交」，不是真實交易所逐筆撮合回放；(2) 沒有考慮自己下單對市場的衝擊；(3) 點值是固定換算係數，沒算進真實世界的滑價、延遲等其他摩擦成本。

**第四步：進場依據——「會不會成交」怎麼判定，這是最常被誤會的一步。** 系統有三種成交機率模型（`depth_haircut`、`queue_side_haircut`、`conservative_l2_stress`，後者最嚴格保守，平均成交機率只有約 9–14%，另兩種約 35–53%）。**判定成交與否的依據,只用「現在」的 L2 深度與佇列位置（跟訓練集的中位數比較常態化），絕對不會用未來的中價、未來的 PnL、或「這筆交易划不划算」來反推要不要讓它成交。** 這是整套回測最重要的紀律，保證「能不能成交」跟「賺不賺錢」是兩條獨立的判定線，不會互相污染。

**第五步：出場依據。** 一旦進場（被判定成交），有兩種結束方式：(a) 正常情況下，等到第 H=20 個事件後**強制平倉**，用那個時點的中價算最終 PnL；(b) 如果倉位已經到上限（`inventory_cap=1`，同時最多只能持有一個單位的曝險），新的報價意圖會被擋下變成 `forced_no_quote`（想報但不能報），這個被擋下的次數也會被記錄下來，用來檢查策略是不是因為倉位限制錯失了很多機會。

**第六步：怎麼算賺/賠。** 每一筆「真的成交了」的交易，看它在出場那一刻的 `excess_pnl_H` 是正還是負——正的算 win，負的算 loss。`win_rate` 是贏的交易數除以總成交交易數。注意：win_rate 不等於賺不賺錢的全貌，因為贏的金額跟輸的金額大小不一定對稱（見下例）。

**實際數字範例（NEW3，`side_by_queue_imbalance_abs_0p1` 規則，`conservative_l2_stress`/cap=1/H=20）：** 序列代理總損益 `sequential_excess_pnl = 5,466,860`；成交次數 `fill_count`；勝率 `win_rate = 0.4406`（44.06%，本身低於五成）；平均每筆交易損益 `avg_trade_excess_pnl = 19.0461`（正值）。**這組數字示範了一個重要觀念：勝率不到五成，但平均每筆還是賺錢，因為贏的時候平均贏得比輸的時候平均輸得多**——這正是造市/動能類策略常見的損益形態，不能只看 win_rate 判斷策略好壞。另外要主動揭露：在 5 倍手續費的壓力測試下（Fee-5 strict view），這條規則在三個診斷切分上全部轉為淨負值——這是目前報告裡的一個重要穩健性警示，手續費假設的敏感度不能被忽略。

[插入圖表：`n2ew_era\02_short_horizon_sequential_proxy_baselines\figures\validation_policy_pnl_heatmap.png`（橫軸：候選策略名稱，無明確程式標籤、依熱力圖列排序；縱軸：在驗證集 NEW1+NEW2 上的短horizon序列代理PnL，單位是模擬代理損益點數，**不是真實台幣**）。**這張圖在說什麼**：把所有候選的非ML規則在驗證集上的PnL一次性排出來，用顏色深淺比大小，方便一眼看出哪個規則/horizon組合在val上表現最好——但提醒：這張圖只反映val結果，是「選擇」用的圖，不是「最終結論」的圖，被選中的規則還要再過NEW3/NEW4跟matched-random檢驗才算數。]

[插入圖表：`n2ew_era\02_short_horizon_sequential_proxy_baselines\figures\random_baseline_distributions.png`（隨機對照組matched-random PnL的分布圖，橫軸為PnL數值區間、縱軸為出現頻率；用來目視檢查策略PnL是否落在隨機分布的尾端外側，亦即是否顯著優於隨機。）]

### 2.5 為什麼表格裡兩筆「val PnL」會一樣？

被指出的這兩筆：

```
side_by_queue_imbalance_abs_0p1  2,040,260  NEW3  5,731,540  True
side_by_queue_imbalance_abs_0p1  2,040,260  NEW4  2,075,260  True
```

`val` 欄位（2,040,260）會一樣，是因為**這兩筆根本是同一個「驗證集選擇事件」**——`side_by_queue_imbalance_abs_0p1` 這個規則只在 development（NEW1+NEW2／val）上被選中**一次**，產生**一個**固定的 val PnL 數字。`AUDIT_evaluation_methodology_external_validation.md` 這份方法論外部驗證稽核，要做的事是「拿這一個已經凍結的 val 結果，分別去跟 NEW3 跟 NEW4 兩個不同的 frozen 診斷切分比對正負號是否一致」。所以表格裡同一個策略出現兩筆，是因為**比對目標（NEW3 vs NEW4）不同，不是 val 數字被重複算了兩次**——val 那欄理應永遠固定，frozen split 跟 frozen pnl 那兩欄才是真正在變動的部分。這不是資料錯誤，是稽核表格本身的設計：一個驗證選擇事件 × 兩個獨立的frozen比對目標 = 兩行記錄。

### 2.6 Oracle headroom——先講清楚「oracle」是什麼，再回答「差距是不是策略笨」

**什麼是 oracle？為什麼要弄一個「作弊版」的對照組？** oracle 不是另一種交易策略，是研究上常見的一種**理論上限估計**：假設有一個全知者，能完全看到未來（每個時間點報bid或ask之後實際會賺多少都先知道），然後回頭挑最賺錢的那些動作來做。這種「先看到答案再選」在真實世界做不到（等於用未來資訊下單，是作弊），所以它的PnL本質上是一個**天花板**，不是一個「可達成的對手」。用 oracle 的PnL去除真實策略的PnL，得到的比例叫 **capture ratio（捕捉率）**，意思是「真實策略拿到了理論天花板的幾成」。

**重要提醒，從這裡開始就要分清楚兩個完全不同的問題，下面會分別回答：**

> 問題A：「PHC贏HGB、HGB贏簡單規則」這個排名，本身可不可信？
> 問題B：「捕捉率只有13.8%/18.6%」這個絕對數字本身，代表策略還有很大改進空間嗎？

這兩個問題証據強度不一樣，舊版文件把它們混著回答，容易讓人誤以為「捕捉率低」這件事已經被交代過去了——其實沒有，下面拆開講。

**先拆解 oracle 本身要分幾層算才公平：**

**第一層：`oracle_full_action`（密集/無約束 oracle，理論天花板，不是公平比較對象）。** 對每個狀態直接取 `argmax(excess_bid, excess_ask, 0)`，**完全不受成交機率、不受倉位上限約束**——它假設「只要划算就一定能報、一定能成交、想報多少次都行」。NEW5 上這個數字是 84,517,995（selection_rate=0.708，意思是它選擇報價的狀態占了70.8%）。**這個數字不該被拿來跟任何真實策略直接比，因為它根本沒有遵守跟真實策略一樣的執行限制——拿它算出來的「21.5倍」混雜了兩件事：策略真的不夠聰明、跟策略被迫遵守的執行限制（倉位上限、成交機率），無法從這一個數字拆開。**

**第二層：causal-fill sequential oracle（同樣的成交模擬器+倉位限制下的 oracle）。** 把上面那個密集oracle的「動作選擇」直接餵進跟真實策略一樣的 `conservative_l2_stress`/`inventory_cap=1`/`H=20` 模擬器跑一遍，得到 3,923,130——這個數字才是「遵守一樣的執行規則下，oracle能拿到多少」，但它選的動作數量（selection_rate=0.708）跟真實策略選的動作數量不一樣，所以還不是嚴格的同條件比較。

**第三層：matched-count oracle（限定跟真實策略「選一樣多次」的 oracle，比第二層更公平）。** 把oracle的動作排序、限制只能選**跟真實策略一樣多次的動作數**，再去比PnL。這才是控制了「出手次數」這個變數之後的對照：

| policy | policy_pnl | matched-count oracle_pnl | capture_ratio |
|---|---:|---:|---:|
| Simple L2 規則（`side_by_queue_imbalance_abs_0p1`） | 11,439,800 | 83,102,000 | 13.8% |
| ML（HGB，`ml_hgb_regressor_shallow_convergence_selected`） | 15,648,100 | 83,916,600 | 18.6% |

**第四層：matched-count + simulator oracle（同時控制「出手次數」+「成交模擬器」，目前最嚴格的版本）。** 第三層雖然控制了出手次數，但它選出來的那些動作，並沒有真的再丟回成交模擬器確認「這些動作在同樣的成交機率/倉位限制下真的能照原順序被成交」——也就是說第三層的oracle其實還是有一點「先知道結果才回頭挑」的味道。第四層的做法是：先看真實策略在模擬器裡**實際成交了多少次**（不是選了多少次，兩者因為成交機率不是100%而不同），再去找一個oracle，讓它在同一個模擬器下**剛好也成交那麼多次**，這樣才是真正同時控制了「出手次數」跟「執行限制」兩個變數。代表性假設選用跟既有報告一致的 `conservative_l2_stress` + `inventory_cap=1`：

| split | 政策 | 政策實際成交數 | oracle最終成交數 | 成交數誤差 | 政策序列PnL | oracle序列PnL | capture ratio（對最嚴格oracle） |
|---|---|---:|---:|---:|---:|---:|---:|
| val | Simple規則 | 6,416 | 6,416 | 0% | 103,260 | 883,050 | 11.87% |
| val | HGB | 6,231 | 6,231 | 0% | 142,435 | 895,440 | 16.68% |
| val | PHC | 6,891 | 6,891 | 0% | 171,835 | 890,490 | 19.04% |
| NEW3 | Simple規則 | 13,821 | 13,821 | 0% | 303,700 | 2,621,550 | 13.30% |
| NEW3 | HGB | 13,955 | 13,955 | 0% | 411,090 | 2,644,305 | 17.91% |
| NEW3 | PHC | 17,907 | 17,907 | 0% | 611,095 | 2,629,305 | 23.23% |
| NEW4 | Simple規則 | 3,820 | 3,820 | 0% | 72,415 | 1,007,170 | 9.46% |
| NEW4 | HGB | 3,629 | 3,629 | 0% | 97,680 | 1,016,380 | 13.04% |
| NEW4 | PHC | 4,237 | 4,237 | 0% | 129,160 | 1,011,325 | 16.13% |
| train（development，僅供方法驗證，不算對外證據） | PHC | 7,532（目標） | 5,920（實際達成） | **21.4%** | 281,935 | 731,515 | 38.5% |

val/NEW3/NEW4 九個OOS格子，oracle最終成交數跟政策實際成交數做到完全一致（誤差0%），這九個capture ratio數字（9.46%～23.23%）是目前能拿出來最嚴格的版本——比第三層的13.8%/18.6%更嚴格，因為這次連「成交模擬器」都真的跑過了，不只是排序後直接砍數量。**development（train）那一格還沒收斂**：PHC在train上想成交7,532次，但oracle最多只湊出5,920次（誤差21.4%）——代表這個「用二分搜尋去湊出剛好成交數」的方法，在選股密度特別高時會卡住，這個方法本身還沒完全成熟，不能宣稱已經定案。

---

**現在回答問題A：排名（PHC>HGB>queue rule）可信嗎？——可信，這部分有扎實證據。**

Codex 做了一次**敏感度掃描**：把 oracle 計算依賴的兩個假設——`inventory_cap`（倉位上限，測了1/2/3）跟 `fill_model`（成交機率怎麼算，測了三種：`depth_haircut`、`queue_side_haircut`、`conservative_l2_stress`）——交叉組成9種組合，每一種組合下都重新算一次三個策略的排名。**敏感度掃描要回答的問題是：「如果我們當初選的假設換一組，結論還會不會一樣？」**這是用來檢查「結論是不是只在某一組特定假設下才成立、換個假設就翻盤」的標準作法。

結果：27個（split×假設組合）OOS觀測點上，排名`PHC>HGB>queue rule`百分之百一致，沒有一個組合反過來。這個排名結論是穩的。

但有一個要老實講的限制：這次掃描的「3個inventory_cap」實際上只有2種真正不同的狀態——`cap=2`跟`cap=3`在所有組合下數字完全相同（連小數點都不差），代表資料在目前的成交頻率下從未真正用滿到cap=2以上。所以嚴格來說，「排名換了假設還穩」這句話，目前只在`fill_model`這個維度上有完整證據，`inventory_cap`維度等於只測了「cap=1 vs cap≥2」兩種狀態，不是真正獨立的三組。

**現在回答問題B：13.8%/18.6%（或第四層的9.46%～23.23%）這個絕對數字偏低，代表策略還有大改進空間嗎？——老實說：目前沒有證據可以回答這個問題，不能說「已經排除」。**

這裡要特別提醒：**敏感度掃描跟第四層oracle，都不是用來回答問題B的證據**——敏感度掃描證明的是排名穩（問題A），第四層oracle算出來的捕捉率（9.46%～23.23%）跟第三層的13.8%/18.6%相比，**沒有變高，部分格子甚至更低**。也就是說，目前測過的兩個方向，都沒有給出「真實捕捉率其實比看起來高」的支持證據。

唯一能講的、結構上合理的一點是：**跟「全知oracle」比，捕捉率本來就會偏低，這是這種比較方法天生的特性，不是這個專案特有的問題**——好比拿股票策略去跟「事後回頭看買在最低點賣在最高點」的對照組比，捕捉率通常也是個位數到二十幾趴，這不直接等於選股能力差，只是代表「跟全知者比」這個基準本身就很嚴苛。但這只能說明「不能因為捕捉率低就直接判定策略差」，**不能說「策略其實夠好，只是oracle太嚴苛」——這兩句話不一樣，後面這句目前沒有證據支持**。

**老實的結論：**

1. 排名可信（問題A有扎實證據），13.8%/18.6%偏低也是真的事實，不用迴避。
2. 但「偏低是不是因為策略真的有改進空間」（問題B）目前**沒有答案**，是一個開放問題，不是已經處理過、只是還沒寫清楚——之前的版本把問題A的證據拿來回答問題B，是邏輯上的混用，這次已經拆開講清楚。
3. 具體可用、應該對外講的數字：用第四層 OOS 九格的 9.46%～23.23% 取代舊版「21.5倍」或單純的13.8%/18.6%，因為它是目前控制變數最嚴謹的版本；但要附帶兩個但書——(a) train split尚未收斂（21.4%誤差），方法本身還沒完全成熟；(b) 這組數字目前只能回答「排名可信」，**不能**拿來回答「策略還有多少改進空間」。
4. 下一步要做的不是再算一個更嚴格的oracle（那只會繼續回答問題A），而是去直接分析「策略沒選到的那些高價值動作，是因為特徵看不出來、還是模型容量不夠、還是門檻設定問題」——這才是真正回答問題B的路徑，目前還沒有人做。

### 2.7 圖表裡的 PnL 數字怎麼來的——通用追溯路徑

所有圖上的 PnL，追根究柢都是同一條鏈：（1）`01_full_l2_action_universe_builder` 產生每個狀態×每個候選動作的 `pnl_H{horizon}` 標籤（用 §2.4 的 excess PnL 公式，未來中價只當標籤）→（2）各階段的 `run_phaseXX_*.py` 或 `common_phaseXX.py` 把這些標籤跑過策略選擇邏輯（哪個狀態選哪個動作）與序列模擬器（成交機率＋倉位上限＋H事件後平倉），輸出累積/總計 PnL 到 `results/*.csv`（例如 `frozen_baseline_test_new4.csv`、`selected_phase02_policy_val_rows.csv`）→（3）對應的 `0X_generate_phaseXX_figures.py` 直接讀這些csv畫圖，沒有額外的二次運算。換句話說，**圖表上的每一個PnL數字都能在某個 `results/*.csv` 裡找到一模一樣的原始數字**，這也是這份簡報堅持每張圖都附上來源檔案路徑的原因——可以隨時打開csv逐行核對。

---

## 3. 目前的成果與發現（聚焦在實際拿到的結果，階段代號只作為溯源標籤）

### 3.1 動作宇宙建構與基準規則

`01_full_l2_action_universe_builder` 從原始 TXF L2 order book 全量事件，建構出「全動作宇宙」——每個可報價的時間點都列出 bid/ask 候選動作並附上短horizon標籤，保證特徵只用當下及過去資訊。`02_short_horizon_sequential_proxy_baselines` 在 H=1,5,10,20,50,100 多個horizon下評測 No quote、Old baseline、Simple rule。

[插入圖表：`n2ew_era\01_full_l2_action_universe_builder\figures\action_count_waterfall.png`（橫軸：四個切分 new4/test/train/val；縱軸：rows/actions，log scale，三個柱子分別是 `raw_rows`、`quote_eligible_rows`、`candidate_action_rows`）。**這張圖的真實數據與篩選步驟**——以train為例：原始事件 `raw_rows = 3,593,426` → 經過book有效性檢查（valid_book_rows，剔除損毀/不合法的L2快照）→ `quote_eligible_rows = 3,592,764`（保留率 99.98%，剔除約660筆不可報價的狀態，例如跨價/鎖價或session邊界）→ 每個quote-eligible狀態固定展開成3個候選動作（bid/ask/no_quote）→ `candidate_action_rows = 10,778,292`，剛好是 quote_eligible_rows 的3倍（`action_multiplier_vs_quote_eligible = 3.0`，四個切分全部精確吻合，無誤差）。其餘切分：val（raw=563,593 → eligible=563,477 → action=1,690,431）、test（raw=1,166,541 → eligible=1,166,321 → action=3,498,963）、new4（raw=2,197,106 → eligible=2,196,640 → action=6,589,920）。這張圖說明的是「資料量的篩選漏斗」，不是PnL，重點是證明篩選步驟保留率極高（>99.9%）且×3展開精確無誤，不是隨意丟資料。]

[插入圖表：`n2ew_era\01_full_l2_action_universe_builder\figures\feature_distribution_by_split.png`（每個子圖橫軸為切分名稱train/val/test/new4，縱軸為對應特徵的數值範圍，箱形圖，已做1/99百分位裁剪去極端值）。**這張圖支持的結論**：六個因果特徵中，`queue_imbalance`（佇列不平衡）在四個切分上的中位數都接近0、標準差都在0.43上下，分布相當一致；但**`volatility_past_50`（過去波動率）在new4上的均值是0.222，明顯高於train的0.157**，`event_rate_past_50`（過去事件率）在test上的均值只有114.7，明顯低於train的154.2——這代表切分之間**不是完全同分布**，new4那幾天市場明顯更活躍/更波動，test那幾天明顯更安靜。這張圖的真正用途不是證明「一切都一樣很安心」，而是誠實揭露切分間存在regime差異，這正是為什麼§2方法論要額外做LOSO穩健性檢查跟PSI/KS drift稽核的理由——如果只看這張圖就以為分布完全一致，是誤讀。]

[插入圖表：`n2ew_era\02_short_horizon_sequential_proxy_baselines\figures\selected_policy_equity_curves.png`（橫軸：依時間順序downsample後的事件索引；縱軸：累積短horizon代理PnL曲線）。**這張圖具體說明什麼，要先講清楚它畫的是哪個策略**：02階段驗證集選出的main策略是 `ask_all`（horizon=H1）——也就是「每一個可報價狀態都無條件報ask」，完全沒有任何選股/選邊邏輯。它在val上selection_rate=1（每次都報）、total_pnl=116,712,000、win_rate=99.31%；在test/new4上同樣selection_rate=1、win_rate約99%、PnL分別是243,492,000與533,655,000。**正確的解讀方式**：這張曲線之所以幾乎單調上升且勝率接近100%，是因為在極短horizon（H=1，只看下一個事件）下，無條件報ask主要在機械性地收半個價差（spread），這是「地板基準」——任何有意義的策略都應該要比這條線更好或至少不差，這張圖本身**不是**用來展示「我們的方法很厲害」，而是用來建立比較基準的下限。被審查者如果問「這張圖證明了什麼alpha」，正確答法是「沒有，這是無選擇性的機械基準曲線，用來確認後續規則/模型有沒有真的比白動報價更好」。]

### 3.2 嚴謹度升級：02D 的 PASS 被 02E 用更嚴格方法推翻

02D 原本用較寬鬆的方法判定基準規則通過稽核；02E 改用更嚴格的「序列庫存代理（sequential inventory proxy）」重新檢驗——真正模擬持倉限制與逐筆平倉，結果原本的結論被推翻。新組員被問到「為什麼結論不一樣」時，要答「後面那版用了更貼近真實執行限制的模擬方式，發現前一版的寬鬆假設下會過度樂觀」。

[插入圖表：`n2ew_era\02d_final_strict_baseline_audit_package\figures\queue_stress_heatmap_selected_policy.png`（橫軸/縱軸為佇列壓力測試情境參數組合，色塊深淺代表對應PnL高低；用來檢查策略在不同佇列壓力假設下是否依然穩健。）]

[插入圖表：`n2ew_era\02d_final_strict_baseline_audit_package\figures\fee_sensitivity_selected_policy.png`（橫軸：假設手續費水準的倍數；縱軸：策略PnL；用來看策略對手續費成本假設有多敏感——這跟§2.4提到的Fee-5壓力測試是同一類分析。）]

### 3.3 ML特徵就緒與簡單ML基準

`03A` 定義 excess PnL（扣除機械性半價差捕捉後的「超額」損益，用來判斷是不是真的有預測能力）並做特徵就緒診斷；`03A5/03A6` 做了根因稽核（特徵因果性、切分重複性、session邊界洩漏），全部判定「未發現洩漏」；`03B6` 解決了HGB（histogram gradient boosting）的收斂問題（三個不同max_iter全部停在n_iter=121，已用`AUDIT_identical_n_iter_121.md`確認是early-stopping自然結果，不是隱藏硬編碼）。

[插入圖表：`n2ew_era\03a_ml_readiness_excess_markout\figures\bucket_lift_queue_imbalance.png`（橫軸：佇列不平衡指標分十分位的bucket；縱軸：每個bucket的平均excess_pnl_H20，單位為H=20horizon下的超額代理PnL）。這張圖用來確認佇列不平衡跟未來超額損益之間是否有單調關係——如果bucket越極端PnL越好/越差呈現一致方向，代表這個特徵確實帶有可用的預測訊號，不是隨機雜訊。]

[插入圖表：`n2ew_era\03b6_hgb_convergence_closure\figures\hgb_val_policy_pnl_by_iter.png`（橫軸：HGB的max_iter訓練輪數上限；縱軸：驗證集上的總excess PnL H20）。這張圖證明三個不同的max_iter上限（80/120/更高）收斂到幾乎同一個validation PnL，佐證§3.3提到的「n_iter=121是自然收斂，不是調出來的」。]

### 3.4 最大的方法論缺口：03B6「ML輸給簡單規則」被04B1的TXF篩選逆轉——**本輪已再次檢視，缺口仍未補起來**

03B6階段（混合商品universe）的結論是ML模型打不過簡單規則。04A2階段發現資料中混入了非TXF的其他商品（產品universe污染），於是04B（`TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT.md`）重新只用TXF商品做開發，結果04B1（`TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT_PATCHED.md`）顯示ML模型**反而打贏**了簡單規則（val/NEW3/NEW4三個切分上`beats_matched_random_95p=True`，causal-fill序列評估上`ml_beats_or_matches_queue=True`，例如NEW3上ML序列PnL=7,456,860 vs queue rule 5,733,600）。

**本輪重新檢視結果：** 已查過三個最可能藏答案的地方——`07_competition_proposal/PHC_HLS_Competition_Proposal.md`（純HLS硬體提案，沒有提到這個逆轉）、`06f_phc_lite_manual_hls/REPORT_SUPPLEMENT.md`（補充的是定點精度/功耗/WNS margin三個缺口，跟這個問題無關）、`04b1_ml_completion_patch/TXF_ONLY_FINAL_DEVELOPMENT_DECISION_REPORT_PATCHED.md`（只給出「篩選後ML贏了」的最終數字，**沒有**拆解「為什麼篩選前後排名會反轉」的機制）。**結論：這個缺口目前仍是開放狀態，沒有任何現存文件解釋根本機制**，舊版報告的描述準確，必須繼續誠實揭露。

**給 Codex 的實驗 prompt（可直接複製貼給 Codex 執行）：**

```
任務：解釋並驗證「03B6 ML輸給簡單規則」與「04B1 TXF篩選後ML打贏」之間的根本機制（reversal root cause）

背景：
- 03B6（混合商品universe，TXF+其他商品）：HGB action-value模型在val/NEW3/NEW4上的表現劣於simple
  L2 rule（side_by_queue_imbalance_abs_0p1）。
- 04B/04B1（僅TXF商品重新開發）：相同類型的HGB模型（max_iter=120, threshold=0.224194）在
  val/NEW3/NEW4上轉為打贏simple rule，且beats_matched_random_95p=True、causal-fill sequential
  ml_beats_or_matches_queue=True。
- 目前沒有任何報告解釋「為什麼篩選掉非TXF商品後ML的相對排名會反轉」的具體機制。

鐵則（絕對不可違反）：
1. 禁止使用NEW5。本任務全程不得載入、讀取、或以任何方式觸碰NEW5資料。
2. 禁止用NEW3/NEW4做任何模型選擇、調參、門檻選擇——NEW3/NEW4只能用於「事後診斷分析」（產生
   統計量、畫圖），不能反過來改變03B6或04B1已經凍結的模型/門檻設定。
3. 不得重新訓練03B6或04B1已存在的模型；本任務是診斷分析，不是重新建模。
4. 所有分析只能用development（NEW1+NEW2 / train+val）與已存在的frozen NEW3/NEW4 diagnostic
   輸出做事後拆解，不能引入新的特徵或新的模型架構。

具體要做的事（任選其一或多個，依時間允許程度排序）：
A. 特徵分布拆解：比較「混合商品universe」與「純TXF universe」在ML模型最重要的幾個特徵
   （用既有HGB feature_importances_或permutation importance，若03B6/04B1已存的模型物件還在，
   直接load出來看）上的分布差異，特別關注：queue_imbalance、deep_imbalance、spread_ticks、
   event_rate_past_50。輸出：每個特徵在「混合」vs「純TXF」資料下的分布統計表（mean/std/分位數），
   以及這些差異是否能解釋HGB決策邊界的改變。
B. 預測誤差分解：對03B6與04B1兩個模型，分別在「同一份純TXF held-out資料」（不是NEW3/NEW4，
   用development內部切一份額外驗證子集，或如果03B6模型物件還留著，直接重新對純TXF development
   資料跑推論）上算prediction error，看03B6模型是否系統性地對「非TXF樣態」學到了會誤導純TXF
   判斷的pattern（例如：不同商品的tick size、價差結構不同，造成03B6學到的threshold在純TXF上失準）。
C. 規則對照分解：simple rule（side_by_queue_imbalance_abs_0p1）在混合vs純TXF資料上的
   policy_pnl/win_rate本身是否也有變化？如果simple rule本身在兩種資料上表現穩定，但ML模型表現
   大幅變動，這會指向「ML對商品異質性更敏感」這個具體機制；反之如果兩者都變動，則可能是市場
   regime差異而非商品污染本身造成。

輸出格式要求：
- 產出一份AUDIT_03b6_04b1_reversal_root_cause.md，結構仿照現有n2ew_era/000_context/AUDIT_*.md
  系列（開頭status: resolved/open/mixed，結尾誠實結論段落）。
- 所有數字必須來自實際重新計算的程式輸出，不得估算或目視推測。
- 明確寫出「development-only」分析與「NEW3/NEW4事後診斷」分析的界線，每張表/每個數字都要標明
  用了哪個split。
- 如果分析完仍無法定論根因，誠實寫「diagnosed but inconclusive」，不要為了給出答案而過度推論。

常見坑（務必避開）：
- 不要因為要做「對照」就去新建一份「mixed universe development子集」再重新調參——這樣會等於
  重新做一次03B6，失去診斷意義，且可能誤觸碰NEW3/NEW4選模型的紅線。
- 不要把NEW3/NEW4當成驗證新假設的「測試集」——它們只能拿來看「同一個既定假設，在這兩個frozen
  split上是否也呈現同樣的pattern」，不能用來篩選哪個假設講得通。
- 若要load舊模型物件（pickle/joblib），先確認版本相容性，避免用新版sklearn載舊版模型導致
  attribute錯誤而誤判為「找不到特徵重要性」。
```

[插入圖表：`n2ew_era\04b_txf_only_final_development_rerun\TXF_NON_ML_BASELINE_REPORT.md` 與 `TXF_SIMPLE_ML_REPORT.md`——這兩份目前查無獨立 `figures/` 子目錄，證據只在報告內表格，沒有PNG可插，提到這個階段時應直接附表格數字而非聲稱有圖。]

### 3.5 NEW5一次性最終holdout與oracle分析

04C 對凍結的NEW5資料做**唯一一次**最終測試，選中的候選是 `ml_hgb_regressor_shallow_convergence_selected`，causal-fill sequential excess PnL = **893,555**（正值）。oracle headroom 分析見 §2.6 的完整三層拆解。

[插入圖表：本階段報告無獨立圖檔目錄，數字證據以 `NEW5_FROZEN_POLICY_EVALUATION_REPORT.md`、`NEW5_CAUSAL_FILL_SEQUENTIAL_REPORT.md` 表格呈現。]

### 3.6 PHC-lite序列模型

PHC-lite是受PHCSSM論文（arXiv 2604.01295）啟發的簡化遞迴序列模型。**重要紅線**：PHC-lite**不是**完整PHCSSM——只實作了「可學習decay/gate遞迴狀態 + within-timestep multi-transmission loop」，**沒有**spike機制、沒有STDP、沒有原論文的surrogate gradient spiking neuron。這個自我聲明在`05c0_freeze_phc_lite_v1/phc_lite_v1_claim_boundary.md`，任何報告場合都不能被誤讀為「實作了完整論文架構」。

**誠實的動機說明**：嘗試PHCSSM最初是和指導教授閒談得知這個架構，屬於機會性嘗試，技術論證是事後補上的合理化敘事——答辯時建議誠實交代這個時間順序。

PHC-lite三個種子(seed 0/1/2)獨立訓練，跨種子穩定性結果：

| seed | split | causal_fill_pnl | beats_queue | beats_hgb |
|---:|---|---:|---|---|
| 0 | val | 170,310 | True | True |
| 0 | new3 | 488,870 | True | True |
| 0 | new4 | 121,765 | True | True |
| 1 | val | 151,530 | True | True |
| 2 | val | 162,625 | True | True |

**PHC-lite訓練用Huber loss——這是什麼，為什麼用它：** Huber loss 是介於 MSE（均方誤差）跟 MAE（平均絕對誤差）之間的損失函數，公式（程式裡 `delta=1.0`）：誤差絕對值 ≤ delta 時用 `0.5 × error²`（跟MSE一樣，靠近零時是平滑的二次曲線，梯度會隨誤差變小而自然縮小，方便模型做精細校準）；誤差絕對值 > delta 時改用 `delta × (|error| − 0.5 × delta)`（跟MAE一樣，是線性的，梯度不會因為誤差很大而爆炸）。**用在PHC-lite這裡的理由**：訓練標籤是逐事件的短horizon markout PnL，這種金融序列標籤天生有重尾分布——少數劇烈波動的事件會產生極端的PnL值。如果用純MSE，這些少數極端值會在平方之後產生超大梯度，主導整個訓練過程，讓模型過度去擬合這幾個outlier而犧牲一般狀態下的準確度；Huber loss對這些大誤差只給線性懲罰，避免outlier主導梯度，同時對小誤差仍保留MSE的平滑特性，讓訓練穩定。

[插入圖表：`n2ew_era\05a_phc_lite_microstructure_sequence_feasibility\figures\phc_lite_loss_curve.png`（橫軸：Epoch；縱軸：Huber loss，數值越低代表模型輸出跟標籤之間的Huber距離越小）。這張圖用來確認訓練是否收斂——如果loss持續下降後在某個epoch附近趨於平緩，代表訓練已經收斂，可以用驗證集指標挑選最佳epoch。]

[插入圖表：`n2ew_era\05a_phc_lite_microstructure_sequence_feasibility\figures\phc_lite_val_policy_pnl_by_epoch.png`（橫軸：Epoch；縱軸：驗證集上的policy PnL）。這張圖直接拿來選epoch——訓練loss最低不代表policy PnL最好，這張圖才是真正用來決定「在哪個epoch停下來、用哪個checkpoint」的依據。]

[插入圖表：`n2ew_era\05a_phc_lite_microstructure_sequence_feasibility\figures\phc_lite_spearman_by_epoch.png`（橫軸：Epoch；縱軸：驗證集Spearman等級相關係數）。這張圖衡量的是「模型分數的排序」跟「真實標籤的排序」吻合程度，跟PnL是兩個不同維度的指標——Spearman高不保證PnL高，因為PnL還跟「選了多少次、選對的那幾次金額多大」有關，排序對不一定金額對。]

[插入圖表：`n2ew_era\05a_phc_lite_microstructure_sequence_feasibility\figures\phc_lite_policy_comparison.png`（不同策略在val/NEW3/NEW4上的PnL長條圖比較，含no_quote/old_bid_only/queue_imbalance/HGB/phc_lite五種策略）。這張圖是整個PHC-lite章節最重要的單一對照圖，直接視覺化PHC-lite是否在三個切分上都站在最高的那一根柱子。]

[插入圖表：`n2ew_era\05a_phc_lite_microstructure_sequence_feasibility\figures\phc_lite_score_bucket_lift.png`（橫軸：模型分數分桶；縱軸：每個分桶的平均excess PnL）。用來看模型分數是否真的單調對應到更高的實際損益——如果分桶越高分PnL越好，代表模型分數本身是有意義的排序訊號，不是雜訊。]

### 3.7 記憶機制驗證（「孤兒CSV」缺口已封閉）

舊版文件曾標記「memory ablation數字找不到生成腳本來源」的缺口。本輪已用`AUDIT_learned_memory_vs_fixed_window.md`封閉：用既有凍結checkpoint做反事實(counterfactual)推論測試(固定decay=0.9、關閉記憶、序列反轉、時間步打亂):

| split | variant | total_excess_pnl_H20 | Δ vs baseline | spearman |
|---|---|---:|---:|---:|
| val | baseline | 1,927,170 | 0 | 0.312 |
| val | fixed_decay_0.9 | 1,927,640 | +470 | 0.312 |
| val | memory_off | 1,123,940 | −803,235 | 0.229 |
| val | reverse_sequence | −837,330 | −2,764,500 | 0.010 |
| val | deterministic_shuffle | −642,420 | −2,569,590 | 0.022 |

**誠實結論（建議答辯逐字引用）**：PHC-lite並未證明實作了完整的生物遞迴機制，但反轉/打亂時間步會讓相關性崩潰、PnL轉負，證明模型確實用到了序列順序；關閉記憶會讓PnL明顯轉弱但仍為正。固定decay=0.9幾乎打平學習到的decay，所以誠實的說法是「序列順序+閘控狀態本身有價值」，不是「每一個學習到的decay channel都各自必不可少」——這也是§1理由三裡提到「PHC贏的具體來源還需再確認」的證據出處。**主動揭露的限制**：這是反事實推論測試，不是重新訓練式的ablation，不能完全排除「從頭用固定decay訓練可能學到別的補償方式」的可能性。

### 3.8 PHC變體系列（05C1-05E）：Dale's Law、STP、動態增益、hidden維度消融

05C1測試Dale's Law（興奮/抑制神經元符號約束），05C2測試STP（短期可塑性）+動態突觸增益，05C3做200epoch乾淨重跑但結果不明確，05D做oracle gap按波動率/事件率bucket拆解，05E測試把hidden_dim降到8（為了HLS資源可行性）的影響。

[插入圖表：`n2ew_era\05c1_phc_dale_law_ablation\figures\dale_val_policy_pnl_seed0.png`（橫軸：Epoch；縱軸：Dale's Law版本的驗證集policy PnL）。]

[插入圖表：`n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\figures\phc_stp_gain_distribution.png`（動態突觸增益值的分布圖）。用來檢查增益值是否合理收斂在有限範圍內，而非發散到不合理的極端值——這是訓練穩定性的健檢圖，不是PnL圖。]

[插入圖表：`n2ew_era\05d_oracle_gap_reduction\figures\oracle_gap_by_volatility_bucket.png`（橫軸：波動率分桶；縱軸：oracle與實際策略的PnL差距）。用來看高波動情境下headroom是否特別大，是§2.6 oracle拆解的延伸切片分析。]

### 3.9 嚴謹度升級：05C3不明確結果被05C4的cosine LR排程修正

05C3用乾淨200epoch重跑STP變體，結果不明確（趨勢不穩定）。05C4改用cosine learning rate排程後，結果穩定下來，`phc_stp_v2_cosine_lr`成為目前PHC系列的**採用候選**——它在05C4階段通過了seed/threshold驗證集選擇與六項採用門檻(adoption gates)，在NEW3/NEW4 frozen split上維持正值(見§3.6表格)。**NEW5尚未對此候選開鎖（見§1完整理由）**。

[插入圖表：`n2ew_era\05c4_phc_stp_cosine_lr\figures\phc_stp_v2_val_policy_pnl_seed2.png`（橫軸：Epoch；縱軸：最終採用種子seed2的驗證集policy PnL）。這是後續HLS實作直接採用的checkpoint來源，務必記住「seed2」這個細節。]

### 3.10 HLS/FPGA硬體實作現況

**06A（queue rule baseline）**：只做到功能等價驗證，**沒有**真正的synthesis資源數字，報告誠實揭露這個限制。

**06F（PHC-lite手寫HLS實作，對應05C4的`phc_stp_v2_cosine_lr` seed2 checkpoint）**：完整跑過 csim → csynth → cosim → **Vivado P&R（Implementation）**全流程，並且**timing met**：

| 資源 | csynth估計 | P&R實測 | xc7z020預算 |
|---|---:|---:|---|
| DSP | 21%（46/220） | 21.36%（47/220） | 220 |
| LUT | 46.68%（24,836/53,200） | **14.20%**（7,555/53,200，合成邏輯優化後更低） | 53,200 |
| FF | 13.65%（14,518/106,400） | 7.72%（8,217/106,400） | 106,400 |
| BRAM | 2.14%（6/280） | 2.14%（6/280） | 280 |
| Latency | 8,316 cycles ≈ 83.16µs @ 100MHz | 同左（無法pipeline，遞迴模型t+1依賴t的state） | — |

時序：`bd_0_wrapper_timing_summary_routed.rpt` 顯示 **WNS=+0.020ns,TNS=0.000ns**，22,941個endpoint全部過。Cosim比對顯示 25,600筆測試向量、mismatches=0、max_abs_diff=0.00725275(RTL與C++模擬結果幾乎完全一致)。**方向一致性（sign agreement）補充數據**：兩維readout聯合方向一致率99.9375%，16個異號案例全部發生在`|ref|<0.00054`（模型本身幾乎無方向性判斷的時刻），若下游策略設confidence threshold，這些時步會被自然濾掉。**功耗**：Total On-Chip Power 0.357W，Junction Temperature 29.1°C（室溫25°C下），confidence level為medium（內部節點切換率用預設0.5估算，非精確SAIF檔案）。

**這是經過8版迭代(v1→v8)才修好的**：v1原本資源爆炸(LUT 2073%、DSP 2636%)，三個根本原因依序是：(1) C標準`tanh()`生成雙精度浮點核心，耗用約2,048個DSP，修復用65點定點LUT取代；(2) 任何`PIPELINE II=X` pragma都會觸發HLS把partial UNROLL升格為complete UNROLL；(3) 刪除PIPELINE後，HLS全域`pipeline_loops`閾值仍會對小迴圈自動套用pipeline，引發連鎖展開。v8(2026-06-15)加上`#pragma HLS PIPELINE off`明確覆寫全域設定才解決。P&R實測LUT(14.20%)比csynth估計(46.68%)低，是Vivado邏輯優化的正常現象，**不是數字矛盾**。

**WNS margin風險（誠實揭露，不能淡化）**：+0.020ns margin極薄，若P&R在不同seed/Vivado版本下佈線略有差異，可能變負值；建議生產部署降頻至90MHz以取得約+1.13ns的margin，但延遲會增加到約92.4µs（+11%）。**競賽/展示用途下，目前100MHz數字已通過Vivado STA，可作為正式報告數字，但必須附上此margin說明，不能單獨秀+0.020ns而不講風險。**

**06B/06C/06D（額外的串流定點等價性與TCN替代架構探索）**：06B驗證PHC-lite定點化後的動作一致性與causal-fill PnL差異；06C/06D是用hls4ml風格TCN（時間卷積網路）做的替代架構探索，屬於並行的硬體可行性備案，非目前主線採用對象。

[插入圖表：`n2ew_era\06b_phc_lite_streaming_fixed_point_equivalence\figures\phc_fixed_point_action_agreement.png`（橫軸：定點量化精度設定；縱軸：與浮點版本的動作一致率）。用來檢查量化後的決策是否仍與原始浮點模型一致——一致率越接近100%代表量化幾乎沒有改變模型的實際決策。]

[插入圖表：`n2ew_era\06b_phc_lite_streaming_fixed_point_equivalence\figures\phc_fixed_point_causal_fill_pnl.png`（不同定點精度下的causal-fill PnL比較柱狀圖）。]

**仍未解決的三個缺口（主動列出，答辯前務必心裡有底）**：
1. **跨種子HLS穩定性未測——本輪確認這是「完全零進度」，不是「進行中」。** 之前交辦Codex去補seed0/seed1的HLS驗證，這次重新審查`AUDIT_hls_cross_seed_resource_stability.md`後發現：整個專案裡找不到任何seed0/seed1的權重匯出腳本、找不到任何獨立的HLS工作目錄、找不到任何csynth報告檔案——`00_export_weights.py`這支匯出腳本仍然硬編碼指向seed2，完全沒被改動。換句話說，**這次交回來的"完成"只是把已知缺口重新用表格的形式複述一次，沒有新增任何實測證據**。狀態應該標記為「未開始」而不是「已知缺口、待補」，這是本輪審查發現的一個重要落差，必須對組員誠實說明：目前唯一有完整HLS證據的就是seed2，seed0/seed1連嘗試都還沒開始。
2. PYNQ-Z2實機上板未做——csim/cosim/P&R timing met是業界認可的完整驗證鏈，但離「插上板子量到83µs」還有一步，需要Vivado block design整合PS-PL介面。
3. 測試向量可能不具代表性——目前用的25,600組測試向量是「validation split前200個sequence」，不是完整validation set，也不是獨立第二份Python reference重新生成的向量。

**給Codex的補做提示（這次必須真的執行export+csynth，不能再只交一份敘事性的md）：**

```
任務：真正執行PHC-lite seed0/seed1的HLS資源驗證（csynth至少），不要只描述缺口。

背景：n2ew_era/06f_phc_lite_manual_hls/00_export_weights.py目前硬編碼指向
n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt，只匯出了seed2的權重並
完成了完整HLS驗證鏈（csim/csynth/cosim/P&R）。seed0/seed1的訓練checkpoint已存在
（phc_stp_v2_seed0_best_val.pt、phc_stp_v2_seed1_best_val.pt），但從未匯出或合成過。

鐵則：
1. 不得更動seed2既有的HLS設定（pragma、interface、clock）——seed0/seed1必須沿用完全相同的
   設定，唯一變數是權重來源checkpoint，這樣資源數字才有可比性。
2. 不得使用NEW5資料做任何驗證測試向量。
3. 這次的產出必須包含可執行、可重跑的腳本（仿照00_export_weights.py，可以是
   00_export_weights_seed0.py/seed1.py或加參數化），以及實際執行csynth後產生的report檔案路徑
   （例如hls/syn/report/csynth.rpt），不能只在md裡寫表格數字卻沒有對應的report檔案佐證。

具體步驟：
1. 仿照00_export_weights.py，把seed0/seed1的checkpoint分別匯出成HLS可用的權重格式，存到獨立的
   工作目錄（例如phc_stp_step_hls_seed0.pt、phc_stp_step_hls_seed1.pt）。
2. 建立獨立的HLS專案/工作目錄跑seed0/seed1的csynth（至少做到csynth這一步即可回報初步資源數字；
   若時間允許，盡量也做cosim確認功能正確性）。
3. 產出AUDIT_hls_cross_seed_resource_stability.md的更新版，比較表格裡seed0/seed1/seed2的
   DSP/LUT/FF/BRAM/latency數字，每一格都要附上對應report檔案的路徑，不能是估算或目視推測。
4. 如果seed0或seed1的csynth資源數字跟seed2差異超過合理範圍（例如LUT/FF差異超過±20%），要額外
   說明可能原因（例如不同種子訓練出來的權重稀疏度/數值分布不同，導致定點化後LUT用量不同）。

常見坑：
- 不要因為環境配置複雜就只跑csim而省略csynth——資源數字必須來自csynth或更後面的階段，csim只
  驗證功能正確，不會有資源數字。
- 如果中途卡在環境限制（例如Vivado授權、執行時間上限），務必在報告裡明確寫出卡在哪一步、
  已完成到哪一步，不要回頭只交一份「目前缺口」的敘事性總結來代替實際執行結果。
```

---

## 3.11 圖表標題/軸標籤狀態更新——本輪重新盤點後，先前懷疑的「cosmetic bug」缺乏新證據，且發現一個沒人做過版本控制的程序問題

舊版這裡寫的是「已確認繪圖script有bug，同一個matplotlib軸物件被多次savefig重用，標題互相污染」。**本輪重新審查 `AUDIT_chart_axis_semantics.md`（這次是Codex的最新版本）後，這個說法需要收回，改成更保守、更誠實的講法。**

先講一個重要的程序限制：**`n2ew_era` 這個git repo目前完全沒有任何commit歷史**（`git status` 顯示 no commits yet）。這代表沒有辦法用 `git diff` 去確認Codex這次到底有沒有真的去改繪圖script、改了哪幾行。我們只能看到檔案「現在這個瞬間」長什麼樣子，看不到「之前」跟「之後」的對比。

重新核對最新一版稽核結果後，實際情況是：

- `01_full_l2_action_universe_builder/03_generate_phase01_figures.py` 裡，`action_count_waterfall.png`、`action_side_distribution.png`、`horizon_label_distribution.png`、`feature_distribution_snapshot.png`、`feature_distribution_by_split.png`、`old_residual_vs_new_universe.png` 六張圖，**程式碼裡各自都有獨立的 `set_title`/`set_xlabel`/`set_ylabel` 呼叫**，標題文字跟檔名語意是對得上的。也就是說，**舊版報告聲稱的「這幾張圖標題互相污染」這件事，目前看不到任何證據支持**——可能是先前的判讀本身有誤，也可能是Codex確實默默修好了，但因為沒有版本歷史，無法分辨是哪一種。
- `05a_phc_lite_microstructure_sequence_feasibility/common_05a.py` 裡的 `phc_lite_*` 系列圖（loss curve、val policy pnl、spearman、grad norm、policy comparison、session pnl、causal-fill comparison、oracle confusion、score bucket lift）**同樣各自有獨立的title/xlabel/ylabel**，跟舊版聲稱的「沿用前一張圖的舊標題」不符。
- 唯二維持「找不到明確軸標籤」狀態（`inferred=True`，但這代表「找不到」不是「標錯」）的，是 `02_short_horizon_sequential_proxy_baselines/06_generate_phase02_figures.py` 裡的 `validation_policy_pnl_heatmap.png` 跟 `validation_family_pnl_heatmap.png` 這兩張熱力圖——程式碼附近沒有明確的xlabel/ylabel設定呼叫，這兩張圖在投影片上應該用口頭/文字補充軸意義，不要假設畫面上有文字。

**對答辯/簡報的實際影響與建議做法**：不要再說「這是已確認的繪圖bug」，這個措辭目前缺乏證據支持，講出來反而會被問「你說的bug在哪裡，我看程式碼是對的」。正確的講法是：「我們重新盤點過所有圖表的標題/軸標籤來源程式碼，多數圖都各自有正確的標籤；只有兩張熱力圖（validation_policy_pnl_heatmap、validation_family_pnl_heatmap）目前沒有找到明確的軸標籤設定，這兩張圖的軸意義要用口頭補充」。**另外要主動揭露一個更高層的問題**：因為這個專案目前完全沒有git版本歷史，任何「之前有問題、現在修好了」的說法都沒辦法被驗證，這是個比單張圖表標籤更根本的嚴謹度缺口——建議近期把整個專案先做一次commit，往後才有辦法用diff證明「真的改了什麼」。

如果仍然想對這兩張heatmap補上明確的軸標籤，可以交給Codex按以下提示處理：

```
任務一：補上 n2ew_era/02_short_horizon_sequential_proxy_baselines/06_generate_phase02_figures.py
裡 validation_policy_pnl_heatmap.png 跟 validation_family_pnl_heatmap.png 兩張熱力圖目前缺失的
明確xlabel/ylabel設定（目前AUDIT_chart_axis_semantics.md把這兩張標記為inferred=True，代表程式碼
附近找不到明確的軸標籤呼叫）。

鐵則：
1. 不得更動任何資料計算邏輯、模型、或現有frozen結果csv——這純粹是畫圖script的文字標籤修正，
   不是分析邏輯修正。
2. 不要去改其他已經有明確title/xlabel/ylabel的圖（已重新核對過01_full_l2_action_universe_builder
   與05a_phc_lite_microstructure_sequence_feasibility兩支script的圖，目前都各自正確，不需要動）。
3. 修正後，重新跑一次 n2ew_era/000_context/run_oracle_sensitivity_and_matched_sim_audit.py 旁的
   chart-axis稽核流程（或依照AUDIT_chart_axis_semantics.md的方法重新生成同格式表格），確認這兩張
   圖的inferred欄位變成False，且title/xlabel/ylabel內容與圖檔語意相符。
4. 輸出更新後的AUDIT_chart_axis_semantics.md供人工覆核，不要只回報「已修好」就結束。

任務二（獨立、優先層級較低，但建議一併處理）：n2ew_era這個git repo目前完全沒有commit歷史
（git status顯示no commits yet），導致任何"之前有問題、現在修好了"的說法都無法用git diff驗證。
請執行 git add -A && git commit -m "baseline snapshot before further fixes"，把現狀定樁，
往後每次修改才能用diff證明實際改了什麼。執行前請先確認 .git/index.lock 沒有殘留鎖檔
（之前審查時遇到過 unable to unlink .git/index.lock 的權限錯誤），如有殘留請先安全移除再commit。
```

---

## 4. 教授/Reviewer視角：答辯前必須心裡有底的完整限制清單

> 這是全篇最重要的一節。把分散在各章節的限制集中列出，逐項都已標明出處，方便答辯前快速複查，**主動講出來比被問到才講更有說服力**。

1. **PHC NEW5尚未開鎖，理由有三項具體且可驗證**（§1）——03B6→04B1逆轉未解、oracle約束敏感度掃描的方法論限制（見第7項）、記憶ablation顯示優勢來源還需再確認，三項都不需要碰NEW5即可先處理。
2. **PHC-lite不是完整PHCSSM**（§3.6）——缺spike/STDP機制，這是自我聲明的紅線。
3. **PHC嘗試動機是事後合理化**（§3.6）——技術論證在嘗試之後才補上。
4. **記憶ablation是反事實推論，非重新訓練**（§3.7）——`fixed_decay_0.9`幾乎打平baseline，不能只強調memory-off的下降。
5. **03B6→04B1的ML輸贏逆轉沒有根因解釋**（§3.4）——本輪已重新檢視三份候選文件，確認缺口仍開放，已附Codex實驗prompt。
6. **Oracle現在有四層，最嚴格的一層（matched-count + simulator）本輪已補上，OOS全部收斂，但development split上有一個已知收斂失敗案例（PHC train誤差21.4%）**（§2.6）——目前對外可用的最誠實headroom數字是9.46%～23.23%（OOS九格capture ratio，最低為NEW4/Simple規則，最高為NEW3/PHC），比舊版「21.5倍」或13.8%/18.6%都更嚴格，但這個方法本身status仍是mixed，不能宣稱已完全成熟。
7. **Oracle約束敏感度掃描已補做（9組合），但其中「3個inventory_cap」實際只有2種真正不同的狀態（cap=2與cap=3數字完全相同）**（§1理由二、§2.6）——對外溝通時必須講清楚這個掃描廣度的限制，否則容易被抓到「3組假設」名實不符。掃描本身的好消息是：27個OOS觀測點上排名`PHC>HGB>queue rule`百分之百一致。
8. **5倍手續費壓力測試下，queue imbalance規則在三個診斷切分上全部轉負**（§2.4）——手續費假設敏感度是重要穩健性警示，不能略過。
9. **方法論外部驗證樣本小且相關**（§2.5）——n=9，跨階段共享市場資料，沒有算信賴區間是對的，不要額外宣稱統計顯著。
10. **selected_policy_equity_curves.png畫的是無選擇性的`ask_all`機械基準**（§3.1）——不是alpha證明，是比較基準的地板線。
11. **特徵分布在切分間並非完全同分布**（§3.1）——new4波動率明顯偏高、test事件率明顯偏低，這是真實的regime差異，不是噪音。
12. **HLS跨種子穩定性目前是「完全零進度」，不是「進行中」**（§3.10）——本輪重新審查交辦給Codex的任務後，確認沒有任何seed0/seed1的匯出腳本、工作目錄或csynth報告存在，只是把已知缺口重新書面化。唯一有完整證據的是seed2。已附Codex補做prompt，要求這次必須真的執行export+csynth。
13. **PYNQ-Z2實機上板未做**（§3.10）——目前最後一步未完成，只到P&R timing met。
14. **HLS測試向量可能不具代表性**（§3.10）——只用了validation split前200組，非完整集合或獨立reference。
15. **WNS margin僅+0.020ns，餘裕極薄**（§3.10）——功能上合法但風險不能淡化，已附降頻建議。
16. **整個n2ew_era專案目前完全沒有git commit歷史**（§3.11）——`git status`顯示no commits yet，導致任何「之前有問題、後來修好了」的說法都無法用diff驗證，這比單一圖表標籤問題更根本，建議近期先做一次commit定樁。
17. **圖表標題/軸標籤問題本輪重新核對後大部分查無實據，只剩兩張熱力圖缺明確軸標籤**（§3.11）——舊版「已確認cosmetic bug」的講法已收回，不要再用這個措辭，正確說法是「多數圖各自有正確標籤，僅validation_policy_pnl_heatmap與validation_family_pnl_heatmap兩張缺軸標籤設定，需口頭補充」。

---

## 5. 給新組員的一句話總結（口頭簡報開場可用）

「我們在做的是用L2 order book的因果特徵去決定怎麼報價，評估方式不是看終端結果，而是模擬真正逐筆執行（含庫存限制、隨機成交機率）算出的短horizon代理損益。目前唯一一次真正算數的最終測試（NEW5 holdout，HGB模型）結果是正的；用更嚴格的方法（同時匹配出手次數又穿過成交模擬器的oracle）重新檢驗headroom，OOS九個格子全部精確收斂，policy capture ratio落在9.46%～23.23%，比之前的13.8%/18.6%更誠實。PHC-lite序列模型目前在NEW3/NEW4上排名都贏HGB跟簡單規則，而且這個排名在9種不同執行假設下都沒變過，但還沒開NEW5鎖，因為三個具體理由（ML逆轉機制未解、oracle方法本身在development split上有已知收斂失敗案例、記憶優勢來源待確認）需要先處理。HLS硬體這邊，PHC-lite seed2已經完整跑通csim到P&R並且timing met，但seed0/seed1的硬體驗證目前是完全零進度，不是進行中。我們很清楚知道哪裡還有缺口，接下來的工作就是照著缺口清單一項一項補，而不是回頭美化已知的限制。」

---

## 附錄：圖檔總清單（依資料夾分組，方便排版時批次插入）

**01_full_l2_action_universe_builder/figures/**：action_count_waterfall.png、action_side_distribution.png、horizon_label_distribution.png、feature_distribution_snapshot.png、feature_distribution_by_split.png、old_residual_vs_new_universe.png

**02_short_horizon_sequential_proxy_baselines/figures/**：validation_policy_pnl_heatmap.png、validation_family_pnl_heatmap.png、selected_policy_equity_curves.png、selected_policy_drawdown_curves.png、random_baseline_distributions.png

**02c_queue_aware_stress/figures/**：queue_stress_heatmap.png、delay_sensitivity.png

**02d_final_strict_baseline_audit_package/figures/**：selected_vs_controls_bar.png、random_hurdle_bar.png、fee_sensitivity_selected_policy.png、queue_stress_heatmap_selected_policy.png、delay_sensitivity_selected_policy.png

**02f_excess_markout_and_causal_fill_proxy/figures/**：sequential_excess_equity_curves.png

**03a_ml_readiness_excess_markout/figures/**：bucket_lift_deep_imbalance.png、bucket_lift_queue_imbalance.png、bucket_lift_volatility_past_50.png

**03b_simple_ml_action_value_baselines/figures/**：validation_threshold_selection_curve.png、frozen_oos_ml_vs_controls.png、matched_random_hurdle.png、causal_fill_sequential_equity_curve.png

**03b6_hgb_convergence_closure/figures/**：hgb_val_policy_pnl_by_iter.png

**05a_phc_lite_microstructure_sequence_feasibility/figures/**：phc_lite_loss_curve.png、phc_lite_val_policy_pnl_by_epoch.png、phc_lite_spearman_by_epoch.png、phc_lite_grad_norm_by_epoch.png、phc_lite_policy_comparison.png、phc_lite_session_pnl.png、phc_lite_causal_fill_comparison.png、phc_lite_oracle_action_confusion.png、phc_lite_score_bucket_lift.png、以及seed1/seed2版本的loss_curve/val_policy_pnl_by_epoch/spearman_by_epoch/grad_norm_by_epoch（共17張)

**05c1_phc_dale_law_ablation/figures/**：dale_loss_curve_seed0/1/2.png、dale_val_policy_pnl_seed0/1/2.png

**05c2_phc_stp_dynamic_synaptic_gain/figures/**：stp_loss_curve_seed0/1/2.png、stp_val_policy_pnl_seed0/1/2.png、phc_stp_gain_distribution.png

**05c3_phc_stp_clean_200epoch/figures/**：phc_stp_v2_loss_curve_seed0/1/2.png、phc_stp_v2_val_policy_pnl_seed0/1/2.png

**05c4_phc_stp_cosine_lr/figures/**：phc_stp_v2_loss_curve_seed0/1/2.png、phc_stp_v2_val_policy_pnl_seed0/1/2.png（**seed2是最終HLS採用來源，務必記住**）

**05d_oracle_gap_reduction/figures/**：oracle_gap_by_volatility_bucket.png、oracle_gap_by_event_rate_bucket.png、oracle_gap_error_type_stack.png

**05e_phc_hidden8_ablation/figures/**：phc_stp_v2_loss_curve_seed0/1/2.png、phc_stp_v2_val_policy_pnl_seed0/1/2.png（hidden_dim=8消融版本，檔名與05c系列相同但內容不同，插圖時務必標明資料夾來源避免混淆）

**06b_phc_lite_streaming_fixed_point_equivalence/figures/**：phc_streaming_action_agreement.png、phc_fixed_point_action_agreement.png、phc_fixed_point_causal_fill_pnl.png

**06c_tcn_action_value_hls4ml/figures/**：tcn_av_16feat_loss_curve_seed0/1/2.png、tcn_av_16feat_val_policy_pnl_seed0/1/2.png、tcn_av_18feat_loss_curve_seed0/1/2.png、tcn_av_18feat_val_policy_pnl_seed0/1/2.png

**06d_tcn_5block_prepad_hls4ml/figures/**：tcn_5block_prepad_18feat_loss_curve_seed0/1/2.png、tcn_5block_prepad_18feat_val_policy_pnl_seed0/1/2.png

**06a_hls_queue_rule_baseline／06f_phc_lite_manual_hls**：這兩個資料夾下沒有自製分析圖檔(只有Vivado工具自動生成的IP logo.png,非分析用圖),HLS資源與時序數字以§3.10的表格與原始log路徑(`hls_run_cosim.log`、`bd_0_wrapper_timing_summary_routed.rpt`)為證據來源,建議在投影片用表格呈現而非圖片。
