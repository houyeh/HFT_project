## PHC / PHCSSM (arXiv 2604.01295v2)

**題目**:Parallelized Hierarchical Connectome — 把 State-Space Model (SSM) 升級成「同時具備空間(跨神經元)與時間遞迴」的框架,具體實例化為 PHCSSM(一個 spiking SSM)。作者 Po-Han Chiang,陽明交大。

**要解決的矛盾**:現代 SSM(S4/S5/Mamba/LinOSS)靠「對角化狀態轉移矩陣」換來 O(log T) 平行掃描(parallel scan)訓練,但代價是同一時間步內神經元彼此解耦——沒有側向連接、沒有 Dale's Law(興奮/抑制符號限制)、沒有狀態依賴的突觸傳遞。反過來,生物真實的 RSNN(spiking neural net)有這些機制,但訓練是嚴格序列的(BPTT through spikes),長序列訓練成本隨 T 線性增長,無法吃到 GPU 世代紅利。PHC 想要兩者都要。

**核心架構(三段式,對應 Fig.3)**:

0. PHC-lite 是 PHCSSM 論文的啟發的實作，可以簡單理解成與GRU同屬閘控RNN家族，但省略了reset gate，且用固定的per-channel學習時間常數取代動態update gate，架構上更接近閘控式漏電積分器/對角SSM)。為了在 FPGA 上的可行性，保留原論文中「可學習的閘控/衰減遞迴狀態更新」，捨棄 spike 機制、STDP、surrogate-gradient spiking neuron、multi-transmission loop等原論文啟發的多數生物機制，並將這部分保留做為未來試驗空間。

1. **Sensory Encoder**:輸入線性投影 + LayerNorm + input mask,產生持續性感覺驅動 `x_sen`,每次迴圈都重新注入。
2. **Multi-Transmission Loop (MTL)**:在同一個時間步內,重複 `N_max` 次「Neuron Layer → Synapse Layer」:
   - **Neuron Layer (NL)**:三個對角 scan 構成 ALIF(adaptive leaky integrate-and-fire)——膜電位衰減(softplus 輸入)、自適應閾值累積、不反應期(refractory)衰減,最後 Heaviside 閾值化產生 binary spike。
   - **Synapse Layer (SL)**:spike 先過 FIFO 延遲緩衝(軸突延遲),再乘上 Tsodyks–Markram STP 的雙狀態(facilitation `u`、recovery `x`)動態增益 `g=u·x`,最後用拓撲遮罩+Dale's Law 鉗制後的權重矩陣 `W_struct = W_syn ⊙ M_topo` 做一次矩陣乘法,產生下一輪的突觸電流。
   - 迴圈用 Cauchy 收斂準則監控 residual,並有 convergence regularizer 把它變成靜態計算圖。
3. **Readout**:對最後一輪 iteration 的電位或 spike 做時間聚合(mean/last/sum/max/weighted/com/ssm 任選)+ 線性分類頭。

**訓練 vs 部署雙模式(全文重點)**:訓練模式把 k(MTL 迭代)當外層迴圈、T(時間)當內層 log-scan;部署模式反過來,把 t 當外層迴圈、每步只跑一次 NL+SL、狀態逐步延續——兩者共用同一組權重,理論上收斂到同一不動點,不需要 ANN-to-SNN 轉換就能把訓練好的模型直接當 RSNN 部署到晶片。這是論文第二個賣點。

**兩條學習通道**:梯度下降(AdamW)+ STDP(pre/post spike timing 的 Hebbian 更新),兩者都要在更新後對 `W_syn` 做 Dale's Law 投影。

**實驗**:6 個 UEA 生理訊號分類資料集(T=405~17,984),參數量僅 1,312~4,891(比最小的基準線 S6 還小 4~10 倍,比最大的 NRDE 小 4 個數量級),準確率與 SSM 家族打平或超越。消融實驗顯示 Dale's Law 影響最大(移除後平均掉 7.4pp,EigenWorms 掉 21pp);STP、STDP 呈現資料集依賴、不是一致正貢獻。跨後端(H100/x86/Cortex-A76/Cortex-M4F)驗證 bit-exact 或接近,並真的把模型跑上 STM32L412KB 微控制器(40KB SRAM/128KB Flash)。

---

## 逐項比較:論文 PHCSSM vs 你專案的 PHC-lite 實作

我讀了 `05a_phc_lite.../models/phc_lite.py`(基準版)、`05c1.../phc_lite_dale.py`、`05c2.../phc_lite_stp.py`、`05f.../phc_diagonal_ssm.py`,以及 `05c0` 的 freeze report 和 claim boundary。你的專案文件本身已經很誠實地標註「這不是完整 PHCSSM」,以下是具體逐項差異:

| 組件 | 論文 PHCSSM | 你的 PHC-lite 實作 | 差異程度 |
|---|---|---|---|
| 神經元模型 (NL) | ALIF:3 個對角 scan(膜電位衰減、自適應閾值、refractory)+ Heaviside 產生 binary spike | 一個 GRU 式 gated 更新:`decay*state + sigmoid(gate)*tanh(candidate)`,連續值,無 spike、無閾值、無 refractory | 完全不同機制,你是連續值 RNN,不是 spiking |
| 突觸層 (SL) 主體 | delay buffer → STP 增益 → `W_struct` 矩陣乘 | 05a:純 `nn.Linear`;05c1/05c2 各補一塊 | 分散實作,單一模型未整合 |
| 突觸延遲 | 固定整數延遲 FIFO(公式 9) | 完全沒有 | 缺 |
| STP(Tsodyks–Markram) | 雙狀態 `u,x` 的仿射遞迴(facilitation+depletion),對數域可平行掃描 | `05c2`:一個學到的 gain = `sigmoid(W1·encoded + W2·prev_state)`,單一無狀態的逐步門控,docstring 自陳「not full biological STP」 | 概念相似(都是動態增益調制),機制不同(無雙狀態 ODE、無時間常數 τf/τd) |
| Dale's Law | 對 `W_syn` 做**參數更新後的原地投影**(E 欄位 clip≥0, I 欄位 clip≤0),不進 autograd | `05c1`:`softplus(raw_weight) * sign` 的**可微分重參數化**,永遠滿足符號但是走梯度圖 | 功能等價(都保證符號一致),但實作路徑不同;且只在 `05c1` 這個消融分支存在,不在主線模型 |
| 拓撲遮罩 / 多區域階層 | `M_topo` 由「宏觀區域數 R + 每區 E/I 比例 + 跨區連接機率」構造,矩陣有 feedforward/feedback/lateral 子塊 | 沒有多區域結構,`hidden_dim=16` 是扁平的一層,無 `M_topo` | 論文核心賣點(hierarchical connectome)完全未實作 |
| Multi-Transmission Loop | 訓練時 `N_max`(生產設定 12)次迭代,每次迭代**重置** NL/SL 狀態,只有 loop input 傳遞;有 Cauchy 收斂判準 + convergence regularizer | `mtl_steps` 預設 **1**(frozen v1 就是 1),迴圈嵌在每個時間步的 for-loop 裡,沒有跨迭代殘差監控、沒有收斂正則項 | 結構上退化成「單次訊息傳遞」,MTL 的「用迭代換深度」核心思路基本沒用上 |
| 訓練 vs 部署雙模式 | 兩套排程共用權重,有明確的漸近等價證明與跨後端驗證 | 只有一種前向(等同論文的「RSNN 部署模式」結構:t 外層、狀態逐步延續),沒有平行掃描訓練模式,也沒做雙模式一致性驗證 | 你等於直接用「部署模式」在做訓練——工程上簡單但沒有論文強調的 O(log T) 訓練加速 |
| 時間軸平行化 | log-domain prefix scan,O(log T) | Python `for t in range(seq_len)` 序列迴圈,O(T) 且是 Python-level 迴圈(非向量化 scan) | 完全沒有平行掃描;seq_len=128 還可接受,若序列變長會是瓶頸 |
| STDP | pre/post spike timing 的雙軌跡 Hebbian 更新,獨立於梯度、真的用 binary spike | 沒有任何實作(grep 全專案找不到程式碼,只有文件明確排除) | 缺,且因為你的模型無 spike,要做「真正的」STDP 需要先有 spiking 機制 |
| 損失函數 | CE + spike-rate 正則 + 電位正則 + 收斂正則 | 你的任務是連續值 action-value 回歸(`excess_pnl_H20`),用回歸/校準損失 | 這是**合理的領域差異**,不是缺陷——論文是分類任務,你是交易訊號回歸 |
| Readout | 多種時間聚合(mean/last/sum/...)可選,取電位或 spike | 固定取序列最後一步的隱狀態 | 對應論文的 `last` 選項,是子集 |
| 應用領域 | UEA 生理訊號多變量分類(6 個學術資料集) | L2 order book 微結構、bid/ask action-value、causal-fill 模擬、matched-random 對照、queue-aware stress test | 論文完全沒碰過金融/HFT,這是你自己的領域遷移,是加分而非差異 |
| 硬體部署驗證 | x86/H100/Cortex-A76/Cortex-M4F 跨後端 bit-exact,真跑 STM32 量測 Flash/SRAM/energy | `06f/06g/06h/06n/06o` 系列:manual HLS、hls4ml、PYNQ-Z2 FPGA、fixed-point 等價性、延遲預算逼近 1us | 你在 FPGA/HLS 這條線做得**比論文更深**,但部署的是簡化版模型(83µs 在 PYNQ-Z2,你自己文件也承認 < 10µs 的 HFT 需求還沒到) |
| 參數量 | 1,312~4,891(依資料集) | frozen v1 約 1,203 | 數量級相近,純屬巧合(架構本質不同) |

---

## 與GRU比較
不建議。我剛去讀了實際的模型程式碼(`n2ew_era/05a_.../models/phc_lite.py`)確認一下，PHC 在架構上其實**不是** GRU，改名會變成一個站不住腳的技術宣稱，而且會跟你自己的消融證據互相打架。差異點：

**1. decay 不是動態的 update gate**

程式碼第52行：`decay = torch.sigmoid(self.decay_param).view(1, -1)` 是寫在時間迴圈**外面**算的——它是一個訓練後固定下來的、per-channel的**常數向量**，不吃當下輸入 x_t，也不吃前一刻狀態 state_{t-1}。GRU 的 update gate `z_t = sigmoid(W_z·x_t + U_z·h_{t-1})` 是**每一步重新算**的，會隨輸入和狀態動態變化。這是本質上不同的機制——你的模型比較接近「固定時間常數的漏電積分器(leaky integrator)/對角線State-Space Model」，這也正是為什麼專案裡有一個獨立的 `05f_phc_diagonal_ssm` 拿它去跟diagonal SSM比較，而不是跟GRU比較。

**2. gate_t 只吃當下輸入，不吃前一刻狀態**

第55行 `gates[:, t, :] = sigmoid(self.gate(encoded[:, t, :]))`，而 `encoded` 只來自 `self.encoder(x_t)`——沒有任何地方把 `state`(前一刻的隱藏狀態)餵進 gate 或 candidate 的計算。GRU 的 reset gate 和 update gate**都同時依賴** x_t 和 h_{t-1}。你的模型裡，只有 STP 版本(05C2+)額外加的 `gain_t = sigmoid(W_g·encoded_t + U_g·state_{t-1})` 才有吃到前一刻狀態，但那是另一層獨立的乘法式調變，不是GRU的gate。

**3. decay 和 gate 不是互補的(non-complementary)**

`state_temporal = decay * state + gate_t * candidate_t`——decay 和 gate_t 是**兩個獨立的 sigmoid 值**，不像 GRU 是 `(1-z_t)*h_{t-1} + z_t*h~_t` 這種強制加起來等於1的凸組合。你的狀態理論上可以持續增長，不是在新舊值之間做內插。

**4. 你自己的消融結果，其實正好證明「這不是GRU式的動態閘門」**

你上一輪報告裡的`fixed_decay_0.9`消融——把學到的per-channel decay直接換成常數0.9，三個split裡兩個PnL幾乎沒變(+470、+460)。如果這是GRU式的動態update gate，把它強制換成一個對所有樣本、所有時間點都相同的常數，應該會嚴重傷害表現(因為GRU的核心價值就是"每一步能根據當下狀況決定要記多少、忘多少")。結果幾乎沒變，直接印證了decay本來就是一個**接近靜態的時間常數**，不是動態閘門——這跟程式碼證據完全吻合，但也代表如果你把它叫成GRU，這個消融結果反而會變成"啊那你的GRU的update gate是假的"這種很難回答的追問。

**建議做法**：不要整批替換成"GRU"。如果是為了讓讀者好懂，可以在第一次出現PHC時加一句定位描述，例如「一個輕量閘控遞迴架構(概念上與GRU同屬閘控RNN家族，但省略了reset gate，且用固定的per-channel學習時間常數取代動態update gate，架構上更接近閘控式漏電積分器/對角SSM)」，之後全文仍用「PHC-lite」這個名字。這樣讀者一樣好懂，但你不會在口試被問到reset gate在哪裡時答不出來。
---

## 如果要往「更接近論文」補強:詳細作法

依實作成本由低到高排序:

**1. 突觸延遲(最便宜)**——在 SL 輸入前加一個固定 depth 的 FIFO:`s_delayed[:, t, :] = s[:, t-d, :]`(`d=0` 時退化成現況),用 `torch.roll` 或手動 index 加零填充即可,幾行程式碼、不增加參數。

**2. 真正的 Tsodyks–Markram STP**——把 `05c2` 現在的「單步學習增益」換成論文式(10)-(13)的雙狀態遞迴:每個 neuron 維護 `u_t, x_t` 兩個狀態,`u` 用 facilitation 時間常數 `τf`、`x` 用 recovery 時間常數 `τd`,兩者都是可學參數,`g_t = u_t·x_t` 再去調制 spike/message。因為你現有 forward 已經是逐步序列迴圈(不是 scan),直接照公式寫兩行遞迴就好,不需要處理平行掃描的仿射重排。

**3. 真正的 Dale's Law(post-update projection)**——現在 `05c1` 用 `softplus+sign` 的可微分寫法,功能上等價但優化行為不同(梯度會被 softplus 的曲率影響)。若要更貼近論文:改用有號的 `raw_weight`,在 `optimizer.step()` 之後用 `with torch.no_grad(): W[:, E].clamp_(min=0); W[:, I].clamp_(max=0)` 做原地投影,並且要記得在 STDP 更新(如果之後做)之後也要再 clamp 一次。

**4. 加上多區域拓撲遮罩 `M_topo`**——把 `hidden_dim=16` 切成 2 個區域(例如 8+8),構造一個固定(不訓練)的 0/1 遮罩,允許 R0→R1(feedforward)、R1→R0(feedback)、R0→R0/R1→R1(lateral)四塊子矩陣按你想要的密度連接,`W_struct = W_syn ⊙ M_topo`。這是論文「用共享參數換深度」的核心,你目前完全沒有,補上後才算真正在做「hierarchical connectome」而不只是一層 dense 遞迴。

**5. MTL 收斂監控 + regularizer**——現在 `mtl_steps=1` 等於沒有迭代。若要驗證迭代真的有幫助:把 `mtl_steps` 調大(例如 4~12),在每次迭代間算 `r^(k) = ||msg^(k)-msg^(k-1)|| / (||msg^(k-1)||+eps)`,加一個 `λ_conv * mean(r^(k)^2)` 到 loss 裡鼓勵收斂。但要先做消融確認在你的短序列(seq_len=128)、regression 任務上,多迭代是否真的比 mtl_steps=1 好——論文的資料集序列長很多,MTL 的「用迭代換深度」在短序列未必有等量增益。

**6. ALIF 神經元(較大工程)**——如果要讓 NL 真正變成 spiking:把現在的 GRU-cell 換成論文式(3)-(8):膜電位用 softplus 輸入的漏積分、疊加自適應閾值(用另一個 exponential accumulator)、疊加 refractory 衰減、最後 Heaviside 閾值化。這一步會把輸出變成 binary,下游整個 pipeline(readout、loss)都要跟著改,是目前清單裡改動面最大的一項,而且訓練會需要 surrogate gradient(因為 Heaviside 不可微)。

**7. STDP(最貴,而且前提是先做完第 6 步)**——沒有真正的 spike 就沒有真正的 spike-timing,所以 STDP 必須建立在 ALIF 之上。屆時可以用 LTP/LTD 兩條指數衰減 eligibility trace(τ+/τ-),在每個 batch 結束後算 `ΔW_hebb`,經 `M_topo ⊙ M_hebb` 遮罩後疊加到 `W_syn`,再過一次 Dale clamp。

**8. 平行掃描訓練模式**——若之後序列長度超過幾百步、Python for-loop 訓練變成瓶頸,可以把時間軸的衰減遞迴(`decay*state + input`)重寫成 log-domain affine scan(用 `torch.logcumsumexp` 或自訂 associative scan),拿到 O(log T) 訓練加速。目前 seq_len=128 這一步的急迫性最低,可以放最後。

一個務實的建議順序:先做 1(延遲)、2(STP)、3(Dale post-update)這三項成本低、可獨立消融驗證增益的;第 4 項(拓撲遮罩)是概念上最貼近論文精神但工程量中等的;5、6、7 屬於「要不要往真正 spiking 走」的架構決策,值得先問自己:HFT 微結構任務需要 spiking 表示嗎,還是現在的連續值 GRU+Dale+STP-like 組合已經足夠——如果目的是 FPGA 低延遲部署,dense 連續值反而比 spiking+STDP 更容易做定點量化與 HLS 綜合。