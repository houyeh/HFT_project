# 方法1：V3 encoder交錯累加與局部pipeline
工作目錄：`n2ew_era/v3_acceleration_experiments/01_encoder_schedule/`。

## 問題與假說
原始16x16 encoder外層channel loop為320 csynth cycles，占532 cycles約60%。優先改善這個序列區段，保留模型與定點運算語義。

## 第一階段：排程與數值可行性
1. 複製凍結來源，保存來源hash；用report與原始碼行號確認320-cycle loop，列出乘法器延遲、權重bank、input讀port及每channel累加依賴。
2. 做baseline與最多兩個候選：2個、4個output channel交錯累加。每channel依原順序处理input i，保留bias初始化、每次乘積轉型與每次acc更新的截位；不要順手改成加法樹。
3. 保留inner unroll8語義；權重banking按實際讀取方向設計。外層pipeline可能強迫內層完全unroll，必須檢查compile log與DSP，不能假設factor仍生效。
4. 不改LayerNorm、SiLU、recurrence或readout公式。baseline與候選以同一組固定輸入/state跑Csim，直接比對原始Q24.12 readout/state bits；再跑4096筆閉環回授replay。
5. 任何bit差異先定位，不可用PnL容忍度掩蓋。以相同算式改排程的目標是bit-exact。
6. Csim通過後跑csynth，列encoder loop cycles、全核latency/II、DSP/LUT/FF/BRAM與推算us。資源超標照實記錄。

## 交付與停止點
輸出 `REPORT.md`、`results/schedule_scan.csv`、來源hash manifest、patch與完整logs。
列baseline/output-channel2/output-channel4全部結果；明確指出收益是latency還是II。
完成csynth後STOP，等待選定候選再cosim/P&R。不重新訓練或重跑PnL；bit-exact只證明所測輸入集合等效，不能宣稱所有輸入形式證明。

## 共用規格（本 prompt 的一部分）

本次以目前已上板的 V3 diagonal16 為起點。只執行本 prompt 指定的第一階段；後續階段先準備計畫，停下等使用者覆核。
- 凍結來源：`n2ew_era/06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/hls_src/`。
- 既有綜合證據：同變體 `hls_component/stage_a_work/hls/syn/report/phc_diag_step_csynth.rpt`。
- 訊號來源：`n2ew_era/05f_phc_diagonal_ssm/` 的 frozen seed2；先從既有 selected policy / export manifest 確認精確 checkpoint 與 threshold，不能猜路徑。
- 上板證據：`n2ew_era/06s2_pynq_z2_wrapper_bringup/V3DIAG_STAGE_E_REPORT.md`。
- 原始基準分層記錄：csynth latency/interval=532/533 cycles；cosim latency/interval=530/531 cycles。5.300 us 是530 cycles在100 MHz timing closure下的推導值，不是獨立晶片量測。
- 既有核心 routed DSP=196/220；帶VIO板級結果另列，不能把核心與板級資源混用。
- 禁止修改任何 frozen source、checkpoint、threshold、原有結果。所有產物放到本實驗獨立目錄；輸出路徑由 CLI 明確傳入，import 不得寫檔。
- 不碰 NEW5，包括載入、評估、用於校正／挑選。NEW3/NEW4只能在development選定候選、寫下凍結manifest後作事後診斷，不得據此挑選／改參數。
- 先確認既有切分，沿用 NEW2=train、NEW1=val；若實際manifest不同，回報後停下，不能顛倒。
- 不修改特徵、標籤、模擬器、matched-random實作。重用既有 conservative_l2_stress、inventory_cap=1、H=20 的 sequential causal-fill 路徑，記錄import入口與所有設定。
- PnL主指標是 sequential_excess_pnl，markout/action_pnl另欄，不能混用。保留率分別對frozen V3與05C4 dense seed2計算，分母必須是同split、同模擬協定。分母<=0時標NA並列絕對差。
- 98%/95%/90%是待使用者選定的development比較檔位，不是已批准的PnL損失。報告成交數、每筆成交PnL、決策一致率；不能只看總PnL。
- 模型/精度改變後，對新模型float vs fixed完整序列比較，讓state由各自結果回授；teacher-forced state測試只作定位，不得代替閉環驗證。
- HLS一律用Vitis/Vivado 2025.2、xc7z020clg400-1、10 ns起步，沿用已驗證settings64環境。每次保存版本、command、config、stdout/stderr、報告與SHA256。
- 同一時間最多一個Vivado/Vitis工作，不與CPU訓練／基準量測並行。開始前檢查現有重負載，不自行終止其他程序。
- 不盲目加頂層PIPELINE、不全域改unroll、不移除已驗證local partition。用schedule、memory-port、recurrence與資源證據決定pragma。
- 新HLS必須先Csim，再csynth，再cosim；不得提高既有testbench容忍度。對近似模型若既有門檻不過，保存結果、停下覆核，不自行改門。
- DSP<220、LUT<53200、FF<106400、BRAM_18K<280是核心必要條件；另留wrapper/debug預算，核心fit不代表板級fit。
- P&R/上板都需另一次核可；只有實際timing closure後才能報routed-derived us。禁止預先承諾sub-us、倍數或PnL改善。
- latency量測固定為完整feature vector到對應readout有效；II用連續事件trace驗證。另列介面輸入／輸出開銷、wrapper等待與排隊延遲，不能互相代換。

每階段報告開頭寫 status、完成到哪個驗證等級、停止原因。失敗保留完整log，不自行延伸另一個架構假設。

