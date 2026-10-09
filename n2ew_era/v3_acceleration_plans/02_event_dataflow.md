# 方法2：encoder / recurrence / readout跨事件DATAFLOW
工作目錄：`n2ew_era/v3_acceleration_experiments/02_event_dataflow/`。

## 問題與假說
V3 encoder不依賴前一事件state。recurrence完成後，下一事件不必等待上一事件readout結束。目標降低II與burst排隊，不預設孤立事件latency會縮短。

## 第一階段：串流功能與schedule驗證
1. 先畫目前依賴圖：encoder -> recurrence -> readout，state_feedback只回到recurrence。列出stage latency/II，inline導致沒有獨立stage報表時明確標未知，不用總latency亂拆。
2. 建立獨立streaming top，stage函式INLINE off；encoder輸出帶event_id的向量token；recurrence在單一process內擁有state，readout接收獨立snapshot。不要建立未初始化的跨process feedback FIFO。
3. 明確保留兩條精度路徑：readout使用本次寬精度state_acc；下次recurrence使用cast成phc_fixed_t的state_next。不能讓readout改用24-bit state，不能在下一事件覆寫前一事件snapshot。
4. 向量token以完整event為單位，或顯式定義16-lane打包協定；不能把每channel II=1說成每event II=1。
5. 明確定義reset/session邊界state歸零，backpressure時不能提前更新state、遺失或重複event。FIFO深度可先2/4做有限掃描，記錄deadlock理由。
6. 與原核心作4096筆閉環bit-exact Csim；另測相鄰事件、隨機input空拍、output stall、session reset。assert event順序／數量與readout/state相等。
7. 跑csynth，檢查DATAFLOW process真的存在及FIFO/PIPO、recurrence dependence、實際stage II。報告瓶頸stage與預測steady-state II，未cosim不得稱實測II。

## 後續核可後
cosim連續供應事件並隨機backpressure，保存每筆input_accept/output_valid/state_commit cycle CSV；從trace算latency與event II，不能只引用pragma target。
burst測試固定相同到達時間與queue規則；較低II不直接代表PnL提升。

## 交付與停止點
`REPORT.md`、process依賴圖、Csim比對、csynth結果與待測cosim腳本。第一階段完成STOP。
此路線可獨立從baseline做，通過後才與方法1合併並重新驗證；不可假設加速倍數可相乘。

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

