# 方法6：無遞迴前饋／淺層樹蒸餾
工作目錄：`n2ew_era/v3_acceleration_experiments/06_feedforward_distillation/`。

## 問題與假說
用固定有限歷史近似V3分數，消除model-state回授以利跨事件pipeline。研究問題是決策與PnL保留程度，不是僅看R2。

## 第一階段：development-only軟體可行性
1. 先讀既有規則蒸餾報告（若存在）及07A TCN分析；列來源，不覆寫。Teacher本實驗固定05F V3 seed2，若要比較dense teacher，另做独立track不可混稱同一模型。
2. 從train取得teacher bid/ask原始分數，禁止把實際PnL labels當蒸餾目標。
3. 預先鎖定小grid：multi-output DecisionTreeRegressor depth={3,4,6}，history lag={0,3}；另加一個小MLP基準（hidden8、ReLU，同樣lag兩檔）。lag3代表當下加前3筆，共64維，不是3維。
4. lag不得跨session；sequence邊界／缺歷史處理必須明確，沿用相同有效樣本mask比較teacher與student。不可使用未來snapshot。
5. 決策沿用V3 threshold與bid/ask/no_quote tie規則。每候選報R2/RMSE、action agreement、val sequential PnL、成交次數、per-fill PnL、樹葉數/最深比較次數或MLP參數/MAC。
6. 只用val選候選；固定tree random_state與MLP seed0/1/2。對選定候選跑既有matched-random，該p值只代表勝過隨機，不等於與teacher無差異。
7. 樹輸出完整if/else與feature名稱/單位。若是標準化特徵，附逆轉換後threshold可讀性說明；不能將標準化門檻說成原始市場單位。
8. tree/MLP固定lag buffer也是狀態，成本隨lag和feature數增加；它沒有遞迴模型state依賴，但並非零儲存。

## 後續核可後
凍結候選再做NEW3/NEW4診斷。HLS另階段：tree按層pipeline，MLP逐層DATAFLOW；bit位寬、threshold量化和葉輸出量化都重驗PnL。不可承諾單cycle或個位數ns，event II必須把feature傳輸與lag更新算清楚。

## 交付與停止點
`REPORT.md`、`results/distillation_grid.csv`、完整規則範例、teacher/student來源hash與val選擇manifest。
結論寫「在development上保留X%」或「未達設定檔位」；純軟體階段完成STOP，不自動合成。

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

