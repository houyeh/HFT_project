# 方法3：混合精度與reciprocal-sqrt近似
工作目錄：`n2ew_era/v3_acceleration_experiments/03_mixed_precision/`。

## 問題與假說
csynth存在40-bit寬乘法與更寬中間乘積；先省DSP，再把資源投入encoder平行化。低位寬不保證cycles自動下降。

## 第一階段：純軟體數值與PnL消融
1. 只用train/val量測weight、encoder activation、mean/variance、inv_std、state與readout動態範圍、分位數、最大值；記錄溢位/飽和比例，校正範圍只能由train決定。
2. 最多六個非baseline設定：權重/activation的18、16、12總位寬三組；保留state24與足夠寬accumulator。每組整數位由train範圍決定，不把ap_fixed<16,6>當通用答案。
3. 另測reciprocal-sqrt查表/分段近似的兩個明確精度設定，起初維持其餘原精度；最後最多一個combined設定。先把grid寫manifest再執行，不因val或OOS結果無限加設定。
4. rsqrt必須保留eps、處理零/近零variance、定義輸入範圍與out-of-range行為。報告最大相對/絕對近似誤差；不可假定variance均勻分布。
5. 定點模擬需模擬各運算節點的cast、AP_TRN/AP_WRAP等模式，不是只量化權重。用Vitis ap_fixed C參考或既有已驗證模擬校對負數、邊界、溢位語義。
6. 評估完整sequence readout誤差、state漂移、action一致率、threshold margin、val sequential PnL。此階段不做QAT或重訓。
7. 報告98/95/90%比較檔位下的候選；matched-random使用既有協定，先只對development選定的候選跑，避免不必要全grid成本。

## 後續核可後
選定候選凍結，再做NEW3/NEW4診斷與HLS Csim/csynth/cosim，確認省下DSP與真實cycles。若PTQ失敗，回報是否值得另開QAT任務，不自行重訓。

## 交付與停止點
`REPORT.md`、`results/precision_grid.csv`、動態範圍、誤差/溢位與PnL表，完整精度manifest。純軟體Phase完成STOP；未合成只寫資源假說。

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

