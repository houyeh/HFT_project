# V3 FPGA Acceleration: Execution Plan
日期：2026-10-05。狀態：規劃與可執行prompt完成；本任務未啟動新實驗。

## 目標與基準
以5.300 us V3 diagonal16作起點，同時探索單筆latency與事件II。
HLS估计532/533 cycles、RTL cosim530/531 cycles分欄；100 MHz對應5.300 us是routed-derived。
上板4096筆readout exact drift=0、decision flips=0是功能證據，不是獨立延遲量測。
不可由「recurrence存在」直接推出II必須等於整個核心latency：只有state回授有跨事件依賴。

## 六條路線與執行顺序
| 次序 | prompt | 第一階段 | 主要收益假說 | 第一個停止點 |
|---|---|---|---|---|
| 1 | [Encoder schedule](01_encoder_schedule.md) | baseline+2/4 channel交錯，Csim/csynth | encoder目前320-cycle瓶頸 | bit-exact與csynth報告 |
| 2 | [Event DATAFLOW](02_event_dataflow.md) | encoder/state/readout拆stage | 降event II與burst queue | stream功能與csynth |
| 3 | [Mixed precision](03_mixed_precision.md) | 數值範圍、PTQ、rsqrt有限grid | 省DSP後增加平行度 | val PnL/誤差報告 |
| 4 | [Affine normalization](04_affine_normalization.md) | 新student三seed | 去動態normalization | val收斂與PnL |
| 5 | [Diagonal hidden8](05_diagonal_hidden8.md) | 三seed容量控制 | 縮小encoder/state | val收斂與PnL |
| 6 | [Feedforward distillation](06_feedforward_distillation.md) | 淺樹與小MLP | 消除model-state回授 | val選擇與完整規則 |

每份prompt都包含共用規則，可獨立交付agent。先做1；2可從原baseline獨立驗證，通過後才合併。3與4先各自消融，再測組合。5、6為後續模型路线。
不把各自加速倍數相乘。合併版本需要新的Csim/cosim與資源報告。
若最終目的主要是burst不塞車，方法2優先；若是單事件arrival-to-score，方法1/3/4優先。

## 核可節點
1. 執行各prompt第一階段並停下；使用者依實際成本／PnL核可候選。
2. 訊號有改變者先凍結development選擇，才一次性NEW3/NEW4事後診斷；診斷結果不能回頭挑另個candidate。NEW5始終禁用。
3. 核可後Csim -> csynth -> cosim（已完成的步驟不無故重跑）。模型改變與C/RTL mismatch分開，近似模型不能成為掩蓋C/RTL錯誤的理由。
4. 核可後獨立P&R，再核可上板；每階段保留停止點與完整失敗證據。

## PnL 預算與選擇
使用者尚未選定容忍範圍。先列相對V3的98%、95%、90%三檔，另列相對dense production的絕對與比例損失。
主指標統一為既有causal-fill sequential_excess_pnl；不得拿06U的action_pnl表當sequential分母。
同一協定下評估teacher baseline與candidate；baseline從既有csv引用時須先核對model/seed/split/fill/fee/inventory/H完全一致。
在development符合預先批准門檻者中列Pareto候選，不以OOS數字挑選。彙報各session差異與seed散布；若使用CI，按session成塊並交代session數有限，不能把逐event當獨立樣本。
matched-random只檢驗勝隨機；PnL保留是否非劣需要另定配對比較與容忍範圍，不能以p>0.05宣稱等效。

## 每候選統一欄位
variant_id、parent_hash、model_hash、feature/hidden_dim、precision、threshold、
split、sequential_excess_pnl、retention_vs_v3、retention_vs_dense、fill_count、
pnl_per_fill、decision_agreement、matched_random_p、selection_or_diagnostic、
csim_bit_diff、cosim_bit_diff、csynth_latency/II、cosim_latency/II、
event_input/output_cycle_trace、freq、WNS/TNS、DSP/LUT/FF/BRAM_18K、
core_or_board、verification_tier、wall_time、source_path。
未量測填NA與原因，不能用0或估計填成實測。
串流另列queue policy、backpressure與端到端等待；報價不得偷看未到達event。

## 成本與停損
開始先檢查其他Vivado/訓練負載，序列化。先1-epoch或一個baseline計時，再提交全批，給出預算估計。
不預設所有實驗都很便宜；保留checkpoint、命令和logs。預算不足標incomplete，不縮減驗收或硬稱收斂。
不安排升頻作第一槓桿：既有111/125 MHz版本cosim失敗且cycles增加；也不重試全域unroll16（DSP452）。
無新使用者核可，不改主線demo bitstream、不改HLS凍結源碼、不燒板。

## 參考
- `../06u_phc_latency_breakthrough/AUDIT_06u_phc_latency_breakthrough.md`
- `../06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/hls_src/phc_diag_step_hls.cpp`
- `../06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/hls_component/stage_a_work/hls/syn/report/phc_diag_step_csynth.rpt`
- `../05f_phc_diagonal_ssm/PHC_DIAGONAL_SSM_ABLATION_REPORT.md`
- `../06s2_pynq_z2_wrapper_bringup/V3DIAG_STAGE_E_REPORT.md`

