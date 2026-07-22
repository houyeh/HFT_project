# C類缺口：phc_stp_v2_cosine_lr的HLS驗證——現況糾正 + 使用者手動複現步驟

**狀態：重大糾正（這個任務在你提問前其實已經做完了）**

## 1. 糾正：Task 7的前提是錯的

`CODEX_PROMPT_close_documentation_gaps.md`任務7寫著「`phc_stp_v2_cosine_lr`...目前**從未跑過**06系列的HLS validation」。我實際查證後，這句話**不成立**。

證據鏈（已逐一核對原始log檔，非僅讀markdown摘要）：

1. `n2ew_era/06f_phc_lite_manual_hls/PHC_WEIGHT_EXPORT_REPORT.md`第7行明確記載：
   > Source checkpoint: `n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt`
   
   這正是task 7要求要拿去跑HLS的同一個checkpoint（`phc_stp_v2_cosine_lr`，seed2）。
2. 我直接讀取了原始log（不是只看report摘要）：
   - `hls_component/hls_component/logs/hls_run_cosim.log`第25925-25927行：`total=25600 mismatches=0`, `max_abs_diff=0.00725275`, `*** C/RTL co-simulation finished: PASS ***`
   - `hls_component/hls_component/hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_timing_summary_routed.rpt`第145行：`WNS=0.020ns, TNS=0.000ns, TNS Failing Endpoints=0`（22,941個endpoint全過），第148行：`All user specified timing constraints are met.`
3. 這兩份原始log與`PHC_HLS_Technical_Report.md`/`PHC_HLS_Reviewer_Briefing.md`裡引用的數字**完全吻合**——不是Codex自己編出來的摘要，是有原始工具輸出佐證的。

**結論**：`phc_stp_v2_cosine_lr`（seed2 checkpoint）已經跑完csim → csynth → cosim → **Vivado P&R（Implementation）**全流程，且P&R timing met（WNS=+0.020ns），這比Task 7原本要求的範圍（只要csim/cosim+資源/延遲數字）還做得更完整。

### 1.1 但有兩個還沒解決、需要你知道的點

1. **這是seed2，不是所有seed**：`05c4_phc_stp_cosine_lr/`下若訓練了多個seed（你的`06_seed_stability.py`暗示有做seed穩定性分析），目前HLS只驗證了seed2這一個checkpoint。如果reviewer問「其他seed跑起來資源/精度會不會不一樣」，目前沒有答案——理論上資源數字應該幾乎不變（架構相同、只是權重值不同，Q24.12量化只要沒有飽和就不影響資源），但`max_abs_diff`可能因weight分布不同而略有差異，**這點未經實測，標注「未明確說明」**。
2. **PYNQ-Z2實機上板測試還沒做**：`PHC_HLS_Reviewer_Briefing.md`§7.1 Q7已經自己承認「上板測試列為future work」。csim/cosim/P&R timing met是業界認可的完整驗證鏈，但離"真的插上板子跑起來量到83µs"還有一步。

## 2. 為什麼memory裡記的「PHC csynth資源爆炸11-26倍」現在看起來是錯的？

不是錯的，是**過時**。那是v1-v7迭代過程中的真實狀態（v1: LUT 2073%、DSP 2636%；v7: DSP仍121%），但v8（2026-06-15 02:25，`#pragma HLS PIPELINE off`修正pipeline_loops自動接管問題後）已經把資源壓到DSP 21%/LUT 46%/FF 13%/BRAM 2%，全部在xc7z020 budget內，且後續P&R也confirm了實際資源更低（LUT csynth估46.68% → P&R實測14.20%，這是Vivado synthesis邏輯優化的正常現象）。我會更新memory把這個結論修正掉，避免下次又用過時資訊回答你。

## 3. 如果你仍想自己手動重新跑一次（驗證或練習用），步驟如下

以下步驟是**重現已有結果**，不是從零開始設計，目的是讓你能在reviewer面前說「我自己手動跑過一次，不是只看Codex的報告」。

### 3.1 前置確認

```powershell
cd C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06f_phc_lite_manual_hls\hls_component
```

確認以下檔案存在（這些是已凍結的成品，**不要修改**）：
- `hls_config.cfg`
- `../hls_src/phc_step_hls.cpp`（含v8最終版pragma：所有linear/layernorm/activation/element-wise loop都有`#pragma HLS PIPELINE off`，頂層`phc_step`沒有任何PIPELINE pragma）
- `../hls_src/phc_weights.hpp`（來源checkpoint已確認是`phc_stp_v2_seed2_best_val.pt`）

### 3.2 重跑csim（功能驗證，數分鐘內完成）

```powershell
vitis-run --mode hls --csim --config hls_config.cfg --work_dir hls_component
```

預期看到（對照`logs/hls_run_csim.log`）：
```
total=25600 mismatches=0
max_abs_diff=0.00725275 mean_abs_diff=0.000372553 tolerance=1
INFO: [SIM 211-1] CSim done with 0 errors.
```
若你看到的`mismatches`不是0，或`max_abs_diff`明顯偏離0.00725275，代表環境/工具版本有差異，**不要直接相信新數字，先回報差異**。

### 3.3 重跑csynth（資源估算，約十分鐘）

```powershell
vitis-run --mode hls --syn --config hls_config.cfg --work_dir hls_component
```

完成後打開：
```
hls_component\hls\syn\report\phc_step_csynth.rpt
```
核對Utilization Estimates表格，預期：DSP 46/220 (21%)、LUT 24,836/53,200 (46%)、FF 14,518/106,400 (13%)、BRAM 6/280 (2%)，Latency 8,316 cycles（83.16µs @ 100MHz）。

### 3.4 重跑cosim（RTL/C一致性驗證，視機器效能可能需要10-30分鐘）

```powershell
vitis-run --mode hls --cosim --config hls_config.cfg --work_dir hls_component
```

只有csynth資源確認在budget內才跑這一步（避免浪費時間在注定要重做的設計上）。預期看到`*** C/RTL co-simulation finished: PASS ***`，且`max_abs_diff`應與csim完全相同（0.00725275）——這是「RTL與C model bit-level一致」的關鍵證據，若兩者不同，代表RTL生成過程引入了額外誤差，需要回報。

### 3.5（可選）重跑P&R（Vivado Implementation，需要較長時間，約30分鐘以上）

只有在你需要重新確認timing margin或要產生新的bitstream時才需要做這步，單純為了答辯準備不一定需要重跑：

```powershell
vitis-run --mode hls --impl --config hls_config.cfg --work_dir hls_component
```

完成後核對：
```
hls_component\hls\impl\verilog\project.runs\impl_1\bd_0_wrapper_timing_summary_routed.rpt
```
重點看WNS（Worst Negative Slack）是否仍為正值（目前紀錄是+0.020ns，margin很小但有過）。若WNS變負，代表timing violated，需要重新評估clock period或pragma設定。

### 3.6 跑完後，請我做什麼

把你實際跑出來的`mismatches`/`max_abs_diff`/資源百分比/WNS數字貼給我，我會逐項跟現有報告比對，確認是否與既有結論一致（同機器、同工具版本通常應該完全一樣；若有落差，可能是Vitis HLS/Vivado版本差異，需要記錄下來而不是假裝沒看到）。

## 4. 真正還沒做、值得列入下一步的事

1. **跨seed資源/精度穩定性**：把`05c4_phc_stp_cosine_lr/`其他seed的checkpoint也跑一次`00_export_weights.py` → csim，確認`max_abs_diff`是否穩定在同一量級。這是用現有06f pipeline直接可做的事，**不需要重新設計HLS C++**，只要改`00_export_weights.py`指向的checkpoint路徑。
2. **PYNQ-Z2實機上板**：目前缺的最後一步，需要Vivado block design整合PS-PL介面，超出目前HLS IP驗證範圍。
3. （若你想嚴謹一點）**用獨立的第二份Python reference重新生成一份新的test vector集合**，而不是沿用06f已用過的25,600組——目前的測試向量是「validation split前200個sequence」，不是full validation set，如果reviewer問「你怎麼知道沒有挑到特別好算的子集」，目前沒有反駁證據。
