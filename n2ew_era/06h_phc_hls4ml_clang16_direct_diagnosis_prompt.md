# Prompt：PHC hls4ml 蒸餾學生模型 — 繞過 Vitis wrapper，直接跑 clang-16 拿真實錯誤訊息

> 給 coding agent 的自包含任務說明。直接複製整份貼給 agent，不需要補充對話脈絡。
> 這是診斷任務，不是修改模型/架構的任務。上一輪（升級 hls4ml 到 GitHub main `1.4.0.dev32+gb90fb067`）已經確認繞過了舊的 `Syntax check failed in clang-tidy` 卡點，但在 Vitis 呼叫 clang-16 這一步失敗，**沒有任何診斷文字**（"zero-diagnostics"）。這次的目標是拿到 clang-16 真正吐出來的錯誤，不是再猜一次版本組合。

---

## 背景

- hls4ml 版本已升級：`1.4.0.dev32+gb90fb067`（GitHub main）。
- 舊卡點（`Syntax check failed in clang-tidy`，出現在 `nnet_utils/*.h` 的 pragma 警告後）**已消失**——代表升版方向是對的，值得繼續往這條路走，不是走錯路。
- 新卡點：Vitis 呼叫 clang-16 做 C-Synthesis 前端編譯時失敗，退出碼非零，但**沒有任何錯誤訊息文字**。`csynth_report_count` 仍是 0。
- 「zero-diagnostics」這個特徵本身要當一條線索看：正常的編譯錯誤（語法錯誤、型別不合）幾乎一定會印出至少一行 `error:`。什麼都沒印，通常代表下列其中一種：
  1. clang-16 這個子行程被作業系統強制終止（segfault、stack overflow、OOM-kill），錯誤訊息根本來不及印出來或印到了但被 Vitis 的 TCL 包裝腳本吞掉沒轉發。
  2. Vitis 的 log 擷取機制本身把 stderr 導向了別的地方（沒有正確 pipe 到目前看到的 build log 檔）。
  3. clang-16 收到的呼叫參數本身就有問題（缺 include path、旗標不合法），導致它連嘗試 parse 都沒開始就退出，這種情況通常*會*印訊息，機率比前兩者低，但不能排除。

前一輪（06H 原始嘗試）留下的線索也支持「模型太複雜、模板展開太深」這個方向：不 clamp 精度時，型別膨脹到 `ap_fixed<156,76>`；clamp32 之後 instruction 數在 unroll/inline 階段仍衝到十幾萬（`172,096 instructions` 是舊版 hls4ml 的數字，新版可能不同但同樣量級可能還在）。深度模板展開 + 巨大 instruction 數是 clang 前端 crash（尤其 stack overflow）的典型觸發條件，跟「zero-diagnostics」的特徵吻合。

## 這次要做什麼：不要再靠 Vitis 包裝跑，直接手動呼叫 clang-16

### 步驟 1：找到失敗當下實際的 clang 呼叫指令

hls4ml build 出來的專案目錄下（`06h_phc_hls4ml_distilled_student/hls4ml_prj/<project>/`）應該有 `firmware/<project>.cpp` 和一堆 `firmware/nnet_utils/*.h`。Vitis csynth 階段呼叫 clang-16 時的完整指令列（include path、`-std=`、`-D` 巨集、優化旗標）通常可以從下列地方之一找到：

- 這次 build 產生的完整 log 檔（不要只看報告裡摘錄的幾行，把整個 `06h_build_full_log_*.txt` 或同等的新檔案從頭到尾看一遍，找 `clang` 或 `Analyzing design file` 附近的完整指令）。
- 如果 log 裡沒有完整指令列，用 process monitor / 或在呼叫 `vitis-run` 前後加上 shell 層級的指令記錄（例如 Windows 下用 `Process Monitor` 或簡單在批次檔裡加 `echo` 攔截），抓出實際執行的 clang 指令。
- 或者查 Vitis 專案裡 `.autopilot` 或 `solution1/.autopilot/db/` 底下有沒有記錄編譯指令的 log/cfg 檔（AutoPilot 是 Vitis HLS 內部代號，這類檔案通常會列出實際呼叫的 compiler flags）。

### 步驟 2：直接執行同一條 clang-16 指令（不透過 vitis-run / TCL）

找到 `clang-16` 執行檔本身的路徑（通常在 `C:\AMDDesignTools\2025.2\...` 底下某個 LLVM/clang 相關子目錄，用 `dir /s clang-16.exe` 或等效指令找）。手動組出跟 Vitis 用的同一條指令（同樣的 `firmware/<project>.cpp`、同樣的 include path、同樣的 `-D` 巨集），直接在終端機執行，不透過 Vitis 的 TCL 腳本。

```bash
"C:\path\to\clang-16.exe" <從步驟1複製出來的完整參數> firmware/<project>.cpp 2>&1 | tee clang_direct_output.txt
```

**這一步的目的只有一個：拿到 clang 未經 Vitis 過濾的原始 stdout/stderr 全文，包含退出碼（`echo %ERRORLEVEL%` 或 `echo $?`）。**

### 步驟 3：如果直接呼叫也是靜默失敗（沒有輸出、行程直接消失）

這代表確實是 crash（segfault/stack overflow），不是單純的錯誤訊息被吞掉。這種情況下：

- 檢查 Windows 事件檢視器（Event Viewer → Windows Logs → Application）在同一時間戳附近有沒有 clang-16.exe 的當機記錄（Application Error / Faulting module）。
- 嘗試把輸入檔案換成**更簡化的版本**做二分排查：先只編譯 `firmware/nnet_utils/` 底下的單一標頭檔（用 `-fsyntax-only` 只做語法檢查、不做完整編譯，加快排查速度），一個一個排除，找出哪個 `nnet_*.h` 或哪個由 hls4ml 生成的巨大模板展開觸發了 crash。
- 如果懷疑是模板展開/instruction數過大導致 stack overflow：嘗試加大 clang 的呼叫堆疊限制（如果是 Windows，`clang-16` 本身通常不直接吃 `ulimit`，需要查 clang 是否有 `-fno-elide-constructors` 之類影響遞迴展開深度的旗標，或考慮下一輪換更保守的精度設定，見下方備案）。

## 備案：如果確認是模板展開過深/資源耗盡（不是 hls4ml 版本問題）

若步驟 2/3 顯示問題本質是「這個特定學生模型（~25 層 Dense/Activation 串接，精度鏈很深）展開後對 clang 前端太重」，那麼繼續升 hls4ml 版本大概率沒用（因為根因不在 hls4ml 的 pragma 生成邏輯，而在生成出的 C++ 本身太複雜）。此時下一步應該是**進一步降低精度或簡化學生模型**，而不是再換 hls4ml 版本：

- 把 clamp 從 32 bit 降到 16 bit（`fixed<16,8>` 或更窄），重新跑一次同樣的 direct clang 呼叫，看 crash 是否消失。
- 如果模型本身的深度（9-step PHC-STP skeleton，約 25 層 Dense/Activation）是問題，考慮用 hls4ml 的 `reuse_factor` 設定換取更小的展開規模（reuse_factor 越高，同一份硬體資源被時間複用的次數越多，生成的 C++ 展開規模通常越小）。
- 這兩個都只是「讓 clang 願意編譯」的權宜之計，不代表原本 06G/06H 想驗證的東西（hls4ml 對這類架構的相容性）有真的解決，如果走到這一步，誠實回報「hls4ml 路線在目前工具版本組合下，即使解決了 pragma 相容性，仍受限於生成程式碼複雜度，需要進一步簡化模型才可能成功」。

## 絕對不動的部分

- 不要重新訓練、不要碰 NEW5、不要改教師模型權重。
- 不要動 `06f_phc_lite_manual_hls`、`06g_phc_hls4ml_a3d3_retry` 的既有凍結產物。
- 這次先不要嘗試同時改 hls4ml 版本又改精度設定——一次只變一個變數，否則出結果也不知道是哪個因素造成的。

## 驗收標準

1. 是否拿到 clang-16 未經 Vitis 過濾的原始輸出（有輸出內容 vs. 直接崩潰無輸出，兩種都算「拿到診斷結果」，都可以停下回報）。
2. 若有錯誤文字：完整貼出，不要摘要。
3. 若確認是崩潰（無輸出、有作業系統層級的當機記錄）：附上 Event Viewer 或等效證據。
4. 若做了二分排查：附上「哪個檔案/哪個規模觸發崩潰」的具體結論。

## 回報格式

- 直接呼叫 clang-16 用的完整指令列。
- 完整、未經摘要的輸出（成功編譯出提示訊息，或崩潰的系統記錄）。
- 明確結論句三選一：「(a) 找到真正的編譯錯誤，內容是 XXX」／「(b) 確認是 clang-16 崩潰（segfault/OOM），不是可修的語法錯誤」／「(c) 降精度後崩潰消失，問題是模型複雜度不是 hls4ml 版本」。
- 如果結論是 (b) 或 (c)：明確建議「這條 hls4ml 路線是否還值得投入時間」，附上到目前為止已經花的輪數（06G 四條路徑、06H 原始+多輪 clamp 嘗試、這次版本升級+direct clang 診斷）當作時間成本的參考，讓使用者自己決定要不要繼續，還是把力氣轉回 TCN pipeline 或其他材料。
