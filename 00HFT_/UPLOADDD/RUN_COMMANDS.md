# FI-2010 GPU Step 0 Probe Commands

> ## 🔴 送出前檢查清單（2026-10-02 補）
>
> 逐項打勾，全部完成才 `sbatch`。
>
> - [ ] **1. 填入站台指令**：`slurm/step0_gpu_probe.sbatch` 內兩處 `REPLACE_ME_*`
>       （`--account`、`--partition`）已改成實際值。
>       查法：`sinfo -s`（分區）、`sacctmgr show assoc user=$USER format=account,partition`
> - [ ] **2. 載入環境**：同檔案中的 `module load` 區塊已取消註解並改成站台實際名稱，
>       或已自行 `export PYTHON_BIN=<有 CUDA PyTorch 的直譯器>`。
>       自檢：`$PYTHON_BIN -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
> - [ ] **3. 額度護欄已就位**：`--time=00:10:00`、`--wall-limit-seconds 420`。
>       最壞情況探針成本約 **0.12 SU**。**不要調高這兩個值。**
> - [ ] **4. 完整性核對**：上傳後在叢集端執行
>       `sha256sum -c <(grep -v '^#' GPU_SWEEP_UPLOAD_SHA256.txt)`，全部 OK。
>
> ### 探針回來只看一個數字
>
> 作業結束後看 stdout 最後的大框，或：
>
> ```bash
> cat results/gpu_probe/cap16_seed0/port_check_verdict.txt
> ```
>
> | 判定 | 意義 | 下一步 |
> |---|---|---|
> | `PASS` | epoch 1 的 `train_loss` 相對差 < 1e-3，移植正確 | 核可 Step 1（9 個 run 並行） |
> | `INVESTIGATE` | 相對差 1e-3 ～ 1e-1 | **不要開 Step 1**，先查資料順序與 batch 組成 |
> | `FAIL` | 相對差 > 1e-1，結構性差異 | **停止**，診斷配方移植 |
> | `INCOMPLETE` | 缺 epoch 1，probe 被 watchdog 截斷 | 代表 GPU 太慢 → 退回 CPU 的 5-run 方案 |
>
> CPU 參考值：**epoch 1 `train_loss` = 1.0716215017748378**
>
> ⚠️ **epoch 2、3 的差異會放大，那是正常的，不可當判準。**
> ⚠️ `val_accuracy` 在 epoch 1 只有 0.3597，是極早期噪音指標，不可當判準。
>
> ### 可能遇到的失敗（都不會浪費額度）
>
> | 症狀 | 原因 | 處理 |
> |---|---|---|
> | 作業被排程器直接拒絕 | account / partition 未填或填錯 | 補正後重送，未消耗額度 |
> | 幾秒內失敗，log 顯示 import error | `PYTHON_BIN` 沒有 CUDA PyTorch | 補 module load 後重送 |
> | RuntimeError 提到 deterministic 實作 | 某個運算沒有確定性 CUDA 版本 | 改 `use_deterministic_algorithms(True, warn_only=True)`，並在報告註明確定性為部分；**不要默默移除此旗標** |
> | 作業跑滿 10 分鐘被砍 | GPU 上仍太慢 | 這就是答案：退回 CPU 5-run 方案 |

---


This package is for **Step 0 only**. It must not be used to launch the nine-run capacity sweep before the probe result is reviewed.

## 1. Upload from the Windows workstation

Replace `<USER>`, `<CLUSTER>`, and the remote path as needed. Run in PowerShell from the repository root:

```powershell
scp -r .\00HFT_\UPLOADDD <USER>@<CLUSTER>:~/fi2010_gpu_probe
```

When VS Code Remote SSH is already connected, uploading the `UPLOADDD` directory through the Explorer is equivalent. Preserve the `data/cache/` and `slurm/` subdirectories.

## 2. Enter the uploaded directory

Run these commands in the cluster shell:

```bash
cd ~/fi2010_gpu_probe
pwd
find . -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
```

## 3. Verify transfer integrity

```bash
sha256sum -c GPU_SWEEP_UPLOAD_SHA256.txt
```

All entries must print `OK`. Do not submit if any entry fails.

## 4. Select the existing CUDA Python environment

Do not install packages into the login node unless required by cluster policy. Activate the cluster's existing CUDA PyTorch environment, then run:

```bash
which python
python -V
python -c 'import torch, numpy, sklearn; print({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": numpy.__version__, "sklearn": sklearn.__version__, "cuda_available_on_login_node": torch.cuda.is_available()})'
export PYTHON_BIN="$(command -v python)"
```

`cuda_available_on_login_node` may be false on clusters that expose GPUs only inside jobs. The job itself performs the authoritative CUDA check and refuses CPU fallback.

If the cluster requires a partition or account, add the site-provided directives to `slurm/step0_gpu_probe.sbatch` before submission. Do not guess them.

## 5. Submit the three-epoch probe

```bash
sbatch --export=ALL,PYTHON_BIN="$PYTHON_BIN" slurm/step0_gpu_probe.sbatch
```

Record the returned job ID as `<JOB_ID>`.

## 6. Monitor without launching any other workload

```bash
squeue -j <JOB_ID> -o '%.18i %.12P %.28j %.8T %.10M %.10l %.6D %R'
tail -f probe-<JOB_ID>.out
```

If the job fails, inspect:

```bash
cat probe-<JOB_ID>.err
cat probe-<JOB_ID>.out
find results/gpu_probe -maxdepth 4 -type f -print | sort
```

## 7. Collect authoritative accounting after completion

Run only after the job leaves the queue:

```bash
sacct -j <JOB_ID> -X -P --format=JobID,JobName,State,Elapsed,ElapsedRaw,AllocTRES,Billing,TotalCPU,MaxRSS
scontrol show job <JOB_ID>
scontrol show config | grep -i TRESBillingWeights || true
```

The measured one-GPU consumption in GPU-hours is also written to:

```bash
cat results/gpu_probe/slurm_accounting/probe_gpu_time_<JOB_ID>.csv
cat results/gpu_probe/slurm_accounting/probe_sacct_in_job_<JOB_ID>.txt
```

Use the cluster's documented conversion from GPU-hours or `Billing` to SU. Do not invent an SU conversion factor if the scheduler output does not define it.

## 8. Inspect the required Step 0 outputs

```bash
cat results/gpu_probe/cap16_seed0/environment.json
cat results/gpu_probe/cap16_seed0/probe_epoch_metrics.csv
cat results/gpu_probe/cap16_seed0/cpu_gpu_first3_trajectory_comparison.csv
cat results/gpu_probe/cap16_seed0/run_metrics.csv
cat results/gpu_probe/cap16_seed0/run_status.json
```

The trajectory comparison contains CPU value, GPU value, signed delta, absolute delta, and relative absolute delta for train loss and every validation metric at epochs 1-3. It intentionally does not hide device-dependent differences behind a single pass/fail threshold.

## 9. Estimate the nine-run Step 1 cost

After reading `seconds_per_epoch` from `run_metrics.csv`, combine it with the existing CPU best-epoch counts for hidden16 and hidden64. Report both:

```text
serial_gpu_hours = sum(estimated_run_seconds) / 3600
one_gpu_nine_process_wall_hours = max(estimated_run_seconds) / 3600
```

For the requested single-GPU concurrent design, billed GPU-hours are based on allocation wall time, not the sum of nine process wall times. Include scheduler overhead and the observed slowdown from concurrency once Step 1 is eventually authorized. Do not submit Step 1 during this step.

## 10. Download the probe evidence

From local PowerShell:

```powershell
scp -r <USER>@<CLUSTER>:~/fi2010_gpu_probe/results/gpu_probe .\00HFT_\UPLOADDD\downloaded_results\
scp <USER>@<CLUSTER>:~/fi2010_gpu_probe/probe-<JOB_ID>.out .\00HFT_\UPLOADDD\downloaded_results\
scp <USER>@<CLUSTER>:~/fi2010_gpu_probe/probe-<JOB_ID>.err .\00HFT_\UPLOADDD\downloaded_results\
```

## Hard stop

Stop after downloading and reporting the probe. Do not create or submit the nine-run Step 1 job until the user reviews the GPU/CPU trajectory and SU estimate.

---

## 🖥️ NCSA Delta 送出指令（2026-10-02 實測環境）

### 1. 取得未截斷的帳號名稱

`sacctmgr` 預設把欄位截斷成 `bche-delt+`，**不要照著打**。加寬欄位：

```bash
sacctmgr -nP show assoc user=$USER format=account | grep -v '^noalloc$'
```

### 2. 送出（帳號自動代入，不手打）

```bash
cd ~/project/UPLOADDD          # 改成你實際的上傳路徑

ACCT=$(sacctmgr -nP show assoc user=$USER format=account | grep -v '^noalloc$' | head -1)
echo "account = $ACCT"         # 先確認不是空的、不是 noalloc

module purge
module load anaconda3_gpu      # 依 Delta 實際可用版本調整，見下方步驟 3
$(command -v python) -c "import torch;print(torch.__version__, torch.cuda.is_available())"

sbatch --account="$ACCT" --export=ALL,PYTHON_BIN="$(command -v python)" \
       slurm/step0_gpu_probe.sbatch
```

分區已寫在指令檔中（`gpuA40x4`），不需在命令列重複指定。

### 3. module 名稱怎麼找

```bash
module avail 2>&1 | grep -i -E "anaconda|pytorch|cuda"
```

挑一個含 CUDA PyTorch 的環境，確認 `torch.cuda.is_available()` 為 `True` 再送。
**若為 `False`，不要送出**——作業會在環境擷取階段失敗。

### 4. 為什麼選 gpuA40x4 而不是預設的 gpuA100x4

| 分區 | 計費係數 | 本工作的實際差異 |
|---|---:|---|
| `gpuA100x4`（預設 `*`） | **1.0** | — |
| **`gpuA40x4`** | **0.5** | 模型僅 1,731 參數、受 kernel-launch 限制，速度差異極小 |

Delta 的 1 SU = 1 個 A100 GPU-hour。你的學期額度 10 SU 在 A40 上等於 **20 GPU-hours**。

**探針最壞情況成本：10 分鐘 × 0.5 = 約 0.083 SU**（不到額度的 1%）。

> ⚠️ Delta 的計費依據是**為作業保留的資源 × 實際耗用時間**，不是你實際用了多少。
> 所以 `--gres=gpu:1`、`--cpus-per-task=4`、`--mem=16G` 都不要往上加——
> 現在的設定已在單張 A40 的配額內，提高任何一項都可能被計成更多 GPU。

若 `gpuA40x4` 排隊太久，可改用 `gpuA40x4-interactive`（1 小時上限，對 10 分鐘的探針綽綽有餘），
但**送出前請先自行確認該分區的計費係數**，官方文件未明列。

### 5. 看結果

```bash
squeue -u $USER                                   # 排隊/執行狀態
cat probe-<JOB_ID>.out | tail -40                 # 判定大框在最後
cat results/gpu_probe/cap16_seed0/port_check_verdict.txt
sacct -j <JOB_ID> -X -P --format=JobID,State,ElapsedRaw,AllocTRES,Billing
```

最後一行的 `Billing` 欄就是這次實際扣掉的額度。

---

## Step 0.5：並行效率探測（決定 Step 1 成本）

Step 0 探針已 PASS（epoch 1 相對差 1.6e-08，val 指標與 CPU 完全相同），
但實測 **62.0 秒／epoch**，意味著：

| Step 1（5 runs：hidden32 x3 + hidden64 x2） | 估計 SU |
|---|---:|
| 序列執行 | **約 2.8** ❌ 超出 1 SU 目標 |
| 5-way 並行且重疊良好 | **約 0.8–1.0** ✅ |

這兩者差 3 倍，必須用實測決定。本探測成本：**最壞 12 分鐘 × 0.5 = 0.1 SU**。

```bash
cd ~/project
ACCT=$(sacctmgr -nP show assoc user=$USER format=account | grep -v '^noalloc$' | head -1)
sbatch --account="$ACCT" --export=ALL,PYTHON_BIN="$(command -v python)" \
       slurm/step0b_gpu_concurrency_probe.sbatch
```

結束後：

```bash
cat results/gpu_concurrency_probe/concurrency_summary.json
tail -20 concprobe-<JOB_ID>.out
```

### 判定

| verdict | worst/solo | 意義 |
|---|---|---|
| `CONCURRENCY_EFFECTIVE` | ≤ 1.35x | 並行有效 → Step 1 用 5-way 並行 |
| `CONCURRENCY_PARTIAL` | 1.35–2.5x | 部分重疊 → 回報數字，人工權衡 |
| `CONCURRENCY_SERIALIZED` | > 2.5x | 被序列化 → 並行無益，改走縮減網格 |

> 本探測使用 **seed 90–94**（不在正式網格 {0,1,2} 內），且結束時會自動把拋棄式
> 輸出從 `results/gpu_sweep/` 搬到 `results/gpu_concurrency_probe/`，
> **不會害 Step 1 撞上 FileExistsError**。

