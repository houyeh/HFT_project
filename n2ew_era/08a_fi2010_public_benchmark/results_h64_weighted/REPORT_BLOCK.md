| seed | checkpoint | accuracy % | weighted-F1 % | macro-F1 % |
|---:|---|---:|---:|---:|
| 0 | `phc_fi2010_h50_seed0_raw_v3_hidden64_capacity.pt` | 66.12 | 66.32 | 64.43 |
| 1 | `phc_fi2010_h50_seed1_raw_v3_hidden64_capacity.pt` | 64.92 | 65.17 | 63.33 |
| 2 | `phc_fi2010_h50_seed2_raw_v3_hidden64_capacity.pt` | 65.58 | 65.75 | 63.87 |
| **mean ± std** | 3 seeds | **65.54 ± 0.60** | **65.74 ± 0.57** | **63.88 ± 0.55** |

### 容量對照（3 seeds vs 3 seeds）

| 設定 | 參數量 | accuracy % | weighted-F1 % | macro-F1 % |
|---|---:|---:|---:|---:|
| hidden16 | 2,132 | 66.45 ± 1.07 | 66.50 ± 1.15 | 63.97 ± 1.45 |
| hidden64 | 23,876 | 65.54 ± 0.60 | 65.74 ± 0.57 | 63.88 ± 0.55 |
| **差（hidden64 - hidden16）** | **11.2×** | **-0.91 pp** | **-0.76 pp** | **-0.09 pp** |

### 檢定力陳述

- **accuracy**：差 -0.91 pp，Welch t = -1.29（df = 3.1，不顯著）；95% CI = [-2.88, +1.06] pp —— 可排除大於 **+1.06 pp** 的提升
- **weighted-F1**：差 -0.76 pp，Welch t = -1.02（df = 2.9，不顯著）；95% CI = [-2.82, +1.30] pp —— 可排除大於 **+1.30 pp** 的提升；與 DeepLOB 的 13.85 pp 差距相比，上界僅佔 9%
- **macro-F1**：差 -0.09 pp，Welch t = -0.10（df = 2.6，不顯著）；95% CI = [-2.58, +2.39] pp —— 可排除大於 **+2.39 pp** 的提升

一致性閘：所有 checkpoint 的 weighted recall 與 accuracy 最大絕對差 = `0.00e+00`
