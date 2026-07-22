00_HFT_project/
├── 📂 artifacts/                    # 依產出目的整理的實驗成果與交付物
│   ├── README.md                    # artifacts 導覽與用途說明
│   ├── 📂 01_project_plans/         # 計畫、pivot memo、策略規劃
│   │   └── first_principles_pivot_plan.md
│   ├── 📂 02_model_runs/            # 訓練、驗證、sweep、跨市場測試輸出
│   │   ├── 📂 cme_eval_smoke/
│   │   └── 📂 cme_eval_transfer/
│   ├── 📂 03_latency_proof/         # 延遲敏感度與 latency proof 主成果
│   │   ├── 📂 latency_proof/
│   │   │   └── frozen_tcn_best_model.pt # 目前主線凍結模型
│   │   ├── 📂 latency_proof_A/          # 實驗比較組 A
│   │   ├── 📂 latency_proof_C/          # 實驗比較組 C
│   │   └── latency_method_comparison.csv
│   └── 📂 04_hardware_handoff/      # INT8 匯出、延遲估算、hls4ml 嘗試
│       ├── 📂 int8_export/
│       ├── 📂 latency_estimate/
│       └── 📂 hls4ml_attempt/
│
├── 📂 cpp_hls_prep/                 # C++ / HLS 前處理與 Vitis HLS 專區
│   ├── 📂 include/                  # 固定點模型 header 與 generated 權重
│   │   └── 📂 generated/            # 自動產生：weights.hpp + test_vectors.hpp
│   ├── 📂 src/                      # fixed_tcn / kernel / host / testbench 實作
│   ├── 📂 hls/                      # HLS TCL、PowerShell 與 project 輸出
│   ├── 📂 build/                    # CMake build 結果與本地執行檔
│   ├── CMakeLists.txt
│   └── README.md
│
├── 📂 data/                         # 資料湖：原始市場資料與處理後張量
│   ├── 📂 raw/
│   │   ├── 📂 CME/                  # CME Logic20 / SetProduct 原始資料
│   │   │   ├── 📂 ES_MES/           # ES + MES 各日資料 (20260319~20260325)
│   │   │   └── 📂 CL_MCL/           # CL + MCL 各日資料
│   │   ├── 📂 TW/                   # 台指 replay 資料
│   │   ├── mock_CL.csv
│   │   ├── mock_ES.csv
│   │   └── mock_NQ.csv
│   └── 📂 processed/
│       ├── CL_tensor.npz
│       ├── ES_tensor.npz
│       └── NQ_tensor.npz
│
├── 📂 logs/                         # HLS 執行 log
│   └── hls_run_tcl.log
│
├── 📂 meeting/                      # 對教授簡報、會議文件與視覺化輸出
│   ├── 📂 w5/
│   │   ├── w5_meeting_outline.md
│   │   ├── 📂 w5_chart/
│   │   └── 2026-03-27_professor_update_week5_pivot.pptx
│   ├── 📂 w7/
│   │   ├── w7meeting.md
│   │   ├── 📂 w6_result/            # CME 驗證圖表與 summary
│   │   └── W7_meeting.pptx
│   └── meeting_explanation.md
│
├── 📂 notebooks/                    # Notebook 工作區（目前幾乎未使用）
│
├── 📂 scripts/                      # 依功能與使用時機排序的可執行入口
│   ├── README.md                    # scripts 導覽與建議執行順序
│   ├── 📂 01_data_prep/             # mock data / 資料前處理 / 早期 pipeline
│   │   ├── generate_mock_data.py
│   │   └── run_pipeline.py
│   ├── 📂 02_train_eval/            # 模型訓練與直接驗證
│   │   ├── train_tw_baseline.py
│   │   ├── train_cme_baseline.py
│   │   └── evaluate_cme_frozen_model.py
│   ├── 📂 03_experiments/           # sweep / ablation / latency 實驗
│   │   ├── sweep_latency.py
│   │   ├── sweep_tw_baseline.py
│   │   └── sweep_tw_improvements.py
│   ├── 📂 04_analysis/              # 實驗後分析、模擬、延遲估算
│   │   └── analyze_*.py / simulate_int8_inference.py / estimate_tcn_latency.py
│   ├── 📂 05_hardware_handoff/      # INT8 匯出、位元驗證、HLS 橋接
│   │   ├── export_tcn_int8_artifacts.py
│   │   ├── verify_cpp_python_bit_accurate.py
│   │   ├── verify_hls_equivalence.py
│   │   └── attempt_hls4ml_conversion.py
│   └── 📂 06_reporting/             # 圖表、週會資產、簡報輸出
│       ├── generate_cme_w6_result_assets.py
│       ├── generate_meeting_charts.py
│       ├── generate_tradeoff_chart.py
│       ├── generate_failed_chart.py
│       └── generate_professor_ppt.py
│
├── 📂 src/                          # 核心研究與策略程式碼
│   ├── 📂 backtest/
│   │   ├── hybrid_market_maker.py  # Dumb Quoting + Smart Canceling 主策略
│   │   ├── event_driven.py         # Queue-aware 事件驅動模擬器
│   │   ├── baseline.py             # 基準策略與 replay 對照
│   │   ├── signals.py              # 訊號邏輯與 gating
│   │   └── symbol_calibration.py   # 商品參數校準
│   ├── 📂 data/
│   │   ├── loader.py               # 清理 / 對齊 / forward-fill
│   │   ├── tw_replay.py            # 台指 replay 載入與處理
│   │   └── cme_logic20.py          # CME Logic20 格式解析
│   ├── 📂 features/
│   │   ├── dimensionless_features.py
│   │   ├── execution_features.py
│   │   └── sliding_window.py
│   ├── 📂 models/
│   │   ├── tcn.py                  # TCN 架構定義（⭐ 最重要）
│   │   ├── hardware.py             # 硬體相關量化 / 匯出輔助
│   │   └── walk_forward_optimize.py
│   ├── 📂 training/
│   │   ├── datasets.py
│   │   ├── metrics.py
│   │   └── trainer.py
│   ├── 📂 utils/
│   │   └── constants.py
│   └── *_explanation.md            # 各模組對應說明文件
│
├── 📂 teaching/                     # 技術教學、研究脈絡與方法論文件
│   ├── 01_High_Frequency_Trading_Basics.md
│   ├── ...
│   ├── 18_CME_Logic20_Adapter_and_Direct_Validation.md
│   ├── 19_TW_TCN_CPP_to_HLS_Mainline.md
│   ├── 20_Model_Features_Params_and_Dataflow.md
│   ├── 21_Hardware_Handoff_hls4ml_CPP_and_Latency.md
│   └── 22_Vitis_HLS_2025_2_Step_By_Step.md
│
├── 📂 tests/                        # 單元測試與 smoke checks
│   ├── test_features.py
│   ├── test_execution_features.py
│   ├── test_event_backtest.py
│   ├── test_hybrid_market_maker.py
│   ├── test_model.py
│   ├── test_model_hardware.py
│   ├── test_cme_logic20.py
│   ├── test_tw_replay.py
│   ├── test_walk_forward_optimize.py
│   └── test_train_cme_baseline_smoke.py
│
├── 📂 vendor/                       # 外部對接 / 廠商交付說明
│   └── w7_vendor_brief.md
│
├── 📂 trash/                        # 暫存不用的大型檔案
│   └── CME_data.zip
│
├── 📂 Weekly_log/                   # 逐週開發記錄（主線進度）
│   ├── week1-2.md ~ week7.md        # 逐週報告
│   ├── w7_next_step.md              # 下週規劃
│   ├── TCN_method.md                # TCN 方法論總整理
│   ├── Backtest_method.md           # 回測方法總整理
│   ├── HLS_PROGRESS_FOR_TEACHER.md  # 給老師的 HLS 進度說明
│   └── HLS_PROGRESS_GUIDE.md        # HLS 工作進度指南
│
├── onboarding.md                    # ⭐ 新人快速上手指南（2026-04-14 新增）
├── guide.md                         # 專案總指南（舊版，保留參考）
├── file_structure.md                # 目錄結構（本文件）
└── requirements.txt                 # Python 套件需求
