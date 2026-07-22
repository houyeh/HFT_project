import os
import sys
from pathlib import Path
import numpy as np
import torch
import hls4ml

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

from src.models.tcn import HLSFriendlyTCNClassifier

def main():
    """
    主程式：
    1. 將已經訓練完成的 PyTorch TCN 模型權重載入。
    2. 定義 hls4ml 的轉換配置，將 PyTorch 模型轉換為硬體友善的 C++ (HLS) 專案。
    3. 特殊設定 io_stream 介面，以支援 HFT 中連續不斷的資料流 (Streaming)。
    """
    print("Loading PyTorch TCN Model...")
    device = torch.device("cpu") # 轉換硬體時建議用 CPU，避免 GPU Tensor 格式問題
    
    # 建立與訓練時一模一樣的模型架構
    model = HLSFriendlyTCNClassifier(
        num_inputs=8, 
        num_classes=3, 
        channel_list=[16, 16, 16],
        kernel_size=3,
        dropout=0.1
    ).to(device)
    
    # 載入權重 (等訓練完全結束後使用)
    weights_path = project_root / "new_era" / "005_tcn_weights.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print("Weights loaded successfully.")
    else:
        print("Warning: Weights not found! Using random weights for test conversion.")
        
    model.eval() # 必須設定為評估模式，關閉 Dropout
    
    # hls4ml 進行 PyTorch 轉換時需要明確指定輸入的形狀 (Shape)
    # 我們的 TCN 輸入是 (Batch_size, Features, Time) -> (1, 8, 100)
    input_shape = (8, 100)
    
    # 建立 hls4ml 配置 (設定為 16-bit 定點數)
    # 若未來想轉為 INT8，只需把 Precision 改為 ap_fixed<8,2> 等 8-bit 設定
    config = hls4ml.utils.config_from_pytorch_model(
        model, 
        input_shape=input_shape, 
        default_precision='ap_fixed<16,6>', # 16位元儲存，其中6位元給整數，10位元給小數
        granularity='name'
    )
    
    # ========================================================
    # 🌟 1. 調整 DSE 參數：資源重用率 (ReuseFactor)
    # ========================================================
    # 原本設為 1 時，DSP 消耗高達 3425 個 (1556%)，PYNQ-Z2 塞不下。
    # 我們將 ReuseFactor 調高到 32，用時間換取空間，大幅降低硬體資源消耗！
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['ReuseFactor'] = 32
        
    print("HLS Configuration:")
    print(config)
    
    output_dir = str(project_root / "new_era" / "007_hls4ml_tcn_prj")
    
    print("\nStarting hls4ml PyTorch to C++ Conversion...")
    # 開始執行模型轉換
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model=model,
        hls_config=config,
        output_dir=output_dir,
        part='xc7z020clg400-1', # PYNQ-Z2 晶片型號
        io_type='io_stream'     # 【關鍵】高頻交易必須使用 Streaming 介面來處理源源不絕的 Tick 資料
    )
    
    # ========================================================
    # 🌟 2. 注入測試數據 (tb_data) 以支援 C-Sim/Co-Sim 驗證
    # ========================================================
    # 為了能跑 Co-simulation，我們必須讓 hls4ml 自動生成 tb_input_features.dat
    # 我們隨機生成一筆形狀為 (1, 8, 100) 的虛擬市場資料 (X_test) 來當作測試台的輸入
    X_test = np.random.rand(1, 8, 100).astype(np.float32)
    
    try:
        print("\n編譯 C++ 模型以準備生成 Testbench...")
        hls_model.compile()
        print("編譯成功！開始執行預測以產生 tb_data...")
        # 這一行超級關鍵！呼叫 predict 會強迫 hls4ml 將輸入寫進 tb_data 資料夾，供 Vitis HLS 模擬用
        y_hls = hls_model.predict(X_test)
        print("成功生成測試數據！C-Simulation/Co-Simulation 現在可以完美運作了。")
    except Exception as e:
        print(f"\n⚠️ 警告: C++ 編譯失敗 ({e})。")
        print("這通常是因為在一般的 Windows 命令字元中找不到 g++ 編譯器。")
        print("但不影響 IP 導出！你仍然可以直接使用 Vitis 進行 Synth 和 Export！")
    
    # 【自動修改 build_prj.tcl】重新開啟 csim 與 cosim，讓它能做完整驗證
    tcl_path = Path(output_dir) / "build_prj.tcl"
    if tcl_path.exists():
        tcl_code = tcl_path.read_text()
        tcl_code = tcl_code.replace("csim       0", "csim       1")
        tcl_code = tcl_code.replace("cosim      0", "cosim       1")
        tcl_path.write_text(tcl_code)
        print("已將 build_prj.tcl 中的 csim 與 cosim 重新開啟！")
    
    print(f"\n🎉 TCN Conversion Completed!")
    print(f"FPGA C++ 專案已生成於：{output_dir}")
    print("請在 Vitis 環境中執行： vitis-run --mode hls --tcl build_prj.tcl")

if __name__ == "__main__":
    # 程式進入點
    main()
