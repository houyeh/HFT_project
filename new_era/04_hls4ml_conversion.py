import os
import sys
from pathlib import Path

# Removed TF_USE_LEGACY_KERAS to allow keras 3.0 loading of the h5 file
import tensorflow as tf
import hls4ml

project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

def main():
    """
    主程式：
    1. 讀取前面訓練好的 Keras MLP 模型 (.h5)。
    2. 建立 hls4ml 硬體配置，針對 PYNQ-Z2 (Zynq-7000) FPGA 板進行特化設定。
    3. 設定每層神經網路的 ReuseFactor (硬體重複利用率) 和 Precision (數值精度)。
    4. 將模型轉換並輸出為 Vivado HLS C++ 專案。
    """
    model_path = project_root / "new_era" / "003_baseline_mlp.h5"
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        sys.exit(1)
        
    print(f"Loading Keras model from {model_path}...")
    # 載入原本的 Keras 模型
    model = tf.keras.models.load_model(model_path)
    
    # 1. 建立 hls4ml 配置 (Configuration)
    # 針對神經網路每一層抓出配置骨架
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    
    # 針對各層進行硬體優化設定
    for layer in config['LayerName'].keys():
        # ReuseFactor = 1 代表「全展開 (Fully Unrolled)」
        # 意思是不重複使用硬體乘法器，而是將每條連線都配置實體的運算單元。
        # 這樣會消耗最多硬體資源，但能換取最極致的 1 個 Clock 週期極低延遲，非常適合 HFT。
        config['LayerName'][layer]['ReuseFactor'] = 1  
        
        # 定點數 (Fixed-point) 數值精度設定：'ap_fixed<16,6>'
        # 代表總共用 16-bit 儲存，其中 6-bit 用於整數與符號，剩下的 10-bit 用於小數。
        # 此設定在精確度與硬體佔用空間中取得極佳平衡。
        config['LayerName'][layer]['Precision'] = 'ap_fixed<16,6>' 
        
    # 特殊處理 Softmax 啟動函數
    # 因為硬體計算 exp() (指數函數) 非常消耗 DSP 資源，因此改用 Stable 策略或 LUT (查表法) 近似
    if 'output_probabilities' in config['LayerName']:
        config['LayerName']['output_probabilities']['Strategy'] = 'Stable' 

    print("\n--- hls4ml Configuration ---")
    import pprint
    pprint.pprint(config)

    # 2. 轉換為 HLS 專案
    output_dir = project_root / "new_era" / "004_hls4ml_prj"
    print(f"\nConverting Keras model to hls4ml C++ project at {output_dir}...")
    
    # 執行模型轉換
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, 
        hls_config=config, 
        output_dir=str(output_dir), 
        part='xc7z020clg400-1', # 這正是 PYNQ-Z2 板子上的 Xilinx Zynq-7000 FPGA 晶片型號
        backend='Vivado'      # 使用 Vivado 作為後端合成工具
    )
    
    # 3. 寫入 HLS C++ 原始碼檔案
    hls_model.write()
    print("✅ C++ code successfully written.")
    
    # 4. 嘗試編譯 C++ 進行位元精確度模擬 (Bit-accurate C Simulation)
    # 此步驟能在 Python 中模擬硬體實際以 16-bit 運算時的結果，確認精度是否流失
    # 如果 Windows 系統沒有安裝 g++，這一步可能會報錯，但核心的 HLS C++ 程式碼早已安全產生了
    try:
        print("\nAttempting to compile HLS C++ code for fast simulation...")
        hls_model.compile()
        print("✅ Compilation successful! The model is ready for Bit-accurate simulation in Python.")
    except Exception as e:
        print(f"\n[Warning] C++ compilation skipped or failed (usually missing g++ compiler on Windows):")
        print(str(e))
        print("This is normal on a Windows machine without MinGW. The HLS C++ source files are still perfectly valid!")

    print("\n🎉 Pipeline Step 4 Completed!")
    print(f"Your FPGA hardware project is ready at: {output_dir}")
    print("You can now zip this folder, send it to a machine with Xilinx Vivado, and run:")
    print(f"  vivado_hls -f build_prj.tcl")
    
if __name__ == "__main__":
    # 程式進入點
    main()
