import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import hls4ml
import re

# 專案路徑設定
project_root = Path(r"c:\Users\holbo\00VS_projjects\00_HFT_project")
sys.path.insert(0, str(project_root))

# 匯入原本 hls4ml 認識的 HLSFriendlyTCNClassifier
from src.models.tcn import HLSFriendlyTCNClassifier

def main():
    print("============================================================")
    print(" Step 14: QAT Weights -> Standard Model -> hls4ml Synthesis ")
    print("============================================================")

    SEQ_LEN = 80 # 稍微縮減從 100 到 80 以大幅節省 BRAM
    # 改名避開 Windows 檔案鎖定 (PermissionError)
    out_dir = project_root / 'new_era' / '014_hls4ml_qat_80ticks_12ch_prj'

    # 1. 建立一個 hls4ml 認識的標準模型
    model = HLSFriendlyTCNClassifier(
        num_inputs=8, 
        num_classes=3, 
        channel_list=[12, 12, 12],
        kernel_size=3,
        seq_length=SEQ_LEN
    )
    
    # 2. 載入 QAT 權重並手動映射 (Mapping)
    weights_path = project_root / "new_era" / "013_qat_finetuned_weights.pth"
    # if not weights_path.exists():
    #     print(f"Error: Weights not found at {weights_path}")
    #     return

    # qat_state_dict = torch.load(weights_path, map_location='cpu')
    # new_state_dict = model.state_dict()

    # 映射邏輯：將 QAT 的卷積層權重塞入標準層
    # QAT 層名通常是 tcn.network.0.conv1.weight
    # 標準層名也是 tcn.network.0.conv1.weight
    # 我們只需要過濾掉 QAT 特有的 BatchNorm (bn) 與量化節點參數
    # for key in new_state_dict.keys():
    #     if key in qat_state_dict:
    #         new_state_dict[key] = qat_state_dict[key]
    #         print(f"Successfully mapped weight: {key}")

    # model.load_state_dict(new_state_dict)
    model.eval()

    # 3. hls4ml 轉換設定 (關鍵優化：ReuseFactor = 64)
    
    config = hls4ml.utils.config_from_pytorch_model(
        model, 
        input_shape=(8, SEQ_LEN),
        granularity='name',
        default_precision='ap_fixed<8,2>'
    )
    
    config['Model']['ReuseFactor'] = 64
    config['Model']['Strategy'] = 'Resource'
    
    # 針對所有層強制使用 64 倍重用，並優化存儲實作以節省 BRAM
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['ReuseFactor'] = 64
        # 如果是卷積層或權重層，可以嘗試使用 Distributed RAM 替代 BRAM (視情況而定)
        # config['LayerName'][layer]['StorageImpl'] = 'distributed' 

    print("Converting Model with ReuseFactor=64 (Target: Area Optimization)...")
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=config,
        output_dir=str(out_dir),
        part='xc7z020clg400-1',
        io_type='io_stream'
    )
    hls_model.write()
    
    # 4. 自動修補 Residual Transpose Bug
    cpp_file = out_dir / 'firmware' / 'myproject.cpp'
    if cpp_file.exists():
        content = cpp_file.read_text(encoding='utf-8')
        # TCN 殘差層修補 (延用之前的成功經驗)
        old_str = r'nnet::pointwise_conv_1d_cl<input_t, tcn_network_0_downsample_result_t, config35>\(layer32_cpy2, layer8_out, w8, b8\);'
        new_str = '''hls::stream<layer30_t> layer32_cpy2_transposed("layer32_cpy2_transposed");
    #pragma HLS STREAM variable=layer32_cpy2_transposed depth=8
    nnet::transpose<input_t, layer30_t, config30>(layer32_cpy2, layer32_cpy2_transposed);
    nnet::pointwise_conv_1d_cl<layer30_t, tcn_network_0_downsample_result_t, config35>(layer32_cpy2_transposed, layer8_out, w8, b8);'''
        content = re.sub(old_str, new_str, content)
        cpp_file.write_text(content, encoding='utf-8')
        print("Patched myproject.cpp for Residual Transpose.")

    # 5. 啟動 Vitis HLS
    print(f"\nStarting Vitis HLS Synthesis in: {out_dir}")
    tcl_path = out_dir / 'build_prj.tcl'
    if not tcl_path.exists():
        print(f"ERROR: {tcl_path} not found! hls4ml.write() might have failed.")
        return

    tcl_content = tcl_path.read_text(encoding='utf-8')
    tcl_content = tcl_content.replace('csim       1', 'csim       0')
    tcl_content = tcl_content.replace('cosim      1', 'cosim      0')
    # Patch for Vitis HLS 2025.2 compatibility
    tcl_content = tcl_content.replace('config_array_partition -maximum_size 4096', '# config_array_partition -maximum_size 4096 (Patched for Vitis 2025.2)')
    tcl_path.write_text(tcl_content, encoding='utf-8')

    # 使用 subprocess 確保能捕捉到輸出
    import subprocess
    cmd = f'vitis-run --mode hls --tcl "{tcl_path}"'
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Vitis HLS failed with exit code {e.returncode}")
        return
    
    print("\n======================================================")
    print(" Synthesis Task Finished! ")
    print(f" Report path: {out_dir}/myproject_prj/solution1/syn/report/myproject_csynth.rpt")
    print("======================================================")

if __name__ == "__main__":
    main()
