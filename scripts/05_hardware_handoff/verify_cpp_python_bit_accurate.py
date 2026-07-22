"""scripts/05_hardware_handoff/verify_cpp_python_bit_accurate.py

Verify that the C++ inference matches the Python INT8 simulation bit-for-bit.

Note: the reference Python INT8 simulator lives in `scripts/04_analysis/`.
This script adjusts `sys.path` so it can be executed from the repo root.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "scripts", "04_analysis")
if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)


from simulate_int8_inference import (
    load_wts,
    conv1d_causal_int8,
    requantize,
    bias_to_int32,
    global_avg_pool
)

def run_python_sim():
    wts = load_wts("artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts")
    inp_ref = np.loadtxt("artifacts/04_hardware_handoff/int8_export/reference_input.txt", dtype=np.int8)
    
    current = inp_ref.reshape(12, 100)
    current_scale = 1.0 / 64.0
    
    for blk in range(3):
        prefix = f"tcn.network.{blk}."
        dil = 2 ** blk
        
        FIXED_ACT_SCALE = 1.0 / 64.0
        
        # conv1
        w1 = wts[prefix + "conv1.conv.weight"]
        b1 = wts[prefix + "conv1.conv.bias"]
        acc1_scale = current_scale * w1['scale']
        out1 = conv1d_causal_int8(current, w1['values'], dil)
        b1_i32 = bias_to_int32(b1['values'], b1['scale'], acc1_scale)
        out1 += b1_i32[:, None]
        out1 = np.maximum(out1, 0)
        q1 = requantize(out1, acc1_scale, FIXED_ACT_SCALE)
        
        # conv2
        w2 = wts[prefix + "conv2.conv.weight"]
        b2 = wts[prefix + "conv2.conv.bias"]
        acc2_scale = FIXED_ACT_SCALE * w2['scale']
        out2 = conv1d_causal_int8(q1, w2['values'], dil)
        b2_i32 = bias_to_int32(b2['values'], b2['scale'], acc2_scale)
        out2 += b2_i32[:, None]
        out2 = np.maximum(out2, 0)
        
        # residual
        if blk == 0:
            wd = wts[prefix + "downsample.weight"]
            bd = wts[prefix + "downsample.bias"]
            res_scale = current_scale * wd['scale']
            res = conv1d_causal_int8(current, wd['values'], 1) 
            bd_i32 = bias_to_int32(bd['values'], bd['scale'], res_scale)
            res += bd_i32[:, None]
        else:
            res = current.astype(np.int32)
            res_scale = current_scale
            
        block_out_scale = FIXED_ACT_SCALE
        res_q = requantize(res, res_scale, block_out_scale)
        out2_q = requantize(out2, acc2_scale, block_out_scale)
        
        merged = out2_q.astype(np.int32) + res_q.astype(np.int32)
        merged = np.maximum(merged, 0)
        current = np.clip(merged, -127, 127).astype(np.int8)
        current_scale = block_out_scale
        
    # GAP
    pooled = global_avg_pool(current)
    pooled_q = requantize(pooled, current_scale, current_scale)
    
    # FC
    fc_w = wts["fc.weight"]
    fc_b = wts["fc.bias"]
    fc_acc_scale = current_scale * fc_w['scale']
    logits_i32 = np.dot(fc_w['values'].astype(np.int32), pooled_q.astype(np.int32))
    fc_b_i32 = bias_to_int32(fc_b['values'], fc_b['scale'], fc_acc_scale)
    logits_i32 += fc_b_i32
    
    return logits_i32

def main():
    print("Running Python INT8 simulator...")
    py_logits = run_python_sim()
    print(f"Python Logits: {py_logits}")
    
    print("\nRunning C++ executable...")
    cmd = [
        "cpp_hls_prep/build/tcn_hls_prep.exe",
        "--weights", "artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts",
        "--ref-input", "artifacts/04_hardware_handoff/int8_export/reference_input.txt"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("C++ execution failed")
        print(r.stderr)
        sys.exit(1)
        
    cpp_logits = None
    for line in r.stdout.splitlines():
        if "Output logits (int32):" in line:
            parts = line.split(":")[-1].strip().split()
            cpp_logits = np.array([int(p) for p in parts], dtype=np.int32)
            break
            
    print(f"C++ Logits:    {cpp_logits}")
    
    if cpp_logits is None:
        print("Could not find C++ output in stdout.")
        sys.exit(1)
        
    if np.array_equal(py_logits, cpp_logits):
        print("\n[SUCCESS] Layer-by-layer mathematical logic is BIT-ACCURATE!")
    else:
        print("\n[FAIL] Output mismatch between Python and C++.")

if __name__ == "__main__":
    main()
