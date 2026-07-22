"""
模擬 C++ INT8 推論練，用於逐層比對 Python ↔ C++。
此腳本讀取 tcn_int8_weights.wts 與 reference_input.txt，
並以 numpy 實作與 C++ 完全相同的整數運算與 requantization 邏輯，
確保兩者 bit-accurate。
"""
import numpy as np

# Module note: this simulator mirrors the fixed-point math used by the C++
# implementation so we can debug layer-by-layer mismatches before running HLS.

def load_wts(path):
    tensors = {}
    with open(path, 'r') as f:
        lines = f.read().split()
    
    idx = 0
    num_tensors = int(lines[idx])
    idx += 1
    
    for _ in range(num_tensors):
        name = lines[idx]
        count = int(lines[idx+1])
        scale = float(lines[idx+2])
        shape_str = lines[idx+3]
        shape = tuple(int(x) for x in shape_str.split('x'))
        idx += 4
        
        vals = []
        for _ in range(count):
            vals.append(int(lines[idx]))
            idx += 1
            
        tensors[name] = {
            'scale': scale,
            'shape': shape,
            'values': np.array(vals, dtype=np.int8).reshape(shape)
        }
    return tensors

def conv1d_causal_int8(x, w, dilation):
    """
    x: (C_in, T) int8
    w: (C_out, C_in, K) int8
    return: (C_out, T) int32
    """
    C_out, C_in, K = w.shape
    T = x.shape[1]
    
    pad_len = (K - 1) * dilation
    padded_x = np.pad(x, ((0,0), (pad_len, 0)), mode='constant')
    
    out = np.zeros((C_out, T), dtype=np.int32)
    for t in range(T):
        for k in range(K):
            in_t = t + k * dilation
            # padded_x[:, in_t] is (C_in,)
            # w[:, :, k] is (C_out, C_in)
            out[:, t] += np.dot(w[:, :, k].astype(np.int32), padded_x[:, in_t].astype(np.int32))
            
    return out

def cpp_round(x: np.ndarray) -> np.ndarray:
        """Round like C++ std::round for our purposes.

        We want *half-away-from-zero* rounding:
            +1.5 -> +2,  +1.4 -> +1
            -1.5 -> -2,  -1.4 -> -1

        NOTE: The previous implementation used floor(x + 0.5), which is wrong for
        negative numbers (e.g. -1.5 would become -1).
        """
        x = x.astype(np.float32)
        return np.sign(x) * np.floor(np.abs(x) + 0.5)

def requantize(acc, acc_scale, target_scale):
    ratio = acc_scale / target_scale
    val = cpp_round(acc.astype(np.float32) * ratio)
    return np.clip(val, -127, 127).astype(np.int8)

def bias_to_int32(bias_vals, bias_scale, acc_scale):
    bias_to_acc = bias_scale / acc_scale
    return cpp_round(bias_vals.astype(np.float32) * bias_to_acc).astype(np.int32)

def global_avg_pool(x):
    return (np.sum(x.astype(np.int64), axis=1) // x.shape[1]).astype(np.int32)

def main():
    wts = load_wts("artifacts/04_hardware_handoff/int8_export/tcn_int8_weights.wts")
    inp_ref = np.loadtxt("artifacts/04_hardware_handoff/int8_export/reference_input.txt", dtype=np.int8)
    
    current = inp_ref.reshape(12, 100)
    current_scale = 1.0 / 64.0
    
    print(f"Input sum: {np.sum(current)}")
    
    for blk in range(3):
        prefix = f"tcn.network.{blk}."
        dil = 2 ** blk
        
        # --- conv1 ---
        w1 = wts[prefix + "conv1.conv.weight"]
        b1 = wts[prefix + "conv1.conv.bias"]
        
        acc1_scale = current_scale * w1['scale']
        out1 = conv1d_causal_int8(current, w1['values'], dil)
        b1_i32 = bias_to_int32(b1['values'], b1['scale'], acc1_scale)
        out1 += b1_i32[:, None]
        out1 = np.maximum(out1, 0)
        
        q1 = requantize(out1, acc1_scale, acc1_scale)
        print(f"Block {blk} conv1 q sum: {np.sum(q1)}")
        
        # --- conv2 ---
        w2 = wts[prefix + "conv2.conv.weight"]
        b2 = wts[prefix + "conv2.conv.bias"]
        
        acc2_scale = acc1_scale * w2['scale']
        out2 = conv1d_causal_int8(q1, w2['values'], dil)
        b2_i32 = bias_to_int32(b2['values'], b2['scale'], acc2_scale)
        out2 += b2_i32[:, None]
        
        # --- residual ---
        if blk == 0:
            wd = wts[prefix + "downsample.weight"]
            bd = wts[prefix + "downsample.bias"]
            res_scale = current_scale * wd['scale']
            res = conv1d_causal_int8(current, wd['values'], 1) # acts as 1x1 conv
            bd_i32 = bias_to_int32(bd['values'], bd['scale'], res_scale)
            res += bd_i32[:, None]
        else:
            res = current.astype(np.int32)
            res_scale = current_scale
            
        # Add residual (scale aligned)
        block_out_scale = acc2_scale
        res_q = requantize(res, res_scale, block_out_scale)
        out2 += res_q.astype(np.int32)
        out2 = np.maximum(out2, 0)
        
        current_scale = block_out_scale
        current = requantize(out2, block_out_scale, block_out_scale)
        print(f"Block {blk} out sum: {np.sum(current)}")
        
    # --- GAP ---
    pooled = global_avg_pool(current)
    print(f"GAP sum: {np.sum(pooled)}")
    pooled_q = requantize(pooled, current_scale, current_scale)
    
    # --- FC ---
    fc_w = wts["fc.weight"]
    fc_b = wts["fc.bias"]
    
    fc_acc_scale = current_scale * fc_w['scale']
    logits_i32 = np.dot(fc_w['values'].astype(np.int32), pooled_q.astype(np.int32))
    fc_b_i32 = bias_to_int32(fc_b['values'], fc_b['scale'], fc_acc_scale)
    logits_i32 += fc_b_i32
    
    print(f"Logits (int32): {logits_i32}")

if __name__ == '__main__':
    main()
