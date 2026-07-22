"""
Week 3 TCN 模型單元測試
=======================
驗證項目：
  1. Forward Pass 形狀正確
  2. Softmax 輸出總和為 1
  3. 因果性 (Causality) — 改變未來不影響過去的輸出
  4. 參數量在 FPGA 友善範圍
  5. 感受野計算正確
"""

import os
import sys
import torch
import numpy as np
import pytest

# Module note: these tests validate the baseline TCN behavior before the model
# is quantized or rewritten into the HLS-friendly operator set.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.tcn import (
    CausalConv1d,
    TemporalBlock,
    TemporalConvNet,
    TCNClassifier,
    build_default_model,
)
from src.utils.constants import (
    NUM_INPUT_FEATURES, NUM_CLASSES, DEFAULT_WINDOW_SIZE,
    TCN_CHANNELS, TCN_KERNEL_SIZE, TCN_DROPOUT,
)


# ============================================================
# Test 1: Forward Pass 形狀正確
# ============================================================
class TestForwardPass:
    """確認模型的輸入輸出形狀符合預期。"""

    def test_dummy_tensor_forward(self):
        """
        用假的 Dummy Tensor 做 Forward Pass。
        這是 guide.md Week 3 的檢核點 (Deliverable)。
        """
        model = build_default_model()
        model.eval()

        batch_size = 32
        seq_len = DEFAULT_WINDOW_SIZE  # 100
        num_features = NUM_INPUT_FEATURES  # 5

        # 建立 Dummy Tensor
        x = torch.randn(batch_size, seq_len, num_features)

        with torch.no_grad():
            output = model(x)

        # 輸出形狀: (batch_size, num_classes)
        assert output.shape == (batch_size, NUM_CLASSES), \
            f"Expected shape ({batch_size}, {NUM_CLASSES}), got {output.shape}"

    def test_single_sample_forward(self):
        """單筆資料也能通過。"""
        model = build_default_model()
        model.eval()
        x = torch.randn(1, DEFAULT_WINDOW_SIZE, NUM_INPUT_FEATURES)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, NUM_CLASSES)

    def test_variable_sequence_length(self):
        """不同序列長度也能通過（TCN 的彈性）。"""
        model = build_default_model()
        model.eval()
        for seq_len in [50, 100, 200, 500]:
            x = torch.randn(4, seq_len, NUM_INPUT_FEATURES)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (4, NUM_CLASSES), \
                f"Failed for seq_len={seq_len}"


# ============================================================
# Test 2: Softmax 輸出驗證
# ============================================================
class TestSoftmaxOutput:
    """確認 predict_proba 的輸出是有效的機率分布。"""

    def test_proba_sums_to_one(self):
        model = build_default_model()
        model.eval()
        x = torch.randn(16, DEFAULT_WINDOW_SIZE, NUM_INPUT_FEATURES)
        with torch.no_grad():
            proba = model.predict_proba(x)
        
        # 每個樣本的機率總和應該 ≈ 1.0
        sums = proba.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5), \
            f"Probability sums: {sums}"

    def test_proba_non_negative(self):
        model = build_default_model()
        model.eval()
        x = torch.randn(8, DEFAULT_WINDOW_SIZE, NUM_INPUT_FEATURES)
        with torch.no_grad():
            proba = model.predict_proba(x)
        assert (proba >= 0).all(), "Probabilities must be non-negative"


# ============================================================
# Test 3: 因果性驗證 (Causality)
# ============================================================
class TestCausality:
    """
    驗證因果性：改變「未來」的輸入不應影響「過去」時間點的中間表示。
    
    方法：
      1. 送入一個輸入 x，取得 TCN backbone 在時間 t 的輸出
      2. 修改 x 在時間 t+1 之後的值
      3. 重新 forward，確認時間 t 的輸出完全不變
    """

    def test_causal_no_future_leakage(self):
        model = build_default_model()
        model.eval()

        x = torch.randn(1, DEFAULT_WINDOW_SIZE, NUM_INPUT_FEATURES)
        t = 50  # 我們觀察第 50 個時間點

        # 第一次 forward
        with torch.no_grad():
            x_perm = x.permute(0, 2, 1)  # (Batch, Feature, Time)
            out1 = model.tcn(x_perm)  # (Batch, Channel, Time)
            val_at_t_1 = out1[:, :, t].clone()

        # 修改 t+1 之後的輸入
        x_modified = x.clone()
        x_modified[:, t + 1:, :] = torch.randn_like(x_modified[:, t + 1:, :]) * 100

        # 第二次 forward
        with torch.no_grad():
            x_mod_perm = x_modified.permute(0, 2, 1)
            out2 = model.tcn(x_mod_perm)
            val_at_t_2 = out2[:, :, t].clone()

        # 時間 t 的輸出必須完全一致
        assert torch.allclose(val_at_t_1, val_at_t_2, atol=1e-5), \
            "因果性違反！修改未來資料影響了過去的輸出。"


# ============================================================
# Test 4: 參數量檢查 (FPGA 友善)
# ============================================================
class TestParameterCount:
    """確認模型參數量在合理範圍內，適合 FPGA 部署。"""

    def test_param_count_reasonable(self):
        model = build_default_model()
        count = model.count_parameters()
        print(f"   模型參數量: {count:,}")
        # 對於 3 層 16-channel TCN，參數量應在數千級，遠小於百萬
        assert count < 100_000, \
            f"參數量 {count:,} 太大，不適合 FPGA 部署"
        assert count > 100, \
            f"參數量 {count:,} 太小，模型可能有問題"


# ============================================================
# Test 5: 感受野 (Receptive Field) 計算
# ============================================================
class TestReceptiveField:
    """確認感受野計算正確。"""

    def test_receptive_field_value(self):
        model = build_default_model()
        rf = model.receptive_field()
        print(f"   感受野: {rf} ticks")
        
        # 手動計算：3 層, kernel=3, dilation=[1,2,4]
        # RF = 1 + 2 * (3-1) * (1+2+4) = 1 + 4 * 7 = 29
        expected_rf = 1 + 2 * (TCN_KERNEL_SIZE - 1) * sum(2**i for i in range(len(TCN_CHANNELS)))
        assert rf == expected_rf, f"Expected RF={expected_rf}, got {rf}"
        assert rf <= DEFAULT_WINDOW_SIZE, \
            f"感受野 ({rf}) 大於視窗大小 ({DEFAULT_WINDOW_SIZE})，可能浪費計算"


# ============================================================
# Test 6: CausalConv1d 單元測試
# ============================================================
class TestCausalConv1d:
    """驗證 CausalConv1d 的輸出長度與因果性。"""

    def test_output_length_preserved(self):
        """因果卷積不應改變時間軸長度。"""
        conv = CausalConv1d(5, 16, kernel_size=3, dilation=1)
        x = torch.randn(1, 5, 100)
        out = conv(x)
        assert out.shape == (1, 16, 100), f"Got {out.shape}"

    def test_output_length_with_dilation(self):
        """膨脹卷積也不應改變時間軸長度。"""
        conv = CausalConv1d(5, 16, kernel_size=3, dilation=4)
        x = torch.randn(1, 5, 100)
        out = conv(x)
        assert out.shape == (1, 16, 100), f"Got {out.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
