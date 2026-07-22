"""
驗證 HLS 友善版 TCN 與原版 TCN 的數值等價性。

測試項目：
  1. 權重遷移後，兩版模型的 state_dict key 完全對應
  2. 同一組 input，兩版模型的 forward output 在 FP32 精度下完全一致
  3. 因果性驗證：未來 input 不影響過去 output
  4. 參數量與感受野相同
  5. torch.jit.trace 可以成功（原版因 F.pad 可能有 warning）
"""

from __future__ import annotations

# Module note: these checks protect the weight-migration path from the training
# model to the HLS-friendly PyTorch variant before the model is exported to C++.

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.tcn import (
    TCNClassifier,
    HLSFriendlyTCNClassifier,
    build_default_model,
    build_default_model_hls,
    migrate_weights_to_hls,
)


def test_weight_migration():
    """測試權重遷移：state_dict key 和 shape 完全相同。"""
    original = build_default_model(num_inputs=12)
    hls_model = build_default_model_hls(num_inputs=12)

    # 遷移前，key 應完全對應
    orig_keys = set(original.state_dict().keys())
    hls_keys = set(hls_model.state_dict().keys())
    assert orig_keys == hls_keys, f"Key mismatch: {orig_keys.symmetric_difference(hls_keys)}"

    # 遷移
    migrate_weights_to_hls(original, hls_model)

    # 遷移後，權重數值完全相同
    for key in orig_keys:
        orig_val = original.state_dict()[key]
        hls_val = hls_model.state_dict()[key]
        assert torch.equal(orig_val, hls_val), f"Weight mismatch at {key}"

    print("[PASS] Weight migration: all keys and values match.")


def test_forward_equivalence():
    """測試前向等價性：同一組 input，兩版 output 完全相同。"""
    torch.manual_seed(42)
    original = build_default_model(num_inputs=12)
    hls_model = build_default_model_hls(num_inputs=12)
    migrate_weights_to_hls(original, hls_model)

    original.eval()
    hls_model.eval()

    # 測試多種 batch size
    for batch_size in [1, 4, 16]:
        # 原版需要 (B, T, F) — time-last input
        x_time_last = torch.randn(batch_size, 100, 12)
        # HLS 版需要 (B, F, T) — channel-first input
        x_channel_first = x_time_last.permute(0, 2, 1)

        with torch.no_grad():
            out_orig = original(x_time_last)
            out_hls = hls_model(x_channel_first)

        max_diff = (out_orig - out_hls).abs().max().item()
        assert max_diff < 1e-5, f"Output mismatch: max_diff={max_diff} for batch_size={batch_size}"
        print(f"  batch_size={batch_size}: max_diff={max_diff:.2e} [OK]")

    print("[PASS] Forward equivalence: all batch sizes match.")


def test_causality():
    """測試因果性：修改未來 input 不影響過去 output。"""
    torch.manual_seed(42)
    hls_model = build_default_model_hls(num_inputs=12)
    hls_model.eval()

    x = torch.randn(1, 12, 100)

    # 取得 backbone 的逐時間步 output
    with torch.no_grad():
        backbone_out = hls_model.tcn(x)  # (1, 16, 100)
        out_t50 = backbone_out[0, :, 50].clone()

    # 修改時間步 51 之後的 input
    x_modified = x.clone()
    x_modified[:, :, 51:] = torch.randn(1, 12, 49) * 100

    with torch.no_grad():
        backbone_out_modified = hls_model.tcn(x_modified)
        out_t50_modified = backbone_out_modified[0, :, 50].clone()

    max_diff = (out_t50 - out_t50_modified).abs().max().item()
    assert max_diff < 1e-5, f"Causality violation: max_diff={max_diff}"
    print(f"[PASS] Causality: t=50 output unchanged after modifying t>50 (diff={max_diff:.2e}).")


def test_parameter_count():
    """測試參數量和感受野相同。"""
    original = build_default_model(num_inputs=12)
    hls_model = build_default_model_hls(num_inputs=12)

    orig_params = original.count_parameters()
    hls_params = hls_model.count_parameters()
    assert orig_params == hls_params, f"Param count mismatch: {orig_params} vs {hls_params}"

    orig_rf = original.receptive_field()
    hls_rf = hls_model.receptive_field()
    assert orig_rf == hls_rf, f"RF mismatch: {orig_rf} vs {hls_rf}"

    print(f"[PASS] Parameters: {orig_params}, Receptive field: {orig_rf}")


def test_jit_trace():
    """測試 torch.jit.trace 可以成功。"""
    hls_model = build_default_model_hls(num_inputs=12)
    hls_model.eval()

    x = torch.randn(1, 12, 100)
    try:
        traced = torch.jit.trace(hls_model, x)
        with torch.no_grad():
            out_traced = traced(x)
            out_direct = hls_model(x)
        max_diff = (out_traced - out_direct).abs().max().item()
        assert max_diff < 1e-5, f"Traced output mismatch: {max_diff}"
        print(f"[PASS] JIT trace: succeeded, max_diff={max_diff:.2e}")
    except Exception as exc:
        print(f"[FAIL] JIT trace: {exc}")
        raise


def test_frozen_model_migration():
    """測試用真實 frozen model 做遷移。"""
    frozen_path = Path("artifacts/03_latency_proof/latency_proof/frozen_tcn_best_model.pt")
    if not frozen_path.exists():
        print("[SKIP] Frozen model not found, skipping real-weight test.")
        return

    original = build_default_model(num_inputs=12)
    state_dict = torch.load(frozen_path, map_location="cpu")
    original.load_state_dict(state_dict)
    original.eval()

    hls_model = build_default_model_hls(num_inputs=12)
    migrate_weights_to_hls(original, hls_model)
    hls_model.eval()

    torch.manual_seed(99)
    x_time_last = torch.randn(8, 100, 12)
    x_channel_first = x_time_last.permute(0, 2, 1)

    with torch.no_grad():
        out_orig = original(x_time_last)
        out_hls = hls_model(x_channel_first)

    max_diff = (out_orig - out_hls).abs().max().item()
    assert max_diff < 1e-5, f"Frozen model migration: max_diff={max_diff}"
    print(f"[PASS] Frozen model migration: max_diff={max_diff:.2e}")

    # 也列出 softmax 機率
    proba_orig = torch.softmax(out_orig, dim=1)
    proba_hls = torch.softmax(out_hls, dim=1)
    print(f"  Sample 0 proba (orig):  {proba_orig[0].tolist()}")
    print(f"  Sample 0 proba (HLS):   {proba_hls[0].tolist()}")


def main():
    print("=" * 60)
    print("HLS-Friendly TCN Equivalence Verification")
    print("=" * 60)

    test_weight_migration()
    print()
    test_forward_equivalence()
    print()
    test_causality()
    print()
    test_parameter_count()
    print()
    test_jit_trace()
    print()
    test_frozen_model_migration()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
