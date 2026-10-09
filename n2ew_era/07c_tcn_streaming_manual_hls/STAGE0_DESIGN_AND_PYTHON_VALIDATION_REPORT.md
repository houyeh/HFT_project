status: stage0_complete_stop_for_review
finding: python_streaming_matches_pytorch_batch_with_float_roundoff
# Stage 0 Design and Python Validation Report

## Scope

This phase creates only a Python streaming design validation for the TCN prepad model. It does not modify 07a/07b artifacts, does not use NEW5, and does not create HLS C++, csim, csynth, cosim, implementation, P&R, or bitstream outputs.

Inputs are the existing 07b 200-row validation test vectors generated from the val split. Checkpoints are the existing frozen 07a selected policies.

## Streaming Form

For one block with dilation d and kernel size 3, the PyTorch valid-conv block can be re-indexed as an event-aligned causal streaming block:

```text
c1_s  = ReLU(b1 + W1[0] * x_{s-2d}  + W1[1] * x_{s-d}  + W1[2] * x_s)
c2_s  = ReLU(b2 + W2[0] * c1_{s-2d} + W2[1] * c1_{s-d} + W2[2] * c1_s)
out_s = ReLU(c2_s + x_s)
emit only after s >= 4d
```

Important residual alignment note: in this event-aligned form, the residual term is the current block input `x_s`. A timestep FIFO that pops `x_{s-4d}` at the residual add would not match the PyTorch batch model. HLS may still need cycle-level delay registers to align arithmetic pipeline latency, but that is a Stage 1 scheduling detail, not a Stage 0 timestep FIFO.

The same relation in PyTorch valid-convolution coordinates is:

```text
batch output index t  <=>  event-aligned index s = t + 4d
PyTorch residual = block_input[t + 4d] = block_input[s]
```

## FIFO and Alignment Table

| block | dilation | input_len | output_len | conv1_input_history_previous_timesteps | conv2_conv1_history_previous_timesteps | total_block_warmup_timesteps | cumulative_warmup_after_block | pytorch_valid_conv_residual_tap | event_aligned_residual_tap | residual_timestep_fifo_depth | residual_cycle_delay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 252 | 248 | 2 | 2 | 4 | 4 | block_input[t+4] | block_input[s], where s=t+4d | 0 | TBD by Stage 1 HLS operator schedule |
| 1 | 2 | 248 | 240 | 4 | 4 | 8 | 12 | block_input[t+8] | block_input[s], where s=t+4d | 0 | TBD by Stage 1 HLS operator schedule |
| 2 | 4 | 240 | 224 | 8 | 8 | 16 | 28 | block_input[t+16] | block_input[s], where s=t+4d | 0 | TBD by Stage 1 HLS operator schedule |
| 3 | 8 | 224 | 192 | 16 | 16 | 32 | 60 | block_input[t+32] | block_input[s], where s=t+4d | 0 | TBD by Stage 1 HLS operator schedule |
| 4 | 16 | 192 | 128 | 32 | 32 | 64 | 124 | block_input[t+64] | block_input[s], where s=t+4d | 0 | TBD by Stage 1 HLS operator schedule |

## Initialization

- Block 0 receives 124 zero vectors followed by the 128 real feature vectors, matching `TCNPrePadRegressor.prepad`.
- Internal histories start empty at sequence reset. The 124 explicit zero events provide the causal warmup.
- For a free-running deployment, reset plus 124 zero ticks is equivalent to the current prepad contract for the first scored 128-event window.

## Validation Inputs

| variant | policy_id | checkpoint | test_vectors | sample_count | seq_len | feature_dim | target_scale | new5_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch | tcn_prepad_scratch_16feat | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07a_tcn_feedforward_phc_tradeoff\models\tcn_prepad_scratch_16feat_seed2_best_val.pt | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07b_tcn_prepad_manual_hls\results\tcn_prepad_scratch_test_vectors.csv | 200 | 128 | 16 | 201.175 | False |
| distilled | tcn_prepad_distilled_16feat | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07a_tcn_feedforward_phc_tradeoff\models\tcn_prepad_distilled_16feat_seed1_best_val.pt | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07b_tcn_prepad_manual_hls\results\tcn_prepad_distilled_test_vectors.csv | 200 | 128 | 16 | 201.175 | False |

## Worst Per-Stage Differences

| variant | stage | shape | max_abs_diff | mean_abs_diff | rmse | max_sample_idx | max_timestep_idx | max_channel_or_output_idx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distilled | readout_scaled_vs_test_vector_csv | 200x2 | 0.00031280518 | 8.0243014e-05 | 9.9696956e-05 | 92 | -1 | 0 |
| distilled | readout_scaled | 200x2 | 0.00029754639 | 8.1141628e-05 | 0.00010136193 | 81 | -1 | 1 |
| scratch | readout_scaled_vs_test_vector_csv | 200x2 | 0.0002746582 | 7.0030373e-05 | 9.0097314e-05 | 95 | -1 | 0 |
| scratch | readout_scaled | 200x2 | 0.00026512146 | 6.7439117e-05 | 8.7616807e-05 | 152 | -1 | 1 |
| scratch | block4_out | 200x128x16 | 2.4795532e-05 | 6.9267114e-07 | 1.2498348e-06 | 17 | 65 | 2 |
| scratch | block3_out | 200x192x16 | 2.0027161e-05 | 4.165757e-07 | 9.1508472e-07 | 18 | 97 | 2 |
| scratch | block4_residual | 200x128x16 | 2.0027161e-05 | 6.071757e-07 | 1.1201484e-06 | 18 | 33 | 2 |
| scratch | block3_conv1_relu | 200x208x16 | 1.7762184e-05 | 1.5411111e-07 | 5.1024761e-07 | 16 | 177 | 3 |
| scratch | block4_conv1_relu | 200x160x16 | 1.6689301e-05 | 1.9803043e-07 | 5.6327428e-07 | 16 | 129 | 9 |
| scratch | block2_conv1_relu | 200x232x16 | 1.5944242e-05 | 1.6707756e-07 | 4.7006566e-07 | 58 | 200 | 1 |
| distilled | block4_out | 200x128x16 | 1.5258789e-05 | 6.3619258e-07 | 1.090256e-06 | 169 | 111 | 8 |
| distilled | block3_out | 200x192x16 | 1.4781952e-05 | 4.0443815e-07 | 8.4106681e-07 | 169 | 175 | 8 |
| distilled | block4_residual | 200x128x16 | 1.4781952e-05 | 5.8295215e-07 | 1.0287452e-06 | 169 | 111 | 8 |
| distilled | block3_residual | 200x192x16 | 1.335144e-05 | 3.4830336e-07 | 7.5473714e-07 | 172 | 79 | 8 |
| distilled | block2_out | 200x224x16 | 1.335144e-05 | 3.0388421e-07 | 6.9927801e-07 | 172 | 111 | 8 |
| distilled | block4_conv1_relu | 200x160x16 | 1.0490417e-05 | 1.8933265e-07 | 4.3856704e-07 | 41 | 130 | 14 |

## Full Per-Stage Summary

| variant | stage | shape | max_abs_diff | mean_abs_diff | rmse | max_sample_idx | max_timestep_idx | max_channel_or_output_idx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distilled | block0_conv1_relu | 200x250x16 | 1.9073486e-06 | 3.5072705e-08 | 9.9745332e-08 | 88 | 230 | 8 |
| distilled | block0_conv2_relu | 200x248x16 | 3.695488e-06 | 8.715601e-08 | 1.9983073e-07 | 39 | 246 | 8 |
| distilled | block0_out | 200x248x16 | 3.695488e-06 | 8.1094819e-08 | 1.9541366e-07 | 39 | 246 | 8 |
| distilled | block0_residual | 200x248x16 | 0 | 0 | 0 | 0 | 0 | 0 |
| distilled | block1_conv1_relu | 200x244x16 | 6.6757202e-06 | 1.38332e-07 | 3.3009373e-07 | 85 | 228 | 1 |
| distilled | block1_conv2_relu | 200x240x16 | 8.3446503e-06 | 1.6295954e-07 | 4.3791154e-07 | 169 | 223 | 8 |
| distilled | block1_out | 200x240x16 | 9.2983246e-06 | 2.1925735e-07 | 5.1951277e-07 | 169 | 223 | 13 |
| distilled | block1_residual | 200x240x16 | 3.695488e-06 | 8.3083965e-08 | 1.9844231e-07 | 39 | 238 | 8 |
| distilled | block2_conv1_relu | 200x232x16 | 6.6757202e-06 | 1.3879094e-07 | 4.0035074e-07 | 19 | 228 | 2 |
| distilled | block2_conv2_relu | 200x224x16 | 5.7220459e-06 | 1.1129176e-07 | 3.1246126e-07 | 41 | 194 | 15 |
| distilled | block2_out | 200x224x16 | 1.335144e-05 | 3.0388421e-07 | 6.9927801e-07 | 172 | 111 | 8 |
| distilled | block2_residual | 200x224x16 | 9.2983246e-06 | 2.3195832e-07 | 5.3738803e-07 | 169 | 207 | 13 |
| distilled | block3_conv1_relu | 200x208x16 | 9.8645687e-06 | 1.7569473e-07 | 4.7276867e-07 | 172 | 95 | 3 |
| distilled | block3_conv2_relu | 200x192x16 | 3.5762787e-06 | 9.381738e-08 | 2.4968523e-07 | 42 | 98 | 1 |
| distilled | block3_out | 200x192x16 | 1.4781952e-05 | 4.0443815e-07 | 8.4106681e-07 | 169 | 175 | 8 |
| distilled | block3_residual | 200x192x16 | 1.335144e-05 | 3.4830336e-07 | 7.5473714e-07 | 172 | 79 | 8 |
| distilled | block4_conv1_relu | 200x160x16 | 1.0490417e-05 | 1.8933265e-07 | 4.3856704e-07 | 41 | 130 | 14 |
| distilled | block4_conv2_relu | 200x128x16 | 3.5464764e-06 | 1.0951157e-07 | 2.4574382e-07 | 170 | 95 | 8 |
| distilled | block4_out | 200x128x16 | 1.5258789e-05 | 6.3619258e-07 | 1.090256e-06 | 169 | 111 | 8 |
| distilled | block4_residual | 200x128x16 | 1.4781952e-05 | 5.8295215e-07 | 1.0287452e-06 | 169 | 111 | 8 |
| distilled | prepad | 200x252x16 | 0 | 0 | 0 | 0 | 0 | 0 |
| distilled | readout_scaled | 200x2 | 0.00029754639 | 8.1141628e-05 | 0.00010136193 | 81 | -1 | 1 |
| distilled | readout_scaled_vs_test_vector_csv | 200x2 | 0.00031280518 | 8.0243014e-05 | 9.9696956e-05 | 92 | -1 | 0 |
| scratch | block0_conv1_relu | 200x250x16 | 1.4305115e-06 | 3.4962433e-08 | 9.5773047e-08 | 11 | 247 | 12 |
| scratch | block0_conv2_relu | 200x248x16 | 3.8146973e-06 | 7.2617195e-08 | 1.7259707e-07 | 88 | 228 | 1 |
| scratch | block0_out | 200x248x16 | 3.8146973e-06 | 6.5368522e-08 | 1.6611507e-07 | 88 | 228 | 1 |
| scratch | block0_residual | 200x248x16 | 0 | 0 | 0 | 0 | 0 | 0 |
| scratch | block1_conv1_relu | 200x244x16 | 4.7683716e-06 | 8.805617e-08 | 2.3366293e-07 | 57 | 232 | 1 |
| scratch | block1_conv2_relu | 200x240x16 | 6.8247318e-06 | 1.3478581e-07 | 3.5002486e-07 | 58 | 208 | 12 |
| scratch | block1_out | 200x240x16 | 8.1062317e-06 | 1.8261031e-07 | 4.2079463e-07 | 3 | 231 | 2 |
| scratch | block1_residual | 200x240x16 | 3.8146973e-06 | 6.6936292e-08 | 1.6876239e-07 | 88 | 220 | 1 |
| scratch | block2_conv1_relu | 200x232x16 | 1.5944242e-05 | 1.6707756e-07 | 4.7006566e-07 | 58 | 200 | 1 |
| scratch | block2_conv2_relu | 200x224x16 | 1.4543533e-05 | 1.2402469e-07 | 4.1669376e-07 | 58 | 192 | 4 |
| scratch | block2_out | 200x224x16 | 1.5258789e-05 | 2.8997943e-07 | 6.8332024e-07 | 58 | 192 | 4 |
| scratch | block2_residual | 200x224x16 | 8.1062317e-06 | 1.9321022e-07 | 4.3538986e-07 | 3 | 215 | 2 |
| scratch | block3_conv1_relu | 200x208x16 | 1.7762184e-05 | 1.5411111e-07 | 5.1024761e-07 | 16 | 177 | 3 |
| scratch | block3_conv2_relu | 200x192x16 | 7.0333481e-06 | 1.1853153e-07 | 3.4991047e-07 | 18 | 97 | 14 |
| scratch | block3_out | 200x192x16 | 2.0027161e-05 | 4.165757e-07 | 9.1508472e-07 | 18 | 97 | 2 |
| scratch | block3_residual | 200x192x16 | 1.5258789e-05 | 3.3241338e-07 | 7.3776181e-07 | 58 | 160 | 4 |
| scratch | block4_conv1_relu | 200x160x16 | 1.6689301e-05 | 1.9803043e-07 | 5.6327428e-07 | 16 | 129 | 9 |
| scratch | block4_conv2_relu | 200x128x16 | 9.059906e-06 | 1.3706662e-07 | 3.5274525e-07 | 16 | 113 | 5 |
| scratch | block4_out | 200x128x16 | 2.4795532e-05 | 6.9267114e-07 | 1.2498348e-06 | 17 | 65 | 2 |
| scratch | block4_residual | 200x128x16 | 2.0027161e-05 | 6.071757e-07 | 1.1201484e-06 | 18 | 33 | 2 |
| scratch | prepad | 200x252x16 | 0 | 0 | 0 | 0 | 0 | 0 |
| scratch | readout_scaled | 200x2 | 0.00026512146 | 6.7439117e-05 | 8.7616807e-05 | 152 | -1 | 1 |
| scratch | readout_scaled_vs_test_vector_csv | 200x2 | 0.0002746582 | 7.0030373e-05 | 9.0097314e-05 | 95 | -1 | 0 |

## Stage Boundary

STOP HERE. Stage 1 HLS work is intentionally not started until this Stage 0 design and Python validation is reviewed.

## Honest Conclusion

The event-aligned streaming Python implementation reproduces the PyTorch batch TCN intermediate tensors at every block output tested here, with only float-order roundoff. The algorithmic design is ready for review, but no HLS implementation or latency claim is made in this stage.
