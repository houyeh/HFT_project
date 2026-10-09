# BRAM Vector Packing

status: `replay_vectors_generated_from_sequential_hls_csim`

- Quantization: signed Q24.12, matching `ap_fixed<24,12>`.
- Integer packing uses AP_TRN bit-drop semantics: `floor(value * 4096)` before two's-complement packing. This matches Vitis `ap_fixed<24,12>` casts for negative values; plain NumPy truncation is not equivalent.
- Hex word width: 24 bits = 6 hex digits, two's complement for negative values.
- `demo_replay_x_readout_q24_12.mem/.coe` line width: 432 bits.
- Word order per line: `x_t_0 ... x_t_15 expected_readout_0 expected_readout_1`.
- The expected readout fields are from the sequential HLS CSim DUT output: `results/demo_replay_seed2_90_hls_csim_sequential_out.csv`. They are not the PyTorch float reference and not an older CSim artifact.
- `demo_replay_expected_state_q24_12.mem/.coe` line width: 384 bits.
- Word order per line: `expected_state_next_0 ... expected_state_next_15`.
- The board wrapper resets recurrent state to zero at replay start, compares readout after each `ap_done`, waits one drain cycle, and feeds captured `state_next` back as the next event's `state_prev`.
- Expected state is provided for XSim debug or optional hardware checking; the primary demo comparator uses readout only to save BRAM.

Comparator note: exact zero-LSB comparison exposed 27 rows with max 1-LSB readout drift under real RTL-feedback recurrence. The wrapper therefore uses a fixed 1-LSB readout tolerance for the positive demo gate. The negative test flips 16 LSB at index 17 and is still caught.

Capacity note: 4096 rows * 432 bits = 1,769,472 bits, approximately 96 BRAM_18K before IP overhead. Full 25,600 rows would require about 614 BRAM_18K for this packed input+readout memory alone, so the demo uses the selected burst subsample.
