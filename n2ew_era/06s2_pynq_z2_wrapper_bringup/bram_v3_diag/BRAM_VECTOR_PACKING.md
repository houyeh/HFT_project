# V3 Diagonal BRAM Vector Packing

status: `generated_from_v3_diagonal_sequential_hls_csim`

- Replay session: `NEW4_20260112_TXFA6`.
- Replay rows: 4096.
- Quantization: signed Q24.12 (`ap_fixed<24,12>`).
- Inputs use AP_TRN bit-drop semantics (`floor(value * 4096)`).
- Expected readout/state are the sequential V3 HLS CSim DUT outputs.
- State starts at zero and each DUT `state_next` is fed into the next row.
- `demo_replay_x_readout_q24_12`: 16 inputs followed by 2 expected readouts.
- `demo_replay_expected_state_q24_12`: 16 expected next-state words.
- Lines different from the legacy 05C4 readout memory: 4096 / 4096.
