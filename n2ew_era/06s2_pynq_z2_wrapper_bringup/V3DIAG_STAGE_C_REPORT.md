# V3 Diagonal Stage C Replay Vector Report

Status: `stage_c_pass_model_matched_bram_vectors_generated`.

Stage C regenerated the replay expectations with the frozen 06U V3 diagonal
HLS C model. It did not use the legacy 05C4 expected readouts, did not select a
new market window, and did not modify the existing `bram/` directory.

## Stage C report block

```text
[C] input event count: 4096
[C] source session: NEW4_20260112_TXFA6
[C] source selection rule: highest one-second rolling event count across val/NEW3/NEW4
[C] recorded peak: 1469 events/second
[C] CSim feedback: state starts at zero; each V3 DUT state_next feeds the next event
[C] new/old readout .mem differing lines: 4096 / 4096
[C] input feature prefix differing lines: 0 / 4096
[C] output files:
    bram_v3_diag/demo_replay_x_readout_q24_12.mem (450560 bytes)
    bram_v3_diag/demo_replay_x_readout_q24_12.coe (454720 bytes)
    bram_v3_diag/demo_replay_expected_state_q24_12.mem (401408 bytes)
    bram_v3_diag/demo_replay_expected_state_q24_12.coe (405568 bytes)
    bram_v3_diag/BRAM_VECTOR_PACKING.md (641 bytes)
    bram_v3_diag/v3diag_replay_manifest.json (2523 bytes)
```

## CSim provenance

The independent replay component references the frozen files directly:

- `06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/hls_src/phc_diag_step_hls.cpp`
- `06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/hls_src/phc_diag_step_hls.hpp`
- `06u_phc_latency_breakthrough/variants/v3_diagonal16_unroll8_localpart_100mhz/hls_src/phc_diag_weights.hpp`

Its testbench consumes the existing replay feature CSV, initializes all 16
state words to zero, invokes `phc_diag_step` once per row, and feeds the DUT's
fixed-point `state_next` into the following row. Vitis HLS 2025.2 reported:

```text
total=4096
feedback_mode=hls_v3diag_dut_state_feedback
CSim done with 0 errors.
```

The generated CSim output is:

- `results_v3_diag/demo_replay_v3diag_hls_csim_sequential_out.csv`

## Packing and self-checks

Each input/readout line contains 18 signed Q24.12 words: 16 input features and
2 V3 expected readouts, for 108 hexadecimal characters (432 bits). Each state
line contains 16 Q24.12 words, for 96 hexadecimal characters (384 bits).

The first 16 words of every new input/readout line are identical to the legacy
file, confirming that Stage C preserved the exact same 4,096-event replay. The
two-word readout suffix differs on all 4,096 rows, confirming that expectations
were regenerated from the V3 diagonal model rather than copied from 05C4.

Machine-readable paths, SHA-256 hashes, sizes, session metadata, and the line
difference count are recorded in `bram_v3_diag/v3diag_replay_manifest.json`.

## Scope boundary

Stage C only establishes model-matched BRAM initialization files. Full
wrapper-plus-core numerical XSim, board-level P&R, bitstream generation, and
on-board comparison have not been run in this stage.
