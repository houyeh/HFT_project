# Tiny MLP Manual HLS

This phase maps the frozen 07G `16 -> 8 -> 3` direct-action Tiny MLP to the
07H-selected `ap_fixed<28,14>` data/weight format and `ap_fixed<40,16>`
accumulator. It does not retrain the model or use NEW5.

Verification completed: 25,600-vector CSim and Verilog C/RTL co-simulation are
bit-exact against the 07H integer emulator. C synthesis reports 172 cycles of
latency and 173 cycles of interval at the 100 MHz target, using 4 DSP, 1 BRAM,
1,268 LUT, and 1,187 FF. Routed implementation remains a later verification
tier; the 1.720 us figure is not yet a routed or on-board measurement.
