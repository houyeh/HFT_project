# 07L Ultrafast Policy Sweep

This phase compares a low-bit QAT Tiny MLP with a compact quantized LUT policy.
Selection uses train/val only; NEW3/NEW4 are frozen diagnostics and NEW5 is not used.

The selected Q12.6 Tiny MLP passes 25,600-vector bit-exact CSim and RTL cosim,
then closes 100 MHz out-of-context routing at WNS +0.467 ns. Its scheduled core
latency is 13 cycles (0.130 us), with routed resources of 4,466 LUT, 2,160 FF,
36 DSP, and 0 BRAM18K. See `AUDIT_ultrafast_policy_sweep.md` for task-value and
claim-boundary details.

The 256-cell LUT policy did not beat the val-selected static queue rule and was
not advanced to HLS.
