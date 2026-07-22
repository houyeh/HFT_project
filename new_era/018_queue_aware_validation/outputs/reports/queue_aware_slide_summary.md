# Queue-aware Validation: Does the Edge Survive Conservative Fill Assumptions?

## Why Queue Matters
- Price touch at best level does not guarantee our fill because queue position is unknown.
- Missing aggressor-side prevents exact exchange queue replay.

## Tested Conservative Assumptions
- Uniform fill haircut (0.75 / 0.50 / 0.25)
- Good-fill haircut vs toxic-fill preference (mild/medium/severe)
- Queue-ahead proxy with best-level volume and future trade-volume windows
- Cancel-delay sensitivity under queue-aware conditions

## Key Original vs Conservative Snapshot
- NEW combined_filter_tcn_055: original 71735.0 -> conservative mean 26103.0 (p10 -23530.0)
- NEW no_gate_tcn_055: original 84980.0 -> conservative mean 13245.2 (p10 -15725.0)
- NEW no_gate_tcn_070: original 63440.0 -> conservative mean 23427.0 (p10 -20975.0)
- NEW spread_filter_tcn_055: original 101455.0 -> conservative mean 41494.6 (p10 80.0)
- NEW val_selected_tcn_070: original 63440.0 -> conservative mean 23427.0 (p10 -20975.0)
- NEW3 combined_filter_tcn_055: original 38010.0 -> conservative mean -22149.6 (p10 -243460.0)
- NEW3 no_gate_tcn_055: original -379275.0 -> conservative mean -384221.5 (p10 -626655.0)

## Final Verdict
- Verdict C: At least one key policy loses edge under conservative queue assumptions.
- FPGA claim impact: latency still matters, but economic edge must be judged under conservative fill stress.