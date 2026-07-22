# Final Checkpoint

1. Original fill reproduction:
- NEW no_gate_tcn_055: net_pnl=84980.00, trade_count=468
- NEW spread_filter_tcn_055: net_pnl=101455.00, trade_count=223
- NEW3 combined_filter_tcn_055: net_pnl=38010.00, trade_count=986

2. Conservative fill result:
- NEW spread_filter_tcn_055: uniform50_mean=20380.50, good_medium_mean=59610.75, queue_q0.5_w5=101455.00
- NEW3 combined_filter_tcn_055: uniform50_mean=61781.00, good_medium_mean=-109088.25, queue_q0.5_w5=38010.00

3. Survival check:
- NEW spread_filter_tcn_055: still_positive=True, still_beats_random=False, still_beats_rule_baseline=False
- NEW3 combined_filter_tcn_055: still_positive=False, still_beats_random=False, still_beats_rule_baseline=False

4. Latency under queue:
- NEW combined_filter_tcn_055 delay_good_medium: delay10-delay0 = -39793.50
- NEW combined_filter_tcn_055 delay_original_like: delay10-delay0 = -71160.00
- NEW combined_filter_tcn_055 delay_queue_q50: delay10-delay0 = -71160.00
- NEW spread_filter_tcn_055 delay_good_medium: delay10-delay0 = -12890.50
- NEW spread_filter_tcn_055 delay_original_like: delay10-delay0 = -68480.00
- NEW spread_filter_tcn_055 delay_queue_q50: delay10-delay0 = -68480.00
- NEW3 combined_filter_tcn_055 delay_good_medium: delay10-delay0 = -228027.50
- NEW3 combined_filter_tcn_055 delay_original_like: delay10-delay0 = -125455.00
- NEW3 combined_filter_tcn_055 delay_queue_q50: delay10-delay0 = -125455.00

5. Final verdict:
- C
- reason: At least one key policy loses edge under conservative queue assumptions.