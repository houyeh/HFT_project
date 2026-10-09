# Demo Replay Staleness Scoreboard

status: `part_a_replay_scoreboard_ready_for_selected_06s2_board_bitstream`

## Scope

- This is a pure software replay/staleness data-prep artifact for the demo scoreboard.
- No model was trained, no checkpoint was changed, and no threshold was selected.
- NEW5 was not touched; the selected replay comes from the existing 04B action universe splits `val/new3/new4` only.
- CPU latency is sampled from the existing measured N=4 empirical distribution, not re-measured per event during this replay.

## Selected Burst Window

| field | value |
|---|---:|
| `split` | `new4` |
| `source_group` | `NEW4` |
| `session_id` | `NEW4_20260112_TXFA6` |
| `symbol` | `TXFA6` |
| `peak_count_1s` | `1469` |
| `peak_event_rate_per_sec` | `1469.0` |
| `segment_event_rate_per_sec` | `281.26072924534776` |
| `segment_duration_us` | `14563000.0` |
| `median_event_rate_past_50` | `131.00633066212754` |
| `p99_event_rate_past_50` | `270.23121341290135` |

Source parquet: `n2ew_era\04b_txf_only_final_development_rerun\data\txf_only_action_universe.parquet\split=new4\part-NEW4_20260112_TXFA6.parquet`

The selected window maximizes the 1-second rolling event count among `val/new3/new4` sessions. This is a reproducible burst criterion, not a hand-picked dramatic interval.

## Latency Sources

- FPGA latency: fixed `25.733` us.
- FPGA latency source: 06S seed2 / 06L unroll16 90 MHz routed timing-met wrapper target; 2316 cycles at 90 MHz = 25.733 us.
- 06U V1 `22.36 us` remains the lower-latency candidate, but the repo-local 06U RTL export is not self-contained for this wrapper bring-up. The selected board bitstream therefore uses the complete 06S seed2 / 06L unroll16 90 MHz RTL.
- CPU latency: empirical raw samples from `n2ew_era\06f_phc_lite_manual_hls\results\cpu_concurrent_benchmark_N4.csv`.
- CPU demo baseline: N=4 concurrent LibTorch/TorchScript CPU benchmark, round 1 fixed as the presentation mouthpiece.
- CPU benchmark metadata, N=4 round 1: mean `72.198630 us`, p50 `65.600000 us`, p95 `126.300000 us`, p99 `262.600000 us`, deadline `83.160000 us`, deadline miss rate `15.1250%`.
- The same benchmark also has N=4 round 2 with deadline miss rate `15.1375%`; the demo narrative uses round 1 consistently to avoid mixing two repeated runs.
- N=4 benchmark reconciliation: see `n2ew_era\06s2_pynq_z2_wrapper_bringup\N4_BENCHMARK_RECONCILIATION.md`. This report intentionally uses the 06F N=4 round 1 demo mouthpiece; 06S N=4 rows are retained as noisy/stress closure evidence only.

## Scoreboard Definition

- For event `i`, arrival time is `timestamp_us[i] + latency_us`.
- Formal scoreboard metric: `stale_top_change_events`.
- `stale_top_change_events` counts future events after decision event `t` where `best_bid` or `best_ask` actually changes before the latency arrival point. This matches the 06T event-shift definition: the kth future top-of-book price change after `t`.
- `stale_strict_time_top_change_events` is also exported as a timestamp-conservative top-change audit variant.
- `stale_strict_time_events` and `stale_all_events` are auxiliary audit counters only, not the formal scoreboard definition. `stale_strict_time_events` counts all real L2 events with `timestamp > timestamp[i]` and `timestamp <= arrival_time`; `stale_all_events` additionally counts later rows with the same timestamp by event-index order.

## Burst vs Calm Sanity Check

| window | events | duration_sec | event_rate/s | CPU scoreboard top-change stale | FPGA scoreboard top-change stale | CPU-FPGA scoreboard delta | CPU nonzero scoreboard rate | FPGA nonzero scoreboard rate | CPU strict-time top-change audit | FPGA strict-time top-change audit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| burst | 4096 | 14.563000 | 281.26 | 1334 | 1331 | 3 | 23.7549% | 23.7305% | 3 | 0 |
| calm_control | 4096 | 50.604000 | 80.94 | 998 | 998 | 0 | 17.8467% | 17.8467% | 0 | 0 |

Interpretation for the demo:

- The official scoreboard should be described as top-of-book-change staleness, not strict timestamp all-event staleness.
- If the burst CPU top-change counter finishes above FPGA, the replay supports the intended latency-backlog story under the 06T-compatible definition.
- If both official counters are close, say that honestly and use the auxiliary strict-time/all-event counters only as audit context.
- The calm window is included as a control; both counters should be materially flatter than the burst window.

## Outputs

- `results/demo_replay_staleness_timeseries.csv`
- `results/demo_replay_staleness_timeseries.json`
- `results/demo_replay_input_features.csv`
- `results/demo_replay_window_selection.csv`
- `results/demo_replay_summary.csv`
- `figures/demo_replay_staleness_scoreboard.png`
