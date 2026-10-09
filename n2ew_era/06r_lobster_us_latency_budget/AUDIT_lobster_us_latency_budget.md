# AUDIT: LOBSTER US Latency Budget Pilot

Status: `resolved_microsecond_interarrival_and_stale_exposure_measured`.

## Correction Notice

Status: `cpu_comparator_corrected_2026_08_08`.

An earlier 06R version used the 06S N=4 concurrent LibTorch p99 rows (`646.318us` and `6299.416us`) as the CPU comparison baseline. Those rows were later reconciled in `n2ew_era/06s2_pynq_z2_wrapper_bringup/N4_BENCHMARK_RECONCILIATION.md` as noisy/stress closure measurements rather than the official demo/paper mouthpiece. This report now uses the cleaner 06F N=4 round 1 p99 (`262.6us`) for the official CPU stale-exposure comparison, while retaining the 06S rows explicitly as contaminated/stress context.

## Timestamp Resolution Statement

LOBSTER message timestamps were parsed as decimal strings representing seconds after midnight and converted to integer nanoseconds by splitting the string at the decimal point. No float64 timestamp conversion was used for interval or stale-exposure calculations.

Across the five official level-10 samples, the verified minimum non-zero adjacent event interval is `0.169us` for all L2 updates and `0.254us` for top-of-book price-change events. Therefore `P(delta_t < 25.733us)` and `P(delta_t < 83.16us)` are directly measurable in this pilot, unlike the 1ms TXF feed audited in 06P.

## Claim Boundary

- This is a five-ticker, single-day 2012 NASDAQ free-sample pilot. It is not TXF and should not be treated as a Taiwan futures market estimate.
- Stale exposure is a probability that the visible top-of-book changes inside a latency window after an L2 event. It is not execution impact, not fill probability, and not PnL.
- The old `83.16us` FPGA baseline is preserved only as a historical 06F comparison. The current demo-aligned FPGA latency is `25.733us` from the routed 06L/06S2 unroll16 90MHz evidence.
- Official CPU comparison in this 06R pilot uses the 06F N=4 round 1 p99 (`262.6us`) from `n2ew_era/06f_phc_lite_manual_hls/results/cpu_concurrent_benchmark_N4.csv`, following the reconciliation in `n2ew_era/06s2_pynq_z2_wrapper_bringup/N4_BENCHMARK_RECONCILIATION.md`. The 06S N=4 rows (`646.318us` and `6299.416us`) are retained only as contaminated/noisy stress windows, not as official demo or paper comparator numbers.

No model inference, fill simulation, execution simulation, or PnL calculation was performed.

## Data Scope

- Source: LOBSTER official free samples page, `2012-06-21`.
- Tickers: AAPL, MSFT, INTC, AMZN, GOOG.
- Level: 10 for all five files.
- Time range: 09:30:00-16:00:00, encoded as `34200000_57600000` in filenames.
- Message schema: `N x 6`.
- Orderbook schema: `N x 40`, ordered ask price/size then bid price/size by level.
- Peak windows: per ticker and event definition, 1-second windows in the top 10% by event count. This includes auction-adjacent 09:30-09:35 and 15:55-16:00 seconds where they fall into the top decile; exact counts are in `results/peak_window_summary.csv`.

## Data Validation

| ticker | message rows | orderbook rows | columns ok | monotonic | min nonzero dt | same-ts | top-price-change count |
|---|---:|---:|---|---|---:|---:|---:|
| AAPL | 400391 | 400391 | True | True | 0.243us | 4.011589% | 64350 |
| MSFT | 668765 | 668765 | True | True | 0.169us | 8.553541% | 4060 |
| INTC | 624040 | 624040 | True | True | 0.171us | 10.557994% | 3217 |
| AMZN | 269748 | 269748 | True | True | 0.230us | 3.144799% | 27557 |
| GOOG | 147916 | 147916 | True | True | 0.303us | 5.771558% | 24084 |

## Aggregate Inter-arrival Results

| ticker | event definition | scope | intervals | same-ts | min nonzero | p50 | p90 | p99 | p999 | P(<25.733us) | P(<83.16us) | P(<646.318us) | P(<6.299ms) | P(<1ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | any_l2_update | all_event_flow | 2110855 | 7.398471% | 0.169us | 145.432us | 113.249ms | 1.044s | 2.878s | 29.873961% | 43.420178% | 63.519569% | 77.385041% | 67.313008% |
| ALL | any_l2_update | peak10_all_event_flow | 1243709 | 8.744489% | 0.169us | 46.121us | 6.748ms | 218.383ms | 997.843ms | 41.893964% | 57.654001% | 77.169016% | 89.758135% | 80.874143% |
| ALL | top_of_book_price_change | all_event_flow | 123263 | 2.987920% | 0.254us | 7.611ms | 1.904s | 14.881s | 67.674s | 5.227846% | 10.911628% | 30.546879% | 48.781062% | 35.726860% |
| ALL | top_of_book_price_change | peak10_all_event_flow | 56250 | 4.798222% | 0.285us | 904.900us | 202.056ms | 4.952s | 42.488s | 7.182222% | 15.116444% | 42.275556% | 67.672889% | 50.947556% |

## Per-Ticker Inter-arrival Results

| ticker | event definition | scope | intervals | same-ts | min nonzero | p50 | p90 | p99 | p999 | P(<25.733us) | P(<83.16us) | P(<646.318us) | P(<6.299ms) | P(<1ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAPL | any_l2_update | all_event_flow | 400390 | 4.011589% | 0.243us | 638.857us | 146.525ms | 997.794ms | 2.383s | 15.819576% | 27.197482% | 50.766003% | 72.793027% | 58.330128% |
| AAPL | top_of_book_price_change | all_event_flow | 64349 | 2.520630% | 0.254us | 7.144ms | 995.447ms | 5.072s | 11.161s | 5.934824% | 12.216196% | 30.922003% | 49.317006% | 37.026216% |
| MSFT | any_l2_update | all_event_flow | 668764 | 8.553541% | 0.169us | 50.580us | 66.673ms | 695.235ms | 1.800s | 42.079568% | 56.489883% | 72.878773% | 82.268782% | 75.141156% |
| MSFT | top_of_book_price_change | all_event_flow | 4059 | 1.182557% | 0.458us | 4.345ms | 17.412s | 74.265s | 142.623s | 2.759300% | 8.647450% | 25.794531% | 51.983247% | 31.116038% |
| INTC | any_l2_update | all_event_flow | 624039 | 10.557994% | 0.171us | 65.997us | 69.279ms | 760.813ms | 1.898s | 38.750142% | 53.944545% | 70.422361% | 81.534327% | 72.886310% |
| INTC | top_of_book_price_change | all_event_flow | 3216 | 0.839552% | 1.399us | 4.776ms | 21.072s | 105.818s | 224.914s | 2.456468% | 10.820896% | 26.026119% | 51.523632% | 30.876866% |
| AMZN | any_l2_update | all_event_flow | 269747 | 3.144799% | 0.230us | 620.242us | 220.845ms | 1.461s | 3.679s | 11.130059% | 22.980422% | 52.253037% | 69.524777% | 56.947807% |
| AMZN | top_of_book_price_change | all_event_flow | 27556 | 3.295108% | 0.312us | 10.783ms | 2.487s | 11.628s | 24.001s | 4.311221% | 10.669183% | 28.763246% | 45.409348% | 31.934969% |
| GOOG | any_l2_update | all_event_flow | 147915 | 5.771558% | 0.303us | 829.702us | 432.681ms | 2.530s | 5.708s | 9.467600% | 21.115506% | 47.150728% | 64.563432% | 51.624920% |
| GOOG | top_of_book_price_change | all_event_flow | 24083 | 4.476187% | 0.683us | 5.960ms | 3.108s | 12.944s | 26.240s | 5.173774% | 8.096998% | 32.990076% | 50.301042% | 38.018519% |

## Stale-Exposure Results

For each L2 event `t`, stale exposure is measured as whether the next later top-of-book price change occurs within latency `L`. Same-timestamp later top-of-book changes count as stale with 0us delay.

| ticker | stale@25.733us FPGA | stale@83.16us legacy FPGA | stale@262.6us 06F N4 official CPU | delta official CPU-FPGA | stale@646.318us 06S stress r1 | stale@6299.416us 06S stress r2 | valid events |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 3.321520% | 5.766392% | 11.051752% | 7.730232% | 13.794585% | 25.357090% | 400389 |
| MSFT | 2.122713% | 3.283930% | 6.162735% | 4.040022% | 9.904898% | 21.223840% | 668437 |
| INTC | 1.857141% | 2.789882% | 5.235118% | 3.377976% | 8.322014% | 18.115146% | 623539 |
| AMZN | 1.581853% | 3.296051% | 6.618424% | 5.036571% | 9.102604% | 19.147572% | 269747 |
| GOOG | 2.561883% | 4.733636% | 10.785063% | 8.223179% | 14.705306% | 24.248305% | 147899 |
| ALL | 2.233353% | 3.712161% | 7.198588% | 4.965235% | 10.409140% | 21.036051% | 2110011 |

Aggregate stale exposure is `2.233353%` at the current demo-aligned FPGA latency (`25.733us`) versus `3.712161%` at the old 06F baseline (`83.16us`). The historical 06F shift therefore contributed `1.478807%` additional stale exposure relative to the current FPGA latency.

Official CPU comparison now uses the 06F N=4 round 1 p99 (`262.6us`), which maps to aggregate stale exposure `7.198588%`. The official CPU-minus-FPGA stale-exposure gap is therefore `4.965235%` relative to the current demo-aligned FPGA latency (`25.733us`, `2.233353%` stale exposure).

The older 06S N=4 rows are preserved only as contaminated/noisy stress context: `10.409140%` at `646.318us` and `21.036051%` at `6299.416us`. They are not official demo or paper comparator numbers.

This is a probability-of-book-change exposure delta only. It is not a PnL delta.

## 06P TXF Comparison

| metric | LOBSTER any L2 | TXF 06P | note |
|---|---:|---:|---|
| all_event_flow: same_timestamp_rate | 7.398471% | 67.874085% | TXF same-timestamp rate is a 1ms-resolution batch indicator. |
| all_event_flow: min_nonzero_dt_us | 0.169us | 1.000ms |  |
| all_event_flow: p50_dt_us | 145.432us | 0.000us |  |
| all_event_flow: p90_dt_us | 113.249ms | 34.000ms |  |
| all_event_flow: p99_dt_us | 1.044s | 414.000ms |  |
| all_event_flow: p999_dt_us | 2.878s | 1.442s |  |
| all_event_flow: P(dt < 23.16us) | 28.703156% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 25.733us) | 29.873961% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 51.0us) | 37.254430% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 74.54us) | 41.666718% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 83.16us) | 43.420178% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 636.7us) | 63.177433% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 646.318us) | 63.519569% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 1000.0us) | 67.313008% | 67.874085% | N/A expected for sub-1ms TXF thresholds. |
| all_event_flow: P(dt < 6299.416us) | 77.385041% | 85.002255% | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: same_timestamp_rate | 8.744489% | 71.999935% | TXF same-timestamp rate is a 1ms-resolution batch indicator. |
| peak10_all_event_flow: min_nonzero_dt_us | 0.169us | 1.000ms |  |
| peak10_all_event_flow: p50_dt_us | 46.121us | 0.000us |  |
| peak10_all_event_flow: p90_dt_us | 6.748ms | 9.000ms |  |
| peak10_all_event_flow: p99_dt_us | 218.383ms | 109.000ms |  |
| peak10_all_event_flow: p999_dt_us | 997.843ms | 340.000ms |  |
| peak10_all_event_flow: P(dt < 23.16us) | 40.274775% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 25.733us) | 41.893964% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 51.0us) | 51.055995% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 74.54us) | 55.964699% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 83.16us) | 57.654001% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 636.7us) | 76.894836% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 646.318us) | 77.169016% | N/A below 1ms timestamp resolution | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 1000.0us) | 80.874143% | 71.999935% | N/A expected for sub-1ms TXF thresholds. |
| peak10_all_event_flow: P(dt < 6299.416us) | 89.758135% | 90.146166% | N/A expected for sub-1ms TXF thresholds. |

## Interpretation

- LOBSTER confirms that a nanosecond-stamped public sample can directly measure sub-100us event intervals; the TXF 1ms feed cannot.
- In the five-stock single-day sample, all-L2-update intervals are often far below both 25.733us and 83.16us, especially in peak windows.
- Top-of-book price-change intervals are less frequent than all L2 updates, but stale exposure still increases as latency rises from 25.733us to the official 06F N=4 CPU p99 window (`262.6us`), and rises further under the older 06S noisy/stress windows.
- These numbers are market/sample-specific and should be treated as method validation plus scale reference only.

## Output Files

- Data manifest: `data_manifest.md`
- File manifest: `results/lobster_file_manifest.csv`
- Validation summary: `results/lobster_validation_summary.csv`
- Inter-arrival summary: `results/event_interarrival_latency_budget_summary.csv`
- Stale exposure summary: `results/stale_exposure_summary.csv`
- Corrected official 06F N=4 stale exposure summary: `results/stale_exposure_summary_06f_corrected.csv`
- CPU-vs-FPGA stale delta table: `results/stale_exposure_cpu_minus_fpga.csv`
- Corrected official 06F N=4 CPU-vs-FPGA stale delta table: `results/stale_exposure_cpu_minus_fpga_06f_corrected.csv`
- Latency comparison table: `results/stale_exposure_latency_comparison.csv`
- 06P comparison: `results/txf_06p_comparison.csv`
- CDF figures: `figures/lobster_interarrival_cdf.png`, `figures/lobster_stale_exposure_cdf.png`

## Sources

- LOBSTER official samples page: <https://php.lobsterdata.com/info/DataSamples.php>
- LOBSTER output structure page: <https://php.lobsterdata.com/info/DataStructure.php>
- TXF 06P audit: `n2ew_era/06p_event_interarrival_latency_budget/AUDIT_event_interarrival_latency_budget.md`
- Official demo/paper CPU N=4 p99 latency source: `n2ew_era/06f_phc_lite_manual_hls/results/cpu_concurrent_benchmark_N4.csv`
- 06R stress-window CPU N=4 p99 latency source retained as non-official context: `n2ew_era/06s_unroll16_verification_closure/results/final_latency_comparison.csv`
- N=4 benchmark reconciliation: `n2ew_era/06s2_pynq_z2_wrapper_bringup/N4_BENCHMARK_RECONCILIATION.md`

## Honest Conclusion

06R resolves the measurement question for this pilot sample: public LOBSTER data can measure microsecond event intervals and stale-exposure probabilities directly. It does not resolve TXF stale exposure, estimate execution/PnL impact, or repair the 06Q fill-proxy limitation.
