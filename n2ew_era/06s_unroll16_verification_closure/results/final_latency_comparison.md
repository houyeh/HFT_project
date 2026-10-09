# 06S Unroll16 Verification Closure

Status: verification_closure_completed_with_caveats.

Scope: this phase copies and verifies existing frozen artifacts; it does not retrain models, alter weights, alter thresholds, or touch NEW5.

## Part A: Unroll16@90MHz Cosim

- 06S seed2 full 25,600-row CSim passed and the produced full cosim CSV is bit-exact against the CSim CSV.
- The 06S full 25,600-row RTL cosim did not produce an HLS `hls_cosim.rpt` before the 60-minute run was stopped, so its latency/II is not accepted as a full 06S cosim report.
- 06S seed2 1,000-row fallback cosim passed: latency 2310 cycles, II/interval 2311 cycles, max_abs_diff 0.007253, bit-exact vs fallback CSim CSV `true`.
- Existing 06L full unroll16 evidence remains the final routed headline: full cosim `Pass`, latency 2316 cycles, interval 2317 cycles, 90MHz routed WNS 0.151ns.

## Part B: Cross-seed 90MHz P&R

Seed0 and seed1 were exported through the same copied unroll16 HLS source and 90MHz flow. Seed0 routed timing met; seed1 did not meet timing at 90MHz. Seed2 uses the already routed 06L artifact as the accepted production latency headline.

## FPGA Cross-seed Evidence

| seed | source | csim | csim rows | csim max abs diff | 06S full cosim note | cosim rpt | cosim rows | bit-exact vs csim | csynth/cosim cycles | latency @90MHz us | routed DSP | routed LUT | routed FF | routed BRAM_18K | WNS ns | timing met |
|---:|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 06s | pass | 25600 | 0.006088 | missing | missing | missing | false | 2310 | 25.666 | 144 (65.5%) | 14981 (28.2%) | 19006 (17.9%) | 0 (0.0%) | 0.027 | true |
| 1 | 06s | pass | 25600 | 0.007211 | missing | missing | missing | false | 2310 | 25.666 | 142 (64.5%) | 14919 (28.0%) | 19055 (17.9%) | 0 (0.0%) | -0.155 | false |
| 2 | 06s_full_cosim_csv_no_rpt_timeout_plus_06l_existing_full_evidence | pass | 25600 | 0.007253 | timeout_or_incomplete_rtl_report_missing | missing | 25600 | true | 2310 | 25.666 | 125 (56.8%) | 14826 (27.9%) | 19805 (18.6%) | 0 (0.0%) | 0.151 | true |

### Seed2 Cosim Status

- 06S full 25,600-row CSim/CSV comparison exists and is bit-exact between CSim CSV and the produced Cosim CSV, but the HLS RTL cosim report was not produced before the 60-minute run was stopped. Therefore the 06S full run is labeled `timeout_or_incomplete_rtl_report_missing`.
- 06S fallback 1,000-row cosim: status `Pass`, rows 1000, max_abs_diff 0.007253, bit-exact vs CSim CSV `true`, latency 2310 cycles / interval 2311 cycles.
- Existing 06L full unroll16 evidence remains the routed headline: full cosim `Pass`, latency 2316 cycles / interval 2317 cycles, routed timing WNS 0.151ns.

Note: the 06S resynthesis at 90MHz reports 2310 cycles, while the already routed 06L unroll16 artifact reports 2316 cycles. The final latency headline keeps the routed 06L artifact at 25.733us.

## Part C: CPU LibTorch Baselines

CPU N=1/N=4/N=16 were measured in this phase with 10,000 iterations per stream. CPU N=8 is carried forward from the existing 06F concurrent benchmark raw/summary files because 06S did not rerun N=8. The CPU deadline miss behavior is not monotonic across N; this is preserved as measured rather than smoothed.

## Final Latency Comparison

| implementation | verification level | mean us | median us | p99 us | max us | jitter p99-median us | miss >83.16us % | miss >25.733us % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FPGA PHC v8 baseline | routed_pnr_met_plus_csim_cosim | 83.160 | 83.160 | 83.160 | 83.160 | 0.000 | 0.000 | 100.000 |
| FPGA PHC unroll16 seed2 @90MHz | routed_pnr_met_06l_full_cosim_plus_06s_cosim1000 | 25.733 | 25.733 | 25.733 | 25.733 | 0.000 | 0.000 | 0.000 |
| CPU LibTorch C++ N=1 | host_benchmark | 898.678 | 73.600 | 21019.382 | 332993.100 | 20945.782 | 41.130 | 100.000 |
| CPU LibTorch C++ N=1 | host_benchmark | 992.151 | 96.700 | 20191.726 | 306264.200 | 20095.026 | 71.090 | 100.000 |
| CPU LibTorch C++ N=4 | host_benchmark | 110.187 | 83.400 | 646.318 | 18518.200 | 562.918 | 50.377 | 100.000 |
| CPU LibTorch C++ N=4 | host_benchmark | 242.065 | 87.800 | 6299.416 | 81096.200 | 6211.616 | 59.812 | 100.000 |
| CPU LibTorch C++ N=8 | host_benchmark | 79.867 | 63.800 | 202.600 | 32380.600 | 138.800 | 8.381 | 100.000 |
| CPU LibTorch C++ N=8 | host_benchmark | 90.140 | 69.900 | 220.200 | 26408.200 | 150.300 | 9.375 | 100.000 |
| CPU LibTorch C++ N=16 | host_benchmark | 1226.428 | 66.800 | 35629.525 | 462182.100 | 35562.725 | 25.868 | 100.000 |
| CPU LibTorch C++ N=16 | host_benchmark | 1282.259 | 79.500 | 28999.447 | 262372.300 | 28919.947 | 40.233 | 100.000 |

Claim boundary: this table compares latency and verification evidence only. It does not claim FPGA PnL improvement and does not use NEW5.
