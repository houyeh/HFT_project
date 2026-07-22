# Latency-Aware Fill Proxy Blocker

The latency-aware CPU scenario requires empirical raw CPU latency samples. Raw samples were recaptured, but the rerun summary does not match the earlier LibTorch CPU summary within the configured 10% relative tolerance.

| metric | original | raw_rerun | relative_delta |
|---|---:|---:|---:|
| mean_us | 74.5444 | 63.9025 | -14.28% |
| median_us | 51 | 45.4 | -10.98% |
| p99_us | 636.7 | 386.4 | -39.31% |
| min_us | 43.8 | 37.3 | -14.84% |

This script refuses to run the CPU-vs-FPGA PnL comparison by default because mixing the older summary with the newly captured empirical raw distribution would weaken the latency-tail claim.

To proceed deliberately with the newly captured raw empirical distribution, rerun with `--allow-raw-drift`. The generated report will keep this caveat explicit.
