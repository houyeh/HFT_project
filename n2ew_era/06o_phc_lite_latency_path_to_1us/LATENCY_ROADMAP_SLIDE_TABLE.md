# Latency Roadmap Slide Table

Status: `slide_ready_summary_updated_with_90mhz_routed_result`.

Resource percentages use the PYNQ-Z2 / Zynq-7020 budget: LUT 53,200, FF 106,400, DSP 220, BRAM_18K 280.

Source reports:

- `n2ew_era/06o_phc_lite_latency_path_to_1us/PHC_LATENCY_PATH_TO_1US_REPORT.md`
- `n2ew_era/06l_phc_lite_pipeline_budget_sweep/AUDIT_unroll16_pnr_timing.md`

| Stage | Latency | Frequency | Verification level | Resource cost | Next step |
|---|---:|---:|---|---|---|
| 06F frozen v8 baseline | 8316 cycles / 83.160us | 100.000MHz | `routed_pnr_met` | DSP 47 (21.4%), LUT 7555 (14.2%), FF 8217 (7.7%), BRAM_18K 6 (2.1%) | Conservative board-wrapper baseline; use when timing evidence matters more than latency. |
| 06L unroll16 routed fallback | 2316 cycles / 25.733us | 90.001MHz | `routed_pnr_met` | DSP 125 (56.8%), LUT 14826 (27.9%), FF 19805 (18.6%), BRAM_18K 0 (0.0%) | Best verified low-latency point; confirm wrapper variant-adapter path before board flow. |
| 06L unroll16 100MHz target | 2316 cycles / 23.160us | 100.000MHz | `csim_cosim_pass` plus `routed_pnr_failed` at 100MHz | failed routed attempts about DSP 125 (56.8%), LUT 14.8k (27.8%), FF 19.8k (18.6%), BRAM_18K 0 (0.0%) | Keep as timing-closure target only. Baseline/aggressive/fanout P&R all failed at 100MHz; high-fanout FSM-to-DSP48E1-PCIN routing is the root mechanism. |
| Sub-10us / ~1us architecture path | <10us target, ~1us long target | TBD | `estimate_only` until redesigned and routed | TBD; 06G diagonal clue is DSP 34 (15.5%), LUT 16058 (30.2%), FF 9066 (8.5%), BRAM_18K 1 (0.4%) but `csynth_only` | Clock/quantization alone cannot get there. Need fewer 16x16 dense blocks or structured/diagonal/feedforward transforms; 06G accuracy regresses vs production sequence model and is not a drop-in replacement. |

Headline wording for slides:

`PHC-lite latency is now 83.160us routed baseline -> 25.733us routed unroll16 at 90MHz. The 23.160us 100MHz point is functionally verified but not timing-closed. Reaching ~1us requires architecture-level simplification, not clock/quantization alone.`

