status: open
# Audit: PHC-Lite HLS Cross-Seed Resource Stability

Hard rules: this audit does not load NEW5 and does not claim cross-seed HLS stability without seed0/seed1 synthesis evidence.

## Scope

Task E asked for a PHC-lite seed0/seed1 HLS resource stability retest, with at least seed0/seed1/seed2 csynth table coverage. I found the trained source checkpoints for seeds 0, 1, and 2, but the existing 06F manual HLS flow is currently a seed2-only export/run: `00_export_weights.py` hardcodes `phc_stp_v2_seed2_best_val.pt` and `phc_stp_step_hls_seed2.pt`.

No seed0 or seed1 HLS export/csynth run was completed in this pass. Therefore the requested stability claim remains open.

## Artifact Inventory

| seed | source checkpoint | step HLS checkpoint | exported HLS weights | csynth report | cosim / implementation reports | evidence status |
| ---- | ----------------- | ------------------- | -------------------- | ------------- | ------------------------------ | --------------- |
| 0 | `n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed0_best_val.pt` | Missing from 06F artifacts | Missing seed-specific export | Not run | Not run | Open |
| 1 | `n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed1_best_val.pt` | Missing from 06F artifacts | Missing seed-specific export | Not run | Not run | Open |
| 2 | `n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt` | `n2ew_era/06f_phc_lite_manual_hls/models/phc_stp_step_hls_seed2.pt` | Existing 06F export | Present | Present | Baseline only |

## csynth Resource Table

| seed | csynth status | top | latency cycles | interval cycles | slack ns | BRAM | DSP | FF | LUT | notes |
| ---- | ------------- | --- | -------------- | --------------- | -------- | ---- | --- | -- | --- | ----- |
| 0 | Not run | - | - | - | - | - | - | - | - | Required retest missing |
| 1 | Not run | - | - | - | - | - | - | - | - | Required retest missing |
| 2 | Existing report | `phc_step` | 8316 | 8317 | 0.15 | 6 | 46 | 14518 | 24836 | Baseline from `hls/syn/report/csynth.rpt` |

## Seed2 Implementation Baseline

| stage | timing status | target ns | achieved ns | slack ns | BRAM | DSP | FF | LUT | SRL |
| ----- | ------------- | --------- | ----------- | -------- | ---- | --- | -- | --- | --- |
| RTL synthesis | Met | 10.000 | 9.729 | 0.271 | 6 | 47 | 8217 | 7754 | 37 |
| Place and route | Met | 10.000 | 9.980 | 0.020 | 6 | 47 | 8217 | 7555 | 26 |

Seed2 cosim status: Verilog pass, latency 8316 cycles, interval 8317 cycles, total execution time 212,915,199 cycles.

## Conclusion

Cross-seed HLS resource stability is not established. The only complete HLS evidence available here is seed2. To close Task E, seed0 and seed1 need separate weight export artifacts and independent HLS work directories, followed by csynth at minimum and ideally cosim plus implementation reports under the same 06F settings. The existing seed2 settings and reports were left unchanged.
