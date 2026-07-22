# Checkpoint Lineage Audit

Conclusion: the deployed 06F HLS checkpoint is from the Phase 05C4 `PHCLiteSTP` STP-like dynamic synaptic gain series, seed 2, not from the original Phase 05A plain `PHCLite` model.

## Evidence

- [00_export_weights.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/00_export_weights.py:18) loads the step checkpoint from `n2ew_era/06e_phc_stp_step_hls4ml/models/phc_stp_step_hls_seed2.pt`.
- [00_export_weights.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/00_export_weights.py:19) records the source checkpoint as `n2ew_era/05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt`.
- [00_export_weights.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/00_export_weights.py:86) requires the loaded step checkpoint phase to be `06e`, and [00_export_weights.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/00_export_weights.py:88) rejects checkpoints marked `new5_used=True`.
- [common_06e.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/common_06e.py:42) points the source checkpoint to the Phase 05C4 seed-2 checkpoint.
- [common_06e.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/common_06e.py:77) dynamically imports `PHCLiteSTP` from the 05C2 STP-like model source.
- [common_06e.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/common_06e.py:95) requires the source checkpoint phase to be `05c4`.
- [common_06e.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/common_06e.py:120) migrates that source model into a single-step HLS graph.
- [common_06e.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/common_06e.py:130) copies `gain_from_encoded`; [common_06e.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/common_06e.py:131) copies `gain_from_prev_state`; [common_06e.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/common_06e.py:132) copies `synapse`; these tensors are the STP-like dynamic gain path.
- [PHC_STP_STEP_HLS4ML_FINAL_REPORT.md](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/PHC_STP_STEP_HLS4ML_FINAL_REPORT.md:7) states the source model is adopted Phase 05C4 `PHCLiteSTP`, seed 2.
- [PHC_STP_STEP_HLS4ML_FINAL_REPORT.md](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06e_phc_stp_step_hls4ml/PHC_STP_STEP_HLS4ML_FINAL_REPORT.md:19) lists the source checkpoint as `05c4_phc_stp_cosine_lr/models/phc_stp_v2_seed2_best_val.pt` and the migrated model as `06e_phc_stp_step_hls4ml/models/phc_stp_step_hls_seed2.pt`.
- [PHC_WEIGHT_EXPORT_REPORT.md](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/PHC_WEIGHT_EXPORT_REPORT.md:7) repeats the same Phase 05C4 source checkpoint.
- [PHC_WEIGHT_EXPORT_REPORT.md](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/PHC_WEIGHT_EXPORT_REPORT.md:43) through [PHC_WEIGHT_EXPORT_REPORT.md](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/PHC_WEIGHT_EXPORT_REPORT.md:45) show exported `GAIN_FROM_ENCODED_*` and `GAIN_FROM_PREV_STATE_W` tensors, which do not exist in the 05A plain PHC-lite model.

## Architecture Comparison

### Phase 05A Plain PHC-lite

- [phc_lite.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py:31) defines `gate`, [phc_lite.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py:32) defines `candidate`, and [phc_lite.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py:33) defines `synapse`.
- [phc_lite.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py:61) computes `message = silu(self.synapse(z))`, then [phc_lite.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py:63) applies `z = loop_norm(z + alpha * message)`.
- There is no `gain_from_encoded`, no `gain_from_prev_state`, and no `gain_t` in the Phase 05A model.

### Phase 05C2/05C4 STP-like PHCLiteSTP

- [phc_lite_stp.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05c2_phc_stp_dynamic_synaptic_gain/models/phc_lite_stp.py:37) defines `gain_from_encoded`.
- [phc_lite_stp.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05c2_phc_stp_dynamic_synaptic_gain/models/phc_lite_stp.py:38) defines `gain_from_prev_state`.
- [phc_lite_stp.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05c2_phc_stp_dynamic_synaptic_gain/models/phc_lite_stp.py:54) computes `gain_t = sigmoid(gain_from_encoded(encoded_t) + gain_from_prev_state(state))`.
- [phc_lite_stp.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05c2_phc_stp_dynamic_synaptic_gain/models/phc_lite_stp.py:56) computes the synaptic message and [phc_lite_stp.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05c2_phc_stp_dynamic_synaptic_gain/models/phc_lite_stp.py:57) applies the dynamic gain: `message = dropout(gain_t * message)`.
- [phc_lite_stp.py](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/05c2_phc_stp_dynamic_synaptic_gain/models/phc_lite_stp.py:58) updates `z = loop_norm(z + alpha * message)`.

### Phase 06F Manual HLS Step

- [phc_step_hls.cpp](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp:239) computes `gain_encoded`.
- [phc_step_hls.cpp](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp:240) computes `gain_prev_state`.
- [phc_step_hls.cpp](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp:243) adds them into `gain_pre`, and [phc_step_hls.cpp](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp:245) applies sigmoid to obtain `gain_t`.
- [phc_step_hls.cpp](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp:247) computes the synaptic message.
- [phc_step_hls.cpp](C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/06f_phc_lite_manual_hls/hls_src/phc_step_hls.cpp:252) applies `ALPHA_SCALAR[0] * gain_t[i] * message[i]`.

Therefore STEP5 in the HLS formula is exactly the 05C2/05C4 STP-like dynamic synaptic gain mechanism. It is not a 05A plain PHC-lite recurrence.

## Impact On Section 3.1.1 Comparison Table

If the current proposal table says the deployed HLS model has removed the STP / synaptic dynamic gain mechanism, that wording is inaccurate.

Recommended correction:

- Do not describe the 06F deployed checkpoint as plain 05A PHC-lite with STP removed.
- Describe it as: `Phase 05C4 PHCLiteSTP cosine-LR selected seed=2, manually exported as a single-step HLS datapath`.
- For the "synaptic mechanism" row, use wording like: `STP-like dynamic synaptic gain retained as gain_t = sigmoid(W_g encoded_t + U_g state_prev); not full biological STP`.

Claim boundary:

- This is not full biological STP, because it lacks explicit short-term plasticity state variables and biological depression/facilitation dynamics.
- This is not Dale's Law, ALIF, STDP, or full PHCSSM.
- The HLS checkpoint lineage is consistent with 05C4 adoption evidence, but any proposal text saying "STP removed" should be revised.
