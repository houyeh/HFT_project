# PHC STP-like Code Audit Report

Overall status: `PASS`.

Audited implementation: `common_05c2.py` and `models/phc_lite_stp.py`.

| audit_item                               | status   | evidence                                                                                          | file_path                                                                                                       |
|:-----------------------------------------|:---------|:--------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|
| train_val_new3_new4_split_handling       | PASS     | Training uses train payload; evaluation loops over val/new3/new4.                                 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| no_new5_usage                            | PASS     | No NEW5 sequence loading found; selected JSON marks new5_used_for_selection=false.                | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| validation_only_epoch_selection          | PASS     | Checkpoint selected by validation policy excess PnL.                                              | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| validation_only_threshold_seed_selection | PASS     | Threshold grid CSV contains only selection_split=val and false OOS selection flags.               | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| feature_list                             | PASS     | Model feature_dim is len(p05a.PHC_FEATURES); inherited forbidden-feature contract from Phase 05A. | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| target_excess_pnl_H20                    | PASS     | Selected policy target is excess_pnl_H20; metrics use Phase 05A horizon constant H20.             | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| matched_random_implementation            | PASS     | Matched random delegates to Phase 05A count/session/side/spread-bucket matcher.                   | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| causal_fill_implementation               | PASS     | Causal-fill uses train-derived norms, stable uniforms, and no future-PnL fill decision.           | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| oracle_capture_computation               | PASS     | Oracle capture explicitly uses future labels and is diagnostic only.                              | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |
| stp_gain_definition                      | PASS     | gain_t = sigmoid(W_g encoded_t + U_g state_{t-1}) exists in model source.                         | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\models\phc_lite_stp.py |
| not_full_biological_stp                  | PASS     | Model docstring states it is not full biological STP.                                             | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\models\phc_lite_stp.py |
| new5_not_in_outputs                      | PASS     | OOS/oracle output splits exclude NEW5.                                                            | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\05c2_phc_stp_dynamic_synaptic_gain\common_05c2.py         |

This audit checks code structure and produced artifacts. It does not adopt STP-like PHC and does not use NEW5.
