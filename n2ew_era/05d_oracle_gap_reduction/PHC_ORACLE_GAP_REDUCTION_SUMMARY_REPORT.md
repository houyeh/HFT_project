# PHC Oracle Gap Reduction Summary Report

This report summarizes controlled Phase 05D attempts to reduce the PHC-window Oracle gap. NEW5 is not used for model selection or threshold selection in this phase.

## Summary Table
| split   |   action_pnl | model                  |   causal_fill_pnl |   oracle_capture | adopted_status                                                    |   false_positive_rate |   missed_opportunity_rate |   side_error_rate |
|:--------|-------------:|:-----------------------|------------------:|-----------------:|:------------------------------------------------------------------|----------------------:|--------------------------:|------------------:|
| val     |  1.92717e+06 | phc_lite_v1            |  170310           |         0.249534 | frozen_reference                                                  |            nan        |                nan        |        nan        |
| new3    |  5.02152e+06 | phc_lite_v1            |  488870           |         0.266517 | frozen_reference                                                  |            nan        |                nan        |        nan        |
| new4    |  1.7625e+06  | phc_lite_v1            |  121765           |         0.239057 | frozen_reference                                                  |            nan        |                nan        |        nan        |
| val     |  2.03792e+06 | phc_stp_v2_cosine_lr   |  171835           |         0.264287 | stp_v2_adopted_as_best_sequence_model                             |            nan        |                nan        |        nan        |
| new3    |  5.23741e+06 | phc_stp_v2_cosine_lr   |  611095           |         0.27863  | stp_v2_adopted_as_best_sequence_model                             |            nan        |                nan        |        nan        |
| new4    |  1.87168e+06 | phc_stp_v2_cosine_lr   |  129160           |         0.254392 | stp_v2_adopted_as_best_sequence_model                             |            nan        |                nan        |        nan        |
| val     |  2.06156e+06 | phc_stp_120epoch       |  181945           |         0.266614 | stp_not_adopted_convergence_incomplete                            |            nan        |                nan        |        nan        |
| new3    |  5.25102e+06 | phc_stp_120epoch       |  606750           |         0.278807 | stp_not_adopted_convergence_incomplete                            |            nan        |                nan        |        nan        |
| new4    |  1.87112e+06 | phc_stp_120epoch       |  129555           |         0.253831 | stp_not_adopted_convergence_incomplete                            |            nan        |                nan        |        nan        |
| val     |  1.98794e+06 | phc_rank_head          |  174780           |         0.256633 | phc_rank_head_promising_not_adopted_seed_stability_caveat         |              0.224723 |                  0.168515 |          0.167635 |
| new3    |  5.1532e+06  | phc_rank_head          |  494545           |         0.273693 | phc_rank_head_promising_not_adopted_seed_stability_caveat         |              0.204346 |                  0.15112  |          0.184347 |
| new4    |  1.81528e+06 | phc_rank_head          |  125720           |         0.24615  | phc_rank_head_promising_not_adopted_seed_stability_caveat         |              0.20255  |                  0.161148 |          0.18636  |
| val     |  1.9222e+06  | phc_uncertainty_gating |  166695           |         0.248583 | phc_uncertainty_gating_not_adopted                                |              0.230991 |                  0.163814 |          0.171649 |
| new3    |  4.98981e+06 | phc_uncertainty_gating |  492030           |         0.26468  | phc_uncertainty_gating_not_adopted                                |              0.205234 |                  0.160992 |          0.182604 |
| new4    |  1.7409e+06  | phc_uncertainty_gating |  117080           |         0.235897 | phc_uncertainty_gating_not_adopted                                |              0.204503 |                  0.168173 |          0.18601  |
| val     |  6.33814e+07 | phc_streaming_dense    |       2.9147e+06  |         0.256251 | streaming_dense_hls_feasibility_diagnostic_not_adoption_candidate |            nan        |                nan        |        nan        |
| new3    |  1.61325e+08 | phc_streaming_dense    |       8.69735e+06 |         0.267151 | streaming_dense_hls_feasibility_diagnostic_not_adoption_candidate |            nan        |                nan        |        nan        |
| new4    |  5.7938e+07  | phc_streaming_dense    |       3.12098e+06 |         0.244843 | streaming_dense_hls_feasibility_diagnostic_not_adoption_candidate |            nan        |                nan        |        nan        |

## Missing Outputs
No missing outputs.

## Final Recommendation
`adopted_sequence_model: phc_stp_v2_cosine_lr`

No model is adopted based only on validation. Oracle metrics remain non-causal diagnostics and cannot be used as a tradable strategy.
