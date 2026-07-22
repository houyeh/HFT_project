# Residual Universe Filtering Waterfall Table

| split | raw_rows | label_rows | valid_book_rows | quote_eligible_rows | candidate_action_rows | bid_candidate_action_rows | ask_candidate_action_rows | baseline_selected_rows | residual_ml_rows | negative_ev_moderate_rows | severe_negative_ev_moderate_rows | negative_ev_rate | severe_negative_ev_rate | session_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3593426 | 3592764 | 3593424 | 3592764 | 1726 | 820 | 906 | 820 | 820 | 226 | 173 | 27.5610% | 21.0976% | 6 |
| val | 563593 | 563477 | 563587 | 563477 | 262 | 134 | 128 | 134 | 134 | 17 | 3 | 12.6866% | 2.2388% | 1 |
| test | 1166541 | 1166321 | 1166540 | 1166321 | 934 | 419 | 515 | 419 | 419 | 16 | 0 | 3.8186% | 0.0000% | 2 |
| NEW4 | 2274678 | 2197106 | 2197106 | 2197106 | 1306 | 714 | 592 | 714 | 714 | 103 | 36 | 14.4258% | 5.0420% | 4 |

Severe negative EV is defined as `quote_ev_moderate <= -5000`, matching the Phase 030 GPU export convention.
