# TCN 5-Block PrePad 18-Feature Sequence Dataset Report

06D reuses the 06C 18-feature TXF-only sequence tensors when available. The model pre-padding is applied by the 06D dataset/model caller, not by rebuilding labels.

- NEW5 is not loaded.
- Feature tensor shape remains `(windows, 128, 18)` before pre-padding.
- 06D model input shape is `(batch, 18, 252)` after caller-side zero prepad.

| split   | source       | path                                                                                                                 |   window_rows |   feature_dim | new5_used   |
|:--------|:-------------|:---------------------------------------------------------------------------------------------------------------------|--------------:|--------------:|:------------|
| train   | 06d_existing | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06d_tcn_5block_prepad_hls4ml\results\feat18_sequences_train.pt |         77194 |            18 | False       |
| val     | 06d_existing | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06d_tcn_5block_prepad_hls4ml\results\feat18_sequences_val.pt   |         82960 |            18 | False       |
| new3    | 06d_existing | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06d_tcn_5block_prepad_hls4ml\results\feat18_sequences_new3.pt  |        176754 |            18 | False       |
| new4    | 06d_existing | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\06d_tcn_5block_prepad_hls4ml\results\feat18_sequences_new4.pt  |         68620 |            18 | False       |
