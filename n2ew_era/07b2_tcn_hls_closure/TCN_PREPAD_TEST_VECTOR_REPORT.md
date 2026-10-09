# TCN PrePad Test Vector Report

- Sequence count per variant: `1000` or fewer if validation payload is smaller.
- Input columns are the unpadded `(128, 16)` PHC feature window. The HLS top function performs the 124-step causal prepad internally.
- Reference readouts are PyTorch float32 model outputs after multiplying by each checkpoint's target scale.
- NEW5 is not loaded.

| variant | sample_count | csv |
| --- | ---: | --- |
| scratch | 1000 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07b2_tcn_hls_closure\results\tcn_prepad_scratch_test_vectors.csv |
| distilled | 1000 | C:\Users\holbo\00VS_projjects\00_HFT_project\n2ew_era\07b2_tcn_hls_closure\results\tcn_prepad_distilled_test_vectors.csv |
