# TCN 18-Feature Sequence Dataset Report

`tcn_av_16feat` reuses Phase 05A PHC-lite sequence tensors. `tcn_av_18feat` adds train-normalized `hawkes_excite` and causal `time_in_session_norm`.

- Product universe: TXF-only inherited from Phase 04B.
- Sequence setting: `seq_len=128`, `stride=32`, target is final timestep H20.
- Scalers: PHC feature scaler and hawkes scaler fit on train only.
- NEW5 is not loaded.

| split   |   event_rows |   window_rows |
|:--------|-------------:|--------------:|
| new3    |      5656822 |        176754 |
| new4    |      2196280 |         68620 |
| train   |      2470559 |         77194 |
| val     |      2655369 |         82960 |
