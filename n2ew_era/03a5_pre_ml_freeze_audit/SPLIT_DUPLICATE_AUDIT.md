# Split Duplicate Audit

Hard rule: train, val, test, NEW3, and NEW4 must not overlap by session_id, timestamp sequence hash, or price sequence hash.

| split_a   | split_b   |   session_id_overlap_count |   timestamp_sequence_hash_overlap_count |   price_sequence_hash_overlap_count | session_id_overlap_values   | independence_pass   | classification   |
|:----------|:----------|---------------------------:|----------------------------------------:|------------------------------------:|:----------------------------|:--------------------|:-----------------|
| train     | val       |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| train     | test      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| train     | new4      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| train     | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| val       | test      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| val       | new4      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| val       | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| test      | new4      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| test      | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |
| new4      | new3      |                          0 |                                       0 |                                   0 |                             | True                | independent      |

Session-level signatures are saved to `results/split_session_signatures.csv`.
