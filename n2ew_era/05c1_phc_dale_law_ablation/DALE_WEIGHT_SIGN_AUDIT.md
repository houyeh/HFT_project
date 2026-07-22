# Dale Weight Sign Audit

Dale's Law is enforced on columns of the synapse matrix because `message = z @ W_eff.T + bias`; column `j` is the outgoing fan-out from presynaptic hidden unit `j`.

Hidden dim: `16`.
Excitatory units: `13`.
Inhibitory units: `3`.
Total sign violations: `0`.

|   hidden_dim |   presynaptic_unit | unit_type   |   sign_value |   outgoing_weight_min |   outgoing_weight_max |   sign_violations | constraint_enforced_during_training   |   excitatory_units |   inhibitory_units |
|-------------:|-------------------:|:------------|-------------:|----------------------:|----------------------:|------------------:|:--------------------------------------|-------------------:|-------------------:|
|           16 |                  0 | excitatory  |            1 |              0.575701 |              0.815184 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  1 | excitatory  |            1 |              0.555551 |              0.858377 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  2 | excitatory  |            1 |              0.572034 |              0.832651 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  3 | excitatory  |            1 |              0.5714   |              0.802735 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  4 | excitatory  |            1 |              0.562696 |              0.804973 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  5 | excitatory  |            1 |              0.575315 |              0.794936 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  6 | excitatory  |            1 |              0.578134 |              0.822722 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  7 | excitatory  |            1 |              0.606655 |              0.803075 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  8 | excitatory  |            1 |              0.561293 |              0.827655 |                 0 | True                                  |                 13 |                  3 |
|           16 |                  9 | excitatory  |            1 |              0.579217 |              0.807626 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 10 | excitatory  |            1 |              0.560044 |              0.822426 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 11 | excitatory  |            1 |              0.584952 |              0.837847 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 12 | excitatory  |            1 |              0.494075 |              0.86131  |                 0 | True                                  |                 13 |                  3 |
|           16 |                 13 | inhibitory  |           -1 |             -0.817954 |             -0.563712 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 14 | inhibitory  |           -1 |             -0.835665 |             -0.538733 |                 0 | True                                  |                 13 |                  3 |
|           16 |                 15 | inhibitory  |           -1 |             -0.878153 |             -0.548977 |                 0 | True                                  |                 13 |                  3 |
