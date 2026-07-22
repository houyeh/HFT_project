status: resolved
finding: positive
# Audit: Learned Memory vs Fixed Window

## 8.1 Learned decay / gate structure

PHC-lite v1 has learned recurrent `decay_param` and a learned gate/candidate path. It does not have the later STP `gain_t`; that exists in 05C2/05C4 only. For v1, the learnable memory evidence is therefore decay, gate/candidate weights, and the recurrent state update.

|   seed | checkpoint                                                                                  |   best_epoch |   decay_min |   decay_max |   decay_mean |   decay_std |   half_life_min_events |   half_life_max_events |   half_life_mean_events |    alpha |
|-------:|:--------------------------------------------------------------------------------------------|-------------:|------------:|------------:|-------------:|------------:|-----------------------:|-----------------------:|------------------------:|---------:|
|      0 | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite_seed0_best_val.pt |           48 |    0.887263 |    0.907933 |     0.899067 |  0.00477381 |                5.7949  |                7.17659 |                 6.5298  | 0.300145 |
|      1 | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite_seed1_best_val.pt |           50 |    0.893325 |    0.908745 |     0.90046  |  0.00378299 |                6.14465 |                7.2436  |                 6.62103 | 0.301709 |
|      2 | n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite_seed2_best_val.pt |           47 |    0.89353  |    0.904919 |     0.899532 |  0.00361039 |                6.15719 |                6.9377  |                 6.55543 | 0.355801 |

The source recurrent update is in `n2ew_era/05a_phc_lite_microstructure_sequence_feasibility/models/phc_lite.py:52-64`: `state_temporal = decay * state + gate_t * candidate_t`, followed by the within-step loop.

## 8.2 Fixed-decay and memory ablations

No retraining was performed. The same frozen checkpoint and validation-selected threshold were evaluated under counterfactual inference changes: fixed decay 0.9, memory-off decay approximately 0, reversed sequence order, and deterministic timestep shuffle.

| split   | variant                         |   total_excess_pnl_H20 |   delta_pnl_vs_baseline |   spearman_bid |   spearman_ask |   selected_action_count | new5_used   |
|:--------|:--------------------------------|-----------------------:|------------------------:|---------------:|---------------:|------------------------:|:------------|
| val     | baseline                        |            1.92717e+06 |             0           |     0.312335   |     0.313338   |                   63766 | False       |
| val     | fixed_decay_0p9                 |            1.92764e+06 |           470           |     0.312361   |     0.313343   |                   64348 | False       |
| val     | memory_off_decay_0              |            1.12394e+06 |       -803235           |     0.228659   |     0.230274   |                   76495 | False       |
| val     | reverse_sequence                |      -837330           |            -2.7645e+06  |     0.0103825  |     0.0105834  |                   63165 | False       |
| val     | deterministic_shuffle_timesteps |      -642420           |            -2.56959e+06 |     0.022197   |     0.0250668  |                   60232 | False       |
| new3    | baseline                        |            5.02152e+06 |             0           |     0.330413   |     0.329118   |                  138261 | False       |
| new3    | fixed_decay_0p9                 |            5.02198e+06 |           460           |     0.330209   |     0.328363   |                  139337 | False       |
| new3    | memory_off_decay_0              |            3.12178e+06 |            -1.89974e+06 |     0.239885   |     0.240654   |                  160691 | False       |
| new3    | reverse_sequence                |           -1.92616e+06 |            -6.94768e+06 |     0.00442807 |     0.0045308  |                  135407 | False       |
| new3    | deterministic_shuffle_timesteps |           -1.49721e+06 |            -6.51872e+06 |     0.0196549  |     0.0214265  |                  127956 | False       |
| new4    | baseline                        |            1.7625e+06  |             0           |     0.30828    |     0.308326   |                   53169 | False       |
| new4    | fixed_decay_0p9                 |            1.75499e+06 |         -7505           |     0.308244   |     0.307806   |                   53636 | False       |
| new4    | memory_off_decay_0              |       980800           |       -781695           |     0.214181   |     0.214913   |                   62282 | False       |
| new4    | reverse_sequence                |      -767660           |            -2.53016e+06 |     0.00188627 |     0.00301776 |                   52086 | False       |
| new4    | deterministic_shuffle_timesteps |      -552025           |            -2.31452e+06 |     0.0228234  |     0.0253098  |                   49476 | False       |

## 8.3 Time-order sanity check

Reversing or shuffling timesteps collapses Spearman correlation close to zero and makes action PnL negative on val/NEW3/NEW4. Memory-off remains positive but materially weaker than baseline. Fixed decay 0.9 is nearly tied with learned decay, which suggests much of the benefit comes from sequential ordering and gated state, while per-channel decay tuning is not the sole driver.

## Seed stability context
|   seed | split   |   best_epoch |         pnl |   causal_fill_pnl | beats_queue_rule   | beats_hgb   | new5_used_for_selection   |
|-------:|:--------|-------------:|------------:|------------------:|:-------------------|:------------|:--------------------------|
|      0 | val     |           48 | 1.92717e+06 |            170310 | True               | True        | False                     |
|      0 | new3    |           48 | 5.02152e+06 |            488870 | True               | True        | False                     |
|      0 | new4    |           48 | 1.7625e+06  |            121765 | True               | True        | False                     |
|      1 | val     |           50 | 1.80182e+06 |            151530 | True               | True        | False                     |
|      1 | new3    |           50 | 4.7472e+06  |            454620 | True               | True        | False                     |
|      1 | new4    |           50 | 1.63951e+06 |            111995 | True               | True        | False                     |
|      2 | val     |           47 | 1.89674e+06 |            162625 | True               | True        | False                     |
|      2 | new3    |           47 | 4.9602e+06  |            482590 | True               | True        | False                     |
|      2 | new4    |           47 | 1.71606e+06 |            116845 | True               | True        | False                     |

## Final answer for `B_CLASS` §B.2

> PHC-lite is not proven to be a full biological recurrent mechanism, but the frozen inference sanity checks show it uses sequence order: reversing or shuffling timesteps destroys correlation and turns PnL negative, while memory-off materially weakens PnL. Fixed decay 0.9 is close to learned decay, so the honest claim is sequence/gated-state value, not that every learned decay channel is individually essential.
