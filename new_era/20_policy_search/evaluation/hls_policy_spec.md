# HLS Policy Specification

This specification maps the validation-selected policy into fixed hardware-friendly logic. It does not require dynamic retraining, online optimization, or floating-point heavy inference in the policy layer.

## Inputs Per Event

- `spread_ticks`
- `bid_prob_toxic` and `ask_prob_toxic` from the frozen 016_v2 risk scorer
- `score_bid_015` and `score_ask_015` from the frozen 015 scorer when the selected policy uses an ensemble

## Output Actions

- `pull_bid`: cancel/passivate the bid quote for the event
- `pull_ask`: cancel/passivate the ask quote for the event

## Selected Logic

- policy_id: `C_ensemble_weighted_w0.75_s3_r0.05`
- policy_type: `ensemble`
- max_spread_ticks: `3.0`

```text
wide_spread = spread_ticks > MAX_SPREAD_TICKS
bid_weighted_score = W_016 * bid_prob_toxic + (1 - W_016) * score_bid_015
ask_weighted_score = W_016 * ask_prob_toxic + (1 - W_016) * score_ask_015
pull_bid = wide_spread OR (bid_weighted_score >= BID_WEIGHTED_THRESHOLD)
pull_ask = wide_spread OR (ask_weighted_score >= ASK_WEIGHTED_THRESHOLD)
```

- `W_016 = 0.75`
- `BID_WEIGHTED_THRESHOLD = 0.14466540329158306`
- `ASK_WEIGHTED_THRESHOLD = 0.1645185947418213`
- audit reference 016 bid threshold: `0.024306584149599075`
- audit reference 016 ask threshold: `0.03156697750091553`

The `0.75` weight can be represented as a fixed-point multiply or as a simple `3/4` shift/add approximation after score quantization.

## Hardware Notes

- Threshold comparisons can be implemented as fixed-point comparators.
- Spread filtering is an integer tick comparison.
- The policy layer is branch/table logic, not online learning.
- The report demonstrates defensive value under proxy stress, not real-world production profitability.