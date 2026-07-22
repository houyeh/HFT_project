# HLS Stress-Robust Policy Specification

This is a fixed policy specification for latency risk-control research. It contains no online optimization, no dynamic retraining, and no test-time threshold adjustment.

## Fixed-Point Inputs

- `spread_ticks`: unsigned fixed-point or integer tick count; expected range `[0, 8]` ticks for the tested universe.
- `bid_prob_toxic`, `ask_prob_toxic`: quantized risk scores, recommended `ap_ufixed<16,1>` or equivalent unsigned Q1.15.
- 015 scores or regime bucket inputs only if used by the selected policy.

## Selected Policy

- policy_id: `EV_s2_lam5_thr0`
- policy_type: `ev_score`
- policy_family: `EV_based_quote_selection`

```text
spread_value = 0.5 * spread_ticks * POINT_VALUE
bid_ev = spread_value - LAMBDA_TOXIC * bid_prob_toxic * POINT_VALUE - MAKER_FEE
ask_ev = spread_value - LAMBDA_TOXIC * ask_prob_toxic * POINT_VALUE - MAKER_FEE
wide_spread = spread_ticks > MAX_SPREAD_TICKS
pull_bid = wide_spread OR (bid_ev < EV_THRESHOLD)
pull_ask = wide_spread OR (ask_ev < EV_THRESHOLD)
```

- `MAX_SPREAD_TICKS = 2.0`
- `LAMBDA_TOXIC = 5.0`
- `EV_THRESHOLD = 0.0`

## Output

- `pull_bid`: one-bit cancel/passivate bid quote action.
- `pull_ask`: one-bit cancel/passivate ask quote action.

This specification frames the policy as a latency risk-control candidate. It does not prove FPGA production necessity or real-world profitability.