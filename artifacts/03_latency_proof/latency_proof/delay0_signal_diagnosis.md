# Delay-0 Signal Diagnosis

This report checks whether the losing delay=0 trades look more like weak alpha or execution-only failures.

```text
symbol position_side  entry_timestamp  exit_timestamp  pnl_ticks  holding_steps  pred_timestamp  y_true  y_pred  prob_down  prob_stationary  prob_up  gated_signal  gated_directional_prob  gated_margin quote_action quote_side  quote_alpha_score  quote_bid_toxicity  quote_ask_toxicity
 MXFJ5         short   42425221000000  42425221000000       -5.0              0  42425221000000       1       0   0.334422         0.333599 0.331979             1                0.334422      0.000822        place        ask          -0.010216            0.043977            0.000000
 MXFJ5          long   43162078000000  43200413000000       -4.0              3  43162078000000       1       2   0.324605         0.336245 0.339149             1                0.339149      0.002904        place        bid           0.012012            0.006696            0.038333
```

## Interpretation

- Average gated directional probability on the losing trades: `0.336786`
- Average gated margin on the losing trades: `0.001863`
- Both losses passed the gate, but the probabilities were still close to the stationary region rather than decisively directional.
- That suggests the problem is not only execution latency. The selected validation rule is likely leaning on weak or regime-specific alpha.
- In other words: Method B is a real concern. The current latency-proof candidate still appears overfit to narrow MXF validation structure.
