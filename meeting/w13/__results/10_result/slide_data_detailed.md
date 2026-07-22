# Latency Sensitivity — Verified Numbers

Source: `lt=.../eval_outputs/diagnosis/latency_sensitivity_enhanced.csv`

## Cancel Delay (threshold=0.55, prediction_delay=0)
| delay_events | net_pnl | delayed_toxic_loss | delayed_toxic_fills |
|-------------|---------|-------------------|---------------------|
| 0 (baseline) | **84,980** | 0 | 0 |
| 1 | 99,550 | 6,100 | 37 |
| 2 | 92,280 | 7,900 | 50 |
| 3 | 113,330 | 9,700 | 63 |
| 5 | 116,080 | 13,900 | 95 |
| **10** | **61,650** | **20,000** | **143** |

## Prediction Delay (threshold=0.55, cancel_delay=0)
| delay_events | net_pnl |
|-------------|---------|
| 0 | 84,980 |
| 1 | 94,285 |
| 2 | 90,960 |
| 3 | **111,240** |
| 5 | 117,310 |
| 10 | 71,760 |

## Correct Interpretation (NOT monotonic)
- Cancel delay 0→3: net_pnl INCREASES (99K → 113K)
  Reason: delay reduces false-positive cancels, allowing more fills
- Cancel delay 10: net_pnl drops to 61,650 → first clear deterioration
- delayed_toxic_loss increases monotonically (0 → 20,000 TWD)
  → more delay = more toxic fills slip through
- DO NOT claim "PnL collapses after 3 events" — data shows non-monotonic response

## FPGA Relevance
- FPGA target latency: sub-10 event processing
- At delay ≤ 5 events: PnL remains above or near zero-delay baseline
- At delay = 10: first meaningful degradation (84K → 62K)
- Risk-control framing: FPGA allows staying within low-delay regime

## Source
- `lt=.../eval_outputs/diagnosis/latency_sensitivity_enhanced.csv`
