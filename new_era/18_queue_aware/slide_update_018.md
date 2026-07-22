# Slide Update 018

## Slide A: Why queue-aware stress?

We froze the 015 and 016_v2 candidates, then weakened the optimistic proxy execution assumptions. The goal is not to prove profitability; it is to test whether cancel signals still reduce downside versus no-cancel and matched random baselines when queue priority, adverse selection, and cancel timing are less favorable.

## Slide B: Moderate stress result

Conclusion: Partial Pass. Profitability does not survive moderate stress, but the defensive edge does.

- 015_common moderate PnL: `-52,345`, gap vs no-cancel `44,925`, gap vs random_v3_mean `33,224`.
- 016_v2_binary moderate PnL: `-38,490`, gap vs no-cancel `58,780`, gap vs random_v3_mean `47,079`.

Message: absolute PnL fails; relative defensive cancel value survives.

## Slide C: Conservative FPGA framing

Latency matters for risk-control, but FPGA necessity is not proven by this audit. The delay sensitivity result motivates low-latency cancel-risk controls as a research direction; it does not establish production PnL or a hardware requirement.
