# Queue-Aware Stress Definition

This stress engine evaluates how candidate cancel policies behave when the optimistic proxy execution assumptions are weakened.

## Stress Dimensions

1. `neutral_fill_haircut`: Uniformly keeps only a fixed fraction of realized maker fills. `1.0` means no queue haircut; `0.5` means each potential fill is kept only when its fixed random mask is below 0.5.

2. `good_fill_haircut`: Preferentially removes favorable fills. A favorable bid fill is one whose future mid return is positive; a favorable ask fill is one whose future mid return is negative. This is intended for pessimistic appendix analysis.

3. `toxic_multiplier`: Multiplies adverse fill damage. At `1.0`, no extra penalty is applied. Above `1.0`, adverse future-mid movement adds an extra penalty proportional to the side-specific adverse ticks.

4. `cancel_delay_ticks`: Delays cancel effectiveness by event ticks. At delay `d`, a cancel signal at time `t` only protects quote events from `t+d` onward.

## Fixed Random Masks

The same random masks are used for every stress setting. This prevents noise from masquerading as stress sensitivity.
