# Matched Random Saturation Report

The Phase 032 exact matched random quote test is non-informative for quote-selection skill.

Reason: the tested policy is `bid_only_spread_ge_1`, and in the quote-eligible action universe every bid action has `spread_ticks >= 1`. Therefore the policy selects the entire exact universe `side=bid and spread>=1`.

When the random baseline is constrained to the exact same candidate universe and must match the same session/spread-bucket counts, there are no alternative quotes left to sample. The random baseline is forced to select the same set as the policy, producing zero variance and empirical p-value `1.0`.

**Correct interpretation:** this exact matched random result is a saturation diagnostic, not evidence that the controller beats matched random quote selection.

**Forbidden wording:** do not claim that `bid_only_spread_ge_1` beats the exact-matched random quote baseline.

## Phase 032 Saturation Evidence

- `new4`: selected `714`, exact candidate universe `714`, random std `0.000000`, p-value `1.000`.
- `test`: selected `419`, exact candidate universe `419`, random std `0.000000`, p-value `1.000`.
