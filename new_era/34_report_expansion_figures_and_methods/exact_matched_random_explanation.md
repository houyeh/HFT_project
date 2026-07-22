# Exact Matched Random Explanation

The Phase 032 exact matched random quote baseline produced p-value `1.0` and zero variance. That is not evidence against or for the controller; it is a saturation diagnostic.

Why it saturates:

- The controller selects `side == bid` and `spread_ticks >= 1`.
- In the quote-eligible action universe, every bid action has `spread_ticks >= 1`.
- Therefore the policy selects the entire exact candidate universe.
- A random selector constrained to the same exact universe and the same session/spread-bucket counts has no alternative quotes left to sample.

Correct wording: the exact matched random test is non-informative for quote-selection skill.

Forbidden wording: the baseline beat exact matched random quote selection.
