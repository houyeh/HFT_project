# Quote-Eligible Universe Definition

Rows are retained only when all rules pass in order:

1. Best bid and ask are finite and strictly positive.
2. Spread is strictly positive.
3. All five bid and ask price/volume levels are finite and strictly positive.
4. The row is not in the first 100 ticks or last 10 ticks of its session.
5. The selected model features and past-only z-scores are finite.

Z-scores use only prior rows from the same session: `shift(1).rolling(...)` supplies the mean and standard deviation.