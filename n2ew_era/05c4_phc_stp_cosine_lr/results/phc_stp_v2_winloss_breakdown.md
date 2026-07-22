# PHC STP v2 Win/Loss Magnitude Breakdown

Scope: post-hoc statistics for the frozen `phc_stp_v2_cosine_lr` selected checkpoint and validation-selected threshold.

No training is run. The script reuses `selected_model()`, `stp_decision_for_payload()`, and `decision_pnl_from_payload()` from the existing 05C4/05A code path.

Splits analyzed: `val`, `new3`, `new4`. No other split is loaded.

Definitions:

- `avg_win`: mean selected-action excess over selected rows where excess is strictly positive.
- `avg_loss`: mean selected-action excess over selected rows where excess is strictly negative.
- `avg_loss_abs`: absolute value of `avg_loss`.
- `win_loss_ratio`: `avg_win / avg_loss_abs`.
- zero excess rows are counted separately and excluded from both win and loss counts.

## Breakdown

| split   | selected_action_count   | win_count   | loss_count   | zero_count   | avg_win       | avg_loss       | avg_loss_abs   | win_loss_ratio   | win_rate_check   | negative_rate_check   | avg_selected_excess_check   | identity_lhs   | identity_rhs   | identity_relative_error   |
|:--------|:------------------------|:------------|:-------------|:-------------|:--------------|:---------------|:---------------|:-----------------|:-----------------|:----------------------|:----------------------------|:---------------|:---------------|:--------------------------|
| val     | 64568                   | 30583       | 33982        | 3            | 146.851355328 | -72.1920428462 | 72.1920428462  | 2.03417647622    | 0.473655680833   | 0.526297856523        | 31.5624612811               | 31.5624612811  | 31.5624612811  | 2.251227271e-16           |
| new3    | 141208                  | 71326       | 69880        | 2            | 155.507248409 | -83.7764739554 | 83.7764739554  | 1.85621620327    | 0.505113024758   | 0.494872811739        | 37.0900373916               | 37.0900373916  | 37.0900373916  | 3.83144793443e-16         |
| new4    | 54183                   | 26850       | 27331        | 2            | 157.081936685 | -85.8353152098 | 85.8353152098  | 1.83003856048    | 0.495542882454   | 0.5044202056          | 34.5437683406               | 34.5437683406  | 34.5437683406  | 2.05693463653e-16         |

## Sanity Checks

The `win_rate_check` value is compared against the existing `results/phc_stp_v2_oos_eval.csv` `win_rate` column. Exact equality is required; otherwise the script stops.

The arithmetic identity checked is:

`win_rate_check * avg_win + negative_rate_check * avg_loss ~= avg_selected_excess_check`

| split   | win_rate_check      | existing_win_rate   | computed_win_rate_text   | existing_win_rate_text   | win_rate_exact_match   | identity_lhs       | identity_rhs       | identity_relative_error   | identity_pass   |
|:--------|:--------------------|:--------------------|:-------------------------|:-------------------------|:-----------------------|:-------------------|:-------------------|:--------------------------|:----------------|
| val     | 0.47365568083261056 | 0.47365568083261056 | 0.47365568083261056      | 0.47365568083261056      | True                   | 31.562461281129977 | 31.56246128112997  | 2.2512272710015402e-16    | True            |
| new3    | 0.50511302475780406 | 0.50511302475780406 | 0.5051130247578041       | 0.5051130247578041       | True                   | 37.090037391649183 | 37.090037391649197 | 3.8314479344260706e-16    | True            |
| new4    | 0.49554288245390621 | 0.49554288245390621 | 0.4955428824539062       | 0.4955428824539062       | True                   | 34.543768340623451 | 34.543768340623444 | 2.0569346365274872e-16    | True            |

## Interpretation

Observed `win_loss_ratio` range: `1.830039` to `2.034176`.

All three splits have `win_loss_ratio > 1`, so the average winning selected action is larger than the absolute average losing selected action. This is direct evidence of asymmetric payoff magnitude, explaining how the selected policy can remain positive even when the selected-action win rate is near or slightly below 50%.

This is a descriptive decomposition of already-selected actions. It does not change the model, threshold, or policy.
