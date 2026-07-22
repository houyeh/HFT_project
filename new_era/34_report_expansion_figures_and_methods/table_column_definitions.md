# Table Column Definitions and Blank Cell Policy

This project intentionally distinguishes missingness from zero-valued metrics.
A blank cell or `NaN` should never be read as a hidden zero.

| Cell representation | Meaning | Typical reason | Interpretation rule |
| --- | --- | --- | --- |
| Empty string / blank cell | Not computed in that artifact | Metric did not belong to the table or the upstream script did not produce it | Do not compare numerically; inspect the table note/source artifact |
| `NaN` | Computation was attempted but undefined | No samples in a subgroup, one-class label vector, or division by zero denominator | Treat as undefined, not as pass/fail by itself |
| `NA` / `not_applicable` | Metric is structurally not applicable | Example: regression metric for a pure classification-only plot | Exclude from aggregate comparisons |
| `source_missing` | Required upstream source artifact or source column was absent | Placeholder reports or compatibility audits | The claim is unsupported until the source is regenerated |
| `0` or `0.0` | Computed numeric zero | Count, PnL, loss, or rate genuinely evaluated to zero | It is a real value and can be compared numerically |

For final advisor-facing tables, any ambiguous blank from older phases should be read alongside the nearest `status`, `note`, `source_note`, or report paragraph.
New Phase 034 follow-up tables use explicit note/status columns where a source was not available or a metric is not applicable.
