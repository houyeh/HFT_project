# AUDIT: Event Inter-arrival Latency Budget

Status: `resolution_limited_microsecond_probabilities_not_measurable`.

## Timestamp Resolution Statement

The verified minimum non-zero adjacent timestamp difference across NEW1-NEW4 TXF-only sessions is `1.000ms`.

The source timestamp column is `timestamp_us`, derived from `info_time` parsed as `hour:minute:second:millisecond:flag`; Phase 04A.1 found the fifth segment always zero and interpreted the fourth segment as milliseconds. Night-session time-of-day values are unwrapped with a session-start anchor, then events are ordered by unwrapped timestamp with `event_index` as a tie-break. This avoids treating raw chunk-order overlaps as real negative time intervals.

Because the effective timestamp granularity is 1ms, the requested probabilities `P(delta_t < 23.16us)` and `P(delta_t < 83.16us)` are **not directly measurable** from this dataset. Any sub-millisecond event ordering inside a single timestamp batch is collapsed into `delta_t = 0`, and any event that crosses a millisecond boundary may have true spacing below 1ms but observed spacing of 1ms. The report therefore presents `P(delta_t < 1ms timestamp tick)` as the measurable resolution-limited substitute, which is the same-timestamp adjacent-event share.

## Data Scope

- Source dataset: `n2ew_era/04b_txf_only_final_development_rerun/data/txf_only_action_universe.parquet/`.
- Splits used: `train` = NEW2, `val` = NEW1, `new3` = NEW3, `new4` = NEW4.
- NEW5: not read, not referenced for selection, and not used.
- Event stream: one row per `(session_id, event_index)`, taken from `action_slot == 0` to remove bid/ask/no_quote action-row duplication.
- Quote-eligible subset: finite top-of-book bid/ask, positive top sizes, positive spread. Since Phase 04B already builds a quote-eligible TXF action universe, this subset is effectively the same state universe; the exact rates are shown below.
- Peak slice: per session, 1-second windows are ranked by event count; intervals whose later event falls in the top 10% windows are retained.
- Calculation script: `n2ew_era/06p_event_interarrival_latency_budget/00_event_interarrival_latency_budget.py`.

## Aggregate NEW1-NEW4 Results

| scope                              |   interval_count | same_timestamp_rate   | min_nonzero_dt_us   | p50_dt_us   | p90_dt_us   | p99_dt_us   | p999_dt_us   | p_lt_23p16us                       | p_lt_83p16us                       | p_lt_1ms_observed   | p_lt_10ms_observed   |
|:-----------------------------------|-----------------:|:----------------------|:--------------------|:------------|:------------|:------------|:-------------|:-----------------------------------|:-----------------------------------|:--------------------|:---------------------|
| all_event_flow                     |         12979011 | 67.874085%            | 1.000ms             | 0.000us     | 34.000ms    | 414.000ms   | 1.442s       | N/A below 1ms timestamp resolution | N/A below 1ms timestamp resolution | 67.874085%          | 85.002255%           |
| peak10_all_event_flow              |          4753764 | 71.999935%            | 1.000ms             | 0.000us     | 9.000ms     | 109.000ms   | 340.000ms    | N/A below 1ms timestamp resolution | N/A below 1ms timestamp resolution | 71.999935%          | 90.146166%           |
| peak10_quote_eligible_state_subset |          4753764 | 71.999935%            | 1.000ms             | 0.000us     | 9.000ms     | 109.000ms   | 340.000ms    | N/A below 1ms timestamp resolution | N/A below 1ms timestamp resolution | 71.999935%          | 90.146166%           |
| quote_eligible_state_subset        |         12979011 | 67.874085%            | 1.000ms             | 0.000us     | 34.000ms    | 414.000ms   | 1.442s       | N/A below 1ms timestamp resolution | N/A below 1ms timestamp resolution | 67.874085%          | 85.002255%           |

## Split-Level All-Event Results

| split   | source_group   |   interval_count | same_timestamp_rate   | min_nonzero_dt_us   | p50_dt_us   | p90_dt_us   | p99_dt_us   | p999_dt_us   | p_lt_1ms_observed   | p_lt_10ms_observed   |
|:--------|:---------------|-----------------:|:----------------------|:--------------------|:------------|:------------|:------------|:-------------|:--------------------|:---------------------|
| new3    | NEW3           |          5656816 | 69.452126%            | 1.000ms             | 0.000us     | 27.000ms    | 359.000ms   | 1.044s       | 69.452126%          | 86.082895%           |
| new4    | NEW4           |          2196276 | 69.350801%            | 1.000ms             | 0.000us     | 27.000ms    | 324.000ms   | 966.000ms    | 69.350801%          | 86.081485%           |
| train   | NEW2           |          2470556 | 65.087333%            | 1.000ms             | 0.000us     | 41.000ms    | 486.000ms   | 1.691s       | 65.087333%          | 83.427455%           |
| val     | NEW1           |          2655363 | 65.883723%            | 1.000ms             | 0.000us     | 46.000ms    | 579.000ms   | 2.162s       | 65.883723%          | 83.272682%           |

## Quote-Eligible Coverage

| split   |   state_count |   quote_eligible_state_count | quote_eligible_state_rate   |
|:--------|--------------:|-----------------------------:|:----------------------------|
| new3    |       5656822 |                      5656822 | 100.000000%                 |
| new4    |       2196280 |                      2196280 | 100.000000%                 |
| train   |       2470559 |                      2470559 | 100.000000%                 |
| val     |       2655369 |                      2655369 | 100.000000%                 |

## Interpretation

- The dataset can support millisecond-level latency-budget claims, not direct 23.16us or 83.16us opportunity-loss probabilities.
- The measurable sub-1ms proxy is the same-timestamp adjacent-event rate. This is not a true microsecond probability; it is a resolution-limited batch indicator.
- In this audit the same-timestamp adjacent-event share is `67.874085%` overall and `71.999935%` in peak 10% event-rate windows. Because same-timestamp batches dominate, this dataset cannot establish that 83.16us or 23.16us misses only a small number of true microstructure opportunities.
- Among timestamp-resolved nonzero moves, the minimum observed adjacent difference is `1.000ms`; observed p90 is `34.000ms` overall and `9.000ms` in peak windows.
- Safe claim: under the available millisecond timestamp feed, sub-millisecond latency loss is resolution-limited and unquantified. Do not present the 23.16us/83.16us thresholds as directly measured opportunity-loss probabilities.
- A fresh data feed with exchange or receive timestamps finer than 83us would be required to make the stronger claim `P(true delta_t < 83.16us)` directly.

## Output Files

- Summary CSV: `results/event_interarrival_latency_budget_summary.csv`
- Session CSV: `results/event_interarrival_session_summary.csv`
- Timestamp resolution CSV: `results/timestamp_resolution_audit.csv`
- CDF figure: `figures/event_interarrival_cdf.png`
