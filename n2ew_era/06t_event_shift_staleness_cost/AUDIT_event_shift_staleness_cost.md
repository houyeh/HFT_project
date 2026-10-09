# AUDIT: Event-Shift Staleness Cost

status: `resolved_with_hypothetical_scale_estimate`

## Scope Boundary

- Splits used: `val`, `NEW3`, `NEW4` only.
- NEW5 was not loaded or touched.
- No model was trained, no checkpoint was changed, and no threshold was selected or tuned.
- Frozen PHC/HGB decisions were reconstructed by loading existing frozen model artifacts because no event-level decision sequence artifact is persisted in the existing phases; the k=0 hard gate below proves the reconstruction path matches frozen summaries exactly.
- Artifact compatibility note: loading the frozen HGB joblib under the current `.venv` emitted sklearn `InconsistentVersionWarning` because the model was pickled with sklearn 1.7.2 and loaded with sklearn 1.8.0; the HGB k=0 exact-match gate across val/NEW3/NEW4 is the compatibility check for this diagnostic.
- No fill model, execution simulation, or queue-position model is used. This is label-side paired diagnostics only.
- 本結果不是因果測量;報告中的合成數字只能稱為「假設下的量級估計」。

## Definition

- Book change event: an event whose `best_bid` or `best_ask` differs from the immediately previous dense action-universe event in the same session.
- k shift: for a selected decision anchored at event `t`, use the event index of the kth future top-of-book price change strictly after `t`; `k=0` is the original event.
- V1 repriced label: query the frozen same-side `excess_pnl_H20` label at the shifted event. This measures signal freshness decay under repricing and is a lower-bound staleness proxy.
- V2 original-quote markout: keep the original bid/ask quote price at `t`, compare it with the shifted event's `future_mid_H20`, and report paired markout deltas as a coarse adverse-selection-direction upper proxy. It does not pass through a fill model.

## Baseline Consistency Gate

The hard gate requires k=0 total `excess_pnl_H20` and selected counts to match the frozen 05c4 summary exactly.

| split | policy_id | reference_total_excess_pnl_H20 | recomputed_k0_total_excess_pnl_H20 | pnl_abs_diff | reference_selected_action_count | recomputed_selected_action_count | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | side_by_queue_imbalance_abs_0p1 | 1237145.0 | 1237145.0 | 0.0 | 68981 | 68981 | True |
| val | ml_hgb_regressor_shallow_convergence_selected | 1671675.0 | 1671675.0 | 0.0 | 61679 | 61679 | True |
| val | phc_stp_v2_cosine_lr | 2037925.0 | 2037925.0 | 0.0 | 64568 | 64568 | True |
| new3 | side_by_queue_imbalance_abs_0p1 | 2954770.0 | 2954770.0 | 0.0 | 144745 | 144745 | True |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 4010580.0 | 4010580.0 | 0.0 | 134210 | 134210 | True |
| new3 | phc_stp_v2_cosine_lr | 5237410.0 | 5237410.0 | 0.0 | 141208 | 141208 | True |
| new4 | side_by_queue_imbalance_abs_0p1 | 1003910.0 | 1003910.0 | 0.0 | 56147 | 56147 | True |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 1395420.0 | 1395420.0 | 0.0 | 51753 | 51753 | True |
| new4 | phc_stp_v2_cosine_lr | 1871685.0 | 1871685.0 | 0.0 | 54183 | 54183 | True |

All k=0 gates passed, so C(k) diagnostics are reported.

## C(k) V1 Repriced-Label Results

Negative `mean_delta` means the shifted same-side label is worse than the original decision label. `Ck_cost_pnl_per_decision = -mean_delta`; therefore positive cost means degradation.

| split | policy_id | k_book_changes_after_t | paired_valid_count | excluded | mean_delta | median_delta_pnl_per_decision | neg_rate | ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | side_by_queue_imbalance_abs_0p1 | 0 | 68981 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| val | side_by_queue_imbalance_abs_0p1 | 1 | 68977 | 4 | -24.877858 | -100.0 | 51.154733% | [-25.924765, -23.852625] |
| val | side_by_queue_imbalance_abs_0p1 | 2 | 68977 | 4 | -30.505386 | 0.0 | 44.058164% | [-31.885556, -29.165773] |
| val | side_by_queue_imbalance_abs_0p1 | 5 | 68971 | 10 | -33.317916 | 0.0 | 46.589146% | [-34.860537, -31.834508] |
| val | ml_hgb_regressor_shallow_convergence_selected | 0 | 61679 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| val | ml_hgb_regressor_shallow_convergence_selected | 1 | 61675 | 4 | -31.093636 | -100.0 | 54.138630% | [-32.155695, -30.068788] |
| val | ml_hgb_regressor_shallow_convergence_selected | 2 | 61675 | 4 | -38.052452 | 0.0 | 46.111066% | [-39.424686, -36.524767] |
| val | ml_hgb_regressor_shallow_convergence_selected | 5 | 61670 | 9 | -41.210799 | 0.0 | 48.517918% | [-42.749919, -39.586880] |
| val | phc_stp_v2_cosine_lr | 0 | 64568 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| val | phc_stp_v2_cosine_lr | 1 | 64565 | 3 | -30.366298 | -100.0 | 54.156277% | [-31.419577, -29.269573] |
| val | phc_stp_v2_cosine_lr | 2 | 64565 | 3 | -39.936266 | 0.0 | 46.791605% | [-41.459072, -38.658329] |
| val | phc_stp_v2_cosine_lr | 5 | 64560 | 8 | -44.085657 | 0.0 | 49.045849% | [-45.602573, -42.575792] |
| new3 | side_by_queue_imbalance_abs_0p1 | 0 | 144745 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| new3 | side_by_queue_imbalance_abs_0p1 | 1 | 144744 | 1 | -23.843372 | -100.0 | 50.636296% | [-24.522637, -23.162152] |
| new3 | side_by_queue_imbalance_abs_0p1 | 2 | 144743 | 2 | -32.145838 | 0.0 | 45.947645% | [-33.000617, -31.153182] |
| new3 | side_by_queue_imbalance_abs_0p1 | 5 | 144742 | 3 | -35.432977 | 0.0 | 47.435437% | [-36.523990, -34.385542] |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 0 | 134210 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 1 | 134210 | 0 | -29.306199 | -100.0 | 53.042992% | [-29.948880, -28.537013] |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 2 | 134209 | 1 | -39.069548 | 0.0 | 47.747916% | [-40.065141, -38.083133] |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 5 | 134208 | 2 | -43.952857 | 0.0 | 49.427754% | [-45.069591, -42.945367] |
| new3 | phc_stp_v2_cosine_lr | 0 | 141208 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| new3 | phc_stp_v2_cosine_lr | 1 | 141207 | 1 | -30.678260 | -100.0 | 54.046896% | [-31.386147, -30.007433] |
| new3 | phc_stp_v2_cosine_lr | 2 | 141206 | 2 | -42.355034 | 0.0 | 48.587171% | [-43.312412, -41.451195] |
| new3 | phc_stp_v2_cosine_lr | 5 | 141206 | 2 | -49.312529 | -100.0 | 50.694730% | [-50.293022, -48.229533] |
| new4 | side_by_queue_imbalance_abs_0p1 | 0 | 56147 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| new4 | side_by_queue_imbalance_abs_0p1 | 1 | 56146 | 1 | -22.686745 | 0.0 | 49.994657% | [-23.843722, -21.434163] |
| new4 | side_by_queue_imbalance_abs_0p1 | 2 | 56146 | 1 | -30.220408 | 0.0 | 45.219606% | [-31.809661, -28.750655] |
| new4 | side_by_queue_imbalance_abs_0p1 | 5 | 56142 | 5 | -32.377543 | 0.0 | 46.765345% | [-34.109058, -30.663318] |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 0 | 51753 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 1 | 51752 | 1 | -27.706562 | -100.0 | 52.301360% | [-28.935176, -26.552541] |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 2 | 51752 | 1 | -37.143589 | 0.0 | 46.970165% | [-38.800508, -35.534769] |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 5 | 51748 | 5 | -40.748241 | 0.0 | 48.548736% | [-42.408547, -38.830653] |
| new4 | phc_stp_v2_cosine_lr | 0 | 54183 | 0 | 0.000000 | 0.0 | 0.000000% | [0.000000, 0.000000] |
| new4 | phc_stp_v2_cosine_lr | 1 | 54181 | 2 | -28.810284 | -100.0 | 53.249294% | [-30.004831, -27.594207] |
| new4 | phc_stp_v2_cosine_lr | 2 | 54181 | 2 | -40.683173 | 0.0 | 47.856260% | [-42.232286, -39.182792] |
| new4 | phc_stp_v2_cosine_lr | 5 | 54177 | 6 | -46.987467 | -100.0 | 50.179966% | [-48.786464, -45.018273] |

Decay curve figure: `figures/staleness_decay_curve.png`.

## V2 Original-Quote Markout Extension

V2 is reported in the same CSV as diagnostic columns prefixed with `v2_original_quote_markout_`. It is a coarse markout sensitivity check, not a fill/PnL simulation.

| split | policy_id | k_book_changes_after_t | v2_original_quote_markout_valid_count | v2_mean | v2_neg_rate |
| --- | --- | --- | --- | --- | --- |
| val | side_by_queue_imbalance_abs_0p1 | 1 | 68976 | -1.000348 | 23.418290% |
| val | side_by_queue_imbalance_abs_0p1 | 2 | 68974 | -0.417549 | 32.987793% |
| val | side_by_queue_imbalance_abs_0p1 | 5 | 68969 | -2.148791 | 41.412809% |
| val | ml_hgb_regressor_shallow_convergence_selected | 1 | 61674 | 0.963129 | 22.479489% |
| val | ml_hgb_regressor_shallow_convergence_selected | 2 | 61673 | 2.461369 | 32.252363% |
| val | ml_hgb_regressor_shallow_convergence_selected | 5 | 61668 | 3.077771 | 40.802361% |
| val | phc_stp_v2_cosine_lr | 1 | 64564 | 4.023914 | 21.792330% |
| val | phc_stp_v2_cosine_lr | 2 | 64563 | 7.059771 | 31.502563% |
| val | phc_stp_v2_cosine_lr | 5 | 64558 | 12.038787 | 39.833948% |
| new3 | side_by_queue_imbalance_abs_0p1 | 1 | 144741 | -0.816631 | 23.925495% |
| new3 | side_by_queue_imbalance_abs_0p1 | 2 | 144741 | -0.310900 | 34.533408% |
| new3 | side_by_queue_imbalance_abs_0p1 | 5 | 144738 | 0.297779 | 42.694386% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 1 | 134207 | 1.207091 | 22.990604% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 2 | 134207 | 3.207731 | 33.646531% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 5 | 134203 | 6.185406 | 41.999806% |
| new3 | phc_stp_v2_cosine_lr | 1 | 141205 | 5.079848 | 21.863957% |
| new3 | phc_stp_v2_cosine_lr | 2 | 141205 | 9.338904 | 32.312595% |
| new3 | phc_stp_v2_cosine_lr | 5 | 141201 | 16.541668 | 40.833280% |
| new4 | side_by_queue_imbalance_abs_0p1 | 1 | 56144 | -1.547806 | 24.326731% |
| new4 | side_by_queue_imbalance_abs_0p1 | 2 | 56143 | -1.334093 | 34.641897% |
| new4 | side_by_queue_imbalance_abs_0p1 | 5 | 56138 | -0.915601 | 42.799886% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 1 | 51750 | 0.498551 | 23.443478% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 2 | 51749 | 1.623220 | 33.861524% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 5 | 51744 | 4.315476 | 42.215523% |
| new4 | phc_stp_v2_cosine_lr | 1 | 54179 | 4.784141 | 22.137729% |
| new4 | phc_stp_v2_cosine_lr | 2 | 54178 | 8.486840 | 32.276939% |
| new4 | phc_stp_v2_cosine_lr | 5 | 54173 | 15.234526 | 40.896757% |

## Synthesis Estimate

Borrowed 06R stale-exposure delta: `6.611245%` from `n2ew_era\06r_lobster_us_latency_budget\results\stale_exposure_cpu_minus_fpga.csv`. This is the LOBSTER pilot `stale_prob(636.7us) - stale_prob(83.16us)` for the five-stock 2012 NASDAQ sample, not TXF.

Formula: `DeltaPnL_daily ~= DeltaStale_06R * C(k) * average_selected_decisions_per_session`, with signed `C(k)=mean_delta_pnl_per_decision`. k=1 and k=2 form the requested range.

| split | policy_id | k_used_as_C | Ck_mean_delta_pnl_per_decision | avg_selected_decisions_per_session | estimated_daily_delta_pnl | estimated_split_delta_pnl | estimated_split_delta_pct_of_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | side_by_queue_imbalance_abs_0p1 | 1 | -24.877858 | 11496.833333333334 | -18909.257633 | -113455.545798 | -9.170756% |
| val | side_by_queue_imbalance_abs_0p1 | 2 | -30.505386 | 11496.833333333334 | -23186.650719 | -139119.904314 | -11.245238% |
| val | ml_hgb_regressor_shallow_convergence_selected | 1 | -31.093636 | 10279.833333333334 | -21132.011839 | -126792.071037 | -7.584732% |
| val | ml_hgb_regressor_shallow_convergence_selected | 2 | -38.052452 | 10279.833333333334 | -25861.397302 | -155168.383811 | -9.282210% |
| val | phc_stp_v2_cosine_lr | 1 | -30.366298 | 10761.333333333334 | -21604.349212 | -129626.095273 | -6.360690% |
| val | phc_stp_v2_cosine_lr | 2 | -39.936266 | 10761.333333333334 | -28412.980913 | -170477.885479 | -8.365268% |
| new3 | side_by_queue_imbalance_abs_0p1 | 1 | -23.843372 | 24124.166666666668 | -38027.979572 | -228167.877434 | -7.722018% |
| new3 | side_by_queue_imbalance_abs_0p1 | 2 | -32.145838 | 24124.166666666668 | -51269.647432 | -307617.884594 | -10.410891% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 1 | -29.306199 | 22368.333333333332 | -43338.750456 | -260032.502736 | -6.483663% |
| new3 | ml_hgb_regressor_shallow_convergence_selected | 2 | -39.069548 | 22368.333333333332 | -57777.038497 | -346662.230981 | -8.643693% |
| new3 | phc_stp_v2_cosine_lr | 1 | -30.678260 | 23534.666666666668 | -47733.362771 | -286400.176627 | -5.468355% |
| new3 | phc_stp_v2_cosine_lr | 2 | -42.355034 | 23534.666666666668 | -65901.659819 | -395409.958913 | -7.549723% |
| new4 | side_by_queue_imbalance_abs_0p1 | 1 | -22.686745 | 14036.75 | -21053.389321 | -84213.557283 | -8.388556% |
| new4 | side_by_queue_imbalance_abs_0p1 | 2 | -30.220408 | 14036.75 | -28044.657667 | -112178.630669 | -11.174172% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 1 | -27.706562 | 12938.25 | -23699.623162 | -94798.492648 | -6.793546% |
| new4 | ml_hgb_regressor_shallow_convergence_selected | 2 | -37.143589 | 12938.25 | -31771.861550 | -127087.446201 | -9.107469% |
| new4 | phc_stp_v2_cosine_lr | 1 | -28.810284 | 13545.75 | -25800.840732 | -103203.362927 | -5.513928% |
| new4 | phc_stp_v2_cosine_lr | 2 | -40.683173 | 13545.75 | -36433.520303 | -145734.081212 | -7.786250% |

Required assumptions:

- (a) stale 深度以 1–2 個事件為主;
- (b) 暴露率借用 2012 NASDAQ pilot,非 TXF 實測;
- (c) fill 處理兩側對稱、未含佇列位置效應。

## Claim Boundary

- This is an event-index paired-label diagnostic and a hypothetical scale estimate, not a causal latency-to-PnL measurement.
- 06R is single-day, five-stock, 2012 NASDAQ data; its stale-exposure delta is borrowed only as a scale reference and must not be treated as measured TXF exposure.
- The experiment intentionally excludes the known-bad 06Q latency fill proxy and makes no claim about execution, queue replay, or realized PnL.

## Outputs

- `results/staleness_cost_by_k.csv`
- `results/synthesis_estimate.csv`
- `results/baseline_consistency_check.csv`
- `figures/staleness_decay_curve.png`

## Honest Conclusion

At k=1, `9` split-policy cells show negative V1 decay and `0` show positive V1 deltas. The sign is therefore an empirical diagnostic result, not forced into a degradation story. The synthesis table should be read as a signed hypothetical scale estimate under the listed assumptions.


### 3.5 微秒級延遲暴露分析（LOBSTER pilot）

實驗設計：TXF 資料時戳粒度 1ms（06P 已驗證），無法量測 P(Δt < 83.16µs)。本節改用
LOBSTER 官方免費樣本（AAPL/MSFT/INTC/AMZN/GOOG，2012-06-21，09:30–16:00，
奈秒級時戳，整數奈秒解析、無浮點精度損失）直接量測兩個量：
(1) 事件間隔分布——相鄰 L2 事件的時間差；(2) stale-exposure——任一事件後 L µs 內
頂檔最佳價已再變動的機率（決策以延遲 L 抵達時 book 已過期的機率）。

| 量測 | 全事件流 | 尖峰窗（事件數前 10% 秒） |
|---|---|---|
| P(Δt < 83.16µs)（任意 L2 更新） | 43.42% | 57.65% |
| P(Δt < 636.7µs) | 63.18% | 76.89% |
| 最小非零間隔 | 0.169µs | — |

| stale-exposure | @83.16µs（FPGA 固定） | @636.7µs（CPU p99） | 差值 |
|---|---|---|---|
| 五檔合併 | 3.71% | 10.32% | +6.61pp |
| 區間（逐檔） | 2.79–5.77% | 8.25–14.52% | +5.46–9.79pp |

正確解讀：FPGA 的暴露是**已知且有界**的常數；CPU 的暴露是隨延遲分布變動的隨機變數，
尾端情境升至 10.3%。此對比的主張是「可預算 vs 不可預算」，非「平均更低」（CPU 平均延遲
74.54µs 低於 83.16µs，其平均暴露與 FPGA 相當）。逐檔差異反映 tick-size regime：MSFT/INTC
（large-tick，頂檔價格變動稀疏）較接近台指期性質，其 stale 差值 +5.5–6.5pp 為對 TXF 的
較保守參考。

![事件間隔 CDF](n2ew_era/06r_lobster_us_latency_budget/figures/lobster_interarrival_cdf.png)
![stale-exposure 對延遲之 CDF](n2ew_era/06r_lobster_us_latency_budget/figures/lobster_stale_exposure_cdf.png)

邊界聲明：單日、五檔、2012 年 NASDAQ 樣本，市場/資產/年代皆與 TXF 不同，本結果為
方法論驗證與量級參考，不得推論 TXF 結論；stale-exposure 是 book 變動機率、非 PnL 影響，
兩者之間隔著成交與逆選擇機制（條件成本估計見 06T，進行中）。


**逐檔 stale-exposure**（支撐上述區間與 tick-size regime 論點）：

| ticker | @83.16µs | @636.7µs | 差值 | 頂檔價變次數/日 | regime |
|---|---|---|---|---|---|
| AAPL | 5.77% | 13.70% | +7.94pp | 64,350 | small-tick |
| GOOG | 4.73% | 14.52% | +9.79pp | 24,084 | small-tick |
| AMZN | 3.30% | 9.05% | +5.75pp | 27,557 | 中間 |
| MSFT | 3.28% | 9.82% | +6.54pp | 4,060 | large-tick |
| INTC | 2.79% | 8.25% | +5.46pp | 3,217 | large-tick |

![事件間隔 CDF](n2ew_era/06r_lobster_us_latency_budget/figures/lobster_interarrival_cdf.png)
![stale-exposure 對延遲之 CDF](n2ew_era/06r_lobster_us_latency_budget/figures/lobster_stale_exposure_cdf.png)

邊界聲明：單日、五檔、2012 年 NASDAQ 樣本，為方法論驗證與量級參考，不得直接推論
TXF；stale-exposure 為 book 變動機率、非 PnL 影響（條件成本見 06T，進行中）。