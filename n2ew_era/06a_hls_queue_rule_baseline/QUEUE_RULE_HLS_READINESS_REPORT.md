# Queue Rule HLS Readiness Report

Logic: `imbalance = (bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1)`. Bid if `> 0.1`; ask if `< -0.1`; otherwise no quote.

The HLS implementation avoids division using integer comparisons: `10*(bid-ask) > bid+ask` and `10*(ask-bid) > bid+ask`.

This phase validates functional equivalence and HLS feasibility only. It does not claim FPGA improves PnL.

## Action Agreement

| comparison              |   row_count |   agreement_rate |   mismatches | cpp_compiled   | cpp_ran   |
|:------------------------|------------:|-----------------:|-------------:|:---------------|:----------|
| python_vs_cpp_reference |      405528 |                1 |            0 | True           | True      |

## HLS Tool Availability

| tool       | path   | available   |
|:-----------|:-------|:------------|
| vitis_hls  |        | False       |
| vivado_hls |        | False       |

If Vitis/Vivado HLS is unavailable, vendor synthesis/resource reports are not fabricated. The included C++ and testbench are ready for synthesis on a PYNQ-Z2 / Zynq-7020 toolchain.

## Static Latency / Resource Estimate

| target_device       | kernel            |   expected_ii | pipeline_pragmas_present   | division_used   | operators                                                 | resource_estimate_type                           | expected_resource_pressure   | fpga_pnl_claim   |
|:--------------------|:------------------|--------------:|:---------------------------|:----------------|:----------------------------------------------------------|:-------------------------------------------------|:-----------------------------|:-----------------|
| PYNQ-Z2 / Zynq-7020 | queue_rule_action |             1 | True                       | False           | 2 add/sub, 2 multiply-by-constant comparisons, branch mux | static_feasibility_estimate_not_vendor_synthesis | very_low                     | forbidden        |

Forbidden claims: real-world profitability, true exchange replay, and FPGA PnL improvement.
