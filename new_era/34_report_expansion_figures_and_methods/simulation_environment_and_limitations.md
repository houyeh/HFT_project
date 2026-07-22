# Simulation Environment and Limitations

| Area | Considered | Omitted |
| --- | --- | --- |
| Market data | Best bid/ask, spread, L2-derived features for ML diagnostics | Full order-by-order queue reconstruction |
| Execution | Passive quote proxy and stress-adjusted quote EV | Real exchange acknowledgements and cancellations |
| Queue | Haircuts and delay stress proxies | True queue priority and order ID matching |
| Labels | Simulator-derived quote EV and toxicity proxies | Ground-truth aggressor-side exchange fills |
| Hardware | HLS-friendly comparator logic and latency framing | Proof that FPGA deployment improves PnL |

All profitability statements must remain proxy statements.

Figure: `figures/simulation_environment_table.png`.
