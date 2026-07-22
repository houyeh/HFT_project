# CPU vs FPGA Latency Plan

Latency should be decomposed into separate components:

- Compute latency: time to evaluate the controller once market data is available.
- I/O latency: time to receive data and transmit orders.
- Exchange/network latency: venue path, gateway, acknowledgement, and matching-engine timing.
- Jitter: variability in latency, often more important than average latency for risk-control systems.

Conservative FPGA claim:

- FPGA implementation can reduce deterministic compute jitter for simple comparator logic.
- This supports a latency risk-control framing.
- It does not prove FPGA improves PnL or is necessary for production profitability.

Figure: `figures/latency_path_diagram.png`.
