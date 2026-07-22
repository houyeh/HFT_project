# Label Hierarchy Contract

| Level | Label / metric | Status | Notes |
|---|---|---|---|
| Diagnostic only | `raw_pnl_H` | diagnostic only | Contains mechanical spread capture and must not be the main target. |
| Diagnostic only | toxic label | diagnostic only | Proxy toxicity label, not true aggressor-side exchange toxicity. |
| Main signal target | `excess_pnl_H20` | primary ML signal target | Half-spread-adjusted future-mid markout target. Future label, never a feature. |
| Strategy evaluation | causal-fill sequential excess PnL | strategy proxy metric | Uses causal fill proxy and sequential inventory cap; still not true exchange replay. |

Future labels and PnL columns are never allowed as model features.
