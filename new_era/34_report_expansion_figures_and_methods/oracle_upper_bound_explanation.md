# Oracle Upper Bound Explanation

Oracle policies use future information and are not tradable. They answer a feasibility question: is positive moderate-stress PnL mathematically available in the proxy environment?

Two relevant oracle concepts:

- Perfect toxic cancel oracle: blocks fills known to be toxic using future information.
- Perfect quote PnL filter oracle: keeps only quotes whose realized stress-adjusted proxy EV is positive.

A large gap between the oracle and causal policies means opportunity exists in the proxy data, but current causal rules or ML features do not capture it reliably.
