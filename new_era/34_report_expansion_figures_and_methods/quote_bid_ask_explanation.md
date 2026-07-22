# Quote Bid/Ask Mechanics

A passive market maker posts limit orders instead of crossing the spread.

- Passive bid: place a buy limit order at or near the best bid. If filled, the strategy buys inventory.
- Passive ask: place a sell limit order at or near the best ask. If filled, the strategy sells inventory.
- Spread capture is the intended compensation for providing liquidity.
- Adverse selection is the risk that the quote is filled because informed flow is moving against it.

Side-specific adverse examples:

- Bid adverse selection: the quote buys, then the mid price falls. The position is immediately underwater.
- Ask adverse selection: the quote sells, then the mid price rises. The strategy has sold too low.

The project uses proxy quote EV and stress-adjusted fill assumptions because true exchange fills and queue position are not available.

Figure: `figures/quote_bid_ask_diagram.png`.
