# Rule Ablation Interpretation

Aggregate `Bid spread>=1` proxy PnL across available groups: `46,497,800`.
Aggregate `Bid all` proxy PnL across available groups: `46,497,800`.
Aggregate `Ask all` proxy PnL across available groups: `-50,540,870`.

Because every quote-eligible bid action has spread >= 1, `Bid spread>=1` and `Bid all` are equivalent in this universe. The safe claim is bid-side regime filtering, not a precise spread-threshold alpha.
