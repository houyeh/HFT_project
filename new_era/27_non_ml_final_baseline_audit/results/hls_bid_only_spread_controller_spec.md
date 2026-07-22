# HLS Bid-Only Spread Controller Specification

Frozen controller: `bid_only_spread_ge_1`.

## Behavioral Rule

- Quote bid when the input event is valid, the side is bid, and `spread_ticks >= 1`.
- Never quote ask in this conservative core controller.
- No ML score, floating point operation, dynamic threshold, online retraining, or session-dependent state is required.

## Module I/O

- `valid`: 1-bit input indicating a valid quote-control decision cycle.
- `side_is_bid`: 1-bit input. `1` means bid side; `0` means ask side.
- `spread_ticks_q0`: unsigned 16-bit integer spread in whole ticks.
- `quote_bid`: 1-bit output enabling a bid quote.
- `quote_ask`: 1-bit output enabling an ask quote; permanently zero for this baseline.
- `quote_enable`: 1-bit aggregate quote enable; equal to `quote_bid`.

## Pipeline

- Target initiation interval: `II=1`.
- Data path: register inputs, compare `spread_ticks_q0 >= 1`, AND with `valid` and `side_is_bid`, register outputs.
- Resource profile: one 16-bit unsigned comparator and a small number of 1-bit gates.

## Fixed-Point Ranges

- `spread_ticks_q0`: integer tick count, unsigned Q16.0. The selected threshold is exactly `1` tick.
- No price multiplication, PnL arithmetic, or probabilistic score is in the controller core.

## Research Caveat

This HLS specification documents a hardware-friendly quote-control primitive. It does not claim FPGA necessity, real-world HFT profitability, true queue priority, or exchange-level fill toxicity.
