from __future__ import annotations

from common_027 import RESULTS, ensure_dirs, load_selected_non_ml_policy, write_markdown


CPP_SOURCE = r'''#include <ap_int.h>

struct QuoteControlInput {
    ap_uint<1> valid;
    ap_uint<1> side_is_bid;
    ap_uint<16> spread_ticks_q0;
};

struct QuoteControlOutput {
    ap_uint<1> quote_bid;
    ap_uint<1> quote_ask;
    ap_uint<1> quote_enable;
};

void bid_only_spread_controller(
    QuoteControlInput in,
    QuoteControlOutput *out
) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE ap_none port=in
#pragma HLS INTERFACE ap_none port=out
#pragma HLS PIPELINE II=1

    const ap_uint<16> MIN_SPREAD_TICKS_Q0 = 1;

    ap_uint<1> pass_valid = in.valid;
    ap_uint<1> pass_side = in.side_is_bid;
    ap_uint<1> pass_spread = (in.spread_ticks_q0 >= MIN_SPREAD_TICKS_Q0);
    ap_uint<1> quote_bid = pass_valid & pass_side & pass_spread;

    out->quote_bid = quote_bid;
    out->quote_ask = 0;
    out->quote_enable = quote_bid;
}
'''


def main() -> None:
    ensure_dirs()
    policy = load_selected_non_ml_policy()
    (RESULTS / "bid_only_spread_controller.cpp").write_text(CPP_SOURCE, encoding="utf-8")
    lines = [
        "# HLS Bid-Only Spread Controller Specification",
        "",
        f"Frozen controller: `{policy['policy_id']}`.",
        "",
        "## Behavioral Rule",
        "",
        "- Quote bid when the input event is valid, the side is bid, and `spread_ticks >= 1`.",
        "- Never quote ask in this conservative core controller.",
        "- No ML score, floating point operation, dynamic threshold, online retraining, or session-dependent state is required.",
        "",
        "## Module I/O",
        "",
        "- `valid`: 1-bit input indicating a valid quote-control decision cycle.",
        "- `side_is_bid`: 1-bit input. `1` means bid side; `0` means ask side.",
        "- `spread_ticks_q0`: unsigned 16-bit integer spread in whole ticks.",
        "- `quote_bid`: 1-bit output enabling a bid quote.",
        "- `quote_ask`: 1-bit output enabling an ask quote; permanently zero for this baseline.",
        "- `quote_enable`: 1-bit aggregate quote enable; equal to `quote_bid`.",
        "",
        "## Pipeline",
        "",
        "- Target initiation interval: `II=1`.",
        "- Data path: register inputs, compare `spread_ticks_q0 >= 1`, AND with `valid` and `side_is_bid`, register outputs.",
        "- Resource profile: one 16-bit unsigned comparator and a small number of 1-bit gates.",
        "",
        "## Fixed-Point Ranges",
        "",
        "- `spread_ticks_q0`: integer tick count, unsigned Q16.0. The selected threshold is exactly `1` tick.",
        "- No price multiplication, PnL arithmetic, or probabilistic score is in the controller core.",
        "",
        "## Research Caveat",
        "",
        "This HLS specification documents a hardware-friendly quote-control primitive. It does not claim FPGA necessity, real-world HFT profitability, true queue priority, or exchange-level fill toxicity.",
    ]
    write_markdown(RESULTS / "hls_bid_only_spread_controller_spec.md", lines)


if __name__ == "__main__":
    main()
    print(f"Wrote HLS spec and C++ controller to {RESULTS}")

