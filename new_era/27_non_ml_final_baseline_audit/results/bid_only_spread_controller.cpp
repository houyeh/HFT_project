#include <ap_int.h>

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
