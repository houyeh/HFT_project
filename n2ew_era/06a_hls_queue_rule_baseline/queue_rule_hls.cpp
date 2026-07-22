#include "queue_rule_hls.hpp"

int8_t queue_rule_action(int32_t bid_sz_1, int32_t ask_sz_1) {
#pragma HLS INLINE
#pragma HLS PIPELINE II=1
    if (bid_sz_1 <= 0 || ask_sz_1 <= 0) {
        return ACTION_NO_QUOTE;
    }

    const int64_t bid = static_cast<int64_t>(bid_sz_1);
    const int64_t ask = static_cast<int64_t>(ask_sz_1);
    const int64_t denom = bid + ask;
    const int64_t diff = bid - ask;

    if (10 * diff > denom) {
        return ACTION_BID;
    }
    if (10 * (-diff) > denom) {
        return ACTION_ASK;
    }
    return ACTION_NO_QUOTE;
}
