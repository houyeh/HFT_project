#pragma once

#include <cstdint>

enum QuoteAction : int8_t {
    ACTION_NO_QUOTE = 0,
    ACTION_BID = 1,
    ACTION_ASK = 2
};

// HLS-friendly queue imbalance rule:
// imbalance = (bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1)
// bid if imbalance > 0.1, ask if imbalance < -0.1.
//
// The comparator avoids division:
//   10 * (bid - ask) > bid + ask
//   10 * (ask - bid) > bid + ask
int8_t queue_rule_action(int32_t bid_sz_1, int32_t ask_sz_1);
