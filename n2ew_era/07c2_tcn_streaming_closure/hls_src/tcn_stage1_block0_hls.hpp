#pragma once

#include "tcn_prepad_hls.hpp"

static const int TCN_BLOCK0_DILATION = 1;
static const int TCN_BLOCK0_HISTORY = 2 * TCN_BLOCK0_DILATION + 1;
static const int TCN_BLOCK0_INPUT_EVENTS = TCN_SEQ_LEN_IN;
static const int TCN_BLOCK0_VALID_OUTPUTS = TCN_BLOCK0_INPUT_EVENTS - 4 * TCN_BLOCK0_DILATION;

void tcn_block0_stream_step(
    const tcn_fixed_t x[TCN_CHANNELS],
    bool reset,
    tcn_fixed_t y[TCN_CHANNELS],
    bool valid[1]
);
