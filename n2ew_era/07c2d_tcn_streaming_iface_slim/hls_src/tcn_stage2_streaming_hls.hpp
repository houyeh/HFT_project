#pragma once

#include "tcn_prepad_hls.hpp"

static const int TCN_STAGE2_BLOCK0_OUT = 248;
static const int TCN_STAGE2_BLOCK1_OUT = 240;
static const int TCN_STAGE2_BLOCK2_OUT = 224;
static const int TCN_STAGE2_BLOCK3_OUT = 192;
static const int TCN_STAGE2_BLOCK4_OUT = 128;

#ifndef TCN_STAGE2_CHANNEL_UNROLL
#define TCN_STAGE2_CHANNEL_UNROLL 4
#endif

void tcn_streaming_step(
    const tcn_fixed_t x[TCN_CHANNELS],
    bool reset,
    tcn_fixed_t readout[TCN_READOUT_DIM],
    bool readout_valid[1]
);

#ifndef __SYNTHESIS__
void tcn_streaming_debug_get_block(
    int block,
    tcn_fixed_t out[TCN_CHANNELS],
    bool valid[1]
);
#endif
