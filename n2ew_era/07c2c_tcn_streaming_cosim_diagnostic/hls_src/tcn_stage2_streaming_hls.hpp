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
    bool readout_valid[1],
    tcn_fixed_t block0_out[TCN_CHANNELS],
    bool block0_valid[1],
    tcn_fixed_t block1_out[TCN_CHANNELS],
    bool block1_valid[1],
    tcn_fixed_t block2_out[TCN_CHANNELS],
    bool block2_valid[1],
    tcn_fixed_t block3_out[TCN_CHANNELS],
    bool block3_valid[1],
    tcn_fixed_t block4_out[TCN_CHANNELS],
    bool block4_valid[1]
);
