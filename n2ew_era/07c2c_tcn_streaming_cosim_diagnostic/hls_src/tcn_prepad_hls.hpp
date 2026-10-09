#pragma once

#if defined(__has_include)
#  if __has_include(<ap_fixed.h>)
#    include <ap_fixed.h>
#  else
// Plain g++ syntax-check fallback only. Vitis HLS must use the real ap_fixed.h.
#    ifndef TCN_AP_FIXED_FALLBACK_DEFINED
#    define TCN_AP_FIXED_FALLBACK_DEFINED
template <int W, int I>
using ap_fixed = double;
#    endif
#  endif
#else
#  include <ap_fixed.h>
#endif

typedef ap_fixed<24, 12> tcn_fixed_t;
typedef ap_fixed<40, 16> tcn_acc_t;

static const int TCN_FEATURE_DIM = 16;
static const int TCN_CHANNELS = 16;
static const int TCN_READOUT_DIM = 2;
static const int TCN_SEQ_LEN = 128;
static const int TCN_PREPAD = 124;
static const int TCN_SEQ_LEN_IN = 252;

// Inner-channel unroll is explicit so Vitis does not infer complete unroll from
// small loop defaults. You can override this via compiler flags in a later
// budget sweep, but the default is the intended first-pass synthesis setting.
#ifndef TCN_CHANNEL_UNROLL
#define TCN_CHANNEL_UNROLL 16
#endif

void tcn_prepad_infer(
    const tcn_fixed_t x[TCN_SEQ_LEN][TCN_FEATURE_DIM],
    tcn_fixed_t readout[TCN_READOUT_DIM]
);

