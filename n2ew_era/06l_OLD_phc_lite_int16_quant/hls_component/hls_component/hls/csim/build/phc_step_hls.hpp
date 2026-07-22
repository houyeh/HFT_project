#pragma once

#if defined(__has_include)
#  if __has_include(<ap_fixed.h>)
#    include <ap_fixed.h>
#  else
// Plain g++ syntax-check fallback only. Vitis HLS must use the real ap_fixed.h.
#    ifndef PHC_AP_FIXED_FALLBACK_DEFINED
#    define PHC_AP_FIXED_FALLBACK_DEFINED
template <int W, int I>
using ap_fixed = double;
#    endif
#  endif
#else
#  include <ap_fixed.h>
#endif

// Phase 06L quantization experiment: external I/O and stored weights use
// Q16.8 == ap_fixed<16,8>. Accumulators remain unchanged below.
typedef ap_fixed<16, 8> phc_fixed_t;

// Dot-products can accumulate 16 products plus bias. Wider integer and
// fractional headroom keeps intermediate products away from overflow while
// preserving enough fractional precision before final Q16.8 outputs.
typedef ap_fixed<40, 16> phc_acc_t;
typedef ap_fixed<48, 24> phc_var_t;

static const int PHC_FEATURE_DIM = 16;
static const int PHC_HIDDEN_DIM = 16;
static const int PHC_READOUT_DIM = 2;

void phc_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
);
