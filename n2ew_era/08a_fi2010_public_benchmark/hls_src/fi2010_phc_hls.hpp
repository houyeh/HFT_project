#pragma once

#if defined(__has_include)
#  if __has_include(<ap_fixed.h>)
#    include <ap_fixed.h>
#  else
#    ifndef FI2010_AP_FIXED_FALLBACK_DEFINED
#    define FI2010_AP_FIXED_FALLBACK_DEFINED
template <int W, int I>
using ap_fixed = double;
#    endif
#  endif
#else
#  include <ap_fixed.h>
#endif

typedef ap_fixed<24, 12> fi_fixed_t;
typedef ap_fixed<40, 16> fi_acc_t;
typedef ap_fixed<48, 24> fi_var_t;

static const int FI2010_FEATURE_DIM = 40;
static const int FI2010_HIDDEN_DIM = 16;
static const int FI2010_READOUT_DIM = 3;

void fi2010_phc_step(
    const fi_fixed_t x_t[FI2010_FEATURE_DIM],
    const fi_fixed_t state_prev[FI2010_HIDDEN_DIM],
    fi_fixed_t logits[FI2010_READOUT_DIM],
    fi_fixed_t state_next[FI2010_HIDDEN_DIM]
);
