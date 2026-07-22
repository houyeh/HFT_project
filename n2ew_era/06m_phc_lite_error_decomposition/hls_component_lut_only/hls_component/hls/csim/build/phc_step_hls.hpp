#pragma once

typedef double phc_fixed_t;
typedef double phc_acc_t;
typedef double phc_var_t;

static const int PHC_FEATURE_DIM = 16;
static const int PHC_HIDDEN_DIM = 16;
static const int PHC_READOUT_DIM = 2;

void phc_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
);
