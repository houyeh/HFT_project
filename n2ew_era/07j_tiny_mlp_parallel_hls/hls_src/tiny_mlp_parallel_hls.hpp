#pragma once

#include <ap_fixed.h>
#include <ap_int.h>

static const int TINY_MLP_INPUT_DIM = 16;
static const int TINY_MLP_HIDDEN_DIM = 8;
static const int TINY_MLP_OUTPUT_DIM = 3;

typedef ap_fixed<28, 14, AP_TRN, AP_WRAP> tiny_data_t;
typedef ap_fixed<40, 16, AP_TRN, AP_WRAP> tiny_acc_t;
typedef ap_uint<2> tiny_action_t;

void tiny_mlp_parallel(
    const tiny_data_t x[TINY_MLP_INPUT_DIM],
    tiny_acc_t logits[TINY_MLP_OUTPUT_DIM],
    tiny_action_t& action
);
