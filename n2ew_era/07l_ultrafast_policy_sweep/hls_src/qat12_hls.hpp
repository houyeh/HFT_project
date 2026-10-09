#pragma once

#include <ap_fixed.h>
#include <ap_int.h>

static const int QAT_INPUT_DIM = 16;
static const int QAT_HIDDEN_DIM = 8;
static const int QAT_OUTPUT_DIM = 3;

typedef ap_fixed<12, 6, AP_TRN, AP_SAT> qat_data_t;
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> qat_acc_t;
typedef ap_uint<2> qat_action_t;

void qat12_tiny_mlp(
    const qat_data_t x[QAT_INPUT_DIM],
    qat_acc_t logits[QAT_OUTPUT_DIM],
    qat_action_t& action
);
