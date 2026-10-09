#pragma once

#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<12, 6, AP_TRN, AP_SAT> qat_data_t;
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> qat_acc_t;
typedef ap_uint<2> qat_action_t;

#define QAT_SCALAR_INPUTS \
    qat_data_t x0, qat_data_t x1, qat_data_t x2, qat_data_t x3, \
    qat_data_t x4, qat_data_t x5, qat_data_t x6, qat_data_t x7, \
    qat_data_t x8, qat_data_t x9, qat_data_t x10, qat_data_t x11, \
    qat_data_t x12, qat_data_t x13, qat_data_t x14, qat_data_t x15

void qat12_scalar_hs_pipeline(
    QAT_SCALAR_INPUTS,
    qat_acc_t& logit0, qat_acc_t& logit1, qat_acc_t& logit2,
    qat_action_t& action
);

void qat12_scalar_none_pipeline(
    QAT_SCALAR_INPUTS,
    qat_acc_t& logit0, qat_acc_t& logit1, qat_acc_t& logit2,
    qat_action_t& action
);
