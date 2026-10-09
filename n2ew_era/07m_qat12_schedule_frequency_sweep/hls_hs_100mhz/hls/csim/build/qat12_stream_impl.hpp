#pragma once

#include "qat12_stream.hpp"
#include "qat12_weights.hpp"

inline qat_data_t qat_from_raw(int raw) {
#pragma HLS INLINE
    qat_data_t value;
    value.range(11, 0) = (ap_uint<12>)raw;
    return value;
}

inline qat_acc_t qat_dot16(const qat_data_t x[16], const int weights[16], int bias) {
#pragma HLS INLINE
    qat_acc_t product[16], l1[8], l2[4], l3[2];
#pragma HLS ARRAY_PARTITION variable=product complete
#pragma HLS ARRAY_PARTITION variable=l1 complete
#pragma HLS ARRAY_PARTITION variable=l2 complete
#pragma HLS ARRAY_PARTITION variable=l3 complete
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        product[i] = (qat_acc_t)(x[i] * qat_from_raw(weights[i]));
    }
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        l1[i] = product[2 * i] + product[2 * i + 1];
    }
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        l2[i] = l1[2 * i] + l1[2 * i + 1];
    }
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        l3[i] = l2[2 * i] + l2[2 * i + 1];
    }
    return (qat_acc_t)qat_from_raw(bias) + l3[0] + l3[1];
}

inline qat_acc_t qat_dot8(const qat_data_t x[8], const int weights[8], int bias) {
#pragma HLS INLINE
    qat_acc_t product[8], l1[4], l2[2];
#pragma HLS ARRAY_PARTITION variable=product complete
#pragma HLS ARRAY_PARTITION variable=l1 complete
#pragma HLS ARRAY_PARTITION variable=l2 complete
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        product[i] = (qat_acc_t)(x[i] * qat_from_raw(weights[i]));
    }
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        l1[i] = product[2 * i] + product[2 * i + 1];
    }
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        l2[i] = l1[2 * i] + l1[2 * i + 1];
    }
    return (qat_acc_t)qat_from_raw(bias) + l2[0] + l2[1];
}

inline void qat_compute(
    const qat_data_t x[16], qat_acc_t& logit0, qat_acc_t& logit1,
    qat_acc_t& logit2, qat_action_t& action
) {
#pragma HLS INLINE
    qat_data_t hidden[8];
    qat_acc_t logits[3];
#pragma HLS ARRAY_PARTITION variable=x complete
#pragma HLS ARRAY_PARTITION variable=hidden complete
#pragma HLS ARRAY_PARTITION variable=logits complete
#pragma HLS ARRAY_PARTITION variable=QAT_W1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=QAT_B1 complete
#pragma HLS ARRAY_PARTITION variable=QAT_W2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=QAT_B2 complete
    for (int h = 0; h < 8; ++h) {
#pragma HLS UNROLL
        const qat_acc_t value = qat_dot16(x, QAT_W1[h], QAT_B1[h]);
        hidden[h] = value > 0 ? (qat_data_t)value : (qat_data_t)0;
    }
    for (int o = 0; o < 3; ++o) {
#pragma HLS UNROLL
        logits[o] = qat_dot8(hidden, QAT_W2[o], QAT_B2[o]);
    }
    logit0 = logits[0];
    logit1 = logits[1];
    logit2 = logits[2];
    qat_action_t best = 0;
    qat_acc_t best_value = logits[0];
    if (logits[1] > best_value) { best = 1; best_value = logits[1]; }
    if (logits[2] > best_value) { best = 2; }
    action = best;
}

#define QAT_PACK_INPUTS \
    qat_data_t x[16] = {x0, x1, x2, x3, x4, x5, x6, x7, \
                        x8, x9, x10, x11, x12, x13, x14, x15}; \
    _Pragma("HLS ARRAY_PARTITION variable=x complete")
