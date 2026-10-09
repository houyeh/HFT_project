#include "qat12_hls.hpp"
#include "qat12_weights.hpp"

namespace {

qat_data_t data_from_raw(int raw) {
#pragma HLS INLINE
    qat_data_t value;
    value.range(11, 0) = (ap_uint<12>)raw;
    return value;
}

qat_acc_t dot16(
    const qat_data_t x[QAT_INPUT_DIM], const int weights[QAT_INPUT_DIM], int bias
) {
#pragma HLS INLINE
    qat_acc_t product[16];
    qat_acc_t l1[8];
    qat_acc_t l2[4];
    qat_acc_t l3[2];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=l1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l3 complete dim=1
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        product[i] = (qat_acc_t)(x[i] * data_from_raw(weights[i]));
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
    return (qat_acc_t)data_from_raw(bias) + l3[0] + l3[1];
}

qat_acc_t dot8(
    const qat_data_t x[QAT_HIDDEN_DIM], const int weights[QAT_HIDDEN_DIM], int bias
) {
#pragma HLS INLINE
    qat_acc_t product[8];
    qat_acc_t l1[4];
    qat_acc_t l2[2];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=l1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l2 complete dim=1
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        product[i] = (qat_acc_t)(x[i] * data_from_raw(weights[i]));
    }
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        l1[i] = product[2 * i] + product[2 * i + 1];
    }
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        l2[i] = l1[2 * i] + l1[2 * i + 1];
    }
    return (qat_acc_t)data_from_raw(bias) + l2[0] + l2[1];
}

}  // namespace

void qat12_tiny_mlp(
    const qat_data_t x[QAT_INPUT_DIM], qat_acc_t logits[QAT_OUTPUT_DIM], qat_action_t& action
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=logits
#pragma HLS INTERFACE ap_none port=action
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=logits complete dim=1
#pragma HLS ARRAY_PARTITION variable=QAT_W1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=QAT_B1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=QAT_W2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=QAT_B2 complete dim=1
    qat_data_t hidden[QAT_HIDDEN_DIM];
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
    for (int h = 0; h < QAT_HIDDEN_DIM; ++h) {
#pragma HLS UNROLL
        const qat_acc_t value = dot16(x, QAT_W1[h], QAT_B1[h]);
        hidden[h] = value > 0 ? (qat_data_t)value : (qat_data_t)0;
    }
    for (int o = 0; o < QAT_OUTPUT_DIM; ++o) {
#pragma HLS UNROLL
        logits[o] = dot8(hidden, QAT_W2[o], QAT_B2[o]);
    }
    qat_action_t best = 0;
    qat_acc_t best_value = logits[0];
    for (int o = 1; o < QAT_OUTPUT_DIM; ++o) {
#pragma HLS PIPELINE off
        if (logits[o] > best_value) {
            best_value = logits[o];
            best = (qat_action_t)o;
        }
    }
    action = best;
}
