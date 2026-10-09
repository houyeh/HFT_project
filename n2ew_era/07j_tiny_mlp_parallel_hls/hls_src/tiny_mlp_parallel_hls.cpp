#include "tiny_mlp_parallel_hls.hpp"
#include "tiny_mlp_weights.hpp"

namespace {

tiny_data_t data_from_raw(long long raw) {
#pragma HLS INLINE
    tiny_data_t value;
    value.range(27, 0) = (ap_uint<28>)raw;
    return value;
}

tiny_acc_t bias_to_acc(long long raw) {
#pragma HLS INLINE
    return (tiny_acc_t)data_from_raw(raw);
}

tiny_acc_t dot16_balanced(
    const tiny_data_t x[TINY_MLP_INPUT_DIM],
    const long long weights[TINY_MLP_INPUT_DIM],
    long long bias
) {
#pragma HLS INLINE
    tiny_acc_t product[16];
    tiny_acc_t level1[8];
    tiny_acc_t level2[4];
    tiny_acc_t level3[2];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=level1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=level2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=level3 complete dim=1

DOT16_PRODUCT:
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        product[i] = (tiny_acc_t)(x[i] * data_from_raw(weights[i]));
    }
DOT16_LEVEL1:
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        level1[i] = product[2 * i] + product[2 * i + 1];
    }
DOT16_LEVEL2:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        level2[i] = level1[2 * i] + level1[2 * i + 1];
    }
DOT16_LEVEL3:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        level3[i] = level2[2 * i] + level2[2 * i + 1];
    }
    return bias_to_acc(bias) + level3[0] + level3[1];
}

tiny_acc_t dot8_balanced(
    const tiny_data_t x[TINY_MLP_HIDDEN_DIM],
    const long long weights[TINY_MLP_HIDDEN_DIM],
    long long bias
) {
#pragma HLS INLINE
    tiny_acc_t product[8];
    tiny_acc_t level1[4];
    tiny_acc_t level2[2];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=level1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=level2 complete dim=1

DOT8_PRODUCT:
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        product[i] = (tiny_acc_t)(x[i] * data_from_raw(weights[i]));
    }
DOT8_LEVEL1:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        level1[i] = product[2 * i] + product[2 * i + 1];
    }
DOT8_LEVEL2:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        level2[i] = level1[2 * i] + level1[2 * i + 1];
    }
    return bias_to_acc(bias) + level2[0] + level2[1];
}

}  // namespace

void tiny_mlp_parallel(
    const tiny_data_t x[TINY_MLP_INPUT_DIM],
    tiny_acc_t logits[TINY_MLP_OUTPUT_DIM],
    tiny_action_t& action
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=logits
#pragma HLS INTERFACE ap_none port=action
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=logits complete dim=1
#pragma HLS ARRAY_PARTITION variable=TINY_MLP_W1_Q complete dim=1
#pragma HLS ARRAY_PARTITION variable=TINY_MLP_W1_Q complete dim=2
#pragma HLS ARRAY_PARTITION variable=TINY_MLP_B1_Q complete dim=1
#pragma HLS ARRAY_PARTITION variable=TINY_MLP_W2_Q complete dim=0
#pragma HLS ARRAY_PARTITION variable=TINY_MLP_B2_Q complete dim=1

    tiny_data_t hidden[TINY_MLP_HIDDEN_DIM];
#pragma HLS ARRAY_PARTITION variable=hidden complete dim=1

HIDDEN_OUTPUT:
    for (int h = 0; h < TINY_MLP_HIDDEN_DIM; ++h) {
#pragma HLS UNROLL
        const tiny_acc_t acc = dot16_balanced(x, TINY_MLP_W1_Q[h], TINY_MLP_B1_Q[h]);
        hidden[h] = (acc > (tiny_acc_t)0) ? (tiny_data_t)acc : (tiny_data_t)0;
    }

OUTPUT_OUTPUT:
    for (int o = 0; o < TINY_MLP_OUTPUT_DIM; ++o) {
#pragma HLS UNROLL
        logits[o] = dot8_balanced(hidden, TINY_MLP_W2_Q[o], TINY_MLP_B2_Q[o]);
    }

    tiny_action_t best = 0;
    tiny_acc_t best_value = logits[0];
ARGMAX:
    for (int o = 1; o < TINY_MLP_OUTPUT_DIM; ++o) {
#pragma HLS PIPELINE off
        if (logits[o] > best_value) {
            best_value = logits[o];
            best = (tiny_action_t)o;
        }
    }
    action = best;
}
