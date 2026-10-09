#include "tiny_mlp_hls.hpp"
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

}  // namespace

void tiny_mlp_infer(
    const tiny_data_t x[TINY_MLP_INPUT_DIM],
    tiny_acc_t logits[TINY_MLP_OUTPUT_DIM],
    tiny_action_t& action
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=logits
#pragma HLS INTERFACE ap_none port=action

    tiny_data_t hidden[TINY_MLP_HIDDEN_DIM];

HIDDEN_OUTPUT:
    for (int h = 0; h < TINY_MLP_HIDDEN_DIM; ++h) {
#pragma HLS PIPELINE off
        tiny_acc_t acc = bias_to_acc(TINY_MLP_B1_Q[h]);
HIDDEN_INPUT:
        for (int i = 0; i < TINY_MLP_INPUT_DIM; ++i) {
#pragma HLS PIPELINE II=1
            const tiny_data_t weight = data_from_raw(TINY_MLP_W1_Q[h][i]);
            acc += x[i] * weight;
        }
        hidden[h] = (acc > (tiny_acc_t)0) ? (tiny_data_t)acc : (tiny_data_t)0;
    }

OUTPUT_OUTPUT:
    for (int o = 0; o < TINY_MLP_OUTPUT_DIM; ++o) {
#pragma HLS PIPELINE off
        tiny_acc_t acc = bias_to_acc(TINY_MLP_B2_Q[o]);
OUTPUT_INPUT:
        for (int h = 0; h < TINY_MLP_HIDDEN_DIM; ++h) {
#pragma HLS PIPELINE II=1
            const tiny_data_t weight = data_from_raw(TINY_MLP_W2_Q[o][h]);
            acc += hidden[h] * weight;
        }
        logits[o] = acc;
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
