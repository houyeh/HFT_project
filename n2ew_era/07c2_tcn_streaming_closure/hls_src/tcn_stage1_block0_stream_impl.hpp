#pragma once

#include "tcn_stage1_block0_hls.hpp"

#if defined(TCN_MODEL_SCRATCH)
#  include "tcn_weights_scratch.hpp"
#elif defined(TCN_MODEL_DISTILLED)
#  include "tcn_weights_distilled.hpp"
#else
#  error "Define TCN_MODEL_SCRATCH or TCN_MODEL_DISTILLED before including tcn_stage1_block0_stream_impl.hpp"
#endif

namespace {

inline tcn_acc_t relu_acc(tcn_acc_t x) {
#pragma HLS INLINE
    return x > (tcn_acc_t)0 ? x : (tcn_acc_t)0;
}

inline int ring_prev1(int cur) {
#pragma HLS INLINE
    return cur == 0 ? 2 : cur - 1;
}

inline int ring_prev2(int cur) {
#pragma HLS INLINE
    return cur == 2 ? 0 : cur + 1;
}

}  // namespace

void tcn_block0_stream_step(
    const tcn_fixed_t x[TCN_CHANNELS],
    bool reset,
    tcn_fixed_t y[TCN_CHANNELS],
    bool valid[1]
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=y
#pragma HLS INTERFACE ap_memory port=valid
#pragma HLS INTERFACE ap_none port=reset
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=1

    static tcn_fixed_t input_hist[TCN_BLOCK0_HISTORY][TCN_CHANNELS];
    static tcn_fixed_t conv1_hist[TCN_BLOCK0_HISTORY][TCN_CHANNELS];
    static int input_pos = 0;
    static int conv1_pos = 0;
    static int event_idx = 0;

#pragma HLS ARRAY_PARTITION variable=input_hist complete dim=0
#pragma HLS ARRAY_PARTITION variable=conv1_hist complete dim=0

    if (reset) {
        for (int r = 0; r < TCN_BLOCK0_HISTORY; ++r) {
#pragma HLS UNROLL
            for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
                input_hist[r][c] = (tcn_fixed_t)0;
                conv1_hist[r][c] = (tcn_fixed_t)0;
            }
        }
        input_pos = 0;
        conv1_pos = 0;
        event_idx = 0;
        valid[0] = false;
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
            y[c] = (tcn_fixed_t)0;
        }
        return;
    }

    const int cur_input = input_pos;
    const int input_m1 = ring_prev1(cur_input);
    const int input_m2 = ring_prev2(cur_input);
    for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
        input_hist[cur_input][c] = x[c];
    }
    input_pos = input_pos == 2 ? 0 : input_pos + 1;

    tcn_fixed_t conv1_vec[TCN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=conv1_vec complete dim=1
    for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
        conv1_vec[c] = (tcn_fixed_t)0;
    }

    if (event_idx >= 2) {
        for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE II=1
            tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV1_B[0][oc];
            for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#pragma HLS UNROLL factor=TCN_CHANNEL_UNROLL
                acc += (tcn_acc_t)tcn_weights::CONV1_W[0][oc][ic][0] * (tcn_acc_t)input_hist[input_m2][ic];
                acc += (tcn_acc_t)tcn_weights::CONV1_W[0][oc][ic][1] * (tcn_acc_t)input_hist[input_m1][ic];
                acc += (tcn_acc_t)tcn_weights::CONV1_W[0][oc][ic][2] * (tcn_acc_t)input_hist[cur_input][ic];
            }
            conv1_vec[oc] = (tcn_fixed_t)relu_acc(acc);
        }

        const int cur_conv1 = conv1_pos;
        const int conv1_m1 = ring_prev1(cur_conv1);
        const int conv1_m2 = ring_prev2(cur_conv1);
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
            conv1_hist[cur_conv1][c] = conv1_vec[c];
        }
        conv1_pos = conv1_pos == 2 ? 0 : conv1_pos + 1;

        if (event_idx >= 4) {
            valid[0] = true;
            for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE II=1
                tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV2_B[0][oc];
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#pragma HLS UNROLL factor=TCN_CHANNEL_UNROLL
                    acc += (tcn_acc_t)tcn_weights::CONV2_W[0][oc][ic][0] * (tcn_acc_t)conv1_hist[conv1_m2][ic];
                    acc += (tcn_acc_t)tcn_weights::CONV2_W[0][oc][ic][1] * (tcn_acc_t)conv1_hist[conv1_m1][ic];
                    acc += (tcn_acc_t)tcn_weights::CONV2_W[0][oc][ic][2] * (tcn_acc_t)conv1_hist[cur_conv1][ic];
                }
                const tcn_acc_t conv2_relu = relu_acc(acc);
                y[oc] = (tcn_fixed_t)relu_acc(conv2_relu + (tcn_acc_t)x[oc]);
            }
        } else {
            valid[0] = false;
            for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
                y[c] = (tcn_fixed_t)0;
            }
        }
    } else {
        valid[0] = false;
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
            y[c] = (tcn_fixed_t)0;
        }
    }

    event_idx++;
}
