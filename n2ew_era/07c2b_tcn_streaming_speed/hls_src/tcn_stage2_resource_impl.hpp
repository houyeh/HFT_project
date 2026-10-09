#pragma once

#include "tcn_stage2_streaming_hls.hpp"

#if defined(TCN_MODEL_SCRATCH)
#  include "tcn_weights_scratch.hpp"
#elif defined(TCN_MODEL_DISTILLED)
#  include "tcn_weights_distilled.hpp"
#else
#  error "Define TCN_MODEL_SCRATCH or TCN_MODEL_DISTILLED before including tcn_stage2_resource_impl.hpp"
#endif

#ifndef TCN_STAGE2_SHARED_HISTORY_STORAGE
#define TCN_STAGE2_SHARED_HISTORY_STORAGE 0
#endif

#ifndef TCN_STAGE2_TOP_PIPELINE
#define TCN_STAGE2_TOP_PIPELINE 0
#endif

#ifndef TCN_STAGE2_BLOCK_PIPELINE
#define TCN_STAGE2_BLOCK_PIPELINE 0
#endif

#ifndef TCN_STAGE2_INNER_PIPELINE
#define TCN_STAGE2_INNER_PIPELINE 0
#endif

#ifndef TCN_STAGE2_LOCAL_TAP_BUFFER
#define TCN_STAGE2_LOCAL_TAP_BUFFER 0
#endif

namespace {

inline tcn_acc_t stage2_resource_relu_acc(tcn_acc_t x) {
#pragma HLS INLINE
    return x > (tcn_acc_t)0 ? x : (tcn_acc_t)0;
}

template <int DEPTH>
inline int stage2_resource_ring_minus(int cur, int offset) {
#pragma HLS INLINE
    int idx = cur - offset;
    while (idx < 0) {
        idx += DEPTH;
    }
    return idx;
}

inline void stage2_resource_zero_vec(tcn_fixed_t x[TCN_CHANNELS]) {
#pragma HLS INLINE
    for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
        x[c] = (tcn_fixed_t)0;
    }
}

template <int BLOCK, int DILATION, int DEPTH>
void stage2_resource_stream_block_step(
    const tcn_fixed_t x[TCN_CHANNELS],
    bool reset,
    bool input_valid,
    tcn_fixed_t y[TCN_CHANNELS],
    bool& output_valid
) {
#pragma HLS INLINE off
    static tcn_fixed_t input_hist[DEPTH][TCN_CHANNELS];
    static tcn_fixed_t conv1_hist[DEPTH][TCN_CHANNELS];
    static int input_pos = 0;
    static int conv1_pos = 0;
    static int event_idx = 0;

#if TCN_STAGE2_SHARED_HISTORY_STORAGE
#pragma HLS BIND_STORAGE variable=input_hist type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv1_hist type=ram_2p impl=bram
#else
#pragma HLS ARRAY_PARTITION variable=input_hist complete dim=0
#pragma HLS ARRAY_PARTITION variable=conv1_hist complete dim=0
#endif

    if (reset) {
        for (int r = 0; r < DEPTH; ++r) {
#pragma HLS PIPELINE off
            for (int c = 0; c < TCN_CHANNELS; ++c) {
#if TCN_STAGE2_SHARED_HISTORY_STORAGE
#pragma HLS PIPELINE off
#else
#pragma HLS UNROLL
#endif
                input_hist[r][c] = (tcn_fixed_t)0;
                conv1_hist[r][c] = (tcn_fixed_t)0;
            }
        }
        input_pos = 0;
        conv1_pos = 0;
        event_idx = 0;
        output_valid = false;
        stage2_resource_zero_vec(y);
        return;
    }

    if (!input_valid) {
        output_valid = false;
        stage2_resource_zero_vec(y);
        return;
    }

    const int cur_input = input_pos;
    const int input_m1 = stage2_resource_ring_minus<DEPTH>(cur_input, DILATION);
    const int input_m2 = stage2_resource_ring_minus<DEPTH>(cur_input, 2 * DILATION);

    for (int c = 0; c < TCN_CHANNELS; ++c) {
#if TCN_STAGE2_SHARED_HISTORY_STORAGE
#pragma HLS PIPELINE off
#else
#pragma HLS UNROLL
#endif
        input_hist[cur_input][c] = x[c];
    }
    input_pos = input_pos == DEPTH - 1 ? 0 : input_pos + 1;

#if TCN_STAGE2_LOCAL_TAP_BUFFER
    tcn_fixed_t input_m2_vec[TCN_CHANNELS];
    tcn_fixed_t input_m1_vec[TCN_CHANNELS];
    tcn_fixed_t input_cur_vec[TCN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=input_m2_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_m1_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_cur_vec complete dim=1
    for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS PIPELINE off
        input_m2_vec[c] = input_hist[input_m2][c];
        input_m1_vec[c] = input_hist[input_m1][c];
        input_cur_vec[c] = x[c];
    }
#endif

    tcn_fixed_t conv1_vec[TCN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=conv1_vec complete dim=1
    stage2_resource_zero_vec(conv1_vec);

    if (event_idx >= 2 * DILATION) {
        for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#if TCN_STAGE2_BLOCK_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#endif
            tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV1_B[BLOCK][oc];
            for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#if TCN_STAGE2_INNER_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL
#endif
#if TCN_STAGE2_LOCAL_TAP_BUFFER
                const tcn_acc_t in_m2 = (tcn_acc_t)input_m2_vec[ic];
                const tcn_acc_t in_m1 = (tcn_acc_t)input_m1_vec[ic];
                const tcn_acc_t in_cur = (tcn_acc_t)input_cur_vec[ic];
#else
                const tcn_acc_t in_m2 = (tcn_acc_t)input_hist[input_m2][ic];
                const tcn_acc_t in_m1 = (tcn_acc_t)input_hist[input_m1][ic];
                const tcn_acc_t in_cur = (tcn_acc_t)input_hist[cur_input][ic];
#endif
                const tcn_acc_t tap_m2 = (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][0] * in_m2;
                const tcn_acc_t tap_m1 = (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][1] * in_m1;
                const tcn_acc_t tap_cur = (tcn_acc_t)tcn_weights::CONV1_W[BLOCK][oc][ic][2] * in_cur;
                const tcn_acc_t tap_sum = (tap_m2 + tap_m1) + tap_cur;
                acc += tap_sum;
            }
            conv1_vec[oc] = (tcn_fixed_t)stage2_resource_relu_acc(acc);
        }

        const int cur_conv1 = conv1_pos;
        const int conv1_m1 = stage2_resource_ring_minus<DEPTH>(cur_conv1, DILATION);
        const int conv1_m2 = stage2_resource_ring_minus<DEPTH>(cur_conv1, 2 * DILATION);
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#if TCN_STAGE2_SHARED_HISTORY_STORAGE
#pragma HLS PIPELINE off
#else
#pragma HLS UNROLL
#endif
            conv1_hist[cur_conv1][c] = conv1_vec[c];
        }
        conv1_pos = conv1_pos == DEPTH - 1 ? 0 : conv1_pos + 1;

#if TCN_STAGE2_LOCAL_TAP_BUFFER
        tcn_fixed_t conv1_m2_vec[TCN_CHANNELS];
        tcn_fixed_t conv1_m1_vec[TCN_CHANNELS];
        tcn_fixed_t conv1_cur_vec[TCN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=conv1_m2_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv1_m1_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv1_cur_vec complete dim=1
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS PIPELINE off
            conv1_m2_vec[c] = conv1_hist[conv1_m2][c];
            conv1_m1_vec[c] = conv1_hist[conv1_m1][c];
            conv1_cur_vec[c] = conv1_vec[c];
        }
#endif

        if (event_idx >= 4 * DILATION) {
            output_valid = true;
            for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#if TCN_STAGE2_BLOCK_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#endif
                tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV2_B[BLOCK][oc];
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#if TCN_STAGE2_INNER_PIPELINE
#pragma HLS PIPELINE II=1
#else
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL
#endif
#if TCN_STAGE2_LOCAL_TAP_BUFFER
                    const tcn_acc_t c1_m2 = (tcn_acc_t)conv1_m2_vec[ic];
                    const tcn_acc_t c1_m1 = (tcn_acc_t)conv1_m1_vec[ic];
                    const tcn_acc_t c1_cur = (tcn_acc_t)conv1_cur_vec[ic];
#else
                    const tcn_acc_t c1_m2 = (tcn_acc_t)conv1_hist[conv1_m2][ic];
                    const tcn_acc_t c1_m1 = (tcn_acc_t)conv1_hist[conv1_m1][ic];
                    const tcn_acc_t c1_cur = (tcn_acc_t)conv1_hist[cur_conv1][ic];
#endif
                    const tcn_acc_t tap_c1_m2 = (tcn_acc_t)tcn_weights::CONV2_W[BLOCK][oc][ic][0] * c1_m2;
                    const tcn_acc_t tap_c1_m1 = (tcn_acc_t)tcn_weights::CONV2_W[BLOCK][oc][ic][1] * c1_m1;
                    const tcn_acc_t tap_c1_cur = (tcn_acc_t)tcn_weights::CONV2_W[BLOCK][oc][ic][2] * c1_cur;
                    const tcn_acc_t tap_sum2 = (tap_c1_m2 + tap_c1_m1) + tap_c1_cur;
                    acc += tap_sum2;
                }
                const tcn_acc_t conv2_relu = stage2_resource_relu_acc(acc);
                y[oc] = (tcn_fixed_t)stage2_resource_relu_acc(conv2_relu + (tcn_acc_t)x[oc]);
            }
        } else {
            output_valid = false;
            stage2_resource_zero_vec(y);
        }
    } else {
        output_valid = false;
        stage2_resource_zero_vec(y);
    }

    event_idx++;
}

}  // namespace

void tcn_streaming_step(
    const tcn_fixed_t x[TCN_CHANNELS],
    bool reset,
    tcn_fixed_t readout[TCN_READOUT_DIM],
    bool readout_valid[1],
    tcn_fixed_t block0_out[TCN_CHANNELS],
    bool block0_valid[1],
    tcn_fixed_t block1_out[TCN_CHANNELS],
    bool block1_valid[1],
    tcn_fixed_t block2_out[TCN_CHANNELS],
    bool block2_valid[1],
    tcn_fixed_t block3_out[TCN_CHANNELS],
    bool block3_valid[1],
    tcn_fixed_t block4_out[TCN_CHANNELS],
    bool block4_valid[1]
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_none port=reset
#pragma HLS INTERFACE ap_memory port=readout
#pragma HLS INTERFACE ap_memory port=readout_valid
#pragma HLS INTERFACE ap_memory port=block0_out
#pragma HLS INTERFACE ap_memory port=block0_valid
#pragma HLS INTERFACE ap_memory port=block1_out
#pragma HLS INTERFACE ap_memory port=block1_valid
#pragma HLS INTERFACE ap_memory port=block2_out
#pragma HLS INTERFACE ap_memory port=block2_valid
#pragma HLS INTERFACE ap_memory port=block3_out
#pragma HLS INTERFACE ap_memory port=block3_valid
#pragma HLS INTERFACE ap_memory port=block4_out
#pragma HLS INTERFACE ap_memory port=block4_valid

#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=readout complete dim=1
#pragma HLS ARRAY_PARTITION variable=block0_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block1_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block2_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block3_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=block4_out complete dim=1
#if TCN_STAGE2_TOP_PIPELINE
#pragma HLS PIPELINE
#endif

    static tcn_fixed_t readout_ring[TCN_STAGE2_BLOCK4_OUT][TCN_CHANNELS];
    static tcn_acc_t readout_sum[TCN_CHANNELS];
    static int readout_pos = 0;
    static int readout_count = 0;

#pragma HLS ARRAY_PARTITION variable=readout_ring complete dim=2
#pragma HLS ARRAY_PARTITION variable=readout_sum complete dim=1

    if (reset) {
        tcn_fixed_t zero[TCN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable=zero complete dim=1
        stage2_resource_zero_vec(zero);
        bool dummy_valid = false;
        stage2_resource_stream_block_step<0, 1, 3>(zero, true, false, block0_out, dummy_valid);
        stage2_resource_stream_block_step<1, 2, 5>(zero, true, false, block1_out, dummy_valid);
        stage2_resource_stream_block_step<2, 4, 9>(zero, true, false, block2_out, dummy_valid);
        stage2_resource_stream_block_step<3, 8, 17>(zero, true, false, block3_out, dummy_valid);
        stage2_resource_stream_block_step<4, 16, 33>(zero, true, false, block4_out, dummy_valid);

        for (int t = 0; t < TCN_STAGE2_BLOCK4_OUT; ++t) {
#pragma HLS PIPELINE off
            for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
                readout_ring[t][c] = (tcn_fixed_t)0;
            }
        }
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
            readout_sum[c] = (tcn_acc_t)0;
        }
        readout_pos = 0;
        readout_count = 0;
        for (int i = 0; i < TCN_READOUT_DIM; ++i) {
#pragma HLS UNROLL
            readout[i] = (tcn_fixed_t)0;
        }
        readout_valid[0] = false;
        block0_valid[0] = false;
        block1_valid[0] = false;
        block2_valid[0] = false;
        block3_valid[0] = false;
        block4_valid[0] = false;
        return;
    }

    bool v0 = false;
    bool v1 = false;
    bool v2 = false;
    bool v3 = false;
    bool v4 = false;

    stage2_resource_stream_block_step<0, 1, 3>(x, false, true, block0_out, v0);
    stage2_resource_stream_block_step<1, 2, 5>(block0_out, false, v0, block1_out, v1);
    stage2_resource_stream_block_step<2, 4, 9>(block1_out, false, v1, block2_out, v2);
    stage2_resource_stream_block_step<3, 8, 17>(block2_out, false, v2, block3_out, v3);
    stage2_resource_stream_block_step<4, 16, 33>(block3_out, false, v3, block4_out, v4);

    block0_valid[0] = v0;
    block1_valid[0] = v1;
    block2_valid[0] = v2;
    block3_valid[0] = v3;
    block4_valid[0] = v4;
    readout_valid[0] = false;

    if (v4) {
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS UNROLL
            if (readout_count >= TCN_STAGE2_BLOCK4_OUT) {
                readout_sum[c] -= (tcn_acc_t)readout_ring[readout_pos][c];
            }
            readout_ring[readout_pos][c] = block4_out[c];
            readout_sum[c] += (tcn_acc_t)block4_out[c];
        }
        if (readout_count < TCN_STAGE2_BLOCK4_OUT) {
            readout_count++;
        }
        readout_pos = readout_pos == TCN_STAGE2_BLOCK4_OUT - 1 ? 0 : readout_pos + 1;

        if (readout_count >= TCN_STAGE2_BLOCK4_OUT) {
            readout_valid[0] = true;
            for (int o = 0; o < TCN_READOUT_DIM; ++o) {
#pragma HLS PIPELINE off
                tcn_acc_t acc = (tcn_acc_t)tcn_weights::FC_B[o];
                for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_STAGE2_CHANNEL_UNROLL
                    const tcn_acc_t pooled = readout_sum[c] / (tcn_acc_t)TCN_STAGE2_BLOCK4_OUT;
                    acc += (tcn_acc_t)tcn_weights::FC_W[o][c] * pooled;
                }
                readout[o] = (tcn_fixed_t)acc;
            }
        }
    }
}
