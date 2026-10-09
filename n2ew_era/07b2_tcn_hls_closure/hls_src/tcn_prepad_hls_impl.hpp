#pragma once

#include "tcn_prepad_hls.hpp"

#if defined(TCN_MODEL_SCRATCH)
#  include "tcn_weights_scratch.hpp"
#elif defined(TCN_MODEL_DISTILLED)
#  include "tcn_weights_distilled.hpp"
#else
#  error "Define TCN_MODEL_SCRATCH or TCN_MODEL_DISTILLED before including tcn_prepad_hls_impl.hpp"
#endif

namespace {

inline tcn_acc_t relu_acc(tcn_acc_t x) {
#pragma HLS INLINE
    return x > (tcn_acc_t)0 ? x : (tcn_acc_t)0;
}

template <int LEN>
void copy_and_prepad(
    const tcn_fixed_t x[TCN_SEQ_LEN][TCN_FEATURE_DIM],
    tcn_fixed_t out[LEN][TCN_CHANNELS]
) {
#pragma HLS INLINE off
    for (int t = 0; t < LEN; ++t) {
#pragma HLS PIPELINE off
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_CHANNEL_UNROLL
            if (t < TCN_PREPAD) {
                out[t][c] = (tcn_fixed_t)0;
            } else {
                out[t][c] = x[t - TCN_PREPAD][c];
            }
        }
    }
}

template <int LIN, int DILATION>
void conv1_relu(
    const tcn_fixed_t in[LIN][TCN_CHANNELS],
    tcn_fixed_t out[LIN - 2 * DILATION][TCN_CHANNELS],
    const int block
) {
#pragma HLS INLINE off
    for (int t = 0; t < LIN - 2 * DILATION; ++t) {
#pragma HLS PIPELINE off
        for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE off
            tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV1_B[block][oc];
            for (int k = 0; k < 3; ++k) {
#pragma HLS PIPELINE off
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_CHANNEL_UNROLL
                    acc += (tcn_acc_t)tcn_weights::CONV1_W[block][oc][ic][k] * (tcn_acc_t)in[t + k * DILATION][ic];
                }
            }
            out[t][oc] = (tcn_fixed_t)relu_acc(acc);
        }
    }
}

template <int LIN, int DILATION>
void conv2_relu_residual(
    const tcn_fixed_t residual_in[LIN][TCN_CHANNELS],
    const tcn_fixed_t conv1_out[LIN - 2 * DILATION][TCN_CHANNELS],
    tcn_fixed_t out[LIN - 4 * DILATION][TCN_CHANNELS],
    const int block
) {
#pragma HLS INLINE off
    for (int t = 0; t < LIN - 4 * DILATION; ++t) {
#pragma HLS PIPELINE off
        for (int oc = 0; oc < TCN_CHANNELS; ++oc) {
#pragma HLS PIPELINE off
            tcn_acc_t acc = (tcn_acc_t)tcn_weights::CONV2_B[block][oc];
            for (int k = 0; k < 3; ++k) {
#pragma HLS PIPELINE off
                for (int ic = 0; ic < TCN_CHANNELS; ++ic) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_CHANNEL_UNROLL
                    acc += (tcn_acc_t)tcn_weights::CONV2_W[block][oc][ic][k] * (tcn_acc_t)conv1_out[t + k * DILATION][ic];
                }
            }
            const tcn_acc_t conv2_relu = relu_acc(acc);
            const tcn_acc_t residual = (tcn_acc_t)residual_in[t + 4 * DILATION][oc];
            out[t][oc] = (tcn_fixed_t)relu_acc(conv2_relu + residual);
        }
    }
}

template <int LIN, int DILATION>
void tcn_block(
    const tcn_fixed_t in[LIN][TCN_CHANNELS],
    tcn_fixed_t out[LIN - 4 * DILATION][TCN_CHANNELS],
    const int block
) {
#pragma HLS INLINE off
    tcn_fixed_t tmp[LIN - 2 * DILATION][TCN_CHANNELS];
    conv1_relu<LIN, DILATION>(in, tmp, block);
    conv2_relu_residual<LIN, DILATION>(in, tmp, out, block);
}

void global_average_readout(
    const tcn_fixed_t in[TCN_SEQ_LEN][TCN_CHANNELS],
    tcn_fixed_t readout[TCN_READOUT_DIM]
) {
#pragma HLS INLINE off
    tcn_acc_t pooled[TCN_CHANNELS];
    for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS PIPELINE off
        tcn_acc_t acc = 0;
        for (int t = 0; t < TCN_SEQ_LEN; ++t) {
#pragma HLS PIPELINE off
            acc += (tcn_acc_t)in[t][c];
        }
        pooled[c] = acc / (tcn_acc_t)TCN_SEQ_LEN;
    }
    for (int o = 0; o < TCN_READOUT_DIM; ++o) {
#pragma HLS PIPELINE off
        tcn_acc_t acc = (tcn_acc_t)tcn_weights::FC_B[o];
        for (int c = 0; c < TCN_CHANNELS; ++c) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=TCN_CHANNEL_UNROLL
            acc += (tcn_acc_t)tcn_weights::FC_W[o][c] * pooled[c];
        }
        readout[o] = (tcn_fixed_t)acc;
    }
}

}  // namespace

void tcn_prepad_infer(
    const tcn_fixed_t x[TCN_SEQ_LEN][TCN_FEATURE_DIM],
    tcn_fixed_t readout[TCN_READOUT_DIM]
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
#pragma HLS INTERFACE ap_memory port=x
#pragma HLS INTERFACE ap_memory port=readout
    tcn_fixed_t buf0[252][TCN_CHANNELS];
    tcn_fixed_t buf1[248][TCN_CHANNELS];
    tcn_fixed_t buf2[240][TCN_CHANNELS];
    tcn_fixed_t buf3[224][TCN_CHANNELS];
    tcn_fixed_t buf4[192][TCN_CHANNELS];
    tcn_fixed_t buf5[128][TCN_CHANNELS];

    copy_and_prepad<252>(x, buf0);
    tcn_block<252, 1>(buf0, buf1, 0);
    tcn_block<248, 2>(buf1, buf2, 1);
    tcn_block<240, 4>(buf2, buf3, 2);
    tcn_block<224, 8>(buf3, buf4, 3);
    tcn_block<192, 16>(buf4, buf5, 4);
    global_average_readout(buf5, readout);
}
