#include "phc_step_hls.hpp"
#include "phc_weights.hpp"

#include <cmath>
#include <hls_math.h>

namespace {

static const phc_acc_t LN_EPS = (phc_acc_t)1.0e-5;

// Quant-only control: keep Q24.12/ap_fixed inputs, weights, accumulators, and
// final casts, but replace the 65-point LUT nonlinearities with real floating
// sigmoid/tanh/SiLU evaluated in double and cast back to phc_acc_t.
phc_acc_t sigmoid_scalar(phc_acc_t x) {
#pragma HLS INLINE
    const double xd = static_cast<double>(x);
    return (phc_acc_t)(1.0 / (1.0 + std::exp(-xd)));
}

phc_acc_t tanh_scalar(phc_acc_t x) {
#pragma HLS INLINE
    return (phc_acc_t)std::tanh(static_cast<double>(x));
}

void sigmoid_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = sigmoid_scalar(in[i]);
    }
}

void tanh_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = tanh_scalar(in[i]);
    }
}

void silu_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_16_16_fixed(const phc_fixed_t in[16], const phc_fixed_t w[16][16], const phc_fixed_t b[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_acc(const phc_acc_t in[16], const phc_fixed_t w[16][16], const phc_fixed_t b[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_fixed_no_bias(const phc_fixed_t in[16], const phc_fixed_t w[16][16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = 0;
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_2_acc(const phc_acc_t in[16], const phc_fixed_t w[2][16], const phc_fixed_t b[2], phc_acc_t out[2]) {
#pragma HLS INLINE
    for (int j = 0; j < 2; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void layernorm_16(const phc_acc_t in[16], const phc_fixed_t gamma[16], const phc_fixed_t beta[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    phc_acc_t mean = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        mean += in[i];
    }
    mean = mean / (phc_acc_t)16.0;
    phc_var_t var = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        phc_acc_t d = in[i] - mean;
        var += (phc_var_t)d * (phc_var_t)d;
    }
    var = var / (phc_var_t)16.0;
    phc_acc_t inv_std = (phc_acc_t)1.0 / (phc_acc_t)hls::sqrt(var + (phc_var_t)LN_EPS);
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = (phc_acc_t)gamma[i] * (in[i] - mean) * inv_std + (phc_acc_t)beta[i];
    }
}

}  // namespace

void phc_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
) {
    phc_acc_t enc_linear_out[16];
    phc_acc_t encoded_pre[16];
    phc_acc_t encoded[16];
    phc_acc_t gate_pre[16];
    phc_acc_t gate_t[16];
    phc_acc_t candidate_pre[16];
    phc_acc_t candidate_t[16];
    phc_acc_t state_temporal[16];
    phc_acc_t gain_encoded[16];
    phc_acc_t gain_prev_state[16];
    phc_acc_t gain_pre[16];
    phc_acc_t gain_t[16];
    phc_acc_t message_pre[16];
    phc_acc_t message[16];
    phc_acc_t loop_input[16];
    phc_acc_t z[16];
    phc_acc_t readout_acc[2];

    using namespace phc_step_weights;

    linear_16_16_fixed(x_t, ENC_LINEAR_W, ENC_LINEAR_B, enc_linear_out);
    layernorm_16(enc_linear_out, ENC_NORM_GAMMA, ENC_NORM_BETA, encoded_pre);
    silu_16(encoded_pre, encoded);

    linear_16_16_acc(encoded, GATE_W, GATE_B, gate_pre);
    sigmoid_16(gate_pre, gate_t);

    linear_16_16_acc(encoded, CANDIDATE_W, CANDIDATE_B, candidate_pre);
    tanh_16(candidate_pre, candidate_t);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_temporal[i] = (phc_acc_t)DECAY[i] * (phc_acc_t)state_prev[i] + gate_t[i] * candidate_t[i];
    }

    linear_16_16_acc(encoded, GAIN_FROM_ENCODED_W, GAIN_FROM_ENCODED_B, gain_encoded);
    linear_16_16_fixed_no_bias(state_prev, GAIN_FROM_PREV_STATE_W, gain_prev_state);
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        gain_pre[i] = gain_encoded[i] + gain_prev_state[i];
    }
    sigmoid_16(gain_pre, gain_t);

    linear_16_16_acc(state_temporal, SYNAPSE_W, SYNAPSE_B, message_pre);
    silu_16(message_pre, message);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        loop_input[i] = state_temporal[i] + (phc_acc_t)ALPHA_SCALAR[0] * gain_t[i] * message[i];
    }
    layernorm_16(loop_input, LOOP_NORM_GAMMA, LOOP_NORM_BETA, z);

    linear_16_2_acc(z, READOUT_W, READOUT_B, readout_acc);

    for (int i = 0; i < 2; ++i) {
#pragma HLS PIPELINE off
        readout[i] = (phc_fixed_t)readout_acc[i];
    }
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_next[i] = (phc_fixed_t)z[i];
    }
}
