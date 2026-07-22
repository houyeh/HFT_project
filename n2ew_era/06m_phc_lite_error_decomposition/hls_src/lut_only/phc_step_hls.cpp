#include "phc_step_hls.hpp"
#include "phc_weights_double.hpp"

#include <cmath>

namespace {

static const phc_acc_t LN_EPS = 1.0e-5;

phc_acc_t tanh_lut_scalar(phc_acc_t x) {
    static const phc_acc_t TANH_LUT[65] = {
        -0.9993292997, -0.9991388858, -0.9988944427, -0.9985806592,
        -0.9981778976, -0.9976609795, -0.9969976355, -0.9961465307,
        -0.9950547537, -0.9936546343, -0.9918597246, -0.9895597486,
        -0.9866142982, -0.9828450292, -0.9780261147, -0.9718727459,
        -0.9640275801, -0.9540452602, -0.9413755385, -0.9253462253,
        -0.9051482536, -0.8798266997, -0.8482836400, -0.8093010702,
        -0.7615941560, -0.7039056039, -0.6351489524, -0.5545997223,
        -0.4621171573, -0.3583573984, -0.2449186624, -0.1243530018,
        0.0000000000,  0.1243530018,  0.2449186624,  0.3583573984,
        0.4621171573,  0.5545997223,  0.6351489524,  0.7039056039,
        0.7615941560,  0.8093010702,  0.8482836400,  0.8798266997,
        0.9051482536,  0.9253462253,  0.9413755385,  0.9540452602,
        0.9640275801,  0.9718727459,  0.9780261147,  0.9828450292,
        0.9866142982,  0.9895597486,  0.9918597246,  0.9936546343,
        0.9950547537,  0.9961465307,  0.9969976355,  0.9976609795,
        0.9981778976,  0.9985806592,  0.9988944427,  0.9991388858,
        0.9993292997
    };
    if (x <= -4.0) {
        return -1.0;
    }
    if (x >= 4.0) {
        return 1.0;
    }
    phc_acc_t scaled = (x + 4.0) * 8.0;
    int idx = (int)scaled;
    if (idx < 0) {
        idx = 0;
    }
    if (idx > 63) {
        idx = 63;
    }
    phc_acc_t frac = scaled - (phc_acc_t)idx;
    phc_acc_t y0 = TANH_LUT[idx];
    phc_acc_t y1 = TANH_LUT[idx + 1];
    return y0 + frac * (y1 - y0);
}

phc_acc_t sigmoid_scalar(phc_acc_t x) {
    return 0.5 * (tanh_lut_scalar(0.5 * x) + 1.0);
}

void sigmoid_16(const phc_acc_t in[16], phc_acc_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        out[i] = sigmoid_scalar(in[i]);
    }
}

void tanh_16(const phc_acc_t in[16], phc_acc_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        out[i] = tanh_lut_scalar(in[i]);
    }
}

void silu_16(const phc_acc_t in[16], phc_acc_t out[16]) {
    for (int i = 0; i < 16; ++i) {
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_16_16_fixed(const phc_fixed_t in[16], const double w[16][16], const double b[16], phc_acc_t out[16]) {
    for (int j = 0; j < 16; ++j) {
        phc_acc_t acc = b[j];
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_acc(const phc_acc_t in[16], const double w[16][16], const double b[16], phc_acc_t out[16]) {
    for (int j = 0; j < 16; ++j) {
        phc_acc_t acc = b[j];
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_fixed_no_bias(const phc_fixed_t in[16], const double w[16][16], phc_acc_t out[16]) {
    for (int j = 0; j < 16; ++j) {
        phc_acc_t acc = 0.0;
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_2_acc(const phc_acc_t in[16], const double w[2][16], const double b[2], phc_acc_t out[2]) {
    for (int j = 0; j < 2; ++j) {
        phc_acc_t acc = b[j];
        for (int i = 0; i < 16; ++i) {
            acc += w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void layernorm_16(const phc_acc_t in[16], const double gamma[16], const double beta[16], phc_acc_t out[16]) {
    phc_acc_t mean = 0.0;
    for (int i = 0; i < 16; ++i) {
        mean += in[i];
    }
    mean = mean / 16.0;
    phc_var_t var = 0.0;
    for (int i = 0; i < 16; ++i) {
        phc_acc_t d = in[i] - mean;
        var += d * d;
    }
    var = var / 16.0;
    phc_acc_t inv_std = 1.0 / std::sqrt(var + LN_EPS);
    for (int i = 0; i < 16; ++i) {
        out[i] = gamma[i] * (in[i] - mean) * inv_std + beta[i];
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
        state_temporal[i] = DECAY[i] * state_prev[i] + gate_t[i] * candidate_t[i];
    }

    linear_16_16_acc(encoded, GAIN_FROM_ENCODED_W, GAIN_FROM_ENCODED_B, gain_encoded);
    linear_16_16_fixed_no_bias(state_prev, GAIN_FROM_PREV_STATE_W, gain_prev_state);
    for (int i = 0; i < 16; ++i) {
        gain_pre[i] = gain_encoded[i] + gain_prev_state[i];
    }
    sigmoid_16(gain_pre, gain_t);

    linear_16_16_acc(state_temporal, SYNAPSE_W, SYNAPSE_B, message_pre);
    silu_16(message_pre, message);

    for (int i = 0; i < 16; ++i) {
        loop_input[i] = state_temporal[i] + ALPHA_SCALAR[0] * gain_t[i] * message[i];
    }
    layernorm_16(loop_input, LOOP_NORM_GAMMA, LOOP_NORM_BETA, z);

    linear_16_2_acc(z, READOUT_W, READOUT_B, readout_acc);

    for (int i = 0; i < 2; ++i) {
        readout[i] = readout_acc[i];
    }
    for (int i = 0; i < 16; ++i) {
        state_next[i] = z[i];
    }
}
