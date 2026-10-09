#include "fi2010_phc_hls.hpp"
#include "fi2010_phc_weights.hpp"

#include <hls_math.h>

namespace {

static const fi_acc_t LN_EPS = (fi_acc_t)1.0e-5;

fi_acc_t tanh_lut_scalar(fi_acc_t x) {
#pragma HLS INLINE
    static const fi_acc_t TANH_LUT[65] = {
        (fi_acc_t)-0.9993292997, (fi_acc_t)-0.9991388858, (fi_acc_t)-0.9988944427, (fi_acc_t)-0.9985806592,
        (fi_acc_t)-0.9981778976, (fi_acc_t)-0.9976609795, (fi_acc_t)-0.9969976355, (fi_acc_t)-0.9961465307,
        (fi_acc_t)-0.9950547537, (fi_acc_t)-0.9936546343, (fi_acc_t)-0.9918597246, (fi_acc_t)-0.9895597486,
        (fi_acc_t)-0.9866142982, (fi_acc_t)-0.9828450292, (fi_acc_t)-0.9780261147, (fi_acc_t)-0.9718727459,
        (fi_acc_t)-0.9640275801, (fi_acc_t)-0.9540452602, (fi_acc_t)-0.9413755385, (fi_acc_t)-0.9253462253,
        (fi_acc_t)-0.9051482536, (fi_acc_t)-0.8798266997, (fi_acc_t)-0.8482836400, (fi_acc_t)-0.8093010702,
        (fi_acc_t)-0.7615941560, (fi_acc_t)-0.7039056039, (fi_acc_t)-0.6351489524, (fi_acc_t)-0.5545997223,
        (fi_acc_t)-0.4621171573, (fi_acc_t)-0.3583573984, (fi_acc_t)-0.2449186624, (fi_acc_t)-0.1243530018,
        (fi_acc_t)0.0000000000,  (fi_acc_t)0.1243530018,  (fi_acc_t)0.2449186624,  (fi_acc_t)0.3583573984,
        (fi_acc_t)0.4621171573,  (fi_acc_t)0.5545997223,  (fi_acc_t)0.6351489524,  (fi_acc_t)0.7039056039,
        (fi_acc_t)0.7615941560,  (fi_acc_t)0.8093010702,  (fi_acc_t)0.8482836400,  (fi_acc_t)0.8798266997,
        (fi_acc_t)0.9051482536,  (fi_acc_t)0.9253462253,  (fi_acc_t)0.9413755385,  (fi_acc_t)0.9540452602,
        (fi_acc_t)0.9640275801,  (fi_acc_t)0.9718727459,  (fi_acc_t)0.9780261147,  (fi_acc_t)0.9828450292,
        (fi_acc_t)0.9866142982,  (fi_acc_t)0.9895597486,  (fi_acc_t)0.9918597246,  (fi_acc_t)0.9936546343,
        (fi_acc_t)0.9950547537,  (fi_acc_t)0.9961465307,  (fi_acc_t)0.9969976355,  (fi_acc_t)0.9976609795,
        (fi_acc_t)0.9981778976,  (fi_acc_t)0.9985806592,  (fi_acc_t)0.9988944427,  (fi_acc_t)0.9991388858,
        (fi_acc_t)0.9993292997
    };
#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1
    if (x <= (fi_acc_t)-4.0) {
        return (fi_acc_t)-1.0;
    }
    if (x >= (fi_acc_t)4.0) {
        return (fi_acc_t)1.0;
    }
    fi_acc_t scaled = (x + (fi_acc_t)4.0) * (fi_acc_t)8.0;
    int idx = (int)scaled;
    if (idx < 0) idx = 0;
    if (idx > 63) idx = 63;
    fi_acc_t frac = scaled - (fi_acc_t)idx;
    fi_acc_t y0 = TANH_LUT[idx];
    fi_acc_t y1 = TANH_LUT[idx + 1];
    return y0 + frac * (y1 - y0);
}

fi_acc_t sigmoid_scalar(fi_acc_t x) {
#pragma HLS INLINE
    return (fi_acc_t)0.5 * (tanh_lut_scalar((fi_acc_t)0.5 * x) + (fi_acc_t)1.0);
}

void sigmoid_16(const fi_acc_t in[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = sigmoid_scalar(in[i]);
    }
}

void tanh_16(const fi_acc_t in[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = tanh_lut_scalar(in[i]);
    }
}

void silu_16(const fi_acc_t in[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_40_16_fixed(const fi_fixed_t in[40], const fi_fixed_t w[16][40], const fi_fixed_t b[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = (fi_acc_t)b[j];
        for (int i = 0; i < 40; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * (fi_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_acc(const fi_acc_t in[16], const fi_fixed_t w[16][16], const fi_fixed_t b[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = (fi_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_16_16_fixed_no_bias(const fi_fixed_t in[16], const fi_fixed_t w[16][16], fi_acc_t out[16]) {
#pragma HLS INLINE
    for (int j = 0; j < 16; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = 0;
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * (fi_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_16_3_acc(const fi_acc_t in[16], const fi_fixed_t w[3][16], const fi_fixed_t b[3], fi_acc_t out[3]) {
#pragma HLS INLINE
    for (int j = 0; j < 3; ++j) {
#pragma HLS PIPELINE off
        fi_acc_t acc = (fi_acc_t)b[j];
        for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
            acc += (fi_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void layernorm_16(const fi_acc_t in[16], const fi_fixed_t gamma[16], const fi_fixed_t beta[16], fi_acc_t out[16]) {
#pragma HLS INLINE
    fi_acc_t mean = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        mean += in[i];
    }
    mean = mean / (fi_acc_t)16.0;

    fi_var_t var = 0;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        fi_acc_t d = in[i] - mean;
        var += (fi_var_t)d * (fi_var_t)d;
    }
    var = var / (fi_var_t)16.0;
    fi_acc_t inv_std = (fi_acc_t)1.0 / (fi_acc_t)hls::sqrt(var + (fi_var_t)LN_EPS);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = (fi_acc_t)gamma[i] * (in[i] - mean) * inv_std + (fi_acc_t)beta[i];
    }
}

}  // namespace

void fi2010_phc_step(
    const fi_fixed_t x_t[FI2010_FEATURE_DIM],
    const fi_fixed_t state_prev[FI2010_HIDDEN_DIM],
    fi_fixed_t logits[FI2010_READOUT_DIM],
    fi_fixed_t state_next[FI2010_HIDDEN_DIM]
) {
    fi_acc_t enc_linear_out[16];
    fi_acc_t encoded_pre[16];
    fi_acc_t encoded[16];
    fi_acc_t gate_pre[16];
    fi_acc_t gate_t[16];
    fi_acc_t candidate_pre[16];
    fi_acc_t candidate_t[16];
    fi_acc_t state_temporal[16];
    fi_acc_t gain_encoded[16];
    fi_acc_t gain_prev_state[16];
    fi_acc_t gain_pre[16];
    fi_acc_t gain_t[16];
    fi_acc_t message_pre[16];
    fi_acc_t message[16];
    fi_acc_t loop_input[16];
    fi_acc_t z[16];
    fi_acc_t logits_acc[3];

    using namespace fi2010_phc_weights;

    linear_40_16_fixed(x_t, ENC_LINEAR_W, ENC_LINEAR_B, enc_linear_out);
    layernorm_16(enc_linear_out, ENC_NORM_GAMMA, ENC_NORM_BETA, encoded_pre);
    silu_16(encoded_pre, encoded);

    linear_16_16_acc(encoded, GATE_W, GATE_B, gate_pre);
    sigmoid_16(gate_pre, gate_t);

    linear_16_16_acc(encoded, CANDIDATE_W, CANDIDATE_B, candidate_pre);
    tanh_16(candidate_pre, candidate_t);

    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_temporal[i] = (fi_acc_t)DECAY[i] * (fi_acc_t)state_prev[i] + gate_t[i] * candidate_t[i];
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
        loop_input[i] = state_temporal[i] + (fi_acc_t)ALPHA_SCALAR[0] * gain_t[i] * message[i];
    }
    layernorm_16(loop_input, LOOP_NORM_GAMMA, LOOP_NORM_BETA, z);

    linear_16_3_acc(z, READOUT_W, READOUT_B, logits_acc);

    for (int i = 0; i < 3; ++i) {
#pragma HLS PIPELINE off
        logits[i] = (fi_fixed_t)logits_acc[i];
    }
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_next[i] = (fi_fixed_t)z[i];
    }
}
