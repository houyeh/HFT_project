#include "phc_diag_step_hls.hpp"
#include "phc_diag_weights.hpp"

#include <hls_math.h>

namespace {

static const phc_acc_t LN_EPS = (phc_acc_t)1.0e-5;

phc_acc_t tanh_lut_scalar(phc_acc_t x) {
#pragma HLS INLINE
    static const phc_acc_t TANH_LUT[65] = {
        (phc_acc_t)-0.9993292997, (phc_acc_t)-0.9991388858, (phc_acc_t)-0.9988944427, (phc_acc_t)-0.9985806592,
        (phc_acc_t)-0.9981778976, (phc_acc_t)-0.9976609795, (phc_acc_t)-0.9969976355, (phc_acc_t)-0.9961465307,
        (phc_acc_t)-0.9950547537, (phc_acc_t)-0.9936546343, (phc_acc_t)-0.9918597246, (phc_acc_t)-0.9895597486,
        (phc_acc_t)-0.9866142982, (phc_acc_t)-0.9828450292, (phc_acc_t)-0.9780261147, (phc_acc_t)-0.9718727459,
        (phc_acc_t)-0.9640275801, (phc_acc_t)-0.9540452602, (phc_acc_t)-0.9413755385, (phc_acc_t)-0.9253462253,
        (phc_acc_t)-0.9051482536, (phc_acc_t)-0.8798266997, (phc_acc_t)-0.8482836400, (phc_acc_t)-0.8093010702,
        (phc_acc_t)-0.7615941560, (phc_acc_t)-0.7039056039, (phc_acc_t)-0.6351489524, (phc_acc_t)-0.5545997223,
        (phc_acc_t)-0.4621171573, (phc_acc_t)-0.3583573984, (phc_acc_t)-0.2449186624, (phc_acc_t)-0.1243530018,
        (phc_acc_t)0.0000000000,  (phc_acc_t)0.1243530018,  (phc_acc_t)0.2449186624,  (phc_acc_t)0.3583573984,
        (phc_acc_t)0.4621171573,  (phc_acc_t)0.5545997223,  (phc_acc_t)0.6351489524,  (phc_acc_t)0.7039056039,
        (phc_acc_t)0.7615941560,  (phc_acc_t)0.8093010702,  (phc_acc_t)0.8482836400,  (phc_acc_t)0.8798266997,
        (phc_acc_t)0.9051482536,  (phc_acc_t)0.9253462253,  (phc_acc_t)0.9413755385,  (phc_acc_t)0.9540452602,
        (phc_acc_t)0.9640275801,  (phc_acc_t)0.9718727459,  (phc_acc_t)0.9780261147,  (phc_acc_t)0.9828450292,
        (phc_acc_t)0.9866142982,  (phc_acc_t)0.9895597486,  (phc_acc_t)0.9918597246,  (phc_acc_t)0.9936546343,
        (phc_acc_t)0.9950547537,  (phc_acc_t)0.9961465307,  (phc_acc_t)0.9969976355,  (phc_acc_t)0.9976609795,
        (phc_acc_t)0.9981778976,  (phc_acc_t)0.9985806592,  (phc_acc_t)0.9988944427,  (phc_acc_t)0.9991388858,
        (phc_acc_t)0.9993292997
    };
#pragma HLS ARRAY_PARTITION variable=TANH_LUT complete dim=1

    if (x <= (phc_acc_t)-4.0) {
        return (phc_acc_t)-1.0;
    }
    if (x >= (phc_acc_t)4.0) {
        return (phc_acc_t)1.0;
    }

    phc_acc_t scaled = (x + (phc_acc_t)4.0) * (phc_acc_t)8.0;
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
#pragma HLS INLINE
    return (phc_acc_t)0.5 * (tanh_lut_scalar((phc_acc_t)0.5 * x) + (phc_acc_t)1.0);
}

void silu_16(const phc_acc_t in[16], phc_acc_t out[16]) {
#pragma HLS INLINE
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_16_16_fixed(
    const phc_fixed_t in[16],
    const phc_fixed_t w[16][16],
    const phc_fixed_t b[16],
    phc_acc_t out[16]
) {
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

void linear_16_2_acc(
    const phc_acc_t in[16],
    const phc_fixed_t w[2][16],
    const phc_fixed_t b[2],
    phc_acc_t out[2]
) {
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

void layernorm_16(
    const phc_acc_t in[16],
    const phc_fixed_t gamma[16],
    const phc_fixed_t beta[16],
    phc_acc_t out[16]
) {
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

void encoder_stage(const phc_fixed_t x_t[16], phc_acc_t encoded[16]) {
#pragma HLS INLINE
    phc_acc_t enc_linear_out[16];
    phc_acc_t encoded_pre[16];
    using namespace phc_diag_weights;
    linear_16_16_fixed(x_t, ENC_LINEAR_W, ENC_LINEAR_B, enc_linear_out);
    layernorm_16(enc_linear_out, ENC_NORM_GAMMA, ENC_NORM_BETA, encoded_pre);
    silu_16(encoded_pre, encoded);
}

void recurrence_stage(
    const phc_fixed_t state_prev[16],
    const phc_acc_t encoded[16],
    phc_acc_t state_acc[16]
) {
#pragma HLS INLINE
    using namespace phc_diag_weights;
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE II=1
        state_acc[i] = (phc_acc_t)DIAG_DECAY[i] * (phc_acc_t)state_prev[i]
                     + (phc_acc_t)DIAG_GAIN[i] * encoded[i];
    }
}

void readout_stage(const phc_acc_t state_acc[16], phc_acc_t readout_acc[2]) {
#pragma HLS INLINE
    using namespace phc_diag_weights;
    linear_16_2_acc(state_acc, READOUT_W, READOUT_B, readout_acc);
}

}  // namespace

void phc_diag_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
) {
    phc_acc_t encoded[16];
    phc_acc_t state_acc[16];
    phc_acc_t readout_acc[2];

    encoder_stage(x_t, encoded);
    recurrence_stage(state_prev, encoded, state_acc);
    readout_stage(state_acc, readout_acc);

    for (int i = 0; i < 2; ++i) {
#pragma HLS PIPELINE off
        readout[i] = (phc_fixed_t)readout_acc[i];
    }
    for (int i = 0; i < 16; ++i) {
#pragma HLS PIPELINE off
        state_next[i] = (phc_fixed_t)state_acc[i];
    }
}
