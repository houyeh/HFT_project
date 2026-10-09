// Phase 06U V2 generated variant.
//
// Source model: Phase 05E PHC hidden8 checkpoint selected by median val PnL.
// Scheduling target: hidden8, unroll8, 10 ns clock.
// Interface, arithmetic type family, and recurrent step contract follow 06F.

#include "phc_step_hls.hpp"
#include "phc_weights.hpp"

#include <hls_math.h>

namespace {

static const int UNROLL_FACTOR = 8;
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

phc_acc_t tanh_scalar(phc_acc_t x) {
#pragma HLS INLINE
    return tanh_lut_scalar(x);
}

void sigmoid_hidden(const phc_acc_t in[PHC_HIDDEN_DIM], phc_acc_t out[PHC_HIDDEN_DIM]) {
#pragma HLS INLINE
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        out[i] = sigmoid_scalar(in[i]);
    }
}

void tanh_hidden(const phc_acc_t in[PHC_HIDDEN_DIM], phc_acc_t out[PHC_HIDDEN_DIM]) {
#pragma HLS INLINE
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        out[i] = tanh_scalar(in[i]);
    }
}

void silu_hidden(const phc_acc_t in[PHC_HIDDEN_DIM], phc_acc_t out[PHC_HIDDEN_DIM]) {
#pragma HLS INLINE
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        out[i] = in[i] * sigmoid_scalar(in[i]);
    }
}

void linear_feature_hidden_fixed(
    const phc_fixed_t in[PHC_FEATURE_DIM],
    const phc_fixed_t w[PHC_HIDDEN_DIM][PHC_FEATURE_DIM],
    const phc_fixed_t b[PHC_HIDDEN_DIM],
    phc_acc_t out[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=UNROLL_FACTOR dim=2
    for (int j = 0; j < PHC_HIDDEN_DIM; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < PHC_FEATURE_DIM; ++i) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=UNROLL_FACTOR
            acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_hidden_hidden_acc(
    const phc_acc_t in[PHC_HIDDEN_DIM],
    const phc_fixed_t w[PHC_HIDDEN_DIM][PHC_HIDDEN_DIM],
    const phc_fixed_t b[PHC_HIDDEN_DIM],
    phc_acc_t out[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=UNROLL_FACTOR dim=2
    for (int j = 0; j < PHC_HIDDEN_DIM; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=UNROLL_FACTOR
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void linear_hidden_hidden_fixed_no_bias(
    const phc_fixed_t in[PHC_HIDDEN_DIM],
    const phc_fixed_t w[PHC_HIDDEN_DIM][PHC_HIDDEN_DIM],
    phc_acc_t out[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=UNROLL_FACTOR dim=2
    for (int j = 0; j < PHC_HIDDEN_DIM; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = 0;
        for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=UNROLL_FACTOR
            acc += (phc_acc_t)w[j][i] * (phc_acc_t)in[i];
        }
        out[j] = acc;
    }
}

void linear_hidden_readout_acc(
    const phc_acc_t in[PHC_HIDDEN_DIM],
    const phc_fixed_t w[PHC_READOUT_DIM][PHC_HIDDEN_DIM],
    const phc_fixed_t b[PHC_READOUT_DIM],
    phc_acc_t out[PHC_READOUT_DIM]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=UNROLL_FACTOR dim=2
    for (int j = 0; j < PHC_READOUT_DIM; ++j) {
#pragma HLS PIPELINE off
        phc_acc_t acc = (phc_acc_t)b[j];
        for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=UNROLL_FACTOR
            acc += (phc_acc_t)w[j][i] * in[i];
        }
        out[j] = acc;
    }
}

void layernorm_hidden(
    const phc_acc_t in[PHC_HIDDEN_DIM],
    const phc_fixed_t gamma[PHC_HIDDEN_DIM],
    const phc_fixed_t beta[PHC_HIDDEN_DIM],
    phc_acc_t out[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE
    phc_acc_t mean = 0;
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        mean += in[i];
    }
    mean = mean / (phc_acc_t)PHC_HIDDEN_DIM;

    phc_var_t var = 0;
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        phc_acc_t d = in[i] - mean;
        var += (phc_var_t)d * (phc_var_t)d;
    }
    var = var / (phc_var_t)PHC_HIDDEN_DIM;
    phc_acc_t inv_std = (phc_acc_t)1.0 / (phc_acc_t)hls::sqrt(var + (phc_var_t)LN_EPS);

    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        out[i] = (phc_acc_t)gamma[i] * (in[i] - mean) * inv_std + (phc_acc_t)beta[i];
    }
}

void gate_branch(const phc_acc_t encoded[PHC_HIDDEN_DIM], phc_acc_t gate_t[PHC_HIDDEN_DIM]) {
#pragma HLS INLINE off
    phc_acc_t gate_pre[PHC_HIDDEN_DIM];
    linear_hidden_hidden_acc(encoded, phc_step_weights::GATE_W, phc_step_weights::GATE_B, gate_pre);
    sigmoid_hidden(gate_pre, gate_t);
}

void candidate_branch(const phc_acc_t encoded[PHC_HIDDEN_DIM], phc_acc_t candidate_t[PHC_HIDDEN_DIM]) {
#pragma HLS INLINE off
    phc_acc_t candidate_pre[PHC_HIDDEN_DIM];
    linear_hidden_hidden_acc(encoded, phc_step_weights::CANDIDATE_W, phc_step_weights::CANDIDATE_B, candidate_pre);
    tanh_hidden(candidate_pre, candidate_t);
}

void fork_encoded_for_gate_candidate(
    const phc_acc_t encoded[PHC_HIDDEN_DIM],
    phc_acc_t encoded_for_gate[PHC_HIDDEN_DIM],
    phc_acc_t encoded_for_candidate[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE off
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        encoded_for_gate[i] = encoded[i];
        encoded_for_candidate[i] = encoded[i];
    }
}

void gate_candidate_dataflow(
    const phc_acc_t encoded[PHC_HIDDEN_DIM],
    phc_acc_t gate_t[PHC_HIDDEN_DIM],
    phc_acc_t candidate_t[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    phc_acc_t encoded_for_gate[PHC_HIDDEN_DIM];
    phc_acc_t encoded_for_candidate[PHC_HIDDEN_DIM];

    fork_encoded_for_gate_candidate(encoded, encoded_for_gate, encoded_for_candidate);
    gate_branch(encoded_for_gate, gate_t);
    candidate_branch(encoded_for_candidate, candidate_t);
}

void fork_inputs_for_main_gain(
    const phc_acc_t encoded[PHC_HIDDEN_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_acc_t encoded_for_main[PHC_HIDDEN_DIM],
    phc_acc_t encoded_for_gain[PHC_HIDDEN_DIM],
    phc_fixed_t state_prev_for_main[PHC_HIDDEN_DIM],
    phc_fixed_t state_prev_for_gain[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE off
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        encoded_for_main[i] = encoded[i];
        encoded_for_gain[i] = encoded[i];
        state_prev_for_main[i] = state_prev[i];
        state_prev_for_gain[i] = state_prev[i];
    }
}

void main_chain_fn(
    const phc_acc_t encoded[PHC_HIDDEN_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_acc_t message_out[PHC_HIDDEN_DIM],
    phc_acc_t state_temporal_out[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE off
    phc_acc_t gate_t[PHC_HIDDEN_DIM];
    phc_acc_t candidate_t[PHC_HIDDEN_DIM];
    phc_acc_t message_pre[PHC_HIDDEN_DIM];

    gate_candidate_dataflow(encoded, gate_t, candidate_t);

    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        state_temporal_out[i] =
            (phc_acc_t)phc_step_weights::DECAY[i] * (phc_acc_t)state_prev[i] + gate_t[i] * candidate_t[i];
    }

    linear_hidden_hidden_acc(state_temporal_out, phc_step_weights::SYNAPSE_W, phc_step_weights::SYNAPSE_B, message_pre);
    silu_hidden(message_pre, message_out);
}

void gain_chain_fn(
    const phc_acc_t encoded[PHC_HIDDEN_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_acc_t gain_t_out[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE off
    phc_acc_t gain_encoded[PHC_HIDDEN_DIM];
    phc_acc_t gain_prev_state[PHC_HIDDEN_DIM];
    phc_acc_t gain_pre[PHC_HIDDEN_DIM];

    linear_hidden_hidden_acc(encoded, phc_step_weights::GAIN_FROM_ENCODED_W, phc_step_weights::GAIN_FROM_ENCODED_B, gain_encoded);
    linear_hidden_hidden_fixed_no_bias(state_prev, phc_step_weights::GAIN_FROM_PREV_STATE_W, gain_prev_state);
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        gain_pre[i] = gain_encoded[i] + gain_prev_state[i];
    }
    sigmoid_hidden(gain_pre, gain_t_out);
}

void step_dataflow_fn(
    const phc_acc_t encoded[PHC_HIDDEN_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_acc_t message_out[PHC_HIDDEN_DIM],
    phc_acc_t state_temporal_out[PHC_HIDDEN_DIM],
    phc_acc_t gain_t_out[PHC_HIDDEN_DIM]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    phc_acc_t encoded_for_main[PHC_HIDDEN_DIM];
    phc_acc_t encoded_for_gain[PHC_HIDDEN_DIM];
    phc_fixed_t state_prev_for_main[PHC_HIDDEN_DIM];
    phc_fixed_t state_prev_for_gain[PHC_HIDDEN_DIM];

    fork_inputs_for_main_gain(
        encoded,
        state_prev,
        encoded_for_main,
        encoded_for_gain,
        state_prev_for_main,
        state_prev_for_gain
    );
    main_chain_fn(encoded_for_main, state_prev_for_main, message_out, state_temporal_out);
    gain_chain_fn(encoded_for_gain, state_prev_for_gain, gain_t_out);
}

}  // namespace

void phc_step(
    const phc_fixed_t x_t[PHC_FEATURE_DIM],
    const phc_fixed_t state_prev[PHC_HIDDEN_DIM],
    phc_fixed_t readout[PHC_READOUT_DIM],
    phc_fixed_t state_next[PHC_HIDDEN_DIM]
) {

    phc_acc_t enc_linear_out[PHC_HIDDEN_DIM];
    phc_acc_t encoded_pre[PHC_HIDDEN_DIM];
    phc_acc_t encoded[PHC_HIDDEN_DIM];
    phc_acc_t state_temporal[PHC_HIDDEN_DIM];
    phc_acc_t gain_t[PHC_HIDDEN_DIM];
    phc_acc_t message[PHC_HIDDEN_DIM];
    phc_acc_t loop_input[PHC_HIDDEN_DIM];
    phc_acc_t z[PHC_HIDDEN_DIM];
    phc_acc_t readout_acc[PHC_READOUT_DIM];

    using namespace phc_step_weights;

    linear_feature_hidden_fixed(x_t, ENC_LINEAR_W, ENC_LINEAR_B, enc_linear_out);
    layernorm_hidden(enc_linear_out, ENC_NORM_GAMMA, ENC_NORM_BETA, encoded_pre);
    silu_hidden(encoded_pre, encoded);

    step_dataflow_fn(encoded, state_prev, message, state_temporal, gain_t);

    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        loop_input[i] = state_temporal[i] + (phc_acc_t)ALPHA_SCALAR[0] * gain_t[i] * message[i];
    }
    layernorm_hidden(loop_input, LOOP_NORM_GAMMA, LOOP_NORM_BETA, z);

    linear_hidden_readout_acc(z, READOUT_W, READOUT_B, readout_acc);

    for (int i = 0; i < PHC_READOUT_DIM; ++i) {
#pragma HLS PIPELINE off
        readout[i] = (phc_fixed_t)readout_acc[i];
    }
    for (int i = 0; i < PHC_HIDDEN_DIM; ++i) {
#pragma HLS PIPELINE off
        state_next[i] = (phc_fixed_t)z[i];
    }
}
