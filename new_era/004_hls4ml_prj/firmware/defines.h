#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> fc1_accum_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> fc1_weight_t;
typedef ap_fixed<16,6> fc1_bias_t;
typedef ap_uint<1> layer3_index;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> bn1_scale_t;
typedef ap_fixed<16,6> bn1_bias_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<16,6> fc2_accum_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> fc2_weight_t;
typedef ap_fixed<16,6> fc2_bias_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> bn2_scale_t;
typedef ap_fixed<16,6> bn2_bias_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_fixed<16,6> output_probabilities_accum_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> output_probabilities_weight_t;
typedef ap_fixed<16,6> output_probabilities_bias_t;
typedef ap_uint<1> layer9_index;
typedef ap_fixed<16,6> output_probabilities_softmax_accum_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> output_probabilities_softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_probabilities_softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_probabilities_softmax_inv_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_probabilities_softmax_inv_inp_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
