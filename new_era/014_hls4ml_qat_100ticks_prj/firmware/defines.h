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
typedef nnet::array<ap_fixed<8,2>, 80*1> input_t;
typedef nnet::array<ap_fixed<8,2>, 8*1> layer30_t;
typedef nnet::array<ap_fixed<8,2>, 8*1> layer2_t;
typedef ap_fixed<8,2> model_default_t;
typedef nnet::array<ap_fixed<22,10>, 12*1> tcn_network_0_conv1_conv_result_t;
typedef ap_fixed<8,2> tcn_network_0_conv1_conv_weight_t;
typedef ap_fixed<8,2> tcn_network_0_conv1_conv_bias_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer4_t;
typedef ap_fixed<18,8> tcn_network_0_relu_table_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer5_t;
typedef nnet::array<ap_fixed<23,11>, 12*1> tcn_network_0_conv2_conv_result_t;
typedef ap_fixed<8,2> tcn_network_0_conv2_conv_weight_t;
typedef ap_fixed<8,2> tcn_network_0_conv2_conv_bias_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer7_t;
typedef ap_fixed<18,8> tcn_network_0_relu_1_table_t;
typedef nnet::array<ap_fixed<20,8>, 12*1> tcn_network_0_downsample_result_t;
typedef ap_fixed<8,2> tcn_network_0_downsample_weight_t;
typedef ap_fixed<8,2> tcn_network_0_downsample_bias_t;
typedef nnet::array<ap_fixed<21,9>, 12*1> add_result_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer10_t;
typedef ap_fixed<18,8> tcn_network_0_relu_2_table_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer11_t;
typedef nnet::array<ap_fixed<23,11>, 12*1> tcn_network_1_conv1_conv_result_t;
typedef ap_fixed<8,2> tcn_network_1_conv1_conv_weight_t;
typedef ap_fixed<8,2> tcn_network_1_conv1_conv_bias_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer13_t;
typedef ap_fixed<18,8> tcn_network_1_relu_table_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer14_t;
typedef nnet::array<ap_fixed<23,11>, 12*1> tcn_network_1_conv2_conv_result_t;
typedef ap_fixed<8,2> tcn_network_1_conv2_conv_weight_t;
typedef ap_fixed<8,2> tcn_network_1_conv2_conv_bias_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer16_t;
typedef ap_fixed<18,8> tcn_network_1_relu_1_table_t;
typedef nnet::array<ap_fixed<9,3>, 12*1> add_1_result_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer18_t;
typedef ap_fixed<18,8> tcn_network_1_relu_2_table_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer19_t;
typedef nnet::array<ap_fixed<23,11>, 12*1> tcn_network_2_conv1_conv_result_t;
typedef ap_fixed<8,2> tcn_network_2_conv1_conv_weight_t;
typedef ap_fixed<8,2> tcn_network_2_conv1_conv_bias_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer21_t;
typedef ap_fixed<18,8> tcn_network_2_relu_table_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer22_t;
typedef nnet::array<ap_fixed<23,11>, 12*1> tcn_network_2_conv2_conv_result_t;
typedef ap_fixed<8,2> tcn_network_2_conv2_conv_weight_t;
typedef ap_fixed<8,2> tcn_network_2_conv2_conv_bias_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer24_t;
typedef ap_fixed<18,8> tcn_network_2_relu_1_table_t;
typedef nnet::array<ap_fixed<9,3>, 12*1> add_2_result_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer26_t;
typedef ap_fixed<18,8> tcn_network_2_relu_2_table_t;
typedef nnet::array<ap_fixed<8,2>, 12*1> layer27_t;
typedef nnet::array<ap_fixed<8,2>, 1*1> layer31_t;
typedef nnet::array<ap_fixed<21,9>, 3*1> result_t;
typedef ap_fixed<8,2> fc_weight_t;
typedef ap_fixed<8,2> fc_bias_t;
typedef ap_uint<1> layer29_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
