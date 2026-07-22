#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer29_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x,layer29_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<tcn_network_0_conv1_conv_weight_t, 384>(w3, "w3.txt");
        nnet::load_weights_from_txt<tcn_network_0_conv1_conv_bias_t, 16>(b3, "b3.txt");
        nnet::load_weights_from_txt<tcn_network_0_conv2_conv_weight_t, 768>(w6, "w6.txt");
        nnet::load_weights_from_txt<tcn_network_0_conv2_conv_bias_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<tcn_network_0_downsample_weight_t, 128>(w8, "w8.txt");
        nnet::load_weights_from_txt<tcn_network_0_downsample_bias_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<tcn_network_1_conv1_conv_weight_t, 768>(w12, "w12.txt");
        nnet::load_weights_from_txt<tcn_network_1_conv1_conv_bias_t, 16>(b12, "b12.txt");
        nnet::load_weights_from_txt<tcn_network_1_conv2_conv_weight_t, 768>(w15, "w15.txt");
        nnet::load_weights_from_txt<tcn_network_1_conv2_conv_bias_t, 16>(b15, "b15.txt");
        nnet::load_weights_from_txt<tcn_network_2_conv1_conv_weight_t, 768>(w20, "w20.txt");
        nnet::load_weights_from_txt<tcn_network_2_conv1_conv_bias_t, 16>(b20, "b20.txt");
        nnet::load_weights_from_txt<tcn_network_2_conv2_conv_weight_t, 768>(w23, "w23.txt");
        nnet::load_weights_from_txt<tcn_network_2_conv2_conv_bias_t, 16>(b23, "b23.txt");
        nnet::load_weights_from_txt<fc_weight_t, 48>(w29, "w29.txt");
        nnet::load_weights_from_txt<fc_bias_t, 3>(b29, "b29.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<input_t> layer32_cpy1("layer32_cpy1");
    #pragma HLS STREAM variable=layer32_cpy1 depth=8

    hls::stream<input_t> layer32_cpy2("layer32_cpy2");
    #pragma HLS STREAM variable=layer32_cpy2 depth=8

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=100

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=102

    hls::stream<tcn_network_0_conv1_conv_result_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=100

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=100

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=102

    hls::stream<tcn_network_0_conv2_conv_result_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=100

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=100

    hls::stream<tcn_network_0_downsample_result_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=100

    hls::stream<add_result_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=100

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=100

    hls::stream<layer10_t> layer33_cpy1("layer33_cpy1");
    #pragma HLS STREAM variable=layer33_cpy1 depth=100

    hls::stream<layer10_t> layer33_cpy2("layer33_cpy2");
    #pragma HLS STREAM variable=layer33_cpy2 depth=100

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=104

    hls::stream<tcn_network_1_conv1_conv_result_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=100

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=100

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=104

    hls::stream<tcn_network_1_conv2_conv_result_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=100

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=100

    hls::stream<add_1_result_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=100

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=100

    hls::stream<layer18_t> layer34_cpy1("layer34_cpy1");
    #pragma HLS STREAM variable=layer34_cpy1 depth=100

    hls::stream<layer18_t> layer34_cpy2("layer34_cpy2");
    #pragma HLS STREAM variable=layer34_cpy2 depth=100

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=108

    hls::stream<tcn_network_2_conv1_conv_result_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=100

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=100

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=108

    hls::stream<tcn_network_2_conv2_conv_result_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=100

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=100

    hls::stream<add_2_result_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=100

    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=100

    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=1

    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=16

    auto& layer28_out = layer31_out;
    nnet::clone_stream<input_t, input_t, 800>(x, layer32_cpy1, layer32_cpy2); // clone_x

    nnet::transpose<input_t, layer30_t, config30>(layer32_cpy1, layer30_out); // transpose_input_for_x

    nnet::zeropad1d_cl<layer30_t, layer2_t, config2>(layer30_out, layer2_out); // tcn_network_0_conv1_pad

    nnet::conv_1d_cl<layer2_t, tcn_network_0_conv1_conv_result_t, config3>(layer2_out, layer3_out, w3, b3); // tcn_network_0_conv1_conv

    nnet::relu<tcn_network_0_conv1_conv_result_t, layer4_t, relu_config4>(layer3_out, layer4_out); // tcn_network_0_relu

    nnet::zeropad1d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out); // tcn_network_0_conv2_pad

    nnet::conv_1d_cl<layer5_t, tcn_network_0_conv2_conv_result_t, config6>(layer5_out, layer6_out, w6, b6); // tcn_network_0_conv2_conv

    nnet::relu<tcn_network_0_conv2_conv_result_t, layer7_t, relu_config7>(layer6_out, layer7_out); // tcn_network_0_relu_1

    hls::stream<layer30_t> layer32_cpy2_transposed("layer32_cpy2_transposed");
    #pragma HLS STREAM variable=layer32_cpy2_transposed depth=8
    nnet::transpose<input_t, layer30_t, config30>(layer32_cpy2, layer32_cpy2_transposed);
    nnet::pointwise_conv_1d_cl<layer30_t, tcn_network_0_downsample_result_t, config35>(layer32_cpy2_transposed, layer8_out, w8, b8); // tcn_network_0_downsample

    nnet::add<layer7_t, tcn_network_0_downsample_result_t, add_result_t, config9>(layer7_out, layer8_out, layer9_out); // add

    nnet::relu<add_result_t, layer10_t, relu_config10>(layer9_out, layer10_out); // tcn_network_0_relu_2

    nnet::clone_stream<layer10_t, layer10_t, 1600>(layer10_out, layer33_cpy1, layer33_cpy2); // clone_tcn_network_0_relu_2

    nnet::zeropad1d_cl<layer10_t, layer11_t, config11>(layer33_cpy1, layer11_out); // tcn_network_1_conv1_pad

    nnet::conv_1d_cl<layer11_t, tcn_network_1_conv1_conv_result_t, config12>(layer11_out, layer12_out, w12, b12); // tcn_network_1_conv1_conv

    nnet::relu<tcn_network_1_conv1_conv_result_t, layer13_t, relu_config13>(layer12_out, layer13_out); // tcn_network_1_relu

    nnet::zeropad1d_cl<layer13_t, layer14_t, config14>(layer13_out, layer14_out); // tcn_network_1_conv2_pad

    nnet::conv_1d_cl<layer14_t, tcn_network_1_conv2_conv_result_t, config15>(layer14_out, layer15_out, w15, b15); // tcn_network_1_conv2_conv

    nnet::relu<tcn_network_1_conv2_conv_result_t, layer16_t, relu_config16>(layer15_out, layer16_out); // tcn_network_1_relu_1

    nnet::add<layer16_t, layer10_t, add_1_result_t, config17>(layer16_out, layer33_cpy2, layer17_out); // add_1

    nnet::relu<add_1_result_t, layer18_t, relu_config18>(layer17_out, layer18_out); // tcn_network_1_relu_2

    nnet::clone_stream<layer18_t, layer18_t, 1600>(layer18_out, layer34_cpy1, layer34_cpy2); // clone_tcn_network_1_relu_2

    nnet::zeropad1d_cl<layer18_t, layer19_t, config19>(layer34_cpy1, layer19_out); // tcn_network_2_conv1_pad

    nnet::conv_1d_cl<layer19_t, tcn_network_2_conv1_conv_result_t, config20>(layer19_out, layer20_out, w20, b20); // tcn_network_2_conv1_conv

    nnet::relu<tcn_network_2_conv1_conv_result_t, layer21_t, relu_config21>(layer20_out, layer21_out); // tcn_network_2_relu

    nnet::zeropad1d_cl<layer21_t, layer22_t, config22>(layer21_out, layer22_out); // tcn_network_2_conv2_pad

    nnet::conv_1d_cl<layer22_t, tcn_network_2_conv2_conv_result_t, config23>(layer22_out, layer23_out, w23, b23); // tcn_network_2_conv2_conv

    nnet::relu<tcn_network_2_conv2_conv_result_t, layer24_t, relu_config24>(layer23_out, layer24_out); // tcn_network_2_relu_1

    nnet::add<layer24_t, layer18_t, add_2_result_t, config25>(layer24_out, layer34_cpy2, layer25_out); // add_2

    nnet::relu<add_2_result_t, layer26_t, relu_config26>(layer25_out, layer26_out); // tcn_network_2_relu_2

    nnet::pooling1d_cl<layer26_t, layer27_t, config27>(layer26_out, layer27_out); // gap

    nnet::transpose<layer27_t, layer31_t, config31>(layer27_out, layer31_out); // transpose_input_for_squeeze

    nnet::dense<layer31_t, result_t, config29>(layer28_out, layer29_out, w29, b29); // fc

}

