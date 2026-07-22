#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_layer[100*8],
    result_t layer10_out[3]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_layer complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_layer,layer10_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<fc1_weight_t, 25600>(w3, "w3.txt");
        nnet::load_weights_from_txt<fc1_bias_t, 32>(b3, "b3.txt");
        nnet::load_weights_from_txt<bn1_scale_t, 32>(s4, "s4.txt");
        nnet::load_weights_from_txt<bn1_bias_t, 32>(b4, "b4.txt");
        nnet::load_weights_from_txt<fc2_weight_t, 512>(w6, "w6.txt");
        nnet::load_weights_from_txt<fc2_bias_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<bn2_scale_t, 16>(s7, "s7.txt");
        nnet::load_weights_from_txt<bn2_bias_t, 16>(b7, "b7.txt");
        nnet::load_weights_from_txt<output_probabilities_weight_t, 48>(w9, "w9.txt");
        nnet::load_weights_from_txt<output_probabilities_bias_t, 3>(b9, "b9.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    auto& layer2_out = input_layer;
    layer3_t layer3_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    layer8_t layer8_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[3];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    nnet::dense<input_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // fc1

    nnet::normalize<layer3_t, layer4_t, config4>(layer3_out, layer4_out, s4, b4); // bn1

    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // relu1

    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // fc2

    nnet::normalize<layer6_t, layer7_t, config7>(layer6_out, layer7_out, s7, b7); // bn2

    nnet::relu<layer7_t, layer8_t, relu_config8>(layer7_out, layer8_out); // relu2

    nnet::dense<layer8_t, layer9_t, config9>(layer8_out, layer9_out, w9, b9); // output_probabilities

    nnet::softmax<layer9_t, result_t, softmax_config10>(layer9_out, layer10_out); // output_probabilities_softmax

}

