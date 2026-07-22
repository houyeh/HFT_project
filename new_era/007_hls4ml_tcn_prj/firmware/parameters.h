#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv1d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"
#include "nnet_utils/nnet_stream.h"
#include "nnet_utils/nnet_transpose.h"
#include "nnet_utils/nnet_transpose_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w12.h"
#include "weights/b12.h"
#include "weights/w15.h"
#include "weights/b15.h"
#include "weights/w20.h"
#include "weights/b20.h"
#include "weights/w23.h"
#include "weights/b23.h"
#include "weights/w29.h"
#include "weights/b29.h"


// hls-fpga-machine-learning insert layer-config
// transpose_input_for_x
struct config30 {
    static const unsigned dims = 2;
    static const unsigned N = 800;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config30_from_shape[2] = {8, 100};
unsigned config30_to_shape[2] = {100, 8};
unsigned config30_perm[2] = {1, 0};
unsigned config30_perm_strides[2] = {1, 100};

const unsigned* const config30::from_shape = config30_from_shape;
const unsigned* const config30::to_shape = config30_to_shape;
const unsigned* const config30::perm = config30_perm;
const unsigned* const config30::perm_strides = config30_perm_strides;

// tcn_network_0_conv1_pad
struct config2 : nnet::padding1d_config {
    static const unsigned in_width = 100;
    static const unsigned n_chan = 8;
    static const unsigned out_width = 102;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 0;
};

// tcn_network_0_conv1_conv
struct config3_mult : nnet::dense_config {
    static const unsigned n_in = 24;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef tcn_network_0_conv1_conv_bias_t bias_t;
    typedef tcn_network_0_conv1_conv_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config3 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 102;
    static const unsigned n_chan = 8;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 102;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef tcn_network_0_conv1_conv_bias_t bias_t;
    typedef tcn_network_0_conv1_conv_weight_t weight_t;
    typedef config3_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config3::filt_width> config3::pixels[] = {0};

// tcn_network_0_relu
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_0_relu_table_t table_t;
};

// tcn_network_0_conv2_pad
struct config5 : nnet::padding1d_config {
    static const unsigned in_width = 100;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 102;
    static const unsigned pad_left = 2;
    static const unsigned pad_right = 0;
};

// tcn_network_0_conv2_conv
struct config6_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef tcn_network_0_conv2_conv_bias_t bias_t;
    typedef tcn_network_0_conv2_conv_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config6 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 102;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 102;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef tcn_network_0_conv2_conv_bias_t bias_t;
    typedef tcn_network_0_conv2_conv_weight_t weight_t;
    typedef config6_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config6::filt_width> config6::pixels[] = {0};

// tcn_network_0_relu_1
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_0_relu_1_table_t table_t;
};

// tcn_network_0_downsample
struct config35_mult : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef tcn_network_0_downsample_bias_t bias_t;
    typedef tcn_network_0_downsample_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config35 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 100;
    static const unsigned n_chan = 8;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 100;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef tcn_network_0_downsample_bias_t bias_t;
    typedef tcn_network_0_downsample_weight_t weight_t;
    typedef config35_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config35::filt_width> config35::pixels[] = {0};

// add
struct config9 : nnet::merge_config {
    static const unsigned n_elem = 100*16;
    static const unsigned n_elem1 = 100*16;
    static const unsigned n_elem2 = 100*16;
    static const unsigned reuse_factor = 32;
};

// tcn_network_0_relu_2
struct relu_config10 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_0_relu_2_table_t table_t;
};

// tcn_network_1_conv1_pad
struct config11 : nnet::padding1d_config {
    static const unsigned in_width = 100;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 104;
    static const unsigned pad_left = 4;
    static const unsigned pad_right = 0;
};

// tcn_network_1_conv1_conv
struct config12_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef tcn_network_1_conv1_conv_bias_t bias_t;
    typedef tcn_network_1_conv1_conv_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config12 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 104;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 2;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 104;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef tcn_network_1_conv1_conv_bias_t bias_t;
    typedef tcn_network_1_conv1_conv_weight_t weight_t;
    typedef config12_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config12::filt_width> config12::pixels[] = {0};

// tcn_network_1_relu
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_1_relu_table_t table_t;
};

// tcn_network_1_conv2_pad
struct config14 : nnet::padding1d_config {
    static const unsigned in_width = 100;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 104;
    static const unsigned pad_left = 4;
    static const unsigned pad_right = 0;
};

// tcn_network_1_conv2_conv
struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef tcn_network_1_conv2_conv_bias_t bias_t;
    typedef tcn_network_1_conv2_conv_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config15 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 104;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 2;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 104;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef tcn_network_1_conv2_conv_bias_t bias_t;
    typedef tcn_network_1_conv2_conv_weight_t weight_t;
    typedef config15_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config15::filt_width> config15::pixels[] = {0};

// tcn_network_1_relu_1
struct relu_config16 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_1_relu_1_table_t table_t;
};

// add_1
struct config17 : nnet::merge_config {
    static const unsigned n_elem = 100*16;
    static const unsigned n_elem1 = 100*16;
    static const unsigned n_elem2 = 100*16;
    static const unsigned reuse_factor = 32;
};

// tcn_network_1_relu_2
struct relu_config18 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_1_relu_2_table_t table_t;
};

// tcn_network_2_conv1_pad
struct config19 : nnet::padding1d_config {
    static const unsigned in_width = 100;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 108;
    static const unsigned pad_left = 8;
    static const unsigned pad_right = 0;
};

// tcn_network_2_conv1_conv
struct config20_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef tcn_network_2_conv1_conv_bias_t bias_t;
    typedef tcn_network_2_conv1_conv_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config20 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 108;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 4;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 108;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef tcn_network_2_conv1_conv_bias_t bias_t;
    typedef tcn_network_2_conv1_conv_weight_t weight_t;
    typedef config20_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config20::filt_width> config20::pixels[] = {0};

// tcn_network_2_relu
struct relu_config21 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_2_relu_table_t table_t;
};

// tcn_network_2_conv2_pad
struct config22 : nnet::padding1d_config {
    static const unsigned in_width = 100;
    static const unsigned n_chan = 16;
    static const unsigned out_width = 108;
    static const unsigned pad_left = 8;
    static const unsigned pad_right = 0;
};

// tcn_network_2_conv2_conv
struct config23_mult : nnet::dense_config {
    static const unsigned n_in = 48;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 32;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef tcn_network_2_conv2_conv_bias_t bias_t;
    typedef tcn_network_2_conv2_conv_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config23 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 108;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 4;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 108;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef tcn_network_2_conv2_conv_bias_t bias_t;
    typedef tcn_network_2_conv2_conv_weight_t weight_t;
    typedef config23_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config23::filt_width> config23::pixels[] = {0};

// tcn_network_2_relu_1
struct relu_config24 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_2_relu_1_table_t table_t;
};

// add_2
struct config25 : nnet::merge_config {
    static const unsigned n_elem = 100*16;
    static const unsigned n_elem1 = 100*16;
    static const unsigned n_elem2 = 100*16;
    static const unsigned reuse_factor = 32;
};

// tcn_network_2_relu_2
struct relu_config26 : nnet::activ_config {
    static const unsigned n_in = 1600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 32;
    typedef tcn_network_2_relu_2_table_t table_t;
};

// gap
struct config27 : nnet::pooling1d_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 1;
    static const unsigned n_filt = 16;
    static const unsigned pool_width = 100;

    static const unsigned filt_width = pool_width;
    static const unsigned n_chan = n_filt;

    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const bool count_pad = true;
    static const unsigned stride_width = 100;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse_factor = 32;
    typedef model_default_t accum_t;
};

// transpose_input_for_squeeze
struct config31 {
    static const unsigned dims = 2;
    static const unsigned N = 16;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config31_from_shape[2] = {1, 16};
unsigned config31_to_shape[2] = {16, 1};
unsigned config31_perm[2] = {1, 0};
unsigned config31_perm_strides[2] = {1, 16};

const unsigned* const config31::from_shape = config31_from_shape;
const unsigned* const config31::to_shape = config31_to_shape;
const unsigned* const config31::perm = config31_perm;
const unsigned* const config31::perm_strides = config31_perm_strides;

// fc
struct config29 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 3;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 32;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 48;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef fc_bias_t bias_t;
    typedef fc_weight_t weight_t;
    typedef layer29_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
