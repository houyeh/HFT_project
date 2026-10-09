open_project hls_component
set_top tcn_prepad_infer
add_files C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/07b2_tcn_hls_closure/hls_src/tcn_prepad_hls_distilled.cpp
add_files C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/07b2_tcn_hls_closure/hls_src/tcn_prepad_hls.hpp
add_files C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/07b2_tcn_hls_closure/hls_src/tcn_prepad_hls_impl.hpp
add_files C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/07b2_tcn_hls_closure/hls_src/tcn_weights_distilled.hpp
add_files -tb C:/Users/holbo/00VS_projjects/00_HFT_project/n2ew_era/07b2_tcn_hls_closure/hls_src/tcn_prepad_testbench.cpp
open_solution -flow_target vivado hls
set_part xc7z020-clg400-1
create_clock -period 10 -name default
csynth_design
exit
