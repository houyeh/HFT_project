open_project -reset hls_component
set_top tcn_streaming_step
add_files ../hls_src/tcn_stage2_resource_distilled_u2.cpp
add_files ../hls_src/tcn_prepad_hls.hpp
add_files ../hls_src/tcn_stage2_streaming_hls.hpp
add_files ../hls_src/tcn_stage2_resource_impl.hpp
add_files ../hls_src/tcn_weights_distilled.hpp
add_files -tb ../hls_src/tcn_stage2_resource_testbench.cpp
open_solution -flow_target vivado hls
set_part xc7z020-clg400-1
create_clock -period 10 -name default
config_compile -pipeline_loops 0
csynth_design
exit
