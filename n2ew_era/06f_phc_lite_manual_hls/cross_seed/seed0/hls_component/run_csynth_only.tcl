open_project hls_component
set_top phc_step
add_files ../hls_src/phc_step_hls.cpp
add_files ../hls_src/phc_step_hls.hpp
add_files ../hls_src/phc_weights.hpp
add_files -tb ../hls_src/phc_step_testbench.cpp
open_solution -flow_target vivado hls
set_part xc7z020-clg400-1
create_clock -period 10 -name default
csynth_design
exit
