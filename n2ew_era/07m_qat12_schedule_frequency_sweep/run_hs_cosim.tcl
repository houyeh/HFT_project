set script_dir [file dirname [file normalize [info script]]]
set vectors [file normalize [file join $script_dir .. 07l_ultrafast_policy_sweep results qat12_test_vectors.csv]]
set output [file normalize [file join $script_dir results hs_cosim_out.csv]]
open_project [file join $script_dir hls_hs_100mhz]
open_solution hls
cosim_design -rtl verilog -tool xsim -argv "$vectors $output"
exit
