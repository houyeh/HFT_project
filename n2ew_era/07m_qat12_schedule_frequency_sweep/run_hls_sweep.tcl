set script_dir [file dirname [file normalize [info script]]]
set src_dir [file join $script_dir hls_src]
set vectors [file normalize [file join $script_dir .. 07l_ultrafast_policy_sweep results qat12_test_vectors.csv]]
file mkdir [file join $script_dir results]

foreach variant {hs none} {
    if {$variant eq "hs"} {
        set top qat12_scalar_hs_pipeline
        set source qat12_scalar_hs_pipeline.cpp
        set cflags ""
    } else {
        set top qat12_scalar_none_pipeline
        set source qat12_scalar_none_pipeline.cpp
        set cflags "-DQAT_TOP_NONE"
    }
    set project [file join $script_dir "hls_${variant}_100mhz"]
    set output [file join $script_dir results "${variant}_csim_out.csv"]
    open_project -reset $project
    set_top $top
    add_files [file join $src_dir $source]
    add_files [file join $src_dir qat12_stream.hpp]
    add_files [file join $src_dir qat12_stream_impl.hpp]
    add_files [file join $src_dir qat12_weights.hpp]
    add_files -tb [file join $src_dir qat12_stream_testbench.cpp] -cflags $cflags
    open_solution -flow_target vivado hls
    set_part xc7z020-clg400-1
    create_clock -period 10 -name default
    config_compile -pipeline_loops 0
    csim_design -argv "$vectors $output"
    csynth_design
    close_project
}
exit
